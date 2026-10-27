"""PCNN+ATT baseline (piecewise CNN, Zeng et al. 2015, + selective attention, Lin et al. 2016).
Uses the same instance files, label mapping and evaluation code as DSRE-LLM; bag-level training and inference.
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dsre import data as D
from dsre import evaluate as E

MAX_LEN = 120
POS_LIMIT = 60
MAX_BAG = 64


def load_w2v(path):
    vocab, vecs = {}, []
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8", errors="ignore").split()
        n, dim = int(header[0]), int(header[1])
        for _ in range(n):
            chars = []
            while True:
                c = f.read(1)
                if c == b" " or c == b"":
                    break
                if c != b"\n":
                    chars.append(c)
            w = b"".join(chars).decode("utf-8", errors="ignore")
            v = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            if len(v) < dim:
                break
            vocab[w] = len(vecs); vecs.append(v)
    return vocab, np.stack(vecs), dim


def tokenize(s):
    return re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", s.lower())


class Featurizer:
    def __init__(self, vocab):
        self.vocab = vocab; self.unk = len(vocab); self.pad = self.unk + 1

    def __call__(self, inst):
        toks = tokenize(inst["text"])[:MAX_LEN]
        hp = self._find(toks, tokenize(inst["h"]["name"])); tp = self._find(toks, tokenize(inst["t"]["name"]))
        ids = [self.vocab.get(t, self.unk) for t in toks]; n = len(ids)
        p1 = [int(np.clip(i - hp, -POS_LIMIT, POS_LIMIT)) + POS_LIMIT for i in range(n)]
        p2 = [int(np.clip(i - tp, -POS_LIMIT, POS_LIMIT)) + POS_LIMIT for i in range(n)]
        a, b = sorted([hp, tp])
        mask = [1 if i <= a else (2 if i <= b else 3) for i in range(n)]
        pad = MAX_LEN - n
        return ids + [self.pad] * pad, p1 + [0] * pad, p2 + [0] * pad, mask + [0] * pad

    @staticmethod
    def _find(toks, ent):
        for i in range(len(toks)):
            if ent and toks[i:i + len(ent)] == ent:
                return i
        return 0


class PCNN(nn.Module):
    def __init__(self, emb, n_rel, pos_dim=5, n_filters=230, window=3, dropout=0.5):
        super().__init__()
        V, Dm = emb.shape
        self.word = nn.Embedding(V + 2, Dm, padding_idx=V + 1)
        with torch.no_grad():
            self.word.weight[:V] = torch.tensor(emb)
        self.p1 = nn.Embedding(2 * POS_LIMIT + 2, pos_dim, padding_idx=0)
        self.p2 = nn.Embedding(2 * POS_LIMIT + 2, pos_dim, padding_idx=0)
        self.conv = nn.Conv1d(Dm + 2 * pos_dim, n_filters, window, padding=window // 2)
        self.drop = nn.Dropout(dropout)
        self.rel = nn.Embedding(n_rel, n_filters * 3)
        self.bias = nn.Parameter(torch.zeros(n_rel)); self.n_rel = n_rel

    def sentence_rep(self, ids, p1, p2, mask):
        x = torch.cat([self.word(ids), self.p1(p1), self.p2(p2)], dim=-1)
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        outs = [torch.where((mask == seg).unsqueeze(-1), c, torch.full_like(c, -1e4)).max(dim=1).values for seg in (1, 2, 3)]
        return self.drop(torch.tanh(torch.cat(outs, dim=-1)))

    def bag_logits(self, reps, bag_index, n_bags):
        A = self.rel.weight; e = reps @ A.t()
        out = reps.new_zeros(n_bags, self.n_rel)
        for b in range(n_bags):
            sel = bag_index == b
            if sel.any():
                alpha = F.softmax(e[sel], dim=0); s = alpha.t() @ reps[sel]
                out[b] = (s * A).sum(-1) + self.bias
        return out


def make_bags(insts, rel2id, na_id):
    bags, order = {}, []
    for i, x in enumerate(insts):
        k = D.bag_key(x)
        if k not in bags:
            bags[k] = {"idx": [], "rel": set()}; order.append(k)
        bags[k]["idx"].append(i); bags[k]["rel"].add(rel2id.get(x["relation"], na_id))
    return order, bags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--w2v", required=True)
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bags-per-batch", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=0, help="> 0: early stopping after this many epochs without validation improvement (needs --val-file)")
    ap.add_argument("--val-file", default=None, help="evaluate on the validation set after every epoch (AUC + Mi-F1 + Ma-F1) and predict the test set with the best epoch")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na_id = rel2id.get("NA", -1)
    train = D.load_jsonl(a.train_file); test = D.load_jsonl(a.test_file)
    vocab, emb, dim = load_w2v(a.w2v); feat = Featurizer(vocab)
    print("[data] train=%d test=%d K=%d  w2v %dx%d" % (len(train), len(test), K, len(vocab), dim), flush=True)

    def encode_all(insts):
        f = [feat(x) for x in insts]
        return tuple(torch.tensor([z[j] for z in f]) for j in range(4))
    tr = encode_all(train); te = encode_all(test)
    tr_order, tr_bags = make_bags(train, rel2id, na_id)
    model = PCNN(emb, K).to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    def predict(insts, enc):
        model.eval(); te_order, te_bags = make_bags(insts, rel2id, na_id)
        bag_probs = np.zeros((len(te_order), K), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(te_order), 64):
                chunk = te_order[s:s + 64]; idxs, bag_index = [], []
                for bi, k in enumerate(chunk):
                    for i in te_bags[k]["idx"]:
                        idxs.append(i); bag_index.append(bi)
                it = torch.tensor(idxs)
                reps = model.sentence_rep(*(t[it].to(dev) for t in enc))
                logits = model.bag_logits(reps, torch.tensor(bag_index, device=dev), len(chunk))
                bag_probs[s:s + len(chunk)] = F.softmax(logits.float(), dim=-1).cpu().numpy()
        # bag level: one row per bag; for sentence-level Macro-F1 each sentence receives the distribution of its bag
        sent_probs = np.zeros((len(insts), K), dtype=np.float32)
        for bi, k in enumerate(te_order):
            for i in te_bags[k]["idx"]:
                sent_probs[i] = bag_probs[bi]
        model.train(); return sent_probs

    val = D.load_jsonl(a.val_file) if a.val_file else None
    if val:
        va = encode_all(val); vgold = np.array([rel2id.get(x["relation"], na_id) for x in val]); vbags = [D.bag_key(x) for x in val]
    best, best_probs, hist = -1.0, None, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(len(tr_order)); tot, nb = 0.0, 0
        for s in range(0, len(perm), a.bags_per_batch):
            chunk = [tr_order[i] for i in perm[s:s + a.bags_per_batch]]
            idxs, bag_index, ys = [], [], []
            for bi, k in enumerate(chunk):
                for i in tr_bags[k]["idx"][:MAX_BAG]:
                    idxs.append(i); bag_index.append(bi)
                rs = sorted(r for r in tr_bags[k]["rel"] if r != na_id)
                ys.append(rs[0] if rs else na_id)
            it = torch.tensor(idxs)
            reps = model.sentence_rep(*(t[it].to(dev) for t in tr))
            logits = model.bag_logits(reps, torch.tensor(bag_index, device=dev), len(chunk))
            loss = F.cross_entropy(logits, torch.tensor(ys, device=dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        print("  epoch %d  loss %.4f  %.1fs" % (ep, tot / max(nb, 1), time.time() - t0), flush=True)
        if val:
            v = E.evaluate(predict(val, va), vgold, vbags, K, na_id=na_id)
            sc = 100 * (v["auc_pr"] + v["micro_f1_nonNA"] + v["macro_f1"]); hist.append({"epoch": ep, "val_score": sc, "val_auc": v["auc_pr"]})
            print("    val AUC %.4f Mi %.4f Ma %.4f score %.2f" % (v["auc_pr"], v["micro_f1_nonNA"], v["macro_f1"], sc), flush=True)
            if sc > best:
                best = sc; best_probs = predict(test, te)
            if a.patience and len(hist) - 1 - max(range(len(hist)), key=lambda i: hist[i]["val_score"]) >= a.patience:
                print("    early stop", flush=True); break
    train_sec = time.time() - t0

    t1 = time.time(); sent_probs = best_probs if best_probs is not None else predict(test, te)
    gold = np.array([rel2id.get(x["relation"], na_id) for x in test]); bags = [D.bag_key(x) for x in test]
    tier_of_class = None
    if a.tiers:
        t = json.load(open(a.tiers))["tier_of_relation"]; tier_of_class = {rel2id[r]: v for r, v in t.items() if r in rel2id}
    res = E.evaluate(sent_probs, gold, bags, K, na_id=na_id, tier_of_class=tier_of_class)
    if a.bootstrap:
        res["bootstrap"] = E.bootstrap_ci(sent_probs, gold, bags, K, na_id=na_id, n_boot=a.bootstrap)
    out = {"model": "PCNN+ATT", "train_file": a.train_file, "test_file": a.test_file, "n_train": len(train),
           "epochs": a.epochs, "seed": a.seed, "result": res,
           "tuning": {"lr": a.lr, "val_file": a.val_file, "best_val_score": best, "history": hist},
           "cost": {"train_seconds": train_sec, "encode_seconds": time.time() - t1}}
    np.save(os.path.join(a.out, "probs.npy"), sent_probs)
    json.dump(out, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("\n[PCNN+ATT]", json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)


if __name__ == "__main__":
    main()
