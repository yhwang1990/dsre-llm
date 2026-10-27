"""Pretrained-language-model baselines: BERT-Focal and RoBERTa-MIL.
Uses exactly the same instance files, label mapping and evaluation code as DSRE-LLM.
  bert-focal  : BERT encoder + entity-position pooling + focal loss (sentence level)
  roberta-mil : RoBERTa encoder + multi-instance learning (within a batch only the sentence most confident in the bag label is kept per bag)
Dynamic padding; training time, peak memory and inference throughput are logged (for the cost table).
Tuned version (enabled by --val-file, identical for all methods): evaluate on the validation set after every epoch and predict the test set with the best epoch by AUC + Mi-F1 + Ma-F1;
optional linear warm-up/decay (--warmup) and punctuation entity markers (--markers: "@" around the head, "#" around the tail; a simplified typed marker of Zhou & Chen 2022).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dsre import data as D
from dsre import evaluate as E
from train import encode_text, pad_batch


def add_markers(x):
    """Wrap the head as "@ h @" and the tail as "# t #"; returns the new text and the (marked) head/tail character spans; no markers if the entities overlap."""
    text = x["text"]; h, t = tuple(x["h"]["pos"]), tuple(x["t"]["pos"])
    if not (h[1] <= t[0] or t[1] <= h[0]):
        return text, [h, t]
    out, prev, new = "", 0, {}
    for (s0, e0), m, k in sorted([(h, "@", "h"), (t, "#", "t")]):
        out += text[prev:s0]; ns = len(out); out += "%s %s %s" % (m, text[s0:e0], m); new[k] = (ns, len(out)); prev = e0
    return out + text[prev:], [new["h"], new["t"]]


class Data(Dataset):
    def __init__(self, insts, rel2id, tok, max_len, na_id, markers=False):
        self.x, self.rel2id, self.tok, self.max_len, self.na_id, self.markers = insts, rel2id, tok, max_len, na_id, markers

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        text, spans = add_markers(x) if self.markers else (x["text"], [tuple(x["h"]["pos"]), tuple(x["t"]["pos"])])
        ids, ent = encode_text(self.tok, text, spans, self.max_len)
        return {"ids": ids, "ent": ent, "y": self.rel2id.get(x["relation"], self.na_id), "idx": i}


def make_collate(pad_id):
    def collate(items):
        ids, am, em = pad_batch([it["ids"] for it in items], [it["ent"] for it in items], pad_id)
        return {"ids": ids, "am": am, "em": em, "y": torch.tensor([it["y"] for it in items]),
                "idx": torch.tensor([it["idx"] for it in items])}
    return collate


class Encoder(nn.Module):
    def __init__(self, path, n_rel, dropout=0.1):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        self.enc = AutoModel.from_pretrained(path)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(2 * AutoConfig.from_pretrained(path).hidden_size, n_rel)

    def forward(self, ids, am, em):
        # pool head and tail separately and concatenate (standard entity representation; keeps the relation direction)
        hs = self.enc(input_ids=ids, attention_mask=am).last_hidden_state

        def span_mean(m):
            m = m.float(); empty = m.sum(1) == 0
            if empty.any():
                m = m.clone(); m[empty] = am[empty].float()
            w = (m * am.float()).unsqueeze(-1)
            return (hs * w).sum(1) / w.sum(1).clamp(min=1e-6)
        h = torch.cat([span_mean(em == 1), span_mean(em == 2)], dim=-1)
        return self.cls(self.drop(h))


def focal_loss(logits, y, gamma=2.0, alpha=0.25):
    ce = F.cross_entropy(logits, y, reduction="none")
    return (alpha * (1 - torch.exp(-ce)) ** gamma * ce).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bert-focal", "roberta-mil", "bert-ce"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=0, help="> 0: early stopping after this many epochs without validation improvement (needs --val-file)")
    ap.add_argument("--val-file", default=None, help="evaluate on the validation set after every epoch and keep the best epoch")
    ap.add_argument("--warmup", type=float, default=0.0, help="> 0: linear warm-up (fraction of total steps) + linear decay")
    ap.add_argument("--markers", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="bf16 automatic mixed precision (speed only)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"; os.makedirs(a.out, exist_ok=True)
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na_id = rel2id.get("NA", -1)
    train = D.load_jsonl(a.train_file); test = D.load_jsonl(a.test_file)
    print("[data] train=%d test=%d K=%d method=%s" % (len(train), len(test), K, a.method), flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = Encoder(a.model_path, K).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    dl = DataLoader(Data(train, rel2id, tok, a.max_len, na_id, a.markers), batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, collate_fn=make_collate(tok.pad_token_id))
    sched = None
    if a.warmup > 0:
        from transformers import get_linear_schedule_with_warmup
        T = a.epochs * len(dl); sched = get_linear_schedule_with_warmup(opt, int(a.warmup * T), T)

    def predict(insts):
        model.eval(); P = np.zeros((len(insts), K), dtype=np.float32); off = 0
        d = DataLoader(Data(insts, rel2id, tok, a.max_len, na_id, a.markers), batch_size=256, shuffle=False,
                       num_workers=4, collate_fn=make_collate(tok.pad_token_id))
        with torch.no_grad():
            for b in d:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    p = F.softmax(model(b["ids"].to(dev), b["am"].to(dev), b["em"].to(dev)).float(), -1).cpu().numpy()
                P[off:off + len(p)] = p; off += len(p)
        model.train(); return P
    val = D.load_jsonl(a.val_file) if a.val_file else None
    if val:
        vgold = np.array([rel2id.get(x["relation"], na_id) for x in val]); vbags = [D.bag_key(x) for x in val]
    best, best_probs, hist = -1.0, None, []
    idx2bag = {}
    for bi, (k, idxs) in enumerate(D.group_into_bags(train).items()):
        for i in idxs:
            idx2bag[i] = bi
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); steps = 0
    for ep in range(a.epochs):
        model.train(); tot, nb = 0.0, 0
        for b in dl:
            ids, am, em, y = b["ids"].to(dev), b["am"].to(dev), b["em"].to(dev), b["y"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                logits = model(ids, am, em).float()
            if a.method == "bert-focal":
                loss = focal_loss(logits, y)
            elif a.method == "bert-ce":      # BERT+NLI step 2: plain cross-entropy on the NLI-consolidated labels (MCRE of DSRE-NLI)
                loss = F.cross_entropy(logits, y)
            else:
                bg = torch.tensor([idx2bag[int(i)] for i in b["idx"]], device=dev)
                keep = torch.ones_like(y, dtype=torch.bool)
                for u in bg.unique():
                    sel = (bg == u).nonzero(as_tuple=True)[0]
                    if sel.numel() > 1:
                        conf = logits.softmax(-1)[sel, y[sel]]
                        keep[sel] = False; keep[sel[conf.argmax()]] = True
                loss = F.cross_entropy(logits[keep], y[keep])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if sched:
                sched.step()
            tot += loss.item(); nb += 1; steps += 1
        print("  epoch %d  loss %.4f  %.1fs" % (ep, tot / max(nb, 1), time.time() - t0), flush=True)
        if val:
            v = E.evaluate(predict(val), vgold, vbags, K, na_id=na_id)
            sc = 100 * (v["auc_pr"] + v["micro_f1_nonNA"] + v["macro_f1"]); hist.append({"epoch": ep, "val_score": sc, "val_auc": v["auc_pr"]})
            print("    val AUC %.4f Mi %.4f Ma %.4f score %.2f" % (v["auc_pr"], v["micro_f1_nonNA"], v["macro_f1"], sc), flush=True)
            if sc > best:
                best = sc; best_probs = predict(test)
            if a.patience and len(hist) - 1 - max(range(len(hist)), key=lambda i: hist[i]["val_score"]) >= a.patience:
                print("    early stop", flush=True); break
    train_sec = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 1e9

    model.eval()
    probs = np.zeros((len(test), K), dtype=np.float32)
    dlt = DataLoader(Data(test, rel2id, tok, a.max_len, na_id, a.markers), batch_size=256, shuffle=False, num_workers=4,
                     collate_fn=make_collate(tok.pad_token_id))
    torch.cuda.synchronize(); t1 = time.time(); off = 0
    if best_probs is not None:
        probs = best_probs
    else:
        with torch.no_grad():
            for b in dlt:
                p = F.softmax(model(b["ids"].to(dev), b["am"].to(dev), b["em"].to(dev)).float(), -1).cpu().numpy()
                probs[off:off + len(p)] = p; off += len(p)
    torch.cuda.synchronize(); enc_sec = time.time() - t1
    gold = np.array([rel2id.get(x["relation"], na_id) for x in test]); bags = [D.bag_key(x) for x in test]
    tier_of_class = None
    if a.tiers:
        t = json.load(open(a.tiers))["tier_of_relation"]; tier_of_class = {rel2id[r]: v for r, v in t.items() if r in rel2id}
    res = E.evaluate(probs, gold, bags, K, na_id=na_id, tier_of_class=tier_of_class)
    if a.bootstrap:
        res["bootstrap"] = E.bootstrap_ci(probs, gold, bags, K, na_id=na_id, n_boot=a.bootstrap)
    out = {"model": a.method, "train_file": a.train_file, "test_file": a.test_file, "n_train": len(train),
           "epochs": a.epochs, "seed": a.seed, "result": res,
           "tuning": {"lr": a.lr, "warmup": a.warmup, "markers": a.markers, "val_file": a.val_file, "best_val_score": best, "history": hist},
           "cost": {"train_seconds": train_sec, "steps": steps, "peak_mem_gb": peak,
                    "encode_seconds": enc_sec, "ms_per_inst": 1000 * enc_sec / len(test)}}
    np.save(os.path.join(a.out, "probs.npy"), probs)
    json.dump(out, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("\n[%s]" % a.method, json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)
    print("[saved]", os.path.join(a.out, "result.json"))


if __name__ == "__main__":
    main()
