"""CGRE / LLM-CG baselines (Liang et al., TKDE 2022, github.com/tmliang/CGRE): PCNN sentence encoder + relation-entity-type constraint graph (GCN) + constraint-aware attention.
Re-written as a single file from the official Net/CGRE.py, Net/graph_encoder.py and data_loader.build_graph, using the same instance files, label mapping and evaluation code as DSRE-LLM:
  - graph nodes = entity types + relations; each constraint (h_type, r, t_type) adds edges h->r and r->t; NA is connected to all types (as in the official code)
  - constraints are counted on the training set: (head type, tail type) combinations observed for each relation
  - entity types come from cache/<ds>/entity_types.jsonl (the same types as our method); unknown -> OTHER
  - GCN layers implement the symmetric normalisation D^-1/2 (A+I) D^-1/2 of PyG GCNConv (no torch_geometric dependency)
  - optimiser and loss as in the official PCNN configuration: SGD (lr 0.5, wd 1e-5), 160 bags per batch, cross-entropy with class weights 1/n^0.05
  - the official code picks the best epoch on the test set; here the test set is never used (fixed number of epochs, or the best validation epoch with --val-file)
Difference from the official code: the embedding layer is the PCNN of pcnn_att.py (word + two position embeddings) instead of the entity-aware embedding (lambda gate).
"""
import argparse, collections, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsre import data as D
from dsre import evaluate as E
from pcnn_att import load_w2v, Featurizer, PCNN, make_bags, MAX_BAG


class GCN(nn.Module):
    def __init__(self, edge, n_node, n_type, emb, hid, out, layers=2, init=None):
        super().__init__()
        A = torch.zeros(n_node, n_node)
        A[edge[1], edge[0]] = 1.0                      # message source -> target
        A = A + torch.eye(n_node)
        d = A.sum(1).pow(-0.5)
        self.register_buffer("A", d[:, None] * A * d[None, :])
        if init is None:      # CGRE: randomly initialised node embeddings
            self.node = nn.Embedding(n_node, emb); nn.init.xavier_uniform_(self.node.weight)
        else:                 # LLM-CG: frozen LLM node representations, linearly projected as initial features and refined by the GCN
            self.node = None; self.register_buffer("init", init); self.proj = nn.Linear(init.shape[1], emb)
        dims = [emb] + [hid] * (layers - 1) + [out]
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.act = nn.GELU(); self.n_type = n_type

    def forward(self):
        x = self.node.weight if self.node is not None else self.proj(self.init)
        for lin in self.lins:
            x = self.act(self.A @ lin(x))
        return x[:self.n_type], x[self.n_type:]


class CGRE(nn.Module):
    def __init__(self, emb, K, n_type, edge, constraint, g_emb=150, g_hid=900, g_out=150, class_dim=300, dropout=0.5, node_init=None):
        super().__init__()
        self.enc = PCNN(emb, K)                       # only its sentence_rep (3 x 230 dims) is used
        self.gnn = GCN(edge, n_type + K, n_type, g_emb, g_hid, g_out, init=node_init)
        self.constraint = constraint                  # one (2, n) tensor of type ids per relation
        self.fc1 = nn.Linear(230 * 3 + 2 * g_out, class_dim)
        self.fc2 = nn.Linear(3 * g_out, class_dim)
        self.cls = nn.Linear(class_dim, K)
        for m in (self.fc1, self.fc2, self.cls):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.drop = nn.Dropout(dropout); self.K = K

    def reps(self, ids, p1, p2, mask, types):
        X = self.enc.sentence_rep(ids, p1, p2, mask)
        T, R = self.gnn()
        X = torch.relu(self.fc1(torch.cat([X, F.embedding(types, T).reshape(types.shape[0], -1)], -1)))
        Tr = torch.stack([F.embedding(c, T).mean(1).reshape(-1) for c in self.constraint])
        C = torch.relu(self.fc2(torch.cat([R, Tr], -1)))
        return X, C

    def train_logits(self, X, C, bag_index, ys):
        out = []
        for b, y in enumerate(ys):
            r = X[bag_index == b]
            out.append(F.softmax(r @ C[y], 0) @ r)
        return self.cls(self.drop(torch.stack(out)))

    def test_probs(self, X, C, bag_index, n_bags):
        att = X @ C.t(); out = []
        for b in range(n_bags):
            sel = bag_index == b
            s = F.softmax(att[sel], 0).t() @ X[sel]                 # (K, dim): one bag representation per relation
            out.append(torch.diagonal(F.softmax(self.cls(s), -1)))
        return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True); ap.add_argument("--test-file", required=True)
    ap.add_argument("--rel2id", required=True); ap.add_argument("--w2v", required=True)
    ap.add_argument("--types", required=True, help="entity_types.jsonl")
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--node-init", default=None, help="output of llm_node_emb.py; given -> LLM-CG, otherwise CGRE")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--bags-per-batch", type=int, default=160)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=0, help="> 0: early stopping after this many epochs without validation improvement (needs --val-file)")
    ap.add_argument("--val-file", default=None, help="evaluate on the validation set after every epoch (AUC + Mi-F1 + Ma-F1) and predict the test set with the best epoch")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na_id = rel2id.get("NA", -1)
    train = D.load_jsonl(a.train_file); test = D.load_jsonl(a.test_file)
    etype = {}
    for l in open(a.types):
        x = json.loads(l); etype[x["k"]] = x["v"]
    tnames = sorted(set(etype.values()) | {"OTHER"}); type2id = {t: i for i, t in enumerate(tnames)}; NT = len(tnames)
    ty = lambda x: [type2id[etype.get(x["h"]["name"], "OTHER")], type2id[etype.get(x["t"]["name"], "OTHER")]]

    # constraint graph: (head type, tail type) observed for each relation in the training set
    pairs = collections.defaultdict(set)
    for x in train:
        r = rel2id.get(x["relation"], na_id)
        if r != na_id:
            pairs[r].add(tuple(ty(x)))
    sub, obj, constraint = [], [], []
    for r in range(K):
        if r == na_id:
            hs = ts = list(range(NT))
            for i in range(NT):
                sub += [i, NT + r]; obj += [NT + r, i]
        else:
            ps = sorted(pairs[r]) or [(type2id["OTHER"], type2id["OTHER"])]
            hs, ts = [p[0] for p in ps], [p[1] for p in ps]
            for h, t in ps:
                sub += [h, NT + r]; obj += [NT + r, t]
        constraint.append(torch.tensor([hs, ts], dtype=torch.long, device=dev))
    edge = torch.tensor([sub, obj], dtype=torch.long)
    json.dump({"type2id": type2id, "constraints": {r: sorted(map(list, pairs[rel2id[r]])) for r in rel2id if r != "NA"}},
              open(os.path.join(a.out, "constraint_graph.json"), "w"), indent=1)

    vocab, emb, dim = load_w2v(a.w2v); feat = Featurizer(vocab)
    print("[data] train=%d test=%d K=%d types=%d edges=%d" % (len(train), len(test), K, NT, edge.shape[1]), flush=True)

    def encode_all(insts):
        f = [feat(x) for x in insts]
        return tuple(torch.tensor([z[j] for z in f]) for j in range(4)) + (torch.tensor([ty(x) for x in insts]),)
    tr = encode_all(train); te = encode_all(test)
    tr_order, tr_bags = make_bags(train, rel2id, na_id)
    node_init = None
    if a.node_init:       # align by name: node order = types (type2id) + relations (rel2id)
        z = np.load(a.node_init); row = {("T", t): i for i, t in enumerate(z["types"])}
        row.update({("R", r): len(z["types"]) + i for i, r in enumerate(z["rels"])})
        order = [row[("T", t)] for t in tnames] + [row[("R", r)] for r in sorted(rel2id, key=rel2id.get)]
        node_init = torch.tensor(z["emb"][order], dtype=torch.float32)
        node_init = F.normalize(node_init - node_init.mean(0), dim=-1)   # centre then row-normalise: LLM hidden states have a few very large dimensions that diverge at lr 0.5
    model = CGRE(emb, K, NT, edge, constraint, node_init=node_init).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, weight_decay=1e-5)
    cnt = collections.Counter(rel2id.get(x["relation"], na_id) for x in train)   # counted per instance, as in the official code
    cw = torch.ones(K)
    for k, v in cnt.items():
        cw[k] = 1.0 / v ** 0.05
    cw = cw.to(dev)
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
                X, C = model.reps(*(t[it].to(dev) for t in enc))
                bag_probs[s:s + len(chunk)] = model.test_probs(X, C, torch.tensor(bag_index, device=dev), len(chunk)).float().cpu().numpy()
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
            X, C = model.reps(*(t[it].to(dev) for t in tr))
            logits = model.train_logits(X, C, torch.tensor(bag_index, device=dev), ys)
            loss = F.cross_entropy(logits, torch.tensor(ys, device=dev), weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
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
    out = {"model": "LLM-CG" if a.node_init else "CGRE", "node_init": a.node_init, "train_file": a.train_file, "test_file": a.test_file, "n_train": len(train),
           "epochs": a.epochs, "lr": a.lr, "seed": a.seed, "result": res,
           "tuning": {"lr": a.lr, "val_file": a.val_file, "best_val_score": best, "history": hist},
           "cost": {"train_seconds": train_sec, "encode_seconds": time.time() - t1}}
    np.save(os.path.join(a.out, "probs.npy"), sent_probs)
    json.dump(out, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("\n[%s]" % out["model"], json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)


if __name__ == "__main__":
    main()
