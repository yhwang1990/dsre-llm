"""c_max selection on Wiki80: for every DSRE-LLM model, c_max is chosen from a grid by Macro-F1 on the noisy validation set val_cmax
(noisy labels with the same noise setting as its training set; ties -> smaller value), then the test set is fused with the selected c_max (Qwen3-8B votes: test8b_votes.jsonl).
The classifier alone and c_max = 0.3 are reported on the test set for reference."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsre import data as D
from dsre import evaluate as E
from dsre.ontology import Ontology
from train import load_cache

R = os.environ.get("DSRE_ROOT", "workspace"); W = R + "/data_prepared/wiki80"; C = R + "/cache/wiki80"
rel2id = json.load(open(W + "/rel2id.json")); K = len(rel2id); rels = [r for r in sorted(rel2id, key=lambda r: rel2id[r])]
onto = Ontology(R + "/data/ontology/relation_ontology_all.json", relations=rels, has_na=False)
types = load_cache(C + "/entity_types.jsonl")
GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
val = D.load_jsonl(W + "/val_cmax.jsonl"); vlab = json.load(open(W + "/val_cmax_labels.json"))
test = D.load_jsonl(W + "/test.jsonl"); tgold = np.array([rel2id[x["relation"]] for x in test])
vv = load_cache(C + "/val_cmax_votes.jsonl"); tv = load_cache(C + "/test8b_votes.jsonl")


def fuse(probs, insts, votes, c):
    if c <= 0:
        return probs
    nd = E.needs_llm(probs, c); f = probs.copy()
    for i in np.flatnonzero(nd):
        x = insts[i]; v = votes[x["uid"]]
        ok = bool(onto.consistent(types.get(x["h"]["name"], "OTHER"), types.get(x["t"]["name"], "OTHER"), v["rel"]))
        f[i] = E.fuse(probs[i], v["rel"], v["conf"], rel2id, ok, c)
    return f


def mf1(p, g):
    y = p.argmax(1)
    return E.macro_f1(g, y, K, na_id=-1), float((y == g).mean())


out = {}
for run in sys.argv[1:]:
    st = run.split("/")[-1].split("_cwf")[0].replace("w80_", "")      # e.g. sym_40
    vp = np.load(run + "/eval_val_cmax/probs.npy"); assert json.load(open(run + "/eval_val_cmax/uids.json")) == [x["uid"] for x in val]
    tp = np.load(run + "/eval_test/probs.npy"); assert json.load(open(run + "/eval_test/uids.json")) == [x["uid"] for x in test]
    g = np.array([rel2id[r] for r in vlab[st]])
    sel = {str(c): mf1(fuse(vp, val, vv, c), g)[0] for c in GRID}
    best = max(GRID, key=lambda c: (round(sel[str(c)], 6), -c))
    res = {"val_macro_f1_by_cmax": sel, "selected_cmax": best}
    for name, c in (("classifier_only", 0.0), ("selected", best), ("cmax_0.3", 0.3)):
        f = fuse(tp, test, tv, c); m, acc = mf1(f, tgold)
        res["test_" + name] = {"c_max": c, "macro_f1": m, "micro_f1": acc, "call_fraction": float(E.needs_llm(tp, c).mean()) if c > 0 else 0.0}
    out[st] = res
    print(st, "selected", best, {k: round(100 * v, 2) for k, v in sel.items()},
          {k: round(100 * v["macro_f1"], 2) for k, v in res.items() if k.startswith("test_")}, flush=True)
json.dump(out, open(R + "/results/w80_cmax_select.json", "w"), indent=1)
