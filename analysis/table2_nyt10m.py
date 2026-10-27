"""NYT10m columns of Table 2: every method re-evaluated on the human-annotated NYT10 test set (NYT10m, Gao et al., 2021).
All NYT10m test sentences are contained in the NYT10 test set, so the per-sentence probabilities of the runs on test_full.jsonl
are reused (no re-training, no new LLM calls) and projected onto the NYT10m label space with the mapping of nyt10m_eval.py.
Qwen3-8B-ICL probabilities are exported with baselines/dump_zs_probs.py to results/zsprobs/knncal.npy.
Output: results/table2_nyt10m.json"""
import json, os
import numpy as np
from common import D, E, R, RUNS, P, C, RES, FN, CMAX, FTLR, SEEDS, BEST, ONTO, Ontology, load_cache
from nyt10m_eval import build_map

ours = json.load(open(P + "/nyt10/rel2id.json")); m = json.load(open(R + "/data_ext/nyt10m/nyt10m_rel2id.json")); K = len(m); na = m["NA"]
full = D.load_jsonl(P + "/nyt10/test_full.jsonl"); uids = [x["uid"] for x in full]
idx = {}
for i, x in enumerate(full):
    idx.setdefault((x["text"], x["h"]["name"], x["t"]["name"]), i)
test = D.load_jsonl(R + "/data_ext/nyt10m/nyt10m_test.txt")
rows = np.array([idx[(x["text"], x["h"]["name"], x["t"]["name"])] for x in test])
gold = np.array([m[x["relation"]] for x in test])
bag_ids = [D.bag_key(x) for x in test]
keys, bags, _ = E.bag_aggregate(np.zeros((len(test), 1)), bag_ids)
members = [np.array(bags[k]) for k in keys]
bag_gold = [set(int(gold[i]) for i in mm) for mm in members]
PM = build_map(ours, m)


def metrics(pf):
    p = pf[rows] @ PM
    bp = np.stack([p[mm].max(axis=0) for mm in members]); yp = p.argmax(1)
    sc, lab = E._facts(bp, bag_gold, K, na)
    pn = E.precision_at_n(sc, lab); s = E.sentence_level_summary(gold, yp, K, na_id=na)
    return [E.pr_curve_auc(sc, lab)[0], pn[100], s["micro_f1_nonNA"], E.macro_f1(gold, yp, K, na_id=na)]


def aligned(d, name="probs.npy"):
    p = np.load(os.path.join(d, name))
    if os.path.exists(os.path.join(d, "uids.json")):
        assert json.load(open(os.path.join(d, "uids.json"))) == uids
    assert p.shape[0] == len(uids)
    return p


votes = load_cache(C + "/nyt10/test_full_votes.jsonl")
onto = Ontology(ONTO, relations=[r for r in sorted(ours, key=lambda r: ours[r]) if r != "NA"], has_na=True)
types = load_cache(C + "/nyt10/entity_types.jsonl")


def fused(d, c=CMAX["nyt10"]):
    p = aligned(d); f = p.copy()
    for i in np.flatnonzero(E.needs_llm(p, c)):
        x = full[i]; v = votes[uids[i]]
        ok = bool(onto.consistent(types.get(x["h"]["name"], "OTHER"), types.get(x["t"]["name"], "OTHER"), v["rel"]))
        f[i] = E.fuse(p[i], v["rel"], v["conf"], ours, ok, c)
    return f


BL = {"PCNN+ATT": "pcnn", "BERT-Focal": "focal", "RoBERTa-MIL": "mil", "BERT-NLI": "nli", "CGRE": "cgre", "LLM-CG": "llmcg"}
out = {}
for name, mth in BL.items():
    out[name] = [metrics(aligned(f"{RUNS}/tune/nyt10_{mth}_{BEST[('nyt10', mth)]}_s{s}")) for s in SEEDS]
pth = f"{RES}/zsprobs/knncal.npy"
if os.path.exists(pth):
    out["Qwen3-8B-ICL"] = [metrics(np.load(pth))]
out["Qwen3-8B-FT"] = [metrics(aligned(f"{RUNS}/nyt10_ft_lr{FTLR}_8b_s{s}_v3/eval_test_full")) for s in SEEDS]
out["DSRE-LLM (Ours)"] = [metrics(fused(f"{RUNS}/nyt10_{FN}_8b_s{s}_v3/eval_test_full")) for s in SEEDS]
res = {}
for k, v in out.items():
    A = 100 * np.array(v)
    res[k] = {"n": len(v), "mean": A.mean(0).tolist(), "std": (A.std(0, ddof=1) if len(v) > 1 else 0 * A[0]).tolist()}
    print(f"{k:20s}", " ".join(f"{a:6.2f}±{b:4.2f}" for a, b in zip(res[k]["mean"], res[k]["std"])), flush=True)
json.dump({"keys": ["auc_pr", "p@100", "micro_f1_nonNA", "macro_f1"], "n_sentences": len(test), "n_bags": len(keys), "rows": res},
          open(RES + "/table2_nyt10m.json", "w"), indent=1)
