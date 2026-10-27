"""Table 10: sensitivity of the trust ceiling c_max for the Qwen3-8B model (seed 42) on the full NYT10 test set.
A setting is reported only when the votes of all instances it sends to the LLM are available.
Usage: python cmax_sensitivity.py 0.1,0.2,0.25,0.3,0.4    Output: results/cmax_sensitivity.json"""
import json, sys
import numpy as np
from common import D, E, P, C, RUNS, RES, FN, ONTO, Ontology, load_cache

NYT = P + "/nyt10"; RUN = f"{RUNS}/nyt10_{FN}_8b_s42_v3/eval_test_full"
rel2id = json.load(open(NYT + "/rel2id.json")); K = len(rel2id); na = rel2id["NA"]
rel_rows = [r for r in sorted(rel2id, key=lambda r: rel2id[r]) if r != "NA"]
insts = D.load_jsonl(NYT + "/test_full.jsonl")
gold = np.array([rel2id.get(x["relation"], na) for x in insts]); bag_ids = [D.bag_key(x) for x in insts]; uids = [x["uid"] for x in insts]
probs = np.load(RUN + "/probs.npy"); assert json.load(open(RUN + "/uids.json")) == uids
votes = load_cache(C + "/nyt10/test_full_votes.jsonl")
onto = Ontology(ONTO, relations=rel_rows, has_na=True)
types = load_cache(C + "/nyt10/entity_types.jsonl")
out = {}
for c in [float(x) for x in sys.argv[1].split(",")]:
    nd = E.needs_llm(probs, c)
    miss = sum(1 for i in np.flatnonzero(nd) if uids[i] not in votes)
    if miss:
        print(c, "skip: missing votes", miss, flush=True); out[str(c)] = {"missing_votes": miss}; continue
    fused = probs.copy()
    for i in np.flatnonzero(nd):
        x = insts[i]; v = votes[uids[i]]
        ok = bool(onto.consistent(types.get(x["h"]["name"], "OTHER"), types.get(x["t"]["name"], "OTHER"), v["rel"]))
        fused[i] = E.fuse(probs[i], v["rel"], v["conf"], rel2id, ok, c)
    r = E.evaluate(fused, gold, bag_ids, K, na_id=na)
    # share of called instances whose final decision equals the LLM vote
    r["final_equals_llm_on_called"] = float(np.mean([fused[i].argmax() == rel2id.get(votes[uids[i]]["rel"], -2) for i in np.flatnonzero(nd)])) if nd.any() else None
    r["call_fraction"] = float(nd.mean())
    out[str(c)] = r
    print(c, {k: round(100 * r[k], 2) for k in ["auc_pr", "p@100", "mean_p@n", "micro_f1_nonNA", "macro_f1", "final_equals_llm_on_called", "call_fraction"]}, flush=True)
json.dump(out, open(RES + "/cmax_sensitivity.json", "w"), indent=1)
