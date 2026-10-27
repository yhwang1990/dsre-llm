"""Per-relation error analysis on NYT10 (Section 4.2.1): DSRE-LLM with Qwen3-8B, seed 42, fused at c_max 0.25.
For every non-NA relation: test and training sentence counts, F1, number of predictions, and the three most frequent
predicted labels of its test sentences. Output: results/error_analysis_nyt10.json"""
import json
from collections import Counter
from common import DS, D, E, P, C, RUNS, RES, FN, CMAX, load_cache

nyt = DS("nyt10", f"{P}/nyt10/test_full.jsonl", f"{C}/nyt10/entity_types.jsonl")
V = load_cache(f"{C}/nyt10/test_full_votes.jsonl")
p, u = nyt.probs(f"{RUNS}/nyt10_{FN}_8b_s42_v3/eval_test_full")
assert u == nyt.uids
f = nyt.fuse(p, u, V, CMAX["nyt10"])
gold = nyt.gold; pred = f.argmax(1)
id2rel = {i: r for r, i in nyt.rel2id.items()}
tf = Counter(x["relation"] for x in D.load_jsonl(f"{P}/nyt10/train_main.jsonl"))
out = {}
for k, (f1, sup) in E.per_class_f1(gold, pred, nyt.K).items():
    if k == nyt.na:
        continue
    dest = Counter(id2rel[int(x)] for x in pred[gold == k])
    out[id2rel[k]] = {"test": int(sup), "train": tf.get(id2rel[k], 0), "f1": f1, "n_predicted": int((pred == k).sum()),
                      "dest": [(r, round(c / sup, 3)) for r, c in dest.most_common(3)]}
json.dump(out, open(f"{RES}/error_analysis_nyt10.json", "w"), indent=1)
for r, v in sorted(out.items(), key=lambda x: -x[1]["test"]):
    print(f"{r:60s} test={v['test']:5d} train={v['train']:5d} F1={100*v['f1']:5.1f} npred={v['n_predicted']:5d}", v["dest"])
