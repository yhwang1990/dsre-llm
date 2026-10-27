"""Tables 6 and 9: ablation with the main backbone Qwen3-8B on the full NYT10 test set (c_max 0.25, Qwen3-8B votes).
Table 9: the full row is DSRE-LLM of Table 2 (mean of three seeds, read from results/table2.json written by table2.py); every variant
is a single seed-42 run, read from the result.json written by reproduce/07_ablation.sh. The three ontology-isolating variants reuse the
seed-42 main model and only change the LLM votes.
Table 6: per-tier Macro-F1 of the main model (mean of three seeds, as the full row of Table 9 and Figure 7) and of the two long-tail variants (w/o LLM centroid init., w/o frequency compensation),
together with the number of relations per tier in training and in the test set. Output: results/ablation/summary.json"""
import collections, json, os
import numpy as np
from common import DS, E, P, C, RUNS, RES, FN, CMAX, SEEDS, load_cache

O = RES + "/ablation"; os.makedirs(O, exist_ok=True)
M = f"{RUNS}/nyt10_{FN}_8b_s42_v3"
D_ = {j: f"{RUNS}/abl_{j}_8b_s42/eval_test_full" for j in
      "no_onto no_contrast proto_only inst_only no_cw no_anchor no_freq no_onto_contrast no_all".split()}
D_.update(keepcot=M + "/eval_test_full_keepcot", coarse=M + "/eval_test_full_onto_coarse", noisy=M + "/eval_test_full_onto_noisy")
K = ("auc_pr", "p@100", "mean_p@n", "micro_f1_nonNA", "macro_f1")

# ---------------- Table 9
t2 = json.load(open(f"{RES}/table2.json")); keys = t2["keys"]; full = t2["rows"]["DSRE-LLM (Ours)"]["nyt10"]["mean"]
t9 = {"full": {k: full[keys.index("p@m" if k == "mean_p@n" else k)] for k in K}}
for j, d in D_.items():
    x = json.load(open(d + "/result.json"))["fused"]
    t9[j] = {k: 100 * x[k] for k in K}

# ---------------- Table 6
tiers = json.load(open(f"{P}/nyt10/tiers_main.json"))["tier_of_relation"]
nyt = DS("nyt10", f"{P}/nyt10/test_full.jsonl", f"{C}/nyt10/entity_types.jsonl")
tier_of = {nyt.rel2id[r]: t for r, t in tiers.items() if r in nyt.rel2id}
V8 = load_cache(f"{C}/nyt10/test_full_votes.jsonl"); per_seed = []
for s in SEEDS:
    p, u = nyt.probs(f"{RUNS}/nyt10_{FN}_8b_s{s}_v3/eval_test_full"); assert u == nyt.uids
    per_seed.append(E.evaluate(nyt.fuse(p, u, V8, CMAX["nyt10"]), nyt.gold, nyt.bags, nyt.K, na_id=nyt.na, tier_of_class=tier_of)["tier_macro_f1"])
cols = {"full": {t: {"macro_f1": float(np.mean([r[t]["macro_f1"] for r in per_seed])), "n_relations": v["n_relations"]}
                 for t, v in per_seed[0].items()}}
for j in ("no_anchor", "no_freq"):
    cols[j] = json.load(open(D_[j] + "/result.json"))["fused"]["tier_macro_f1"]
n_train = collections.Counter(t for r, t in tiers.items() if r != "NA")       # relations with at least one training sentence
t6 = {t: {"n_rel_train": n_train.get(t, 0), "n_rel_test": cols["full"][t]["n_relations"],
          **{c: 100 * v[t]["macro_f1"] for c, v in cols.items() if t in v}} for t in cols["full"]}

json.dump({"table9": t9, "table6": t6}, open(f"{O}/summary.json", "w"), indent=1)
print("Table 9  variant | AUC-PR P@100 MeanP@N Micro-F1 Macro-F1")
for j, r in t9.items():
    print(j, "|", " ".join(f"{r[k]:.2f}" for k in K))
print("Table 6  tier | #rel train / test | full  w/o init  w/o freq")
for t, r in t6.items():
    print(t, "|", r["n_rel_train"], r["n_rel_test"], "|", " ".join(f"{r.get(c, np.nan):.2f}" for c in ("full", "no_anchor", "no_freq")))
