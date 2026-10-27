"""Tables 2 and 3 (NYT10 / GDS test sets): baselines, Qwen3-8B-ICL, Qwen3-8B-FT and DSRE-LLM, 3 seeds, mean ± sample std.
DSRE-LLM is fused at the validation-selected c_max of each dataset; Qwen3-8B-FT is evaluated on the classifier output only.
P@M is the mean of P@100/200/300, computed per run and then averaged over seeds.
Output: results/table2.json"""
import json, os
import numpy as np
from common import DS, P, C, RUNS, RES, FN, CMAX, FTLR, SEEDS, BEST, load_cache

KEYS = ["auc_pr", "p@100", "p@200", "p@300", "micro_f1_nonNA", "macro_f1"]
CFG = {"nyt10": dict(test=f"{P}/nyt10/test_full.jsonl", votes=f"{C}/nyt10/test_full_votes.jsonl", ev="eval_test_full", bl=""),
       "gids": dict(test=f"{P}/gids/test.jsonl", votes=f"{C}/gids/test_votes.jsonl", ev="eval_test", bl="gids_")}
DATA = {}


def dataset(ds):
    if ds not in DATA:
        DATA[ds] = (DS(ds, CFG[ds]["test"], f"{C}/{ds}/entity_types.jsonl"), load_cache(CFG[ds]["votes"]))
    return DATA[ds]


def result(path, key="result"):
    return json.load(open(path))[key] if os.path.exists(path) else None


def ours(ds, s):
    d = f"{RUNS}/{ds}_{FN}_8b_s{s}_v3/{CFG[ds]['ev']}"
    if not os.path.exists(f"{d}/probs.npy"):
        return None
    data, votes = dataset(ds)
    p, u = data.probs(d)
    assert u == data.uids
    return data.ev(data.fuse(p, u, votes, CMAX[ds]))


def baseline(m):
    return lambda ds, s: result(f"{RUNS}/tune/{ds}_{m}_{BEST[(ds, m)]}_s{s}/result.json")


ROWS = [("PCNN+ATT", baseline("pcnn")), ("BERT-Focal", baseline("focal")), ("RoBERTa-MIL", baseline("mil")),
        ("BERT-NLI", baseline("nli")), ("CGRE", baseline("cgre")), ("LLM-CG", baseline("llmcg")),
        # retrieved demonstrations + validation calibration; deterministic, a single run
        ("Qwen3-8B-ICL", lambda ds, s: result(f"{RUNS}/{CFG[ds]['bl']}bl_knncal_8b/result.json") if s == 42 else None),
        ("Qwen3-8B-FT", lambda ds, s: result(f"{RUNS}/{ds}_ft_lr{FTLR}_8b_s{s}_v3/{CFG[ds]['ev']}/result.json", "classifier_only")),
        ("DSRE-LLM (Ours)", ours)]

out = {}
for name, fn in ROWS:
    out[name] = {}
    for ds in CFG:
        rs = [r for r in (fn(ds, s) for s in SEEDS) if r]
        if not rs:
            out[name][ds] = None; continue
        A = 100 * np.array([[r[k] for k in KEYS] + [(r["p@100"] + r["p@200"] + r["p@300"]) / 3] for r in rs])
        out[name][ds] = dict(n=len(rs), mean=A.mean(0).tolist(), std=(A.std(0, ddof=1) if len(rs) > 1 else 0 * A[0]).tolist())

json.dump(dict(cmax=CMAX, keys=KEYS + ["p@m"], rows=out), open(f"{RES}/table2.json", "w"), indent=1)
print("%-18s %-6s " % ("method", "data") + " ".join("%13s" % k for k in ["AUC-PR", "P@100", "Micro-F1", "Macro-F1", "P@M"]))
for name, v in out.items():
    for ds, x in v.items():
        if x:
            cells = [(x["mean"][i], x["std"][i]) for i in (0, 1, 4, 5, 6)]
            print("%-18s %-6s " % (name, ds) + " ".join("%7.2f±%5.2f" % c for c in cells))
