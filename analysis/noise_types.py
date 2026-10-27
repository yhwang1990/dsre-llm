"""Table 5: confidence weights by noise type (Wiki80, 40% symmetric / asymmetric noise, Qwen3-8B, weights dumped at the end of training).
Groups (only injected-noise samples; "clean" = samples without injected noise):
  confident mismatch: the nearest prototype is not the annotated label (p_label < p_max) and the prototype entropy H is at most
                      the median H over all training samples (more concentrated than a typical sample);
  ambiguous noise   : all other noisy samples.
The threshold depends only on the distribution of H, not on the labels. W is the dumped weight (entropy weight x agreement factor a);
"entropy only" is W / a. AUROC uses the group as positives and the clean samples as negatives, with score 1 - W.
Input: runs/w80_<setting>_cwf5l1_8b_s42_v4/weights_dump.jsonl (train.py --dump-weights). Output: results/noise_types.json"""
import json
import numpy as np
from common import RUNS, RES, FN


def auroc(score_pos, score_neg):
    """AUROC with average ranks for ties; score_pos (noisy) should be higher than score_neg (clean)."""
    s = np.concatenate([score_pos, score_neg]); y = np.concatenate([np.ones(len(score_pos)), np.zeros(len(score_neg))])
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    n1, n0 = len(score_pos), len(score_neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


out = {}
for st in ("asym_40", "sym_40"):
    rows = {}
    for l in open(f"{RUNS}/w80_{st}_{FN}_8b_s42_v4/weights_dump.jsonl"):
        r = json.loads(l)
        if r["p_label"] >= 0:
            rows[r["uid"]] = r
    rows = list(rows.values()); Hmed = float(np.median([r["H"] for r in rows]))
    clean = [r for r in rows if not r["noisy"]]; noisy = [r for r in rows if r["noisy"]]
    is_conf = lambda r: r["p_label"] < r["p_max"] and r["H"] <= Hmed
    conf = [r for r in noisy if is_conf(r)]; amb = [r for r in noisy if not is_conf(r)]
    W = lambda rs: np.array([r["W"] for r in rs]); We = lambda rs: np.array([r["W"] / max(r["a"], 1e-9) for r in rs])
    res = {"H_median": Hmed, "n_total": len(rows)}
    for g, rs in (("clean", clean), ("ambiguous", amb), ("confident", conf)):
        res[g] = {"n": len(rs), "median_W": float(np.median(W(rs))) if rs else None}
        if g != "clean":
            res[g]["auroc_1-W"] = auroc(1 - W(rs), 1 - W(clean)); res[g]["auroc_entropy_only"] = auroc(1 - We(rs), 1 - We(clean))
            res[g]["auroc_1-a"] = auroc(1 - np.array([r["a"] for r in rs]), 1 - np.array([r["a"] for r in clean]))
    out[st] = res
json.dump(out, open(f"{RES}/noise_types.json", "w"), indent=1)
print(json.dumps(out, indent=1))
