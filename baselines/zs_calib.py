"""Calibration of the probabilistic zero-shot / ICL scores (parameters selected on validation data only; the test set is not used for selection):
zero-shot prompting assigns a non-NA relation to most NA sentences of NYT10 and prefers a few options (contains / administrative_divisions, ...),
so two training-free steps are applied: (1) option-prior correction p'(o) ~ p(o) / prior(o)^beta, prior = mean option probability on the validation subset (label-free, contextual calibration);
(2) NA scaling p'(NA) *= w. beta and w are selected on the validation subset by AUC + Mi-F1 + Ma-F1 and applied unchanged to the cached test probabilities.
Usage: python zs_calib.py --val-file val_sub.jsonl --val-cache ... --test-file test_full.jsonl --test-cache ... --rel2id ... --tiers ... --out <dir>"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D
from dsre import evaluate as E


def load(file, cache_f, rel2id):
    na = rel2id["NA"]; rows = [na] + [rel2id[r] for r in sorted(rel2id, key=rel2id.get) if r != "NA"]
    cache = {}
    for l in open(cache_f):
        d = json.loads(l); cache[d["k"]] = d["v"]
    X = [x for x in D.load_jsonl(file) if x["uid"] in cache]
    O = np.array([cache[x["uid"]] for x in X]) + 1e-8           # option order (0 = NA)
    gold = np.array([rel2id.get(x["relation"], na) for x in X]); bags = [D.bag_key(x) for x in X]
    return O, gold, bags, rows


def to_cls(O, rows, K):
    P = np.zeros((len(O), K)); P[:, rows] = O
    return P / P.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ["--val-file", "--val-cache", "--test-file", "--test-cache", "--rel2id", "--out"]:
        ap.add_argument(k, required=True)
    ap.add_argument("--tiers", default=None); ap.add_argument("--bootstrap", type=int, default=200)
    a = ap.parse_args()
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na = rel2id["NA"]
    Ov, gv, bv, rows = load(a.val_file, a.val_cache, rel2id)
    Ot, gt, bt, _ = load(a.test_file, a.test_cache, rel2id)
    prior = Ov.mean(0)
    grid = []
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for w in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
            Q = Ov / prior ** beta; Q[:, 0] *= w
            r = E.evaluate(to_cls(Q, rows, K), gv, bv, K, na_id=na)
            s = 100 * (r["auc_pr"] + r["micro_f1_nonNA"] + r["macro_f1"])
            grid.append(dict(beta=beta, w=w, score=s, auc=100 * r["auc_pr"], mi=100 * r["micro_f1_nonNA"], ma=100 * r["macro_f1"]))
            print("val beta %.2f w %4d  AUC %.2f Mi %.2f Ma %.2f score %.2f" % (beta, w, grid[-1]["auc"], grid[-1]["mi"], grid[-1]["ma"], s), flush=True)
    best = max(grid, key=lambda g: g["score"])
    print("selected on val:", best, flush=True)
    tier_of_class = None
    if a.tiers:
        t = json.load(open(a.tiers))["tier_of_relation"]; tier_of_class = {rel2id[r]: v for r, v in t.items() if r in rel2id}
    Q = Ot / prior ** best["beta"]; Q[:, 0] *= best["w"]; P = to_cls(Q, rows, K)
    res = E.evaluate(P, gt, bt, K, na_id=na, tier_of_class=tier_of_class)
    if a.bootstrap:
        res["bootstrap"] = E.bootstrap_ci(P, gt, bt, K, na_id=na, n_boot=a.bootstrap)
    res["answered_NA_frac"] = float((P.argmax(1) == na).mean())
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": "Qwen3-8B-ZeroShot-Prob-Calibrated", "selected": best, "val_grid": grid, "n_val": len(Ov), "result": res},
              open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
