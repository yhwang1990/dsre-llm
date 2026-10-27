"""Table 7 (backbones), Table 8 (low-resource) and the data of Figure 6 (Wiki80 robustness), CPU only.
Fusion uses the c_max of each dataset (NYT10 0.25, GDS 0.05, Wiki80 selected on the noisy validation set by w80_cmax_select.py),
with votes produced by the same backbone. NYT10 rows of Tables 7 and 8 are evaluated on the 20% bag-level test subset.
Single-seed cells report the std of 200 bag-level bootstrap resamples. Output: results/tables_cpu.json"""
import json, os
import numpy as np
from common import DS, P, C, RUNS, RES, FN, CMAX, FTLR, SEEDS, BEST, load_cache, subset_rows, pick

nyt = DS("nyt10", f"{P}/nyt10/test_full.jsonl", f"{C}/nyt10/entity_types.jsonl")
sub = DS("nyt10", f"{P}/nyt10/test_sub20.jsonl", f"{C}/nyt10/entity_types.jsonl")
rows = subset_rows(nyt, sub)


def merged(*paths):
    v = {}
    for f in paths:
        v.update({k: x for k, x in load_cache(f).items() if k not in v})
    return v


V8 = merged(f"{C}/nyt10/test_full_votes.jsonl", f"{C}/nyt10/test_sub20_votes.jsonl")
V4 = load_cache(f"{C}/nyt10/test_sub20_4b_votes.jsonl")
VL = load_cache(f"{C}/nyt10/test_full_llama_votes.jsonl")


def nyt_full_on_sub(d, votes=None, c=CMAX["nyt10"], boot=0):
    """A run evaluated on the full NYT10 test set, scored on the 20% subset."""
    p, u = nyt.probs(d); assert u == nyt.uids
    if votes is not None:
        p = nyt.fuse(p, u, votes, c)
    return pick(nyt.ev(p[rows], rows, boot=boot))


def nyt_sub(d, votes=None, c=CMAX["nyt10"], boot=0):
    """A run evaluated directly on the 20% subset (eval_test_sub20)."""
    p, u = sub.probs(d); assert u == sub.uids
    if votes is not None:
        p = sub.fuse(p, u, votes, c)
    return pick(sub.ev(p, boot=boot))


def summarize(runs):
    A = np.array([[r["auc_pr"], r["macro_f1"]] for r in runs])
    return {"runs": runs, "mean": A.mean(0).tolist(), "std": (A.std(0, ddof=1) if len(runs) > 1 else 0 * A[0]).tolist()}


out = {"table7": {}, "table8": {}, "fig6": {}}

# ---------------- Table 7: backbones (NYT10 20% subset at c_max 0.25; GDS at c_max 0.05)
t7 = out["table7"]
t7["nyt10_qwen3-8b"] = summarize([nyt_full_on_sub(f"{RUNS}/nyt10_{FN}_8b_s{s}_v3/eval_test_full", V8) for s in SEEDS])
t7["nyt10_qwen3-4b"] = summarize([nyt_sub(f"{d}/eval_test_sub20", V4) for d in
                                  (f"{RUNS}/nyt10_{FN}_4b_s13_v3", f"{RUNS}/nyt10_{FN}_4b_s42_v3", f"{RUNS}/nyt10_{FN}_4b_s123_v3")])
t7["nyt10_llama-3-8b"] = summarize([nyt_full_on_sub(f"{RUNS}/nyt10_{FN}_llama3_s42_v3/eval_test_full", VL, boot=200)])
for b, d in (("qwen3-8b", "bl_zeroshot_probcal_8b_sub20"), ("qwen3-4b", "bl_zeroshot_probcal_4b_sub20")):
    r = json.load(open(f"{RUNS}/{d}/result.json")); r = r.get("result", r)
    t7[f"nyt10_zeroshot_{b}"] = {"auc_pr": 100 * r["auc_pr"], "macro_f1": 100 * r["macro_f1"]}
gds = DS("gids", f"{P}/gids/test.jsonl", f"{C}/gids/entity_types.jsonl")


def gds_run(d, votes, boot=0):
    p, u = gds.probs(d); assert u == gds.uids
    return pick(gds.ev(gds.fuse(p, u, votes, CMAX["gids"]), boot=boot))


for b, tag, votes in (("qwen3-8b", "8b", f"{C}/gids/test_votes.jsonl"), ("qwen3-4b", "4b", f"{C}/gids/test_4b_votes.jsonl")):
    V = load_cache(votes)
    t7[f"gids_{b}"] = summarize([gds_run(f"{RUNS}/gids_{FN}_{tag}_s{s}_v3/eval_test", V) for s in SEEDS])
t7["gids_llama-3-8b"] = summarize([gds_run(f"{RUNS}/gids_{FN}_llama3_s42_v3/eval_test", load_cache(f"{C}/gids/test_llama_votes.jsonl"), boot=200)])
for k, v in t7.items():
    print("T7", k, np.round(v["mean"], 2).tolist() if "mean" in v else v, np.round(v["std"], 2).tolist() if "std" in v else "", flush=True)

# ---------------- Table 8: low-resource (NYT10 20% subset, single seed 42 + bootstrap)
t8 = out["table8"]
for name, lr, m in (("PCNN+ATT", "pcnn_att", "pcnn"), ("BERT-Focal", "bert-focal", "focal"), ("RoBERTa-MIL", "roberta-mil", "mil")):
    t8[name] = {"1pct": nyt_full_on_sub(f"{RUNS}/lowres_1pct_bl_{lr}_s42", boot=200),
                "5pct": nyt_full_on_sub(f"{RUNS}/lowres_5pct_bl_{lr}_s42", boot=200),
                "10pct": nyt_full_on_sub(f"{RUNS}/tune/nyt10_{m}_{BEST[('nyt10', m)]}_s42", boot=200)}
t8["Qwen3-8B-FT"] = {"1pct": nyt_sub(f"{RUNS}/lowres_1pct_ft_lr{FTLR}_8b_s42_v4/eval_test_sub20", boot=200),
                     "5pct": nyt_sub(f"{RUNS}/lowres_5pct_ft_lr{FTLR}_8b_s42_v4/eval_test_sub20", boot=200),
                     "10pct": nyt_full_on_sub(f"{RUNS}/nyt10_ft_lr{FTLR}_8b_s42_v3/eval_test_full", boot=200)}
t8["DSRE-LLM (Ours)"] = {"1pct": nyt_sub(f"{RUNS}/lowres_1pct_{FN}_8b_s42_v4/eval_test_sub20", V8, boot=200),
                         "5pct": nyt_sub(f"{RUNS}/lowres_5pct_{FN}_8b_s42_v4/eval_test_sub20", V8, boot=200),
                         "10pct": nyt_full_on_sub(f"{RUNS}/nyt10_{FN}_8b_s42_v3/eval_test_full", V8, boot=200)}
for k, v in t8.items():
    print("T8", k, {f: (round(x["auc_pr"], 2), round(x["macro_f1"], 2), round(x["std_macro_f1"], 2)) for f, x in v.items()}, flush=True)

# ---------------- Figure 6: Wiki80 controlled noise (Macro-F1 + bootstrap)
w = DS("wiki80", f"{P}/wiki80/test.jsonl", f"{C}/wiki80/entity_types.jsonl")
TV = load_cache(f"{C}/wiki80/test8b_votes.jsonl"); sel = json.load(open(f"{RES}/w80_cmax_select.json"))   # written by ../w80_cmax_select.py
f6 = out["fig6"]
KW = ("macro_f1", "micro_f1_nonNA")


def w80(d, votes=None, c=0.0, boot=200):
    p, u = w.probs(d); assert u == w.uids
    if votes is not None:
        p = w.fuse(p, u, votes, c)
    return pick(w.ev(p, boot=boot), KW)


for st in ["sym_0"] + [f"{k}_{r}" for k in ("sym", "asym") for r in (10, 20, 30, 40)]:
    for m, lab in (("pcnn_att", "PCNN+ATT"), ("bert-focal", "BERT-Focal"), ("roberta-mil", "RoBERTa-MIL")):
        f6.setdefault(lab, {})[st] = w80(f"{RUNS}/w80_{st}_bl_{m}_s42", boot=200)
    d = f"{RUNS}/w80_{st}_ft_lr{FTLR}_8b_s42_v4/eval_test"
    if os.path.exists(d + "/probs.npy"):
        f6.setdefault("Qwen3-8B-FT", {})[st] = w80(d, boot=200)
    d = f"{RUNS}/w80_{st}_{FN}_8b_s42_v4/eval_test"
    if os.path.exists(d + "/probs.npy") and st in sel:
        c = sel[st]["selected_cmax"]
        f6.setdefault("DSRE-LLM (Ours)", {})[st] = dict(w80(d, TV, c, boot=200), c_max=c)
for k, v in f6.items():
    print("F6", k, {s: (round(x["macro_f1"], 2), round(x.get("std_macro_f1", 0), 2)) for s, x in v.items()}, flush=True)
json.dump(out, open(f"{RES}/tables_cpu.json", "w"), indent=1)
