"""Figure 5 (PR curves on NYT10 and GDS) and Figure 7 (long-tail evaluation on NYT10) from the output of fig_data.py.
Usage: python plot_figs.py [--data-dir results/figdata] [--out-dir results/figs]"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RES
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=RES + "/figdata"); ap.add_argument("--out-dir", default=RES + "/figs")
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
STYLE = {  # baselines as thin dashed lines, DSRE-LLM as a thick red solid line
    "PCNN+ATT": dict(color="#7f7f7f", ls="--"), "BERT-Focal": dict(color="#1f77b4", ls=":"), "RoBERTa-MIL": dict(color="#9467bd", ls="--"),
    "BERT-NLI": dict(color="#8c564b", ls="-."), "CGRE": dict(color="#2ca02c", ls="-."), "LLM-CG": dict(color="#17becf", ls="--"),
    "Qwen3-8B-ICL": dict(color="#ff7f0e", ls=":"), "Qwen3-8B-FT": dict(color="#bcbd22", ls="-."), "DSRE-LLM (Ours)": dict(color="#d62728", ls="-", lw=1.8)}
TITLE = {"nyt10": "NYT10", "gids": "GDS"}

def load(ds):
    f = os.path.join(a.data_dir, f"figdata_{ds}.json")
    return json.load(open(f)) if os.path.exists(f) else None

# ---------------- Figure 5
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
for ax, ds in zip(axes, ("nyt10", "gids")):
    d = load(ds)
    ax.set_title(f"PR Curve on {TITLE[ds]}", fontweight="bold"); ax.set_xlabel("Recall", fontweight="bold"); ax.set_ylabel("Precision", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", alpha=0.4)
    if not d:
        continue
    for name, st in STYLE.items():
        m = d["methods"].get(name)
        if not m or "pr" not in m: continue
        r, p = np.array(m["pr"]["recall"]), np.array(m["pr"]["precision"])
        ax.plot(r, p, label=f"{name} (AUC={m['pr']['auc']:.3f})", lw=st.get("lw", 1.1), ls=st["ls"], color=st["color"])
        if "Ours" in name: ax.fill_between(r, p, color=st["color"], alpha=0.08)
    ax.legend(loc="upper right" if ds == "nyt10" else "lower left", fontsize=7, framealpha=0.9)   # GDS curves sit high, so the legend goes bottom-left
fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "Fig5-PR-Curves.pdf")); fig.savefig(os.path.join(a.out_dir, "Fig5-PR-Curves.png"), dpi=130)

# ---------------- Figure 7 (NYT10)
d = load("nyt10")
if d:
    names = ["PCNN+ATT", "CGRE", "BERT-Focal", "Qwen3-8B-FT", "DSRE-LLM (Ours)"]
    colors = ["#e67e22", "#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    tiers = ["1-10", "11-30", "31-100", "101-300", "300+"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    w = 0.16; x = np.arange(len(tiers))
    for i, (n, c) in enumerate(zip(names, colors)):
        v = [100 * d["methods"][n]["tier_macro_f1"].get(t, np.nan) for t in tiers]
        a1.bar(x + (i - 2) * w, v, w, label=n, color=c)
    a1.set_xticks(x); a1.set_xticklabels(tiers); a1.set_xlabel("Training Instances per Relation"); a1.set_ylabel("Macro-F1 (%)")
    a1.set_title("(a) Macro-F1 by Relation Frequency", fontweight="bold"); a1.grid(True, axis="y", alpha=0.4); a1.legend(fontsize=8)
    freq = d["train_freq"]
    scored = [r for r in d["methods"]["DSRE-LLM (Ours)"]["per_class_f1"]]
    rare = sorted(scored, key=lambda r: (freq.get(r, 0), r))          # all relations present in the test set, by training frequency
    mk = ["v", "D", "^", "s", "o"]
    for n, c, m in zip(names, colors, mk):
        v = [100 * d["methods"][n]["per_class_f1"].get(r, 0.0) for r in rare]
        a2.plot(range(1, len(rare) + 1), v, marker=m, ms=4, color=c, lw=1.6 if "Ours" in n else 1.1, ls="-" if "Ours" in n else "--", label=n)
    a2.set_xlabel("Relation Category (sorted by training frequency)"); a2.set_ylabel("F1 Score (%)"); a2.set_xticks(range(1, len(rare) + 1, 3))
    a2.set_title("(b) Per-Relation F1 (%d test relations)" % len(rare), fontweight="bold"); a2.grid(True, alpha=0.4); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "Fig7-Longtail-Eval.pdf")); fig.savefig(os.path.join(a.out_dir, "Fig7-Longtail-Eval.png"), dpi=130)
    json.dump({"rare_relations": [(r, freq.get(r, 0)) for r in rare]}, open(os.path.join(a.out_dir, "Fig7_rare_relations.json"), "w"), indent=1)
print("done")
