"""Figure 6: Macro-F1 under controlled label noise on Wiki80 (symmetric / asymmetric), from results/tables_cpu.json.
All models are trained at 0-40% noise in steps of 10%; error bars are bag-level bootstrap standard deviations (single seed).
Output: results/figs/Fig6-Robustness-Noise.{pdf,png}"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
STYLE = {"DSRE-LLM (Ours)": dict(color="#e74c3c", ls="-", marker="o", lw=2.2),
         "Qwen3-8B-FT": dict(color="#9b59b6", ls="--", marker="s"),
         "BERT-Focal": dict(color="#3498db", ls="-.", marker="^"),
         "RoBERTa-MIL": dict(color="#2ecc71", ls=":", marker="D"),
         "PCNN+ATT": dict(color="#e67e22", ls="--", marker="v")}
F = json.load(open(f"{RES}/tables_cpu.json"))["fig6"]
RATES = [0, 10, 20, 30, 40]
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
for ax, kind, title, xl in ((axes[0], "sym", "(a) Symmetric Label Noise", "Symmetric Noise Rate (%)"),
                            (axes[1], "asym", "(b) Asymmetric Label Noise", "Asymmetric Noise Rate (%)")):
    for name, st in STYLE.items():
        pts = [(r, F[name][k]) for r in RATES for k in ["sym_0" if r == 0 else f"{kind}_{r}"] if k in F.get(name, {})]
        if not pts:
            continue
        xs = [r for r, _ in pts]; ys = [v["macro_f1"] for _, v in pts]; es = [v["std_macro_f1"] for _, v in pts]
        ax.errorbar(xs, ys, yerr=es, label=name, lw=st.get("lw", 1.6), ls=st["ls"], color=st["color"], marker=st["marker"], ms=7, capsize=2.5)
    ax.set_xticks(RATES); ax.set_xlabel(xl, fontsize=12); ax.set_ylabel("Macro-F1 (%)", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(10, 95); ax.set_yticks([10, 30, 50, 70, 90]); ax.set_xlim(-1, 41); ax.grid(alpha=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9, edgecolor="#999999")
fig.tight_layout()
os.makedirs(f"{RES}/figs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{RES}/figs/Fig6-Robustness-Noise.{ext}", dpi=300, bbox_inches="tight")
print("done")
