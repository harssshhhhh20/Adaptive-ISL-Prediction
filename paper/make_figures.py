"""Paper figures + LaTeX tables, computed from the saved result files.  python paper/make_figures.py"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import stats as S  # noqa: E402

OUT = ROOT / "paper" / "figures"
FIXED, ADAPT = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (dataviz reference palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})


def jl(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / pattern)))]


# ------------------------------------------------------------------ collect per-seed numbers
inc50 = jl("results/include50/*.json")
inc24 = jl("results/include/*.json")
demo = json.load(open(ROOT / "results/demo/metrics.json"))["runs"]
loso = {c: S.pooled_wer_by_seed(S.load(c, "loso")) for c in ("G_concat", "L_trel_gate")}

settings = [
    ("INCLUDE-50\nsign chains", [r["test_sequences"]["wer"] for r in inc50 if r["fusion"] == "concat"],
     [r["test_sequences"]["wer"] for r in inc50 if r["fusion"] == "trel_gate"]),
    ("INCLUDE-24\nsign chains", [r["test_sequences"]["wer"] for r in inc24 if r["fusion"] == "concat"],
     [r["test_sequences"]["wer"] for r in inc24 if r["fusion"] == "trel_gate"]),
    ("ISL-CSLTR, 40 sent.\nvideo, signer-seen", [v["test"]["wer"] for k, v in demo.items() if k.startswith("fixed")],
     [v["test"]["wer"] for k, v in demo.items() if k.startswith("adaptive")]),
    ("ISL-CSLTR, 97 sent.\nunseen signer (LOSO)", list(loso["G_concat"]), list(loso["L_trel_gate"])),
]

# ------------------------------------------------------------------ Figure 1: fixed vs adaptive WER
fig, ax = plt.subplots(figsize=(6.8, 3.0), facecolor=SURF)
ax.set_facecolor(SURF)
x = np.arange(len(settings))
w = 0.36
for j, (lab, col, off) in enumerate([("Fixed fusion (concat)", FIXED, -w / 2 - 0.01), ("Adaptive fusion (proposed)", ADAPT, w / 2 + 0.01)]):
    vals = [s[1 + j] for s in settings]
    m = [np.mean(v) for v in vals]
    sd = [np.std(v, ddof=1) for v in vals]
    ax.bar(x + off, m, w, color=col, label=lab, zorder=2)
    ax.errorbar(x + off, m, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
    for xi, (mi, vi) in enumerate(zip(m, vals)):
        ax.scatter(np.full(len(vi), xi + off), vi, s=9, color=INK, zorder=4, linewidths=0)
        ax.text(xi + off, mi + max(sd[xi], 0) + 2.5, f"{mi:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
ax.hlines(82.5, 3 - 0.45, 3 + 0.45, colors=INK2, linestyles="--", linewidth=1, zorder=5)
ax.text(3 + 0.47, 82.5, "constant\n'I YOU'", va="center", fontsize=7, color=INK2)
ax.set_xticks(x, [s[0] for s in settings])
ax.set_ylabel("Test WER (%) ↓")
ax.set_ylim(0, 100)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("Fixed vs adaptive fusion across four settings (3 seeds; bars = mean, whiskers = SD, dots = seeds)",
             fontsize=8.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig_fixed_vs_adaptive.pdf"); fig.savefig(OUT / "fig_fixed_vs_adaptive.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: INCLUDE-50 accuracy vs published
fig, ax = plt.subplots(figsize=(3.4, 2.6), facecolor=SURF)
ax.set_facecolor(SURF)
for j, (fu, lab, col) in enumerate([("concat", "Fixed", FIXED), ("trel_gate", "Adaptive", ADAPT)]):
    v = [r["test_isolated_accuracy_closed"] for r in inc50 if r["fusion"] == fu]
    ax.scatter(np.full(len(v), j) + np.linspace(-0.06, 0.06, len(v)), v, s=36, color=col, zorder=3, edgecolors=SURF, linewidths=2)
    ax.hlines(np.mean(v), j - 0.2, j + 0.2, colors=col, linewidth=2, zorder=2)
    ax.text(j + 0.24, np.mean(v), f"{np.mean(v):.1f}", va="center", fontsize=8, color=INK)
ax.axhline(94.5, color=INK2, linestyle="--", linewidth=1)
ax.text(-0.45, 94.25, "INCLUDE paper (94.5)", va="top", fontsize=7, color=INK2)
ax.set_xticks([0, 1], ["Fixed\nfusion", "Adaptive\nfusion"])
ax.set_xlim(-0.5, 1.6)
ax.set_ylim(90, 100)
ax.set_ylabel("INCLUDE-50 test accuracy (%)")
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_include50_accuracy.pdf"); fig.savefig(OUT / "fig_include50_accuracy.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ LaTeX tables
def ms(v):
    return f"{np.mean(v):.1f} $\\pm$ {np.std(v, ddof=1):.1f}"


rows = []
for fu, lab in [("concat", "Fixed fusion"), ("trel_gate", "Adaptive fusion")]:
    rs = [r for r in inc50 if r["fusion"] == fu]
    rows.append(f"{lab} & {ms([r['test_isolated_accuracy_closed'] for r in rs])} & "
                f"{ms([100 - r['test_isolated_free']['wer'] for r in rs])} & {ms([r['test_sequences']['wer'] for r in rs])} & "
                f"{ms([r['test_sequences']['sent_acc'] for r in rs])} & 1.3M \\\\")
t1 = r"""\begin{table}[t]\centering\small
\caption{INCLUDE-50, official split (192 test clips). Mean $\pm$ SD over 3 seeds. Sequence metrics on 400 chains of test clips (no boundaries given).}
\label{tab:include50}
\begin{tabular}{lccccc}\toprule
Model & Acc. (closed) & Acc. (free) & Seq. WER$\downarrow$ & Exact seq. & Params\\\midrule
INCLUDE paper (best reported) & 94.5 & -- & -- & -- & --\\
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}"""

t2rows = [f"{s[0].replace(chr(10), ' ')} & {ms(s[1])} & {ms(s[2])} \\\\" for s in settings]
t2 = r"""\begin{table}[t]\centering\small
\caption{Test WER (\%, lower is better) of fixed vs adaptive fusion, mean $\pm$ SD over 3 seeds. A constant output (``I YOU'') scores 82.5 on the LOSO protocol.}
\label{tab:fusion}
\begin{tabular}{lcc}\toprule
Setting & Fixed fusion & Adaptive fusion\\\midrule
""" + "\n".join(t2rows) + r"""
\bottomrule\end{tabular}\end{table}"""
(ROOT / "paper" / "tables.tex").write_text(t1 + "\n\n" + t2 + "\n")
for s in settings:
    print(s[0].replace("\n", " "), "fixed", round(np.mean(s[1]), 2), "adaptive", round(np.mean(s[2]), 2))
