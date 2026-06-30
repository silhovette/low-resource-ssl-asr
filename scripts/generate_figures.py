"""Generate publication-quality figures for the NeurIPS paper."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import csv
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Style: match LaTeX Times body text at 10pt ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1.3,
    "pdf.fonttype": 42,     # embed fonts as TrueType/Type1 (not outline)
    "ps.fonttype": 42,
})

SYSTEMS = {
    "mel_1h_hf_cuda":         ("Log-mel",        "#d62728", "-"),
    "wav2vec2_1h_frozen_hf_cuda": ("wav2vec2 last",  "#1f77b4", "-"),
    "wav2vec2_1h_layer6_hf_cuda": ("wav2vec2 layer 6","#ff7f0e", "-"),
    "hubert_1h_frozen_hf_cuda":   ("HuBERT last",     "#2ca02c", "-"),
}
COLORS = [v[1] for v in SYSTEMS.values()]
LABELS = [v[0] for v in SYSTEMS.values()]

# ============================================================
# Figure 1: Training curves — loss + dev WER side-by-side
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6))

for key, (label, color, _) in SYSTEMS.items():
    m = json.loads((Path("outputs") / key / "metrics.json").read_text("utf-8"))
    epochs = [h["epoch"] for h in m["history"]]
    loss = [h["train_loss"] for h in m["history"]]
    wer = [h["dev_wer"] for h in m["history"]]
    ax1.plot(epochs, loss, color=color, marker="o", ms=3, label=label)
    ax2.plot(epochs, wer, color=color, marker="o", ms=3, label=label)

ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train CTC Loss")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dev WER")
ax1.legend(fontsize=8, framealpha=0.7, loc="upper right")
ax2.legend(fontsize=8, framealpha=0.7, loc="upper right")
ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
fig.tight_layout()
fig.savefig(OUT / "training_curves.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2: WER / CER bar chart across systems
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 2.4))
x = np.arange(len(SYSTEMS))
width = 0.32

wer_vals, cer_vals, rtf_vals = [], [], []
for key in SYSTEMS:
    m = json.loads((Path("outputs") / key / "metrics.json").read_text("utf-8"))
    wer_vals.append(m.get("test_wer", m.get("best_dev_wer", 0)))
    cer_vals.append(m.get("test_cer", m.get("best_dev_cer", 0)))
    rtf_vals.append(m.get("test_rtf", m.get("dev_rtf", 0)))

b1 = ax.bar(x - width/2, wer_vals, width, color="#d62728", edgecolor="white", linewidth=0.3, label="Test WER")
b2 = ax.bar(x + width/2, cer_vals, width, color="#1f77b4", edgecolor="white", linewidth=0.3, label="Test CER")
ax.bar_label(b1, fmt=lambda v: f"{v:.3f}" if v < 1 else "1.000", fontsize=5.5, padding=2)
ax.bar_label(b2, fmt=lambda v: f"{v:.3f}" if v < 1 else "1.000", fontsize=5.5, padding=2)

ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8)
ax.set_ylabel("Error Rate"); ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
fig.tight_layout()
fig.savefig(OUT / "wer_cer_bars.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 3: Error type breakdown (subs / dels / ins) — donut + legend
# ============================================================
from src.metrics import aggregate_error_stats

# Only the three SSL systems (skip log-mel since it's all deletions)
ssl_systems = {k: v for k, v in SYSTEMS.items() if k != "mel_1h_hf_cuda"}
ssl_labels = [v[0] for v in ssl_systems.values()]
ssl_colors_list = [v[1] for v in ssl_systems.values()]

fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.6))
wedge_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
cat_names = ["Substitutions", "Deletions", "Insertions", "Correct"]
error_data = {}

for key, (label, color, _) in ssl_systems.items():
    preds = list(csv.DictReader(open(f"outputs/{key}/test_predictions.csv", encoding="utf-8")))
    total_subs = total_dels = total_ins = total_correct = total_ref = 0
    for row in preds:
        s = aggregate_error_stats(row["ref"], row["hyp"])
        total_subs += s["subs"]; total_dels += s["dels"]
        total_ins += s["ins"]; total_correct += s["correct"]
        total_ref += s["ref_words"]
    error_data[label] = {
        "subs": total_subs, "dels": total_dels, "ins": total_ins, "correct": total_correct
    }

for ax, (label, color) in zip(axes, zip(ssl_labels, ssl_colors_list)):
    d = error_data[label]
    total = d["subs"] + d["dels"] + d["ins"] + d["correct"]
    sizes = [d["subs"]/total, d["dels"]/total, d["ins"]/total, d["correct"]/total]

    def make_autopct(sizes):
        def autopct(pct):
            return f"{pct:.0f}%" if pct >= 5 else ""
        return autopct

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,           # no direct wedge labels — use legend instead
        colors=wedge_colors,
        autopct=make_autopct(sizes),
        pctdistance=0.70,
        startangle=90,
        textprops={"fontsize": 8, "fontweight": "bold"},
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    ax.set_title(label, fontsize=9, fontweight="bold", color=color)

# Shared legend below the three pies, one row
fig.legend(
    wedges, cat_names,
    loc="lower center", ncol=4,
    fontsize=8, framealpha=0.7,
    bbox_to_anchor=(0.5, -0.02),
)
fig.suptitle("Word-Level Error Breakdown per System", fontsize=10, y=1.02)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT / "error_breakdown.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 4: System architecture schematic (as simple flow chart via matplotlib)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 1.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 3)
ax.axis("off")

boxes = [
    (0.5, 1.5, 1.2, 0.7, "Raw Audio\n(16 kHz)", "#e8e8e8"),
    (2.3, 1.5, 1.4, 0.7, "SSL Encoder\n(frozen)", "#c5e0b4"),
    (4.3, 1.5, 1.2, 0.7, "CTC Projection\nHead", "#bdd7ee"),
    (6.1, 1.5, 1.2, 0.7, "CTC Greedy\nDecode", "#f4b4c2"),
    (7.9, 1.5, 1.2, 0.7, "Text\nOutput", "#e8e8e8"),
]
for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y - h/2), w, h, facecolor=color, edgecolor="black",
                          linewidth=0.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y, text, ha="center", va="center", fontsize=6.5, zorder=3)

# arrows
arrow_y = 1.5
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i+1][0]
    ax.annotate("", xy=(x2, arrow_y), xytext=(x1, arrow_y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# annotation
ax.text(3.0, 2.5, "wav2vec 2.0 / HuBERT base", fontsize=7, ha="center", style="italic",
        color="#2ca02c")
ax.annotate("", xy=(3.0, 2.3), xytext=(3.0, 2.2),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.6))

ax.set_title("System Architecture Overview", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 5: Layer comparison — wav2vec2 last vs layer 6 side-by-side
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 2.0))
x = np.arange(2)
width = 0.28
keys = ["wav2vec2_1h_frozen_hf_cuda", "wav2vec2_1h_layer6_hf_cuda"]
lbls = ["wav2vec2\nLast Layer", "wav2vec2\nLayer 6"]
mets = []
for key in keys:
    m = json.loads((Path("outputs") / key / "metrics.json").read_text("utf-8"))
    mets.append((m["test_wer"], m["test_cer"]))

b1 = ax.bar(x - width/2, [m[0] for m in mets], width, color="#d62728", edgecolor="white", linewidth=0.3, label="Test WER")
b2 = ax.bar(x + width/2, [m[1] for m in mets], width, color="#1f77b4", edgecolor="white", linewidth=0.3, label="Test CER")
ax.bar_label(b1, fmt=lambda v: f"{v:.3f}", fontsize=8, padding=2)
ax.bar_label(b2, fmt=lambda v: f"{v:.3f}", fontsize=8, padding=2)
# delta annotation
delta_wer = mets[0][0] - mets[1][0]
delta_cer = mets[0][1] - mets[1][1]
ax.annotate(f"Δ WER = -{delta_wer:.3f}", xy=(0.8, 0.35), fontsize=7, color="green",
            ha="center", fontweight="bold")
ax.annotate(f"Δ CER = -{delta_cer:.3f}", xy=(0.8, 0.25), fontsize=7, color="green",
            ha="center", fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=8)
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax.set_title("Layer Ablation: wav2vec 2.0 Last vs Layer 6", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "layer_ablation.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 6: Per-epoch convergence table as a heatmap-style plot
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 1.8))
wer_matrix = []
row_labels = []
for key, (label, _, _) in SYSTEMS.items():
    m = json.loads((Path("outputs") / key / "metrics.json").read_text("utf-8"))
    wer_row = [h["dev_wer"] for h in m["history"]]
    # pad to same length
    max_epochs = 8
    while len(wer_row) < max_epochs:
        wer_row.append(np.nan)
    wer_matrix.append(wer_row)
    row_labels.append(label)

wer_matrix = np.array(wer_matrix)
masked = np.ma.masked_invalid(wer_matrix)
im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r", vmin=0.5, vmax=1.0)
ax.set_xticks(range(wer_matrix.shape[1]))
ax.set_xticklabels([f"Ep{e+1}" for e in range(wer_matrix.shape[1])], fontsize=7)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=7)
for i in range(wer_matrix.shape[0]):
    for j in range(wer_matrix.shape[1]):
        if not np.isnan(wer_matrix[i, j]):
            ax.text(j, i, f"{wer_matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=6, color="black" if wer_matrix[i,j] < 0.8 else "white")
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Dev WER", fontsize=7)
cbar.ax.tick_params(labelsize=6)
ax.set_title("Dev WER Over Training Epochs", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "convergence_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

print(f"Generated 6 figures in {OUT}/")
for f in sorted(OUT.glob("*.pdf")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
