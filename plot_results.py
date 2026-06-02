import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_PATH = "results.json"
OUT_PATH = "xai_output/results_comparison.png"

LABELS = {
    "nn_lpq":         "LPQ",
    "nn_gradients":   "Gradients",
    "nn_raw_gray":    "Raw Gray",
    "nn_high_freq":   "High Freq",
    "nn_spatial_lbp": "Spatial LBP",
}
COLORS = {"APCER": "#E05C5C", "BPCER": "#5C9AE0", "ACER": "#5CB87A"}

with open(RESULTS_PATH) as f:
    results = json.load(f)

modes = list(results.keys())
display = [LABELS.get(m, m) for m in modes]
metrics = ["APCER", "BPCER", "ACER"]

x = np.arange(len(modes))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in modes]
    bars = ax.bar(x + i * width, vals, width, label=metric,
                  color=COLORS[metric], edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}", ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(display, fontsize=10)
ax.set_ylabel("Error Rate (%)", fontsize=11)
ax.set_title("Liveness Detection — APCER / BPCER / ACER per Feature Mode\n(CASIA-FASD, lower is better)", fontsize=12)
ax.set_ylim(0, max(results[m]["BPCER"] for m in modes) + 8)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=10, framealpha=0.8)
ax.spines[['top', 'right']].set_visible(False)

import os; os.makedirs("xai_output", exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_PATH}")
