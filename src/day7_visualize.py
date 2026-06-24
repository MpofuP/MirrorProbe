"""
MirrorProbe - Day 7 Visualization
=====================================
Plots multi-seed mean accuracy +/- std per layer for both real
probe and shuffled control. This is the statistically robust version
of the Day 5 chart - error bars show stability across 10 seeds.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("data/day7_multiseed_results.json", "r", encoding="utf-8-sig") as f:
    results = json.load(f)

layers = list(range(12))
real_means = [results["real_probe"][f"layer_{l}"]["mean"] for l in layers]
real_stds  = [results["real_probe"][f"layer_{l}"]["std"]  for l in layers]
ctrl_means = [results["control_shuffled"][f"layer_{l}"]["mean"] for l in layers]
ctrl_stds  = [results["control_shuffled"][f"layer_{l}"]["std"]  for l in layers]

x = np.arange(len(layers))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))

bars1 = ax.bar(x - width/2, real_means, width,
               yerr=real_stds, capsize=4,
               label="Real probe (mean ± std, 10 seeds)",
               color="#3b82f6", alpha=0.9)
bars2 = ax.bar(x + width/2, ctrl_means, width,
               yerr=ctrl_stds, capsize=4,
               label="Shuffled-label control (mean ± std, 10 seeds)",
               color="#9ca3af", alpha=0.9)

ax.axhline(y=0.5, color="red", linestyle="--",
           linewidth=1.2, label="Chance level (50%)")

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Test accuracy", fontsize=12)
ax.set_title(
    "MirrorProbe Day 7: Pressure-detection probe vs. shuffled control\n"
    "(GPT-2 Small, n=100, 10 random seeds, error bars = ±1 std dev)",
    fontsize=12
)
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("data/day7_multiseed_chart.png", dpi=150)
print("Saved to data/day7_multiseed_chart.png")