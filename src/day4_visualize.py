"""
MirrorProbe - Day 4 Visualization
====================================
Plots REAL probe test accuracy vs CONTROL (shuffled-label) test
accuracy, per layer, side by side. This is the visual proof of
why we can't claim a real signal yet - the bars overlap too much.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("data/day4_probe_results.json", "r", encoding="utf-8-sig") as f:
    results = json.load(f)

layers = list(range(12))
real_acc = [results["real_probe"][f"layer_{l}"]["test_acc"] for l in layers]
control_acc = [results["control_shuffled"][f"layer_{l}"]["test_acc"] for l in layers]

x = np.arange(len(layers))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, real_acc, width, label="Real probe (pressure vs neutral)", color="#3b82f6")
ax.bar(x + width/2, control_acc, width, label="Control (shuffled labels)", color="#9ca3af")

ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Chance level (50%)")

ax.set_xlabel("Layer")
ax.set_ylabel("Test accuracy")
ax.set_title("Day 5: Real probe vs. shuffled-label control, per layer\n"
             "(n=100, 50 pairs - clear gap emerging, confound checks still needed)")
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig("data/day4_probe_vs_control.png", dpi=150)
print("Saved plot to data/day4_probe_vs_control.png")