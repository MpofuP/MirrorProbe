"""
MirrorProbe - Day 6 Summary Visualization
============================================
Combines all four control results into one chart: GPT-2 real vs
control, Pythia real vs control, side by side, showing the pattern
replicates across architecture.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("data/day4_probe_results.json", "r", encoding="utf-8-sig") as f:
    gpt2_results = json.load(f)
with open("data/day6_control4_results.json", "r", encoding="utf-8-sig") as f:
    pythia_results = json.load(f)

layers = list(range(12))
gpt2_real = [gpt2_results["real_probe"][f"layer_{l}"]["test_acc"] for l in layers]
gpt2_control = [gpt2_results["control_shuffled"][f"layer_{l}"]["test_acc"] for l in layers]
pythia_real = [pythia_results["real_probe"][f"layer_{l}"] for l in layers]
pythia_control = [pythia_results["control_shuffled"][f"layer_{l}"] for l in layers]

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

x = np.arange(len(layers))
width = 0.35

axes[0].bar(x - width/2, gpt2_real, width, label="Real probe", color="#3b82f6")
axes[0].bar(x + width/2, gpt2_control, width, label="Shuffled control", color="#9ca3af")
axes[0].axhline(y=0.5, color="red", linestyle="--", linewidth=1)
axes[0].set_title("GPT-2 Small")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Test accuracy")
axes[0].set_xticks(x)
axes[0].legend()

axes[1].bar(x - width/2, pythia_real, width, label="Real probe", color="#10b981")
axes[1].bar(x + width/2, pythia_control, width, label="Shuffled control", color="#9ca3af")
axes[1].axhline(y=0.5, color="red", linestyle="--", linewidth=1)
axes[1].set_title("Pythia-160M (different architecture)")
axes[1].set_xlabel("Layer")
axes[1].set_xticks(x)
axes[1].legend()

fig.suptitle("MirrorProbe Day 6: Pressure-detection signal replicates across two architectures\n"
             "(blue/green = real probe, gray = random-label control, red line = chance)")
plt.tight_layout()
plt.savefig("data/day6_summary_two_models.png", dpi=150)
print("Saved summary chart to data/day6_summary_two_models.png")