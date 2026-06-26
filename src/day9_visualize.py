"""
MirrorProbe - Day 9 Visualization
=====================================
Compares Control 2 results across both models:
- GPT-2 Small (base model, degenerate outputs - weaker test)
- TinyLlama 1.1B Chat (instruction-tuned, real stances - stronger test)

Shows probe accuracy on output-similar held-out pairs per layer.
The TinyLlama result is the cleaner, more meaningful one.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("data/day6_control2_results.json", "r", encoding="utf-8-sig") as f:
    gpt2_c2 = json.load(f)

with open("data/day8_control2_redo_results.json", "r", encoding="utf-8-sig") as f:
    tinyllama_c2 = json.load(f)

gpt2_layers = sorted([int(k.split("_")[1]) for k in gpt2_c2.keys()])
tiny_layers = sorted([int(k.split("_")[1]) for k in tinyllama_c2.keys()])

gpt2_acc = [gpt2_c2[f"layer_{l}"] for l in gpt2_layers]
tiny_acc = [tinyllama_c2[f"layer_{l}"] for l in tiny_layers]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(gpt2_layers, gpt2_acc, color="#3b82f6", linewidth=2, marker="o", markersize=5)
axes[0].axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
axes[0].fill_between(gpt2_layers, 0.5, gpt2_acc, alpha=0.15, color="#3b82f6")
axes[0].set_title("GPT-2 Small — Control 2\n(base model, degenerate outputs — weaker test)", fontsize=11)
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Accuracy on output-similar pairs")
axes[0].set_ylim(0.3, 1.05)
axes[0].legend()
axes[0].set_xticks(gpt2_layers)

axes[1].plot(tiny_layers, tiny_acc, color="#10b981", linewidth=2, marker="o", markersize=5)
axes[1].axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
axes[1].fill_between(tiny_layers, 0.5, tiny_acc, alpha=0.15, color="#10b981")
axes[1].set_title("TinyLlama 1.1B Chat — Control 2 redo\n(instruction-tuned, real stances — stronger test)", fontsize=11)
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Accuracy on output-similar pairs")
axes[1].set_ylim(0.3, 1.05)
axes[1].legend()
axes[1].set_xticks(tiny_layers[::2])

fig.suptitle(
    "MirrorProbe Day 9: Internal pressure signal survives identical outputs\n"
    "Probe trained on 35 pairs, tested on 15 most output-similar pairs per model",
    fontsize=12
)
plt.tight_layout()
plt.savefig("data/day9_control2_comparison.png", dpi=150)
print("Saved to data/day9_control2_comparison.png")