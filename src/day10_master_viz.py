"""
MirrorProbe - Day 10
=====================
Master summary visualization combining all key results:
- GPT-2 Small multi-seed (0.92 +/- 0.03, 10 seeds)
- Pythia-160M cross-architecture replication
- TinyLlama Control 2 bootstrap (instruction-tuned, real stances)
- Control 1 lexical confound results

This is the single definitive chart for all applications and posts.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Load all results
with open("data/day7_multiseed_results.json", "r", encoding="utf-8-sig") as f:
    gpt2_multi = json.load(f)

with open("data/day6_control4_results.json", "r", encoding="utf-8-sig") as f:
    pythia = json.load(f)

with open("data/day9_tinyllama_bootstrap.json", "r", encoding="utf-8-sig") as f:
    tinyllama = json.load(f)

with open("data/day6_control1_results.json", "r", encoding="utf-8-sig") as f:
    control1 = json.load(f)

layers_12 = list(range(12))
layers_22 = list(range(22))

# GPT-2 multi-seed
gpt2_real_mean = [gpt2_multi["real_probe"][f"layer_{l}"]["mean"] for l in layers_12]
gpt2_real_std  = [gpt2_multi["real_probe"][f"layer_{l}"]["std"]  for l in layers_12]
gpt2_ctrl_mean = [gpt2_multi["control_shuffled"][f"layer_{l}"]["mean"] for l in layers_12]
gpt2_ctrl_std  = [gpt2_multi["control_shuffled"][f"layer_{l}"]["std"]  for l in layers_12]

# Pythia
pythia_real = [pythia["real_probe"][f"layer_{l}"] for l in layers_12]
pythia_ctrl = [pythia["control_shuffled"][f"layer_{l}"] for l in layers_12]

# TinyLlama bootstrap
tiny_mean = [tinyllama[f"layer_{l}"]["mean"] for l in layers_22]
tiny_std  = [tinyllama[f"layer_{l}"]["std"]  for l in layers_22]

# Control 1 lexical confound
ctrl1_pct = [control1[f"layer_{l}"] / 100 for l in layers_12]

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#0a0a0f")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor("#111118")
    ax.tick_params(colors="#8888a0", labelsize=9)
    ax.spines["bottom"].set_color("#333344")
    ax.spines["left"].set_color("#333344")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Panel 1: GPT-2 multi-seed ──
x = np.array(layers_12)
w = 0.35
ax1.bar(x - w/2, gpt2_real_mean, w, yerr=gpt2_real_std, capsize=3,
        label="Real probe", color="#6366f1", alpha=0.9)
ax1.bar(x + w/2, gpt2_ctrl_mean, w, yerr=gpt2_ctrl_std, capsize=3,
        label="Shuffled control", color="#4a4a5a", alpha=0.9)
ax1.axhline(0.5, color="#ef4444", linestyle="--", linewidth=1)
ax1.set_title("GPT-2 Small — Primary result\n(10 seeds, mean ± std)",
              color="#e8e8f0", fontsize=10, pad=8)
ax1.set_xlabel("Layer", color="#8888a0", fontsize=9)
ax1.set_ylabel("Test accuracy", color="#8888a0", fontsize=9)
ax1.set_ylim(0, 1.1)
ax1.set_xticks(x)
ax1.legend(fontsize=8, facecolor="#1a1a24", edgecolor="#333344",
           labelcolor="#e8e8f0")
ax1.text(10, 0.95, "0.92 ± 0.03", color="#6366f1", fontsize=9, ha="center")

# ── Panel 2: Pythia cross-architecture ──
ax2.plot(layers_12, pythia_real, color="#10b981", linewidth=2,
         marker="o", markersize=4, label="Real probe (Pythia-160M)")
ax2.plot(layers_12, pythia_ctrl, color="#4a4a5a", linewidth=2,
         marker="o", markersize=4, label="Shuffled control")
ax2.axhline(0.5, color="#ef4444", linestyle="--", linewidth=1)
ax2.fill_between(layers_12, 0.5, pythia_real, alpha=0.1, color="#10b981")
ax2.set_title("Pythia-160M — Cross-architecture replication\n(different model, different training data)",
              color="#e8e8f0", fontsize=10, pad=8)
ax2.set_xlabel("Layer", color="#8888a0", fontsize=9)
ax2.set_ylabel("Test accuracy", color="#8888a0", fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.set_xticks(layers_12)
ax2.legend(fontsize=8, facecolor="#1a1a24", edgecolor="#333344",
           labelcolor="#e8e8f0")

# ── Panel 3: TinyLlama bootstrap ──
ax3.plot(layers_22, tiny_mean, color="#f59e0b", linewidth=2,
         marker="o", markersize=3, label="Real probe (bootstrap mean)")
ax3.fill_between(layers_22,
                 [m - s for m, s in zip(tiny_mean, tiny_std)],
                 [m + s for m, s in zip(tiny_mean, tiny_std)],
                 alpha=0.2, color="#f59e0b", label="± 1 std (50 iterations)")
ax3.axhline(0.5, color="#ef4444", linestyle="--", linewidth=1)
ax3.set_title("TinyLlama 1.1B Chat — Control 2 redo\n(instruction-tuned, output-identical pairs, bootstrap)",
              color="#e8e8f0", fontsize=10, pad=8)
ax3.set_xlabel("Layer", color="#8888a0", fontsize=9)
ax3.set_ylabel("Accuracy on output-similar pairs", color="#8888a0", fontsize=9)
ax3.set_ylim(0, 1.1)
ax3.set_xticks(layers_22[::2])
ax3.legend(fontsize=8, facecolor="#1a1a24", edgecolor="#333344",
           labelcolor="#e8e8f0")

# ── Panel 4: Control 1 lexical confound ──
colors = ["#ef4444" if v > 0.6 else "#10b981" for v in ctrl1_pct]
ax4.bar(layers_12, ctrl1_pct, color=colors, alpha=0.85)
ax4.axhline(0.5, color="#ffffff", linestyle="--", linewidth=1,
            label="Chance (50%)")
ax4.axhline(0.6, color="#ef4444", linestyle=":", linewidth=1,
            label="Concern threshold (60%)")
ax4.set_title("Control 1 — Lexical confound check\n(% decoys wrongly called pressured — lower is better)",
              color="#e8e8f0", fontsize=10, pad=8)
ax4.set_xlabel("Layer", color="#8888a0", fontsize=9)
ax4.set_ylabel("% decoys misclassified", color="#8888a0", fontsize=9)
ax4.set_ylim(0, 1.1)
ax4.set_xticks(layers_12)
ax4.legend(fontsize=8, facecolor="#1a1a24", edgecolor="#333344",
           labelcolor="#e8e8f0")
ax4.text(0.5, 0.92, "Red = lexically sensitive\nGreen = robust",
         transform=ax4.transAxes, color="#8888a0", fontsize=8, ha="center")

fig.suptitle(
    "MirrorProbe — Complete Results Summary\n"
    "Social-pressure framing leaves a detectable internal signature across 3 model architectures",
    color="#ffffff", fontsize=13, fontweight="500", y=0.98
)

plt.savefig("data/day10_master_summary.png", dpi=150,
            facecolor="#0a0a0f", bbox_inches="tight")
print("Saved to data/day10_master_summary.png")