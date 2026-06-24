"""
MirrorProbe - Day 7
====================
Multi-seed rerun of the probe training pipeline.

Runs probe training 10 times with different random seeds on GPT-2
Small activations, reporting mean accuracy +/- standard deviation
per layer for both real probe and shuffled control.

This directly addresses the single biggest outstanding weakness in
the experiment: all previous results were single-seed, single-split.
A stable mean with small std across seeds is much stronger evidence
than one lucky split.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the 100-record activation dataset (50 pairs x 2 conditions)
with open("data/day3_activations.json", "r", encoding="utf-8-sig") as f:
    records = json.load(f)

print(f"Loaded {len(records)} records.\n")

LAYER_COUNT = 12
N_SEEDS = 10
SEEDS = list(range(42, 42 + N_SEEDS))  # seeds: 42,43,44...51

def build_dataset(layer_name, records):
    X, y = [], []
    for r in records:
        X.append(r["activations"][layer_name])
        y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

print("=" * 70)
print(f"MULTI-SEED RESULTS ({N_SEEDS} seeds) - real probe")
print("Format: mean_acc +/- std_acc | min | max")
print("=" * 70)

real_summary = {}
control_summary = {}
rng_meta = np.random.default_rng(0)

for layer in range(LAYER_COUNT):
    layer_name = f"layer_{layer}"
    X, y = build_dataset(layer_name, records)

    real_accs = []
    control_accs = []

    for seed in SEEDS:
        # Real probe
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        real_accs.append(clf.score(X_test, y_test))

        # Shuffled label control - different shuffle per seed
        rng = np.random.default_rng(seed)
        y_shuffled = rng.permutation(y)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y_shuffled, test_size=0.3, random_state=seed, stratify=y_shuffled
        )
        clf2 = LogisticRegression(max_iter=1000)
        clf2.fit(X_train2, y_train2)
        control_accs.append(clf2.score(X_test2, y_test2))

    real_mean = np.mean(real_accs)
    real_std = np.std(real_accs)
    ctrl_mean = np.mean(control_accs)
    ctrl_std = np.std(control_accs)

    real_summary[layer_name] = {
        "mean": float(real_mean),
        "std": float(real_std),
        "min": float(np.min(real_accs)),
        "max": float(np.max(real_accs)),
        "all_accs": real_accs
    }
    control_summary[layer_name] = {
        "mean": float(ctrl_mean),
        "std": float(ctrl_std),
        "min": float(np.min(control_accs)),
        "max": float(np.max(control_accs)),
        "all_accs": control_accs
    }

    print(f"Layer {layer:2d} | "
          f"REAL: {real_mean:.2f} +/- {real_std:.2f} "
          f"[{np.min(real_accs):.2f}-{np.max(real_accs):.2f}] | "
          f"CTRL: {ctrl_mean:.2f} +/- {ctrl_std:.2f} "
          f"[{np.min(control_accs):.2f}-{np.max(control_accs):.2f}]")

# Save full results
output = {
    "n_seeds": N_SEEDS,
    "seeds_used": SEEDS,
    "real_probe": real_summary,
    "control_shuffled": control_summary
}
with open("data/day7_multiseed_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nSaved to data/day7_multiseed_results.json")
print("\nINTERPRETATION:")
print("Small std (< 0.05) = stable, reliable result across splits")
print("Large std (> 0.10) = noisy, seed-dependent, less trustworthy")
print("Real mean >> Control mean = genuine signal, not noise")