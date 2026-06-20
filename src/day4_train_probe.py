"""
MirrorProbe - Day 4
====================
Trains a logistic regression probe per layer to distinguish neutral
vs. pressured prompts from internal activations ALONE (no text access).

CRITICAL: Immediately after the real probe, we run Control 5 - shuffle
the labels randomly and retrain. If the shuffled version also scores
well, our pipeline is broken or our sample is too small to trust.
We report both results honestly, no matter what they show.

Honest caveat going in: 20 data points (10 pairs) is a SMALL dataset.
Today's numbers are a rough first signal, not a strong claim.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Day 3's labeled activation data
with open("data/day3_activations.json", "r", encoding="utf-8-sig") as f:
    records = json.load(f)

print(f"Loaded {len(records)} activation records.\n")

LAYER_COUNT = 12

def build_dataset(layer_name, records):
    """
    Builds X (activation vectors) and y (0=neutral, 1=pressured)
    for one specific layer.
    """
    X = []
    y = []
    for r in records:
        X.append(r["activations"][layer_name])
        y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

print("=" * 70)
print("REAL PROBE - per layer accuracy (train/test split)")
print("=" * 70)

real_results = {}
for layer in range(LAYER_COUNT):
    layer_name = f"layer_{layer}"
    X, y = build_dataset(layer_name, records)

    # With only 20 examples, we use a 50/50 split to have enough
    # test examples to mean anything at all - still very small.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    real_results[layer_name] = {"train_acc": train_acc, "test_acc": test_acc}
    print(f"Layer {layer:2d} | train acc: {train_acc:.2f} | test acc: {test_acc:.2f}")

print("\n" + "=" * 70)
print("CONTROL 5 - SHUFFLED LABELS (should score ~0.50 if pipeline is sound)")
print("=" * 70)

control_results = {}
rng = np.random.default_rng(42)

for layer in range(LAYER_COUNT):
    layer_name = f"layer_{layer}"
    X, y = build_dataset(layer_name, records)

    # Shuffle the labels randomly - breaks any real relationship
    y_shuffled = rng.permutation(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shuffled, test_size=0.5, random_state=42, stratify=y_shuffled
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    control_results[layer_name] = {"test_acc": test_acc}
    print(f"Layer {layer:2d} | shuffled-label test acc: {test_acc:.2f}")

print("\n" + "=" * 70)
print("HONEST INTERPRETATION GUIDE")
print("=" * 70)
print("If REAL test acc is notably higher than CONTROL (shuffled) acc")
print("at a given layer, that's a real signal worth investigating further.")
print("If they're similar, the 'real' result is likely noise from our")
print("very small (20-example) dataset - not evidence of anything yet.")

# Save full results
output = {"real_probe": real_results, "control_shuffled": control_results}
with open("data/day4_probe_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nSaved full results to data/day4_probe_results.json")