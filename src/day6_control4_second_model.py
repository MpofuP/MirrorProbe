"""
MirrorProbe - Day 6 (continued) - Control 4 (Second Model Check)
=====================================================================
Repeats the full extraction + probe + shuffle-control pipeline on a
SECOND model with a different architecture (Pythia-160M, not just a
bigger GPT-2), to test whether the sycophancy-pressure signature
transfers across models or is specific to GPT-2 Small.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer

MODEL_NAME = "EleutherAI/pythia-160m"
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
print("Model loaded.\n")

LAYER_COUNT = model.cfg.n_layers
print(f"{MODEL_NAME} has {LAYER_COUNT} layers.\n")

with open("data/prompt_pairs.json", "r", encoding="utf-8-sig") as f:
    pairs = json.load(f)["pairs"]

print(f"Extracting activations for {len(pairs)} pairs x 2 conditions...")
records = []
for pair in pairs:
    for condition in ["neutral", "pressured"]:
        prompt = pair[condition]
        logits, cache = model.run_with_cache(prompt)
        layer_acts = {}
        for layer in range(LAYER_COUNT):
            layer_acts[f"layer_{layer}"] = cache["resid_post", layer][0, -1, :].detach().tolist()
        records.append({
            "pair_id": pair["id"],
            "condition": condition,
            "activations": layer_acts
        })

print("Extraction done.\n")

with open("data/day6_control4_pythia_activations.json", "w") as f:
    json.dump(records, f)

def build_dataset(layer_name, records):
    X, y = [], []
    for r in records:
        X.append(r["activations"][layer_name])
        y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

print("=" * 70)
print(f"CONTROL 4 RESULTS on {MODEL_NAME} - real probe vs shuffled control")
print("=" * 70)

results = {"real_probe": {}, "control_shuffled": {}}
rng = np.random.default_rng(42)

for layer in range(LAYER_COUNT):
    layer_name = f"layer_{layer}"
    X, y = build_dataset(layer_name, records)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    real_acc = clf.score(X_test, y_test)

    y_shuffled = rng.permutation(y)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_shuffled, test_size=0.3, random_state=42, stratify=y_shuffled
    )
    clf2 = LogisticRegression(max_iter=1000)
    clf2.fit(X_train2, y_train2)
    control_acc = clf2.score(X_test2, y_test2)

    results["real_probe"][layer_name] = real_acc
    results["control_shuffled"][layer_name] = control_acc
    print(f"Layer {layer:2d} | real: {real_acc:.2f} | shuffled control: {control_acc:.2f}")

with open("data/day6_control4_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results. INTERPRETATION: if {MODEL_NAME} shows a similar")
print("real-vs-control gap as GPT-2 Small did, the signature generalizes")
print("across architectures. If the gap disappears, it may be specific")
print("to GPT-2's particular training/architecture.")