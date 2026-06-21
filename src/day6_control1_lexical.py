"""
MirrorProbe - Day 6 - Control 1 (Lexical Confound Check)
============================================================
Tests whether our probe is detecting genuine social pressure, or
just surface mention of an authority figure / person.

Method:
1. Extract activations for the 50 decoy prompts (person mentioned,
   NO pressure/endorsement attached).
2. Train the probe on the REAL data (neutral vs pressured), same as
   Day 4/5.
3. Run the trained probe on the decoys. If decoys get classified as
   "pressured" at a high rate, the probe is fooled by surface mention
   of a person - that's a real problem. If decoys get classified as
   "neutral" (correctly), the probe is tracking something deeper.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2")
print("Model loaded.\n")

LAYER_COUNT = 12

# -----------------------------------------------------------------
# STEP 1: Extract decoy activations
# -----------------------------------------------------------------
with open("data/decoy_prompts.json", "r", encoding="utf-8-sig") as f:
    decoy_data = json.load(f)["decoys"]

print(f"Extracting activations for {len(decoy_data)} decoy prompts...")
decoy_activations = []  # list of dicts: {id, activations per layer}

for item in decoy_data:
    logits, cache = model.run_with_cache(item["decoy"])
    layer_acts = {}
    for layer in range(LAYER_COUNT):
        layer_acts[f"layer_{layer}"] = cache["resid_post", layer][0, -1, :].detach().tolist()
    decoy_activations.append({"id": item["id"], "activations": layer_acts})

print("Decoy extraction done.\n")

# -----------------------------------------------------------------
# STEP 2: Load real data, train probe per layer (same as Day 4/5)
# -----------------------------------------------------------------
with open("data/day3_activations.json", "r", encoding="utf-8-sig") as f:
    real_records = json.load(f)

def build_real_dataset(layer_name):
    X, y = [], []
    for r in real_records:
        X.append(r["activations"][layer_name])
        y.append(1 if r["condition"] == "pressured" else 0)  # 1=pressured, 0=neutral
    return np.array(X), np.array(y)

def build_decoy_X(layer_name):
    return np.array([d["activations"][layer_name] for d in decoy_activations])

# -----------------------------------------------------------------
# STEP 3: For each layer, train on ALL real data, predict on decoys
# -----------------------------------------------------------------
print("=" * 70)
print("CONTROL 1 RESULTS - % of decoys WRONGLY classified as 'pressured'")
print("(Low % = good, probe is NOT fooled by surface mention of a person)")
print("(High % near real pressured rate = bad, probe is fooled)")
print("=" * 70)

control1_results = {}
for layer in range(LAYER_COUNT):
    layer_name = f"layer_{layer}"
    X_real, y_real = build_real_dataset(layer_name)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_real, y_real)  # train on ALL real data this time

    X_decoy = build_decoy_X(layer_name)
    decoy_preds = clf.predict(X_decoy)  # 1=classified as pressured, 0=neutral

    pct_wrongly_pressured = float(np.mean(decoy_preds == 1)) * 100
    control1_results[layer_name] = pct_wrongly_pressured

    print(f"Layer {layer:2d} | {pct_wrongly_pressured:5.1f}% of decoys called 'pressured'")

with open("data/day6_control1_results.json", "w", encoding="utf-8") as f:
    json.dump(control1_results, f, indent=2)

print("\nSaved results to data/day6_control1_results.json")
print("\nINTERPRETATION: if these percentages are close to 50% (chance),")
print("the probe correctly ignores mere mention of a person without")
print("pressure. If they're close to 90%+ (like real pressured prompts),")
print("the probe is likely keying on surface words, not real pressure.")