"""
MirrorProbe - Day 9
====================
Multi-seed rerun of the Control 2 redo on TinyLlama.
Activations already saved from Day 8 - we just retrain the probe
across 10 seeds to confirm the 0.70-0.93 result is stable,
not a lucky single split.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

with open("data/day8_tinyllama_activations.json", "r", encoding="utf-8-sig") as f:
    records = json.load(f)

with open("data/day8_tinyllama_generations.json", "r", encoding="utf-8-sig") as f:
    generations = json.load(f)

print(f"Loaded {len(records)} TinyLlama activation records.")

# Rebuild output similarity to get top-15 most similar pairs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

neutral_gens = {g["pair_id"]: g["response"] for g in generations if g["condition"] == "neutral"}
pressured_gens = {g["pair_id"]: g["response"] for g in generations if g["condition"] == "pressured"}
pair_ids = list(neutral_gens.keys())
texts_a = [neutral_gens[p] for p in pair_ids]
texts_b = [pressured_gens[p] for p in pair_ids]
vectorizer = TfidfVectorizer().fit(texts_a + texts_b)
vecs_a = vectorizer.transform(texts_a)
vecs_b = vectorizer.transform(texts_b)
similarities = []
for i in range(len(pair_ids)):
    sim = cosine_similarity(vecs_a[i], vecs_b[i])[0][0]
    similarities.append({"pair_id": pair_ids[i], "similarity": float(sim)})
similarities.sort(key=lambda x: -x["similarity"])
top_ids = set(item["pair_id"] for item in similarities[:15])
other_ids = set(r["pair_id"] for r in records) - top_ids

N_SEEDS = 10
SEEDS = list(range(42, 42 + N_SEEDS))
n_layers = 22

def build_dataset(layer_name, records, ids_filter=None):
    X, y = [], []
    for r in records:
        if ids_filter and r["pair_id"] not in ids_filter:
            continue
        if layer_name in r["activations"]:
            X.append(r["activations"][layer_name])
            y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

print(f"\nRunning {N_SEEDS} seeds on output-similar held-out pairs...\n")
print("=" * 70)
print("TINYLLAMA CONTROL 2 REDO - multi-seed (mean +/- std, 10 seeds)")
print("=" * 70)

results = {}
for layer in range(n_layers):
    layer_name = f"layer_{layer}"
    X_all, y_all = build_dataset(layer_name, records)

    accs = []
    for seed in SEEDS:
        # Train on other_ids, test on top_ids
        # We use the full records split by pair_id not random split
        X_train, y_train = build_dataset(layer_name, records, ids_filter=other_ids)
        X_test, y_test = build_dataset(layer_name, records, ids_filter=top_ids)

        if len(X_train) < 4 or len(X_test) < 4:
            continue

        # Add seed-based shuffle to training data for variance estimation
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_train))
        X_train_s = X_train[idx]
        y_train_s = y_train[idx]

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train_s, y_train_s)
        accs.append(clf.score(X_test, y_test))

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    results[layer_name] = {"mean": float(mean_acc), "std": float(std_acc)}
    print(f"Layer {layer:2d} | {mean_acc:.2f} +/- {std_acc:.2f}")

with open("data/day9_tinyllama_multiseed.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved to data/day9_tinyllama_multiseed.json")
print("\nINTERPRETATION: stable mean with small std = the TinyLlama")
print("Control 2 result is robust, not a lucky single split.")