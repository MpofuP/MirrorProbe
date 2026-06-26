"""
MirrorProbe - Day 9 corrected multi-seed
==========================================
Fixes the zero-std problem from the first attempt.
Uses bootstrap resampling of training data to get genuine
variance estimates, not just shuffling order.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("data/day8_tinyllama_activations.json", "r", encoding="utf-8-sig") as f:
    records = json.load(f)
with open("data/day8_tinyllama_generations.json", "r", encoding="utf-8-sig") as f:
    generations = json.load(f)

# Rebuild top-15 similar pairs
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

def build_dataset(layer_name, records, ids_filter=None):
    X, y = [], []
    for r in records:
        if ids_filter and r["pair_id"] not in ids_filter:
            continue
        if layer_name in r["activations"]:
            X.append(r["activations"][layer_name])
            y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

N_BOOTSTRAP = 50  # more iterations since bootstrap is fast
n_layers = 22

print("=" * 70)
print("TINYLLAMA CONTROL 2 - bootstrap resampling (50 iterations)")
print("Each run: random subsample of training data, fixed test set")
print("This gives genuine variance, not zero from order shuffling")
print("=" * 70)

results = {}
for layer in range(n_layers):
    layer_name = f"layer_{layer}"
    X_train_full, y_train_full = build_dataset(layer_name, records, ids_filter=other_ids)
    X_test, y_test = build_dataset(layer_name, records, ids_filter=top_ids)

    if len(X_train_full) < 4 or len(X_test) < 4:
        continue

    accs = []
    rng = np.random.default_rng(42)
    for _ in range(N_BOOTSTRAP):
        # Bootstrap: sample training data WITH replacement
        idx = rng.integers(0, len(X_train_full), size=len(X_train_full))
        X_boot = X_train_full[idx]
        y_boot = y_train_full[idx]

        # Skip if bootstrap sample has only one class
        if len(np.unique(y_boot)) < 2:
            continue

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_boot, y_boot)
        accs.append(clf.score(X_test, y_test))

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    results[layer_name] = {"mean": mean_acc, "std": std_acc, "n_bootstrap": len(accs)}
    print(f"Layer {layer:2d} | {mean_acc:.2f} +/- {std_acc:.2f}")

with open("data/day9_tinyllama_bootstrap.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved to data/day9_tinyllama_bootstrap.json")
print("Non-zero std = genuine variance estimate from bootstrap resampling")