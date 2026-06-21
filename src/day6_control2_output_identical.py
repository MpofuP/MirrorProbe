"""
MirrorProbe - Day 6 (continued) - Control 2 (Output-Identical Pairs)
=========================================================================
Generates a short continuation for each neutral/pressured prompt pair,
measures how similar the two continuations are, and identifies pairs
where the model said something very similar in BOTH conditions.

On those output-similar pairs specifically, we check whether the probe
can STILL tell neutral from pressured using internals alone. If yes,
that's strong evidence of an internal signature beyond what's visible
in the output text - the core hypothesis of this project.

CAVEAT: GPT-2 Small is a base model, not instruction-tuned, so it
continues text statistically rather than giving a clear yes/no verdict.
This limits how cleanly "identical output" maps to "identical judgment."
Noted as a real limitation, not hidden.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformer_lens import HookedTransformer

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2")
print("Model loaded.\n")

with open("data/prompt_pairs.json", "r", encoding="utf-8-sig") as f:
    pairs = json.load(f)["pairs"]

GEN_TOKENS = 15  # how many tokens to generate as the model's "answer"

print(f"Generating {GEN_TOKENS}-token continuations for all 50 pairs...")
generations = []  # {id, neutral_text, pressured_text}

for pair in pairs:
    neutral_out = model.generate(
        pair["neutral"], max_new_tokens=GEN_TOKENS, temperature=0, verbose=False
    )
    pressured_out = model.generate(
        pair["pressured"], max_new_tokens=GEN_TOKENS, temperature=0, verbose=False
    )
    # Strip the original prompt off to keep just the NEW generated text
    neutral_new = neutral_out[len(pair["neutral"]):]
    pressured_new = pressured_out[len(pair["pressured"]):]

    generations.append({
        "id": pair["id"],
        "neutral_continuation": neutral_new,
        "pressured_continuation": pressured_new
    })

print("Generation done.\n")

# -----------------------------------------------------------------
# Measure similarity between neutral and pressured continuations
# -----------------------------------------------------------------
texts_a = [g["neutral_continuation"] for g in generations]
texts_b = [g["pressured_continuation"] for g in generations]

vectorizer = TfidfVectorizer().fit(texts_a + texts_b)
vecs_a = vectorizer.transform(texts_a)
vecs_b = vectorizer.transform(texts_b)

similarities = []
for i in range(len(generations)):
    sim = cosine_similarity(vecs_a[i], vecs_b[i])[0][0]
    similarities.append(sim)
    generations[i]["output_similarity"] = float(sim)

# Sort to see the most output-similar pairs
sorted_gens = sorted(generations, key=lambda g: -g["output_similarity"])

print("=" * 70)
print("TOP 10 MOST OUTPUT-SIMILAR PAIRS (model said almost the same thing)")
print("=" * 70)
for g in sorted_gens[:10]:
    print(f"{g['id']} | similarity: {g['output_similarity']:.2f}")
    print(f"  neutral:   ...{g['neutral_continuation'].strip()}")
    print(f"  pressured: ...{g['pressured_continuation'].strip()}")

with open("data/day6_control2_generations.json", "w", encoding="utf-8") as f:
    json.dump(generations, f, indent=2)

# -----------------------------------------------------------------
# Test probe accuracy specifically on the TOP 15 most output-similar
# pairs - this is the real test of the core hypothesis
# -----------------------------------------------------------------
top_similar_ids = set(g["id"] for g in sorted_gens[:15])

with open("data/day3_activations.json", "r", encoding="utf-8-sig") as f:
    records = json.load(f)

def build_dataset(layer_name, records, ids_filter=None):
    X, y = [], []
    for r in records:
        if ids_filter is not None and r["pair_id"] not in ids_filter:
            continue
        X.append(r["activations"][layer_name])
        y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

print("\n" + "=" * 70)
print("CONTROL 2 RESULT: probe accuracy on the 15 MOST output-similar pairs")
print("(trained on the OTHER 35 pairs, tested only on these held-out 15)")
print("=" * 70)

control2_results = {}
for layer in range(12):
    layer_name = f"layer_{layer}"
    other_ids = set(p["id"] for p in pairs) - top_similar_ids

    X_train, y_train = build_dataset(layer_name, records, ids_filter=other_ids)
    X_test, y_test = build_dataset(layer_name, records, ids_filter=top_similar_ids)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    control2_results[layer_name] = acc
    print(f"Layer {layer:2d} | accuracy on output-similar held-out pairs: {acc:.2f}")

with open("data/day6_control2_results.json", "w", encoding="utf-8") as f:
    json.dump(control2_results, f, indent=2)

print("\nSaved results. INTERPRETATION: high accuracy here, even when the")
print("model's actual generated text was nearly identical between")
print("conditions, would be the strongest evidence for this project's")
print("core hypothesis - that pressure leaves an internal trace beyond")
print("what's visible in output text.")