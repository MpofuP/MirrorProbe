"""
MirrorProbe - Day 3
====================
Runs all 10 contrastive prompt pairs through GPT-2 Small, extracts
the resid_post activation at the FINAL token of each prompt (where
the model has "seen" the whole question and is about to commit to
its answer), and saves everything to a structured file for later
probing.

Labels (topic_domain, pressure_source) are carried through so we can
run confound checks later (Control 1 and Control 3 from the project plan).
"""

import json
import torch
from transformer_lens import HookedTransformer

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2")
print("Model loaded.\n")

# Load our labeled prompt pairs
with open("data/prompt_pairs.json", "r") as f:
    data = json.load(f)

pairs = data["pairs"]

# We'll extract activations from every layer (0-11) so we can later
# check WHICH layer best separates neutral vs pressured - we don't
# assume in advance which layer matters.
LAYER_COUNT = 12

results = []

for pair in pairs:
    pair_id = pair["id"]
    print(f"Processing {pair_id}...")

    for condition in ["neutral", "pressured"]:
        prompt = pair[condition]

        # Run forward pass + cache all activations
        logits, cache = model.run_with_cache(prompt)

        # Extract resid_post at the LAST token position, for every layer.
        # Shape per layer before slicing: [1, num_tokens, 768]
        # After slicing to last token: [768]
        layer_activations = {}
        for layer in range(LAYER_COUNT):
            act = cache["resid_post", layer][0, -1, :].detach().tolist()
            layer_activations[f"layer_{layer}"] = act

        results.append({
            "pair_id": pair_id,
            "condition": condition,  # "neutral" or "pressured"
            "topic_domain": pair["topic_domain"],
            "pressure_source": pair["pressure_source"],
            "prompt_text": prompt,
            "num_tokens": logits.shape[1],
            "activations": layer_activations
        })

# Save everything - this becomes the dataset our probe trains on later
with open("data/day3_activations.json", "w") as f:
    json.dump(results, f)

print(f"\nDone. Saved {len(results)} labeled activation records "
      f"(10 pairs x 2 conditions = 20 records) to data/day3_activations.json")
