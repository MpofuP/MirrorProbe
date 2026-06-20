"""
MirrorProbe - Day 2
====================
Goal: Load GPT-2 Small, run one forward pass, and extract ("hook")
one internal activation tensor so we can inspect its shape.

This is the foundational technique the whole project depends on:
peeking inside the model's internal numbers, not just reading its
final text output.
"""

from transformer_lens import HookedTransformer

# -----------------------------------------------------------------
# STEP 1: Load the model
# -----------------------------------------------------------------
print("Loading GPT-2 Small... (first run will download the model)")
model = HookedTransformer.from_pretrained("gpt2")
print("Model loaded successfully.")

# -----------------------------------------------------------------
# STEP 2: Define a simple input prompt
# -----------------------------------------------------------------
prompt = "The weather today is"

# -----------------------------------------------------------------
# STEP 3: Run the model WITH CACHING
# -----------------------------------------------------------------
logits, cache = model.run_with_cache(prompt)

print(f"\nPrompt used: '{prompt}'")
print(f"Logits shape (model's raw output scores): {logits.shape}")

# -----------------------------------------------------------------
# STEP 4: Pull out ONE specific internal activation
# -----------------------------------------------------------------
layer_to_inspect = 6
activation = cache["resid_post", layer_to_inspect]

print(f"\nActivation tensor extracted from layer {layer_to_inspect}")
print(f"Shape: {activation.shape}")
print("Shape meaning: [batch_size, num_tokens, hidden_dimension]")
print(f"  - batch_size = {activation.shape[0]} (we only ran 1 prompt)")
print(f"  - num_tokens = {activation.shape[1]} (tokens in our prompt)")
print(f"  - hidden_dimension = {activation.shape[2]} "
      f"(size of GPT-2 Small's internal representation per token)")

print("\nDay 2 core mechanic complete: model loaded, "
      "forward pass run, internal activation extracted.")
