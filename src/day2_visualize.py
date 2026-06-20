"""
MirrorProbe - Day 2 Visualization
===================================
Plots the 768 internal activation values for one token, just to
SEE what "768 numbers representing a word" actually looks like.
This is purely illustrative - not analysis yet, just intuition-building.
"""

import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

print("Loading GPT-2 Small (should be cached now, fast this time)...")
model = HookedTransformer.from_pretrained("gpt2")

prompt = "The weather today is"
logits, cache = model.run_with_cache(prompt)

layer_to_inspect = 6
activation = cache["resid_post", layer_to_inspect]

# Grab the activation for the LAST token only ("is"), for one clean plot
# Shape goes from [1, 5, 768] -> just the 768 numbers for token index 4
last_token_activation = activation[0, -1, :].detach().numpy()

plt.figure(figsize=(12, 4))
plt.plot(last_token_activation, linewidth=0.8)
plt.title(f"768 internal activation values for the word 'is'\n"
          f"(Layer {layer_to_inspect} of GPT-2 Small, prompt: '{prompt}')")
plt.xlabel("Dimension (0 to 767)")
plt.ylabel("Activation value")
plt.tight_layout()
plt.savefig("data/day2_activation_plot.png", dpi=150)
print("Saved plot to data/day2_activation_plot.png")
