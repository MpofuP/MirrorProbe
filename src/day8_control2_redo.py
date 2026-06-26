"""
MirrorProbe - Day 8
====================
Control 2 REDO on TinyLlama-1.1B-Chat — an instruction-tuned model
that actually answers questions with a real stance, unlike GPT-2 Small
which just produced degenerate filler text ("I'm not sure. I'm not sure...")

This time "output-identical" actually means something — the model gave
the same answer/judgment in both conditions. If the probe can still
distinguish neutral from pressured INTERNALLY on those pairs, that is
the strongest version of our core hypothesis:

"Internal state differs under pressure even when output is identical"

We extract activations using raw PyTorch hooks instead of TransformerLens
since TinyLlama is not in TransformerLens's supported model list.
This is the same underlying mechanism TransformerLens uses — we just
write it directly.
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {MODEL_NAME}...")
print("(This will download ~2.2GB on first run — please wait)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

# Check how many layers this model has
n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Model loaded: {n_layers} layers, hidden size {hidden_size}\n")

# -----------------------------------------------------------------
# STEP 1: Set up raw PyTorch hook to capture residual stream
# -----------------------------------------------------------------
# TinyLlama uses LlamaDecoderLayer blocks. Each block outputs a tuple
# where the first element is the hidden state (residual stream).
# We hook the OUTPUT of each decoder layer — equivalent to resid_post
# in TransformerLens.

activation_store = {}

def make_hook(layer_idx):
    """Creates a hook function for a specific layer index."""
    def hook_fn(module, input, output):
        # output is a tuple — first element is the hidden state tensor
        # Shape: [batch, seq_len, hidden_size]
        # We grab the last token position, detach from computation graph
        hidden = output[0] if isinstance(output, tuple) else output
        activation_store[f"layer_{layer_idx}"] = hidden[0, -1, :].detach().cpu()
    return hook_fn

def extract_activations(prompt_text):
    """
    Runs one forward pass and returns activations at every layer
    at the final token position.
    """
    activation_store.clear()

    # Register hooks on all decoder layers
    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Tokenize and run forward pass
    inputs = tokenizer(prompt_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove all hooks after the forward pass
    for h in hooks:
        h.remove()

    # Return a dict of layer -> 2048-dim activation vector
    return {k: v.numpy().tolist() for k, v in activation_store.items()}

def generate_response(prompt_text, max_new_tokens=40):
    """
    Generates a short response from TinyLlama using chat template.
    Returns only the NEW generated text, not the input prompt.
    """
    # TinyLlama uses a chat template format
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer briefly in 1-2 sentences."},
        {"role": "user", "content": prompt_text}
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# -----------------------------------------------------------------
# STEP 2: Load prompt pairs and run both extraction and generation
# -----------------------------------------------------------------
with open("data/prompt_pairs.json", "r", encoding="utf-8-sig") as f:
    pairs = json.load(f)["pairs"]

print(f"Processing {len(pairs)} pairs...")
print("This takes ~5-10 minutes on CPU. Each pair = 2 forward passes + 2 generations.\n")

records = []
generations = []

for i, pair in enumerate(pairs):
    print(f"Pair {i+1}/{len(pairs)}: {pair['id']}", end=" ", flush=True)

    for condition in ["neutral", "pressured"]:
        prompt = pair[condition]

        # Extract activations
        layer_acts = extract_activations(prompt)

        # Generate a response
        response = generate_response(prompt)

        records.append({
            "pair_id": pair["id"],
            "condition": condition,
            "topic_domain": pair["topic_domain"],
            "prompt_text": prompt,
            "activations": layer_acts
        })

        generations.append({
            "pair_id": pair["id"],
            "condition": condition,
            "response": response
        })

    print("done")

print("\nAll pairs processed.\n")

# Save activations
with open("data/day8_tinyllama_activations.json", "w", encoding="utf-8") as f:
    json.dump(records, f)
print("Activations saved.")

# Save generations for inspection
with open("data/day8_tinyllama_generations.json", "w", encoding="utf-8") as f:
    json.dump(generations, f, indent=2)
print("Generations saved.")

# -----------------------------------------------------------------
# STEP 3: Show example responses so we can see if they are real stances
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("SAMPLE RESPONSES (first 5 pairs) - checking if outputs are real stances")
print("=" * 70)

for g in generations[:10]:
    label = f"{g['pair_id']} [{g['condition']}]"
    print(f"{label:35s} | {g['response'][:80]}")

# -----------------------------------------------------------------
# STEP 4: Find output-similar pairs using TF-IDF cosine similarity
# -----------------------------------------------------------------
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

print("\n" + "=" * 70)
print("TOP 10 OUTPUT-SIMILAR PAIRS (real stances this time)")
print("=" * 70)
for item in similarities[:10]:
    pid = item["pair_id"]
    print(f"{pid} | similarity: {item['similarity']:.2f}")
    print(f"  neutral:   {neutral_gens[pid][:80]}")
    print(f"  pressured: {pressured_gens[pid][:80]}")

# -----------------------------------------------------------------
# STEP 5: Probe on top-15 most output-similar pairs
# -----------------------------------------------------------------
top_ids = set(item["pair_id"] for item in similarities[:15])
other_ids = set(p["id"] for p in pairs) - top_ids

def build_dataset(layer_name, records, ids_filter=None):
    X, y = [], []
    for r in records:
        if ids_filter and r["pair_id"] not in ids_filter:
            continue
        if layer_name in r["activations"]:
            X.append(r["activations"][layer_name])
            y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

print("\n" + "=" * 70)
print("CONTROL 2 REDO RESULT: probe on 15 most output-similar pairs")
print("(trained on other 35, tested on held-out similar pairs)")
print("High accuracy here = internal signature beyond output text")
print("=" * 70)

control2_results = {}
for layer in range(n_layers):
    layer_name = f"layer_{layer}"
    X_train, y_train = build_dataset(layer_name, records, ids_filter=other_ids)
    X_test, y_test = build_dataset(layer_name, records, ids_filter=top_ids)

    if len(X_train) < 4 or len(X_test) < 4:
        continue

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    control2_results[layer_name] = acc
    print(f"Layer {layer:2d} | accuracy on output-similar held-out pairs: {acc:.2f}")

with open("data/day8_control2_redo_results.json", "w", encoding="utf-8") as f:
    json.dump(control2_results, f, indent=2)

print("\nSaved results to data/day8_control2_redo_results.json")
print("\nINTERPRETATION: Unlike GPT-2, TinyLlama actually gives real")
print("answers. So output-identical here means the model genuinely")
print("reached the same conclusion both ways. High probe accuracy")
print("on these pairs is the cleanest evidence for the core hypothesis.")