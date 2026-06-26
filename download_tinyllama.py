from huggingface_hub import snapshot_download
import os
token = os.environ.get("HF_TOKEN")
print("Downloading TinyLlama with resumable download...")
path = snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    token=token,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"]
)
print(f"Downloaded to: {path}")