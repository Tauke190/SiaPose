from huggingface_hub import snapshot_download
import os
target_dir = "weights/ViCLIP-B-16-hf"
os.makedirs(target_dir, exist_ok=True)

snapshot_download(
    repo_id="OpenGVLab/ViCLIP-B-16-hf",
    repo_type="model",
    local_dir=target_dir
)
