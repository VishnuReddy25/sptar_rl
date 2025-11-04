from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="/home/aiml_user/vishnu/sptar_rl/models/llama-7b",
    allow_patterns=["*.safetensors"],  # only download safetensors
)
