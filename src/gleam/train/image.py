"""Modal app, container image, volumes, and container imports."""

from __future__ import annotations

import modal

# ─── Modal app ─────────────────────────────────────────────────────────

app = modal.App("gleam-sft-finetune")

# ─── Container image ──────────────────────────────────────────────────

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("cmake", "git", "curl", "build-essential", "libcurl4-openssl-dev", "libssl-dev")
    .uv_pip_install(
        "accelerate>=1.9.0",
        "datasets>=3.6.0",
        "hf-transfer>=0.1.9",
        "huggingface_hub>=0.34.2",
        "peft>=0.16.0",
        "trackio>=0.1.0",
        "transformers>=4.54.0",
        "trl>=0.19.1",
        "unsloth>=2026.6.1",
        "unsloth_zoo>=2026.6.1",
        # llama.cpp convert_hf_to_gguf.py dependencies
        "gguf>=0.17.0",
        "sentencepiece>=0.2.0",
        "protobuf>=5.29.0",
    )
    # Clone llama.cpp for convert_hf_to_gguf.py + conversion Python module
    # and download pre-built llama-quantize binary (avoids slow in-image compilation).
    # Unsloth's bundled converter produces broken GGUFs for Gemma 4 —
    # we use llama.cpp's own converter directly instead.
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
        # Download pre-built llama-quantize for Ubuntu x64 (includes shared libs)
        'curl -sL -o /tmp/llama-bin.tar.gz https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-b9581-bin-ubuntu-x64.tar.gz',
        'mkdir -p /opt/llama-cpp-bin && tar xzf /tmp/llama-bin.tar.gz -C /opt/llama-cpp-bin --strip-components=1',
        'rm -f /tmp/llama-bin.tar.gz',
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "UNSLOTH_LLAMA_CPP_PATH": "/root/.unsloth/llama.cpp",
            "LLAMA_CPP_PATH": "/opt/llama.cpp",
            "LLAMA_CPP_BIN": "/opt/llama-cpp-bin",
        }
    )
)

# Pre-imported inside every @app.function(image=train_image) container.
# NOTE: unsloth and torch are NOT pre-imported here because unsloth raises
# NotImplementedError on CPU containers. They're imported locally inside
# each @app.function body that needs them (which runs on GPU).
with train_image.imports():
    import datasets  # noqa: F401
    import trackio  # noqa: F401

# ─── Volumes ───────────────────────────────────────────────────────────

model_cache = modal.Volume.from_name("gleam-model-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("gleam-training-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("gleam-checkpoints", create_if_missing=True)

# ─── Secrets ───────────────────────────────────────────────────────────

# HF secret — create with: modal secret create huggingface-secret HF_TOKEN=hf_xxx
# Falls back to empty placeholder so --validate works before secret is created.
# NOTE: Training will fail without a real HF token.
try:
    hf_secret: list = [modal.Secret.from_name("huggingface-secret")]
except Exception:
    hf_secret = [modal.Secret.from_dict({"HF_TOKEN": ""})]
