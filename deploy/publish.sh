#!/usr/bin/env bash
# Deploy gemma-4-e4b-gleam-sft to HuggingFace Hub and Ollama registry.
#
# Usage:
#   ./deploy/publish.sh --push-hf          # Push LoRA + GGUF to HuggingFace
#   ./deploy/publish.sh --push-ollama      # Push to Ollama registry (from forge)
#   ./deploy/publish.sh --push-all         # Do both
#
# Prerequisites:
#   - Modal CLI authenticated
#   - Ollama key registered at ollama.com (for --push-ollama)
#   - GGUF already exported on Modal volume or local disk

set -euo pipefail

EXPERIMENT="gleam-sft-r32-ep2-20260608-173504"
HF_REPO="kasuboski/gemma-4-e4b-gleam-sft"
GGUF_FILE="gemma-4-e4b-gleam-sft-Q4_K_M.gguf"
LOCAL_GGUF="/tmp/gleam-sft-llamacpp-q4km.gguf"
FORGE_HOST="josh@forge"

PUSH_HF=false
PUSH_OLLAMA=false

for arg in "$@"; do
    case "$arg" in
        --push-hf)     PUSH_HF=true ;;
        --push-ollama) PUSH_OLLAMA=true ;;
        --push-all)    PUSH_HF=true; PUSH_OLLAMA=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if ! $PUSH_HF && ! $PUSH_OLLAMA; then
    echo "Usage: $0 --push-hf | --push-ollama | --push-all"
    exit 1
fi

# ── Step 1: Push to HuggingFace Hub ──────────────────────────────
if $PUSH_HF; then
    echo "=========================================="
    echo "📦 Pushing to HuggingFace Hub"
    echo "=========================================="
    echo ""
    echo "This pushes:"
    echo "  LoRA adapter → ${HF_REPO}"
    echo "  GGUF (Q4_K_M) → ${HF_REPO}-GGUF"
    echo ""
    echo "Experiment: ${EXPERIMENT}"
    echo ""

    # Use Modal to push from the volume (has both LoRA and GGUF)
    uv run modal run -m src.gleam.train \
        --experiment "${EXPERIMENT}" \
        --gguf \
        --push-hub \
        --hub-id "${HF_REPO}"

    echo ""
    echo "✅ HuggingFace push complete!"
    echo "  LoRA:  https://huggingface.co/${HF_REPO}"
    echo "  GGUF:  https://huggingface.co/${HF_REPO}-GGUF"
fi

# ── Step 2: Push to Ollama registry ─────────────────────────────
if $PUSH_OLLAMA; then
    echo ""
    echo "=========================================="
    echo "🦙 Pushing to Ollama registry"
    echo "=========================================="
    echo ""
    echo "This creates: kasuboski/gemma-4-e4b-gleam-sft"
    echo ""

    # Build and push from forge (has the GGUF + Ollama)
    # The Modelfile at deploy/Modelfile.gleam-sft has the system prompt + template
    scp deploy/Modelfile.gleam-sft "${FORGE_HOST}:/tmp/Modelfile.gleam-sft"

    ssh "${FORGE_HOST}" <<'REMOTE'
        set -euo pipefail

        GGUF="/tmp/gleam-sft-llamacpp-q4km.gguf"
        MODELFILE="/tmp/Modelfile.gleam-sft"

        if [ ! -f "$GGUF" ]; then
            echo "❌ GGUF not found at ${GGUF}"
            echo "Download from HF first:"
            echo "  wget -O ${GGUF} https://huggingface.co/kasuboski/gemma-4-e4b-gleam-sft-GGUF/resolve/main/gemma-4-e4b-gleam-sft-Q4_K_M.gguf"
            exit 1
        fi

        # Copy both build inputs explicitly; /tmp is not necessarily shared with the container.
        docker cp "$GGUF" ollama-nvidia:/tmp/gemma-4-e4b-gleam-sft-Q4_K_M.gguf
        docker cp "$MODELFILE" ollama-nvidia:/tmp/Modelfile.gleam-sft

        # Build inside Docker container
        docker exec -i ollama-nvidia bash -c '
            cd /tmp
            ollama rm kasuboski/gemma-4-e4b-gleam-sft 2>/dev/null || true
            ollama create kasuboski/gemma-4-e4b-gleam-sft -f /tmp/Modelfile.gleam-sft
        '

        echo ""
        echo "Model built. To push to ollama.com:"
        echo "  docker exec -it ollama-nvidia ollama push kasuboski/gemma-4-e4b-gleam-sft"
        echo ""
        echo "⚠️  This requires your Ollama public key to be registered at ollama.com."
        echo "Push manually when ready with the command above."
REMOTE
fi

echo ""
echo "=========================================="
echo "✅ Publish script complete"
echo "=========================================="
