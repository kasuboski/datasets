# Gleam SFT Fine-tuning

QLoRA fine-tuning of Gemma 4 E4B on a mixed dataset of 13,538 instruction-response pairs, using Unsloth + Modal + Trackio.

The validated local dataset is `data/sft/mixed_training.jsonl`:

| Source | Records |
|--------|---------:|
| gleam-instruct | 9,355 |
| ultrachat | 3,000 |
| code-290k-functional | 683 |
| OpenCodeInstruct | 500 |
| **Total** | **13,538** |

## Architecture

```
src/gleam/train/
├── __init__.py   # Re-exports; registers @app.local_entrypoint on import
├── config.py     # TrainingConfig dataclass + chat template constants
├── image.py      # Modal app, container image, volumes, secrets
├── helpers.py    # load_model, setup_lora, dataset prep (local imports for GPU-only deps)
├── remote.py     # All @app.function decorated functions (train, export, validate, etc.)
├── local.py      # upload_dataset + @app.local_entrypoint dispatcher
└── README.md     # You are here
```

## Prerequisites

1. **Modal account** — [modal.com](https://modal.com) ($30/mo free credits)
2. **Modal CLI** — `uv add modal` (already in project)
3. **HuggingFace token** — stored as Modal secret:
   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
   ```
4. **Trackio Space** — auto-created on the first run.

## Quick Start

```bash
# 1. Validate everything works (no GPU, ~1 min)
uv run modal run -m src.gleam.train --validate

# 2. Upload dataset to Modal Volume
uv run modal run -m src.gleam.train --upload-data

# 3. Start training (detached so it survives disconnect)
uv run modal run --detach -m src.gleam.train \
  --trackio-space kasuboski/trackio

# 4. Monitor live at:
#    https://huggingface.co/spaces/kasuboski/trackio
```

## CLI Reference

All commands use `uv run modal run -m src.gleam.train` as the base.

### Validate (no GPU, no cost)

```bash
# Checks imports, config, chat template markers, dataset existence
uv run modal run -m src.gleam.train --validate
```

Three-part check:
1. **Local** — config construction, path resolution, dataset file check
2. **CPU (Modal)** — datasets, trl, peft, trackio, transformers imports
3. **GPU (Modal L4)** — unsloth, FastModel, chat templates, API signatures

### Upload Dataset

```bash
uv run modal run -m src.gleam.train --upload-data
# Optional: --dataset-file path/to/other.jsonl
```

Reads local file, writes to `gleam-training-data` Modal Volume at `/data/`.

### Train

```bash
# Fresh run (auto-generates experiment name like gleam-sft-r32-ep2-20260608-155553)
uv run modal run --detach -m src.gleam.train \
  --trackio-space kasuboski/trackio

# With custom hyperparameters
uv run modal run --detach -m src.gleam.train \
  --trackio-space kasuboski/trackio \
  --lora-r 64 --epochs 3 --learning-rate 1e-4
```

**Always use `--detach`** for training. Without it, the run stops when your terminal disconnects.

Training computes checkpoint and eval cadence dynamically at roughly four checkpoints per epoch. Early stopping triggers after 3 consecutive evals with no improvement in eval loss. At the end, the best checkpoint is loaded automatically.

### Resume

If a training run stops (Modal timeout, early stop, disconnect), resume from the last checkpoint:

```bash
# Pass the experiment name from the previous run
uv run modal run --detach -m src.gleam.train \
  --experiment gleam-sft-r32-ep2-20260608-155553 \
  --trackio-space kasuboski/trackio
```

Auto-detects the latest `checkpoint-*` subdirectory and restores:
- LoRA adapter weights
- Optimizer state (Adam momentum)
- LR scheduler position (cosine curve)
- Step counter

### Test Inference

```bash
# Against the base model
uv run modal run -m src.gleam.train --test

# Against a trained checkpoint
uv run modal run -m src.gleam.train --test --test-experiment gleam-sft-r32-ep2-20260608-155553

# Custom prompt
uv run modal run -m src.gleam.train --test --test-prompt "Write a binary tree in Gleam"
```

### Export

```bash
# Export LoRA adapter to GGUF (for llama.cpp / Ollama)
uv run modal run -m src.gleam.train \
  --experiment gleam-sft-r32-ep2-20260608-155553 \
  --gguf

# Push to HuggingFace Hub
uv run modal run -m src.gleam.train \
  --experiment gleam-sft-r32-ep2-20260608-155553 \
  --push-hub --hub-id kasuboski/gemma4-e4b-gleam-lora
```

## Monitoring

### Trackio Dashboard

Live dashboard: https://huggingface.co/spaces/kasuboski/trackio

### Trackio CLI

```bash
# List runs
uv run trackio --space kasuboski/trackio list runs --project huggingface

# Get loss history
uv run trackio --space kasuboski/trackio get metric \
  --project huggingface \
  --run gleam-sft-r32-ep2-20260608-155553 \
  --metric train/loss

# Get eval loss
uv run trackio --space kasuboski/trackio get metric \
  --project huggingface \
  --run gleam-sft-r32-ep2-20260608-155553 \
  --metric eval/loss
```

### Modal CLI

```bash
# List running apps
uv run modal app list

# Stream logs from a running app
uv run modal app logs <app-id>
```

## All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `unsloth/gemma-4-E4B-it` | Base model (Unsloth-optimized) |
| `--max-seq-length` | `4096` | Max token length per sample |
| `--lora-r` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA scaling factor |
| `--use-rslora` / `--no-use-rslora` | `True` | Rank-stabilized LoRA |
| `--epochs` | `2` | Max training epochs |
| `--batch-size` | `2` | Per-device batch size |
| `--gradient-accumulation-steps` | `8` | Effective batch = 2 × 8 = 16 |
| `--learning-rate` | `2e-4` | Peak learning rate |
| `--trackio-space` | `None` | HF Space ID for Trackio logging |
| `--experiment` | auto-generated | Experiment name (pass to resume) |
| `--validate` | `False` | Run smoke tests only |
| `--upload-data` | `False` | Upload dataset to Modal Volume |
| `--dataset-file` | `data/sft/mixed_training.jsonl` | Local dataset path (for upload) |
| `--test` | `False` | Run inference test |
| `--test-prompt` | *"Write a function that reverses a list in Gleam"* | Prompt for inference test |
| `--test-experiment` | `None` | Experiment to test (defaults to base model) |
| `--gguf` | `False` | Export to GGUF after training |
| `--push-hub` | `False` | Push to HuggingFace Hub |
| `--hub-id` | `None` | Hub model ID (e.g. `user/model-name`) |

## Infrastructure

Three Modal Volumes persist data across runs:

| Volume | Mount | Contents |
|--------|-------|----------|
| `gleam-model-cache` | `/root/.cache/huggingface` | Downloaded model weights |
| `gleam-training-data` | `/data` | `mixed_training.jsonl` |
| `gleam-checkpoints` | `/checkpoints` | Per-experiment checkpoints + final model |

Training GPU: **NVIDIA RTX-PRO-6000** on Modal, with a maximum runtime of 12 hours. Validation's GPU smoke test uses an L4.

## Training Details

- **Method**: QLoRA (4-bit quantized base + LoRA adapters)
- **Trainable params**: ~73M / 8.07B (0.91%)
- **Response masking**: Only trains on assistant responses, not user prompts
- **Precision**: BF16 when supported by the training GPU
- **Optimizer**: AdamW 8-bit
- **LR schedule**: Cosine with 5% warmup
- **Early stopping**: Patience=3 evals, metric=eval_loss
- **Best model**: Loaded automatically at end of training
