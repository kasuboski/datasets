---
name: modal
description: >
  Build and run Python code on Modal serverless cloud infrastructure (GPU, CPU, storage).
  Use when writing or debugging Modal apps, choosing GPUs, managing Volumes/Secrets,
  deploying functions, or running training/inference workloads on cloud hardware.
---

# Modal Serverless Cloud

Modal runs Python functions in the cloud with zero infrastructure management. You define container images, attach GPUs/storage, and Modal handles the rest — billing is per-second with no idle costs.

## CLI Quick Reference

```bash
# Run a function (foreground, dies if local process disconnects)
modal run app.py::my_func

# Run detached — survives local disconnect. CRITICAL for long jobs (>5min).
modal run --detach app.py::my_func

# Stream logs from a detached run
modal app logs <app-name>

# Deploy as a persistent service
modal deploy app.py

# Interactive shell inside a container (same image/gpu/volumes as a function)
modal shell app.py::my_func
modal shell --gpu=A100 --image=debian-slim  # ad-hoc shell

# Volumes (persistent distributed filesystem)
modal volume list
modal volume ls <name> [path]
modal volume put <name> <local-path> [remote-path]
modal volume get <name> <remote-path> [local-dest]
modal volume create <name>

# Secrets (environment variables injected into containers)
modal secret list
modal secret create <name> KEY=VALUE ...

# Manage running/stopped apps
modal app list
modal app stop <app-name>
modal app logs <app-name>

# Billing
modal billing
```

See [CLI details](references/cli.md) for flags and less-common commands.

## App Skeleton

Every Modal script follows this pattern:

```python
import modal

# 1. Define the container image (dependencies)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .uv_pip_install("torch", "transformers")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 2. Attach resources
volume = modal.Volume.from_name("my-data", create_if_missing=True)
secret = modal.Secret.from_name("my-secret")

# 3. Create the app
app = modal.App("my-app", image=image)

# 4. Define remote functions
@app.function(gpu="A100", volumes={"/data": volume}, secrets=[secret], timeout=86400)
def train():
    import torch  # imports run inside the container
    assert torch.cuda.is_available()

# 5. Local entrypoint — runs on YOUR machine, calls remote functions
@app.local_entrypoint()
def main():
    train.remote()           # blocking call
    # train.spawn()           # fire-and-forget (returns a handle)
```

## Key Patterns & Gotchas

### Imports: container vs local
Code at module top-level runs **locally** when the script is parsed. Code inside `@app.function` bodies runs **inside the remote container**. Import heavy libs (torch, unsloth) inside the function body, not at module level.

Pre-import common lightweight deps with `image.imports()`:
```python
with image.imports():
    import datasets  # available in all @app.function bodies
```

### Long-running jobs: always `--detach`
`modal run` (foreground) kills the remote when the local process disconnects. For anything >5 minutes:
```bash
modal run --detach app.py::main
modal app logs <app-name>   # watch progress
```

### Volumes: write-once, read-many
Modal Volumes are distributed filesystems. They persist across runs but are optimized for large files. Use `Volume.batch_upload()` for uploading local files programmatically:
```python
@app.function(volumes={"/data": volume})
def upload():
    with volume.batch_upload() as batch:
        batch.put_file("local/train.jsonl", "/train.jsonl")
```
Or use the CLI: `modal volume put <name> <local> <remote>`.

### Retries for interruptible training
```python
@app.function(
    gpu="A100",
    timeout=86400,  # max 24h
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    volumes={...},
)
def train_interruptible():
    # Check for existing checkpoint, resume if found
    ...
```

### `.remote()` vs `.spawn()`
- `.remote()` — blocking, returns result. Call handle expires after 24h.
- `.spawn()` — fire-and-forget, returns a `FunctionCall` handle. Use `.spawn().get()` for detached long jobs.

### GPU fallbacks
```python
@app.function(gpu=["H100", "A100-40GB:2"])  # tries H100 first, falls back
def train(): ...
```

## GPU Selection

| GPU | VRAM | $/hr | Best for |
|-----|------|------|----------|
| T4 | 16 GB | $0.59 | Light inference, testing |
| L4 | 24 GB | $0.80 | Small models, dev/debug |
| A10 | 24 GB | $1.10 | Medium inference |
| L40S | 48 GB | $1.95 | Inference, medium training |
| A100-40GB | 40 GB | $2.10 | Training up to ~7B |
| A100-80GB | 80 GB | $2.50 | Training up to ~13B |
| RTX PRO 6000 | 48 GB | $3.03 | Training (high bandwidth) |
| H100 | 80 GB | $3.95 | Large training, fast inference |
| H200 | 141 GB | $4.54 | Largest models |
| B200 | 192 GB | $6.25 | Cutting-edge workloads |

See [GPU guide](references/gpu-guide.md) for detailed selection guidance and pricing math.

## Plans

| Plan | Cost | Free credits | GPU concurrency |
|------|------|-------------|-----------------|
| Starter | $0/mo | $30/mo | 10 |
| Team | $250/mo | $100/mo | 50 |

Starter plan covers most single-developer training runs. Volume storage: $0.09/GiB/mo (1 TiB free).

## Common Workflows

See reference docs:
- [GPU selection guide](references/gpu-guide.md) — how to pick the right GPU for your workload
- [CLI reference](references/cli.md) — detailed CLI commands and flags
- [Training patterns](references/training-patterns.md) — checkpointing, multi-GPU, resumable jobs
