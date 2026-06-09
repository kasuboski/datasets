# Training Patterns on Modal

## Resumable Training with Checkpoints

The fundamental pattern for any training job that might be interrupted:

```python
from pathlib import Path
import modal

volume = modal.Volume.from_name("training-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = Path("/checkpoints")

@app.function(
    gpu="A100",
    volumes={CHECKPOINT_DIR: volume},
    timeout=86400,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
)
def train(experiment: str):
    exp_dir = CHECKPOINT_DIR / experiment
    latest = find_latest_checkpoint(exp_dir)

    if latest:
        print(f"Resuming from {latest}")
        model = load_checkpoint(latest)
        start_step = model.global_step
    else:
        print("Starting from scratch")
        model = init_model()
        start_step = 0

    for step in range(start_step, total_steps):
        train_step(model)
        if step % save_every == 0:
            save_checkpoint(exp_dir, step, model)
            volume.commit()  # force flush to distributed storage

    save_final(exp_dir / "final_model", model)
    volume.commit()
```

Key points:
- **`volume.commit()`** after saves — forces flush to distributed storage. Without it, data may not survive container restart.
- **`retries=modal.Retries(...)`** — automatically restarts on timeout/preemption. Combined with checkpointing, this gives interruptible training.
- **`timeout=86400`** — max 24 hours per function call. With 10 retries, that's up to 10 days total.

## Running Detached Training

```bash
# Start detached
modal run --detach train.py::main --experiment my-exp-v1

# Watch logs
modal app logs -f gleam-sft-finetune

# Stop if needed
modal app stop gleam-sft-finetune
```

**Critical:** `modal run` without `--detach` kills the remote when your local process disconnects. This is the #1 cause of "my training died" issues.

Also: `.remote()` call handles expire after 24h. For detached long jobs, use `.spawn(...).get()` instead:

```python
@app.local_entrypoint()
def main(experiment: str = typer.Option(None)):
    handle = train.spawn(experiment)
    handle.get()  # blocks until done, survives local disconnect
```

## Multi-GPU Training

```python
@app.function(gpu="A100:4", timeout=86400)
def train_multi_gpu():
    import subprocess, sys
    # If your framework re-executes the entrypoint (e.g., PyTorch Lightning),
    # run the script as a subprocess:
    subprocess.run(
        ["torchrun", "--nproc_per_node=4", "train.py"],
        stdout=sys.stdout, stderr=sys.stderr, check=True,
    )
```

- Multi-GPU always on same physical machine
- Requesting >2 GPUs usually means longer queue times
- Cost scales linearly: `A100:4` = 4 × $2.50/hr = $10/hr

## Image Construction for ML

```python
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("cmake", "git", "curl", "build-essential")
    .uv_pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "accelerate>=1.0.0",
        "peft>=0.10.0",
        "trl>=0.10.0",
        "datasets>=3.0.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model downloads
        "HF_HOME": "/root/.cache/huggingface",
    })
)

# Pre-import lightweight deps (heavy deps like torch/unsloth should be
# imported inside function bodies to avoid CPU-side import errors)
with train_image.imports():
    import datasets
```

Gotchas:
- **Unsloth raises `NotImplementedError` on CPU.** Don't import it in `image.imports()` or at module level. Import inside `@app.function` body only.
- **Use `uv_pip_install`** (not `pip_install`) — faster, Modal's default.
- **Pin versions** for reproducibility, especially for torch/transformers compatibility.

## Volume Data Flow

Typical data lifecycle:

1. **Build dataset locally** → `data/train.jsonl`
2. **Upload to Volume** → `modal volume put my-data ./data/train.jsonl /data/`
   - Or programmatically: `volume.batch_upload()` inside a function
3. **Training reads** from mounted Volume → `/data/train.jsonl`
4. **Checkpoints write** to mounted Volume → `/checkpoints/exp-name/`
5. **Download final model** → `modal volume get my-data /checkpoints/exp/final_model/ ./output/`

Mount multiple volumes:
```python
@app.function(volumes={
    "/data": data_volume,
    "/checkpoints": checkpoint_volume,
    "/root/.cache/huggingface": model_cache_volume,
})
def train(): ...
```

## Monitoring Training

Options for tracking metrics:
1. **Modal dashboard** — `modal app logs -f <name>` for stdout
2. **Weights & Biases** — `pip_install("wandb")`, set `WANDB_API_KEY` via Secret
3. **Trackio** — free, HuggingFace-native, `report_to="trackio"` in TRL/transformers
4. **TensorBoard** — write logs to Volume, download and view locally

## Cost Estimation

Formula: `cost = hours × $/hr`

Before a training run:
1. Know your total steps and seconds/step (run 10 steps on a small GPU to estimate)
2. `total_hours = (steps × sec/step) / 3600`
3. `cost = total_hours × gpu_price`

Example: 1,600 steps × 11s/step = 17,600s ≈ 4.9h on A100-40GB ($2.10/hr) = **$10.29**
