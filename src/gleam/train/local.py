"""Local entrypoint and upload helper — runs on your machine, not in Modal."""

from __future__ import annotations

import pathlib
from typing import Optional

from src.gleam.train.config import INSTRUCTION_PART, RESPONSE_PART, TrainingConfig
from src.gleam.train.helpers import get_paths
from src.gleam.train.image import app, data_volume
from src.gleam.train.remote import (
    export_gguf,
    inference_test,
    push_to_hub,
    train,
    validate_gpu,
    validate_remote,
)


# ─── Upload helper (local, not decorated) ─────────────────────────────


def upload_dataset(local_path: str = "data/sft/mixed_training.jsonl"):
    """Upload the mixed training dataset to a Modal Volume.

    This runs LOCALLY (not decorated with @app.function) because it reads
    from the local filesystem. It writes directly into the Volume API.
    """
    src = pathlib.Path(local_path)
    if not src.exists():
        print(f"❌ File not found: {src}")
        return

    remote_name = f"/{src.name}"
    size_mb = src.stat().st_size / 1e6
    print(f"Uploading {src} ({size_mb:.1f} MB) → Volume:{remote_name}")

    with data_volume.batch_upload() as batch:
        batch.put_file(str(src), remote_name)

    print(f"✅ Uploaded to Volume 'gleam-training-data' as /data{remote_name}")


# ─── Local entrypoint ─────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    # Model
    model_name: str = "unsloth/gemma-4-E4B-it",
    max_seq_length: int = 4096,
    # LoRA
    lora_r: int = 32,
    lora_alpha: int = 64,
    use_rslora: bool = True,
    # Training
    epochs: int = 2,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    # Tracking
    trackio_space: Optional[str] = None,
    # Export
    gguf: bool = False,
    push_hub: bool = False,
    hub_id: Optional[str] = None,
    # Test
    test: bool = False,
    test_prompt: str = "Write a function that reverses a list in Gleam",
    test_experiment: Optional[str] = None,
    # Reuse an existing experiment (skip training, just export/test)
    experiment: Optional[str] = None,
    # Data
    upload_data: bool = False,
    dataset_file: str = "data/sft/mixed_training.jsonl",
    # Validation
    validate: bool = False,
):
    """Train (and optionally export/test/validate) a Gleam specialist model."""

    # ── Validate: local config check + remote import check ──
    if validate:
        print("=" * 60)
        print("🔍 Validating train module (no GPU, no cost)")
        print("=" * 60)

        # Part 1: Local config sanity check (instant, no Modal)
        print("\n── Part 1: Local config check ──")
        config = TrainingConfig(
            model_name=model_name,
            max_seq_length=max_seq_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            use_rslora=use_rslora,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            trackio_space_id=trackio_space,
            hub_model_id=hub_id,
        )
        print(f"  ✅ TrainingConfig created: {config.experiment_name}")

        # Check paths
        paths = get_paths(config)
        print(
            f"  ✅ Paths: checkpoints={paths['checkpoints']}, "
            f"final={paths['final_model']}"
        )

        # Check training math
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        print(f"  ✅ Effective batch size: {effective_batch}")

        # Check dataset file exists locally
        dataset_local = pathlib.Path(dataset_file)
        if dataset_local.exists():
            size_mb = dataset_local.stat().st_size / 1e6
            print(f"  ✅ Dataset exists: {dataset_local} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  Dataset not found locally: {dataset_local}")

        # Check chat template markers match known Gemma 4 format
        assert INSTRUCTION_PART == "<|turn>user\n", (
            f"Unexpected instruction_part: {INSTRUCTION_PART!r}"
        )
        assert RESPONSE_PART == "<|turn>model\n", (
            f"Unexpected response_part: {RESPONSE_PART!r}"
        )
        print(
            f"  ✅ Chat markers: instruction={INSTRUCTION_PART!r}, "
            f"response={RESPONSE_PART!r}"
        )

        # Part 2: Remote CPU check (builds image, no GPU)
        print("\n── Part 2: Remote CPU check (Modal image, no GPU) ──")
        cpu_results = validate_remote.remote()
        for r in cpu_results:
            print(r)

        # Part 3: Remote GPU check (L4, verifies unsloth + torch.cuda)
        print("\n── Part 3: Remote GPU check (Modal L4, validates unsloth) ──")
        gpu_results = validate_gpu.remote()
        for r in gpu_results:
            print(r)

        # Summary
        all_results = cpu_results + gpu_results
        failures = [r for r in all_results if r.startswith("  ❌")]
        sep = "=" * 60
        print(f"\n{sep}")
        if failures:
            print(f"❌ VALIDATION FAILED — {len(failures)} issue(s):")
            for f in failures:
                print(f)
        else:
            print("✅ ALL CHECKS PASSED — ready to train!")
        print(sep)
        return

    # ── Build config for training/test/export ──
    config = TrainingConfig(
        model_name=model_name,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_rslora=use_rslora,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        trackio_space_id=trackio_space,
        hub_model_id=hub_id,
        # When --gguf is passed with training, export inside the train function
        # (cleanest merge — model is already in memory with LoRA attached).
        # When --gguf is passed with --experiment (reuse), export separately.
        export_gguf=gguf and experiment is None,
    )

    # Upload dataset to Modal Volume if requested
    if upload_data:
        upload_dataset(local_path=dataset_file)
        # Don't fall through to training unless explicitly requested
        if not test and experiment is None:
            print("\n✅ Upload complete. Run training with:")
            print("  uv run modal run -m src.gleam.train --trackio-space kasuboski/gleam-sft-tracking")
            return

    if test:
        # Inference test: against a trained checkpoint if given, else base model
        inference_test.remote(
            prompt=test_prompt,
            experiment_name=test_experiment,
            model_name=model_name,
        )
        return

    # Use existing experiment name if provided, otherwise train fresh
    if experiment:
        experiment_name = experiment
        print(f"Reusing existing experiment: {experiment_name}")
    else:
        experiment_name = train.remote(config)
        print(f"\n🎉 Experiment: {experiment_name}")

    # Export GGUF separately only when reusing an existing experiment.
    # For fresh training runs, GGUF export happens inside train.remote()
    # for the cleanest merge (model in memory with LoRA attached).
    if gguf and experiment:
        export_gguf.remote(experiment_name)

    if push_hub and hub_id:
        push_to_hub.remote(experiment_name, hub_id)
