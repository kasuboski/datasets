"""All @app.function-decorated remote functions (GPU and CPU)."""

from __future__ import annotations

from typing import Optional

from src.gleam.train.config import INSTRUCTION_PART, RESPONSE_PART, TrainingConfig
from src.gleam.train.helpers import (
    get_paths,
    load_and_prepare_dataset,
    load_model,
    setup_lora,
)
from src.gleam.train.image import (
    app,
    checkpoint_volume,
    data_volume,
    hf_secret,
    model_cache,
    train_image,
)

import modal


# ─── GGUF conversion helper (llama.cpp) ─────────────────────────────────


def _convert_to_gguf(
    merged_model_dir: str,
    output_dir: str,
    quantization_method: str = "q4_k_m",
) -> str:
    """Convert merged 16-bit safetensors → GGUF using llama.cpp tools.

    Two-step process:
    1. convert_hf_to_gguf.py — safetensors → F16 GGUF
    2. llama-quantize — F16 GGUF → quantized GGUF (e.g. Q4_K_M)

    This replaces Unsloth's save_pretrained_gguf() which produces broken
    GGUFs for Gemma 4 (tokenization bug, see llama.cpp PR #21343).
    """
    import os
    import subprocess

    llama_cpp = os.environ.get("LLAMA_CPP_PATH", "/opt/llama.cpp")
    llama_cpp_bin = os.environ.get("LLAMA_CPP_BIN", "/opt/llama-cpp-bin")
    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_bin, "llama-quantize")

    # Step 1: Convert safetensors → F16 GGUF
    f16_gguf = os.path.join(output_dir, "f16.gguf")
    if not os.path.exists(f16_gguf):
        print(f"  [1/2] Converting safetensors → F16 GGUF …")
        result = subprocess.run(
            [
                "python3", convert_script,
                merged_model_dir,
                "--outfile", f16_gguf,
                "--outtype", "f16",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  convert_hf_to_gguf.py FAILED:")
            print(f"  stdout: {result.stdout[-2000:]}")
            print(f"  stderr: {result.stderr[-2000:]}")
            raise RuntimeError("convert_hf_to_gguf.py failed")
        print(f"  F16 GGUF → {f16_gguf}")
    else:
        print(f"  [1/2] Reusing existing F16 GGUF: {f16_gguf}")

    # Step 2: Quantize F16 → target format
    final_gguf = os.path.join(output_dir, f"{quantization_method}.gguf")
    if not os.path.exists(final_gguf):
        print(f"  [2/2] Quantizing F16 → {quantization_method} …")
        # llama-quantize uses uppercase format names (Q4_K_M)
        qm_upper = quantization_method.upper()
        result = subprocess.run(
            [quantize_bin, f16_gguf, final_gguf, qm_upper],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "LD_LIBRARY_PATH": f"{llama_cpp_bin}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            },
        )
        if result.returncode != 0:
            print(f"  llama-quantize FAILED:")
            print(f"  stdout: {result.stdout[-2000:]}")
            print(f"  stderr: {result.stderr[-2000:]}")
            raise RuntimeError("llama-quantize failed")
        print(f"  {qm_upper} GGUF → {final_gguf}")
    else:
        print(f"  [2/2] Reusing existing quantized GGUF: {final_gguf}")

    return final_gguf


# ─── Training (GPU) ───────────────────────────────────────────────────


@app.function(
    image=train_image,
    gpu="RTX-PRO-6000",
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=hf_secret,
    timeout=12 * 60 * 60,  # 12 hours max (longer seq = more time)
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    single_use_containers=True,
)
def train(config: TrainingConfig) -> str:
    paths = get_paths(config)
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)

    # ── Auto-detect checkpoint to resume from ──
    resume_checkpoint = None
    existing_checkpoints = sorted(
        [d for d in paths["checkpoints"].iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if existing_checkpoints:
        resume_checkpoint = str(existing_checkpoints[-1])
        print(f"📂 Found checkpoint to resume from: {resume_checkpoint}")
    else:
        print("🆕 No existing checkpoint — starting fresh")

    # ── Load model ──
    model, tokenizer = load_model(config)
    model = setup_lora(model, config)

    # ── Load dataset ──
    train_ds, eval_ds = load_and_prepare_dataset(config, tokenizer)

    # ── Compute training steps ──
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * config.epochs
    save_steps = max(steps_per_epoch // 4, 10)  # ~4 checkpoints per epoch

    print(f"\nTraining plan:")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Total steps:         {total_steps}")
    print(f"  Save every:          {save_steps} steps")
    print(f"  Expected loss:       13-15 initial (normal for Gemma 4), target 1-3 final")

    # ── Configure tracking ──
    report_to = None
    if config.trackio_space_id:
        import os

        os.environ["TRACKIO_SPACE_ID"] = config.trackio_space_id
        os.environ["TRACKIO_PROJECT"] = "gleam-specialist"
        report_to = "trackio"

    # ── Create trainer ──
    from trl import SFTConfig, SFTTrainer
    import torch

    training_args = SFTConfig(
        # Core
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        max_steps=total_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        # Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # Data
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=config.packing,
        # Checkpointing
        output_dir=str(paths["checkpoints"]),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=4,
        resume_from_checkpoint=resume_checkpoint,
        # Eval
        eval_strategy="steps",
        eval_steps=save_steps,
        # Early stopping — stop if eval loss doesn't improve for 3 evals
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=5,
        report_to=report_to,
        run_name=config.experiment_name,
        seed=config.seed,
        # Hub
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # ── Early stopping — halt if eval loss plateaus for 3 consecutive evals ──
    from transformers import EarlyStoppingCallback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    print("Early stopping enabled: patience=3 evals (stops if eval_loss doesn't improve for 3 checks)")

    # ── Train on completions only (masks user/system tokens) ──
    if config.train_on_responses:
        from unsloth.chat_templates import train_on_responses_only

        print("\nEnabling train_on_responses_only (masking user/system tokens)")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_PART,
            response_part=RESPONSE_PART,
        )

    # ── Train ──
    print(f"\n{'=' * 60}")
    print(f"Starting training: {config.experiment_name}")
    print(f"{'=' * 60}\n")

    trainer.train()

    # ── Save final model (LoRA adapters + tokenizer) ──
    final_path = paths["final_model"]
    print(f"\nSaving LoRA adapters to {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # ── Export GGUF via llama.cpp (Unsloth's converter is broken for Gemma 4) ──
    if config.export_gguf:
        qm = config.gguf_quantization_method
        print(f"\nExporting GGUF via llama.cpp (method={qm})…")

        # Step 1: Merge LoRA into base and save full 16-bit safetensors.
        # save_pretrained_merged actually merges LoRA into base weights
        # and saves full safetensors + config.json. Regular save_pretrained
        # only saves the LoRA adapter (no config.json, no model weights).
        merged_path = str(paths["checkpoints"] / "merged_16bit")
        print(f"  Merging LoRA into base → {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer)

        # Step 2: Convert to GGUF using llama.cpp's convert_hf_to_gguf.py
        _convert_to_gguf(merged_path, str(paths["checkpoints"]), qm)

        # Commit after export
        checkpoint_volume.commit()

    # Commit volumes so data persists
    checkpoint_volume.commit()

    print(f"\n{'=' * 60}")
    print(f"✅ Training complete: {config.experiment_name}")
    print(f"   Model saved to: {final_path}")
    print(f"{'=' * 60}")

    return config.experiment_name


# ─── Export (GPU) ─────────────────────────────────────────────────────


@app.function(
    image=train_image,
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/checkpoints": checkpoint_volume,
    },
    secrets=hf_secret,
    timeout=60 * 60,  # merge + convert + quantize can take ~30min
)
def export_gguf(experiment_name: str, quantization_method: str = "q4_k_m") -> str:
    """Export trained LoRA adapter to GGUF for llama.cpp / Ollama.

    Uses llama.cpp's convert_hf_to_gguf.py instead of Unsloth's
    save_pretrained_gguf(), which produces broken GGUFs for Gemma 4
    (see: https://github.com/ggml-org/llama.cpp/pull/21343).

    Steps:
    1. Load LoRA adapter (Unsloth auto-merges with base model)
    2. Save merged 16-bit safetensors
    3. Convert to F16 GGUF via convert_hf_to_gguf.py
    4. Quantize to requested format (e.g. Q4_K_M) via llama-quantize
    """
    import subprocess

    model_path = f"/checkpoints/{experiment_name}/final_model"
    output_dir = f"/checkpoints/{experiment_name}"

    # Check if merged_16bit already exists (from a previous export attempt)
    merged_path = f"{output_dir}/merged_16bit"
    import os

    if not os.path.exists(os.path.join(merged_path, "config.json")):
        from unsloth import FastModel

        print(f"Loading fine-tuned model from {model_path}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
        )

        # save_pretrained_merged actually merges LoRA into base weights
        # and saves full safetensors + config.json. Regular save_pretrained
        # only saves the LoRA adapter (no config.json, no model weights).
        print(f"Merging LoRA into base model → {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer)
    else:
        print(f"Reusing existing merged 16-bit model at {merged_path}")

    # Convert + quantize
    _convert_to_gguf(merged_path, output_dir, quantization_method)

    checkpoint_volume.commit()

    gguf_path = f"{output_dir}/{quantization_method}.gguf"
    print(f"✅ Exported GGUF: {gguf_path}")
    return gguf_path


@app.function(
    image=train_image,
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/checkpoints": checkpoint_volume,
    },
    secrets=hf_secret,
    timeout=60 * 60,
)
def push_to_hub(experiment_name: str, hub_model_id: str) -> str:
    """Push trained LoRA adapter and GGUF to HuggingFace Hub (two repos).

    Creates two repos following HF convention:
    - {hub_model_id}           → LoRA adapter (safetensors + tokenizer + config)
    - {hub_model_id}-GGUF      → GGUF quantization for llama.cpp / Ollama

    Both repos get a README.md model card with training details and usage.
    """
    import os
    from huggingface_hub import HfApi

    model_path = f"/checkpoints/{experiment_name}/final_model"
    gguf_path = f"/checkpoints/{experiment_name}/q4_k_m.gguf"

    api = HfApi()

    # ── Repo 1: LoRA adapter ──
    lora_repo = hub_model_id
    print(f"\n{'='*60}")
    print(f"📦 Pushing LoRA adapter to {lora_repo}")
    print(f"{'='*60}")

    api.create_repo(repo_id=lora_repo, exist_ok=True)

    print(f"  Uploading LoRA adapter from {model_path}")
    api.upload_folder(
        folder_path=model_path,
        repo_id=lora_repo,
        # Don't overwrite our model card if we write one separately
        ignore_patterns=["README.md"],
    )

    # Write model card for LoRA repo
    lora_readme = _lora_model_card(experiment_name, hub_model_id)
    api.upload_file(
        path_or_fileobj=lora_readme.encode(),
        path_in_repo="README.md",
        repo_id=lora_repo,
    )
    print(f"  ✅ LoRA adapter + model card → https://huggingface.co/{lora_repo}")

    # ── Repo 2: GGUF ──
    gguf_repo = f"{hub_model_id}-GGUF"
    print(f"\n{'='*60}")
    print(f"📦 Pushing GGUF to {gguf_repo}")
    print(f"{'='*60}")

    api.create_repo(repo_id=gguf_repo, exist_ok=True)

    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / 1e9
        print(f"  Uploading GGUF ({size_gb:.1f} GB) from {gguf_path}")
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo="gemma-4-e4b-gleam-sft-Q4_K_M.gguf",
            repo_id=gguf_repo,
        )
    else:
        print(f"  ⚠️ No GGUF found at {gguf_path} — skipping GGUF upload")

    # Write model card for GGUF repo
    gguf_readme = _gguf_model_card(experiment_name, hub_model_id)
    api.upload_file(
        path_or_fileobj=gguf_readme.encode(),
        path_in_repo="README.md",
        repo_id=gguf_repo,
    )
    print(f"  ✅ GGUF + model card → https://huggingface.co/{gguf_repo}")

    return lora_repo


def _lora_model_card(experiment_name: str, hub_model_id: str) -> str:
    """Generate model card for the LoRA adapter repo."""
    return f"""---
base_model: unsloth/gemma-4-e4b-it-unsloth-bnb-4bit
language:
- en
license: gemma
library_name: peft
tags:
- unsloth
- gleam
- code-generation
- functional-programming
- lora
- gemma4
---

# {hub_model_id}

A LoRA adapter fine-tuned on [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it) for Gleam code generation.

This is the adapter-only repo. For a ready-to-run GGUF (Ollama / llama.cpp), see [{hub_model_id}-GGUF](https://huggingface.co/{hub_model_id}-GGUF).

## Dataset

Trained on a mixed dataset of 13,538 instruction-response pairs:

| Source | Count | % | Purpose |
|--------|-------|---|----------|
| [gleam-instruct](https://huggingface.co/datasets/kasuboski/gleam-instruct) | 9,355 | 69.1% | Gleam code generation and completion |
| [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 3,000 | 22.2% | General instruction — anti-forgetting |
| [code-290k-functional](https://huggingface.co/datasets/MathAndMagic/Code-290k-ShareGPT-MarkedLanguage) | 683 | 5.0% | Functional code — anti-forgetting |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | 500 | 3.7% | Python code — anti-forgetting |

### gleam-instruct (69.1%)

The Gleam contribution contains 9,355 instruction-response pairs. The retained task types are:
- **code_gen** — write new Gleam code from requirements
- **completion** — complete partial Gleam implementations

The `explanation` task type was excluded from training.

### Anti-forgetting mix (30.9%)

To preserve general capabilities while specializing on Gleam:
- **General instruction** (ultrachat_200k, MIT) — 3,000 samples to maintain conversational ability
- **Functional code** (code-290k-functional) — 683 samples to maintain functional programming knowledge
- **Python code** (OpenCodeInstruct, CC-BY-4.0) — 500 samples to maintain general code ability

## Usage

### With Unsloth (recommended)

```python
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
model.load_adapter("{hub_model_id}")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

messages = [
    {{"role": "system", "content": "You are a helpful Gleam programming assistant."}},
    {{"role": "user", "content": "Write a Gleam function that reverses a list."}},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.3)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

### With PEFT + Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("google/gemma-4-e4b-it", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "{hub_model_id}")
tokenizer = AutoTokenizer.from_pretrained("{hub_model_id}")
```

## Recommended System Prompt

For best results, use this system prompt:

```
You are a Gleam programming expert. You ONLY write valid Gleam code. Gleam syntax rules: comments use //, function signatures use -> not ::, no where clauses, no do notation, imports use gleam/module style, use pub fn for public functions, pipe operator |> for chaining. Do NOT use Haskell syntax.
```

## Known Issues

- **Haskell syntax bleed**: The model may occasionally output Haskell syntax (`::` type signatures, `--` comments, `Data.*` imports). This is caused by 156 Haskell samples in the anti-forgetting mix. A strong system prompt (above) mitigates this. A future training run should exclude Haskell.
- **Unsloth GGUF export is broken for Gemma 4**: Do NOT use `model.save_pretrained_gguf()`. Use llama.cpp's `convert_hf_to_gguf.py` instead (see the GGUF repo for details).

## License

This model is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
"""


def _gguf_model_card(experiment_name: str, hub_model_id: str) -> str:
    """Generate model card for the GGUF repo."""
    gguf_repo = f"{hub_model_id}-GGUF"
    return f"""---
base_model: unsloth/gemma-4-e4b-it-unsloth-bnb-4bit
language:
- en
license: gemma
tags:
- gguf
- gleam
- code-generation
- functional-programming
- ollama
- llama-cpp
- gemma4
quantized_by: kasuboski
---

# {gguf_repo}

Quantized GGUF of [{hub_model_id}](https://huggingface.co/{hub_model_id}) for use with Ollama and llama.cpp.

Fine-tuned on [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it) for Gleam code generation. See the LoRA repo for [full dataset details](https://huggingface.co/{hub_model_id}#dataset).

## Available Quantizations

| File | Format | Size |
|------|--------|------|
| `gemma-4-e4b-gleam-sft-Q4_K_M.gguf` | Q4_K_M | ~5.0 GB |

## Export Pipeline

This GGUF was exported using **llama.cpp** — NOT Unsloth's `save_pretrained_gguf()` (which produces broken GGUFs for Gemma 4).

1. Merge LoRA into base via `model.save_pretrained_merged()` → 16-bit safetensors
2. Convert to F16 GGUF via `python convert_hf_to_gguf.py`
3. Quantize to Q4_K_M via `llama-quantize`

## Usage

### Ollama

```bash
# Pull and run directly from HuggingFace
ollama run hf.co/{gguf_repo}:Q4_K_M
```

When using via HF, you must set a system prompt for best results. For a batteries-included experience, use the [Ollama registry version](https://ollama.com/kasuboski/gemma-4-e4b-gleam-sft) which bakes in the system prompt, chat template, and 128K context:

```bash
ollama run kasuboski/gemma-4-e4b-gleam-sft
```

### llama.cpp

```bash
./llama-cli -m gemma-4-e4b-gleam-sft-Q4_K_M.gguf \\
  --system-prompt "You are a Gleam programming expert. You ONLY write valid Gleam code." \\
  -p "Write a Gleam function that reverses a list." \\
  --temp 0.3
```

## Recommended System Prompt

```
You are a Gleam programming expert. You ONLY write valid Gleam code. Gleam syntax rules: comments use //, function signatures use -> not ::, no where clauses, no do notation, imports use gleam/module style, use pub fn for public functions, pipe operator |> for chaining. Do NOT use Haskell syntax.
```

## Chat Template

Uses **Gemma 4** format (`<|turn>role\n...<turn|>`), NOT Gemma 2 (`<start_of_turn>`). The GGUF does not embed the chat template or system prompt — configure these in your inference engine.

## Known Issues

- **Haskell syntax bleed**: May occasionally output Haskell syntax. A strong system prompt mitigates this.

## License

This model is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
"""


@app.function(
    image=train_image,
    gpu="L4",
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/checkpoints": checkpoint_volume,
    },
    timeout=10 * 60,
)
def inference_test(
    prompt: str = "Write a function that reverses a list in Gleam",
    experiment_name: Optional[str] = None,
    model_name: str = "unsloth/gemma-4-E4B-it",
) -> str:
    """Quick inference test — on a fine-tuned checkpoint if experiment_name
    given, otherwise on the base model (useful for baseline comparison)."""
    if experiment_name:
        model_path = f"/checkpoints/{experiment_name}/final_model"
        print(f"Loading fine-tuned model from {model_path}")
    else:
        model_path = model_name
        print(f"Loading base model: {model_path}")

    print(f"Prompt: {prompt}\n")

    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(model_name=model_path)
    FastModel.for_inference(model)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful Gleam programming assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"Response:\n{response}")
    return response


# ─── Validate ─────────────────────────────────────────────────────────


@app.function(
    image=train_image,
    timeout=5 * 60,
)
def validate_remote() -> list[str]:
    """CPU-safe checks — verify non-GPU imports and packages are installed."""
    results = []

    def check(label: str, fn):
        try:
            fn()
            results.append(f"  ✅ {label}")
        except Exception as e:
            results.append(f"  ❌ {label}: {e}")

    check("import datasets", lambda: __import__("datasets"))
    check("import trackio", lambda: __import__("trackio"))
    check(
        "from trl import SFTTrainer, SFTConfig",
        lambda: (__import__("trl").SFTTrainer, __import__("trl").SFTConfig),
    )
    check("from peft import LoraConfig", lambda: __import__("peft").LoraConfig)
    check(
        "from transformers import AutoTokenizer",
        lambda: __import__("transformers").AutoTokenizer,
    )

    # Verify llama.cpp conversion tools are installed
    def _check_llama_cpp():
        import os
        llama_cpp = os.environ.get("LLAMA_CPP_PATH", "/opt/llama.cpp")
        llama_cpp_bin = os.environ.get("LLAMA_CPP_BIN", "/opt/llama-cpp-bin")
        convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
        quantize_bin = os.path.join(llama_cpp_bin, "llama-quantize")
        assert os.path.exists(convert_script), f"convert_hf_to_gguf.py not found at {convert_script}"
        assert os.path.exists(quantize_bin), f"llama-quantize not found at {quantize_bin}"

    check("llama.cpp convert_hf_to_gguf.py + llama-quantize installed", _check_llama_cpp)

    return results


@app.function(
    image=train_image,
    gpu="L4",
    timeout=5 * 60,
)
def validate_gpu() -> list[str]:
    """GPU checks — verify unsloth imports and API surfaces work with a GPU."""
    results = []

    def check(label: str, fn):
        try:
            fn()
            results.append(f"  ✅ {label}")
        except Exception as e:
            results.append(f"  ❌ {label}: {e}")

    # 1. Core GPU imports
    check("import torch", lambda: __import__("torch"))
    check(
        "torch.cuda.is_available()",
        lambda: __import__("torch").cuda.is_available(),
    )
    check("import unsloth", lambda: __import__("unsloth"))
    check(
        "from unsloth import FastModel",
        lambda: __import__("unsloth").FastModel,
    )

    # 2. Chat template imports
    def _check_chat_imports():
        from unsloth.chat_templates import (
            get_chat_template,
            standardize_data_formats,
            train_on_responses_only,
        )

        assert callable(get_chat_template)
        assert callable(standardize_data_formats)
        assert callable(train_on_responses_only)

    check(
        "from unsloth.chat_templates import get_chat_template, "
        "standardize_data_formats, train_on_responses_only",
        _check_chat_imports,
    )

    # 3. Verify train_on_responses_only accepts the right kwargs
    def _check_toro_signature():
        import inspect

        from unsloth.chat_templates import train_on_responses_only

        sig = inspect.signature(train_on_responses_only)
        params = list(sig.parameters.keys())
        assert "instruction_part" in params, f"missing instruction_part, got: {params}"
        assert "response_part" in params, f"missing response_part, got: {params}"

    check(
        "train_on_responses_only accepts instruction_part / response_part",
        _check_toro_signature,
    )

    # 4. Verify FastModel.get_peft_model accepts finetune_* flags
    def _check_peft_flags():
        import inspect

        from unsloth import FastModel

        sig = inspect.signature(FastModel.get_peft_model)
        params = list(sig.parameters.keys())
        for flag in [
            "finetune_vision_layers",
            "finetune_language_layers",
            "finetune_attention_modules",
            "finetune_mlp_modules",
        ]:
            assert flag in params, f"missing {flag}, got: {params}"

    check(
        "FastModel.get_peft_model has finetune_vision/language/attention/mlp_modules flags",
        _check_peft_flags,
    )

    # 5. Verify FastModel.from_pretrained accepts full_finetuning kwarg
    def _check_loader_kwargs():
        import inspect

        from unsloth import FastModel

        sig = inspect.signature(FastModel.from_pretrained)
        params = list(sig.parameters.keys())
        assert "full_finetuning" in params, f"missing full_finetuning, got: {params}"

    check(
        "FastModel.from_pretrained has full_finetuning kwarg",
        _check_loader_kwargs,
    )

    return results
