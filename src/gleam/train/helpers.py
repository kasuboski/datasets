"""Model loading, LoRA setup, and dataset preparation helpers.

These functions run inside @app.function containers, so they use local
imports for unsloth/datasets/torch (available via train_image.imports()).
"""

from __future__ import annotations

import pathlib

from src.gleam.train.config import INSTRUCTION_PART, RESPONSE_PART, TrainingConfig


def get_paths(config: TrainingConfig) -> dict:
    """Structured paths inside Modal volumes."""
    return {
        "checkpoints": pathlib.Path("/checkpoints") / config.experiment_name,
        "final_model": pathlib.Path("/checkpoints")
        / config.experiment_name
        / "final_model",
    }


def load_model(config: TrainingConfig):
    """Load base model + tokenizer with Unsloth optimizations.

    Uses FastModel (Unsloth's unified API for Gemma 4) which handles
    the multimodal quirks (KV-shared layers, use_cache bugs) automatically.
    """
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading model: {config.model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=config.load_in_4bit,
        full_finetuning=False,  # LoRA, not full FT
    )

    # Apply Gemma 4 chat template (non-thinking for E4B per Unsloth docs)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    return model, tokenizer


def setup_lora(model, config: TrainingConfig):
    """Attach LoRA adapters using Gemma 4 best practices.

    Uses FastModel.get_peft_model with the selective layer flags that
    Unsloth recommends for Gemma 4 (language + attention + MLP on,
    vision off for text-only fine-tuning).
    """
    from unsloth import FastModel

    print(
        f"Setting up LoRA: r={config.lora_r}, alpha={config.lora_alpha}, "
        f"rslora={config.use_rslora}"
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Text-only fine-tuning
        finetune_language_layers=True,  # Core language model
        finetune_attention_modules=True,  # Attention good for SFT
        finetune_mlp_modules=True,  # Always leave on
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
    return model


def load_and_prepare_dataset(config: TrainingConfig, tokenizer):
    """Load mixed training JSONL, format with chat template, split.

    Follows the Unsloth Gemma 4 recipe:
    1. standardize_data_formats() to normalize message structures
    2. apply_chat_template() to render conversations as text
    3. Strip leading <bos> (the tokenizer adds it during training)
    """
    import datasets

    print(f"Loading dataset from {config.dataset_path}")

    ds = datasets.load_dataset("json", data_files=config.dataset_path, split="train")
    print(f"  Total samples: {len(ds):,}")
    print(f"  Columns: {ds.column_names}")

    # Our mixed dataset uses 'messages' in standard role/content format.
    # standardize_data_formats expects a 'conversations' key — skip it.
    # Instead, just normalize the column name to 'conversations' if needed.
    if "conversations" not in ds.column_names and "messages" in ds.column_names:
        ds = ds.rename_column("messages", "conversations")
        print("  Renamed 'messages' → 'conversations'")

    # Apply chat template to convert conversations → tokenizable text.
    # Per Unsloth Gemma 4 guide: strip <bos> since the tokenizer adds it.
    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        return {"text": text}

    ds = ds.map(
        apply_template,
        batched=False,
        desc="Applying chat template",
        remove_columns=ds.column_names,
    )

    # Train/eval split
    split = ds.train_test_split(test_size=config.eval_split, seed=config.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    return train_ds, eval_ds
