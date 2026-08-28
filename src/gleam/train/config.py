"""Training hyperparameters and constants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Gemma 4 chat template markers for train_on_responses_only.
# These match the gemma-4 template: <|turn>user\n ... <|turn>model\n
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


@dataclass
class TrainingConfig:
    """All tunables for a single training run."""

    # Model
    model_name: str = "unsloth/gemma-4-E4B-it"
    max_seq_length: int = 4094
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True

    # Training
    epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    packing: bool = False
    use_gradient_checkpointing: str = "unsloth"

    # Train on completions only (masks user/system tokens)
    train_on_responses: bool = True

    # Data
    dataset_path: str = "/data/mixed_training.jsonl"
    eval_split: float = 0.05  # 5% held out for eval
    seed: int = 42

    # Experiment tracking
    experiment_name: Optional[str] = None
    trackio_space_id: Optional[str] = None  # e.g. "kasuboski/gleam-sft-tracking"

    # HF hub push
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None  # e.g. "kasuboski/gemma4-e4b-gleam-lora"

    # GGUF export (done inside training function for cleanest merge)
    export_gguf: bool = False
    gguf_quantization_method: str = "q4_k_m"

    def __post_init__(self):
        if self.experiment_name is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.experiment_name = f"gleam-sft-r{self.lora_r}-ep{self.epochs}-{ts}"
