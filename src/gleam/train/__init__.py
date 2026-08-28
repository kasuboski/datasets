"""Gleam SFT fine-tuning — Modal + Unsloth + Gemma 4.

Usage:
    modal run -m gleam.train                           # train with defaults
    modal run -m gleam.train --validate                # smoke test (no GPU)
    modal run -m gleam.train --upload-data             # upload dataset
    modal run -m gleam.train --test                    # inference test
    modal run -m gleam.train --gguf --push-hub --hub-id kasuboski/gemma4-e4b-gleam-lora
"""

# Import local to register @app.local_entrypoint with Modal
from src.gleam.train import local  # noqa: F401

# Re-export primary public API
from src.gleam.train.config import INSTRUCTION_PART, RESPONSE_PART, TrainingConfig
from src.gleam.train.image import app, train_image
