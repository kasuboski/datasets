#!/usr/bin/env python3
"""
Sample and prepare general instruction and functional code datasets for Gleam SFT mixing.

This script:
1. Samples 3K general instruction pairs from HuggingFaceH4/ultrachat_200k
2. Samples 500 Python code pairs from nvidia/OpenCodeInstruct (converts to ChatML)
3. Extracts all ~683 functional language pairs from MathAndMagic/Code-290k-ShareGPT-MarkedLanguage
   (Haskell, Elixir, Erlang, Scala, Clojure, F#, OCaml, Lisp — converts ShareGPT → ChatML)

All outputs are saved as JSONL in ChatML format with a 'messages' field.

Usage:
    uv run python scripts/sample_general_datasets.py
    uv run python scripts/sample_general_datasets.py --output-dir data/sft
"""

import json
import argparse
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

FUNCTIONAL_LANGUAGES = ["haskell", "elixir", "erlang", "scala", "clojure", "fsharp", "ocaml", "lisp"]


def sample_ultrachat(output_dir: Path) -> Path:
    """Sample 3K general instruction pairs from ultrachat_200k."""
    out_path = output_dir / "ultrachat_3k.jsonl"
    if out_path.exists():
        count = sum(1 for _ in open(out_path))
        print(f"  ✅ Already exists: {out_path} ({count:,} samples) — skipping")
        return out_path

    print("=" * 60)
    print("1. Sampling ultrachat_200k (general instruction)")
    print("=" * 60)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    print(f"  Loaded {len(ds):,} samples")

    sampled = ds.shuffle(seed=42).select(range(3000))

    out_path = output_dir / "ultrachat_3k.jsonl"
    count = 0
    with open(out_path, "w") as f:
        for example in sampled:
            messages = example["messages"]
            # Add system prompt if missing
            if not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages

            record = {"messages": messages, "source": "ultrachat_200k"}
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"  ✅ Saved {count:,} samples → {out_path}")
    return out_path


def sample_opencode(output_dir: Path) -> Path:
    """Sample 500 Python code pairs from OpenCodeInstruct and convert to ChatML.

    Uses streaming to avoid downloading the full 5M row dataset.
    """
    out_path = output_dir / "opencode_500_chatml.jsonl"
    if out_path.exists():
        count = sum(1 for _ in open(out_path))
        print(f"  ✅ Already exists: {out_path} ({count:,} samples) — skipping")
        return out_path

    print("\n" + "=" * 60)
    print("2. Sampling nvidia/OpenCodeInstruct (general code)")
    print("=" * 60)

    TARGET = 500
    out_path = output_dir / "opencode_500_chatml.jsonl"

    # Use streaming to avoid loading 5M rows into memory
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    # Shuffle via a buffer (not perfect but avoids full download)
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    count = 0
    with open(out_path, "w") as f:
        for example in ds:
            if count >= TARGET:
                break
            messages = [
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]},
            ]
            record = {"messages": messages, "source": "opencodeinstruct"}
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"  ✅ Saved {count:,} samples → {out_path}")
    return out_path


def extract_functional(output_dir: Path) -> Path:
    """Extract all functional language pairs from Code-290k-ShareGPT-MarkedLanguage."""
    out_path = output_dir / "functional_683_chatml.jsonl"
    if out_path.exists():
        count = sum(1 for _ in open(out_path))
        print(f"  ✅ Already exists: {out_path} ({count:,} samples) — skipping")
        return out_path

    print("\n" + "=" * 60)
    print("3. Extracting functional language pairs from Code-290k-ShareGPT-MarkedLanguage")
    print("=" * 60)

    ds = load_dataset("MathAndMagic/Code-290k-ShareGPT-MarkedLanguage", split="train")
    print(f"  Loaded {len(ds):,} total samples")

    # Filter for functional languages
    functional_ds = ds.filter(lambda x: x["language"] in FUNCTIONAL_LANGUAGES)
    print(f"  Filtered to {len(functional_ds):,} functional language samples")

    out_path = output_dir / "functional_683_chatml.jsonl"
    count = 0
    lang_counts = {}
    with open(out_path, "w") as f:
        for example in functional_ds:
            lang = example["language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            # Convert ShareGPT format (from/value) → ChatML (role/content)
            messages = []
            for conv in example["conversations"]:
                role = "user" if conv["from"] == "human" else "assistant"
                messages.append({"role": role, "content": conv["value"]})

            # Add system prompt
            messages = [
                {"role": "system", "content": "You are a helpful programming assistant."}
            ] + messages

            # Validate: last message must be assistant
            if messages[-1]["role"] != "assistant":
                continue

            record = {
                "messages": messages,
                "source": "code-290k-functional",
                "language": lang,
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"  Language breakdown:")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {cnt}")
    print(f"  ✅ Saved {count:,} samples → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Sample general datasets for Gleam SFT mixing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sft"),
        help="Output directory for sampled datasets",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ultrachat_path = sample_ultrachat(args.output_dir)
    opencode_path = sample_opencode(args.output_dir)
    functional_path = extract_functional(args.output_dir)

    print("\n" + "=" * 60)
    print("Done! Output files:")
    print(f"  {ultrachat_path}")
    print(f"  {opencode_path}")
    print(f"  {functional_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
