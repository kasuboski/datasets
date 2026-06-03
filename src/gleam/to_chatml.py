#!/usr/bin/env python3
"""Convert train.jsonl to ChatML format for publication.

Reconstructs the full conversation (system + user + assistant messages) from:
  - train.jsonl: assistant responses + metadata
  - gleam_corpus.parquet: source code files
  - prompts.py: system prompt + task templates

For code_gen tasks, a fresh random instruction is assigned (the original
instruction was chosen via random.choice in threaded execution, so exact
replay is not possible — but the resulting pairs are equally valid).

Output schema matches docs/sft-strategy.md specification.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

# Add repo root to path for src.* imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

# Reuse the same templates from the generation pipeline
from src.gleam.pilot.prompts import (
    CODE_GEN_INSTRUCTIONS,
    SYSTEM_PROMPT,
    SINGLE_PROMPTS,
)


def build_user_message(
    task_type: str,
    code: str,
    repo_name: str,
    instruction: Optional[str] = None,
) -> str:
    """Reconstruct the user message for a given task type and source code.

    Mirrors build_prompt_for_file() in generate.py but returns just the user text.
    """
    source_label = f"repo {repo_name}"

    if task_type == "code_gen":
        if instruction is None:
            instruction = random.choice(CODE_GEN_INSTRUCTIONS)
        return SINGLE_PROMPTS["code_gen"].format(
            code=code,
            source_label=source_label,
            instruction=instruction,
        )

    elif task_type == "explanation":
        return SINGLE_PROMPTS["explanation"].format(code=code)

    elif task_type == "completion":
        lines = code.split("\n")
        cutoff = max(5, int(len(lines) * 0.6))
        partial = "\n".join(lines[:cutoff])
        partial += "\n// TODO: implement the remaining functions"
        return SINGLE_PROMPTS["completion"].format(partial_code=partial)

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def main():
    parser = argparse.ArgumentParser(description="Convert train.jsonl to ChatML format")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for code_gen instruction assignment (default: 42)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Input JSONL path (default: data/sft/train.jsonl)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output ChatML JSONL path (default: data/sft/gleam-instruct.jsonl)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    input_path = Path(args.input) if args.input else repo_root / "data" / "sft" / "train.jsonl"
    output_path = Path(args.output) if args.output else repo_root / "data" / "sft" / "gleam-instruct.jsonl"
    corpus_path = repo_root / "data" / "processed" / "gleam_corpus.parquet"

    random.seed(args.seed)

    # Load corpus: id → (code, repo_name, file_path, stars)
    print(f"Loading corpus from {corpus_path} ...")
    corpus = pl.read_parquet(str(corpus_path))
    code_lookup = {}
    for row in corpus.iter_rows(named=True):
        code_lookup[row["id"]] = {
            "code": row["code"],
            "repo_name": row["repo_name"],
            "file_path": row["file_path"],
            "stars": row["stars"],
        }
    print(f"  {len(code_lookup):,} source files")

    # Load train records
    print(f"Loading train records from {input_path} ...")
    train = pl.read_ndjson(str(input_path))
    print(f"  {train.height:,} records")

    # Convert to ChatML
    print("Converting to ChatML ...")
    records_out = []
    skipped = 0
    idx = 0

    for row in train.sort(["task_type", "source_file_id"]).iter_rows(named=True):
        file_id = row["source_file_id"]
        task_type = row["task_type"]

        if file_id not in code_lookup:
            print(f"  WARNING: {file_id} not in corpus, skipping", file=sys.stderr)
            skipped += 1
            continue

        source = code_lookup[file_id]

        user_content = build_user_message(
            task_type=task_type,
            code=source["code"],
            repo_name=source["repo_name"],
        )

        record = {
            "id": f"sft_gleam_{idx + 1:05d}",
            "source_file_id": file_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row["content"]},
            ],
            "task_type": task_type,
            "source_language": None,
            "model_used": row["model"],
            "license": "Apache-2.0",
            "schema_version": 1,
        }
        records_out.append(record)
        idx += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nWrote {len(records_out):,} records to {output_path} ({size_mb:.1f} MB)")
    if skipped:
        print(f"Skipped {skipped} records (source file not in corpus)")

    # Summary
    by_type = {}
    for rec in records_out:
        t = rec["task_type"]
        by_type[t] = by_type.get(t, 0) + 1
    print("\nBy task_type:")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n:,}")

    # Validate: check all messages have the right roles
    for rec in records_out:
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"], f"Bad roles: {roles} in {rec['id']}"
    print("\n✓ All records have correct message roles (system → user → assistant)")


if __name__ == "__main__":
    main()
