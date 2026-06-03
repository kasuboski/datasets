#!/usr/bin/env python3
"""Clean generation.jsonl into a deduplicated, filtered training dataset.

Pipeline:
  1. Dedupe: keep latest record per (source_file_id, task_type) by timestamp
  2. Filter: remove truncated (finish_reason=length) and failed (null finish_reason/content)
  3. Select training-relevant columns
  4. Write to train.jsonl
"""

import sys
from pathlib import Path

import polars as pl


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sft"
    src = data_dir / "generation.jsonl"
    dst = data_dir / "train.jsonl"

    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {src} ...")
    df = pl.read_ndjson(str(src))
    print(f"  Raw records: {df.height:,}")

    # Step 1: Dedupe — keep latest per (source_file_id, task_type)
    deduped = (
        df.sort("timestamp")
        .group_by(["source_file_id", "task_type"])
        .last()
    )
    print(f"  After dedupe: {deduped.height:,} (removed {df.height - deduped.height:,} older duplicates)")

    # Step 2: Filter — only clean records
    clean = deduped.filter(
        (pl.col("finish_reason") == "stop")
        & pl.col("content").is_not_null()
        & (pl.col("content").str.len_chars() > 0)
    )
    removed = deduped.height - clean.height
    print(f"  After filter: {clean.height:,} (removed {removed:,} bad records)")

    # Breakdown of removed
    truncated = deduped.filter(pl.col("finish_reason") == "length").height
    failed = deduped.filter(pl.col("finish_reason").is_null()).height
    empty = deduped.filter(
        pl.col("finish_reason") == "stop",
        pl.col("content").is_not_null(),
        (pl.col("content").str.len_chars() == 0),
    ).height
    print(f"    Truncated: {truncated}, Failed/null: {failed}, Empty content: {empty}")

    # Step 3: Select training-relevant columns
    training_cols = [
        "task_type",
        "source_file_id",
        "source_file_path",
        "source_repo",
        "source_stars",
        "content",
        "input_tokens",
        "output_tokens",
        "provider",
        "model",
    ]
    out = clean.select(training_cols).sort(["task_type", "source_file_id"])

    # Step 4: Write
    out.write_ndjson(str(dst))
    size_mb = dst.stat().st_size / 1e6
    print(f"\nWrote {out.height:,} records to {dst} ({size_mb:.1f} MB)")

    # Summary
    print("\nBy task_type:")
    for task in ["code_gen", "explanation", "completion"]:
        n = out.filter(pl.col("task_type") == task).height
        print(f"  {task}: {n:,}")


if __name__ == "__main__":
    main()
