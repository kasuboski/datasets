#!/usr/bin/env python3
"""Clean generation.jsonl into a deduplicated, filtered training dataset.

Pipeline:
  1. Dedupe: keep latest record per (source_file_id, task_type) by timestamp
  2. Filter: remove truncated, failed, and degenerate repetitive content
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

    # Step 2a: Filter — only clean records
    clean = deduped.filter(
        (pl.col("finish_reason") == "stop")
        & pl.col("content").is_not_null()
        & (pl.col("content").str.len_chars() > 0)
    )
    truncated = deduped.filter(pl.col("finish_reason") == "length").height
    failed = deduped.filter(pl.col("finish_reason").is_null()).height
    empty = deduped.filter(
        pl.col("finish_reason") == "stop",
        pl.col("content").is_not_null(),
        (pl.col("content").str.len_chars() == 0),
    ).height
    print(f"  After basic filter: {clean.height:,} (removed {deduped.height - clean.height:,}: {truncated} truncated, {failed} failed, {empty} empty)")

    # Step 2b: Filter degenerate repetitive content
    # Unique line ratio < 0.3 means >70% of lines are duplicates — model looping
    clean = clean.with_columns(
        pl.col("content").map_elements(
            lambda text: len(set(text.strip().split("\n"))) / max(len(text.strip().split("\n")), 1),
            return_dtype=pl.Float64,
        ).alias("unique_line_ratio")
    )
    degenerate = clean.filter(pl.col("unique_line_ratio") < 0.3)
    if degenerate.height > 0:
        print(f"  Degenerate (unique_line_ratio < 0.3): {degenerate.height}")
        for row in degenerate.select(["source_file_id", "task_type", "unique_line_ratio"]).iter_rows(named=True):
            print(f"    {row['source_file_id'][:60]:60s} | {row['task_type']:12s} | ratio={row['unique_line_ratio']:.3f}")
    clean = clean.filter(pl.col("unique_line_ratio") >= 0.3).drop("unique_line_ratio")
    print(f"  After degenerate filter: {clean.height:,}")

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
