#!/usr/bin/env python3
"""
Create mixed training dataset for Gleam SFT fine-tuning.

Combines the validated JSONL inputs:
- gleam-instruct.jsonl (9,355 pairs, ChatML) — domain data, 69.1%
- functional_683_chatml.jsonl (683 pairs) — functional code, 5.0%
- ultrachat_3k.jsonl (3,000 pairs) — general instruction following, 22.2%
- opencode_500_chatml.jsonl (500 pairs) — general code, 3.7%

The validated output contains 13,538 records; proportions reflect the available inputs (69.1/5.0/22.2/3.7).
The script normalises all entries to {messages: [...], source: str} format.

Output: data/sft/mixed_training.jsonl (ChatML format, shuffled)

Usage:
    uv run python scripts/create_mixed_dataset.py
    uv run python scripts/create_mixed_dataset.py --output data/sft/mixed_v2.jsonl
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter

random.seed(42)

DEFAULT_GLEAM = Path("data/sft/gleam-instruct.jsonl")
DEFAULT_FUNCTIONAL = Path("data/sft/functional_683_chatml.jsonl")
DEFAULT_ULTRACHAT = Path("data/sft/ultrachat_3k.jsonl")
DEFAULT_OPENCODE = Path("data/sft/opencode_500_chatml.jsonl")


def load_jsonl(path: Path, label: str) -> list[dict]:
    """Load a JSONL file and normalise to {messages, source} format."""
    if not path.exists():
        print(f"  ⚠️  Not found: {path} — skipping {label}")
        return []

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Normalise: ensure every record has a top-level 'messages' list
    normalised = []
    for rec in records:
        if "messages" in rec:
            normalised.append(rec)
        else:
            print(f"  ⚠️  Skipping record without 'messages' in {label}")
            continue

    print(f"  ✅ {label}: {len(normalised):,} records from {path.name}")
    return normalised


def validate_messages(messages: list[dict]) -> bool:
    """Quick sanity check on a messages list."""
    if not messages:
        return False
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ("system", "user", "assistant"):
            return False
    if messages[-1]["role"] != "assistant":
        return False
    return True


def compute_stats(data: list[dict]) -> dict:
    """Compute summary statistics on a mixed dataset."""
    source_counts = Counter()
    total_msgs = 0
    content_lengths = []
    roles = Counter()

    for rec in data:
        source = rec.get("source", "unknown")
        source_counts[source] += 1
        for msg in rec["messages"]:
            total_msgs += 1
            content_lengths.append(len(msg.get("content", "")))
            roles[msg.get("role", "unknown")] += 1

    return {
        "total_records": len(data),
        "total_messages": total_msgs,
        "source_counts": source_counts,
        "role_counts": roles,
        "avg_content_len": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
        "max_content_len": max(content_lengths) if content_lengths else 0,
        "min_content_len": min(content_lengths) if content_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Create mixed SFT training dataset")
    parser.add_argument("--gleam", type=Path, default=DEFAULT_GLEAM, help="Gleam instruct JSONL")
    parser.add_argument("--functional", type=Path, default=DEFAULT_FUNCTIONAL, help="Functional code JSONL")
    parser.add_argument("--ultrachat", type=Path, default=DEFAULT_ULTRACHAT, help="Ultrachat sample JSONL")
    parser.add_argument("--opencode", type=Path, default=DEFAULT_OPENCODE, help="OpenCodeInstruct sample JSONL")
    parser.add_argument("--output", type=Path, default=Path("data/sft/mixed_training.jsonl"), help="Output path")
    args = parser.parse_args()

    print("=" * 60)
    print("Creating Mixed Training Dataset")
    print("=" * 60)

    # Load all sources
    print("\nLoading datasets...")
    gleam = load_jsonl(args.gleam, "gleam-instruct")
    functional = load_jsonl(args.functional, "functional-code")
    ultrachat = load_jsonl(args.ultrachat, "ultrachat")
    opencode = load_jsonl(args.opencode, "opencode")

    # Tag source for each record (keep existing source if present, otherwise label)
    for rec in gleam:
        rec.setdefault("source", "gleam-instruct")
    for rec in functional:
        rec.setdefault("source", "functional-code")
    for rec in ultrachat:
        rec.setdefault("source", "ultrachat")
    for rec in opencode:
        rec.setdefault("source", "opencode")

    # Filter out explanation task type from gleam-instruct
    # Explanations are very long (P50~4500 tokens, barely fit 4096) and teach
    # code narration rather than code writing. Completion + code_gen are more
    # useful and fit much better in shorter seq lengths.
    exclude_task_types = {"explanation"}
    before = len(gleam)
    gleam = [rec for rec in gleam if rec.get("task_type") not in exclude_task_types]
    filtered = before - len(gleam)
    if filtered:
        print(f"  🗑️  Filtered {filtered} 'explanation' records from gleam-instruct")
        print(f"     Kept: {len(gleam):,} (completion + code_gen only)")

    # Combine and shuffle
    mixed = gleam + functional + ultrachat + opencode
    print(f"\n  Combined: {len(mixed):,} records (before validation)")

    # Validate
    valid = []
    skipped = 0
    for rec in mixed:
        if validate_messages(rec["messages"]):
            valid.append(rec)
        else:
            skipped += 1

    if skipped:
        print(f"  ⚠️  Skipped {skipped} invalid records")

    # Shuffle deterministically
    random.shuffle(valid)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in valid:
            # Strip extra fields for clean output — keep messages + source
            clean = {"messages": rec["messages"]}
            if "source" in rec:
                clean["source"] = rec["source"]
            # Preserve gleam task_type and functional language if present
            if "task_type" in rec:
                clean["task_type"] = rec["task_type"]
            if "language" in rec:
                clean["language"] = rec["language"]
            f.write(json.dumps(clean) + "\n")

    # Stats
    stats = compute_stats(valid)
    print(f"\n{'=' * 60}")
    print("Mixed Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Total records:   {stats['total_records']:,}")
    print(f"  Total messages:  {stats['total_messages']:,}")
    print(f"  Avg content len: {stats['avg_content_len']:.0f} chars")
    print(f"  Max content len: {stats['max_content_len']:,} chars")
    print(f"\n  Source breakdown:")
    for source, count in stats["source_counts"].most_common():
        pct = count / stats["total_records"] * 100
        print(f"    {source:<25s} {count:>6,}  ({pct:.1f}%)")
    print(f"\n  Role distribution:")
    for role, count in stats["role_counts"].most_common():
        print(f"    {role:<15s} {count:>6,}")
    print(f"\n  ✅ Saved to {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
