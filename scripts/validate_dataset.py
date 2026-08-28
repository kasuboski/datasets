#!/usr/bin/env python3
"""
Validate mixed training dataset for SFT.

Checks:
1. ChatML format compliance
2. Message sequence validity (system → user → assistant)
3. Field presence and types
4. Statistics on task types, lengths, etc.

Usage:
    uv run python scripts/validate_dataset.py --path data/sft/mixed_training.jsonl
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter
import statistics


def validate_entry(entry, idx):
    """Validate a single dataset entry. Returns (is_valid, errors)."""
    errors = []

    # Check required fields
    if "messages" not in entry:
        return False, [f"Entry {idx}: missing 'messages' field"]

    messages = entry["messages"]
    if not isinstance(messages, list):
        return False, [f"Entry {idx}: 'messages' is not a list"]

    if not messages:
        return False, [f"Entry {idx}: 'messages' is empty"]

    if len(messages) < 2:
        errors.append(f"Entry {idx}: 'messages' has fewer than 2 messages (need at least user+assistant)")

    # Check each message
    for j, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Entry {idx}, message {j}: not a dict")
            continue

        # Check required fields
        if "role" not in msg:
            errors.append(f"Entry {idx}, message {j}: missing 'role'")
        elif msg["role"] not in ["system", "user", "assistant"]:
            errors.append(f"Entry {idx}, message {j}: invalid role '{msg.get('role')}'")

        if "content" not in msg:
            errors.append(f"Entry {idx}, message {j}: missing 'content'")
        elif not isinstance(msg["content"], str):
            errors.append(f"Entry {idx}, message {j}: 'content' is not a string")

    # Check message sequence is valid
    roles = [m.get("role") for m in messages if "role" in m]

    # First message should be system or user
    if roles and roles[0] not in ("system", "user"):
        errors.append(f"Entry {idx}: first message must be 'system' or 'user', got '{roles[0]}'")

    # Check alternating pattern
    for i in range(1, len(roles)):
        prev_role = roles[i-1]
        curr_role = roles[i]

        if prev_role == "system" and curr_role != "user":
            errors.append(f"Entry {idx}: after 'system' must be 'user', got '{curr_role}'")
        elif prev_role == "user" and curr_role != "assistant":
            errors.append(f"Entry {idx}: after 'user' must be 'assistant', got '{curr_role}'")
        elif prev_role == "assistant" and curr_role not in ("user",):
            errors.append(f"Entry {idx}: after 'assistant' must be 'user', got '{curr_role}'")

    # Last message should be assistant
    if roles and roles[-1] != "assistant":
        errors.append(f"Entry {idx}: last message must be 'assistant', got '{roles[-1]}'")

    return len(errors) == 0, errors


def compute_statistics(data):
    """Compute dataset statistics."""
    stats = {
        "total_entries": len(data),
        "num_messages": [],
        "message_lengths": [],
        "roles": Counter(),
        "sources": Counter(),
        "task_types": Counter(),
        "languages": Counter(),
    }

    for entry in data:
        messages = entry.get("messages", [])

        # Number of messages
        stats["num_messages"].append(len(messages))

        # Message lengths
        for msg in messages:
            content = msg.get("content", "")
            stats["message_lengths"].append(len(content))
            stats["roles"][msg.get("role", "unknown")] += 1

        # Source
        if "source" in entry:
            stats["sources"][entry["source"]] += 1

        # Task type (if present)
        if "task_type" in entry:
            stats["task_types"][entry["task_type"]] += 1

        # Language (for functional code)
        if "language" in entry:
            stats["languages"][entry["language"]] += 1

    # Compute summary statistics
    if stats["num_messages"]:
        stats["avg_messages"] = statistics.mean(stats["num_messages"])
        stats["median_messages"] = statistics.median(stats["num_messages"])
        stats["max_messages"] = max(stats["num_messages"])
        stats["min_messages"] = min(stats["num_messages"])

    if stats["message_lengths"]:
        stats["avg_length"] = statistics.mean(stats["message_lengths"])
        stats["median_length"] = statistics.median(stats["message_lengths"])
        stats["max_length"] = max(stats["message_lengths"])
        stats["min_length"] = min(stats["message_lengths"])

    return stats


def print_statistics(stats):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    print(f"\nTotal entries: {stats['total_entries']}")

    if "avg_messages" in stats:
        print(f"\nMessages per entry:")
        print(f"  Average: {stats['avg_messages']:.1f}")
        print(f"  Median: {stats['median_messages']}")
        print(f"  Range: {stats['min_messages']} - {stats['max_messages']}")

    if "avg_length" in stats:
        print(f"\nMessage length (characters):")
        print(f"  Average: {stats['avg_length']:.0f}")
        print(f"  Median: {stats['median_length']:.0f}")
        print(f"  Range: {stats['min_length']} - {stats['max_length']}")

    print(f"\nRole distribution:")
    for role, count in stats["roles"].most_common():
        print(f"  {role}: {count}")

    if stats["sources"]:
        print(f"\nSource breakdown:")
        for source, count in stats["sources"].most_common():
            pct = count / stats['total_entries'] * 100
            print(f"  {source:<25s} {count:>6,}  ({pct:.1f}%)")

    if stats["task_types"]:
        print(f"\nTask types:")
        for task_type, count in stats["task_types"].most_common():
            print(f"  {task_type}: {count}")

    if stats["languages"]:
        print(f"\nLanguages (functional code):")
        for lang, count in stats["languages"].most_common():
            print(f"  {lang}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Validate mixed SFT dataset")
    parser.add_argument("--path", required=True, help="Path to JSONL file")
    parser.add_argument("--max-errors", type=int, default=20,
                       help="Maximum errors to display")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return 1

    print("=" * 60)
    print(f"Validating: {path}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"✅ Loaded {len(data)} entries")

    # Validate all entries
    print("\nValidating entries...")
    all_errors = ["Dataset contains no entries"] if not data else []
    valid_count = 0

    for idx, entry in enumerate(data):
        is_valid, errors = validate_entry(entry, idx)
        if is_valid:
            valid_count += 1
        else:
            all_errors.extend(errors)

    # Print results
    percentage = valid_count / len(data) * 100 if data else 0.0
    print(f"\n✅ Valid entries: {valid_count} / {len(data)} ({percentage:.1f}%)")

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} errors:")
        for error in all_errors[:args.max_errors]:
            print(f"  - {error}")
        if len(all_errors) > args.max_errors:
            print(f"  ... and {len(all_errors) - args.max_errors} more")
    else:
        print("\n✅ No errors found!")

    # Compute and print statistics
    stats = compute_statistics(data)
    print_statistics(stats)

    # Final verdict
    print("\n" + "=" * 60)
    if all_errors:
        print("❌ VALIDATION FAILED")
    else:
        print("✅ VALIDATION PASSED")
    print("=" * 60)

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
