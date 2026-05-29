"""CLI entry point for the Gleam datasets pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Gleam datasets pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect    Run Phase 1: collect Gleam code corpus
  convert    Convert JSONL corpus to Parquet and optionally push to HF
  validate   Validate an existing JSONL corpus
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- collect ---
    p_collect = subparsers.add_parser("collect", help="Collect Gleam code corpus from GitHub + Hex.pm")
    p_collect.add_argument("--max-repos", type=int, default=None, help="Limit repos (for testing)")
    p_collect.add_argument("--dry-run", action="store_true", help="Discover only, don't clone")
    p_collect.add_argument("--no-resume", action="store_true", help="Start fresh")
    p_collect.add_argument("--clone-timeout", type=int, default=120)
    p_collect.add_argument("--corpus-path", type=Path, default=Path("data/raw/gleam_corpus.jsonl"))
    p_collect.add_argument("--state-path", type=Path, default=Path("data/raw/collection_state.json"))

    # --- convert ---
    p_convert = subparsers.add_parser("convert", help="Convert JSONL → Parquet, optionally push to HF")
    p_convert.add_argument("--jsonl-path", type=Path, default=Path("data/raw/gleam_corpus.jsonl"))
    p_convert.add_argument("--parquet-dir", type=Path, default=Path("data/processed"))
    p_convert.add_argument("--push", action="store_true", help="Push to Hugging Face")
    p_convert.add_argument("--dataset-id", type=str, default=None, help="HF dataset ID (e.g. user/gleam-code-corpus)")
    p_convert.add_argument("--skip-filters", action="store_true", help="Skip quality filters (keep raw corpus)")

    # --- validate ---
    p_validate = subparsers.add_parser("validate", help="Validate a JSONL corpus file")
    p_validate.add_argument("--corpus-path", type=Path, default=Path("data/raw/gleam_corpus.jsonl"))

    # --- push ---
    p_push = subparsers.add_parser("push", help="Push already-processed files to Hugging Face")
    p_push.add_argument("--parquet-dir", type=Path, default=Path("data/processed"))
    p_push.add_argument("--dataset-id", type=str, default="kasuboski/gleam-code-corpus", help="HF dataset ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "collect":
        from src.gleam.build import build_corpus
        build_corpus(
            corpus_path=args.corpus_path,
            state_path=args.state_path,
            max_repos=args.max_repos,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            clone_timeout=args.clone_timeout,
        )

    elif args.command == "convert":
        from src.core.hf_hub import run_conversion
        run_conversion(
            jsonl_path=args.jsonl_path,
            parquet_dir=args.parquet_dir,
            dataset_id=args.dataset_id,
            push=args.push,
            skip_filters=args.skip_filters,
        )

    elif args.command == "validate":
        from src.core.hf_hub import validate_corpus
        import polars as pl
        if not args.corpus_path.exists():
            print(f"ERROR: {args.corpus_path} not found")
            sys.exit(1)
        print(f"Loading {args.corpus_path}...")
        df = pl.read_ndjson(args.corpus_path)
        print(f"Loaded {len(df):,} records")
        warnings = validate_corpus(df)
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  ⚠ {w}")
        else:
            print("✅ No warnings")

    elif args.command == "push":
        from src.core.hf_hub import push_only
        push_only(
            parquet_dir=args.parquet_dir,
            dataset_id=args.dataset_id,
        )


if __name__ == "__main__":
    main()
