"""Convert JSONL corpus to Parquet and push to Hugging Face Hub.

Handles:
- JSONL → Parquet conversion (with Polars)
- Splitting into HF config subsets (all, official, community_top)
- Dataset card generation
- Upload to HF
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

SCHEMA_VERSION = 1


def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> pl.DataFrame:
    """Convert JSONL corpus to a Parquet file.

    Returns the loaded DataFrame for further processing.
    """
    df = pl.read_ndjson(jsonl_path)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path, compression="zstd")
    return df


def split_configs(
    df: pl.DataFrame,
    output_dir: Path,
    community_top_stars: int = 10,
    community_top_downloads: int = 1000,
) -> dict[str, Path]:
    """Split the corpus into HF config subsets.

    Returns a dict of config_name → parquet_path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    configs: dict[str, Path] = {}

    # All permissive-license records
    all_path = output_dir / "all.parquet"
    df_all = df.filter(pl.col("license").is_not_null())
    df_all.write_parquet(all_path, compression="zstd")
    configs["all"] = all_path

    # Official (is_official == true)
    official_path = output_dir / "official.parquet"
    df_official = df.filter(pl.col("is_official") == True)  # noqa: E712
    df_official.write_parquet(official_path, compression="zstd")
    configs["official"] = official_path

    # Community top (high stars or high downloads, excluding official)
    community_path = output_dir / "community_top.parquet"
    df_community = df.filter(
        (pl.col("is_official") == False)  # noqa: E712
        & (
            (pl.col("stars") >= community_top_stars)
            | (pl.col("hex_downloads") >= community_top_downloads)
        )
    )
    df_community.write_parquet(community_path, compression="zstd")
    configs["community_top"] = community_path

    return configs


def generate_dataset_card(df: pl.DataFrame) -> str:
    """Generate a README dataset card for the HF dataset."""
    total_repos = df.select(pl.col("repo_url").n_unique()).item()
    total_files = len(df)
    total_lines = df.select(
        pl.col("code").str.count_matches("\n").sum()
    ).item() + total_files
    total_chars = df.select(pl.col("code").str.len_chars().sum()).item()

    license_counts = (
        df.filter(pl.col("license").is_not_null())
        .group_by("license")
        .len()
        .sort("len", descending=True)
    )

    official_count = df.filter(pl.col("is_official") == True).height  # noqa: E712
    community_count = total_files - official_count

    card = f"""# Gleam Code Corpus

A structured, attributed corpus of Gleam programming language source code collected from GitHub and enriched with Hex.pm metadata.

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total `.gleam` files | {total_files:,} |
| Unique repositories | {total_repos:,} |
| Total lines of code | {total_lines:,} |
| Total characters | {total_chars:,} |
| Official files | {official_count:,} |
| Community files | {community_count:,} |

## Schema

Each record represents a single `.gleam` source file:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Deterministic ID: `{{owner}}_{{repo}}_{{path}}` |
| `code` | string | Raw file contents |
| `file_path` | string | Path within the repository |
| `repo_url` | string | GitHub repository URL |
| `repo_owner` | string | Repository owner/org |
| `repo_name` | string | Repository name |
| `repo_description` | string | Repository description |
| `stars` | int | GitHub stars |
| `is_official` | bool | From an official Gleam org |
| `license` | string | Normalized SPDX identifier |
| `license_raw` | string | Original license string from source |
| `authors` | list[str] | Repository/Hex owners |
| `hex_package` | string | Hex.pm package name (if published) |
| `hex_downloads` | int | Total Hex.pm downloads |
| `fork` | bool | Is a fork |
| `archived` | bool | Repository is archived |
| `collected_at` | string | ISO 8601 timestamp |
| `commit_sha` | string | Git HEAD at collection time |
| `schema_version` | int | Schema version (currently {SCHEMA_VERSION}) |

## Configs

- **`all`**: All files with identified permissive licenses
- **`official`**: Files from official Gleam organizations (gleam-lang, lustre-labs, gleam-wisp)
- **`community_top`**: Community packages with ≥10 stars or ≥1,000 Hex downloads

## License Distribution

| License | Files |
|---------|-------|
"""
    for row in license_counts.iter_rows(named=True):
        card += f"| {row['license']} | {row['len']:,} |\n"

    card += f"""
## Intended Use

This dataset is designed for:
- **Continued pre-training** of language models on Gleam code
- **Instruction fine-tuning** (after conversion to SFT pairs)
- **Code analysis** and ecosystem research

## Collection Method

1. GitHub search API: `language:gleam` (sorted by stars)
2. Hex.pm search API: `search=gleam` for metadata enrichment
3. Shallow clone of each repository
4. Extraction of all `.gleam` files with attribution

## License

Each file retains its original repository license (see `license` field).
This dataset is provided for research purposes. Individual files are
subject to their respective licenses.
"""
    return card


def validate_corpus(df: pl.DataFrame) -> list[str]:
    """Run validation checks on the corpus. Returns list of warnings."""
    warnings = []

    # Check for duplicate IDs
    dup_count = df.filter(pl.col("id").is_duplicated()).height
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate IDs found")

    # Check for empty code
    empty_count = df.filter(pl.col("code").str.len_chars() == 0).height
    if empty_count > 0:
        warnings.append(f"{empty_count} records with empty code")

    # Check for missing licenses
    no_license = df.filter(pl.col("license").is_null()).height
    if no_license > 0:
        warnings.append(f"{no_license} records without license info")

    # Check for very large files (might be generated)
    large = df.filter(pl.col("code").str.len_chars() > 100_000).height
    if large > 0:
        warnings.append(f"{large} files over 100KB (possibly generated)")

    return warnings


def push_to_hf(
    parquet_dir: Path,
    dataset_id: str,
    dataset_card: str,
    token: str | None = None,
) -> None:
    """Push the processed corpus to Hugging Face Hub.

    Args:
        parquet_dir: Directory containing config parquet files.
        dataset_id: HF dataset ID (e.g., "username/gleam-code-corpus").
        dataset_card: README content.
        token: HF API token (uses env var HF_TOKEN if not provided).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = dataset_id

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Upload dataset card
    card_path = parquet_dir / "README.md"
    card_path.write_text(dataset_card)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Upload each config's parquet file
    for config_name in ["all", "official", "community_top"]:
        parquet_file = parquet_dir / f"{config_name}.parquet"
        if parquet_file.exists():
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=f"{config_name}/data.parquet",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  Uploaded {config_name}/data.parquet")

    print(f"  Dataset pushed to https://huggingface.co/datasets/{repo_id}")


def run_conversion(
    jsonl_path: Path = Path("data/raw/gleam_corpus.jsonl"),
    parquet_dir: Path = Path("data/processed"),
    dataset_id: str | None = None,
    push: bool = False,
    token: str | None = None,
    skip_filters: bool = False,
) -> None:
    """Full conversion pipeline: JSONL → filter → validate → split → push."""
    print("=" * 60)
    print("CONVERTING CORPUS: JSONL → PARQUET")
    print("=" * 60)

    if not jsonl_path.exists():
        print(f"  ERROR: {jsonl_path} not found")
        return

    # Load
    print(f"\n[1/5] Loading {jsonl_path}...")
    df = pl.read_ndjson(jsonl_path)
    print(f"  Loaded {len(df):,} records")

    # Save raw (unfiltered) Parquet
    parquet_dir.mkdir(parents=True, exist_ok=True)
    raw_path = parquet_dir / "gleam_corpus_raw.parquet"
    df.write_parquet(raw_path, compression="zstd")
    print(f"  Raw corpus saved to {raw_path}")

    # Filter
    if skip_filters:
        print(f"\n[2/5] Skipping filters (--skip-filters)")
    else:
        print(f"\n[2/5] Filtering...")
        from src.core.filters import filter_corpus
        df = filter_corpus(df)

    # Validate
    print(f"\n[3/5] Validating...")
    warnings = validate_corpus(df)
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    ⚠ {w}")
    else:
        print("  ✅ No warnings")

    # Save filtered Parquet
    filtered_path = parquet_dir / "gleam_corpus.parquet"
    df.write_parquet(filtered_path, compression="zstd")
    print(f"  Filtered corpus saved to {filtered_path}")

    # Split into configs
    print(f"\n[4/5] Splitting into configs...")
    configs = split_configs(df, parquet_dir)
    for name, path in configs.items():
        config_df = pl.read_parquet(path)
        print(f"  {name}: {len(config_df):,} records → {path}")

    # Generate dataset card
    card = generate_dataset_card(df)
    card_path = parquet_dir / "README.md"
    card_path.write_text(card)
    print(f"\n  Dataset card written to {card_path}")

    # Push to HF
    if push and dataset_id:
        print(f"\n[4/4] Pushing to HF as {dataset_id}...")
        push_to_hf(parquet_dir, dataset_id, card, token=token)
    else:
        print(f"\n[4/4] Skipping HF push (use --push --dataset-id <id> to push)")

    print("\nDone!")


def push_only(
    parquet_dir: Path = Path("data/processed"),
    dataset_id: str = "kasuboski/gleam-code-corpus",
    token: str | None = None,
) -> None:
    """Push already-processed files to Hugging Face without re-running conversion."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Read dataset card
    card_path = parquet_dir / "README.md"
    if not card_path.exists():
        print(f"  ERROR: {card_path} not found. Run `convert` first.")
        return

    card = card_path.read_text()

    # Verify at least one config exists
    configs_found = []
    for config_name in ["all", "official", "community_top"]:
        if (parquet_dir / f"{config_name}.parquet").exists():
            configs_found.append(config_name)

    if not configs_found:
        print(f"  ERROR: No config parquet files found in {parquet_dir}. Run `convert` first.")
        return

    print(f"  Found configs: {configs_found}")
    print(f"  Dataset card: {card_path}")

    # Create repo and upload
    api.create_repo(repo_id=dataset_id, repo_type="dataset", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=dataset_id,
        repo_type="dataset",
    )
    print(f"  Uploaded README.md")

    for config_name in configs_found:
        parquet_file = parquet_dir / f"{config_name}.parquet"
        api.upload_file(
            path_or_fileobj=str(parquet_file),
            path_in_repo=f"{config_name}/data.parquet",
            repo_id=dataset_id,
            repo_type="dataset",
        )
        size_mb = parquet_file.stat().st_size / (1024 * 1024)
        print(f"  Uploaded {config_name}/data.parquet ({size_mb:.1f} MB)")

    print(f"\n  Dataset pushed to https://huggingface.co/datasets/{dataset_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert and publish Gleam corpus")
    parser.add_argument("--jsonl-path", type=Path, default=Path("data/raw/gleam_corpus.jsonl"))
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face")
    parser.add_argument("--dataset-id", type=str, default=None, help="HF dataset ID")
    args = parser.parse_args()

    run_conversion(
        jsonl_path=args.jsonl_path,
        parquet_dir=args.parquet_dir,
        dataset_id=args.dataset_id,
        push=args.push,
    )
