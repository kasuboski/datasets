"""Corpus quality filters — applied post-collection to produce clean datasets.

Filters are split into two categories:
1. Per-file filters (can be applied row-by-row during streaming)
2. Cross-file filters (need the full corpus, e.g. deduplication)

Usage:
    from datasets.core.filters import filter_corpus
    df_clean = filter_corpus(df_raw)
"""

from __future__ import annotations

import polars as pl

# --- Constants ---

# Files larger than this are almost certainly auto-generated (schemas, lookup tables).
MAX_FILE_CHARS = 50_000

# Files smaller than this are likely stubs or boilerplate.
# The default `gleam new` test is ~174 chars.
MIN_FILE_CHARS = 20

# Repos whose name or owner matches these patterns are exercise/tutorial repos
# with low diversity (same problems solved repeatedly).
EXERCISM_PATTERNS = [
    "exercism",
]

# Repos on these topics tend to be auto-generated or low-quality.
# We don't filter by topic (GitHub API doesn't give us reliable topics),
# but we note them for awareness.


def filter_generated(df: pl.DataFrame) -> pl.DataFrame:
    """Remove files over MAX_FILE_CHARS (auto-generated schemas, data dumps)."""
    before = df.height
    df = df.filter(pl.col("code").str.len_chars() <= MAX_FILE_CHARS)
    removed = before - df.height
    print(f"  filter_generated: removed {removed:,} files >{MAX_FILE_CHARS:,} chars ({before:,} → {df.height:,})")
    return df


def filter_stubs(df: pl.DataFrame) -> pl.DataFrame:
    """Remove trivially small files (empty, stubs, boilerplate placeholders)."""
    before = df.height
    df = df.filter(pl.col("code").str.len_chars() >= MIN_FILE_CHARS)
    removed = before - df.height
    print(f"  filter_stubs: removed {removed:,} files <{MIN_FILE_CHARS} chars ({before:,} → {df.height:,})")
    return df


def filter_unlicensed(df: pl.DataFrame) -> pl.DataFrame:
    """Remove files without a permissive license.

    Keeps files where:
    - license is identified AND is permissive (MIT, Apache-2.0, etc.)
    - license is null but repo has a Hex package (give benefit of doubt,
      Hex packages tend to have licenses even if GitHub didn't report one)
    """
    before = df.height
    df = df.filter(
        pl.col("license").is_not_null()
        | pl.col("hex_package").is_not_null()
    )
    removed = before - df.height
    print(f"  filter_unlicensed: removed {removed:,} unlicensed files ({before:,} → {df.height:,})")
    return df


def filter_non_permissive(df: pl.DataFrame) -> pl.DataFrame:
    """Remove files with identified non-permissive licenses (GPL, AGPL)."""
    from datasets.core.license_checker import NON_PERMISSIVE
    before = df.height
    df = df.filter(
        pl.col("license").is_null()  # null = unknown, keep (benefit of doubt)
        | ~pl.col("license").is_in(list(NON_PERMISSIVE))
    )
    removed = before - df.height
    print(f"  filter_non_permissive: removed {removed:,} copyleft files ({before:,} → {df.height:,})")
    return df


def filter_exercism(df: pl.DataFrame) -> pl.DataFrame:
    """Remove exercism solution repos (low diversity, same exercises repeated)."""
    before = df.height
    # Build a combined owner/name expression for matching
    owner_match = pl.col("repo_owner").str.contains("(?i)exercism")
    name_match = pl.col("repo_name").str.contains("(?i)exercism")
    df = df.filter(~owner_match & ~name_match)
    removed = before - df.height
    print(f"  filter_exercism: removed {removed:,} exercism files ({before:,} → {df.height:,})")
    return df


def deduplicate_content(df: pl.DataFrame) -> pl.DataFrame:
    """Remove files with identical code content.

    When multiple files have the same code, keeps the one from the repo
    with the most stars (most canonical source).
    """
    before = df.height
    df = df.with_columns(
        pl.col("code").hash().alias("_content_hash")
    )
    # Sort by stars desc so the first occurrence is from the most-starred repo
    df = df.sort("stars", descending=True)
    df = df.unique(subset=["_content_hash"], keep="first")
    df = df.drop("_content_hash")
    removed = before - df.height
    print(f"  deduplicate_content: removed {removed:,} duplicate files ({before:,} → {df.height:,})")
    return df


def filter_corpus(
    df: pl.DataFrame,
    *,
    remove_generated: bool = True,
    remove_stubs: bool = True,
    remove_unlicensed: bool = True,
    remove_non_permissive: bool = True,
    remove_exercism: bool = True,
    deduplicate: bool = True,
) -> pl.DataFrame:
    """Apply the full filter pipeline to a corpus DataFrame.

    Returns the filtered DataFrame. Prints stats for each step.
    """
    print("=" * 60)
    print("FILTERING CORPUS")
    print("=" * 60)
    print(f"  Input: {df.height:,} files")
    print()

    if remove_generated:
        df = filter_generated(df)
    if remove_stubs:
        df = filter_stubs(df)
    if remove_unlicensed:
        df = filter_unlicensed(df)
    if remove_non_permissive:
        df = filter_non_permissive(df)
    if remove_exercism:
        df = filter_exercism(df)
    if deduplicate:
        df = deduplicate_content(df)

    print()
    total_chars = df.select(pl.col("code").str.len_chars().sum()).item()
    total_lines = df.select((pl.col("code").str.count_matches("\n") + 1).sum()).item()
    total_repos = df.select(pl.col("repo_url").n_unique()).item()
    print(f"  Output: {df.height:,} files from {total_repos:,} repos")
    print(f"          {total_lines:,} lines, {total_chars:,} characters")
    print("=" * 60)
    return df
