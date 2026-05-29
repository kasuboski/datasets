"""Build orchestrator: registry → clone → extract → store as JSONL.

This is the main entry point for Phase 1 corpus collection.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.core.git_tools import process_repo
from src.core.license_checker import is_permissive
from src.gleam.registry import (
    RepoInfo,
    discover_github_repos,
    discover_hex_packages,
    enrich_repos_with_hex,
)

# Default paths (relative to project root)
DEFAULT_CORPUS_PATH = Path("data/raw/gleam_corpus.jsonl")
DEFAULT_STATE_PATH = Path("data/raw/collection_state.json")


def _load_state(state_path: Path) -> dict:
    """Load collection state (tracks which repos have been processed)."""
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"processed_repos": {}, "stats": {}}


def _save_state(state: dict, state_path: Path) -> None:
    """Save collection state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def _append_records(records: list[dict], corpus_path: Path) -> None:
    """Append records to the JSONL corpus file."""
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_corpus(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    max_repos: int | None = None,
    skip_non_permissive: bool = True,
    skip_no_gleam_files: bool = True,
    clone_timeout: int = 120,
    resume: bool = True,
    dry_run: bool = False,
) -> dict:
    """Run the full corpus collection pipeline.

    Args:
        corpus_path: Output JSONL file path.
        state_path: Collection state file for resumability.
        max_repos: Limit number of repos (for testing). None = all.
        skip_non_permissive: Skip repos with non-permissive licenses.
        skip_no_gleam_files: Skip repos that yielded 0 .gleam files.
        clone_timeout: Seconds before clone is killed.
        resume: Skip repos already in state file.
        dry_run: Discover repos but don't clone/extract.

    Returns:
        Stats dict with counts.
    """
    stats = {
        "repos_discovered": 0,
        "repos_skipped_license": 0,
        "repos_skipped_already_processed": 0,
        "repos_processed": 0,
        "repos_clone_failed": 0,
        "repos_no_gleam_files": 0,
        "files_collected": 0,
        "files_skipped_non_permissive": 0,
    }

    # --- Step 1: Discover repos ---
    print("=" * 60)
    print("PHASE 1: GLEAM CODE CORPUS COLLECTION")
    print("=" * 60)

    print("\n[1/4] Discovering GitHub repos (language:gleam)...")
    repos = discover_github_repos(max_repos=max_repos)
    stats["repos_discovered"] = len(repos)
    print(f"  Found {len(repos)} repos")

    # --- Step 2: Hex.pm enrichment ---
    print("\n[2/4] Fetching Hex.pm metadata for enrichment...")
    hex_packages = discover_hex_packages()
    print(f"  Found {len(hex_packages)} Hex packages matching 'gleam'")

    print("  Cross-referencing...")
    repos = enrich_repos_with_hex(repos, hex_packages)
    enriched = sum(1 for r in repos if r.hex_package)
    print(f"  {enriched} repos matched to Hex packages")

    # --- Step 3: Filter ---
    if skip_non_permissive:
        before = len(repos)
        # Keep repos where license is permissive OR unknown (give benefit of doubt)
        repos = [
            r for r in repos
            if r.license_raw is None or is_permissive(r.license_raw)
        ]
        stats["repos_skipped_license"] = before - len(repos)
        print(f"\n  Filtered: {before - len(repos)} repos with non-permissive licenses removed")

    # Pre-clone filter: skip repos with no license AND no Hex package.
    # These repos contributed 42K of our 69K files in the first run, all discarded post-collection.
    # Skipping them saves ~60% of clone bandwidth/time on re-runs.
    before = len(repos)
    repos = [
        r for r in repos
        if r.license_raw is not None or r.hex_package is not None
    ]
    skipped_unlicensed = before - len(repos)
    if skipped_unlicensed:
        print(f"  Pre-clone filter: skipped {skipped_unlicensed} repos with no license and no Hex package")
        stats["repos_skipped_unlicensed"] = skipped_unlicensed

    if dry_run:
        print(f"\n[DRY RUN] Would process {len(repos)} repos. Stopping.")
        stats["repos_to_process"] = len(repos)
        return stats

    # --- Step 4: Clone, extract, store ---
    print(f"\n[3/4] Processing {len(repos)} repos...")
    state = _load_state(state_path) if resume else {"processed_repos": {}, "stats": {}}

    for i, repo in enumerate(repos):
        repo_key = f"{repo.owner}/{repo.name}"

        # Resume: skip already-processed repos
        if resume and repo_key in state["processed_repos"]:
            stats["repos_skipped_already_processed"] += 1
            continue

        print(f"  [{i+1}/{len(repos)}] {repo_key} (⭐{repo.stars})", end="")

        result = process_repo(
            repo_url=repo.url,
            owner=repo.owner,
            name=repo.name,
            repo_description=repo.description,
            stars=repo.stars,
            license_raw=repo.license_raw,
            authors=repo.authors,
            hex_package=repo.hex_package,
            hex_downloads=repo.hex_downloads,
            fork=repo.fork,
            archived=repo.archived,
            clone_timeout=clone_timeout,
        )

        if result.error:
            stats["repos_clone_failed"] += 1
            print(f" → ERROR: {result.error}")
            state["processed_repos"][repo_key] = {
                "status": "error",
                "error": result.error,
                "timestamp": time.time(),
            }
            _save_state(state, state_path)
            continue

        if not result.files:
            stats["repos_no_gleam_files"] += 1
            print(f" → no .gleam files")
            state["processed_repos"][repo_key] = {
                "status": "no_gleam_files",
                "timestamp": time.time(),
            }
            _save_state(state, state_path)
            continue

        # Write records
        _append_records(result.files, corpus_path)
        stats["repos_processed"] += 1
        stats["files_collected"] += len(result.files)
        print(f" → {len(result.files)} .gleam files (SHA: {result.commit_sha[:8] if result.commit_sha else 'unknown'})")

        # Update state
        state["processed_repos"][repo_key] = {
            "status": "collected",
            "files": len(result.files),
            "commit_sha": result.commit_sha,
            "timestamp": time.time(),
        }
        state["stats"] = stats
        _save_state(state, state_path)

    # --- Summary ---
    print(f"\n[4/4] Collection complete!")
    print(f"  Repos discovered:     {stats['repos_discovered']}")
    print(f"  Repos processed:      {stats['repos_processed']}")
    print(f"  Repos already done:   {stats['repos_skipped_already_processed']}")
    print(f"  Repos clone failed:   {stats['repos_clone_failed']}")
    print(f"  Repos no .gleam:      {stats['repos_no_gleam_files']}")
    print(f"  Repos skipped license:{stats['repos_skipped_license']}")
    print(f"  Files collected:      {stats['files_collected']}")
    print(f"\n  Output: {corpus_path}")

    state["stats"] = stats
    _save_state(state, state_path)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Gleam code corpus")
    parser.add_argument("--max-repos", type=int, default=None, help="Limit repos (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Discover only, don't clone")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore previous state")
    parser.add_argument("--clone-timeout", type=int, default=120, help="Clone timeout in seconds")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    args = parser.parse_args()

    build_corpus(
        corpus_path=args.corpus_path,
        state_path=args.state_path,
        max_repos=args.max_repos,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        clone_timeout=args.clone_timeout,
    )
