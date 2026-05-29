"""Stratified file sampling from the Gleam code corpus.

Picks N files across diversity dimensions:
  - Complexity: simple (<100 LOC), medium (100-500), complex (500+)
  - Source path: src/, test/, examples/
  - Quality tier: official (gleam-lang/*), community_top (stars>=10), community

Excludes monks_of_style (semi-generated).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

# Reproducible sampling
SEED = 42


@dataclass
class SampledFile:
    """A single file sampled from the corpus."""
    id: str
    code: str
    file_path: str
    repo_name: str
    repo_owner: str
    stars: int
    is_official: bool
    license: str
    loc: int  # lines of code


def _classify_complexity(loc: int) -> str:
    if loc < 100:
        return "simple"
    elif loc < 500:
        return "medium"
    else:
        return "complex"


def _classify_source(file_path: str) -> str:
    prefix = file_path.split("/")[0]
    if prefix == "src":
        return "src"
    elif prefix == "test":
        return "test"
    elif prefix == "examples":
        return "examples"
    else:
        return "other"


def _classify_tier(is_official: bool, stars: int) -> str:
    if is_official:
        return "official"
    elif stars >= 10:
        return "community_top"
    else:
        return "community"


def sample_files(
    corpus_path: str = "data/processed/gleam_corpus.parquet",
    n: int = 20,
    seed: int = SEED,
) -> list[SampledFile]:
    """Sample N files with diversity across complexity, source, and tier."""
    rng = random.Random(seed)

    df = pl.read_parquet(corpus_path)

    # Exclude monks_of_style
    df = df.filter(pl.col("repo_name") != "monks_of_style")

    # Add computed columns
    df = df.with_columns([
        pl.col("code").str.count_matches("\n").add(1).alias("loc"),
    ])

    # Build classification columns
    records = df.to_dicts()

    classified = []
    for r in records:
        loc = r["loc"]
        fp = r["file_path"]
        classified.append({
            "id": r["id"],
            "code": r["code"],
            "file_path": fp,
            "repo_name": r["repo_name"],
            "repo_owner": r["repo_owner"],
            "stars": r["stars"],
            "is_official": r["is_official"],
            "license": r["license"],
            "loc": loc,
            "complexity": _classify_complexity(loc),
            "source": _classify_source(fp),
            "tier": _classify_tier(r["is_official"], r["stars"]),
        })

    # Target distribution (balanced across dimensions)
    # We want coverage, not strict proportions
    target_complexity = {"simple": 7, "medium": 7, "complex": 6}
    target_source = {"src": 10, "test": 6, "examples": 4}
    target_tier = {"official": 4, "community_top": 8, "community": 8}

    sampled_ids = set()
    samples = []

    def _pick_from_pool(pool: list[dict], count: int) -> list[dict]:
        available = [r for r in pool if r["id"] not in sampled_ids]
        if not available:
            return []
        count = min(count, len(available))
        chosen = rng.sample(available, count)
        return chosen

    # Phase 1: Pick officials first (limited pool — only 296 total)
    officials = [r for r in classified if r["tier"] == "official"]
    chosen = _pick_from_pool(officials, target_tier["official"])
    for c in chosen:
        sampled_ids.add(c["id"])
        samples.append(SampledFile(**{k: c[k] for k in [
            "id", "code", "file_path", "repo_name", "repo_owner",
            "stars", "is_official", "license", "loc",
        ]}))

    # Phase 2: Pick by complexity bucket
    for comp, count in target_complexity.items():
        remaining = count - sum(
            1 for s in samples
            if _classify_complexity(s.loc) == comp
        )
        if remaining <= 0:
            continue
        pool = [r for r in classified if r["complexity"] == comp]
        chosen = _pick_from_pool(pool, remaining)
        for c in chosen:
            sampled_ids.add(c["id"])
            samples.append(SampledFile(**{k: c[k] for k in [
                "id", "code", "file_path", "repo_name", "repo_owner",
                "stars", "is_official", "license", "loc",
            ]}))

    # Phase 3: Fill remaining from source buckets
    remaining = n - len(samples)
    if remaining > 0:
        # Prioritize underrepresented sources
        for src, count in target_source.items():
            current = sum(
                1 for s in samples
                if _classify_source(s.file_path) == src
            )
            need = max(0, count - current)
            if need > 0:
                pool = [r for r in classified if r["source"] == src]
                chosen = _pick_from_pool(pool, need)
                for c in chosen:
                    sampled_ids.add(c["id"])
                    samples.append(SampledFile(**{k: c[k] for k in [
                        "id", "code", "file_path", "repo_name", "repo_owner",
                        "stars", "is_official", "license", "loc",
                    ]}))

    # Phase 4: Fill to target N with random remaining
    remaining = n - len(samples)
    if remaining > 0:
        pool = [r for r in classified if r["id"] not in sampled_ids]
        chosen = _pick_from_pool(pool, remaining)
        for c in chosen:
            sampled_ids.add(c["id"])
            samples.append(SampledFile(**{k: c[k] for k in [
                "id", "code", "file_path", "repo_name", "repo_owner",
                "stars", "is_official", "license", "loc",
            ]}))

    return samples[:n]


def print_sample_summary(samples: list[SampledFile]) -> None:
    """Print a summary of the sampled files."""
    print(f"\n{'='*60}")
    print(f"SAMPLED FILES: {len(samples)} total")
    print(f"{'='*60}")

    by_complexity = {}
    by_source = {}
    by_tier = {}
    for s in samples:
        c = _classify_complexity(s.loc)
        src = _classify_source(s.file_path)
        t = _classify_tier(s.is_official, s.stars)
        by_complexity[c] = by_complexity.get(c, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1
        by_tier[t] = by_tier.get(t, 0) + 1

    print(f"\n  Complexity: {by_complexity}")
    print(f"  Source:     {by_source}")
    print(f"  Tier:       {by_tier}")
    print(f"\n  Files:")
    for i, s in enumerate(samples, 1):
        comp = _classify_complexity(s.loc)
        src = _classify_source(s.file_path)
        tier = _classify_tier(s.is_official, s.stars)
        print(f"    {i:2d}. [{comp:>8s} | {src:>8s} | {tier:>14s}] "
              f"{s.loc:>5d} LOC  ★{s.stars:<4d}  {s.id[:60]}")
    print()
