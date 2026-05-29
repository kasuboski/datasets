"""Registry: discover Gleam repos via GitHub, enrich with Hex.pm metadata.

GitHub search (language:gleam) is the primary source for repos.
Hex.pm search is used to enrich repos that also publish packages there.

GitHub search API limits each query to 1,000 results. We work around
this by chunking queries into star-count ranges (and date ranges for
the large 0-star slice).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

GITHUB_API = "https://api.github.com"
HEX_API = "https://hex.pm/api"


@dataclass
class RepoInfo:
    """A repo to be collected, with metadata from GitHub + optional Hex enrichment."""
    owner: str
    name: str
    url: str
    description: str = ""
    stars: int = 0
    license_raw: str | None = None
    authors: list[str] = field(default_factory=list)
    fork: bool = False
    archived: bool = False
    # Hex enrichment (set later)
    hex_package: str | None = None
    hex_downloads: int | None = None


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers, using GH_TOKEN if available."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_repo_item(item: dict) -> RepoInfo:
    """Parse a GitHub API repo item into a RepoInfo."""
    license_info = item.get("license")
    license_raw = license_info.get("spdx_id") if license_info else None
    if license_raw == "NOASSERTION":
        license_raw = None

    owner_login = item.get("owner", {}).get("login", "")
    full_name = item.get("full_name", "")
    name = full_name.split("/", 1)[-1] if "/" in full_name else full_name

    return RepoInfo(
        owner=owner_login,
        name=name,
        url=item.get("html_url", f"https://github.com/{full_name}"),
        description=item.get("description", "") or "",
        stars=item.get("stargazers_count", 0),
        license_raw=license_raw,
        authors=[owner_login] if owner_login else [],
        fork=item.get("fork", False),
        archived=item.get("archived", False),
    )


def _github_search_page(
    query: str,
    headers: dict,
    page: int = 1,
    per_page: int = 100,
    rate_limit_pause: float = 2.0,
) -> tuple[list[dict], int, bool]:
    """Execute a single GitHub search API page request.

    Returns (items, total_count, has_more).
    Handles rate limiting with automatic retry.
    """
    while True:
        resp = requests.get(
            f"{GITHUB_API}/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
            headers=headers,
            timeout=30,
        )

        if resp.status_code == 403:
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset:
                wait = max(float(reset) - time.time() + 1, 1)
                print(f"\n    Rate limited. Waiting {wait:.0f}s until reset...")
                time.sleep(wait)
                continue
            else:
                pause = rate_limit_pause * 10
                print(f"\n    Rate limited (no reset header). Pausing {pause:.0f}s...")
                time.sleep(pause)
                continue

        if resp.status_code == 422:
            return [], 0, False

        if resp.status_code != 200:
            print(f"\n    GitHub API error: HTTP {resp.status_code} — {resp.text[:200]}")
            return [], 0, False

        data = resp.json()
        items = data.get("items", [])
        total_count = data.get("total_count", 0)
        has_more = len(items) == per_page and page * per_page < 1000

        return items, total_count, has_more


def _fetch_query(
    query: str,
    headers: dict,
    per_page: int = 100,
    rate_limit_pause: float = 2.0,
    max_repos: int | None = None,
) -> list[RepoInfo]:
    """Fetch all results for a single GitHub search query (up to 1000).

    Paginates through all available pages for this query.
    """
    repos: list[RepoInfo] = []
    page = 1

    while True:
        items, total_count, has_more = _github_search_page(
            query, headers, page, per_page, rate_limit_pause,
        )

        if not items:
            break

        for item in items:
            repos.append(_parse_repo_item(item))

        if max_repos and len(repos) >= max_repos:
            return repos[:max_repos]

        if not has_more:
            break

        page += 1
        time.sleep(rate_limit_pause)

    return repos


def _generate_star_chunks() -> list[str]:
    """Generate star-range query qualifiers to cover all repos.

    GitHub search returns max 1,000 results per query.
    We split by star count ranges so each slice fits under 1,000.

    Current Gleam distribution (~4,660 repos):
      stars:>=100     → ~44 repos
      stars:50..99    → small
      stars:20..49    → small
      stars:10..19    → small
      stars:5..9      → ~273 repos
      stars:3..4      → ~236 repos
      stars:1..2      → ~744 repos
      stars:0         → ~3,023 repos  ← split by creation date
    """
    chunks = [
        "stars:>=100",
        "stars:50..99",
        "stars:20..49",
        "stars:10..19",
        "stars:5..9",
        "stars:3..4",
        "stars:1..2",
    ]

    # The 0-star slice is too large for a single query.
    # Split by creation date into windows that should each be <1,000.
    date_windows = [
        ("2025-01-01", "2025-06-30"),
        ("2025-07-01", "2025-12-31"),
        ("2026-01-01", "2026-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
        ("2023-01-01", "2023-06-30"),
        ("2023-07-01", "2023-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2020-01-01", "2021-12-31"),
        ("2015-01-01", "2019-12-31"),
    ]
    for start, end in date_windows:
        chunks.append(f"stars:0 created:{start}..{end}")

    return chunks


def discover_github_repos(
    max_repos: int | None = None,
    sort: str = "stars",
    per_page: int = 100,
    rate_limit_pause: float = 2.0,
) -> list[RepoInfo]:
    """Discover all GitHub repos with language:gleam.

    Uses GitHub search API with star-range chunking to get past the
    1,000-result-per-query limit. Requires GH_TOKEN for reasonable
    rate limits (without token: ~10 req/min, you WILL hit limits).

    Strategy: partition the repo space into star-count slices.
    Each slice should be <1,000 repos, fitting within one query.
    The 0-star slice (largest) is further split by creation date.

    Args:
        max_repos: Stop after this many repos (for testing). None = all.
        sort: Ignored (we always sort by stars via chunking order).
        per_page: Results per page (max 100).
        rate_limit_pause: Seconds to pause between API calls.
    """
    headers = _github_headers()
    seen: set[str] = set()  # deduplicate across chunks
    repos: list[RepoInfo] = []

    chunks = _generate_star_chunks()
    print(f"  Using {len(chunks)} star/date chunks to cover all repos")

    for i, star_qualifier in enumerate(chunks):
        query = f"language:gleam {star_qualifier}"
        print(f"  Chunk {i+1}/{len(chunks)}: {star_qualifier}", end="")

        # First, check total count for this chunk
        _, total_count, _ = _github_search_page(
            query, headers, page=1, per_page=1, rate_limit_pause=rate_limit_pause,
        )
        print(f" ({total_count:,} repos)", end="")

        if total_count == 0:
            print(" — empty, skipping")
            continue

        # If a chunk still exceeds 1,000, warn but proceed
        if total_count > 1000:
            print(f"\n    WARNING: {total_count} > 1000, some repos may be missed", end="")

        chunk_repos = _fetch_query(
            query, headers, per_page, rate_limit_pause, max_repos,
        )

        # Deduplicate across chunks
        new_count = 0
        for repo in chunk_repos:
            key = f"{repo.owner}/{repo.name}"
            if key not in seen:
                seen.add(key)
                repos.append(repo)
                new_count += 1

        print(f" → {new_count} new ({len(repos)} total)")

        if max_repos and len(repos) >= max_repos:
            repos = repos[:max_repos]
            break

        time.sleep(rate_limit_pause)

    print(f"  Discovered {len(repos)} unique repos across {len(chunks)} chunks")
    return repos


def discover_hex_packages(
    rate_limit_pause: float = 0.3,
) -> dict[str, dict]:
    """Fetch all Hex packages matching search='gleam'.

    Returns a dict keyed by lowercase package name → {downloads, license, github_url}.
    Used for enriching GitHub repos with Hex metadata.
    """
    packages: dict[str, dict] = {}
    page = 1

    while True:
        resp = requests.get(
            f"{HEX_API}/packages",
            params={"search": "gleam", "page": page, "per_page": 100},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  Hex API error: HTTP {resp.status_code}")
            break

        data = resp.json()
        if not data:
            break

        for pkg in data:
            name = pkg.get("name", "").lower()
            meta = pkg.get("meta", {})
            links = meta.get("links", {})
            gh_url = links.get("GitHub", "")

            # Extract owner/repo from GitHub URL for matching
            gh_owner_repo = None
            if gh_url and "github.com/" in gh_url:
                parts = gh_url.rstrip("/").split("github.com/", 1)[-1]
                gh_owner_repo = parts.lower()

            licenses = meta.get("licenses", [])
            downloads = pkg.get("downloads", {})

            packages[name] = {
                "hex_downloads": downloads.get("all") if isinstance(downloads, dict) else None,
                "license_raw": licenses[0] if licenses else None,
                "github_url": gh_url,
                "github_owner_repo": gh_owner_repo,
            }

        if len(data) < 100:
            break

        page += 1
        time.sleep(rate_limit_pause)

    return packages


def enrich_repos_with_hex(
    repos: list[RepoInfo],
    hex_packages: dict[str, dict],
) -> list[RepoInfo]:
    """Cross-reference GitHub repos with Hex.pm packages.

    Matches by:
    1. GitHub URL match (hex_package.github_owner_repo == repo.owner/repo.name)
    2. Package name match (hex package name == repo name, lowercased)
    """
    # Build lookup: github owner/repo (lowercase) → hex data
    gh_to_hex: dict[str, tuple[str, dict]] = {}
    for pkg_name, pkg_data in hex_packages.items():
        if pkg_data.get("github_owner_repo"):
            gh_to_hex[pkg_data["github_owner_repo"]] = (pkg_name, pkg_data)
        # Also map by package name for name-based matching
        gh_to_hex[f"_pkg_{pkg_name}"] = (pkg_name, pkg_data)

    for repo in repos:
        # Try GitHub URL match first
        key = f"{repo.owner.lower()}/{repo.name.lower()}"
        match = gh_to_hex.get(key)

        # Fallback: match by repo name == hex package name
        if not match:
            match = gh_to_hex.get(f"_pkg_{repo.name.lower()}")

        if match:
            pkg_name, pkg_data = match
            repo.hex_package = pkg_name
            repo.hex_downloads = pkg_data.get("hex_downloads")
            # Use Hex license if GitHub didn't have one
            if not repo.license_raw and pkg_data.get("license_raw"):
                repo.license_raw = pkg_data["license_raw"]

    return repos
