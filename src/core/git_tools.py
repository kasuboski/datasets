"""Shallow clone repos and extract .gleam files.

Handles clone failures gracefully, uses temp dirs for isolation,
and supports resumability via commit SHA tracking.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.core.license_checker import check_license, is_permissive

# Official Gleam orgs/repos — used to set is_official flag.
OFFICIAL_OWNERS = frozenset({
    "gleam-lang",
    "gleam-wisp",
    "lustre-labs",
})

# Extraction-time filter constants.
MAX_FILE_CHARS = 50_000  # files larger than this are likely auto-generated
MIN_FILE_CHARS = 20      # files smaller than this are likely stubs


@dataclass
class RepoResult:
    """Result of processing a single repo."""
    owner: str
    name: str
    url: str
    files: list[dict]       # corpus records (one per .gleam file)
    commit_sha: str | None
    error: str | None = None


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a git command, raising on failure."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result


def get_head_sha(repo_dir: Path) -> str | None:
    """Get HEAD commit SHA of a cloned repo."""
    result = _run_git(["rev-parse", "HEAD"], cwd=repo_dir)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def shallow_clone(repo_url: str, dest: Path, timeout: int = 120) -> bool:
    """Shallow clone a repo. Returns True on success."""
    result = _run_git(
        ["clone", "--depth", "1", "--quiet", repo_url, str(dest)],
        timeout=timeout,
    )
    if result.returncode != 0:
        # Some repos fail with --depth 1 due to server config; try full clone
        result2 = _run_git(
            ["clone", "--quiet", repo_url, str(dest)],
            timeout=timeout * 3,
        )
        return result2.returncode == 0
    return True


def extract_gleam_files(
    repo_dir: Path,
    repo_url: str,
    owner: str,
    name: str,
    *,
    repo_description: str = "",
    stars: int = 0,
    license_raw: str | None = None,
    authors: list[str] | None = None,
    hex_package: str | None = None,
    hex_downloads: int | None = None,
    fork: bool = False,
    archived: bool = False,
) -> list[dict]:
    """Extract all .gleam files from a cloned repo into corpus records."""
    spdx, raw_license, is_perm = check_license(license_raw)
    is_official = owner in OFFICIAL_OWNERS
    commit_sha = get_head_sha(repo_dir)
    now = datetime.now(timezone.utc).isoformat()

    records = []
    for gleam_path in sorted(repo_dir.rglob("*.gleam")):
        # Skip files in hidden dirs or build artifacts
        parts = gleam_path.relative_to(repo_dir).parts
        if any(p.startswith(".") for p in parts):
            continue
        if "build" in parts and "src" not in parts:
            continue

        try:
            code = gleam_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Extraction-time filter: skip generated files and stubs
        file_len = len(code)
        if file_len > MAX_FILE_CHARS:
            continue
        if file_len < MIN_FILE_CHARS:
            continue

        file_path = str(gleam_path.relative_to(repo_dir))
        # Build deterministic ID
        record_id = f"{owner}/{name}/{file_path}"

        records.append({
            "id": record_id,
            "source": "github",
            "code": code,
            "file_path": file_path,
            "repo_url": repo_url,
            "repo_owner": owner,
            "repo_name": name,
            "repo_description": repo_description,
            "stars": stars,
            "is_official": is_official,
            "license": spdx,
            "license_raw": raw_license,
            "authors": authors or [],
            "hex_package": hex_package,
            "hex_downloads": hex_downloads,
            "fork": fork,
            "archived": archived,
            "collected_at": now,
            "commit_sha": commit_sha,
            "schema_version": 1,
        })

    return records


def process_repo(
    repo_url: str,
    owner: str,
    name: str,
    *,
    repo_description: str = "",
    stars: int = 0,
    license_raw: str | None = None,
    authors: list[str] | None = None,
    hex_package: str | None = None,
    hex_downloads: int | None = None,
    fork: bool = False,
    archived: bool = False,
    clone_timeout: int = 120,
) -> RepoResult:
    """Clone a repo, extract .gleam files, clean up.

    Uses a temp directory — no persistent checkout.
    """
    with tempfile.TemporaryDirectory(prefix=f"gleam_{owner}_{name}_") as tmpdir:
        repo_dir = Path(tmpdir) / name
        try:
            ok = shallow_clone(repo_url, repo_dir, timeout=clone_timeout)
            if not ok:
                return RepoResult(
                    owner=owner, name=name, url=repo_url,
                    files=[], commit_sha=None,
                    error="clone_failed",
                )

            commit_sha = get_head_sha(repo_dir)
            records = extract_gleam_files(
                repo_dir, repo_url, owner, name,
                repo_description=repo_description,
                stars=stars,
                license_raw=license_raw,
                authors=authors,
                hex_package=hex_package,
                hex_downloads=hex_downloads,
                fork=fork,
                archived=archived,
            )
            return RepoResult(
                owner=owner, name=name, url=repo_url,
                files=records, commit_sha=commit_sha,
            )
        except subprocess.TimeoutExpired:
            return RepoResult(
                owner=owner, name=name, url=repo_url,
                files=[], commit_sha=None,
                error="clone_timeout",
            )
        except Exception as e:
            return RepoResult(
                owner=owner, name=name, url=repo_url,
                files=[], commit_sha=None,
                error=f"error: {e}",
            )
