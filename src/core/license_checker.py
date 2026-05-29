"""Best-effort permissive license identification.

Normalizes variant license strings to SPDX identifiers and checks
whether they're on the permissive list. Not a legal tool — just
a practical filter for corpus collection.
"""

from __future__ import annotations

# SPDX identifiers we consider permissive for training data purposes.
PERMISSIVE_LICENSES = frozenset({
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "0BSD",
    "ISC",
    "Unlicense",
    "CC0-1.0",
    "MPL-2.0",   # weak copyleft, generally acceptable
})

# Mapping from common variant strings → canonical SPDX.
# Keys are lowercased for matching.
_NORMALIZE_MAP: dict[str, str] = {
    # MIT variants
    "mit": "MIT",
    "mit license": "MIT",
    "the mit license": "MIT",
    "expat": "MIT",
    # Apache variants
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache 2": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache license version 2.0": "Apache-2.0",
    "apache-2": "Apache-2.0",
    # BSD variants
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd": "BSD-3-Clause",  # ambiguous, assume 3-clause
    "3-clause bsd": "BSD-3-Clause",
    "2-clause bsd": "BSD-2-Clause",
    "simplified bsd": "BSD-2-Clause",
    "new bsd": "BSD-3-Clause",
    # ISC
    "isc": "ISC",
    "isc license": "ISC",
    # Unlicense
    "unlicense": "Unlicense",
    "the unlicense": "Unlicense",
    # CC0
    "cc0-1.0": "CC0-1.0",
    "cc0": "CC0-1.0",
    # MPL
    "mpl-2.0": "MPL-2.0",
    "mpl 2.0": "MPL-2.0",
    "mozilla public license 2.0": "MPL-2.0",
}

# Non-permissive licenses we want to flag (not exhaustive).
NON_PERMISSIVE = frozenset({
    "GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0",
    "GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only",
    "GPL-2.0-or-later", "GPL-3.0-or-later",
})


def normalize(raw: str | None) -> str | None:
    """Normalize a license string to an SPDX identifier.

    Returns None if the string is empty, None, or unrecognized.
    """
    if not raw:
        return None

    cleaned = raw.strip().lower()

    # Direct lookup
    if cleaned in _NORMALIZE_MAP:
        return _NORMALIZE_MAP[cleaned]

    # Check if it's already a valid SPDX id (case-insensitive)
    upper = raw.strip()
    for spdx in PERMISSIVE_LICENSES | NON_PERMISSIVE:
        if spdx.lower() == cleaned:
            return spdx

    # Heuristic: check for keywords
    if "mit" in cleaned:
        return "MIT"
    if "apache" in cleaned:
        return "Apache-2.0"
    if "bsd" in cleaned:
        if "2" in cleaned or "simplified" in cleaned:
            return "BSD-2-Clause"
        return "BSD-3-Clause"
    if "isc" in cleaned:
        return "ISC"

    return None


def is_permissive(raw: str | None) -> bool:
    """Check if a license string is permissive.

    Returns False for None, unrecognized, or copyleft licenses.
    """
    spdx = normalize(raw)
    if spdx is None:
        return False
    return spdx in PERMISSIVE_LICENSES


def check_license(raw: str | None) -> tuple[str | None, str | None, bool]:
    """Full license check returning (spdx, raw, is_permissive).

    Useful for building corpus records where you want all three.
    """
    spdx = normalize(raw)
    perm = spdx in PERMISSIVE_LICENSES if spdx else False
    return spdx, raw, perm
