"""Key handling. Comparisons are constant-time; keys never appear in logs."""

from __future__ import annotations

import hashlib
import hmac
import os


def parse_keys(raw: str | None) -> list[str]:
    """`a,b,c` (or newline separated) -> ['a', 'b', 'c']. Blank entries dropped.

    Multiple keys let you rotate without downtime: add the new key, hand it over,
    remove the old one on the next restart.
    """
    if not raw:
        return []
    return [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]


def key_matches(candidate: str | None, allowed: list[str]) -> bool:
    if not candidate or not allowed:
        return False
    # compare_digest against every key regardless of an early match, so response
    # time does not leak which key (or how many) matched.
    matched = False
    for key in allowed:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


def key_fingerprint(key: str | None) -> str:
    """Short stable id for logs/metrics. Never log the key itself."""
    if not key:
        return "-"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def env_keys(name: str) -> list[str]:
    return parse_keys(os.environ.get(name))
