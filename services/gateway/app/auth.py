"""Client authentication and rate limiting.

One key per consumer (`CLIENT_API_KEYS=key1,key2`) so you can hand the CRM team
their own and revoke it without touching anyone else's. Keys are compared in
constant time and only ever logged as a 12-char fingerprint.

The limiter is a per-key token bucket held in process memory. That is correct
while the gateway is a single pod — which it is. If you ever run two gateway
replicas, move this to Redis or the limit becomes per-replica.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import Request

from crm_common.errors import RateLimited, Unauthorized
from crm_common.security import env_keys, key_fingerprint, key_matches

from . import config

log = logging.getLogger(__name__)

_buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (float(config.RATE_LIMIT_BURST), 0.0))
_bucket_lock = Lock()


def client_keys() -> list[str]:
    return env_keys("CLIENT_API_KEYS")


def authenticate(request: Request) -> str:
    """Return the caller's key fingerprint, or raise 401."""
    presented = request.headers.get("X-API-Key")
    if not presented:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()

    allowed = client_keys()
    if not allowed:
        raise Unauthorized(
            "The gateway has no CLIENT_API_KEYS configured, so it is refusing every request."
        )
    if not key_matches(presented, allowed):
        raise Unauthorized("Invalid or missing API key. Send it as 'X-API-Key'.")
    return key_fingerprint(presented)


def check_rate_limit(fingerprint: str) -> None:
    """Token bucket: RATE_LIMIT_PER_MINUTE sustained, RATE_LIMIT_BURST instantaneous."""
    rate_per_second = config.RATE_LIMIT_PER_MINUTE / 60.0
    now = time.monotonic()
    with _bucket_lock:
        tokens, last = _buckets[fingerprint]
        if last:
            tokens = min(float(config.RATE_LIMIT_BURST), tokens + (now - last) * rate_per_second)
        if tokens < 1.0:
            _buckets[fingerprint] = (tokens, now)
            retry_after = max(1, int((1.0 - tokens) / rate_per_second) + 1)
            raise RateLimited(
                f"Rate limit exceeded ({config.RATE_LIMIT_PER_MINUTE}/min). "
                f"Retry in ~{retry_after}s.",
                detail={"retry_after_s": retry_after},
            )
        _buckets[fingerprint] = (tokens - 1.0, now)
