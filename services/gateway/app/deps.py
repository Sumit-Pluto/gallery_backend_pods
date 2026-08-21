"""Shared route dependencies."""

from __future__ import annotations

from fastapi import Request

from . import auth


async def require_client(request: Request) -> str:
    """Authenticate the caller and charge them a rate-limit token.

    Returns the key fingerprint, which is used as the job owner so one client
    cannot poll another's render.

    Rate limiting is charged against the *fair key* — the end user behind the
    shared API key when the caller names one — while the returned fingerprint
    stays the authorisation boundary. Keeping those separate is deliberate: a
    client that forges or omits the end-user header can only affect its own
    throughput, never what it is allowed to read.
    """
    fingerprint = auth.authenticate(request)
    fair = auth.fair_key(request, fingerprint)
    auth.check_rate_limit(fair)
    request.state.client = fingerprint
    request.state.fair_key = fair
    return fingerprint
