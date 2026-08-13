"""Shared route dependencies."""

from __future__ import annotations

from fastapi import Request

from . import auth


async def require_client(request: Request) -> str:
    """Authenticate the caller and charge them a rate-limit token.

    Returns the key fingerprint, which is used as the job owner so one client
    cannot poll another's render.
    """
    fingerprint = auth.authenticate(request)
    auth.check_rate_limit(fingerprint)
    request.state.client = fingerprint
    return fingerprint
