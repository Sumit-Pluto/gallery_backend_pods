"""Typed errors that map cleanly onto HTTP status codes.

Every service raises `ApiError` (or a subclass) instead of returning
`{"error": "..."}` dicts the way the old RunPod handlers did. The FastAPI
exception handler installed by `crm_common.service.create_app` turns them into a
consistent JSON body so the gateway — and therefore the client — never has to
guess whether a 200 actually meant failure.
"""

from __future__ import annotations


class ApiError(Exception):
    """Base class for every expected failure. `status` reaches the caller."""

    status = 500
    code = "internal_error"

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None, detail=None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code
        self.detail = detail

    def to_dict(self) -> dict:
        body = {"error": {"code": self.code, "message": self.message}}
        if self.detail is not None:
            body["error"]["detail"] = self.detail
        return body


class BadRequest(ApiError):
    status = 400
    code = "bad_request"


class Unauthorized(ApiError):
    status = 401
    code = "unauthorized"


class PayloadTooLarge(ApiError):
    status = 413
    code = "payload_too_large"


class RateLimited(ApiError):
    status = 429
    code = "rate_limited"


class NotReady(ApiError):
    """Models are still loading. The caller should retry shortly."""

    status = 503
    code = "not_ready"


class UpstreamError(ApiError):
    """A downstream service (vision / diffusion / cpu / chat-pod) failed."""

    status = 502
    code = "upstream_error"


class UpstreamTimeout(ApiError):
    status = 504
    code = "upstream_timeout"
