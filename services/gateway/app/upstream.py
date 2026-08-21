"""HTTP client for the backend pods.

One pooled AsyncClient for the process (creating one per request is the classic
way to exhaust ephemeral ports under load). Retries cover the failures that are
genuinely transient on RunPod:

  * connect errors — a pod restarting, or the proxy briefly 502ing
  * 503 not_ready  — a pod that is up but still streaming models off the volume

4xx is never retried: a bad request will be just as bad the second time.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from crm_common.errors import ApiError, UpstreamError, UpstreamTimeout
from crm_common.logging import request_id_var
from crm_common.security import env_keys

from . import config

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None
_RETRYABLE_STATUSES = {502, 503, 504}

# 503 means two very different things and they need opposite handling.
#
#   not_ready   a pod is up but still streaming models off the volume. Transient,
#               resolves on its own, retrying is exactly right.
#   gpu_busy /  the backend is SATURATED. Retrying does not wait for capacity, it
#   pool_busy   adds load — and each retry re-enters the queue ahead of somebody,
#   queue_full  so under pressure the retries themselves become the pressure.
#               Classic congestion collapse: the system is slowest precisely when
#               it is being asked hardest, and UPSTREAM_RETRIES=2 turned every
#               overloaded request into three.
#
# So retry on status, except where the backend has explicitly told us it is full.
_SATURATION_CODES = {"gpu_busy", "pool_busy", "queue_full", "rate_limited"}


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
            follow_redirects=False,
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class Upstream:
    def __init__(self, name: str, base_url: str, timeout: float, *, internal: bool = True,
                 extra_headers: dict | None = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.internal = internal
        self.extra_headers = extra_headers or {}

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            # Propagate the id so one grep reconstructs the chain across pods.
            "X-Request-ID": request_id_var.get(),
            **self.extra_headers,
        }
        if self.internal:
            keys = env_keys("INTERNAL_API_KEY")
            if keys:
                headers["X-Internal-Key"] = keys[0]
        return headers

    async def post(self, path: str, payload: dict, *, timeout: float | None = None,
                   retries: int | None = None) -> dict:
        if not self.configured:
            raise ApiError(
                f"The '{self.name}' backend is not configured on the gateway "
                f"(set {self.name.upper()}_URL).",
                status=503,
                code="not_configured",
            )
        url = f"{self.base_url}{path}"
        attempts = (retries if retries is not None else config.UPSTREAM_RETRIES) + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = await client().post(
                    url, json=payload, headers=self._headers(), timeout=timeout or self.timeout
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                log.warning("upstream timeout", extra={"upstream": self.name, "attempt": attempt})
                raise UpstreamTimeout(
                    f"The '{self.name}' service did not respond in time."
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning(
                    "upstream connect failed",
                    extra={"upstream": self.name, "attempt": attempt, "error": str(exc)},
                )
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2 ** attempt, 5))
                    continue
                raise UpstreamError(f"Could not reach the '{self.name}' service.") from exc

            if response.status_code < 400:
                return response.json()

            body = _safe_json(response)
            error_code = ((body or {}).get("error") or {}).get("code")
            saturated = error_code in _SATURATION_CODES
            if response.status_code in _RETRYABLE_STATUSES and attempt + 1 < attempts and not saturated:
                log.warning(
                    "upstream retryable status",
                    extra={"upstream": self.name, "status": response.status_code, "attempt": attempt},
                )
                await asyncio.sleep(min(2 ** attempt, 5))
                continue
            if saturated:
                # Surface it immediately so the caller can back off with the
                # `retry_after_s` the backend supplied, instead of us spending
                # its capacity re-asking on their behalf.
                log.warning(
                    "upstream saturated, not retrying",
                    extra={"upstream": self.name, "code": error_code},
                )

            # Surface the backend's own typed error rather than flattening it.
            error = (body or {}).get("error") or {}
            raise ApiError(
                error.get("message") or f"The '{self.name}' service failed.",
                status=_client_status(response.status_code),
                code=error.get("code") or "upstream_error",
                detail=error.get("detail"),
            )

        raise UpstreamError(f"The '{self.name}' service is unavailable.") from last_error

    async def ready(self) -> dict:
        if not self.configured:
            return {"configured": False}
        try:
            response = await client().get(f"{self.base_url}/readyz", headers=self._headers(), timeout=10)
            return {"configured": True, "ready": response.status_code == 200, **(_safe_json(response) or {})}
        except httpx.HTTPError as exc:
            return {"configured": True, "ready": False, "error": f"{type(exc).__name__}: {exc}"}


def _client_status(upstream_status: int) -> int:
    """Map a backend status onto what the caller should see.

    A backend 500 must not be echoed as 500 — that would claim the *gateway*
    broke — so it becomes 502. But 503 and 429 mean the same thing at both
    layers, "temporarily unavailable, come back", and flattening them to 502
    destroys the distinction the client acts on: 502 says "something failed,
    retry once", 503 says "we are full, wait `retry_after_s`". Saturation
    arriving as 502 is how a caller learns to hammer a queue that is already
    over capacity.
    """
    if upstream_status < 500 or upstream_status in (503,):
        return upstream_status
    return 502


def _safe_json(response: httpx.Response) -> dict | None:
    try:
        return response.json()
    except Exception:
        return {"error": {"message": response.text[:300]}}


# --------------------------------------------------------------------------- #
# The registry. Everything else in the gateway routes through these four.
# --------------------------------------------------------------------------- #

vision = Upstream("vision", config.VISION_URL, config.TIMEOUT_VISION)
diffusion = Upstream("diffusion", config.DIFFUSION_URL, config.TIMEOUT_DIFFUSION)
cpu = Upstream("cpu", config.CPU_URL, config.TIMEOUT_FAST)

# The chat pod predates this repo and has its own auth (CHAT_POD_KEY on the
# translate API; Ollama itself has none), so it is not an "internal" upstream.
chat = Upstream(
    "chat",
    config.CHAT_URL,
    config.TIMEOUT_CHAT,
    internal=False,
    extra_headers={"Authorization": f"Bearer {config.CHAT_POD_KEY}"} if config.CHAT_POD_KEY else {},
)
translate = Upstream(
    "translate",
    config.TRANSLATE_URL,
    config.TIMEOUT_CHAT,
    internal=False,
    extra_headers={"Authorization": f"Bearer {config.CHAT_POD_KEY}"} if config.CHAT_POD_KEY else {},
)

BY_NAME = {"vision": vision, "diffusion": diffusion, "cpu": cpu, "chat": chat, "translate": translate}
