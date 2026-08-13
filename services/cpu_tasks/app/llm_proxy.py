"""Groq passthrough — the server holds only the keys.

The split of responsibility is the whole point and it is preserved exactly from
the serverless build:

  caller  owns the prompt, the model, and the structured-output schema
  server  owns the API keys, and nothing else

The body you send is forwarded to Groq verbatim and Groq's full JSON response
comes back untouched. Keys rotate on rate-limit/quota so a 429 on one key falls
through to the next; a malformed request fails fast instead of burning every key.
"""

from __future__ import annotations

import logging
import os
import threading

import httpx

from crm_common.errors import ApiError, BadRequest, UpstreamError
from crm_common.security import parse_keys

from . import config

log = logging.getLogger(__name__)

# Module state: advances past a rate-limited key so the next request does not
# re-hit it first. Guarded because uvicorn may serve concurrent requests.
_cursor = 0
_cursor_lock = threading.Lock()

# Groq rejects these for reasons another key will not fix.
_FATAL_STATUSES = {400, 404, 405, 413, 415, 422}

# One pooled client for the process. Building an AsyncClient per request means a
# fresh connection pool and a fresh TLS handshake every time, which on a
# detection-heavy workload is most of the latency.
_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=config.GROQ_TIMEOUT,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def keys() -> list[str]:
    return parse_keys(os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY"))


def configured() -> bool:
    return bool(keys())


async def chat(body: dict) -> dict:
    """POST body to Groq /chat/completions, rotating keys on transient failure."""
    if not body.get("messages"):
        raise BadRequest("Missing 'messages' in the LLM request body.")

    body = dict(body)
    body.setdefault("model", config.GROQ_MODEL)
    # /chat here is request/response; streaming would need a different transport
    # all the way through the gateway, so it is explicitly disabled.
    body["stream"] = False

    api_keys = keys()
    if not api_keys:
        raise ApiError(
            "Server is missing GROQ_API_KEYS (set it on the cpu pod).",
            status=503,
            code="not_configured",
        )

    global _cursor
    with _cursor_lock:
        start = _cursor

    url = f"{config.GROQ_BASE_URL}/chat/completions"
    errors: list[str] = []

    http = client()
    for attempt in range(len(api_keys)):
        index = (start + attempt) % len(api_keys)
        try:
            response = await http.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_keys[index]}",
                    "Content-Type": "application/json",
                    "User-Agent": config.GROQ_USER_AGENT,
                },
            )
        except httpx.TimeoutException as exc:
            errors.append(f"key#{index} timeout: {exc}")
            continue
        except httpx.HTTPError as exc:
            errors.append(f"key#{index} {type(exc).__name__}: {exc}")
            continue

        if response.status_code < 400:
            with _cursor_lock:
                _cursor = index  # stick with the key that just worked
            return {"response": response.json(), "key_index": index}

        detail = response.text[:400]
        errors.append(f"key#{index} HTTP {response.status_code}: {detail}")
        if response.status_code in _FATAL_STATUSES:
            raise BadRequest(f"Groq rejected the request (HTTP {response.status_code}): {detail}")
        # 429 rate-limit / 401-403 dead key / 5xx transient -> next key.
        log.warning("groq key failed over", extra={"key_index": index, "status": response.status_code})

    raise UpstreamError(
        "All Groq keys failed or are rate-limited.",
        detail=errors,
    )


async def health() -> dict:
    return {"groq_keys": len(keys()), "groq_model": config.GROQ_MODEL}
