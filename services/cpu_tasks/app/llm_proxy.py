"""OpenAI-shaped passthrough with provider failover — the server holds only keys.

The split of responsibility is the whole point and it is preserved exactly from
the serverless build:

  caller  owns the prompt, the model, and the structured-output schema
  server  owns the API keys, and nothing else

What changed is that there is now more than one place those keys can point. Both
Alibaba Cloud Model Studio and Groq expose the same `/chat/completions` shape, so
one transport serves both and the only per-provider state is a base URL, a key
list and a model name.

Failure handling has two nested loops, and the nesting matters:

  * within a provider, keys rotate — a 429 on one key falls through to the next
  * across providers, the chain advances — every key rate-limited or the whole
    provider down moves to the next provider entirely

That second loop is what the YOLO fallback used to be for, except it answers with
the same open vocabulary instead of a fixed class list. A malformed request fails
fast rather than burning every key everywhere.
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

# Rejected for reasons another key — or another provider — will not fix.
_FATAL_STATUSES = {400, 404, 405, 413, 415, 422}

# Module state: advances past a rate-limited key so the next request does not
# re-hit it first. Keyed by provider; guarded because uvicorn serves concurrently.
_cursors: dict[str, int] = {}
_cursor_lock = threading.Lock()

# One pooled client per provider. Building an AsyncClient per request means a
# fresh connection pool and a fresh TLS handshake every time, which on a
# detection-heavy workload is most of the latency.
_clients: dict[str, httpx.AsyncClient] = {}
_clients_lock = threading.Lock()


def client(provider: config.Provider) -> httpx.AsyncClient:
    existing = _clients.get(provider.name)
    if existing is not None:
        return existing
    with _clients_lock:
        if provider.name not in _clients:
            _clients[provider.name] = httpx.AsyncClient(
                timeout=provider.timeout,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
            )
        return _clients[provider.name]


async def aclose() -> None:
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for http in clients:
        await http.aclose()


def keys(provider: config.Provider) -> list[str]:
    for name in provider.keys_env:
        found = parse_keys(os.environ.get(name))
        if found:
            return found
    return []


def configured() -> bool:
    return bool(config.providers())


def _headers(provider: config.Provider, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **provider.extra_headers,
    }
    if provider.user_agent:
        headers["User-Agent"] = provider.user_agent
    return headers


def _model_for(provider: config.Provider, model_kind: str | None) -> str | None:
    if model_kind == "vision":
        return provider.vision_model
    if model_kind == "chat":
        return provider.model
    return None


def _log_usage(provider: config.Provider, model: str, payload: dict, task: str) -> None:
    """Token accounting. You cannot optimise a spend you are not measuring, and
    the old code threw `usage` away."""
    usage = (payload or {}).get("usage") or {}
    if not usage:
        return
    log.info(
        "llm usage",
        extra={
            "task": task,
            "provider": provider.name,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    )


async def _call_provider(
    provider: config.Provider, body: dict, model_kind: str | None, task: str
) -> tuple[dict | None, list[str], bool]:
    """Try every key on one provider.

    Returns (result, errors, fatal). `fatal` means the request itself was
    rejected — do not keep spending keys on it.
    """
    api_keys = keys(provider)
    if not api_keys:
        return None, [f"{provider.name}: no keys configured"], False

    body = dict(body)
    model = _model_for(provider, model_kind)
    if model:
        body["model"] = model
    model = body.get("model") or provider.model

    with _cursor_lock:
        start = _cursors.get(provider.name, 0)

    url = f"{provider.base_url}/chat/completions"
    errors: list[str] = []
    http = client(provider)

    for attempt in range(len(api_keys)):
        index = (start + attempt) % len(api_keys)
        try:
            response = await http.post(url, json=body, headers=_headers(provider, api_keys[index]))
        except httpx.TimeoutException as exc:
            errors.append(f"{provider.name} key#{index} timeout: {exc}")
            continue
        except httpx.HTTPError as exc:
            errors.append(f"{provider.name} key#{index} {type(exc).__name__}: {exc}")
            continue

        if response.status_code < 400:
            with _cursor_lock:
                _cursors[provider.name] = index  # stick with the key that just worked
            payload = response.json()
            _log_usage(provider, model, payload, task)
            return (
                {"response": payload, "provider": provider.name, "model": model, "key_index": index},
                errors,
                False,
            )

        detail = response.text[:400]
        errors.append(f"{provider.name} key#{index} HTTP {response.status_code}: {detail}")
        if response.status_code in _FATAL_STATUSES:
            # The request is wrong for this provider (an unsupported
            # response_format, say). Another key cannot help; another provider
            # might, so stop here rather than raising.
            return None, errors, True
        # 429 rate-limit / 401-403 dead key / 5xx transient -> next key.
        log.warning(
            "llm key failed over",
            extra={"provider": provider.name, "key_index": index, "status": response.status_code},
        )

    return None, errors, False


async def chat(body: dict, *, model_kind: str | None = None, task: str = "chat") -> dict:
    """POST to the first provider that answers, rotating keys within each.

    `model_kind` picks the model per provider ("chat" / "vision"). Leave it None
    to honour whatever `model` the caller put in the body — names are not
    portable between providers, so a caller-pinned model pins the provider too.
    """
    if not body.get("messages"):
        raise BadRequest("Missing 'messages' in the LLM request body.")

    body = dict(body)
    # /chat here is request/response; streaming would need a different transport
    # all the way through the gateway, so it is explicitly disabled.
    body["stream"] = False

    chain = config.providers()
    if not chain:
        raise ApiError(
            "No LLM provider is configured on the cpu pod. Set DASHSCOPE_API_KEYS "
            "(and/or GROQ_API_KEYS) plus LLM_PROVIDERS.",
            status=503,
            code="not_configured",
        )

    # A caller-supplied model name only exists on one provider, so do not shop it
    # around the chain and get a confusing 404 from the second one.
    if model_kind is None and body.get("model"):
        chain = chain[:1]

    errors: list[str] = []
    fatal = False
    for provider in chain:
        result, provider_errors, provider_fatal = await _call_provider(provider, body, model_kind, task)
        errors.extend(provider_errors)
        if result is not None:
            if provider is not chain[0]:
                log.warning(
                    "llm served by failover provider",
                    extra={"provider": provider.name, "task": task, "skipped": len(errors)},
                )
            return result
        fatal = fatal or provider_fatal

    if fatal:
        raise BadRequest("Every provider rejected the request.", detail=errors)
    raise UpstreamError("All LLM providers failed or are rate-limited.", detail=errors)


async def health() -> dict:
    chain = config.providers()
    return {
        "providers": [
            {"name": p.name, "keys": len(keys(p)), "model": p.model, "vision_model": p.vision_model}
            for p in chain
        ],
        "order": config.LLM_PROVIDER_ORDER,
    }
