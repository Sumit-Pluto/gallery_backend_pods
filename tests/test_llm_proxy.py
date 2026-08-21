"""Provider failover.

Two nested loops, and the nesting is the whole design:

  within a provider   keys rotate      — one key rate-limited, try the next
  across providers    the chain moves  — provider exhausted, try the next one

That second loop is what replaced the automatic YOLO fallback. It matters that it
fails over *without changing the shape of the answer*, which YOLO could not do.

These drive coroutines with a plain `asyncio.run` rather than an async-test
plugin: the subject here is ordering, not concurrency, and one less plugin in the
CI dependency chain is one less way for a green pipeline to turn red without a
commit of ours.
"""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest

from test_detect_vlm import _load_cpu_app  # registers the aliased package

_load_cpu_app()
llm_proxy = importlib.import_module("cpu_app.llm_proxy")
cpu_config = importlib.import_module("cpu_app.config")

BODY = {"messages": [{"role": "user", "content": "hi"}]}


def run(coro):
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeClient:
    """Records every call and replays a scripted set of responses."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "body": json, "auth": (headers or {}).get("Authorization")})
        result = self.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


OK = FakeResponse(200, {"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 12}})


@pytest.fixture
def two_providers(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEYS", "ds-1,ds-2")
    monkeypatch.setenv("GROQ_API_KEYS", "groq-1")
    monkeypatch.setattr(cpu_config, "LLM_PROVIDER_ORDER", ["dashscope", "groq"])
    # Cursors persist across calls by design; reset so tests do not leak state.
    llm_proxy._cursors.clear()
    yield
    llm_proxy._cursors.clear()


def _install(monkeypatch, script):
    fake = FakeClient(list(script))
    monkeypatch.setattr(llm_proxy, "client", lambda provider: fake)
    return fake


def test_first_provider_wins_when_healthy(two_providers, monkeypatch):
    fake = _install(monkeypatch, [OK])
    result = run(llm_proxy.chat(BODY, model_kind="vision"))
    assert result["provider"] == "dashscope"
    assert result["model"] == cpu_config.DASHSCOPE.vision_model
    assert len(fake.calls) == 1


def test_keys_rotate_within_a_provider(two_providers, monkeypatch):
    """A 429 on one key must not cost the request — just the next key."""
    fake = _install(monkeypatch, [FakeResponse(429, text="rate limited"), OK])
    result = run(llm_proxy.chat(BODY, model_kind="vision"))
    assert result["provider"] == "dashscope"
    assert [c["auth"] for c in fake.calls] == ["Bearer ds-1", "Bearer ds-2"]


def test_chain_advances_when_a_provider_is_exhausted(two_providers, monkeypatch):
    """Every Model Studio key throttled -> Groq answers. Same response shape,
    which is exactly what the YOLO fallback could never guarantee."""
    fake = _install(monkeypatch, [FakeResponse(429), FakeResponse(429), OK])
    result = run(llm_proxy.chat(BODY, model_kind="vision"))
    assert result["provider"] == "groq"
    assert result["model"] == cpu_config.GROQ.vision_model
    assert len(fake.calls) == 3


def test_a_fatal_status_stops_key_burn_but_still_tries_the_next_provider(two_providers, monkeypatch):
    """400 usually means an unsupported response_format. No other key of the same
    provider will fix it; a different provider might."""
    fake = _install(monkeypatch, [FakeResponse(400, text="unsupported response_format"), OK])
    result = run(llm_proxy.chat(BODY, model_kind="vision"))
    assert result["provider"] == "groq"
    # Only ONE dashscope key was spent, not both.
    assert [c["auth"] for c in fake.calls] == ["Bearer ds-1", "Bearer groq-1"]


def test_all_providers_rejecting_raises_bad_request(two_providers, monkeypatch):
    from crm_common.errors import BadRequest

    _install(monkeypatch, [FakeResponse(400), FakeResponse(400)])
    with pytest.raises(BadRequest):
        run(llm_proxy.chat(BODY, model_kind="vision"))


def test_all_providers_throttled_raises_upstream_error(two_providers, monkeypatch):
    from crm_common.errors import UpstreamError

    _install(monkeypatch, [FakeResponse(429), FakeResponse(429), FakeResponse(429)])
    with pytest.raises(UpstreamError):
        run(llm_proxy.chat(BODY, model_kind="vision"))


def test_a_caller_pinned_model_pins_the_provider(two_providers, monkeypatch):
    """Model names are not portable: shopping `qwen-vl-plus` around the chain
    would just earn a confusing 404 from the second provider."""
    from crm_common.errors import UpstreamError

    # Two dashscope keys, so a 5xx on the first rotates to the second — and stops.
    fake = _install(monkeypatch, [FakeResponse(500), FakeResponse(500)])
    with pytest.raises(UpstreamError):
        run(llm_proxy.chat({**BODY, "model": "qwen-vl-plus"}))
    assert len(fake.calls) == 2  # both dashscope keys, never groq
    assert all(c["auth"].startswith("Bearer ds-") for c in fake.calls)


def test_missing_messages_fails_before_spending_a_key(two_providers, monkeypatch):
    from crm_common.errors import BadRequest

    fake = _install(monkeypatch, [OK])
    with pytest.raises(BadRequest):
        run(llm_proxy.chat({"model": "x"}))
    assert fake.calls == []


def test_no_configured_provider_is_a_503(monkeypatch):
    from crm_common.errors import ApiError

    for name in ("DASHSCOPE_API_KEYS", "DASHSCOPE_API_KEY", "GROQ_API_KEYS", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert cpu_config.providers() == []

    with pytest.raises(ApiError) as excinfo:
        run(llm_proxy.chat(BODY, model_kind="vision"))
    assert excinfo.value.status == 503
    assert excinfo.value.code == "not_configured"
