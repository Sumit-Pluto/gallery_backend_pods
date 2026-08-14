"""Health, readiness and warmup behaviour from crm_common.service.

The warmup tests exist because of a real failure found while running the cpu
image: `import rembg` pulls in pymatting -> numba, which JIT-compiles at import
time and pegged every core for over 20 minutes on a cold cache. Throughout that,
/readyz reported `{"ready": false, "error": null}` — indistinguishable from a
pod that is merely slow. A stalled pod must be able to say so.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from crm_common import service
from crm_common.service import Readiness, create_app


def wait_for(client, predicate, *, timeout: float = 5.0, interval: float = 0.05):
    """Poll /readyz until `predicate(body)` holds, or give up after `timeout`.

    Deliberately wall-clock bounded rather than a fixed iteration count. An
    earlier version spun N times with no delay, which passed locally and failed
    in CI: with uvloop installed the polls completed faster than the warmup
    timeout being tested, so the condition had not had time to occur yet. Any
    "loop N times and hope" assertion is a race waiting for a faster machine.
    """
    deadline = time.monotonic() + timeout
    body = None
    while time.monotonic() < deadline:
        body = client.get("/readyz").json()
        if predicate(body):
            return body
        time.sleep(interval)
    return body


def test_healthz_is_up_before_readyz():
    """Liveness must not wait on warmup, or an orchestrator kills a loading pod."""
    readiness = Readiness()

    async def slow_warmup():
        await asyncio.sleep(30)

    app = create_app("test-slow", readiness=readiness, warmup=slow_warmup, internal_auth=False)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["ready"] is False


def test_readyz_reports_200_once_warm():
    readiness = Readiness()

    async def warmup():
        readiness.mark_ready(loaded=["thing"])

    app = create_app("test-warm", readiness=readiness, warmup=warmup, internal_auth=False)
    with TestClient(app) as client:
        body = wait_for(client, lambda b: b.get("ready") is True)
    assert body["ready"] is True
    assert body["loaded"] == ["thing"]
    assert body["warmup_s"] is not None


def test_a_hung_warmup_eventually_reports_an_error(monkeypatch):
    """The bug this guards: ready=false, error=null, forever."""
    monkeypatch.setattr(service, "WARMUP_TIMEOUT", 0.3)
    monkeypatch.setattr(service, "WARMUP_HEARTBEAT", 0.1)
    readiness = Readiness()

    async def never_finishes():
        await asyncio.sleep(60)

    app = create_app("test-hung", readiness=readiness, warmup=never_finishes, internal_auth=False)
    with TestClient(app) as client:
        # Must outlast WARMUP_TIMEOUT above, with room for a slow runner.
        body = wait_for(client, lambda b: b.get("error"), timeout=10.0)
    assert body["ready"] is False
    assert body["error"] is not None, "a hung warmup must surface an error, not sit at error=null"
    assert "exceeded" in body["error"]


def test_a_failing_warmup_surfaces_the_exception():
    readiness = Readiness()

    async def broken():
        raise RuntimeError("model file is corrupt")

    app = create_app("test-broken", readiness=readiness, warmup=broken, internal_auth=False)
    with TestClient(app) as client:
        body = wait_for(client, lambda b: b.get("error"))
    assert body["error"] == "RuntimeError: model file is corrupt"


# --------------------------------------------------------------------------- #
# Internal auth — the backend pods' only protection, since proxy URLs are public
# --------------------------------------------------------------------------- #


def test_internal_auth_rejects_a_caller_without_the_shared_secret(monkeypatch):
    monkeypatch.setattr(service, "ALLOW_INSECURE", False)
    monkeypatch.setenv("INTERNAL_API_KEY", "shared-secret")

    app = create_app("test-internal", internal_auth=True)

    @app.post("/work")
    async def work():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200          # probes stay open
        assert client.post("/work").status_code == 401
        assert client.post("/work", headers={"X-Internal-Key": "wrong"}).status_code == 401
        ok = client.post("/work", headers={"X-Internal-Key": "shared-secret"})
        assert ok.status_code == 200


def test_request_id_is_returned_even_when_auth_rejects(monkeypatch):
    """Observability must wrap auth, not sit inside it.

    Starlette runs the LAST-registered middleware outermost, so this asserts the
    registration order in create_app has not been flipped back.
    """
    monkeypatch.setattr(service, "ALLOW_INSECURE", False)
    monkeypatch.setenv("INTERNAL_API_KEY", "shared-secret")

    app = create_app("test-order", internal_auth=True)

    @app.post("/work")
    async def work():
        return {"ok": True}

    with TestClient(app) as client:
        rejected = client.post("/work")
    assert rejected.status_code == 401
    assert rejected.headers.get("X-Request-ID"), "a rejected request must still be traceable"


def test_body_size_limit_rejects_before_parsing(monkeypatch):
    monkeypatch.setattr(service, "MAX_BODY_BYTES", 100)
    app = create_app("test-size", internal_auth=False)

    @app.post("/work")
    async def work(payload: dict):
        return payload

    with TestClient(app) as client:
        response = client.post("/work", json={"blob": "x" * 500})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
