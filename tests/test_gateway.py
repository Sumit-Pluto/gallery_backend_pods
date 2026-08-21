"""End-to-end gateway behaviour with the pods stubbed out.

Covers the things that are easy to get wrong and expensive to discover in
production: auth, per-key job isolation, the 202/poll contract, and the
detection fallback.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config, jobs, upstream
from app.main import app

KEY = "test-client-key"
OTHER_KEY = "other-client-key"
HEAD = {"X-API-Key": KEY}
IMG = "aGVsbG8="


@pytest.fixture(autouse=True)
def _stub_upstreams(monkeypatch):
    """Replace every pod call with a recorder. No network, no GPU."""
    calls: list[tuple[str, str, dict]] = []

    def make(name, result=None, fail=None):
        async def post(path, payload, **kwargs):
            calls.append((name, path, payload))
            if fail is not None:
                raise fail
            return result if result is not None else {"image": "cmVzdWx0", "width": 8, "height": 8}

        return post

    for name in ("vision", "diffusion", "cpu", "chat", "translate"):
        monkeypatch.setattr(upstream.BY_NAME[name], "post", make(name))
        monkeypatch.setattr(upstream.BY_NAME[name], "base_url", f"http://{name}.test")
    monkeypatch.setenv("CLIENT_API_KEYS", f"{KEY},{OTHER_KEY}")
    yield calls


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CLIENT_API_KEYS", f"{KEY},{OTHER_KEY}")
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


def test_health_is_public(client):
    assert client.get("/healthz").status_code == 200


def test_missing_key_is_rejected(client):
    r = client.post("/v1/image/upscale", json={"image": IMG, "scale": 2})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_wrong_key_is_rejected(client):
    r = client.post("/v1/image/upscale", json={"image": IMG}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_bearer_token_is_accepted_too(client):
    r = client.post("/v1/image/upscale", json={"image": IMG},
                    headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_unknown_op_is_a_422_not_a_gpu_call(client, _stub_upstreams):
    r = client.post("/v1/image/edit", json={"image": IMG, "op": {"type": "rm -rf"}}, headers=HEAD)
    assert r.status_code == 422
    assert _stub_upstreams == []


def test_unknown_top_level_field_is_rejected(client):
    r = client.post("/v1/image/upscale",
                    json={"image": IMG, "scale": 2, "surprise": 1}, headers=HEAD)
    assert r.status_code == 422


def test_every_error_uses_one_envelope(client):
    """docs/API.md promises {"error": {code, message}} for everything.

    FastAPI's defaults would otherwise emit a validation array and a bare
    "detail" string, forcing the client to write three parsers.
    """
    cases = [
        client.post("/v1/image/upscale", json={"image": IMG}),                       # 401
        client.post("/v1/image/edit", json={"image": IMG, "op": {"type": "x"}},
                    headers=HEAD),                                                   # 422
        client.get("/v1/jobs/nope", headers=HEAD),                                   # 404
        client.get("/v1/does-not-exist", headers=HEAD),                              # 404
    ]
    for response in cases:
        assert response.status_code >= 400
        body = response.json()
        assert "error" in body, f"{response.url} returned {body}"
        assert set(body["error"]) >= {"code", "message"}
        assert isinstance(body["error"]["message"], str)


# --------------------------------------------------------------------------- #
# Sync vs async
# --------------------------------------------------------------------------- #


def test_upscale_answers_inline(client, _stub_upstreams):
    r = client.post("/v1/image/edit", json={"image": IMG, "op": {"type": "upscale", "factor": 4}},
                    headers=HEAD)
    assert r.status_code == 200
    assert r.json()["image"] == "cmVzdWx0"
    assert _stub_upstreams[0][0] == "vision"


def test_diffusion_returns_202_then_the_job_completes(client):
    r = client.post("/v1/image/edit", json={"image": IMG, "op": {"type": "colorize"}}, headers=HEAD)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    assert body["poll"].endswith(f"/v1/jobs/{job_id}")

    for _ in range(50):
        poll = client.get(f"/v1/jobs/{job_id}", headers=HEAD)
        assert poll.status_code == 200
        if poll.json()["status"] in ("done", "error"):
            break
    assert poll.json()["status"] == "done"
    assert poll.json()["result"]["image"] == "cmVzdWx0"


def test_wait_parameter_can_answer_inline(client):
    r = client.post("/v1/image/edit?wait=5", json={"image": IMG, "op": {"type": "colorize"}},
                    headers=HEAD)
    assert r.status_code == 200
    assert r.json()["image"] == "cmVzdWx0"


def test_a_job_is_scoped_to_the_key_that_created_it(client):
    r = client.post("/v1/image/edit", json={"image": IMG, "op": {"type": "colorize"}}, headers=HEAD)
    job_id = r.json()["job_id"]
    stolen = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": OTHER_KEY})
    assert stolen.status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/v1/jobs/deadbeef", headers=HEAD).status_code == 404


# --------------------------------------------------------------------------- #
# Detection routing
# --------------------------------------------------------------------------- #


def test_detect_prefers_the_vlm_on_the_cpu_pod(client, _stub_upstreams, monkeypatch):
    monkeypatch.setattr(config, "DETECT_BACKEND", "vlm")
    client.post("/v1/vision/detect", json={"image": IMG}, headers=HEAD)
    assert _stub_upstreams[0][0] == "cpu"


def test_detect_backend_query_forces_yolo(client, _stub_upstreams):
    client.post("/v1/vision/detect?backend=yolo", json={"image": IMG}, headers=HEAD)
    assert _stub_upstreams[0][0] == "vision"


def test_detect_falls_back_to_yolo_when_groq_fails(client, monkeypatch):
    from crm_common.errors import UpstreamError

    seen = []

    async def failing_cpu(path, payload, **kwargs):
        seen.append("cpu")
        raise UpstreamError("all keys rate limited")

    async def ok_vision(path, payload, **kwargs):
        seen.append("vision")
        return {"detections": [], "source": "yolo"}

    monkeypatch.setattr(upstream.cpu, "post", failing_cpu)
    monkeypatch.setattr(upstream.vision, "post", ok_vision)
    monkeypatch.setattr(config, "DETECT_FALLBACK_TO_YOLO", True)

    r = client.post("/v1/vision/detect", json={"image": IMG}, headers=HEAD)
    assert r.status_code == 200
    assert seen == ["cpu", "vision"]


def test_forced_backend_does_not_silently_fall_back(client, monkeypatch):
    from crm_common.errors import UpstreamError

    async def failing(path, payload, **kwargs):
        raise UpstreamError("down")

    monkeypatch.setattr(upstream.cpu, "post", failing)
    r = client.post("/v1/vision/detect?backend=vlm", json={"image": IMG}, headers=HEAD)
    assert r.status_code == 502


def test_detect_does_not_fall_back_to_yolo_by_default(client, monkeypatch):
    """The automatic fallback is off now: YOLO answers a narrower question."""
    from crm_common.errors import UpstreamError

    seen = []

    async def failing_cpu(path, payload, **kwargs):
        seen.append("cpu")
        raise UpstreamError("all providers rate limited")

    async def ok_vision(path, payload, **kwargs):
        seen.append("vision")
        return {"detections": [], "source": "yolo"}

    monkeypatch.setattr(upstream.cpu, "post", failing_cpu)
    monkeypatch.setattr(upstream.vision, "post", ok_vision)
    monkeypatch.setattr(config, "DETECT_FALLBACK_TO_YOLO", False)

    r = client.post("/v1/vision/detect", json={"image": IMG}, headers=HEAD)
    assert r.status_code == 502
    assert seen == ["cpu"]


def test_detect_never_falls_back_on_saturation(client, monkeypatch):
    """A rate limit means 'retry shortly', not 'spend the GPU'.

    Redirecting a throttled bulk upload onto the vision pod's single GPU slot is
    how one provider hiccup stalls upscale and transcribe too.
    """
    from crm_common.errors import RateLimited

    seen = []

    async def throttled_cpu(path, payload, **kwargs):
        seen.append("cpu")
        raise RateLimited("slow down")

    async def ok_vision(path, payload, **kwargs):
        seen.append("vision")
        return {"detections": [], "source": "yolo"}

    monkeypatch.setattr(upstream.cpu, "post", throttled_cpu)
    monkeypatch.setattr(upstream.vision, "post", ok_vision)
    monkeypatch.setattr(config, "DETECT_FALLBACK_TO_YOLO", True)  # even when enabled

    r = client.post("/v1/vision/detect", json={"image": IMG}, headers=HEAD)
    assert r.status_code == 429
    assert seen == ["cpu"]


def test_detect_batch_routes_to_the_cpu_pod(client, monkeypatch):
    async def cpu_batch(path, payload, **kwargs):
        assert path == "/detect-batch"
        return {
            "results": [
                {"index": 0, "detections": [{"name": "hard hat", "confidence": 0.9}]},
                {"index": 1, "error": {"code": "bad_request", "message": "not an image"}},
            ],
            "ok": 1,
            "failed": 1,
        }

    monkeypatch.setattr(upstream.cpu, "post", cpu_batch)
    r = client.post("/v1/vision/detect-batch", json={"images": [IMG, IMG]}, headers=HEAD)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] == 1 and body["failed"] == 1
    # A box is no longer required — the gallery only ever consumed labels.
    assert body["results"][0]["detections"][0]["box"] is None


def test_detect_batch_rejects_an_oversized_batch(client, monkeypatch):
    """Reject at the gateway rather than fanning out past the pod's semaphore."""
    monkeypatch.setattr(config, "DETECT_MAX_BATCH", 2)
    r = client.post("/v1/vision/detect-batch", json={"images": [IMG] * 3}, headers=HEAD)
    assert r.status_code == 400
    assert r.json()["error"]["detail"]["max_batch"] == 2


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


def test_rate_limit_returns_429_once_the_burst_is_spent(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(config, "RATE_LIMIT_BURST", 2)
    from app import auth

    auth._buckets.clear()

    codes = [
        client.post("/v1/image/upscale", json={"image": IMG}, headers=HEAD).status_code
        for _ in range(4)
    ]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]
