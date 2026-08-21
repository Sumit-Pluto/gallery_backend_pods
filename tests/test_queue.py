"""Queue admission, fairness, and the retry rule.

These cover the failure modes that only appear with more than one user, which is
exactly the set that a single-developer smoke test will never surface.
"""

from __future__ import annotations

import asyncio

import pytest

from app import config, jobs


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    store = jobs.JobStore()
    monkeypatch.setattr(jobs, "store", store)
    jobs._durations.clear()
    yield store


async def _noop():
    return {"ok": True}


def test_queue_full_rejects_immediately_with_a_retry_hint(_fresh_store, monkeypatch):
    """Fail fast, and say how long to wait.

    The old default accepted 64 jobs against one GPU. Job 64 would have waited
    half an hour, except the pod's 300 s semaphore killed it first — so callers
    waited five minutes and *then* got an error.
    """
    monkeypatch.setattr(config, "JOB_MAX_QUEUED", 3)

    async def main():
        for _ in range(3):
            await _fresh_store.submit("edit", "keyA", _noop)
        with pytest.raises(Exception) as excinfo:
            await _fresh_store.submit("edit", "keyA", _noop)
        exc = excinfo.value
        assert exc.status == 503 and exc.code == "queue_full"
        # A number to back off by, not a guess.
        assert exc.detail["retry_after_s"] >= 1

    asyncio.run(main())


def test_queued_jobs_report_position_and_eta(_fresh_store, monkeypatch):
    monkeypatch.setattr(config, "JOB_MAX_QUEUED", 10)
    monkeypatch.setattr(config, "JOB_ASSUMED_DURATION", 30.0)
    monkeypatch.setattr(jobs, "_DEFAULT_DURATION", 30.0)

    async def main():
        first = await _fresh_store.submit("edit", "keyA", _noop)
        second = await _fresh_store.submit("edit", "keyA", _noop)
        third = await _fresh_store.submit("edit", "keyA", _noop)

        # Nothing is dispatched (no dispatcher running), so positions are 0,1,2.
        assert _fresh_store.get(first.id).public()["position"] == 0
        body = _fresh_store.get(third.id).public()
        assert body["position"] == 2
        # A spinner with no number is indistinguishable from a hang.
        assert body["eta_seconds"] == 60

    asyncio.run(main())


def test_dispatch_is_fair_across_users(_fresh_store, monkeypatch):
    """One user bulk-editing must not put everyone else behind all of their work.

    Under plain FIFO, userA's 5 jobs would all precede userB's single one.
    """
    monkeypatch.setattr(config, "JOB_MAX_QUEUED", 20)

    async def main():
        for _ in range(5):
            await _fresh_store.submit("edit", "shared-key", _noop, fair_key="userA")
        late = await _fresh_store.submit("edit", "shared-key", _noop, fair_key="userB")

        # Nothing running yet, so the oldest wins: userA goes first.
        assert _fresh_store._next_id() == _fresh_store._queue[0]

        # Once userA holds the slot, the next turn belongs to userB — not to
        # userA's four remaining jobs.
        running = _fresh_store._queue.popleft()
        _fresh_store._running.add(running)
        assert _fresh_store._next_id() == late.id

    asyncio.run(main())


def test_fairness_falls_back_to_the_api_key(_fresh_store, monkeypatch):
    """A client that does not send X-End-User still works — it just shares one
    bucket, which is where we started."""
    monkeypatch.setattr(config, "JOB_MAX_QUEUED", 10)

    async def main():
        job = await _fresh_store.submit("edit", "keyA", _noop)
        assert job.fair_key == "keyA"

    asyncio.run(main())


def test_finished_jobs_drop_their_payload(_fresh_store, monkeypatch):
    """The closure captures a base64 image. Holding it for the full 30-minute
    TTL is a slow leak that ends with the pod OOM-killed."""
    monkeypatch.setattr(config, "JOB_MAX_QUEUED", 10)

    async def main():
        job = await _fresh_store.submit("edit", "keyA", _noop)
        await _fresh_store._run(job)
        assert job.status == "done"
        assert job.coro_factory is None

    asyncio.run(main())


def test_completed_durations_feed_the_eta(_fresh_store, monkeypatch):
    """ETA should learn from reality rather than stay on the seeded guess."""
    monkeypatch.setattr(config, "JOB_MAX_QUEUED", 10)

    async def main():
        job = await _fresh_store.submit("edit", "keyA", _noop)
        await _fresh_store._run(job)

    asyncio.run(main())
    assert len(jobs._durations) == 1
    assert jobs.average_duration() == jobs._durations[0]
