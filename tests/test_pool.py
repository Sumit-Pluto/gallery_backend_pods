"""Bounded pools, and the scoping bug they exist to prevent.

The distinction under test is per-request vs per-process. On a pod there is one
long-lived process serving every user, so a limit constructed inside a request
handler limits that one request and nothing else — which is not a limit at all.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time

import pytest

from crm_common import pool
from crm_common.errors import ApiError

from test_detect_vlm import _load_cpu_app

_load_cpu_app()
detect_vlm = importlib.import_module("cpu_app.detect_vlm")


def test_pool_never_exceeds_its_worker_count():
    """The whole point: N workers means at most N run at once, no matter how
    many callers arrive together."""
    p = pool.BoundedPool("t", workers=3, queue_timeout=5)
    live = 0
    peak = 0
    guard = threading.Lock()

    def work():
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with guard:
            live -= 1

    async def main():
        await asyncio.gather(*(p.run(work) for _ in range(20)))

    asyncio.run(main())
    assert peak <= 3
    assert p.stats()["completed"] == 20


def test_pool_rejects_rather_than_queueing_forever():
    """Fail fast beats a long wait that fails anyway: a caller told 'busy' in
    milliseconds can retry, one that waits minutes has wasted the slot too."""
    p = pool.BoundedPool("t", workers=1, queue_timeout=0.05)

    async def main():
        slow = asyncio.create_task(p.run(time.sleep, 0.5))
        await asyncio.sleep(0.01)
        with pytest.raises(ApiError) as excinfo:
            await p.run(lambda: None)
        assert excinfo.value.status == 503
        assert excinfo.value.code == "pool_busy"
        await slow

    asyncio.run(main())
    assert p.stats()["rejected"] == 1


def test_pool_releases_its_slot_when_work_raises():
    """A leaked semaphore slot is a pod that gets slower every hour and recovers
    only on restart."""
    p = pool.BoundedPool("t", workers=1, queue_timeout=1)

    def boom():
        raise ValueError("nope")

    async def main():
        for _ in range(3):
            with pytest.raises(ValueError):
                await p.run(boom)
        await p.run(lambda: "fine")  # slot still available

    asyncio.run(main())
    assert p.stats()["failed"] == 3


def test_batch_gate_is_shared_across_requests():
    """Regression: the batch semaphore was built inside detect_batch(), so ten
    concurrent batches each got their own allowance and put 10x the intended
    calls in flight — the exact rate-limit stampede the bound was added to stop.
    """
    detect_vlm._batch_semaphore = None

    async def main():
        first = detect_vlm._batch_slot()
        second = detect_vlm._batch_slot()
        # Same object, so the budget is the process's, not the request's.
        assert first is second

    asyncio.run(main())
    detect_vlm._batch_semaphore = None


# --------------------------------------------------------------------------- #
# Core detection
# --------------------------------------------------------------------------- #


def test_cores_reads_the_cgroup_quota_not_the_host(tmp_path, monkeypatch):
    """Regression: a 4 vCPU pod reported 192 workers.

    os.cpu_count() answers for the machine, not the container. Sizing a pool off
    it is worse than not bounding at all — every thread gets admitted and then
    spends its life context-switching.
    """
    cgroup = tmp_path / "cpu.max"
    cgroup.write_text("400000 100000")  # 4 CPUs

    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return real_open(cgroup, *args, **kwargs)
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(pool.os, "cpu_count", lambda: 192)
    assert pool._cores() == 4


def test_cores_falls_back_when_no_quota_is_set(monkeypatch):
    """An unrestricted cgroup must still not hand back the host's core count."""

    def fake_open(path, *args, **kwargs):
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(pool.os, "sched_getaffinity", lambda pid: set(range(192)), raising=False)
    # Clamped, not 192.
    assert pool._cores() == 16
