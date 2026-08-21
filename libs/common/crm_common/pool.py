"""Bounded thread pools for blocking work on the CPU services.

`asyncio.to_thread` is the obvious way to keep a blocking call off the event
loop, and on a pod it is a trap. It dispatches to the interpreter's *default*
executor — `min(32, cpu_count + 4)` threads, 8 on a 4-core pod — which is shared
by every caller in the process. Two consequences, both of which only show up
under the concurrency a long-lived pod actually sees:

  * **No admission control.** Fifty concurrent background-removal requests do not
    fail fast; they queue on those 8 threads, each holding a fully decoded image
    in memory while it waits. The pod degrades instead of shedding load.
  * **Cross-starvation.** A 32-image detection batch floods the same pool that
    rembg and OCR need, so unrelated endpoints stall on each other. Serverless
    hid this by giving every worker its own process.

This is the CPU-side counterpart to `crm_common.gpu.run_exclusive`: a named pool
per workload class, each with its own thread budget and its own queue timeout, so
one workload can saturate without taking the others down with it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger(__name__)


class BoundedPool:
    """A named thread pool with a queue depth limit and a wait timeout.

    Rejecting at the door beats queueing forever: a caller that is told "busy"
    in 50 ms can retry, while one that waits three minutes and *then* fails has
    wasted everybody's time — the client's, and the slot it was holding.
    """

    def __init__(self, name: str, workers: int, *, queue_timeout: float = 30.0):
        self.name = name
        self.workers = max(1, workers)
        self.queue_timeout = queue_timeout
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=name)
        self._semaphore: asyncio.Semaphore | None = None
        self._stats = {"in_flight": 0, "waiting": 0, "completed": 0, "failed": 0, "rejected": 0}

    def _sem(self) -> asyncio.Semaphore:
        # Built lazily: a Semaphore binds to the running loop, and these are
        # constructed at import time, before uvicorn has one.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.workers)
        return self._semaphore

    def stats(self) -> dict:
        return dict(self._stats, workers=self.workers)

    async def run(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        from .errors import ApiError

        self._stats["waiting"] += 1
        queued_at = time.monotonic()
        try:
            await asyncio.wait_for(self._sem().acquire(), timeout=self.queue_timeout)
        except asyncio.TimeoutError as exc:
            self._stats["waiting"] -= 1
            self._stats["rejected"] += 1
            raise ApiError(
                f"The '{self.name}' worker pool is saturated; retry shortly.",
                status=503,
                code="pool_busy",
                detail={"pool": self.name, "workers": self.workers},
            ) from exc
        self._stats["waiting"] -= 1
        self._stats["in_flight"] += 1
        wait_ms = int((time.monotonic() - queued_at) * 1000)
        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))
            self._stats["completed"] += 1
            if wait_ms > 1000:
                # Only worth a line when the queue actually bit. Logging every
                # call on a busy pod is how you lose the signal.
                log.info(
                    "pool queue wait",
                    extra={"pool": self.name, "queue_ms": wait_ms,
                           "run_ms": int((time.monotonic() - started) * 1000)},
                )
            return result
        except Exception:
            self._stats["failed"] += 1
            raise
        finally:
            self._stats["in_flight"] -= 1
            self._sem().release()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _cores() -> int:
    """Cores this *container* may use — not the host's.

    `os.cpu_count()` reports the machine's cores, and a pod is a container on a
    much bigger machine. Measured on a 4 vCPU RunPod pod it returned **192**, so
    both pools were sized to 192 threads against 4 usable cores. That is worse
    than leaving the work unbounded: the threads all get admitted, then spend
    their time context-switching instead of finishing.

    cgroup quota is the authoritative answer inside a container, so ask that
    first and treat everything after it as a fallback.
    """
    # cgroup v2 — "<quota> <period>", or "max <period>" when unrestricted.
    try:
        with open("/sys/fs/cgroup/cpu.max") as handle:
            quota, period = handle.read().split()
        if quota != "max" and int(period) > 0:
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as handle:
            quota = int(handle.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as handle:
            period = int(handle.read())
        if quota > 0 and period > 0:
            return max(1, round(quota / period))
    except (OSError, ValueError):
        pass

    # No quota set. The affinity mask is the next best signal, and the host count
    # the last — both clamped, because an unclamped host count is exactly how
    # this pod ended up with 192.
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        available = os.cpu_count() or 4
    return max(1, min(16, available))


# Heavy, genuinely CPU-bound work: rembg, PP-OCR, ffmpeg denoise. Sized to the
# cores the pod actually has, because oversubscribing these makes every one of
# them slower rather than running more of them.
HEAVY = BoundedPool(
    "heavy",
    int(os.environ.get("HEAVY_POOL_WORKERS", str(_cores()))),
    queue_timeout=float(os.environ.get("HEAVY_POOL_TIMEOUT", "60")),
)

# Short image decode/downscale/encode ahead of a network call. Cheaper per item
# and latency-sensitive, so it gets its own budget — a detection batch must not
# be able to starve background removal, and vice versa.
PREP = BoundedPool(
    "prep",
    int(os.environ.get("PREP_POOL_WORKERS", str(max(2, _cores())))),
    queue_timeout=float(os.environ.get("PREP_POOL_TIMEOUT", "30")),
)


def stats() -> dict:
    return {"heavy": HEAVY.stats(), "prep": PREP.stats()}


def shutdown() -> None:
    HEAVY.shutdown()
    PREP.shutdown()
