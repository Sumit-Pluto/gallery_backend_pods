"""Async job queue for the slow work.

FLUX and SDXL take 30-90 s. Holding an HTTP connection open that long fails in
every direction — client timeouts, proxy idle limits, a load balancer that gives
up at 60 s. So diffusion returns 202 + a job id immediately and the caller polls.

## Why the gateway does the queueing

The obvious implementation — accept a job, fire `asyncio.create_task`, let the
GPU pod's semaphore sort it out — was what this used to do, and it produced the
worst possible behaviour under load. Every accepted job hit the diffusion pod at
once and queued on `GPU_QUEUE_TIMEOUT` (300 s). At ~30 s a render only about ten
fit inside that window, so job eleven onward **waited five minutes and then
failed** with `gpu_busy`. A long wait that ends in an error is worse than an
immediate rejection, and the gateway had no idea any of it was happening: it
could not report a position, because it was not holding a queue.

So the queue lives here now. Jobs wait in this process, where their position is
known and reportable, and the dispatcher keeps only `JOB_DISPATCH_CONCURRENCY`
in flight at the pod — matched to the GPU's real capacity. The pod-side semaphore
becomes a backstop that should never fire rather than the thing everybody waits
on.

## Fairness

Dispatch is round-robin across owners, not FIFO. Under plain FIFO one user
submitting twenty edits puts nineteen other users behind all twenty of them; the
gallery is a multi-user product, so one person bulk-editing an album must not
mean everybody else waits. Within an owner it stays FIFO — your own edits still
come back in the order you asked for them.

## Scope

This store is in-process and single-replica, which matches the deployment: one
gateway pod. Two things follow, and both are deliberate:

  * a gateway restart loses in-flight jobs. The client sees the job id 404 and
    retries. That is acceptable for image edits and much simpler than Redis.
  * if you ever scale the gateway to two replicas, a poll can land on the wrong
    one. At that point swap this class for Redis — the interface is small enough
    that nothing above it changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from crm_common.errors import ApiError

from . import config

log = logging.getLogger(__name__)

# Rolling window of recent render durations, used to turn a queue position into
# an ETA. Seeded with a sane guess so the very first caller is not told "unknown".
_durations: deque[float] = deque(maxlen=20)
_DEFAULT_DURATION = float(config.JOB_ASSUMED_DURATION)


def average_duration() -> float:
    if not _durations:
        return _DEFAULT_DURATION
    return sum(_durations) / len(_durations)


@dataclass
class Job:
    id: str
    kind: str
    owner: str
    # Fairness key: the end user behind the shared API key, when the caller
    # identifies one. Queue turns are taken per fair_key, while `owner` remains
    # the authorisation boundary — see auth.py.
    fair_key: str
    coro_factory: Callable[[], Awaitable[dict]] | None = None
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: dict | None = None
    # Filled in by the store at read time; not part of the job's own state.
    position: int | None = None

    def public(self) -> dict:
        body = {
            "job_id": self.id,
            "status": self.status,
            "created_at": round(self.created_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
        }
        if self.status == "queued":
            # A spinner with no information is the difference between "this is
            # taking a while" and "this is broken". Position and ETA cost us
            # nothing and answer both.
            body["position"] = self.position
            body["eta_seconds"] = (
                round((self.position or 0) * average_duration()) if self.position is not None else None
            )
        if self.status == "running":
            body["eta_seconds"] = max(
                0, round(average_duration() - (time.time() - (self.started_at or time.time())))
            )
        if self.status == "done":
            body["result"] = self.result
        if self.status == "error":
            body["error"] = self.error
        return body


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        # Insertion-ordered queue of job ids awaiting dispatch.
        self._queue: deque[str] = deque()
        self._running: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._lock: asyncio.Lock | None = None
        self._wakeup: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- internals ---------------------------------------------------------- #

    def _bind(self) -> None:
        """(Re)bind loop-affine primitives to the loop that is actually running.

        `asyncio.Event` and `asyncio.Lock` attach to the loop that first awaits
        them, and this store is a module-level singleton. So anything that
        starts a fresh loop — a test client, a restarted worker — leaves the
        cached objects owned by a loop that no longer exists, and the next await
        dies with "got Future attached to a different loop".

        Rebinding on loop change also has to drop `_running`: those jobs were
        `asyncio.Task`s on the dead loop and can never complete, so leaving them
        counted would permanently consume dispatch slots and wedge the queue.
        """
        loop = asyncio.get_running_loop()
        if self._loop is loop and self._lock is not None and self._wakeup is not None:
            return
        self._loop = loop
        self._lock = asyncio.Lock()
        self._wakeup = asyncio.Event()
        self._running.clear()
        self._tasks.clear()

    def _event(self) -> asyncio.Event:
        self._bind()
        assert self._wakeup is not None
        return self._wakeup

    def _mutex(self) -> asyncio.Lock:
        self._bind()
        assert self._lock is not None
        return self._lock

    def _depth(self) -> int:
        return len(self._queue) + len(self._running)

    def _next_id(self) -> str | None:
        """Round-robin across owners; FIFO within one.

        Picks the oldest queued job belonging to whichever fair_key has the
        fewest jobs already running, so a single caller cannot monopolise the
        GPU by submitting in bulk.
        """
        if not self._queue:
            return None
        running_by_key: dict[str, int] = {}
        for jid in self._running:
            job = self._jobs.get(jid)
            if job:
                running_by_key[job.fair_key] = running_by_key.get(job.fair_key, 0) + 1
        best_id, best_rank = None, None
        for jid in self._queue:
            job = self._jobs.get(jid)
            if job is None:
                continue
            rank = running_by_key.get(job.fair_key, 0)
            # deque order is submission order, so the first job at the lowest
            # rank is also the oldest one for that owner — FIFO within a key.
            if best_rank is None or rank < best_rank:
                best_id, best_rank = jid, rank
                if rank == 0:
                    break
        return best_id

    def _position_of(self, job: Job) -> int:
        """How many jobs stand between this one and the GPU.

        Submission order plus whatever is already running. Under round-robin the
        true turn order can differ — a bulk submitter's later jobs get overtaken
        — so treat this as an upper bound: callers may be served sooner than
        quoted, never later. An honest over-estimate is the right error to make
        when the number drives a progress message.
        """
        if job.status != "queued":
            return 0
        ahead = len(self._running)
        for jid in self._queue:
            if jid == job.id:
                break
            ahead += 1
        return ahead

    # -- public API --------------------------------------------------------- #

    async def submit(
        self,
        kind: str,
        owner: str,
        coro_factory: Callable[[], Awaitable[dict]],
        *,
        fair_key: str | None = None,
    ) -> Job:
        async with self._mutex():
            if self._depth() >= config.JOB_MAX_QUEUED:
                # Reject in milliseconds rather than after a long wait. Tell the
                # caller how long to wait so backoff is informed, not guessed.
                raise ApiError(
                    "The render queue is full; retry shortly.",
                    status=503,
                    code="queue_full",
                    detail={
                        "max_queued": config.JOB_MAX_QUEUED,
                        "retry_after_s": max(1, round(self._depth() * average_duration())),
                    },
                )
            job = Job(
                id=uuid.uuid4().hex,
                kind=kind,
                owner=owner,
                fair_key=fair_key or owner,
                coro_factory=coro_factory,
            )
            self._jobs[job.id] = job
            self._queue.append(job.id)
            job.position = self._position_of(job)
        self._event().set()
        return job

    async def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        factory = job.coro_factory
        try:
            if factory is None:
                raise RuntimeError("job has no work attached")
            job.result = await factory()
            job.status = "done"
            duration = time.time() - job.started_at
            _durations.append(duration)
            log.info(
                "job done",
                extra={"job_id": job.id, "kind": job.kind, "duration_ms": int(duration * 1000)},
            )
        except ApiError as exc:
            job.status = "error"
            job.error = exc.to_dict()["error"]
            log.warning("job failed", extra={"job_id": job.id, "kind": job.kind, "code": exc.code})
        except Exception as exc:
            job.status = "error"
            job.error = {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
            log.exception("job crashed", extra={"job_id": job.id, "kind": job.kind})
        finally:
            job.finished_at = time.time()
            # Drop the closure: it captures the request payload, which for an
            # edit is a base64 image. Holding it for the whole TTL is a slow
            # memory leak that ends with the pod OOM-killed at 3am.
            job.coro_factory = None
            self._running.discard(job.id)
            self._event().set()

    def get(self, job_id: str, owner: str | None = None) -> Job:
        job = self._jobs.get(job_id)
        # Scope by owner so one client's key cannot read another's render.
        if job is None or (owner is not None and job.owner != owner):
            raise ApiError("No such job (it may have expired).", status=404, code="job_not_found")
        if job.status == "queued":
            job.position = self._position_of(job)
        return job

    async def wait(self, job: Job, seconds: float) -> Job:
        """Block briefly so a fast render can answer inline. Best-effort."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and job.status in ("queued", "running"):
            await asyncio.sleep(0.25)
        return job

    def sweep(self) -> int:
        """Drop finished jobs past TTL, and the oldest if the store is oversized.

        Results carry base64 images, so an unbounded store is a slow memory leak
        that ends in the pod being OOM-killed at 3am.
        """
        now = time.time()
        expired = [
            jid for jid, job in self._jobs.items()
            if job.finished_at and now - job.finished_at > config.JOB_TTL_SECONDS
        ]
        for jid in expired:
            self._jobs.pop(jid, None)

        if len(self._jobs) > config.JOB_MAX_STORED:
            finished = sorted(
                (j for j in self._jobs.values() if j.finished_at),
                key=lambda j: j.finished_at or 0,
            )
            for job in finished[: len(self._jobs) - config.JOB_MAX_STORED]:
                self._jobs.pop(job.id, None)
        return len(expired)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for job in self._jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        return {
            "total": len(self._jobs),
            "queued_depth": len(self._queue),
            "running": len(self._running),
            "avg_duration_s": round(average_duration(), 1),
            **counts,
        }

    async def dispatcher(self) -> None:
        """Keep JOB_DISPATCH_CONCURRENCY jobs in flight, chosen fairly.

        Sized to the GPU's real capacity (one diffusion pod with
        GPU_CONCURRENCY=1 means 1). Raise it as you add pods; the pod-side
        semaphore then stays a backstop rather than the queue everyone waits in.
        """
        while True:
            try:
                event = self._event()
                if len(self._running) >= config.JOB_DISPATCH_CONCURRENCY or not self._queue:
                    event.clear()
                    await event.wait()
                    continue
                async with self._mutex():
                    if len(self._running) >= config.JOB_DISPATCH_CONCURRENCY:
                        continue
                    job_id = self._next_id()
                    if job_id is None:
                        continue
                    self._queue.remove(job_id)
                    self._running.add(job_id)
                    job = self._jobs[job_id]
                task = asyncio.create_task(self._run(job))
                # Hold a reference: a bare create_task can be garbage-collected
                # mid-flight.
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job dispatcher error")
                await asyncio.sleep(0.5)


store = JobStore()


async def sweeper() -> None:
    while True:
        try:
            await asyncio.sleep(60)
            removed = store.sweep()
            if removed:
                log.info("job sweep", extra={"removed": removed, **store.stats()})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("job sweep failed")
