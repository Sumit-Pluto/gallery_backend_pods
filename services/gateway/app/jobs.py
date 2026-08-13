"""Async job queue for the slow work.

FLUX and SDXL take 30-90 s. Holding an HTTP connection open that long fails in
every direction — client timeouts, proxy idle limits, a load balancer that gives
up at 60 s. So diffusion returns 202 + a job id immediately and the caller polls.

This store is in-process and single-replica, which matches the deployment: one
gateway pod. Two things follow from that, and both are deliberate:

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
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from crm_common.errors import ApiError

from . import config

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    owner: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: dict | None = None

    def public(self) -> dict:
        body = {
            "job_id": self.id,
            "status": self.status,
            "created_at": round(self.created_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
        }
        if self.status == "done":
            body["result"] = self.result
        if self.status == "error":
            body["error"] = self.error
        return body


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def _queued_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    async def submit(self, kind: str, owner: str, coro_factory: Callable[[], Awaitable[dict]]) -> Job:
        async with self._lock:
            if self._queued_count() >= config.JOB_MAX_QUEUED:
                raise ApiError(
                    "The render queue is full; retry shortly.",
                    status=503,
                    code="queue_full",
                    detail={"max_queued": config.JOB_MAX_QUEUED},
                )
            job = Job(id=uuid.uuid4().hex, kind=kind, owner=owner)
            self._jobs[job.id] = job

        task = asyncio.create_task(self._run(job, coro_factory))
        # Hold a reference: a bare create_task can be garbage-collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: Job, coro_factory: Callable[[], Awaitable[dict]]) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            job.result = await coro_factory()
            job.status = "done"
            log.info(
                "job done",
                extra={"job_id": job.id, "kind": job.kind,
                       "duration_ms": int((time.time() - job.started_at) * 1000)},
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

    def get(self, job_id: str, owner: str | None = None) -> Job:
        job = self._jobs.get(job_id)
        # Scope by owner so one client's key cannot read another's render.
        if job is None or (owner is not None and job.owner != owner):
            raise ApiError("No such job (it may have expired).", status=404, code="job_not_found")
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
        return {"total": len(self._jobs), **counts}


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
