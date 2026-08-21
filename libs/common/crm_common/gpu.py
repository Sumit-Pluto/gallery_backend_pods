"""Serialise GPU work and keep the event loop free.

Serverless gave us one in-flight request per worker for free. A pod does not: it
will happily accept 50 concurrent requests, run them all against one card, and
CUDA-OOM. Every GPU entrypoint therefore goes through `run_exclusive`, which

  1. takes a semaphore (default 1 concurrent job per card), and
  2. runs the blocking, GIL-holding inference in a worker thread so health checks
     and job polling still answer while a 90 s FLUX render is in flight.

`GPU_CONCURRENCY` can be raised for small models on a big card, but the default
of 1 is the setting that keeps production alive.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger(__name__)

GPU_CONCURRENCY = max(1, int(os.environ.get("GPU_CONCURRENCY", "1")))
GPU_QUEUE_TIMEOUT = float(os.environ.get("GPU_QUEUE_TIMEOUT", "300"))

_semaphore: asyncio.Semaphore | None = None
_executor = ThreadPoolExecutor(max_workers=GPU_CONCURRENCY, thread_name_prefix="gpu")
_stats = {"in_flight": 0, "queued": 0, "completed": 0, "failed": 0}


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(GPU_CONCURRENCY)
    return _semaphore


def stats() -> dict:
    return dict(_stats, concurrency=GPU_CONCURRENCY)


async def run_exclusive(fn: Callable[..., Any], *args, task: str = "gpu", **kwargs) -> Any:
    """Await `fn(*args, **kwargs)` with at most GPU_CONCURRENCY running at once."""
    from .errors import ApiError

    _stats["queued"] += 1
    queued_at = time.monotonic()
    try:
        await asyncio.wait_for(_sem().acquire(), timeout=GPU_QUEUE_TIMEOUT)
    except asyncio.TimeoutError as exc:
        _stats["queued"] -= 1
        raise ApiError(
            "GPU queue is saturated; retry shortly.", status=503, code="gpu_busy"
        ) from exc
    _stats["queued"] -= 1
    _stats["in_flight"] += 1
    wait_ms = int((time.monotonic() - queued_at) * 1000)
    started = time.monotonic()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))
        _stats["completed"] += 1
        log.info(
            "gpu task complete",
            extra={"task": task, "queue_ms": wait_ms, "run_ms": int((time.monotonic() - started) * 1000)},
        )
        return result
    except Exception:
        _stats["failed"] += 1
        log.exception("gpu task failed", extra={"task": task})
        raise
    finally:
        _stats["in_flight"] -= 1
        _sem().release()


def self_test() -> dict:
    """Prove the card can EXECUTE, not merely allocate.

    Loading a pipeline only allocates VRAM, and allocation succeeds on any GPU
    the driver enumerates. Kernel *execution* is what fails when the torch build
    carries no kernels for the device's compute capability — and warmup never
    ran a forward pass, so a pod could report `ready: true` on a card that then
    500s every single request.

    Measured for real: torch 2.6.0+cu124 on an RTX PRO 4000 (Blackwell, sm_120)
    loaded 16 GB of FLUX without complaint, passed readiness, and died at the
    first kernel launch. cu124 predates Blackwell. The failure surfaced as an
    opaque `internal_error` on a user's render instead of a refusal at boot.

    One small matmul forces a kernel launch, which is all it takes to catch it.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - CPU images
        return {"cuda": False}

    if not torch.cuda.is_available():
        return {"cuda": False}

    name = torch.cuda.get_device_name(0)
    capability = "%d.%d" % torch.cuda.get_device_capability(0)
    try:
        probe = torch.ones((8, 8), device="cuda")
        # .item() forces a synchronise, so an async kernel failure is raised
        # here rather than surfacing later on somebody's request.
        value = float((probe @ probe).sum().item())
        del probe
    except Exception as exc:
        raise RuntimeError(
            f"GPU '{name}' (sm_{capability.replace('.', '')}) cannot execute kernels from this "
            f"torch build ({getattr(torch, '__version__', '?')}): {type(exc).__name__}: {exc}. "
            "The usual cause is a card newer than the torch CUDA version — Blackwell needs "
            "cu128 or later, and this image ships cu124. Deploy on an Ada or Ampere card, "
            "or rebuild with a newer torch."
        ) from exc

    if value != 512.0:  # 8x8 of ones, squared, summed
        raise RuntimeError(f"GPU '{name}' returned {value} for a matmul that must equal 512.")

    return {"cuda": True, "device": name, "capability": capability,
            "torch": getattr(torch, "__version__", "?")}


def empty_cache() -> None:
    """Best-effort VRAM release. Safe to call when torch is absent."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch missing on CPU images
        pass


def vram_report() -> dict:
    """Used by /healthz so you can watch headroom without SSHing into the pod."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "total_gb": round(total / 1e9, 2),
            "free_gb": round(free / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
    except Exception as exc:  # pragma: no cover
        return {"cuda": False, "error": f"{type(exc).__name__}: {exc}"}
