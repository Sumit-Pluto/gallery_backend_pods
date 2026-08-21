"""Object detection.

The only automatic path is the Qwen vision model on the cpu pod: open
vocabulary, label-only, no GPU cost, and redundant across providers (Model
Studio -> Groq) inside the cpu pod itself.

YOLO is **retained but not automatic**. It stays wired so a future
construction-specific need — tight boxes, a trained class list, counting
instances — can turn it back on without re-plumbing anything, and it is still
reachable right now with `?backend=yolo`. What changed is that it is no longer a
silent fallback, because:

  * it answers a narrower question. A fixed class list cannot produce the
    open-vocabulary labels the gallery indexes, so falling back to it quietly
    changed what got tagged.
  * it runs on the vision pod's single GPU semaphore, shared with upscale and
    transcribe. Redirecting a rate-limited bulk upload onto that queue is how one
    provider hiccup becomes a stalled GPU pod.

Set DETECT_FALLBACK_TO_YOLO=true to restore the old behaviour; even then, rate
limits and saturation never trigger it (see DETECT_NO_FALLBACK_CODES).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from crm_common.errors import ApiError, BadRequest
from crm_common.schemas import DetectBatchIn, DetectBatchOut, DetectIn, DetectOut

from .. import config, upstream
from ..deps import require_client

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/vision", tags=["vision"], dependencies=[Depends(require_client)])


def _fallback_allowed(exc: ApiError, forced: bool) -> bool:
    """Whether a failed VLM call should be retried on YOLO."""
    if forced or not config.DETECT_FALLBACK_TO_YOLO:
        return False
    if not upstream.vision.configured:
        return False
    # Saturation is a "come back shortly", not a reason to spend the GPU.
    return exc.code not in config.DETECT_NO_FALLBACK_CODES


@router.post("/detect", response_model=DetectOut, summary="Detect objects in an image")
async def detect(
    req: DetectIn,
    backend: str | None = Query(
        None,
        pattern="^(vlm|yolo)$",
        description="Force a backend. Default: DETECT_BACKEND (vlm). 'yolo' uses the "
        "construction-trained model on the vision pod — fixed class list, tight boxes.",
    ),
):
    chosen = (backend or config.DETECT_BACKEND).lower()
    payload = req.model_dump(exclude_none=True)

    if chosen == "yolo":
        return await upstream.vision.post("/detect", payload)

    try:
        return await upstream.cpu.post("/detect", payload)
    except ApiError as exc:
        if not _fallback_allowed(exc, forced=backend is not None):
            raise
        log.warning(
            "vlm detect failed, falling back to yolo",
            extra={"code": exc.code, "status": exc.status},
        )
        return await upstream.vision.post("/detect", payload)


@router.post(
    "/detect-batch",
    response_model=DetectBatchOut,
    summary="Detect objects across several images in one call",
)
async def detect_batch(req: DetectBatchIn):
    """One call, many images — the shape a gallery upload actually has.

    The cpu pod fans these out concurrently under its own semaphore, so this is
    both faster than N sequential calls and gentler on the provider's rate limit
    than N parallel ones from the client. Each image reports its own result or
    its own error; one bad image never fails the batch.

    No YOLO fallback here by design: a partial failure is already expressed
    per-image, and a batch is exactly the workload that must not be redirected
    onto a single-slot GPU queue.
    """
    if len(req.images) > config.DETECT_MAX_BATCH:
        raise BadRequest(
            f"Batch of {len(req.images)} exceeds the limit of {config.DETECT_MAX_BATCH}; "
            "split it across calls.",
            detail={"max_batch": config.DETECT_MAX_BATCH},
        )
    return await upstream.cpu.post("/detect-batch", req.model_dump(exclude_none=True))
