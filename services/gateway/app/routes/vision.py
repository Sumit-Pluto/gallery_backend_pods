"""Object detection.

Primary path is the Qwen vision model on the cpu pod (no GPU cost). YOLO on the
vision pod is the fallback — it is only reached if the VLM call fails and
DETECT_FALLBACK_TO_YOLO is on, which keeps detection working through a Groq
outage or a rate limit without paying for a GPU the rest of the time.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from crm_common.errors import ApiError
from crm_common.schemas import DetectIn, DetectOut

from .. import config, upstream
from ..deps import require_client

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/vision", tags=["vision"], dependencies=[Depends(require_client)])


@router.post("/detect", response_model=DetectOut, summary="Detect objects in an image")
async def detect(
    req: DetectIn,
    backend: str | None = Query(
        None, pattern="^(vlm|yolo)$", description="Force a backend. Default: DETECT_BACKEND (vlm)."
    ),
):
    chosen = (backend or config.DETECT_BACKEND).lower()
    payload = req.model_dump(exclude_none=True)

    if chosen == "yolo":
        return await upstream.vision.post("/detect", payload)

    try:
        return await upstream.cpu.post("/detect", payload)
    except ApiError as exc:
        if backend is not None or not config.DETECT_FALLBACK_TO_YOLO:
            raise
        if not upstream.vision.configured:
            raise
        log.warning(
            "vlm detect failed, falling back to yolo",
            extra={"code": exc.code, "status": exc.status},
        )
        return await upstream.vision.post("/detect", payload)
