"""Image endpoints: edit, upscale, background removal, OCR."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from crm_common.errors import ApiError
from crm_common.schemas import EditIn, ImageOut, JobAccepted, OcrIn, OcrOut, RemoveBgIn, UpscaleIn

from .. import config, jobs, ops, upstream
from ..deps import require_client

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/image", tags=["image"], dependencies=[Depends(require_client)])


def _poll_url(job_id: str) -> str:
    return f"{config.PUBLIC_BASE_URL}/v1/jobs/{job_id}"


@router.post(
    "/edit",
    summary="Apply an allow-listed edit operation",
    response_model=None,
    responses={202: {"model": JobAccepted}, 200: {"model": ImageOut}},
)
async def edit(
    request: Request,
    req: EditIn,
    client: str = Depends(require_client),
    wait: float = Query(
        0,
        ge=0,
        description=(
            "Seconds to block waiting for a diffusion result before returning a job id. "
            "0 (default) returns 202 immediately. Capped by SYNC_WAIT_MAX."
        ),
    ),
):
    """The single entry point for every generative edit.

    You send an **op name**, not a prompt — the gateway owns the wording (see
    ops.py). Fixed-function ops (upscale, restore, remove-background) answer
    inline; diffusion ops return `202 {job_id}` and you poll `/v1/jobs/{id}`.
    """
    route = ops.resolve(req.op, req.image, req.mask, req.params)
    service = upstream.BY_NAME[route.service]

    if not route.async_job:
        return await service.post(route.path, route.payload)

    job = await jobs.store.submit(
        f"edit:{req.op.type}",
        client,
        lambda: service.post(route.path, route.payload),
        # Queue turns are taken per end user, so one person bulk-editing an
        # album cannot put everybody else behind all twenty of their renders.
        fair_key=getattr(request.state, "fair_key", None),
    )

    if wait > 0:
        await jobs.store.wait(job, min(wait, config.SYNC_WAIT_MAX))
        if job.status == "done":
            return job.result
        if job.status == "error":
            err = job.error or {}
            raise ApiError(err.get("message", "Edit failed."), status=502,
                           code=err.get("code", "upstream_error"), detail=err.get("detail"))

    return JSONResponse(
        status_code=202,
        content={"job_id": job.id, "status": job.status, "poll": _poll_url(job.id)},
    )


@router.post("/upscale", response_model=ImageOut, summary="Real-ESRGAN upscale (2x or 4x)")
async def upscale(req: UpscaleIn):
    return await upstream.vision.post("/upscale", req.model_dump())


@router.post("/remove-bg", response_model=ImageOut, summary="Background removal (U^2-Net)")
async def remove_bg(req: RemoveBgIn):
    return await upstream.cpu.post("/remove-bg", req.model_dump(exclude_none=True))


ocr_router = APIRouter(prefix="/v1", tags=["ocr"], dependencies=[Depends(require_client)])


@ocr_router.post("/ocr", response_model=OcrOut, summary="PP-OCRv6 text detection + recognition")
async def ocr(req: OcrIn):
    return await upstream.cpu.post("/ocr", req.model_dump())
