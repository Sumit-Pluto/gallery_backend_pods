"""crm-cpu — background removal, audio denoise, OCR, Groq LLM proxy, and the
primary (VLM) object-detection path.

No GPU, no torch. Cheap pod, so it also hosts anything that is pure IO.
"""

from __future__ import annotations

import asyncio

from . import config  # noqa: F401  — sets cache dirs before rembg/onnx import

from crm_common.schemas import (
    AudioOut,
    DenoiseIn,
    DetectIn,
    DetectOut,
    ImageOut,
    LlmIn,
    OcrIn,
    OcrOut,
    RemoveBgIn,
)
from crm_common.service import Readiness, create_app, require_ready

from . import detect_vlm, llm_proxy, tasks

readiness = Readiness()


async def warmup() -> None:
    detail = await asyncio.to_thread(tasks.warm)
    detail["groq_keys"] = len(llm_proxy.keys())
    readiness.mark_ready(**detail)


app = create_app(
    "cpu",
    readiness=readiness,
    warmup=warmup,
    extra_health=lambda: {"ocr": tasks.ocr_available(), "groq_configured": llm_proxy.configured()},
    on_shutdown=llm_proxy.aclose,
)


@app.post("/remove-bg", response_model=ImageOut)
async def remove_bg(req: RemoveBgIn):
    require_ready(readiness)
    return await asyncio.to_thread(tasks.remove_bg, req.model_dump(exclude_none=True))


@app.post("/denoise", response_model=AudioOut)
async def denoise(req: DenoiseIn):
    require_ready(readiness)
    return await asyncio.to_thread(tasks.denoise, req.model_dump())


@app.post("/ocr", response_model=OcrOut)
async def ocr(req: OcrIn):
    require_ready(readiness)
    return await asyncio.to_thread(tasks.ocr, req.model_dump())


@app.post("/detect", response_model=DetectOut)
async def detect(req: DetectIn):
    """Primary object detection: Qwen vision via Groq. No GPU used here."""
    return await detect_vlm.detect(req.model_dump(exclude_none=True))


@app.post("/llm")
async def llm(req: LlmIn):
    """Groq passthrough. The caller owns prompt/model/schema; we own the keys."""
    return await llm_proxy.chat(req.model_dump(exclude_none=True))
