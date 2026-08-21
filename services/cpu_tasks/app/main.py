"""crm-cpu — background removal, audio denoise, OCR, the LLM proxy, and object
detection.

No GPU, no torch. Cheap pod, so it also hosts anything that is pure IO.

Detection lives here rather than on a GPU pod because it costs us no GPU at all:
the image goes out to a hosted Qwen vision model and structured JSON comes back.
That is also why it scales differently from everything else in this repo — it is
IO-bound, so the pod will hold many in flight at once. The ceiling is the
provider's rate limit, not this process.
"""

from __future__ import annotations

import asyncio

from . import config  # noqa: F401  — sets cache dirs before rembg/onnx import

from crm_common.schemas import (
    AudioOut,
    DenoiseIn,
    DetectBatchIn,
    DetectBatchOut,
    DetectIn,
    DetectOut,
    ImageOut,
    LlmIn,
    OcrIn,
    OcrOut,
    RemoveBgIn,
)
from crm_common import pool
from crm_common.service import Readiness, create_app, require_ready

from . import detect_vlm, llm_proxy, tasks

readiness = Readiness()


async def _shutdown() -> None:
    await llm_proxy.aclose()
    pool.shutdown()


async def warmup() -> None:
    detail = await asyncio.to_thread(tasks.warm)
    detail["llm_providers"] = config.provider_names()
    readiness.mark_ready(**detail)


app = create_app(
    "cpu",
    readiness=readiness,
    warmup=warmup,
    extra_health=lambda: {
        "ocr": tasks.ocr_available(),
        "llm_providers": config.provider_names(),
        "llm_configured": llm_proxy.configured(),
        "pools": pool.stats(),
    },
    on_shutdown=_shutdown,
)


@app.post("/remove-bg", response_model=ImageOut)
async def remove_bg(req: RemoveBgIn):
    require_ready(readiness)
    return await pool.HEAVY.run(tasks.remove_bg, req.model_dump(exclude_none=True))


@app.post("/denoise", response_model=AudioOut)
async def denoise(req: DenoiseIn):
    require_ready(readiness)
    return await pool.HEAVY.run(tasks.denoise, req.model_dump())


@app.post("/ocr", response_model=OcrOut)
async def ocr(req: OcrIn):
    require_ready(readiness)
    return await pool.HEAVY.run(tasks.ocr, req.model_dump())


@app.post("/detect", response_model=DetectOut)
async def detect(req: DetectIn):
    """Object detection: a Qwen vision model, label-only. No GPU used here.

    Deliberately not gated on `readiness` — detection needs no local model, so it
    can serve while rembg and OCR are still warming.
    """
    return await detect_vlm.detect(req.model_dump(exclude_none=True))


@app.post("/detect-batch", response_model=DetectBatchOut)
async def detect_batch(req: DetectBatchIn):
    """Several images in one call, fanned out concurrently and bounded.

    One request per image upstream — batching images into a single model call
    would not save tokens and would let one bad image fail the whole set.
    """
    return await detect_vlm.detect_batch(req.model_dump(exclude_none=True))


@app.post("/llm")
async def llm(req: LlmIn):
    """Provider passthrough. The caller owns prompt/model/schema; we own the keys.

    A caller-supplied `model` pins the provider, because model names are not
    portable between Model Studio and Groq.
    """
    return await llm_proxy.chat(req.model_dump(exclude_none=True))
