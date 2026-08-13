"""crm-vision — Real-ESRGAN upscale, GFPGAN face restore, Whisper transcribe,
YOLO detect (fallback only).

Reachable only from the gateway (X-Internal-Key). Runs on a 16 GB card: the
heavy resident model is Whisper (~2.5 GB) since ESRGAN tiles and YOLO is lazy.
"""

from __future__ import annotations

from . import config  # noqa: F401  — must import first; sets HF_HOME before torch

from crm_common import gpu
from crm_common.schemas import DetectIn, DetectOut, ImageOut, TranscribeIn, TranscribeOut, UpscaleIn
from crm_common.service import Readiness, create_app, require_ready

from . import models, tasks

readiness = Readiness()


async def warmup() -> None:
    detail = await gpu.run_exclusive(models.warm, task="warmup")
    readiness.mark_ready(**detail)


app = create_app(
    "vision",
    readiness=readiness,
    warmup=warmup,
    extra_health=lambda: {"gpu": gpu.vram_report(), "queue": gpu.stats()},
)


@app.post("/detect", response_model=DetectOut)
async def detect(req: DetectIn):
    require_ready(readiness)
    return await gpu.run_exclusive(tasks.detect, req.model_dump(exclude_none=True), task="detect")


@app.post("/upscale", response_model=ImageOut)
async def upscale(req: UpscaleIn):
    require_ready(readiness)
    return await gpu.run_exclusive(tasks.upscale, req.model_dump(), task="upscale")


@app.post("/transcribe", response_model=TranscribeOut)
async def transcribe(req: TranscribeIn):
    require_ready(readiness)
    return await gpu.run_exclusive(tasks.transcribe, req.model_dump(exclude_none=True), task="transcribe")
