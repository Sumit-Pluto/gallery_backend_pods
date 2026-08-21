"""crm-diffusion — FLUX.2 [klein] img2img and SDXL inpainting.

Reachable only from the gateway. One job at a time on the card (see
crm_common.gpu). The gateway fronts this with an async job queue, so a 60-90 s
render never holds the client's HTTP socket open.
"""

from __future__ import annotations

import asyncio

from . import config  # noqa: F401  — must import first; sets HF_HOME before torch

from crm_common import gpu
from crm_common.schemas import ImageOut, Img2ImgIn, InpaintIn
from crm_common.service import Readiness, create_app, require_ready

from . import models, tasks

readiness = Readiness()


async def warmup() -> None:
    # Execute before you load. A card the torch build has no kernels for will
    # happily accept 16 GB of weights and only fail when something actually
    # runs — which meant readiness passed and every user request 500'd instead.
    # Probing first also means a bad card is rejected in a second rather than
    # after a multi-GB download.
    probe = await asyncio.to_thread(gpu.self_test)
    detail = await gpu.run_exclusive(models.warm, task="warmup")
    readiness.mark_ready(**detail, gpu=probe)


app = create_app(
    "diffusion",
    readiness=readiness,
    warmup=warmup,
    extra_health=lambda: {"gpu": gpu.vram_report(), "queue": gpu.stats(), "pipelines": models.report()},
)


@app.post("/img2img", response_model=ImageOut)
async def img2img(req: Img2ImgIn):
    require_ready(readiness)
    return await gpu.run_exclusive(tasks.img2img, req.model_dump(exclude_none=True), task="img2img")


@app.post("/inpaint", response_model=ImageOut)
async def inpaint(req: InpaintIn):
    require_ready(readiness)
    return await gpu.run_exclusive(tasks.inpaint, req.model_dump(exclude_none=True), task="inpaint")
