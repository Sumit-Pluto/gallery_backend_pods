"""img2img (FLUX.2 klein) and inpaint (SDXL). Ports of the serverless handler.

The parameter maths is copied verbatim — the guidance curve, the strength floor,
the mask resize mode, the `_fit` multiples. Changing any of it changes what the
client's existing images look like, so it stays byte-for-byte identical.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from crm_common import media
from crm_common.errors import BadRequest
from crm_common.schemas import ImageOut

from . import config, models

log = logging.getLogger(__name__)


def img2img(payload: dict) -> dict:
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise BadRequest("Missing 'prompt'.")

    image = media.fit(media.load_image(payload["image"], mode="RGB"), config.MAX_SIZE, multiple=16)

    # strength 0..1 maps onto guidance 1.0..4.0 — FLUX klein is guidance-light,
    # so this narrow band is what keeps edits from over-cooking.
    guidance = 1.0
    strength = payload.get("strength")
    if strength is not None:
        guidance = round(1.0 + max(0.0, min(1.0, float(strength))) * 3.0, 2)

    seed = payload.get("seed")
    generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None

    result = models.flux()(
        prompt=prompt,
        image=image,
        height=image.height,
        width=image.width,
        guidance_scale=guidance,
        num_inference_steps=config.FLUX_STEPS,
        generator=generator,
    ).images[0]

    return ImageOut(image=media.encode_png(result), width=result.width, height=result.height).model_dump()


def inpaint(payload: dict) -> dict:
    prompt = (payload.get("prompt") or "").strip()
    negative = (payload.get("negative_prompt") or "").strip() or None

    image = media.fit(media.load_image(payload["image"], mode="RGB"), config.MAX_SIZE, multiple=8)
    mask = media.load_image(payload["mask"], mode="L", field="mask").resize(image.size, Image.NEAREST)

    steps = int(payload.get("num_inference_steps") or 35)
    guidance = float(payload.get("guidance_scale") or 7.0)

    # No prompt means "erase" — full-strength fill. With a prompt, clamp into
    # 0.7..1.0 so the mask region is genuinely regenerated rather than nudged.
    raw_strength = payload.get("strength")
    if not prompt:
        strength = 1.0
    elif raw_strength is not None:
        strength = max(0.7, min(1.0, float(raw_strength)))
    else:
        strength = 0.9

    seed = payload.get("seed")
    generator = (
        torch.Generator(device=models.DEVICE).manual_seed(int(seed)) if seed is not None else None
    )

    effective_prompt = prompt or "clean, seamless, photorealistic background, natural continuation"

    result = models.inpaint()(
        prompt=effective_prompt,
        negative_prompt=negative,
        image=image,
        mask_image=mask,
        height=image.height,
        width=image.width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
    ).images[0]

    return ImageOut(image=media.encode_png(result), width=result.width, height=result.height).model_dump()
