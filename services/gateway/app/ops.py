"""The op -> instruction table, and the op -> upstream routing.

This is the product logic that used to live in apps/web/src/app/api/ai/edit/route.ts
on Vercel. It is the reason the client can send an *op name* instead of a prompt:
the gateway owns the wording, so nobody can push arbitrary text at the GPU and
nobody has to reimplement these prompts on their side.

The prompt strings and the routing decisions below are ported verbatim, including
the two workarounds that were learned the hard way:

  * colorize does NOT go to a dedicated colorize model — the DDColor endpoint
    kept hard-crashing on modelscope, so it is an instruction to img2img instead.
  * replace-sky without a mask degrades to a low-strength (0.4) img2img so the
    foreground survives, rather than failing. With a mask it is real inpainting.

Changing anything here changes what the client's images look like. Do it on
purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from crm_common.errors import BadRequest
from crm_common.schemas import EditOp, EditParams

# --------------------------------------------------------------------------- #
# Instructions
# --------------------------------------------------------------------------- #

OP_PROMPTS: dict[str, str] = {
    "restore": (
        "Restore and enhance this photograph: improve sharpness and clarity, correct "
        "exposure and white balance, reduce noise and compression artifacts, recover "
        "detail. Keep it natural and photorealistic."
    ),
    "colorize": "Colorize this image with natural, realistic, well-balanced colors.",
    "replace-sky": (
        "Replace the sky with a dramatic, beautiful golden-hour sky with soft clouds. "
        "Keep the foreground subject unchanged and the result photorealistic."
    ),
}

MAGIC_ERASER_PROMPT = (
    "Fill the selected region with a clean, seamless, plausible background. Photorealistic."
)

# Ops that need a mask to mean anything.
MASK_REQUIRED = {"magic-eraser", "generative-fill"}

# Ops that produce no instruction at all (pure fixed-function models).
NO_PROMPT = {"upscale", "restore", "remove-background"}


def build_instruction(op: EditOp) -> str:
    """Allow-listed op -> the prompt we actually send. Never trusts free text."""
    caller_prompt = (op.prompt or "").strip()

    if op.type in ("prompt", "generative-fill"):
        if not caller_prompt:
            raise BadRequest(f"op.type '{op.type}' requires a non-empty op.prompt.")
        return caller_prompt[:500]

    if op.type == "replace-sky":
        if caller_prompt:
            return (
                f"Replace the sky with: {caller_prompt[:300]}. "
                "Keep the foreground unchanged and photorealistic."
            )
        return OP_PROMPTS["replace-sky"]

    if op.type == "magic-eraser":
        return MAGIC_ERASER_PROMPT

    if op.type in OP_PROMPTS:
        return OP_PROMPTS[op.type]

    if op.type in NO_PROMPT:
        return ""

    raise BadRequest(f"Unsupported operation '{op.type}'.")


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

Service = Literal["vision", "diffusion", "cpu"]


@dataclass
class Route:
    service: Service
    path: str
    payload: dict = field(default_factory=dict)
    # Diffusion is slow enough that the gateway always runs it as a background
    # job; the fixed-function ops answer fast enough to return inline.
    async_job: bool = False


def _diffusion_params(params: EditParams) -> dict:
    """EditParams -> the diffusion service's field names. Already clamped by pydantic."""
    out: dict = {}
    if params.negative_prompt:
        out["negative_prompt"] = params.negative_prompt
    if params.strength is not None:
        out["strength"] = params.strength
    if params.steps is not None:
        out["num_inference_steps"] = params.steps
    if params.guidance_scale is not None:
        out["guidance_scale"] = params.guidance_scale
    if params.seed is not None:
        out["seed"] = params.seed
    return out


def resolve(op: EditOp, image: str, mask: str | None, params: EditParams) -> Route:
    """Pick the upstream and build its payload. Mirrors editRunPod()."""
    instruction = build_instruction(op)
    extra = _diffusion_params(params)

    if op.type in MASK_REQUIRED and not mask:
        raise BadRequest(f"op.type '{op.type}' needs a mask/selection.")

    if op.type == "remove-background":
        return Route("cpu", "/remove-bg", {"image": image})

    if op.type == "restore":
        # Real-ESRGAN with the GFPGAN face pass = "Restore & Enhance".
        return Route("vision", "/upscale", {"image": image, "scale": 4, "face_enhance": True})

    if op.type == "upscale":
        factor = 4 if op.factor == 4 else 2
        return Route("vision", "/upscale", {"image": image, "scale": factor, "face_enhance": False})

    if op.type in ("colorize", "prompt"):
        return Route(
            "diffusion",
            "/img2img",
            {"image": image, "prompt": instruction, **_img2img_only(extra)},
            async_job=True,
        )

    if op.type == "replace-sky":
        if mask:
            return Route(
                "diffusion",
                "/inpaint",
                {"image": image, "mask": mask, "prompt": instruction, **extra},
                async_job=True,
            )
        # No mask: low-strength img2img keeps the foreground mostly intact.
        payload = {"image": image, "prompt": instruction, **_img2img_only(extra)}
        payload.setdefault("strength", 0.4)
        return Route("diffusion", "/img2img", payload, async_job=True)

    if op.type in ("magic-eraser", "generative-fill"):
        return Route(
            "diffusion",
            "/inpaint",
            {"image": image, "mask": mask, "prompt": instruction, **extra},
            async_job=True,
        )

    raise BadRequest(f"Unsupported operation '{op.type}'.")


def _img2img_only(extra: dict) -> dict:
    """FLUX img2img takes only strength and seed; the rest are SDXL-inpaint fields."""
    return {k: v for k, v in extra.items() if k in ("strength", "seed")}
