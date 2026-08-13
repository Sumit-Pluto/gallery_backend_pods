"""The op -> prompt -> upstream table.

This is the logic that moved off Vercel, so it is the part most worth pinning
down: if a prompt string or a routing decision drifts, the client's images
silently change. Each assertion below mirrors a branch of the original
editRunPod() / instruction builder.
"""

from __future__ import annotations

import pytest

from crm_common.errors import BadRequest
from crm_common.schemas import EditOp, EditParams

from app import ops

IMG = "aGVsbG8="  # placeholder; ops never decodes, it only routes
MASK = "bWFzaw=="
NO_PARAMS = EditParams()


def route(op_type, prompt=None, factor=None, mask=None, params=NO_PARAMS):
    return ops.resolve(EditOp(type=op_type, prompt=prompt, factor=factor), IMG, mask, params)


# --------------------------------------------------------------------------- #
# Fixed-function ops -> vision / cpu, answered inline
# --------------------------------------------------------------------------- #


def test_restore_is_upscale_with_face_pass():
    r = route("restore")
    assert (r.service, r.path, r.async_job) == ("vision", "/upscale", False)
    assert r.payload["scale"] == 4
    assert r.payload["face_enhance"] is True


@pytest.mark.parametrize("factor,expected", [(4, 4), (2, 2), (None, 2)])
def test_upscale_factor_defaults_to_2(factor, expected):
    r = route("upscale", factor=factor)
    assert r.payload["scale"] == expected
    assert r.payload["face_enhance"] is False
    assert r.async_job is False


def test_remove_background_goes_to_cpu():
    r = route("remove-background")
    assert (r.service, r.path, r.async_job) == ("cpu", "/remove-bg", False)


# --------------------------------------------------------------------------- #
# Diffusion ops -> async jobs
# --------------------------------------------------------------------------- #


def test_colorize_routes_through_img2img_not_a_colorize_model():
    # The dedicated DDColor endpoint kept crashing; colorize is an instruction.
    r = route("colorize")
    assert (r.service, r.path, r.async_job) == ("diffusion", "/img2img", True)
    assert r.payload["prompt"] == ops.OP_PROMPTS["colorize"]


def test_prompt_op_uses_caller_text_and_truncates_at_500():
    r = route("prompt", prompt="x" * 900)
    assert len(r.payload["prompt"]) == 500
    assert r.async_job is True


def test_prompt_op_rejects_empty_text():
    with pytest.raises(BadRequest):
        route("prompt", prompt="   ")


def test_replace_sky_without_mask_degrades_to_low_strength_img2img():
    r = route("replace-sky")
    assert r.path == "/img2img"
    # 0.4 keeps the foreground mostly intact when there is nothing to mask with.
    assert r.payload["strength"] == 0.4
    assert r.payload["prompt"] == ops.OP_PROMPTS["replace-sky"]


def test_replace_sky_with_mask_is_real_inpainting():
    r = route("replace-sky", mask=MASK)
    assert r.path == "/inpaint"
    assert r.payload["mask"] == MASK


def test_replace_sky_wraps_a_custom_prompt():
    r = route("replace-sky", prompt="a stormy night sky")
    assert r.payload["prompt"] == (
        "Replace the sky with: a stormy night sky. "
        "Keep the foreground unchanged and photorealistic."
    )


def test_magic_eraser_uses_the_fixed_instruction():
    r = route("magic-eraser", mask=MASK)
    assert r.path == "/inpaint"
    assert r.payload["prompt"] == ops.MAGIC_ERASER_PROMPT


@pytest.mark.parametrize("op_type", ["magic-eraser"])
def test_mask_required_ops_reject_a_missing_mask(op_type):
    with pytest.raises(BadRequest, match="mask"):
        route(op_type)


def test_generative_fill_needs_both_prompt_and_mask():
    with pytest.raises(BadRequest):
        route("generative-fill", mask=MASK)  # no prompt
    with pytest.raises(BadRequest, match="mask"):
        route("generative-fill", prompt="a wooden fence")  # no mask
    r = route("generative-fill", prompt="a wooden fence", mask=MASK)
    assert r.path == "/inpaint"


# --------------------------------------------------------------------------- #
# Parameter handling
# --------------------------------------------------------------------------- #


def test_inpaint_receives_full_params_img2img_only_strength_and_seed():
    params = EditParams(negative_prompt="blurry", strength=0.8, steps=30, guidance_scale=7.5, seed=42)

    inpaint = route("magic-eraser", mask=MASK, params=params)
    assert inpaint.payload["negative_prompt"] == "blurry"
    assert inpaint.payload["num_inference_steps"] == 30
    assert inpaint.payload["guidance_scale"] == 7.5

    # FLUX img2img takes neither negative prompts nor step/guidance overrides.
    img2img = route("prompt", prompt="a red car", params=params)
    assert set(img2img.payload) == {"image", "prompt", "strength", "seed"}


def test_params_are_clamped_by_the_schema_not_the_caller():
    import pydantic

    for bad in ({"steps": 500}, {"guidance_scale": 99}, {"strength": 5}, {"seed": -1}):
        with pytest.raises(pydantic.ValidationError):
            EditParams(**bad)


def test_unknown_op_is_rejected_by_the_schema():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EditOp(type="delete-everything")
