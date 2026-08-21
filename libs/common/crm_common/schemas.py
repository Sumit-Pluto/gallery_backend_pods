"""Request/response models shared between the gateway and the backend services.

The gateway validates the public contract with these; the services validate their
internal contract with the same classes. One definition, so a field cannot drift
between the two hops.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #


class DetectIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(..., description="base64 (optionally data: prefixed) or an http(s) URL")
    conf: float | None = Field(None, ge=0.0, le=1.0)
    # iou / imgsz are YOLO-only knobs. The VLM path ignores them, but they stay in
    # the contract so a caller written against the YOLO backend still validates.
    iou: float | None = Field(None, ge=0.0, le=1.0)
    imgsz: int | None = Field(None, ge=320, le=2048)


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    name: str
    confidence: float
    # Optional because the VLM path is label-only: nothing in the gallery renders
    # a box, so we do not pay a vision model to reason about coordinates it is
    # bad at. When DETECT_EMIT_BOX is on, a whole-image box is filled in here so
    # clients typed against the YOLO contract keep working unchanged.
    box: Box | None = None


class DetectOut(BaseModel):
    detections: list[Detection]
    source: Literal["vlm", "yolo"] = "vlm"
    # Which provider/model actually answered. Worth returning: with failover in
    # play, "why did this image tag differently?" is otherwise unanswerable.
    provider: str | None = None
    model: str | None = None


class DetectBatchIn(BaseModel):
    """Many images, one call. Each image is still its own upstream request."""

    model_config = ConfigDict(extra="forbid")
    images: list[str] = Field(..., min_length=1)
    conf: float | None = Field(None, ge=0.0, le=1.0)


class DetectBatchItem(BaseModel):
    index: int
    # Exactly one of these is set. A single bad image must not fail the batch.
    detections: list[Detection] | None = None
    error: dict | None = None
    source: Literal["vlm", "yolo"] | None = None
    provider: str | None = None
    model: str | None = None


class DetectBatchOut(BaseModel):
    results: list[DetectBatchItem]
    ok: int
    failed: int


class UpscaleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    scale: Literal[2, 4] = 4
    face_enhance: bool = False


class ImageOut(BaseModel):
    image: str = Field(..., description="base64 PNG")
    width: int | None = None
    height: int | None = None


class TranscribeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str
    language: str | None = None
    timestamps: bool = False


class Segment(BaseModel):
    text: str
    start_sec: float
    end_sec: float


class TranscribeOut(BaseModel):
    transcript: str
    language: str | None = None
    segments: list[Segment] | None = None


# --------------------------------------------------------------------------- #
# Diffusion
# --------------------------------------------------------------------------- #


class Img2ImgIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    strength: float | None = Field(None, ge=0.0, le=1.0)
    seed: int | None = None


class InpaintIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    mask: str
    prompt: str = Field("", max_length=2000)
    negative_prompt: str | None = Field(None, max_length=2000)
    strength: float | None = Field(None, ge=0.0, le=1.0)
    num_inference_steps: int | None = Field(None, ge=1, le=100)
    guidance_scale: float | None = Field(None, ge=0.0, le=30.0)
    seed: int | None = None


# --------------------------------------------------------------------------- #
# CPU tasks
# --------------------------------------------------------------------------- #


class RemoveBgIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    model_name: str | None = None


class DenoiseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str


class AudioOut(BaseModel):
    audio: str = Field(..., description="base64 WAV, 48 kHz mono")


class OcrLine(BaseModel):
    text: str
    score: float
    box: list[list[float]]


class OcrIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str


class OcrOut(BaseModel):
    lines: list[OcrLine]
    text: str


class LlmIn(BaseModel):
    """OpenAI chat/completions body, forwarded to Groq verbatim.

    `extra="allow"` on purpose: the caller owns the prompt, the model and the
    structured-output schema. The server owns only the API keys.
    """

    model_config = ConfigDict(extra="allow")
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    model: str | None = None


class LlmOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    response: dict[str, Any]


# --------------------------------------------------------------------------- #
# Gateway: public image-edit contract
# --------------------------------------------------------------------------- #

OpType = Literal[
    "restore",
    "colorize",
    "replace-sky",
    "magic-eraser",
    "generative-fill",
    "prompt",
    "upscale",
    "remove-background",
]


class EditParams(BaseModel):
    """Optional diffusion tuning. Every field is clamped, never passed through raw.

    Same bounds as the old `sanitizeParams` in edit/route.ts — a caller cannot
    ask for 500 steps and pin the GPU.
    """

    model_config = ConfigDict(extra="forbid")
    negative_prompt: str | None = Field(None, max_length=300)
    strength: float | None = Field(None, ge=0.0, le=1.0)
    steps: int | None = Field(None, ge=1, le=60)
    guidance_scale: float | None = Field(None, ge=1.0, le=20.0)
    seed: int | None = Field(None, ge=0, le=2_147_483_647)


class EditOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: OpType
    prompt: str | None = Field(None, max_length=1000)
    factor: Literal[2, 4] | None = None


class EditIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    op: EditOp
    mask: str | None = Field(None, description="base64/URL PNG. White = the region to regenerate.")
    params: EditParams = Field(default_factory=EditParams)


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    poll: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
