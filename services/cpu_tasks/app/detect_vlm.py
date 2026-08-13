"""Object detection via the Qwen vision model — the primary detection path.

This replaces YOLO for day-to-day detection. It costs no GPU on our side: the
image goes to Groq as a data URL and comes back as structured JSON.

Two things worth being clear-eyed about, because they change how the client
should use the result:

* A VLM returns boxes it *reasons* to, not boxes a detector regressed. Expect
  looser localisation than YOLO gave, and treat `confidence` as the model's
  self-report rather than a calibrated score.
* Coordinates come back normalised 0..1 and are converted to the source image's
  pixel space here, so the response shape is identical to the YOLO path and the
  client cannot tell which backend served it (apart from `source`).

The YOLO fallback lives on the vision pod; the gateway decides when to use it.
"""

from __future__ import annotations

import asyncio
import json
import logging

from crm_common import media
from crm_common.errors import UpstreamError
from crm_common.schemas import DetectOut

from . import config, llm_proxy

log = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["name", "confidence", "box"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}

_PROMPT = (
    "List every distinct physical object visible in this image. For each one give:\n"
    "  name       a short lowercase noun (e.g. \"excavator\", \"hard hat\", \"scaffolding\")\n"
    "  confidence 0.0-1.0, how sure you are the object is present\n"
    "  box        [x1, y1, x2, y2] as fractions of image width/height, 0.0-1.0,\n"
    "             top-left origin, x1<x2 and y1<y2\n"
    "Report each instance separately. Omit anything you are not reasonably sure of. "
    "Return JSON only."
)


def _prepare(raw: str) -> tuple[str, int, int]:
    """Decode (or fetch) and re-encode as a data URL. Blocking — runs in a thread.

    A URL input means an outbound HTTP fetch and a PNG re-encode. Doing that on
    the event loop would stall every other request on this pod for the duration,
    which on a shared CPU pod means background removal and denoise queue behind
    somebody's slow image host.
    """
    image = media.load_image(raw, mode="RGB")
    # Re-encode from the decoded image so a URL input and a base64 input take
    # exactly the same path into the model.
    return "data:image/png;base64," + media.encode_png(image), image.width, image.height


async def detect(payload: dict) -> dict:
    data_url, width, height = await asyncio.to_thread(_prepare, payload["image"])

    body = {
        "model": config.GROQ_VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "detections", "schema": _SCHEMA, "strict": True},
        },
    }

    result = await llm_proxy.chat(body)
    try:
        content = result["response"]["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise UpstreamError(f"Could not parse the vision model response: {type(exc).__name__}") from exc

    detections = []
    for item in parsed.get("objects") or []:
        box = item.get("box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        # Tolerate a model that answers in pixels instead of fractions.
        if max(x1, y1, x2, y2) > 1.5:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        detections.append(
            {
                "name": str(item.get("name") or "object").strip().lower(),
                "confidence": max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                # Pixel coords in the SOURCE image, matching the YOLO contract.
                "box": {"x1": x1 * width, "y1": y1 * height, "x2": x2 * width, "y2": y2 * height},
            }
        )

    conf_floor = payload.get("conf")
    if conf_floor is not None:
        detections = [d for d in detections if d["confidence"] >= float(conf_floor)]

    log.info("vlm detect", extra={"count": len(detections), "model": config.GROQ_VISION_MODEL})
    return DetectOut(detections=detections, source="vlm").model_dump()
