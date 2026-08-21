"""Object detection via a Qwen vision model — the only detection path.

The gallery consumes **labels**, not geometry. Nothing in the client renders a
bounding box: the web SDK reduces a detection list straight to a unique set of
label strings and drops everything else. So this asks the model for names and
confidences and nothing more, which is a better trade in three directions at
once:

  * a VLM reasons its way to a box rather than regressing one, so coordinates
    were always the weakest thing it produced — and they were being thrown away
  * dropping the box field takes roughly 60% off output tokens, the expensive side
  * recognition does not need source resolution, so the image is downscaled and
    JPEG-encoded before it is sent. Pixels are what a vision model bills for.

Two more things worth knowing:

* `confidence` is the model's self-report, not a calibrated score. The client
  filters on it, so it is kept, but do not read precision into it.
* The prompt steers toward a canonical vocabulary (config.DETECT_VOCABULARY).
  Unconstrained, a VLM will call the same object "hard hat", "hardhat" and
  "safety helmet" across three photos and each becomes its own album downstream.

The YOLO path still exists on the vision pod and is still reachable by an
explicit `?backend=yolo`, but it is no longer an automatic fallback: it has a
fixed class list and cannot answer the open-vocabulary question this one does.
Provider failover in llm_proxy is what covers an outage now.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from crm_common import media, pool
from crm_common.errors import ApiError, BadRequest, UpstreamError
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
                },
                "required": ["name", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _prompt() -> str:
    vocabulary = ", ".join(config.DETECT_VOCABULARY)
    return (
        "List every distinct physical thing you can see in this construction-site "
        "photo. For each one give:\n"
        '  name        a short lowercase noun, singular (e.g. "hard hat", "excavator")\n'
        "  confidence  0.0-1.0, how sure you are the thing is present\n"
        "\n"
        "Prefer these terms whenever one of them fits, so the same object is named "
        f"the same way every time:\n{vocabulary}\n"
        "\n"
        "If nothing in the list fits, use your own short lowercase noun. Name each "
        "distinct kind of thing once — do not repeat a name for multiple copies of "
        "it. Omit anything you are not reasonably sure of. Return JSON only, as "
        '{"objects": [...]}.'
    )


# Remembers, per provider, that strict json_schema was rejected and json_object
# worked. Negotiated once at runtime instead of demanding you configure it right.
_json_object_only: set[str] = set()


def _response_format(strict: bool) -> dict:
    if strict:
        return {
            "type": "json_schema",
            "json_schema": {"name": "detections", "schema": _SCHEMA, "strict": True},
        }
    return {"type": "json_object"}


def _prepare(raw: str) -> tuple[str, int, int]:
    """Decode (or fetch), downscale, JPEG-encode. Blocking — runs in a thread.

    A URL input means an outbound HTTP fetch and a re-encode. Doing that on the
    event loop would stall every other request on this pod for the duration,
    which on a shared CPU pod means background removal and denoise queue behind
    somebody's slow image host.

    The original dimensions are returned, not the downscaled ones: they are what
    a whole-image compatibility box is expressed in.
    """
    image = media.load_image(raw, mode="RGB")
    width, height = image.width, image.height
    small = media.fit(image, config.DETECT_MAX_SIDE, multiple=1)
    encoded = media.encode_jpeg(small, quality=config.DETECT_JPEG_QUALITY)
    return "data:image/jpeg;base64," + encoded, width, height


def _body(data_url: str, strict: bool) -> dict:
    return {
        "temperature": 0,
        "max_tokens": config.DETECT_MAX_OUTPUT_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": _response_format(strict),
    }


def _content(result: dict) -> str | dict | list:
    try:
        return result["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise UpstreamError(
            f"Could not read the vision model response: {type(exc).__name__}"
        ) from exc


def _parse(content: str | dict | list) -> list[dict]:
    """Tolerant of the shapes a model actually returns under json_object mode.

    Strict schema mode makes this trivial; json_object mode does not guarantee
    the wrapper key, and some models still fence their JSON in markdown.
    """
    if isinstance(content, str):
        text = _JSON_FENCE.sub("", content.strip())
        try:
            content = json.loads(text)
        except json.JSONDecodeError as exc:
            raise UpstreamError(f"Vision model did not return JSON: {exc}") from exc

    if isinstance(content, list):
        items = content
    elif isinstance(content, dict):
        items = content.get("objects")
        if not isinstance(items, list):
            # Some models answer {"detections": [...]} or {"items": [...]}.
            for key in ("detections", "results", "labels", "items"):
                if isinstance(content.get(key), list):
                    items = content[key]
                    break
            else:
                items = []
    else:
        items = []

    return [item for item in items if isinstance(item, dict)]


def _detections(items: list[dict], width: int, height: int, conf_floor: float | None) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        name = str(item.get("name") or item.get("label") or "").strip().lower()
        if not name or name in seen:
            continue
        try:
            confidence = float(item.get("confidence", item.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        if conf_floor is not None and confidence < conf_floor:
            continue
        seen.add(name)
        detection = {"name": name, "confidence": confidence}
        if config.DETECT_EMIT_BOX:
            # A whole-image box. Nothing reads it — it exists so a client typed
            # against the YOLO contract (box is required there) keeps validating.
            detection["box"] = {"x1": 0.0, "y1": 0.0, "x2": float(width), "y2": float(height)}
        out.append(detection)
    return out


async def _ask(data_url: str) -> dict:
    """One upstream call, negotiating structured-output mode as needed."""
    mode = config.DETECT_STRUCTURED_MODE
    if mode == "json_object":
        return await llm_proxy.chat(_body(data_url, strict=False), model_kind="vision", task="detect")
    if mode == "json_schema":
        return await llm_proxy.chat(_body(data_url, strict=True), model_kind="vision", task="detect")

    # auto: strict first, and remember a rejection so it is paid for once.
    chain = config.provider_names()
    if chain and all(name in _json_object_only for name in chain):
        return await llm_proxy.chat(_body(data_url, strict=False), model_kind="vision", task="detect")

    try:
        return await llm_proxy.chat(_body(data_url, strict=True), model_kind="vision", task="detect")
    except BadRequest as exc:
        # Every provider rejected the strict schema — the usual cause is a model
        # that only implements json_object mode. Downgrade once and remember.
        log.warning(
            "strict json_schema rejected; falling back to json_object",
            extra={"providers": chain, "detail": str(exc)},
        )
        _json_object_only.update(chain)
        return await llm_proxy.chat(_body(data_url, strict=False), model_kind="vision", task="detect")


# Process-wide, NOT per-request. A semaphore constructed inside the request
# handler bounds one batch and nothing else: ten concurrent batches would each
# get their own allowance of 8 and put 80 calls in flight, which is precisely the
# rate-limit stampede the bound exists to prevent. On a pod — one long-lived
# process serving every user — the only meaningful scope is the process.
_batch_semaphore: asyncio.Semaphore | None = None


def _batch_slot() -> asyncio.Semaphore:
    global _batch_semaphore
    if _batch_semaphore is None:
        # Lazily: a Semaphore binds to the running loop, which does not exist at
        # import time.
        _batch_semaphore = asyncio.Semaphore(max(1, config.DETECT_BATCH_CONCURRENCY))
    return _batch_semaphore


async def detect(payload: dict) -> dict:
    data_url, width, height = await pool.PREP.run(_prepare, payload["image"])
    result = await _ask(data_url)

    items = _parse(_content(result))
    conf = payload.get("conf")
    detections = _detections(items, width, height, float(conf) if conf is not None else None)

    log.info(
        "vlm detect",
        extra={
            "count": len(detections),
            "provider": result.get("provider"),
            "model": result.get("model"),
        },
    )
    return DetectOut(
        detections=detections,
        source="vlm",
        provider=result.get("provider"),
        model=result.get("model"),
    ).model_dump()


async def detect_batch(payload: dict) -> dict:
    """Many images, bounded fan-out, per-image error isolation.

    Deliberately N independent calls rather than N images in one request. Putting
    several images in a single call saves no tokens — you pay for the same pixels
    — and costs you the ability to fail one image without failing the rest.

    The semaphore is the important part: unbounded fan-out on a 200-photo upload
    walks straight into the provider's rate limit and burns every key doing it.
    """
    images = payload.get("images") or []
    if not images:
        raise BadRequest("Missing 'images'.")
    if len(images) > config.DETECT_MAX_BATCH:
        raise BadRequest(
            f"Batch of {len(images)} exceeds the limit of {config.DETECT_MAX_BATCH}; "
            "split it across calls."
        )

    conf = payload.get("conf")

    async def one(index: int, image: str) -> dict:
        async with _batch_slot():
            try:
                result = await detect({"image": image, **({"conf": conf} if conf is not None else {})})
            except ApiError as exc:
                return {"index": index, "error": exc.to_dict()["error"]}
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("batch detect crashed", extra={"index": index})
                return {
                    "index": index,
                    "error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"},
                }
            return {
                "index": index,
                "detections": result["detections"],
                "source": result.get("source"),
                "provider": result.get("provider"),
                "model": result.get("model"),
            }

    results = await asyncio.gather(*(one(i, img) for i, img in enumerate(images)))
    failed = sum(1 for r in results if r.get("error"))
    log.info("vlm detect batch", extra={"total": len(results), "failed": failed})
    return {"results": results, "ok": len(results) - failed, "failed": failed}
