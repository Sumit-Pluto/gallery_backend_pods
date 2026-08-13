"""The three vision tasks. Bodies are ports of the original serverless handler.

These are synchronous and blocking on purpose — `crm_common.gpu.run_exclusive`
runs them in a worker thread behind a semaphore, so the event loop stays free and
only one job touches the card at a time.
"""

from __future__ import annotations

import logging
import os
import tempfile

from crm_common import media
from crm_common.errors import BadRequest
from crm_common.schemas import DetectOut, ImageOut, TranscribeOut

from . import config, models

log = logging.getLogger(__name__)


def detect(payload: dict) -> dict:
    """YOLO detection — the fallback path.

    Primary object detection is the Qwen vision model on the cpu service; this
    stays wired so a Groq outage or a rate limit has somewhere to fall back to.
    Boxes come back in the ORIGINAL image's pixel coordinates (ultralytics
    rescales from the inference size), which the caller normalises to 0..1.
    """
    img = media.load_image(payload["image"], mode="RGB")
    conf = payload.get("conf") if payload.get("conf") is not None else config.YOLO_CONF
    imgsz = payload.get("imgsz") or config.YOLO_IMGSZ
    iou = payload.get("iou") if payload.get("iou") is not None else config.YOLO_IOU

    results = models.yolo().predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    detections = []
    for result in results:
        names = result.names
        for box in result.boxes or []:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "name": str(names[cls]),
                    "confidence": float(box.conf[0]),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
    return DetectOut(detections=detections, source="yolo").model_dump()


def upscale(payload: dict) -> dict:
    import numpy as np
    from PIL import Image

    scale = int(payload.get("scale") or 4)
    if scale not in (2, 4):
        scale = 4
    face = bool(payload.get("face_enhance", False))

    pil = media.load_image(payload["image"], mode="RGB")
    rgb = np.array(pil)
    bgr = rgb[:, :, ::-1].copy()

    if face:
        _, _, restored = models.face_enhancer().enhance(
            bgr, has_aligned=False, only_center_face=False, paste_back=True
        )
        out = Image.fromarray(restored[:, :, ::-1]).resize(
            (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.LANCZOS
        )
    else:
        restored, _ = models.upsampler().enhance(bgr, outscale=scale)
        out = Image.fromarray(restored[:, :, ::-1])

    return ImageOut(image=media.encode_png(out), width=out.width, height=out.height).model_dump()


def transcribe(payload: dict) -> dict:
    raw = media.decode_media(payload["audio"], field="audio")
    if not raw:
        raise BadRequest("Missing 'audio'.")
    language = payload.get("language") or None
    want_timestamps = bool(payload.get("timestamps", False))

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as handle:
            handle.write(raw)
            path = handle.name
        segments, info = models.whisper().transcribe(path, language=language, vad_filter=True)
        parts, out_segments = [], []
        for seg in segments:
            parts.append(seg.text)
            if want_timestamps:
                out_segments.append(
                    {"text": seg.text.strip(), "start_sec": round(seg.start, 2), "end_sec": round(seg.end, 2)}
                )
        return TranscribeOut(
            transcript="".join(parts).strip(),
            language=info.language,
            segments=out_segments if want_timestamps else None,
        ).model_dump(exclude_none=True)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
