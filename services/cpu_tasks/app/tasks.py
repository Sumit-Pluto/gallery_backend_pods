"""Background removal, audio denoise and OCR. All CPU, all blocking.

Run through `asyncio.to_thread` from main.py so one slow rembg call does not
stall health checks for everything else on the pod.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import threading

from crm_common import media
from crm_common.errors import ApiError
from crm_common.schemas import AudioOut, ImageOut, OcrOut

from . import config

log = logging.getLogger(__name__)

_sessions: dict = {}
_session_lock = threading.Lock()
_ocr: tuple | None = None
_ocr_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Background removal (rembg / U^2-Net)
# --------------------------------------------------------------------------- #
def _session(name: str):
    with _session_lock:
        if name not in _sessions:
            from rembg import new_session

            _sessions[name] = new_session(name)
            log.info("rembg session ready", extra={"model": name})
        return _sessions[name]


def remove_bg(payload: dict) -> dict:
    from rembg import remove

    name = (payload.get("model_name") or config.BG_REMOVE_MODEL).strip() or "u2net"
    img = media.load_image(payload["image"], mode="RGBA")
    out = remove(img, session=_session(name))
    if out.mode != "RGBA":
        out = out.convert("RGBA")
    return ImageOut(image=media.encode_png(out), width=out.width, height=out.height).model_dump()


# --------------------------------------------------------------------------- #
# Audio denoise (ffmpeg RNNoise, afftdn fallback)
# --------------------------------------------------------------------------- #
def denoise(payload: dict) -> dict:
    raw = media.decode_media(payload["audio"], field="audio")
    audio_filter = (
        f"arnndn=m={config.RNNOISE_MODEL}"
        if os.path.isfile(config.RNNOISE_MODEL)
        else "afftdn=nf=-25"
    )
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in")
        out_path = os.path.join(tmp, "out.wav")
        with open(in_path, "wb") as handle:
            handle.write(raw)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", in_path,
             "-af", audio_filter, "-ar", "48000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True,
            timeout=float(os.environ.get("FFMPEG_TIMEOUT", "300")),
        )
        if proc.returncode != 0:
            raise ApiError(
                "Audio denoise failed.",
                status=422,
                code="ffmpeg_failed",
                detail=proc.stderr.decode("utf-8", "replace")[-600:],
            )
        with open(out_path, "rb") as handle:
            out_bytes = handle.read()
    return AudioOut(audio=media.encode_bytes(out_bytes)).model_dump()


# --------------------------------------------------------------------------- #
# OCR (PP-OCRv6 ONNX, CPU)
# --------------------------------------------------------------------------- #
def ocr_available() -> bool:
    from . import ppocr

    return config.OCR_ENABLED and os.path.isfile(ppocr.DET_MODEL) and os.path.isfile(ppocr.REC_MODEL)


def _engine():
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                from . import ppocr

                if not ocr_available():
                    raise ApiError(
                        f"OCR models are not on the volume. Expected {ppocr.DET_MODEL} and "
                        f"{ppocr.REC_MODEL} — see deploy/volume/README.md.",
                        status=503,
                        code="model_unavailable",
                    )
                _ocr = (
                    ppocr.TextDetector(ppocr.DET_MODEL, limit_side_len=config.OCR_LIMIT_SIDE),
                    ppocr.TextRecognizer(ppocr.REC_MODEL, ppocr.REC_CONFIG),
                )
                log.info("ppocr ready", extra={"models": config.OCR_MODEL_DIR})
    return _ocr


def ocr(payload: dict) -> dict:
    import cv2
    import numpy as np

    from . import ppocr

    detector, recognizer = _engine()

    pil = media.load_image(payload["image"], mode="RGB")
    bgr = np.array(pil)[:, :, ::-1].copy()

    boxes = detector(bgr)
    crops = [ppocr.rotate_crop(bgr, box) for box in boxes]
    recognized = recognizer(crops)

    lines = []
    for box, (text, conf) in zip(boxes, recognized):
        if text and conf >= config.OCR_MIN_CONF:
            lines.append(
                {"text": text, "score": round(float(conf), 4), "box": box.round(1).tolist()}
            )

    return OcrOut(lines=lines, text="\n".join(line["text"] for line in lines)).model_dump()


def warm() -> dict:
    loaded = []
    if config.WARM_REMBG:
        _session(config.BG_REMOVE_MODEL)
        loaded.append(f"rembg:{config.BG_REMOVE_MODEL}")
    if config.WARM_OCR and ocr_available():
        _engine()
        loaded.append("ppocr")
    return {
        "loaded": loaded,
        "ocr_available": ocr_available(),
        "rnnoise": os.path.isfile(config.RNNOISE_MODEL),
    }
