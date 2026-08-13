"""Lazy, cached model loaders.

Same singleton pattern the serverless handler used, with two changes for pods:

* weights live on the network volume, so a restart is seconds not minutes;
* `warm()` pulls the ones we actually serve at boot, so request #1 is not the
  one that pays a 90 s load.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_yolo = None
_upsampler = None
_face = None
_whisper = None


def _download(url: str, dest: str) -> str:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    log.info("downloading weights", extra={"url": url, "dest": dest})
    tmp = dest + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def upsampler():
    global _upsampler
    if _upsampler is None:
        with _lock:
            if _upsampler is None:
                import torch
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer

                path = _download(config.ESRGAN_URL, os.path.join(config.WEIGHTS_DIR, "RealESRGAN_x4plus.pth"))
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                _upsampler = RealESRGANer(
                    scale=4,
                    model_path=path,
                    model=net,
                    tile=config.ESRGAN_TILE,
                    tile_pad=config.ESRGAN_TILE_PAD,
                    pre_pad=0,
                    half=torch.cuda.is_available(),
                    gpu_id=None,
                )
                log.info("real-esrgan ready", extra={"tile": config.ESRGAN_TILE})
    return _upsampler


def face_enhancer():
    global _face
    if _face is None:
        with _lock:
            if _face is None:
                from gfpgan import GFPGANer

                path = _download(config.GFPGAN_URL, os.path.join(config.WEIGHTS_DIR, "GFPGANv1.4.pth"))
                _face = GFPGANer(
                    model_path=path,
                    upscale=4,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=upsampler(),
                )
                log.info("gfpgan ready")
    return _face


def whisper():
    global _whisper
    if _whisper is None:
        with _lock:
            if _whisper is None:
                from faster_whisper import WhisperModel

                _whisper = WhisperModel(
                    config.WHISPER_MODEL,
                    device=config.WHISPER_DEVICE,
                    compute_type=config.WHISPER_COMPUTE,
                    download_root=config.HF_HOME,
                )
                log.info("whisper ready", extra={"model": config.WHISPER_MODEL})
    return _whisper


def yolo():
    """Fallback detector. Never warmed — costs zero VRAM unless called."""
    global _yolo
    if _yolo is None:
        with _lock:
            if _yolo is None:
                from ultralytics import YOLO

                if not os.path.isfile(config.YOLO_MODEL):
                    from crm_common.errors import ApiError

                    raise ApiError(
                        f"YOLO weights not found at {config.YOLO_MODEL}. Upload them to the "
                        "network volume (see deploy/volume/README.md) or set YOLO_MODEL.",
                        status=503,
                        code="model_unavailable",
                    )
                _yolo = YOLO(config.YOLO_MODEL)
                log.info("yolo ready (fallback path)", extra={"weights": config.YOLO_MODEL})
    return _yolo


def warm() -> dict:
    """Preload everything this pod actually serves. Called once at startup."""
    loaded = []
    if config.WARM_UPSCALE:
        upsampler()
        loaded.append("real-esrgan")
    if config.WARM_WHISPER:
        whisper()
        loaded.append(f"whisper:{config.WHISPER_MODEL}")
    return {"loaded": loaded, "yolo": "lazy (fallback only)"}
