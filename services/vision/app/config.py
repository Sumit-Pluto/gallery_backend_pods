"""Vision service configuration.

Imported first by every other module in this service because it sets HF_HOME and
friends *before* faster-whisper / ultralytics get imported. Get that ordering
wrong and the models land on the container disk instead of the network volume,
which means a 3 GB re-download on every pod restart.
"""

from __future__ import annotations

import os

# The RunPod network volume. `/runpod-volume` was the serverless mount point;
# pods mount at `/workspace`. DATA_DIR keeps both (and local dev) working.
DATA_DIR = os.environ.get("DATA_DIR") or ("/workspace" if os.path.isdir("/workspace") else "/tmp/crm-data")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR") or os.path.join(DATA_DIR, "weights")
HF_HOME = os.environ.get("HF_HOME") or os.path.join(DATA_DIR, "huggingface")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TORCH_HOME", os.path.join(DATA_DIR, "torch"))
os.environ.setdefault("ULTRALYTICS_DIR", os.path.join(DATA_DIR, "ultralytics"))

# --- Real-ESRGAN -----------------------------------------------------------
# tile=0 means "process the whole image in one tensor", which makes peak VRAM
# scale with input resolution — a 1024px source at 4x allocates a 4096x4096
# activation and can spike past 12 GB. Tiling caps it at a flat ~2-3 GB for a
# ~10-15% throughput cost, which is what lets this run on a 16 GB card.
ESRGAN_TILE = int(os.environ.get("ESRGAN_TILE", "400"))
ESRGAN_TILE_PAD = int(os.environ.get("ESRGAN_TILE_PAD", "10"))
ESRGAN_URL = os.environ.get(
    "ESRGAN_URL",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
)
GFPGAN_URL = os.environ.get(
    "GFPGAN_URL",
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
)

# --- Whisper ---------------------------------------------------------------
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")

# --- YOLO (fallback only) --------------------------------------------------
# Object detection now runs through the Qwen vision model on the cpu service.
# YOLO stays for fallback, so it is loaded lazily and never warmed: it costs
# nothing until something explicitly asks for it.
YOLO_MODEL = os.environ.get("YOLO_MODEL") or os.path.join(WEIGHTS_DIR, "yolo_large_v3.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.15"))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.7"))

# Which models warmup should preload. YOLO is deliberately absent.
WARM_UPSCALE = os.environ.get("WARM_UPSCALE", "true").lower() == "true"
WARM_WHISPER = os.environ.get("WARM_WHISPER", "true").lower() == "true"
