"""cpu_tasks configuration. No GPU, no torch — this pod is cheap and stateless."""

from __future__ import annotations

import os

DATA_DIR = os.environ.get("DATA_DIR") or ("/workspace" if os.path.isdir("/workspace") else "/tmp/crm-data")

# --- numba / pymatting JIT cache -------------------------------------------
# `import rembg` pulls in pymatting -> numba, which JIT-compiles at import time
# and pegs every core for minutes. Measured cold on a 4-core container: several
# minutes of 100%-per-core compilation before a single request is served.
#
# Numba caches the compiled artifacts, but only if it has somewhere durable to
# put them. Pointing it at the network volume turns that into a one-time cost
# instead of a tax on every pod restart. This MUST be set before rembg is
# imported anywhere, which is why it lives at the top of config.
NUMBA_CACHE_DIR = os.environ.get("NUMBA_CACHE_DIR") or os.path.join(DATA_DIR, "numba")
os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)

# Cap the thread pools these libraries spawn. Left unset they each size
# themselves to the host's core count, oversubscribe a shared CPU pod, and make
# everything slower rather than faster.
_threads = os.environ.get("CPU_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, _threads)

# --- background removal ----------------------------------------------------
BG_REMOVE_MODEL = os.environ.get("BG_REMOVE_MODEL", "u2net")
# rembg caches its ONNX weights here; on the volume so a restart is instant.
os.environ.setdefault("U2NET_HOME", os.path.join(DATA_DIR, "rembg"))
os.makedirs(os.environ["U2NET_HOME"], exist_ok=True)

# --- audio denoise ---------------------------------------------------------
RNNOISE_MODEL = os.environ.get("RNNOISE_MODEL", "/app/models/rnnoise.rnnn")

# --- Groq (LLM + the primary object-detection path) ------------------------
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", GROQ_MODEL)
GROQ_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "120"))
# Groq sits behind Cloudflare, which 403s the default Python HTTP client UA
# (error 1010) — the request is blocked at the edge before the key is checked.
GROQ_USER_AGENT = os.environ.get(
    "GROQ_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

# --- OCR -------------------------------------------------------------------
OCR_MODEL_DIR = os.environ.get("OCR_MODEL_DIR") or os.path.join(DATA_DIR, "ocr")
os.environ.setdefault("OCR_MODEL_DIR", OCR_MODEL_DIR)
OCR_LIMIT_SIDE = int(os.environ.get("OCR_LIMIT_SIDE", "1280"))
OCR_MIN_CONF = float(os.environ.get("OCR_MIN_CONF", "0.3"))
OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() == "true"

WARM_REMBG = os.environ.get("WARM_REMBG", "true").lower() == "true"
WARM_OCR = os.environ.get("WARM_OCR", "true").lower() == "true"
