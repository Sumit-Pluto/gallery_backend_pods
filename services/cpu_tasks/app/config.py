"""cpu_tasks configuration. No GPU, no torch — this pod is cheap and stateless."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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


# --------------------------------------------------------------------------- #
# LLM / VLM providers
# --------------------------------------------------------------------------- #
# Both providers speak the OpenAI /chat/completions shape, so the transport is
# identical and only the base URL, keys and model names differ. They are tried
# in LLM_PROVIDERS order, which is what turns "Groq is rate-limiting us" from an
# outage into a slower request.
#
# Model names are NOT portable between providers (`qwen-vl-plus` on Model Studio
# vs `qwen/qwen3.6-27b` on Groq), so each provider carries its own.


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    keys_env: tuple[str, ...]
    model: str
    vision_model: str
    # Groq sits behind Cloudflare, which 403s the default Python HTTP client UA
    # (error 1010) — the request is blocked at the edge before the key is even
    # checked. Model Studio has no such quirk, hence per-provider.
    user_agent: str | None = None
    timeout: float = 120.0
    extra_headers: dict = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and _first_env(self.keys_env))


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value
    return ""


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Alibaba Cloud Model Studio (DashScope), OpenAI-compatible mode.
#   Singapore  https://dashscope-intl.aliyuncs.com/compatible-mode/v1
#   Virginia   https://dashscope-us.aliyuncs.com/compatible-mode/v1
#   Beijing    https://dashscope.aliyuncs.com/compatible-mode/v1
# Pick the region deliberately — Beijing is cheaper and is the wrong answer if
# any client of yours has a data-locality clause.
DASHSCOPE = Provider(
    name="dashscope",
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/"),
    keys_env=("DASHSCOPE_API_KEYS", "DASHSCOPE_API_KEY"),
    model=os.environ.get("DASHSCOPE_MODEL", "qwen-plus"),
    vision_model=os.environ.get("DASHSCOPE_VISION_MODEL", "qwen-vl-plus"),
    timeout=float(os.environ.get("DASHSCOPE_TIMEOUT", "120")),
)

GROQ = Provider(
    name="groq",
    base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
    keys_env=("GROQ_API_KEYS", "GROQ_API_KEY"),
    model=os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b"),
    vision_model=os.environ.get("GROQ_VISION_MODEL", os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")),
    user_agent=os.environ.get("GROQ_USER_AGENT", _BROWSER_UA),
    timeout=float(os.environ.get("GROQ_TIMEOUT", "120")),
)

_REGISTRY = {p.name: p for p in (DASHSCOPE, GROQ)}

# Ordered failover chain. Model Studio first: first-party Qwen, production
# status, cheaper per token, and limits you can raise on a paid account.
LLM_PROVIDER_ORDER = [
    name.strip().lower()
    for name in os.environ.get("LLM_PROVIDERS", "dashscope,groq").split(",")
    if name.strip()
]


def providers() -> list[Provider]:
    """Configured providers, in failover order. Unknown names are ignored."""
    return [
        _REGISTRY[name]
        for name in LLM_PROVIDER_ORDER
        if name in _REGISTRY and _REGISTRY[name].configured
    ]


def provider_names() -> list[str]:
    return [p.name for p in providers()]


# --- VLM object detection --------------------------------------------------
# The gallery consumes labels only — nothing renders a bounding box — so the
# model is asked for names and confidences and nothing else. That removes the
# spatial reasoning a VLM is worst at, and takes roughly 60% off output tokens.

# Long edge the image is downscaled to before it is sent. Detection recognises;
# it does not localise, so it does not need the source resolution. Going from a
# 4032px phone photo to 768px is ~28x fewer pixels, and pixels are what a vision
# model bills for. Raise it if small objects (a clock on a far wall) are missed.
DETECT_MAX_SIDE = int(os.environ.get("DETECT_MAX_SIDE", "768"))
DETECT_JPEG_QUALITY = int(os.environ.get("DETECT_JPEG_QUALITY", "85"))

# "auto" tries strict json_schema and remembers, per provider, if it is rejected
# and json_object worked instead. Groq supports strict schemas; Model Studio's
# support varies by model, and this negotiates it at runtime rather than making
# you get it right in config.
DETECT_STRUCTURED_MODE = os.environ.get("DETECT_STRUCTURED_MODE", "auto").lower()

# Fill in a whole-image box so clients typed against the YOLO contract (the web
# SDK's DetectedObject requires `box`) keep working with no change. Nothing
# reads the value; turn this off once the client's type makes box optional.
DETECT_EMIT_BOX = os.environ.get("DETECT_EMIT_BOX", "true").lower() == "true"

# Batch fan-out. Each image is its own upstream call — batching several images
# into ONE model request saves nothing (you pay the same image tokens) and loses
# per-image error isolation. The semaphore is what stops a 200-photo upload
# burning through every key on a rate limit.
DETECT_BATCH_CONCURRENCY = int(os.environ.get("DETECT_BATCH_CONCURRENCY", "8"))
DETECT_MAX_BATCH = int(os.environ.get("DETECT_MAX_BATCH", "32"))

# Anchor vocabulary. Left unconstrained a VLM emits "hard hat", "hardhat",
# "safety helmet" and "helmet" for the same object across four photos, and each
# one becomes its own album in the client's Objects browser. Steering the model
# to a canonical list fixes that at the source instead of leaving users to clean
# it up with rename rules afterwards. New terms are still allowed.
_DEFAULT_VOCABULARY = (
    "worker,person,hard hat,safety vest,safety harness,gloves,boots,"
    "excavator,bulldozer,crane,forklift,dump truck,concrete mixer,truck,van,"
    "scaffolding,ladder,formwork,rebar,brick,cement bag,pipe,timber,"
    "tile,sand pile,gravel pile,toolbox,power tool,generator,"
    "house,building,wall,roof,window,door,staircase,floor,ceiling,fence,gate,"
    "road,pavement,sign,barrier,traffic cone,clock,light fixture,electrical panel"
)
DETECT_VOCABULARY = [
    term.strip().lower()
    for term in os.environ.get("DETECT_VOCABULARY", _DEFAULT_VOCABULARY).split(",")
    if term.strip()
]

# Cap on how much the model may return. A schema-constrained label list is
# small; this only exists so one pathological image cannot run the bill up.
DETECT_MAX_OUTPUT_TOKENS = int(os.environ.get("DETECT_MAX_OUTPUT_TOKENS", "1024"))

# --- OCR -------------------------------------------------------------------
OCR_MODEL_DIR = os.environ.get("OCR_MODEL_DIR") or os.path.join(DATA_DIR, "ocr")
os.environ.setdefault("OCR_MODEL_DIR", OCR_MODEL_DIR)
OCR_LIMIT_SIDE = int(os.environ.get("OCR_LIMIT_SIDE", "1280"))
OCR_MIN_CONF = float(os.environ.get("OCR_MIN_CONF", "0.3"))
OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() == "true"

WARM_REMBG = os.environ.get("WARM_REMBG", "true").lower() == "true"
WARM_OCR = os.environ.get("WARM_OCR", "true").lower() == "true"

# --- deprecated aliases ----------------------------------------------------
# Kept so anything still importing the old flat names keeps working through the
# migration. Prefer config.GROQ.* / config.DASHSCOPE.*.
GROQ_BASE_URL = GROQ.base_url
GROQ_MODEL = GROQ.model
GROQ_VISION_MODEL = GROQ.vision_model
GROQ_TIMEOUT = GROQ.timeout
GROQ_USER_AGENT = GROQ.user_agent
