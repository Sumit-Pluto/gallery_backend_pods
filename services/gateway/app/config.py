"""Gateway configuration — the only place that knows where the pods live.

The client never sees any of this. They get CRM_AI_BASE_URL + CRM_AI_API_KEY and
nothing else, which is what lets you move, resize or rebuild a pod without them
changing a line.
"""

from __future__ import annotations

import os

# --- upstream pods ---------------------------------------------------------
# On RunPod each pod's HTTP port is reachable at
#   https://<POD_ID>-8000.proxy.runpod.net
# Those URLs are public, which is why every backend service also requires
# INTERNAL_API_KEY. Once your pods sit on RunPod Global Networking you can swap
# these for the private `<pod>.runpod.internal` names and drop the exposure.
VISION_URL = os.environ.get("VISION_URL", "").rstrip("/")
DIFFUSION_URL = os.environ.get("DIFFUSION_URL", "").rstrip("/")
CPU_URL = os.environ.get("CPU_URL", "").rstrip("/")

# The pre-existing chat pod (Qwen via Ollama + NLLB translate). Not managed by
# this repo — the gateway only fronts it so the client has one base URL.
CHAT_URL = os.environ.get("CHAT_URL", "").rstrip("/")
TRANSLATE_URL = os.environ.get("TRANSLATE_URL", "").rstrip("/")
CHAT_POD_KEY = os.environ.get("CHAT_POD_KEY", "")

# --- timeouts --------------------------------------------------------------
# Generous, because these are model inferences behind a queue, not web requests.
TIMEOUT_FAST = float(os.environ.get("TIMEOUT_FAST", "120"))     # remove-bg, denoise, ocr, detect
TIMEOUT_VISION = float(os.environ.get("TIMEOUT_VISION", "600")) # upscale, transcribe
TIMEOUT_DIFFUSION = float(os.environ.get("TIMEOUT_DIFFUSION", "900"))
TIMEOUT_CHAT = float(os.environ.get("TIMEOUT_CHAT", "300"))
UPSTREAM_RETRIES = int(os.environ.get("UPSTREAM_RETRIES", "2"))

# --- jobs ------------------------------------------------------------------
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "1800"))
JOB_MAX_QUEUED = int(os.environ.get("JOB_MAX_QUEUED", "64"))
JOB_MAX_STORED = int(os.environ.get("JOB_MAX_STORED", "500"))
# How long POST /v1/image/edit?wait=N may block before falling back to a job id.
SYNC_WAIT_MAX = float(os.environ.get("SYNC_WAIT_MAX", "60"))

# --- detection -------------------------------------------------------------
# "vlm"  -> Qwen vision on the cpu pod (default, no GPU cost, open vocabulary)
# "yolo" -> the construction-trained YOLO on the vision pod
DETECT_BACKEND = os.environ.get("DETECT_BACKEND", "vlm").lower()

# Automatic VLM -> YOLO fallback, now OFF by default. Two reasons it stopped
# being the right default:
#
#   1. It answers a different question. YOLO has a fixed class list; the product
#      needs open-vocabulary labels ("clock", "house") that it cannot produce. A
#      silent fallback degraded the label vocabulary without telling anyone.
#   2. It is a cascade risk. YOLO detect runs through the vision pod's
#      GPU_CONCURRENCY=1 semaphore, the same one serving upscale and transcribe.
#      A provider rate limit during a bulk upload would redirect every detection
#      onto that queue and stall unrelated GPU work with it.
#
# Redundancy now lives in the cpu pod's provider chain (Model Studio -> Groq),
# which fails over without changing the answer's shape. Turn this back on only
# if you accept a narrower label set during an outage.
DETECT_FALLBACK_TO_YOLO = os.environ.get("DETECT_FALLBACK_TO_YOLO", "false").lower() == "true"

# A rate limit means "retry shortly", not "burn the GPU" — so these codes never
# trigger the fallback even when it is enabled.
DETECT_NO_FALLBACK_CODES = {"rate_limited", "queue_full", "gpu_busy"}

# Upper bound the gateway will forward to the cpu pod's /detect-batch. Keep it at
# or below the pod's own DETECT_MAX_BATCH.
DETECT_MAX_BATCH = int(os.environ.get("DETECT_MAX_BATCH", "32"))

# --- auth / limits ---------------------------------------------------------
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "30"))
# Comma-separated origins. Empty (the default) = CORS off = backend-to-backend
# only, which is the posture the API key assumes.
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
