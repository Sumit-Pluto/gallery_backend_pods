# crm-vision — Real-ESRGAN + GFPGAN + Whisper (+ YOLO fallback).
# CUDA 12.1 + torch cu121, matching the wheels realesrgan/basicsr were built
# against. Build context = repo root:  docker build -f services/vision/Dockerfile .
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE=vision \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first and pinned: pip would otherwise resolve a CPU wheel while
# installing realesrgan, and the pod would silently run inference on CPU.
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

COPY services/vision/requirements.txt .
RUN pip install -r requirements.txt

# basicsr imports a torchvision symbol that moved in 0.16 — the single most
# common Real-ESRGAN crash. Rewrite it at build time.
RUN F=$(python3 -c "import basicsr, os; print(os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'))") && \
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$F"

COPY libs/common /libs/common
RUN pip install /libs/common

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY services/vision/app /app/app

ARG BUILD_SHA=dev
ENV BUILD_SHA=${BUILD_SHA}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/entrypoint.sh"]
