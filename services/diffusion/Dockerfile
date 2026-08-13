# crm-diffusion — FLUX.2 [klein] img2img + SDXL inpaint.
#
# Slim Python base rather than nvidia/cuda: torch's cu124 wheels bundle their own
# CUDA runtime and cuDNN, and RunPod mounts the host driver. That halves the
# image, which matters because container disk is billed and pull time is downtime.
# No CUDA version pinned in the base = runs on any recent driver.
# Build context = repo root:  docker build -f services/diffusion/Dockerfile .
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE=diffusion \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgomp1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu124

COPY services/diffusion/requirements.txt .
RUN pip install -r requirements.txt

COPY libs/common /libs/common
RUN pip install /libs/common

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY services/diffusion/app /app/app

ARG BUILD_SHA=dev
ENV BUILD_SHA=${BUILD_SHA}

EXPOSE 8000
# start-period is long on purpose: FLUX + SDXL stream ~20 GB off the network
# volume on a cold pod. Killing the container mid-load would loop forever.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/entrypoint.sh"]
