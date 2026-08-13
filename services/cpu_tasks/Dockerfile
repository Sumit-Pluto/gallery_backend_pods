# crm-cpu — rembg, ffmpeg denoise, PP-OCRv6, Groq proxy, VLM detection.
# No CUDA. Build context = repo root: docker build -f services/cpu_tasks/Dockerfile .
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE=cpu_tasks \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# RNNoise model for ffmpeg's arnndn. If the download fails the service falls back
# to afftdn at runtime, so a flaky build network does not break the image.
RUN mkdir -p /app/models && \
    (curl -fsSL -o /app/models/rnnoise.rnnn \
      https://raw.githubusercontent.com/GregorR/rnnoise-models/master/beguiling-drafts-2018-08-30/bd.rnnn || \
     echo "rnnoise download skipped; afftdn fallback will be used")

WORKDIR /app

RUN pip install --upgrade pip
COPY services/cpu_tasks/requirements.txt .
RUN pip install -r requirements.txt

COPY libs/common /libs/common
RUN pip install /libs/common

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY services/cpu_tasks/app /app/app

ARG BUILD_SHA=dev
ENV BUILD_SHA=${BUILD_SHA}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/entrypoint.sh"]
