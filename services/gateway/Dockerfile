# crm-gateway — the only pod exposed to the client.
# Build context = repo root: docker build -f services/gateway/Dockerfile .
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE=gateway \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip
COPY services/gateway/requirements.txt .
RUN pip install -r requirements.txt

COPY libs/common /libs/common
RUN pip install /libs/common

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY services/gateway/app /app/app

ARG BUILD_SHA=dev
ENV BUILD_SHA=${BUILD_SHA}

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/entrypoint.sh"]
