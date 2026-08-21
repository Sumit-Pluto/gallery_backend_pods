"""crm-gateway — the single public surface of the CRM AI backend.

Everything the client team touches is here: one base URL, one API key. Behind it
the gateway owns auth, rate limiting, request validation, the op -> prompt table,
routing to the right pod, retries, and the async job queue.

The backend pods (vision / diffusion / cpu) are never exposed to the client, and
the chat pod is fronted rather than replaced.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from crm_common.service import Readiness, create_app

from . import auth, config, jobs, upstream
from .deps import require_client
from .routes import audio, image, jobs as jobs_routes, llm, vision

log = logging.getLogger(__name__)

readiness = Readiness()
# The gateway holds no models, so it is ready the moment the process is up.
# Upstream readiness is reported separately at /v1/status.
readiness.mark_ready(role="gateway")


def _health() -> dict:
    return {"jobs": jobs.store.stats(), "clients_configured": len(auth.client_keys())}


_sweeper_task: asyncio.Task | None = None


async def _startup() -> None:
    global _sweeper_task
    if not auth.client_keys():
        log.critical("CLIENT_API_KEYS is not set — every client request will be rejected.")
    missing = [
        name
        for name, up in (("vision", upstream.vision), ("diffusion", upstream.diffusion),
                         ("cpu", upstream.cpu))
        if not up.configured
    ]
    if missing:
        log.warning("upstreams not configured", extra={"missing": missing})
    _sweeper_task = asyncio.create_task(jobs.sweeper())


async def _shutdown() -> None:
    if _sweeper_task:
        _sweeper_task.cancel()
    await upstream.aclose()


app: FastAPI = create_app(
    "gateway",
    readiness=readiness,
    extra_health=_health,
    internal_auth=False,
    on_startup=_startup,
    on_shutdown=_shutdown,
)

if config.CORS_ORIGINS:
    # Off by default. Turning this on means the API key will live in a browser,
    # which is a different security model — see docs/API.md.
    log.warning("CORS enabled", extra={"origins": config.CORS_ORIGINS})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Request-ID"],
    )

app.include_router(image.router)
app.include_router(image.ocr_router)
app.include_router(vision.router)
app.include_router(audio.router)
app.include_router(llm.router)
app.include_router(jobs_routes.router)

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "crm-ai-gateway",
        "docs": "/v1/openapi.json",
        "health": "/healthz",
        "endpoints": [
            "POST /v1/vision/detect",
            "POST /v1/vision/detect-batch",
            "POST /v1/image/edit",
            "POST /v1/image/upscale",
            "POST /v1/image/remove-bg",
            "POST /v1/audio/transcribe",
            "POST /v1/audio/denoise",
            "POST /v1/ocr",
            "POST /v1/llm/chat",
            "POST /v1/chat/completions",
            "POST /v1/translate",
            "GET  /v1/jobs/{job_id}",
        ],
    }


@app.get("/v1/openapi.json", include_in_schema=False)
async def openapi_spec():
    """The machine-readable contract to hand the client team."""
    return app.openapi()


@app.get("/v1/status", include_in_schema=False, dependencies=[Depends(require_client)])
async def status():
    """Fan out to every pod's /readyz. This is the one call to make at 3am."""
    names = ["vision", "diffusion", "cpu", "chat", "translate"]
    results = await asyncio.gather(
        *(upstream.BY_NAME[n].ready() for n in names), return_exceptions=True
    )
    upstreams = {}
    for name, result in zip(names, results):
        upstreams[name] = (
            {"ready": False, "error": f"{type(result).__name__}: {result}"}
            if isinstance(result, Exception)
            else result
        )
    degraded = [n for n, r in upstreams.items() if r.get("configured") and not r.get("ready")]
    return JSONResponse(
        status_code=200 if not degraded else 207,
        content={"ok": not degraded, "degraded": degraded, "jobs": jobs.store.stats(),
                 "upstreams": upstreams},
    )
