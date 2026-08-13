"""FastAPI app factory shared by all four services.

Every service gets the same middleware, error envelope, health contract and
internal auth from three lines of setup, so `vision`, `diffusion` and `cpu_tasks`
differ only in the models they load.

Health contract (the gateway and your uptime monitor both rely on it):

  GET /healthz  -> 200 as soon as the process is up. Liveness only. If this stops
                   answering, the container is wedged and should be restarted.
  GET /readyz   -> 503 until warmup finishes, then 200. The gateway refuses to
                   route to a pod that is not ready, so a restarting diffusion pod
                   returns a clean 503 to the client instead of a 90 s hang.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .errors import ApiError, NotReady, Unauthorized
from .logging import new_request_id, request_id_var, setup_logging
from .security import env_keys, key_matches

MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(32 * 1024 * 1024)))
ALLOW_INSECURE = os.environ.get("ALLOW_INSECURE", "false").lower() == "true"


class Readiness:
    """Warmup state for one service."""

    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.detail: dict = {}
        self.started_at = time.time()
        self.ready_at: float | None = None

    def mark_ready(self, **detail) -> None:
        self.detail.update(detail)
        self.ready = True
        self.error = None
        self.ready_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.ready = False
        self.error = error

    def snapshot(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "uptime_s": round(time.time() - self.started_at, 1),
            "warmup_s": round(self.ready_at - self.started_at, 1) if self.ready_at else None,
            **self.detail,
        }


def require_ready(readiness: Readiness) -> None:
    if not readiness.ready:
        raise NotReady(readiness.error or "Models are still loading; retry in a few seconds.")


def create_app(
    service: str,
    *,
    readiness: Readiness | None = None,
    warmup: Callable[[], Awaitable[None]] | None = None,
    extra_health: Callable[[], dict] | None = None,
    internal_auth: bool = True,
    on_startup: Callable[[], Awaitable[None]] | None = None,
    on_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    log = setup_logging(service)
    readiness = readiness or Readiness()
    internal_keys = env_keys("INTERNAL_API_KEY")

    if internal_auth and not internal_keys and not ALLOW_INSECURE:
        log.critical(
            "INTERNAL_API_KEY is not set — every request will be rejected. "
            "Set it on the pod, or set ALLOW_INSECURE=true for local development only."
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = None
        if warmup is not None:
            # Warm as a task, not inline: the port binds immediately so /healthz
            # answers while a 16 GB model set streams off the network volume.
            async def runner():
                try:
                    log.info("warmup started")
                    await warmup()
                    log.info("warmup complete", extra=readiness.snapshot())
                except Exception as exc:
                    readiness.mark_failed(f"{type(exc).__name__}: {exc}")
                    log.exception("warmup failed")

            task = asyncio.create_task(runner())
        if on_startup is not None:
            await on_startup()
        try:
            yield
        finally:
            if task is not None and not task.done():
                task.cancel()
            if on_shutdown is not None:
                await on_shutdown()

    app = FastAPI(
        title=f"crm-{service}",
        version=os.environ.get("BUILD_SHA", "dev"),
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.readiness = readiness
    app.state.service = service

    # Middleware registration order is inverted at runtime: Starlette inserts
    # each new one at the front, so the LAST registered runs OUTERMOST. Internal
    # auth is registered first on purpose — that puts observability outside it,
    # so a rejected request still gets a request id and still shows up in the
    # logs. Auth failures you cannot trace are the ones that waste an afternoon.
    if internal_auth:

        @app.middleware("http")
        async def _internal_auth(request: Request, call_next):
            if request.url.path in ("/healthz", "/readyz"):
                return await call_next(request)
            if ALLOW_INSECURE:
                return await call_next(request)
            presented = request.headers.get("X-Internal-Key")
            if not key_matches(presented, internal_keys):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"code": "unauthorized",
                                       "message": "This service is reachable only through the gateway."}},
                )
            return await call_next(request)

    @app.middleware("http")
    async def _observability(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_var.set(rid)
        started = time.monotonic()
        try:
            length = request.headers.get("content-length")
            if length and int(length) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"code": "payload_too_large",
                                       "message": f"Body exceeds {MAX_BODY_BYTES} bytes."}},
                    headers={"X-Request-ID": rid},
                )
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path})
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "internal_error", "message": "Internal error."}},
                headers={"X-Request-ID": rid},
            )
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        if request.url.path not in ("/healthz", "/readyz"):
            log.info(
                "request",
                extra={
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
        return response

    # One error envelope for everything the caller can see. FastAPI's defaults
    # ({"detail": ...}) would otherwise leak two extra shapes — a validation
    # array and a bare string — which is exactly the kind of inconsistency that
    # makes a client write three parsers.
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        errors = [
            {"field": ".".join(str(p) for p in err.get("loc", ())[1:]) or "body",
             "message": err.get("msg", "invalid")}
            for err in exc.errors()
        ]
        first = errors[0] if errors else {"field": "body", "message": "invalid"}
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": f"{first['field']}: {first['message']}",
                    "detail": errors,
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found",
                405: "method_not_allowed", 413: "payload_too_large"}.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail)}},
            headers=getattr(exc, "headers", None),
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        body = {"ok": True, "service": service, "build": os.environ.get("BUILD_SHA", "dev")}
        if extra_health:
            try:
                body.update(extra_health())
            except Exception as exc:  # health must never throw
                body["health_error"] = f"{type(exc).__name__}: {exc}"
        return body

    @app.get("/readyz", include_in_schema=False)
    async def readyz():
        snap = readiness.snapshot()
        if not readiness.ready:
            return JSONResponse(status_code=503, content=snap)
        return snap

    return app


async def verify_internal(x_internal_key: str = Header(default="")) -> None:
    """Route-level fallback when a service opts out of the global middleware."""
    if ALLOW_INSECURE:
        return
    if not key_matches(x_internal_key, env_keys("INTERNAL_API_KEY")):
        raise Unauthorized("This service is reachable only through the gateway.")
