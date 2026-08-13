"""Structured JSON logging with a request id that follows a call across pods.

The gateway generates an id per request and forwards it as `X-Request-ID`; each
backend service picks the header back up, so one grep of `request_id` reconstructs
the whole chain (gateway -> diffusion -> back) even though they are separate pods
with separate log streams.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Anything passed via logger.info("...", extra={"k": v}) rides along.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(service: str) -> logging.Logger:
    """Install the JSON formatter on the root logger. Idempotent."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own colourised handlers; force them through ours so
    # the pod's stdout is a single parseable stream.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # Silence uvicorn's access log outright. Our own middleware already logs
    # every request with the request id, status and duration, so letting this
    # through doubles log volume for strictly less information.
    #
    # This has to be done here rather than with uvicorn's --no-access-log:
    # uvicorn configures logging before importing the app, so setup_logging()
    # runs afterwards and would otherwise re-enable propagation.
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    access.disabled = True

    return logging.getLogger(service)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]
