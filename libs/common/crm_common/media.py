"""Media IO shared by every service: base64 <-> bytes <-> PIL, plus URL fetch.

This replaces the three near-identical `_decode_b64` / `_b64_img` / `_to_b64` /
`_fit` helpers that were copy-pasted across the old gpu-vision, gpu-diffusion and
cpu-tasks handlers.

Two behaviours worth knowing:

* Every `image` / `audio` field accepts **either** raw base64 (with or without a
  `data:` prefix) **or** an http(s) URL. Base64 in JSON costs ~33% overhead, so
  large media should come in by URL.
* URL fetching is SSRF-guarded: the resolved address must be public. Without that
  a caller could make the pod fetch `http://169.254.169.254/...` or reach a
  sibling pod on the private network.
"""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import os
import socket
from urllib.parse import urlparse

import httpx
from PIL import Image

from .errors import BadRequest, PayloadTooLarge

MAX_MEDIA_BYTES = int(os.environ.get("MAX_MEDIA_BYTES", str(25 * 1024 * 1024)))
FETCH_TIMEOUT = float(os.environ.get("MEDIA_FETCH_TIMEOUT", "30"))
ALLOW_PRIVATE_FETCH = os.environ.get("ALLOW_PRIVATE_MEDIA_FETCH", "false").lower() == "true"

# Pillow refuses absurd images by default (decompression-bomb guard). Keep it on
# but raise the ceiling to something a 4x upscale source can legitimately hit.
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(64_000_000)))


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _assert_public_url(url: str) -> None:
    if ALLOW_PRIVATE_FETCH:
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BadRequest(f"Unsupported URL scheme '{parsed.scheme}'.")
    host = parsed.hostname
    if not host:
        raise BadRequest("URL has no host.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise BadRequest(f"Could not resolve '{host}': {exc}") from exc
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            raise BadRequest(f"Refusing to fetch a non-public address ({addr}).")


def fetch_url(url: str, *, max_bytes: int = MAX_MEDIA_BYTES) -> bytes:
    _assert_public_url(url)
    try:
        with httpx.stream("GET", url, timeout=FETCH_TIMEOUT, follow_redirects=True) as resp:
            if resp.status_code >= 400:
                raise BadRequest(f"Fetching media failed with HTTP {resp.status_code}.")
            chunks, total = [], 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise PayloadTooLarge(f"Remote media exceeds {max_bytes} bytes.")
                chunks.append(chunk)
            return b"".join(chunks)
    except httpx.HTTPError as exc:
        raise BadRequest(f"Fetching media failed: {type(exc).__name__}: {exc}") from exc


def decode_media(value: str, *, field: str = "media", max_bytes: int = MAX_MEDIA_BYTES) -> bytes:
    """base64 (optionally `data:` prefixed) or an http(s) URL -> raw bytes."""
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"Missing '{field}'.")
    value = value.strip()

    if _is_url(value):
        return fetch_url(value, max_bytes=max_bytes)

    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    # A base64 payload is ~4/3 the size of the bytes it encodes; reject before
    # allocating rather than after.
    if len(value) > max_bytes * 4 // 3 + 1024:
        raise PayloadTooLarge(f"'{field}' exceeds the {max_bytes} byte limit.")
    try:
        raw = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise BadRequest(f"'{field}' is not valid base64: {exc}") from exc
    if not raw:
        raise BadRequest(f"'{field}' decoded to zero bytes.")
    if len(raw) > max_bytes:
        raise PayloadTooLarge(f"'{field}' exceeds the {max_bytes} byte limit.")
    return raw


def load_image(value: str, *, mode: str = "RGB", field: str = "image") -> Image.Image:
    raw = decode_media(value, field=field)
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise BadRequest(f"'{field}' is not a decodable image: {type(exc).__name__}") from exc
    return img.convert(mode) if img.mode != mode else img


def encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_bytes(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def fit(img: Image.Image, max_size: int, multiple: int = 8) -> Image.Image:
    """Downscale so the long edge is <= max_size, snapped to a multiple.

    Diffusion pipelines need dimensions divisible by 8 (SDXL) or 16 (FLUX); this
    is the same `_fit` the old diffusion handler used, kept identical so output
    framing does not change after the migration.
    """
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    nw = max(multiple, (int(w * scale) // multiple) * multiple)
    nh = max(multiple, (int(h * scale) // multiple) * multiple)
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.LANCZOS)
