#!/usr/bin/env python3
"""End-to-end smoke test against a live gateway.

Run this after every deploy, before telling anyone it works.

    python scripts/smoke_test.py \
        --base-url https://<gateway-pod>-8000.proxy.runpod.net \
        --api-key  <client key>

It generates its own image and audio, so there are no fixtures to ship. Add
--concurrency to fire N simultaneous upscales: that is the test that proves the
GPU semaphore is holding, which is the single most likely way this stack falls
over under real traffic.

Exit code is non-zero if any required check fails, so it drops straight into CI
or a post-deploy hook.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import struct
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("pip install Pillow")


GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def make_image(width: int = 256, height: int = 256) -> str:
    """A small image with shapes and text, so detection/OCR have something to find."""
    img = Image.new("RGB", (width, height), (28, 32, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 120, 110], fill=(220, 180, 60))
    draw.ellipse([140, 40, 230, 130], fill=(70, 140, 220))
    draw.rectangle([30, 150, 226, 200], fill=(240, 240, 240))
    draw.text((45, 168), "INVOICE 2024", fill=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_mask(width: int = 256, height: int = 256) -> str:
    """White = regenerate. A centred blob."""
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).ellipse([90, 90, 180, 180], fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_audio(seconds: float = 2.0, rate: int = 16000) -> str:
    """A 440 Hz tone with a little noise — enough for denoise and Whisper to chew on."""
    frames = bytearray()
    for i in range(int(rate * seconds)):
        value = 0.4 * math.sin(2 * math.pi * 440 * i / rate)
        value += 0.05 * math.sin(2 * math.pi * 3300 * i / rate)
        frames += struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode()


class Runner:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.client = httpx.Client(timeout=timeout)
        self.results: list[tuple[str, bool, str, float]] = []

    def check(self, name: str, method: str, path: str, payload=None, *,
              required: bool = True, expect: int = 200, validate=None) -> dict | None:
        started = time.monotonic()
        try:
            if method == "GET":
                response = self.client.get(f"{self.base}{path}", headers=self.headers)
            else:
                response = self.client.post(f"{self.base}{path}", headers=self.headers, json=payload)
            elapsed = time.monotonic() - started

            if response.status_code != expect:
                body = response.text[:200].replace("\n", " ")
                self._record(name, False, f"HTTP {response.status_code}: {body}", elapsed, required)
                return None

            data = response.json() if response.content else {}
            if validate:
                problem = validate(data)
                if problem:
                    self._record(name, False, problem, elapsed, required)
                    return None
            self._record(name, True, "", elapsed, required)
            return data
        except Exception as exc:
            self._record(name, False, f"{type(exc).__name__}: {exc}",
                         time.monotonic() - started, required)
            return None

    def _record(self, name, ok, detail, elapsed, required):
        self.results.append((name, ok, detail, elapsed))
        mark = f"{GREEN}PASS{RESET}" if ok else (f"{RED}FAIL{RESET}" if required else f"{YELLOW}SKIP{RESET}")
        line = f"  {mark}  {name:<34} {DIM}{elapsed:6.2f}s{RESET}"
        if detail:
            line += f"\n        {DIM}{detail}{RESET}"
        print(line, flush=True)
        if not required and not ok:
            self.results[-1] = (name, True, detail, elapsed)  # optional failures do not fail the run

    def poll_job(self, name: str, job_id: str, timeout: float = 600) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(f"{self.base}/v1/jobs/{job_id}", headers=self.headers)
            if response.status_code != 200:
                self._record(name, False, f"poll HTTP {response.status_code}", 0, True)
                return None
            body = response.json()
            if body["status"] == "done":
                return body["result"]
            if body["status"] == "error":
                self._record(name, False, json.dumps(body.get("error"))[:200], 0, True)
                return None
            time.sleep(2)
        self._record(name, False, f"job did not finish in {timeout}s", 0, True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Fire N simultaneous upscales to prove the GPU semaphore holds.")
    parser.add_argument("--skip-diffusion", action="store_true",
                        help="Skip the slow FLUX/SDXL jobs.")
    args = parser.parse_args()

    runner = Runner(args.base_url, args.api_key, args.timeout)
    image, mask, audio = make_image(), make_mask(), make_audio()

    print(f"\n{DIM}target {args.base_url}{RESET}\n")

    print("health")
    runner.check("healthz", "GET", "/healthz")
    status = runner.check("status (all pods)", "GET", "/v1/status", expect=200)
    if status is None:
        status = runner.check("status (degraded)", "GET", "/v1/status", expect=207, required=False)
    if status:
        for name, info in (status.get("upstreams") or {}).items():
            if info.get("configured"):
                flag = f"{GREEN}ready{RESET}" if info.get("ready") else f"{RED}NOT READY{RESET}"
                print(f"        {DIM}{name:<10}{RESET} {flag}")

    print("\nauth")
    bad = httpx.Client(timeout=30).post(
        f"{args.base_url.rstrip('/')}/v1/image/upscale",
        json={"image": image}, headers={"X-API-Key": "definitely-wrong"},
    )
    ok = bad.status_code == 401
    print(f"  {GREEN + 'PASS' + RESET if ok else RED + 'FAIL' + RESET}  "
          f"{'rejects a bad key':<34} {DIM}(got {bad.status_code}){RESET}")
    runner.results.append(("auth rejects bad key", ok, "", 0.0))

    print("\ncpu pod")
    runner.check("remove-bg", "POST", "/v1/image/remove-bg", {"image": image},
                 validate=lambda d: None if d.get("image") else "no image in response")
    runner.check("audio denoise", "POST", "/v1/audio/denoise", {"audio": audio},
                 validate=lambda d: None if d.get("audio") else "no audio in response")
    runner.check("ocr", "POST", "/v1/ocr", {"image": image}, required=False,
                 validate=lambda d: None if "lines" in d else "no lines in response")
    runner.check("detect (vlm)", "POST", "/v1/vision/detect", {"image": image},
                 validate=lambda d: None if "detections" in d else "no detections key")
    runner.check("llm chat", "POST", "/v1/llm/chat", required=False, payload={
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 5,
    })

    print("\nvision pod")
    runner.check("upscale x2", "POST", "/v1/image/upscale", {"image": image, "scale": 2},
                 validate=lambda d: None if d.get("image") else "no image in response")
    runner.check("transcribe", "POST", "/v1/audio/transcribe", {"audio": audio},
                 validate=lambda d: None if "transcript" in d else "no transcript key")
    runner.check("detect (yolo fallback)", "POST", "/v1/vision/detect?backend=yolo",
                 {"image": image}, required=False)

    print("\nchat pod (pre-existing)")
    runner.check("translate", "POST", "/v1/translate", required=False,
                 payload={"text": "Hello", "source": "eng_Latn", "target": "hin_Deva"})
    runner.check("qwen chat", "POST", "/v1/chat/completions", required=False, payload={
        "messages": [{"role": "user", "content": "Say ok"}], "max_tokens": 5,
    })

    if not args.skip_diffusion:
        print("\ndiffusion pod (async jobs)")
        accepted = runner.check("edit: colorize -> 202", "POST", "/v1/image/edit",
                                {"image": image, "op": {"type": "colorize"}}, expect=202)
        if accepted:
            result = runner.poll_job("edit: colorize result", accepted["job_id"])
            if result and result.get("image"):
                runner._record("edit: colorize result", True, "", 0.0, True)

        accepted = runner.check("edit: magic-eraser -> 202", "POST", "/v1/image/edit",
                                {"image": image, "mask": mask, "op": {"type": "magic-eraser"}},
                                expect=202)
        if accepted:
            result = runner.poll_job("edit: magic-eraser result", accepted["job_id"])
            if result and result.get("image"):
                runner._record("edit: magic-eraser result", True, "", 0.0, True)

    if args.concurrency > 1:
        print(f"\nconcurrency ({args.concurrency} simultaneous upscales)")
        print(f"  {DIM}all should succeed — the GPU semaphore serialises them{RESET}")

        def one(i):
            started = time.monotonic()
            response = httpx.Client(timeout=args.timeout).post(
                f"{args.base_url.rstrip('/')}/v1/image/upscale",
                json={"image": image, "scale": 2},
                headers={"X-API-Key": args.api_key},
            )
            return i, response.status_code, time.monotonic() - started

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            outcomes = list(pool.map(one, range(args.concurrency)))
        failed = [o for o in outcomes if o[1] != 200]
        slowest = max(o[2] for o in outcomes)
        ok = not failed
        print(f"  {GREEN + 'PASS' + RESET if ok else RED + 'FAIL' + RESET}  "
              f"{'no OOM under concurrency':<34} {DIM}slowest {slowest:.1f}s, "
              f"{len(failed)} failed{RESET}")
        for i, code, _ in failed[:5]:
            print(f"        {DIM}request {i} -> HTTP {code}{RESET}")
        runner.results.append(("concurrency", ok, "", slowest))

    passed = sum(1 for _, ok, _, _ in runner.results if ok)
    total = len(runner.results)
    failures = [name for name, ok, _, _ in runner.results if not ok]
    print(f"\n{'-' * 60}")
    if failures:
        print(f"{RED}{passed}/{total} passed{RESET} — failed: {', '.join(failures)}\n")
        return 1
    print(f"{GREEN}{passed}/{total} passed{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
