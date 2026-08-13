# Environment reference

Full copy-paste template: [`../.env.example`](../.env.example).
Per-pod values to actually set: [`../deploy/pods/README.md`](../deploy/pods/README.md).

This page is the *why* — what each variable costs you if you get it wrong.

---

## The five that matter most

| Variable | Pod | Default | Get it wrong and… |
|---|---|---|---|
| `GPU_CONCURRENCY` | vision, diffusion | `1` | >1 puts two inferences on one card. **CUDA OOM under load.** This is the single most likely way this stack dies. |
| `ESRGAN_TILE` | vision | `400` | `0` disables tiling, so peak VRAM scales with image size. A large 4× upscale OOMs a 16 GB card. Only set `0` on 24 GB+. |
| `RESIDENCY` | diffusion | `auto` | `both` on a card that cannot hold FLUX + SDXL fails the second model load. Measure first (below), then pin. |
| `INTERNAL_API_KEY` | all four | — | Unset means the backend pods reject everything (fail-closed). Mismatched between gateway and a pod means that one endpoint family 401s. |
| `CLIENT_API_KEYS` | gateway | — | Unset means every client request is 401. The gateway logs this at CRITICAL on boot. |

### Measuring before you pin `RESIDENCY`

After the diffusion pod's first successful boot:

```bash
curl -s https://<diffusion-pod>-8000.proxy.runpod.net/healthz \
  -H "X-Internal-Key: $INTERNAL_API_KEY" | jq '.gpu, .pipelines'
```

`pipelines.footprint_gb` is the measured VRAM cost of each pipeline. If both fit
with headroom, pin `RESIDENCY=both`. If not, pin `swap` — task switches then cost
15–30 s, which is far better than a dead pod. Pin it either way: `auto`
re-decides on every restart, and you want restarts to be boring.

---

## Gateway

| Variable | Default | Notes |
|---|---|---|
| `CLIENT_API_KEYS` | — | Comma-separated, one per consumer. Add-then-remove gives zero-downtime rotation. |
| `INTERNAL_API_KEY` | — | Shared secret to the backend pods. `openssl rand -hex 32`. |
| `VISION_URL` `DIFFUSION_URL` `CPU_URL` | — | `https://<POD_ID>-8000.proxy.runpod.net`. Empty ⇒ those endpoints return `503 not_configured`. |
| `CHAT_URL` `TRANSLATE_URL` `CHAT_POD_KEY` | — | The pre-existing chat pod. Not managed by this repo. |
| `PUBLIC_BASE_URL` | — | Used to build the `poll` URL in a 202. Unset ⇒ clients get a relative path. |
| `DETECT_BACKEND` | `vlm` | `vlm` (Groq, no GPU) or `yolo` (vision pod). |
| `DETECT_FALLBACK_TO_YOLO` | `true` | On a VLM failure, retry against YOLO. Never applies when the caller forced a backend with `?backend=`. |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per key, token bucket. Per-process — correct for one gateway pod. |
| `RATE_LIMIT_BURST` | `30` | Instantaneous allowance. |
| `JOB_TTL_SECONDS` | `1800` | How long a finished job stays pollable. Results hold base64 images, so this is a memory dial. |
| `JOB_MAX_QUEUED` | `64` | Beyond this, submits get `503 queue_full` instead of queueing forever. |
| `SYNC_WAIT_MAX` | `60` | Ceiling on `?wait=N`. |
| `CORS_ORIGINS` | *(empty)* | Empty = CORS off = backend-to-backend only. Enabling it puts the API key in a browser. |
| `TIMEOUT_FAST` / `_VISION` / `_DIFFUSION` | `120` / `600` / `900` | Upstream read timeouts, seconds. |
| `UPSTREAM_RETRIES` | `2` | Retries on connect errors and 502/503/504 only. Never on 4xx. |

## Vision pod

| Variable | Default | Notes |
|---|---|---|
| `DATA_DIR` | `/workspace` | The network volume. Weights and HF cache live under it. |
| `ESRGAN_TILE` | `400` | See above. |
| `WHISPER_MODEL` | `medium` | `base`/`small` are much lighter if transcription accuracy allows. |
| `WHISPER_COMPUTE` | `float16` | `int8_float16` halves VRAM at a small accuracy cost. |
| `YOLO_MODEL` | `$DATA_DIR/weights/yolo_large_v3.pt` | Fallback only. Missing ⇒ that one route returns `503 model_unavailable`; nothing else is affected. |
| `WARM_UPSCALE` `WARM_WHISPER` | `true` | Preload at boot so request #1 is not the slow one. YOLO is deliberately never warmed. |

## Diffusion pod

| Variable | Default | Notes |
|---|---|---|
| `HF_HOME` | `$DATA_DIR/huggingface` | ~20 GB of weights. On the volume, or every restart re-downloads them. |
| `FLUX2_MODEL` | `black-forest-labs/FLUX.2-klein-4B` | |
| `FLUX2_STEPS` | `6` | Klein is few-step by design. Raising this costs latency for little gain. |
| `SD_INPAINT_MODEL` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | |
| `MAX_SIZE` | `1024` | Long edge. Raising it raises VRAM quadratically. |
| `RESIDENCY` | `auto` | See above. |
| `FULL_GPU_MIN_GB` | `20` | Below this, pipelines use `enable_model_cpu_offload()` — works, but slower. |

## CPU pod

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEYS` | — | Comma-separated. Rotated automatically on 429/quota. More keys = more headroom. Unset ⇒ LLM and VLM detection return `503 not_configured`. |
| `GROQ_VISION_MODEL` | `qwen/qwen3.6-27b` | Used for object detection. |
| `GROQ_USER_AGENT` | *(browser UA)* | Groq sits behind Cloudflare, which 403s default Python client UAs (error 1010). Only change this if a Cloudflare rule changes. |
| `BG_REMOVE_MODEL` | `u2net` | Any rembg model name. |
| `OCR_MODEL_DIR` | `$DATA_DIR/ocr` | Missing models ⇒ `/v1/ocr` returns `503`; everything else keeps serving. |
| `OCR_MIN_CONF` | `0.3` | Recognition confidence floor. |

## All services

| Variable | Default | Notes |
|---|---|---|
| `LOG_LEVEL` | `INFO` | JSON logs on stdout. |
| `MAX_BODY_BYTES` | `33554432` (32 MB) | Rejected at the middleware, before any decode. |
| `MAX_MEDIA_BYTES` | `26214400` (25 MB) | Per media field. |
| `ALLOW_PRIVATE_MEDIA_FETCH` | `false` | Leave it false. `true` lets a caller make the pod fetch internal addresses — an SSRF hole. |
| `ALLOW_INSECURE` | `false` | Disables internal auth. **Local development only.** |
| `CODE_REPO` / `CODE_REF` | — | Boot-time code refresh. Fail-open. |
| `BUILD_SHA` | `dev` | Set by CI; surfaced at `/healthz` so you can confirm what is actually running. |
