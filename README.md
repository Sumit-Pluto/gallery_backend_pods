# crm-ai-backend

Backend-only AI services for the CRM, running on **RunPod Pods** (not serverless).
Python and Docker throughout — no Node, no Next.js, no Vercel. The client team
has their own frontend; this repo hands them **one URL and one API key**.

```
Client backend ──HTTPS + X-API-Key──▶  ┌──────────────────────────────┐
                                       │  GATEWAY   (CPU pod)         │
                                       │  auth · validation · op→     │
                                       │  prompt · routing · jobs     │
                                       └──┬────────┬────────┬─────────┘
                                          │  internal key, never public
                            ┌─────────────▼┐  ┌────▼─────┐  ┌▼─────────┐
                            │ vision  GPU  │  │diffusion │  │ cpu      │
                            │ upscale      │  │  GPU     │  │ rembg    │
                            │ transcribe   │  │ flux     │  │ denoise  │
                            │ yolo(fallbk) │  │ sdxl     │  │ ocr, llm │
                            └──────────────┘  └──────────┘  └──────────┘
                                                   └── chat pod (pre-existing,
                                                       fronted not replaced)
```

## Why pods instead of serverless

RunPod serverless kept failing to acquire a GPU when the region was contended,
which meant paying for active workers anyway — the cost of a pod without the
control. Pods trade that for a different failure mode: scarcity moves from *every
request* to *pod start*. Three rules follow, and they matter more than anything
else in this repo:

1. **On-Demand, never Spot.** A running On-Demand pod is not reclaimed.
2. **Do not Stop/Resume in production.** Resume must re-acquire a GPU and can
   fail. Run 24/7.
3. **Never Terminate.** That destroys the pod id and its proxy URL.

## Layout

```
services/
  gateway/      the only public surface — auth, op→prompt, routing, job queue
  vision/       Real-ESRGAN, GFPGAN, Whisper, YOLO (fallback only)
  diffusion/    FLUX.2 [klein] img2img, SDXL inpaint
  cpu_tasks/    rembg, ffmpeg denoise, PP-OCRv6, Groq proxy, VLM detection
libs/common/    schemas, media IO, logging, health, GPU semaphore — shared by all
deploy/         per-pod RunPod config, volume layout, entrypoint, redeploy script
docs/           API.md (give this to the client), ENV.md, RUNBOOK.md
tests/          gateway behaviour: auth, routing, jobs, the op table
scripts/        smoke_test.py — run after every deploy
```

## Two things make this modular

**One image per service, path-filtered CI.** Touch `services/diffusion/**` and
only `crm-diffusion` builds. Images are tagged by git SHA, never `latest`, so
rollback is redeploying yesterday's tag.

**Code separated from dependencies.** The slow part of a rebuild is torch and
diffusers, never your handler code. So deps are baked into the image and Python
is optionally refreshed from GitHub at boot (`CODE_REF`):

| Changed | Path | Time |
|---|---|---|
| `requirements.txt`, Dockerfile | CI builds one image, repoint the pod | ~15 min |
| Python code, a prompt, a threshold | restart the pod | **~15 s, no build** |

The pull is fail-open — if GitHub is unreachable, the baked-in code runs.

## Quick start

```bash
cp .env.example .env
make dev                 # gateway + cpu on localhost:8000
make test                # gateway test suite
curl localhost:8000/healthz
```

The GPU services need CUDA and are absent locally; the gateway returns a clean
`503 not_configured` for anything routed to them, which is exactly what it does
in production when a pod is down. Every code path except literal model inference
is exercisable on a laptop.

## Deploying

1. Create the network volume — [`deploy/volume/README.md`](deploy/volume/README.md)
2. Create the four pods — [`deploy/pods/README.md`](deploy/pods/README.md)
3. Verify:
   ```bash
   python scripts/smoke_test.py --base-url https://<gateway>.proxy.runpod.net \
                                --api-key <key> --concurrency 10
   ```
   `--concurrency 10` proves the GPU semaphore serialises work instead of OOMing.
   Do not skip it.
4. Hand over `CRM_AI_BASE_URL`, `CRM_AI_API_KEY`, and
   [`docs/API.md`](docs/API.md).

When it breaks: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Design decisions worth knowing before you change something

**The gateway owns the prompts.** Clients send an op name (`colorize`,
`magic-eraser`), never prompt text. The wording lives in
[`services/gateway/app/ops.py`](services/gateway/app/ops.py), ported verbatim from
the Vercel route it replaced. Editing those strings changes what the client's
images look like — do it deliberately. `tests/test_ops.py` pins every branch.

**One GPU job at a time.** `crm_common.gpu.run_exclusive` wraps every inference
in a semaphore of 1. Serverless gave us this for free by handing each worker its
own card; a pod will happily accept 50 concurrent requests and OOM. Raising
`GPU_CONCURRENCY` is the fastest way to break production.

**Diffusion is asynchronous.** FLUX and SDXL take 30–90 s, so `/v1/image/edit`
returns `202 {job_id}` and the client polls. Job state is in-process, which is
correct for one gateway pod and would need Redis for two.

**Models live on the network volume, not in images.** Restarts are seconds
instead of minutes, and swapping the YOLO weights needs no rebuild.

**Detection defaults to the VLM.** Qwen vision via Groq costs no GPU. YOLO stays
wired as a fallback and is loaded lazily, so it holds zero VRAM until something
asks for it.

## Not in this repo

The chat pod (Qwen + NLLB) predates it and still runs from
`chat-pod/start.sh` in the old repo. The gateway fronts it at
`/v1/chat/completions` and `/v1/translate` so the client sees one host — but it
is not built, deployed, or managed here.
