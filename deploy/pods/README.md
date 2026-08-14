# RunPod pod configuration

Four pods. Create the **network volume first** — pods attach to it at creation and
it is region-locked, so every pod must live in the same region as the volume.

| Pod | GPU | Container disk | Image | Port |
|---|---|---|---|---|
| `crm-gateway` | CPU, 2 vCPU / 8 GB | 10 GB | `crm-gateway:<sha>` | 8000 |
| `crm-cpu` | CPU, 4 vCPU / 16 GB | 20 GB | `crm-cpu_tasks:<sha>` | 8000 |
| `crm-vision` | RTX A4000 16 GB | 30 GB | `crm-vision:<sha>` | 8000 |
| `crm-diffusion` | RTX A5000 24 GB | 40 GB | `crm-diffusion:<sha>` | 8000 |

All four mount the same network volume at `/workspace`.

---

## Three rules that keep this alive

**1. On-Demand, never Spot.** A running On-Demand pod is not reclaimed. Spot is,
and it will be reclaimed at the worst possible moment.

**2. Do not Stop/Resume in production.** Resume has to re-acquire a GPU, and if
the region is dry at that moment you do not get one back — the same scarcity you
left serverless to escape, just relocated to restart time. Run 24/7.

**3. Never Terminate.** Terminate destroys the pod id, and the proxy URL with it.
The gateway shields the client from a backend pod changing, but nothing shields
you from losing the *gateway's* URL.

---

## Order of operations

### 1. Network volume

Storage → Network Volumes → New. ~100 GB, in a region that actually has the GPUs
you want. Note the region: every pod goes there.

Contents are documented in [`../volume/README.md`](../volume/README.md). Seed it
before the pods need it (YOLO weights, OCR models).

### 2. Registry access

Images are private on GHCR. RunPod → Settings → Container Registry Auth → add:

```
Registry:  ghcr.io
Username:  <your github username>
Password:  <a GitHub PAT with read:packages>
```

Alternative worth considering: keep the YOLO weights on the volume rather than
baked into the vision image (already how this repo is configured), publish the
images publicly, and skip registry auth entirely. Nothing else in these images is
proprietary — every secret arrives as an env var.

### 3. Create the pods

Pods → Deploy → pick the GPU (or CPU) → **Edit Template**:

- **Container Image** — the full `ghcr.io/sumit-pluto/gallery_backend_pods/crm-<service>:<sha>` tag.
  Pin the SHA, never `:latest`. A moving tag cannot be rolled back.
- **Container Disk** — per the table above.
- **Network Volume** — attach the one from step 1, mount path `/workspace`.
- **Expose HTTP Ports** — `8000`.
- **Environment Variables** — from the per-pod sections below.
- Leave the Docker command empty; the image's entrypoint handles startup.

### 4. Wire the gateway

Each pod's HTTP port is now at `https://<POD_ID>-8000.proxy.runpod.net`. Collect
the three backend URLs, put them in the gateway's env, restart the gateway.

### 5. Verify

```bash
python scripts/smoke_test.py \
  --base-url https://<gateway-pod>-8000.proxy.runpod.net \
  --api-key  <client key> \
  --concurrency 10
```

`--concurrency 10` is the important one: it proves the GPU semaphore serialises
work instead of OOMing. Do not skip it.

---

## Per-pod environment

Generate the two secrets once and reuse them:

```bash
openssl rand -hex 32   # INTERNAL_API_KEY — same value on all four pods
openssl rand -hex 32   # CLIENT_API_KEYS  — the value you hand the client
```

### crm-gateway

```
CLIENT_API_KEYS=<client key>
INTERNAL_API_KEY=<shared secret>
VISION_URL=https://<vision-pod>-8000.proxy.runpod.net
DIFFUSION_URL=https://<diffusion-pod>-8000.proxy.runpod.net
CPU_URL=https://<cpu-pod>-8000.proxy.runpod.net
CHAT_URL=https://<chat-pod>-11434.proxy.runpod.net
TRANSLATE_URL=https://<chat-pod>-8000.proxy.runpod.net
CHAT_POD_KEY=<only if the chat pod sets one>
PUBLIC_BASE_URL=https://<gateway-pod>-8000.proxy.runpod.net
DETECT_BACKEND=vlm
RATE_LIMIT_PER_MINUTE=120
```

`PUBLIC_BASE_URL` is a chicken-and-egg: the pod id does not exist until the pod
does. Create it, read the id, set the variable, restart. One extra restart, once.

### crm-cpu

```
INTERNAL_API_KEY=<shared secret>
DATA_DIR=/workspace
GROQ_API_KEYS=<key1>,<key2>,<key3>
GROQ_VISION_MODEL=qwen/qwen3.6-27b
OCR_MODEL_DIR=/workspace/ocr
```

More Groq keys = more headroom. They rotate automatically on 429.

### crm-vision

```
INTERNAL_API_KEY=<shared secret>
DATA_DIR=/workspace
ESRGAN_TILE=400
WHISPER_MODEL=medium
YOLO_MODEL=/workspace/weights/yolo_large_v3.pt
GPU_CONCURRENCY=1
```

`ESRGAN_TILE=400` is what makes 16 GB enough — `tile=0` makes VRAM scale with
image size and will OOM on a large 4× upscale. Only set 0 on a 24 GB+ card.

### crm-diffusion

```
INTERNAL_API_KEY=<shared secret>
DATA_DIR=/workspace
HF_HOME=/workspace/huggingface
FLUX2_STEPS=6
MAX_SIZE=1024
RESIDENCY=auto
GPU_CONCURRENCY=1
```

After the first boot, read `GET /healthz` on this pod and look at
`readiness.footprint_gb`. That is the measured VRAM cost of FLUX and SDXL. If
they comfortably fit together, pin `RESIDENCY=both`. If not, pin `swap` — a task
switch then costs 15–30 s but the pod never OOMs. Pinning beats `auto` because
`auto` re-decides on every restart.

---

## Optional: code hot-reload

Set on any pod to pull Python from GitHub at boot instead of rebuilding the image:

```
CODE_REPO=https://x-access-token:<github-pat>@github.com/Sumit-Pluto/gallery_backend_pods.git
CODE_REF=main
```

A code-only change is then a pod restart (~15 s) instead of a build (~15 min).
Dependency changes still need a rebuild. The pull is fail-open: if GitHub is
unreachable the pod starts on its baked-in code.

Use a pinned tag (`CODE_REF=v1.4.2`) rather than `main` if you want a pod restart
to be perfectly predictable.
