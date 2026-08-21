# Deployment runbook

The exact sequence, in order, from an empty RunPod account to a URL you hand the
client. Every step has a verification — **do not move on until it passes.** Most
production incidents in a stack like this trace back to a step someone assumed
had worked.

Budget **3–4 hours**, of which about an hour is unattended downloading.

---

## Before you start

Have these open: the RunPod console, GitHub → your repo → Actions, and a
terminal in this repo.

### Generate the two secrets now

```bash
openssl rand -hex 32   # INTERNAL_API_KEY — the SAME value on all four pods
openssl rand -hex 32   # CLIENT_API_KEYS  — the value you give the CRM team
```

Paste them somewhere you will still have in an hour. `INTERNAL_API_KEY` must be
byte-identical across all four pods; a typo there produces a 401 that looks like
a code bug and costs you twenty minutes.

### Pick your region first, not last

Network volumes are **region-locked**, and pods must live in the volume's region.
Choose a region that actually has RTX A4000 and A5000 capacity *before* creating
the volume. Getting this wrong means deleting the volume and starting over.

### The image tags

Tags are the full 40-char commit SHA. Because CI only rebuilds what changed,
**the four services are not all on the same SHA:**

| Service | Image |
|---|---|
| gateway | `ghcr.io/sumit-pluto/gallery_backend_pods/crm-gateway:dcb76c174aff13f6d38c4e37fee9771d5f3f81c9` |
| vision | `ghcr.io/sumit-pluto/gallery_backend_pods/crm-vision:dcb76c174aff13f6d38c4e37fee9771d5f3f81c9` |
| cpu_tasks | `ghcr.io/sumit-pluto/gallery_backend_pods/crm-cpu_tasks:dcb76c174aff13f6d38c4e37fee9771d5f3f81c9` |
| diffusion | `ghcr.io/sumit-pluto/gallery_backend_pods/crm-diffusion:f04d6bea76fb18aeba82dffd8072ece67b008b13` |

Every service also carries `:prod`, which always points at its most recent build.
`:prod` is more convenient; the SHA is what lets you roll back. **Use the SHA.**
Each Actions job's Summary tab prints the exact tag it pushed.

---

## Step 1 — Registry access (10 min)

The packages are private. Pick one:

**Option A — make them public (simpler, recommended).** Nothing proprietary is
baked into these images: the YOLO weights live on the volume and every secret
arrives as a pod env var. GitHub → your profile → Packages → each `crm-*`
package → Package settings → Change visibility → Public.

**Option B — keep them private.** Create a GitHub PAT with `read:packages`, then
RunPod → Settings → Container Registry Auth → Add:

```
Registry:  ghcr.io
Username:  Sumit-Pluto
Password:  <the PAT>
```

**Verify:** with Option A, `docker pull <gateway image>` from any machine works
with no login. With Option B, the pod will fail with `manifest unknown` or
`unauthorized` if the credential is wrong — and RunPod surfaces that only in the
pod's log, so check the log rather than assuming.

---

## Step 2 — Network volume (20 min + upload time)

RunPod → Storage → Network Volumes → **New**.

| Setting | Value |
|---|---|
| Size | **100 GB** |
| Region | the one you chose above |

100 GB is deliberate headroom; steady state is ~25 GB. Volume storage bills
whether pods run or not — it is the one cost that does not stop when you stop.

### Seed it

Two model sets need a human; everything else downloads itself on first boot.

Attach the volume to any cheap pod temporarily (a CPU pod is fine), then:

```bash
# from your machine, using the pod's SSH details from the RunPod console
scp -P <port> yolo_large_v3.pt              root@<host>:/workspace/weights/
scp -P <port> ppocrv6_det.onnx              root@<host>:/workspace/ocr/
scp -P <port> ppocrv6_rec.onnx              root@<host>:/workspace/ocr/
scp -P <port> ppocrv6_rec.config.yml        root@<host>:/workspace/ocr/
```

Create the directories first if they do not exist (`mkdir -p /workspace/weights
/workspace/ocr`).

Sources in the old repo:

| File | Location |
|---|---|
| `yolo_large_v3.pt` | `advance-photo-gallery-web-sdk/workers/gpu-vision/` |
| `ppocrv6_*.onnx`, `ppocrv6_rec.config.yml` | `OCR/handoff/models/` |

**Verify:** `shasum -a 256 -c CHECKSUMS.txt` in the OCR handoff directory. A
truncated ONNX file fails at inference with a confusing error, not at load — so
check now rather than debugging it later.

If a file is missing, nothing crashes: the affected route returns a clean
`503 model_unavailable` naming the exact path it wanted, and every other endpoint
keeps serving.

---

## Step 3 — The three backend pods (45 min)

Deploy in this order: **cpu → vision → diffusion → gateway last.** The gateway
needs the other three URLs, and the cpu pod is the cheapest way to discover that
your registry auth or volume mount is wrong.

For every pod: Pods → Deploy → pick the hardware → **Edit Template** →

- **Container Image** — the full tag from the table above
- **Container Disk** — per the table below
- **Network Volume** — the one from Step 2, mount path **`/workspace`**
- **Expose HTTP Ports** — **`8000`**
- **Environment Variables** — the block for that pod, below
- **Leave the Docker command empty** — the image's entrypoint handles startup

Use **On-Demand / Secure Cloud**, never Spot.

| Pod | Hardware | Container disk |
|---|---|---|
| `crm-cpu` | CPU, 4 vCPU / 16 GB | 20 GB |
| `crm-vision` | **RTX A4000 16 GB** | 30 GB |
| `crm-diffusion` | **RTX A5000 24 GB** | 40 GB |
| `crm-gateway` | CPU, 2 vCPU / 8 GB | 10 GB |

### 3a. crm-cpu

```
INTERNAL_API_KEY=<shared secret>
DATA_DIR=/workspace
# Providers are tried in order; the chain advances when one is down or every
# key of it is throttled. You need at least one. OpenRouter is the quickest to
# get billing for; Model Studio is cheapest once its account is activated.
LLM_PROVIDERS=openrouter,dashscope,groq

OPENROUTER_API_KEYS=<key>
OPENROUTER_VISION_MODEL=qwen/qwen3.7-flash

DASHSCOPE_API_KEYS=<key>
DASHSCOPE_VISION_MODEL=qwen3-vl-flash

GROQ_API_KEYS=<key1>,<key2>
GROQ_VISION_MODEL=qwen/qwen3.6-27b
OCR_MODEL_DIR=/workspace/ocr
```

Each provider's keys are comma-separated and rotate automatically on 429. Note
that Groq's limits are **per organisation**, so extra Groq keys on one account
buy no headroom — only a second *provider* does. More keys
means more headroom. This pod serves object detection (via the Qwen vision
model), so an empty value here disables detection *and* the LLM endpoint.

**Verify:**
```bash
curl https://<cpu-pod-id>-8000.proxy.runpod.net/healthz
curl https://<cpu-pod-id>-8000.proxy.runpod.net/readyz
```
`/healthz` answers within seconds. `/readyz` returns 503 until warmup finishes.

**Expect the first `/readyz` to take several minutes.** `import rembg` pulls in
numba, which JIT-compiles at import and pegs every core. It is cached to
`/workspace/numba` afterwards, so subsequent restarts are fast. If it is still
503 after ~10 minutes, read the pod log — you will see either `warmup still
running` (be patient) or `warmup failed` with the actual exception.

### 3b. crm-vision

```
INTERNAL_API_KEY=<shared secret>
DATA_DIR=/workspace
ESRGAN_TILE=400
WHISPER_MODEL=medium
WHISPER_COMPUTE=float16
YOLO_MODEL=/workspace/weights/yolo_large_v3.pt
GPU_CONCURRENCY=1
```

**`ESRGAN_TILE=400` is what makes a 16 GB card enough.** With tiling off, peak
VRAM scales with input resolution and a large 4× upscale will OOM. Do not set it
to 0 unless you move to a 24 GB+ card.

**`GPU_CONCURRENCY=1` is not a performance setting to tune.** One card, one job.
Serverless gave you this for free by handing each worker its own GPU; a pod will
happily accept 50 concurrent requests and OOM. Raising this is the fastest way to
break production.

First boot downloads Whisper medium (~1.5 GB) and Real-ESRGAN weights to the
volume — a few minutes.

**Verify:** `/readyz` returns 200 with `loaded` listing `real-esrgan` and
`whisper:medium`, and `yolo: "lazy (fallback only)"`.

### 3c. crm-diffusion

```
INTERNAL_API_KEY=<shared secret>
DATA_DIR=/workspace
HF_HOME=/workspace/huggingface
FLUX2_MODEL=black-forest-labs/FLUX.2-klein-4B
FLUX2_STEPS=6
SD_INPAINT_MODEL=diffusers/stable-diffusion-xl-1.0-inpainting-0.1
MAX_SIZE=1024
RESIDENCY=auto
GPU_CONCURRENCY=1
```

**This pod's first boot takes ~30 minutes** — it is pulling roughly 20 GB of FLUX
and SDXL weights onto an empty volume. Every restart after that is seconds,
because they live on the volume. Start this one and go do Step 4 while it runs.

If warmup genuinely hangs rather than being slow, it now fails loudly after 45
minutes with `warmup exceeded ...` instead of sitting silently at
`ready: false`.

**Verify, then do the one tuning step that matters:**

```bash
curl https://<diffusion-pod-id>-8000.proxy.runpod.net/healthz | jq '.gpu, .pipelines'
```

Read `pipelines.footprint_gb` — the measured VRAM cost of each pipeline. Then:

- both fit with headroom on the card → set **`RESIDENCY=both`**
- they do not → set **`RESIDENCY=swap`** (a task switch then costs 15–30 s, which
  is far better than an OOM)

**Pin it either way and restart.** `auto` re-decides on every boot, and you want
restarts to be boring.

---

## Step 4 — The gateway (15 min)

Collect the three proxy URLs first. Each is
`https://<POD_ID>-8000.proxy.runpod.net`.

```
CLIENT_API_KEYS=<client key>
INTERNAL_API_KEY=<shared secret>
VISION_URL=https://<vision-pod-id>-8000.proxy.runpod.net
DIFFUSION_URL=https://<diffusion-pod-id>-8000.proxy.runpod.net
CPU_URL=https://<cpu-pod-id>-8000.proxy.runpod.net
CHAT_URL=https://<chat-pod-id>-11434.proxy.runpod.net
TRANSLATE_URL=https://<chat-pod-id>-8000.proxy.runpod.net
CHAT_POD_KEY=<only if your chat pod sets one>
PUBLIC_BASE_URL=https://<gateway-pod-id>-8000.proxy.runpod.net
DETECT_BACKEND=vlm
# Off by default: YOLO cannot produce the open-vocabulary labels the gallery
# indexes, and it shares the vision pod's single GPU slot with upscale.
DETECT_FALLBACK_TO_YOLO=false

# Must match the GPU's real capacity: one diffusion pod at GPU_CONCURRENCY=1
# means 1. This is what keeps the queue in the gateway (where a caller can be
# told their position) instead of at the pod's semaphore (where they wait blind).
JOB_DISPATCH_CONCURRENCY=1
JOB_MAX_QUEUED=12
RATE_LIMIT_PER_MINUTE=120
```

`CHAT_URL` and `TRANSLATE_URL` point at your **existing** Qwen/NLLB pod. This
repo does not manage it; the gateway only fronts it so the client sees one host.

`PUBLIC_BASE_URL` is a chicken-and-egg — the pod id does not exist until the pod
does. Create the pod, read its id, set the variable, restart. One extra restart,
once. Without it, the `poll` URL in a 202 comes back relative.

**Verify:**
```bash
curl https://<gateway>.proxy.runpod.net/healthz
curl -H "X-API-Key: <client key>" https://<gateway>.proxy.runpod.net/v1/status | jq
```

`/v1/status` fans out to every pod. It returns **200** when all are healthy and
**207** when something is degraded, and names what. This is the single call to
make whenever anything looks wrong.

---

## Step 5 — Prove it works (30 min)

```bash
pip install httpx Pillow
python scripts/smoke_test.py \
  --base-url https://<gateway>.proxy.runpod.net \
  --api-key  <client key> \
  --concurrency 10
```

It generates its own image and audio, so there is nothing to prepare. It exercises
every endpoint, follows a diffusion job through the 202/poll cycle, and confirms
a bad key is rejected.

**Do not skip `--concurrency 10`.** It fires ten simultaneous upscales. All ten
must return 200. That is the test that proves the GPU semaphore is serialising
work rather than letting requests pile onto one card — the single most likely way
this stack fails under real traffic, and the one thing you cannot discover from a
single-request test.

Expect the diffusion checks to take a minute or two each. Optional endpoints
(OCR, LLM, chat, translate) report `SKIP` rather than failing if unconfigured.

---

## Step 6 — Hand over (10 min)

The client gets two values and one document:

```
CRM_AI_BASE_URL=https://<gateway>.proxy.runpod.net
CRM_AI_API_KEY=<their key>
```

Send them [`docs/API.md`](API.md). It has the full contract, a worked Python
example, and the integration notes that matter (retry 502/503/504, never retry
4xx, log `X-Request-ID`).

Tell them explicitly: **call it from their backend, not the browser.** The key
authorises GPU work — in a client bundle, anyone can drain your pods. CORS is off
by default for exactly this reason.

They never learn pod ids, RunPod keys, or Groq keys. That is what lets you move,
resize, or rebuild any pod without telling them.

---

## Step 7 — Production hygiene

### Three rules that keep this alive

**1. On-Demand, never Spot.** A running On-Demand pod is not reclaimed.

**2. Do not Stop/Resume in production.** Resume must re-acquire a GPU, and if the
region is dry at that moment you do not get one back — the same scarcity you left
serverless to escape, relocated to restart time. Run 24/7 and budget for it.

**3. Never Terminate.** That destroys the pod id and its proxy URL. The gateway
shields the client from a backend pod changing; nothing shields you from losing
the gateway's own URL.

### Monitoring

Point an uptime monitor at `https://<gateway>.proxy.runpod.net/healthz` (public,
no key). Check `/v1/status` weekly, or whenever anything feels slow.

Logs are JSON on stdout. Every response carries `X-Request-ID`, and the same id
appears in every pod's log for that request:

```bash
... | jq 'select(.request_id=="abc123")'
```

### Deploying a change

| Changed | What to do | Time |
|---|---|---|
| Python code, a prompt, a threshold | push, then **restart that pod** (needs `CODE_REF` set) | ~15 s |
| `requirements.txt` or a Dockerfile | push → CI builds one image → repoint that pod's tag → restart | ~15 min |

To enable the fast path, set on each pod:

```
CODE_REPO=https://x-access-token:<github-pat>@github.com/Sumit-Pluto/gallery_backend_pods.git
CODE_REF=main
```

The pull is fail-open — if GitHub is unreachable the pod starts on its baked-in
code. Use a pinned tag (`CODE_REF=v1.0.0`) instead of `main` if you want restarts
to be perfectly predictable.

### Rollback

Set the pod's image back to the previous SHA tag and restart. This is the entire
reason the pods point at SHA tags and not `:latest` — a moving tag has nothing to
roll back to.

### Keep the old serverless endpoints for a week

They cost almost nothing idle, and they are your only fallback if something here
misbehaves under real traffic. Delete them after this stack has had a quiet week.

---

## Things that will bite you

Ordered by how likely they are.

| Symptom | Cause | Fix |
|---|---|---|
| Everything 401s | `CLIENT_API_KEYS` unset | Gateway logs this at CRITICAL on boot. Set it, restart. |
| One endpoint family 401s | `INTERNAL_API_KEY` differs between gateway and that pod | Make them byte-identical. |
| `503 not_configured` | Gateway does not know that pod's URL | Set `VISION_URL` / `DIFFUSION_URL` / `CPU_URL`. Also what you see after recreating a pod — new id, new URL. |
| `503 not_ready` for ages | Still loading | Normal for 1–2 min; ~30 min on diffusion's first boot. Check the pod log for `warmup complete`. |
| CUDA OOM | `GPU_CONCURRENCY > 1`, or `ESRGAN_TILE=0` on 16 GB, or FLUX+SDXL both resident on too small a card | Set concurrency to 1, tile to 400, `RESIDENCY=swap`. |
| Detection labels get narrower | `source` flipped to `yolo` (only if you enabled the fallback) | Check `provider` in the response and `/v1/status`. Add a provider to `LLM_PROVIDERS` rather than more keys to one. |
| `queue_full` under load | More renders submitted than one GPU can drain | Expected: it now rejects fast with `retry_after_s` instead of timing out at 5 min. Add a diffusion pod and raise `JOB_DISPATCH_CONCURRENCY`. |
| Job ids 404 right after a 202 | Gateway restarted between submit and poll | Job state is in-process by design. Client retries. If frequent, the gateway is crash-looping — read its log. |
| Pod will not start, `unauthorized` | GHCR credentials | Fix registry auth, or make the packages public. |

Deeper diagnostics: [`RUNBOOK.md`](RUNBOOK.md). Every environment variable and
what it costs you to get wrong: [`ENV.md`](ENV.md).

---

## What this costs

Roughly, at 24/7. Confirm current rates in the console.

| Pod | ~$/mo |
|---|---|
| crm-gateway (CPU) | ~$30 |
| crm-cpu (CPU) | ~$45 |
| crm-vision (A4000 16 GB) | ~$130 |
| crm-diffusion (A5000 24 GB) | ~$220 |
| 100 GB network volume | ~$7 |
| **Total** | **~$430/mo** |

Your existing chat pod is on top of this and unchanged.

If cost matters more than restart isolation, vision and diffusion can share one
24 GB card (~$295/mo) — vision needs only ~3 GB now that detection runs on the
VLM. You lose the ability to redeploy one without bouncing the other.
