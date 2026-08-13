# Runbook

For whoever is on the phone when it breaks. Written to be read at 3am.

## First move, always

```bash
curl -s https://<gateway>.proxy.runpod.net/healthz | jq
curl -s -H "X-API-Key: $KEY" https://<gateway>.proxy.runpod.net/v1/status | jq
```

`/v1/status` fans out to every pod's `/readyz` and names the ones that are down.
It answers 200 when everything is healthy and 207 when something is degraded, so
one call tells you whether this is a gateway problem or a pod problem.

---

## Symptom → cause

### Everything returns 401

`CLIENT_API_KEYS` is unset or the client is sending the wrong key. Check the
gateway logs for `CLIENT_API_KEYS is not set` at CRITICAL — the gateway says this
loudly at boot precisely so this is a ten-second diagnosis.

### One endpoint family returns 503 `not_configured`

The gateway does not know where that pod is. `VISION_URL` / `DIFFUSION_URL` /
`CPU_URL` is empty or wrong. This is the usual first-deploy mistake, and it is
also what you see after recreating a pod (new id, new proxy URL).

### 503 `not_ready`, and `/v1/status` shows a pod `ready: false`

The pod is up but still loading models. Normal for 1–2 minutes after a restart;
up to ~30 minutes on the very first boot when FLUX and SDXL are downloading to
the volume. Watch that pod's log for `warmup complete`.

If it never becomes ready, the pod's log will have `warmup failed` with the
exception. Most common causes: the network volume did not mount, or HuggingFace
is rate-limiting the download.

### 502 / 504 on the slow endpoints only

Upscale or transcribe exceeded `TIMEOUT_VISION` (600 s). Either the input is
enormous, or the pod is thrashing. Check the vision pod's `/healthz` →
`queue.in_flight` and `queue.queued`. A queue depth that only grows means demand
exceeds one GPU and you need a second vision pod behind the gateway.

### CUDA OOM in a GPU pod log

Check `GPU_CONCURRENCY` is `1`. If someone raised it, that is the cause — one
card, one job. If it is already 1:

- **vision** — `ESRGAN_TILE` is probably `0`. Set it to `400`. With tiling off,
  peak VRAM scales with input resolution and a large 4× upscale will not fit.
- **diffusion** — FLUX and SDXL do not both fit on this card. Set
  `RESIDENCY=swap` and restart. Task switches then cost 15–30 s, which is much
  better than a dead pod.

### Detection suddenly returns fewer/looser boxes

Check `source` in the response. If it flipped from `vlm` to `yolo`, the Groq path
failed and the fallback took over — look for `vlm detect failed, falling back to
yolo` in the gateway log. Usually all Groq keys are rate-limited. Add another key
to `GROQ_API_KEYS` on the cpu pod.

### Jobs 404 immediately after a 202

The gateway restarted between the submit and the poll. Job state is in-process
and deliberately so — the fix is for the client to retry the whole request.
If this happens often, the gateway is crash-looping; check its logs.

### Everything is slow but nothing is failing

Look at `/v1/status` → `jobs`. A large `running` count means diffusion is the
bottleneck; jobs are queueing behind one GPU. Either accept the latency, raise
`JOB_MAX_QUEUED` so callers get 202s instead of 503s, or add a second diffusion
pod.

---

## Deploys

### Code-only change (a prompt, a threshold, a bugfix)

If the pod has `CODE_REF` set: push to that ref, restart the pod. ~15 seconds, no
build. Confirm with `/healthz` → `build` showing the new short SHA.

### Dependency change

Push. CI builds only the changed service. Update that pod's image tag to the new
SHA and restart it.

### Rollback

Set the pod's image back to the previous SHA tag and restart. This is why the
pods point at SHA tags and not `:prod` or `:latest` — a moving tag has nothing to
roll back to.

---

## Restarting a pod safely

Use **Restart**, not Stop→Resume. Resume has to re-acquire a GPU and can fail
when the region is dry. If you must stop a GPU pod, do it knowing you may not get
the same card back.

Order matters when restarting several: backends first, gateway last. The gateway
holds no state that matters, so it is the safe one to bounce.

---

## Rotating secrets

**Client key:** add the new key to `CLIENT_API_KEYS` (comma-separated), restart
the gateway, hand it over, remove the old one at the next restart. Both work in
between, so there is no cutover moment.

**Internal key:** it must match across all four pods. Set the new value on the
three backends first, then the gateway — during the overlap the backends accept
only the new key, so flip the gateway promptly.

**Groq keys:** `GROQ_API_KEYS` on the cpu pod, comma-separated. Rotation on 429
is automatic; adding a key is a restart of one cheap pod.

---

## What to check weekly

- `/v1/status` — all pods ready
- Volume usage — models plus caches should sit around 25 GB and stay flat
- Gateway logs for `rate_limited` — if the client is hitting it constantly, raise
  `RATE_LIMIT_PER_MINUTE` rather than making them retry
- RunPod billing — the pods bill 24/7 by design; a surprise means something extra
  is running

## Escalation facts worth having to hand

- Every response carries `X-Request-ID`; the same id appears in every pod's log
  for that request. One grep reconstructs the full chain.
- Logs are JSON on stdout. `... | jq 'select(.request_id=="abc123")'`.
- The old RunPod serverless endpoints are the fallback. They are near-free when
  idle. Do not delete them until this stack has a quiet week.
