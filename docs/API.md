# CRM AI Backend — API

Everything runs behind one host. You need exactly two values:

```
CRM_AI_BASE_URL=https://<gateway>.proxy.runpod.net
CRM_AI_API_KEY=<your key>
```

Send the key on every request as `X-API-Key` (or `Authorization: Bearer <key>`).

> **Call this from your backend, not the browser.**
> The key authorises GPU work. If it ships in a client bundle, anyone who opens
> devtools can spend your compute. Your server holds the key and proxies. CORS is
> disabled by default for exactly this reason.

Machine-readable spec: `GET /v1/openapi.json`.

---

## Conventions

**Media fields** (`image`, `mask`, `audio`) accept either:
- base64 — with or without a `data:image/png;base64,` prefix, or
- an `http(s)` URL we fetch server-side.

Base64 costs ~33% overhead in JSON. For anything above a few MB, send a URL.
Default limits: 25 MB per media field, 32 MB per request body.

**Errors** are always the same shape, never a 200 with an error inside:

```json
{ "error": { "code": "bad_request", "message": "...", "detail": {} } }
```

| Status | Meaning | What to do |
|---|---|---|
| 400 / 422 | Bad input | Fix the request; retrying will not help |
| 401 | Bad or missing key | Check `X-API-Key` |
| 413 | Body or media too large | Send a URL instead of base64 |
| 429 | Rate limited | Back off; `error.detail.retry_after_s` tells you how long |
| 502 / 504 | A model backend failed or timed out | Retry once with backoff |
| 503 | A pod is restarting or still loading models | Retry in ~30 s |

Every response carries `X-Request-ID`. **Quote it when reporting a problem** — it
is how we find your request across the pods.

---

## 1. Object detection

```http
POST /v1/vision/detect
{ "image": "<base64 or url>", "conf": 0.25 }
```

```json
{
  "detections": [
    { "name": "excavator", "confidence": 0.91,
      "box": { "x1": 0.0, "y1": 0.0, "x2": 4032.0, "y2": 3024.0 } }
  ],
  "source": "vlm",
  "provider": "dashscope",
  "model": "qwen-vl-plus"
}
```

**Detection is label-only.** You get *what* is in the photo, not where. Names are
short lowercase nouns steered toward a canonical construction vocabulary, so the
same object is named the same way across uploads and your albums do not fragment
into "hard hat" / "hardhat" / "safety helmet".

`confidence` is the model's self-report, not a calibrated score. Filter on it,
but do not read precision into it.

`box`, when present, is the **whole image** — a compatibility shim for clients
whose types require the field. It carries no information; do not draw it. Set
`DETECT_EMIT_BOX=false` server-side to omit it once your client allows that.
If you need real boxes, use `?backend=yolo` (below).

`provider` and `model` name whoever actually answered. With failover in play,
"why did this photo tag differently?" is otherwise unanswerable.

`source` tells you which backend answered:
- `vlm` — a Qwen vision model (default). No GPU, open vocabulary, labels only.
- `yolo` — the construction-trained model on the vision pod. Real boxes, but a
  **fixed class list**: it cannot return labels outside what it was trained on.

Add `?backend=yolo` to force YOLO. It is no longer an automatic fallback —
falling back silently changed which labels you got, and it competes for the same
single GPU slot as upscale and transcribe. Redundancy now happens one layer down,
across model providers, which fails over without changing the answer's shape.

### Batch

```http
POST /v1/vision/detect-batch
{ "images": ["<base64 or url>", "..."], "conf": 0.25 }
```

```json
{ "ok": 1, "failed": 1,
  "results": [
    { "index": 0, "detections": [ ... ], "source": "vlm", "provider": "dashscope" },
    { "index": 1, "error": { "code": "bad_request", "message": "not a decodable image" } }
  ] }
```

Use this for gallery uploads. The server fans the images out concurrently under
its own semaphore, so it is faster than N sequential calls **and** gentler on the
provider rate limit than N parallel ones from your side. Each image reports its
own result or its own error at its own `index` — one bad image never fails the
batch, and the response always has one entry per input, in order.

Default cap is 32 images per call; over that you get a 400 telling you the limit.

## 2. Image editing

```http
POST /v1/image/edit
{ "image": "<base64 or url>",
  "op":    { "type": "colorize" },
  "mask":  "<base64 or url>",          // required for some ops, see table
  "params": { "strength": 0.8, "seed": 42 } }
```

**You send an op name, not a prompt.** The server owns the wording — that keeps
results consistent and stops prompt text reaching the GPU unchecked.

| `op.type` | Needs mask | Extra | Returns |
|---|---|---|---|
| `upscale` | — | `op.factor`: 2 or 4 (default 2) | 200 inline |
| `restore` | — | Real-ESRGAN + face restore at 4× | 200 inline |
| `remove-background` | — | — | 200 inline |
| `colorize` | — | — | **202 job** |
| `prompt` | — | `op.prompt` **required** | **202 job** |
| `replace-sky` | optional | `op.prompt` describes the sky | **202 job** |
| `magic-eraser` | **yes** | — | **202 job** |
| `generative-fill` | **yes** | `op.prompt` **required** | **202 job** |

Masks are PNG, same aspect as the image. **White = the region to regenerate.**

`replace-sky` without a mask degrades to a gentle whole-image pass so the
foreground mostly survives. Send a mask for a real sky replacement.

`params` (all optional, all clamped server-side):

| Field | Range | Applies to |
|---|---|---|
| `strength` | 0.0–1.0 | all diffusion ops |
| `seed` | 0–2147483647 | all diffusion ops |
| `negative_prompt` | ≤300 chars | mask ops only |
| `steps` | 1–60 | mask ops only |
| `guidance_scale` | 1.0–20.0 | mask ops only |

### The 202 / poll contract

Diffusion takes 30–90 s, so it does not block your HTTP connection:

```json
// 202 Accepted
{ "job_id": "9f2c…", "status": "queued", "poll": "https://…/v1/jobs/9f2c…" }
```

```http
GET /v1/jobs/9f2c…
```

```json
{ "job_id": "9f2c…", "status": "done",
  "result": { "image": "<base64 png>", "width": 1024, "height": 768 } }
```

`status` is `queued` → `running` → `done` | `error`. Poll every 2–3 s. Jobs are
kept 30 minutes after finishing, then the id 404s. A job is visible only to the
key that created it.

If you would rather wait inline for a fast render, add `?wait=30` — you get a
200 with the result if it finishes in time, or the usual 202 if it does not.

Shortcuts that skip the op table when you know what you want:

```http
POST /v1/image/upscale     { "image": "...", "scale": 4, "face_enhance": false }
POST /v1/image/remove-bg   { "image": "..." }
```

## 3. Audio

```http
POST /v1/audio/transcribe
{ "audio": "<base64 or url>", "language": "en", "timestamps": true }
```
```json
{ "transcript": "...", "language": "en",
  "segments": [{ "text": "...", "start_sec": 0.0, "end_sec": 3.2 }] }
```
Omit `language` to auto-detect. Any format ffmpeg reads works.

```http
POST /v1/audio/denoise
{ "audio": "<base64 or url>" }
```
Returns `{ "audio": "<base64 wav>" }` — 48 kHz mono.

## 4. OCR

```http
POST /v1/ocr
{ "image": "<base64 or url>" }
```
```json
{ "text": "INVOICE 2024\nTotal: 1,240.00",
  "lines": [{ "text": "INVOICE 2024", "score": 0.98,
              "box": [[30,150],[226,150],[226,200],[30,200]] }] }
```
`box` is a quad (4 corner points), not a rectangle — text is not always axis-aligned.
Lines come back in reading order.

## 5. LLM and vision (Groq)

```http
POST /v1/llm/chat
{ "model": "qwen/qwen3.6-27b",
  "messages": [ ... ],
  "response_format": { "type": "json_schema", "json_schema": { ... } } }
```

A **passthrough**. The body is a standard OpenAI `chat/completions` payload and it
reaches the provider untouched — you own the prompt, the model, and the
structured-output schema. We hold only the API keys and rotate them on rate
limits.

```json
{ "response": { ...full chat completion... }, "key_index": 0 }
```
Read the answer at `response.choices[0].message.content`.

Vision works the same way — put an `image_url` content part in your message.
Streaming is not supported through the gateway; one JSON response per call.

## 6. Self-hosted chat and translation

```http
POST /v1/chat/completions        // Qwen2.5-14B, OpenAI-shaped
POST /v1/translate               // NLLB-200
{ "text": "When can you deliver the cement?",
  "source": "eng_Latn", "target": "hin_Deva" }
-> { "translation": "..." }
```

Language codes are FLORES-200: `eng_Latn` `hin_Deva` `spa_Latn` `fra_Latn`
`deu_Latn` `arb_Arab` `zho_Hans` `rus_Cyrl` `por_Latn` `ben_Beng` `tam_Taml`
`tel_Telu` `mar_Deva` `guj_Gujr` `pan_Guru` `jpn_Jpan` `kor_Hang` `vie_Latn`
`ind_Latn`.

## 7. Health

```http
GET /healthz     // public, no key. Is the gateway alive?
GET /v1/status   // authenticated. Per-pod readiness — check this first when something is slow.
```

---

## Worked example

```python
import base64, time, requests

BASE = "https://<gateway>.proxy.runpod.net"
HEAD = {"X-API-Key": "<key>"}

image = base64.b64encode(open("site.jpg", "rb").read()).decode()

# Fast op — answers inline.
r = requests.post(f"{BASE}/v1/image/upscale",
                  json={"image": image, "scale": 2}, headers=HEAD, timeout=300)
r.raise_for_status()
open("upscaled.png", "wb").write(base64.b64decode(r.json()["image"]))

# Slow op — 202 then poll.
r = requests.post(f"{BASE}/v1/image/edit",
                  json={"image": image, "op": {"type": "colorize"}},
                  headers=HEAD, timeout=60)
job = r.json()["job_id"]

while True:
    s = requests.get(f"{BASE}/v1/jobs/{job}", headers=HEAD, timeout=30).json()
    if s["status"] == "done":
        open("colorized.png", "wb").write(base64.b64decode(s["result"]["image"]))
        break
    if s["status"] == "error":
        raise RuntimeError(s["error"]["message"])
    time.sleep(2)
```

## Integration notes

- **Retry 502/503/504 once or twice with backoff.** A pod restart is a normal
  event, not an outage — the gateway returns a clean 503 rather than hanging.
- **Do not retry 4xx.** The request is wrong; the second attempt will be too.
- **Set generous client timeouts.** 300 s for upscale and transcribe. Diffusion
  never needs a long timeout because it is a job.
- **Log `X-Request-ID` alongside your own request ids.** It turns a support
  question into a two-minute lookup.
- Rate limit is 120 requests/minute per key with a burst of 30. Ask if you need
  more; it is a config change, not a redeploy.
