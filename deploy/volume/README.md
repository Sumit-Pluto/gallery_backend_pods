# The network volume

One ~100 GB RunPod network volume, mounted at `/workspace` on all four pods.

Everything large lives here rather than in an image. Two reasons, and both are
operational rather than aesthetic:

* **Restarts are seconds, not minutes.** FLUX + SDXL + Whisper is roughly 20 GB.
  Baked into images, every restart re-pulls it. On the volume, it is already there.
* **Swapping a model does not need a rebuild.** Drop a new `yolo_large_v3.pt` on
  the volume, restart the vision pod, done.

## Layout

```
/workspace/
├── huggingface/          HF_HOME — FLUX, SDXL, Whisper. Populated automatically
│                         on first boot (~20 GB, ~30 min). Never delete casually.
├── weights/
│   ├── RealESRGAN_x4plus.pth     auto-downloaded on first vision boot
│   ├── GFPGANv1.4.pth            auto-downloaded on first face-restore call
│   └── yolo_large_v3.pt          ← YOU upload this. Not auto-downloadable.
├── ocr/
│   ├── ppocrv6_det.onnx          ← YOU upload these three
│   ├── ppocrv6_rec.onnx
│   └── ppocrv6_rec.config.yml
├── rembg/                U2NET_HOME — u2net.onnx, auto-downloaded
└── torch/                torch hub cache
```

Only the two `← YOU upload` sets need a human. Everything else populates itself.

## Seeding it

Easiest path: create the volume, attach it to any cheap pod, and copy the files
in over the pod's terminal or SSH.

```bash
# From your machine, with the pod's SSH details from the RunPod console:
scp -P <port> yolo_large_v3.pt root@<host>:/workspace/weights/
scp -P <port> ppocrv6_*.onnx ppocrv6_rec.config.yml root@<host>:/workspace/ocr/
```

Sources in the old repo:

| File | Where it is now |
|---|---|
| `yolo_large_v3.pt` | `advance-photo-gallery-web-sdk/workers/gpu-vision/yolo_large_v3.pt` |
| `ppocrv6_det.onnx` | `OCR/handoff/models/` |
| `ppocrv6_rec.onnx` | `OCR/handoff/models/` |
| `ppocrv6_rec.config.yml` | `OCR/handoff/models/` |

`OCR/handoff/CHECKSUMS.txt` has sha256 sums — verify after transfer with
`shasum -a 256 -c CHECKSUMS.txt`. A truncated ONNX file fails at inference time
with a confusing error, not at load.

## If a file is missing

Nothing crashes. The affected route returns a clean `503 model_unavailable`
naming the exact path it wanted, and everything else keeps serving. That is
deliberate — a missing OCR model should not take background removal down.

## Capacity

~25 GB in use once everything is warm. 100 GB leaves room for a second diffusion
model or a larger Whisper. Volume storage is billed per GB-month whether the pods
are running or not, so it is the one line item that does not stop when you stop.
