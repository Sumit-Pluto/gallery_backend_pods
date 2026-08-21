"""Diffusion service configuration.

Imported before torch/diffusers so HF_HOME points at the network volume — FLUX
and SDXL together are ~20 GB of weights and you do not want to re-download them
on every pod restart.

RESIDENCY is the setting that decides which GPU you need:

  "both"  keep FLUX and SDXL on the card at once. Fastest. Needs the headroom.
  "swap"  keep one pipeline resident, evict it when the other task arrives.
          MEASURED on an RTX 4090 (24 GB) with FLUX.2-klein + SDXL-inpaint:
          ~45 s per switch, not the 15-30 s this used to claim. For scale, a
          FLUX render with its pipeline already resident is 5 s — so an
          alternating workload runs about 9x slower than a batched one.
          Group work by op type where you can. FLUX ops are colorize / prompt /
          replace-sky-without-mask; SDXL ops are magic-eraser / generative-fill /
          masked replace-sky.
  "auto"  measure free VRAM at boot and pick. This is the default.
"""

from __future__ import annotations

import os

DATA_DIR = os.environ.get("DATA_DIR") or ("/workspace" if os.path.isdir("/workspace") else "/tmp/crm-data")
HF_HOME = os.environ.get("HF_HOME") or os.path.join(DATA_DIR, "huggingface")

os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

FLUX_MODEL = os.environ.get("FLUX2_MODEL", "black-forest-labs/FLUX.2-klein-4B")
FLUX_STEPS = int(os.environ.get("FLUX2_STEPS", "6"))
INPAINT_MODEL = os.environ.get("SD_INPAINT_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")

MAX_SIZE = int(os.environ.get("MAX_SIZE", "1024"))

# both | swap | auto
RESIDENCY = os.environ.get("RESIDENCY", "auto").lower()
# Below this much total VRAM, "auto" chooses swap instead of keeping both loaded.
#
# 23 is too low, and measurement says so: FLUX.2-klein alone is 16.03 GB (not the
# ~9 GB the sizing notes assumed), and SDXL-inpaint adds ~7 GB. Both resident is
# therefore ~23 GB before a single activation, so a 24 GB card that passes this
# threshold would OOM on the first render. Wanting "both" means a 32 GB card at
# minimum, realistically 48 GB.
BOTH_RESIDENT_MIN_GB = float(os.environ.get("BOTH_RESIDENT_MIN_GB", "30"))
# Below this, pipelines are placed with enable_model_cpu_offload() instead of .to(cuda).
FULL_GPU_MIN_GB = float(os.environ.get("FULL_GPU_MIN_GB", "20"))

WARM_FLUX = os.environ.get("WARM_FLUX", "true").lower() == "true"
WARM_INPAINT = os.environ.get("WARM_INPAINT", "true").lower() == "true"
