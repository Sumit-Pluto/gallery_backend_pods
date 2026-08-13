"""FLUX.2 [klein] + SDXL-inpaint loading, placement and residency management.

The old serverless handler loaded both lazily and kept both forever, which is
fine when the platform hands each worker its own card and kills it after the
job. A long-lived pod is different: if the two pipelines together exceed VRAM,
the second load OOMs and the pod is dead until someone restarts it.

So this module measures what each pipeline actually costs at load time, logs it,
and either keeps both resident or swaps them depending on the card. The measured
numbers show up in /healthz — that is how you find out whether this pod can drop
to a cheaper GPU.
"""

from __future__ import annotations

import gc
import logging
import threading

import torch

from . import config

log = logging.getLogger(__name__)

_lock = threading.RLock()
_flux = None
_inpaint = None
_mode: str | None = None
_footprint: dict[str, float] = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def total_vram_gb() -> float:
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def residency_mode() -> str:
    global _mode
    if _mode is None:
        if config.RESIDENCY in ("both", "swap"):
            _mode = config.RESIDENCY
        else:
            _mode = "both" if total_vram_gb() >= config.BOTH_RESIDENT_MIN_GB else "swap"
        log.info(
            "residency resolved",
            extra={"mode": _mode, "total_vram_gb": round(total_vram_gb(), 1), "requested": config.RESIDENCY},
        )
    return _mode


def _place(pipe):
    """Full GPU residency when there is headroom, CPU offload when there isn't.

    `enable_model_cpu_offload` streams module weights from RAM per forward pass:
    it makes a 16 GB card work, at a real latency cost. Keeping the original
    threshold behaviour so output is unchanged from the serverless build.
    """
    if DEVICE == "cuda" and total_vram_gb() >= config.FULL_GPU_MIN_GB:
        return pipe.to("cuda")
    log.warning("placing pipeline with cpu_offload (low VRAM)", extra={"total_vram_gb": total_vram_gb()})
    pipe.enable_model_cpu_offload()
    return pipe


def _measure(name: str, before: int) -> None:
    if DEVICE != "cuda":
        return
    used = (torch.cuda.memory_allocated() - before) / 1e9
    _footprint[name] = round(used, 2)
    free, total = torch.cuda.mem_get_info()
    log.info(
        "pipeline loaded",
        extra={"pipeline": name, "vram_gb": _footprint[name], "free_gb": round(free / 1e9, 2)},
    )


def _evict(name: str) -> None:
    global _flux, _inpaint
    target = _flux if name == "flux" else _inpaint
    if target is None:
        return
    log.info("evicting pipeline to free VRAM", extra={"pipeline": name})
    if name == "flux":
        _flux = None
    else:
        _inpaint = None
    del target
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def flux():
    global _flux
    with _lock:
        if residency_mode() == "swap":
            _evict("inpaint")
        if _flux is None:
            from diffusers import Flux2KleinPipeline

            before = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0
            pipe = Flux2KleinPipeline.from_pretrained(config.FLUX_MODEL, torch_dtype=torch.bfloat16)
            pipe.set_progress_bar_config(disable=True)
            _flux = _place(pipe)
            _measure("flux", before)
        return _flux


def inpaint():
    global _inpaint
    with _lock:
        if residency_mode() == "swap":
            _evict("flux")
        if _inpaint is None:
            from diffusers import AutoPipelineForInpainting

            before = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0
            try:
                pipe = AutoPipelineForInpainting.from_pretrained(
                    config.INPAINT_MODEL, torch_dtype=torch.float16, variant="fp16"
                )
            except Exception:
                # Not every mirror publishes the fp16 variant; fall back to the
                # default weights rather than failing the whole pod.
                pipe = AutoPipelineForInpainting.from_pretrained(
                    config.INPAINT_MODEL, torch_dtype=torch.float16
                )
            pipe.set_progress_bar_config(disable=True)
            _inpaint = _place(pipe)
            _measure("inpaint", before)
        return _inpaint


def warm() -> dict:
    mode = residency_mode()
    loaded = []
    if mode == "swap":
        # Warming both would immediately evict one. Warm the common path only.
        if config.WARM_FLUX:
            flux()
            loaded.append("flux")
    else:
        if config.WARM_FLUX:
            flux()
            loaded.append("flux")
        if config.WARM_INPAINT:
            inpaint()
            loaded.append("inpaint")
    return {
        "residency": mode,
        "loaded": loaded,
        "footprint_gb": dict(_footprint),
        "total_vram_gb": round(total_vram_gb(), 1),
    }


def report() -> dict:
    return {
        "residency": _mode,
        "resident": [n for n, p in (("flux", _flux), ("inpaint", _inpaint)) if p is not None],
        "footprint_gb": dict(_footprint),
    }
