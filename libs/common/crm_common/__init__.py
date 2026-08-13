"""Shared primitives for every CRM AI backend service.

Nothing in here imports torch, diffusers, or any model runtime — it is safe to
install into the CPU-only images as well as the GPU ones.
"""

__version__ = "0.1.0"
