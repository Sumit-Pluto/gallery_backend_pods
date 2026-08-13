"""Test bootstrap: put the gateway package and crm_common on the path.

Tests run against the gateway because that is where the logic that moved off
Vercel lives. The vision/diffusion/cpu services are thin wrappers around model
calls and are verified on the pod (scripts/smoke_test.py), not here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs" / "common"))
sys.path.insert(0, str(ROOT / "services" / "gateway"))

os.environ.setdefault("ALLOW_INSECURE", "true")
os.environ.setdefault("CLIENT_API_KEYS", "test-client-key,other-client-key")
os.environ.setdefault("VISION_URL", "http://vision.test")
os.environ.setdefault("DIFFUSION_URL", "http://diffusion.test")
os.environ.setdefault("CPU_URL", "http://cpu.test")
os.environ.setdefault("CHAT_URL", "http://chat.test")
os.environ.setdefault("TRANSLATE_URL", "http://translate.test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
