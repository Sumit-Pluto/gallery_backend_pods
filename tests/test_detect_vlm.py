"""Parsing and shaping for the VLM detection path.

Worth unit-testing specifically because `json_object` mode gives no schema
guarantee: the model may wrap the list under a different key, fence it in
markdown, or hand back a bare array. Strict `json_schema` mode makes all of that
moot — but the whole point of the auto-negotiation is that we cannot rely on
having it, so the tolerant path has to be the tested one.

The cpu_tasks package is loaded under an alias because the gateway already owns
the module name `app` on the test path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_cpu_app():
    if "cpu_app" in sys.modules:
        return sys.modules["cpu_app"]
    spec = importlib.util.spec_from_file_location(
        "cpu_app",
        ROOT / "services" / "cpu_tasks" / "app" / "__init__.py",
        submodule_search_locations=[str(ROOT / "services" / "cpu_tasks" / "app")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cpu_app"] = module
    spec.loader.exec_module(module)
    return module


_load_cpu_app()
detect_vlm = importlib.import_module("cpu_app.detect_vlm")
cpu_config = importlib.import_module("cpu_app.config")


# --------------------------------------------------------------------------- #
# _parse — the shapes a model actually returns
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "content",
    [
        {"objects": [{"name": "hard hat", "confidence": 0.9}]},
        '{"objects": [{"name": "hard hat", "confidence": 0.9}]}',
        '```json\n{"objects": [{"name": "hard hat", "confidence": 0.9}]}\n```',
        [{"name": "hard hat", "confidence": 0.9}],
        {"detections": [{"name": "hard hat", "confidence": 0.9}]},
    ],
    ids=["dict", "json-string", "fenced", "bare-list", "alias-key"],
)
def test_parse_accepts_every_shape_a_model_returns(content):
    assert detect_vlm._parse(content) == [{"name": "hard hat", "confidence": 0.9}]


def test_parse_rejects_non_json_rather_than_guessing():
    from crm_common.errors import UpstreamError

    with pytest.raises(UpstreamError):
        detect_vlm._parse("I can see a hard hat and two workers.")


def test_parse_returns_empty_for_an_unrecognised_wrapper():
    assert detect_vlm._parse({"answer": "nothing here"}) == []


# --------------------------------------------------------------------------- #
# _detections — dedup, clamping, the compatibility box
# --------------------------------------------------------------------------- #


def test_detections_dedupe_by_label(monkeypatch):
    """Repeats collapse. The client stores a unique label set anyway, so paying
    to transmit "worker" five times buys nothing."""
    monkeypatch.setattr(cpu_config, "DETECT_EMIT_BOX", False)
    items = [
        {"name": "Worker", "confidence": 0.9},
        {"name": "worker", "confidence": 0.8},
        {"name": "  WORKER  ", "confidence": 0.7},
    ]
    out = detect_vlm._detections(items, 100, 50, None)
    assert out == [{"name": "worker", "confidence": 0.9}]


def test_detections_clamp_confidence_and_drop_unnamed(monkeypatch):
    monkeypatch.setattr(cpu_config, "DETECT_EMIT_BOX", False)
    items = [
        {"name": "crane", "confidence": 5.0},
        {"name": "rebar", "confidence": -2},
        {"name": "", "confidence": 0.9},
        {"name": "clock", "confidence": "not a number"},
    ]
    out = detect_vlm._detections(items, 100, 50, None)
    assert out == [
        {"name": "crane", "confidence": 1.0},
        {"name": "rebar", "confidence": 0.0},
        {"name": "clock", "confidence": 0.0},
    ]


def test_detections_apply_the_confidence_floor(monkeypatch):
    monkeypatch.setattr(cpu_config, "DETECT_EMIT_BOX", False)
    items = [{"name": "crane", "confidence": 0.9}, {"name": "clock", "confidence": 0.1}]
    out = detect_vlm._detections(items, 100, 50, 0.5)
    assert [d["name"] for d in out] == ["crane"]


def test_detections_emit_a_whole_image_box_for_legacy_clients(monkeypatch):
    """The web SDK's DetectedObject requires `box`. Nothing reads the value, so a
    whole-image box keeps existing clients validating with no change."""
    monkeypatch.setattr(cpu_config, "DETECT_EMIT_BOX", True)
    out = detect_vlm._detections([{"name": "crane", "confidence": 0.9}], 640, 480, None)
    assert out[0]["box"] == {"x1": 0.0, "y1": 0.0, "x2": 640.0, "y2": 480.0}


# --------------------------------------------------------------------------- #
# Prompt / request shaping
# --------------------------------------------------------------------------- #


def test_prompt_carries_the_canonical_vocabulary():
    """Unanchored, the same object comes back as 'hard hat', 'hardhat' and
    'safety helmet' across three photos — three albums for one thing."""
    prompt = detect_vlm._prompt()
    assert "hard hat" in prompt and "excavator" in prompt
    assert "JSON only" in prompt


def test_request_body_asks_for_no_coordinates():
    """The schema is the cost control: no box field, no spatial reasoning, and
    roughly 60% off the output tokens."""
    body = detect_vlm._body("data:image/jpeg;base64,AAAA", strict=True)
    schema = body["response_format"]["json_schema"]["schema"]
    properties = schema["properties"]["objects"]["items"]["properties"]
    assert set(properties) == {"name", "confidence"}
    assert body["temperature"] == 0
    assert body["max_tokens"] == cpu_config.DETECT_MAX_OUTPUT_TOKENS


def test_json_object_mode_drops_the_schema():
    body = detect_vlm._body("data:image/jpeg;base64,AAAA", strict=False)
    assert body["response_format"] == {"type": "json_object"}
