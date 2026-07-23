"""Tests for the reasoning core (mock + HF planner parsing + fallback)."""
from __future__ import annotations

import pytest

from majestic.config import Settings
from majestic.core.hf_core import _parse_steps
from majestic.core.mock import MockReasoningCore
from majestic.factory import build_core
from majestic.types import Request


# --- mock core ---------------------------------------------------------- #
def test_mock_core_plan_and_synthesize():
    core = MockReasoningCore(default_target="echo")
    plan = core.plan(Request(content="hi"))
    assert len(plan.steps) == 1
    assert plan.steps[0].target == "echo"
    answer = core.synthesize(Request(content="hi"), ["world"], grounding=["ctx"])
    assert "world" in answer
    assert "ctx" in answer


# --- HF planner JSON parsing (pure, offline) ---------------------------- #
def test_parse_steps_valid_json():
    raw = '[{"description": "search", "target": "web"}, {"description": "sum", "target": "echo"}]'
    steps = _parse_steps(raw, ("web", "echo"), "content")
    assert [s.target for s in steps] == ["web", "echo"]
    assert all(s.args["text"] == "content" for s in steps)


def test_parse_steps_unknown_target_becomes_echo():
    steps = _parse_steps('[{"description": "x", "target": "nope"}]', ("web", "echo"), "c")
    assert steps[0].target == "echo"


def test_parse_steps_with_surrounding_prose():
    raw = 'Sure! Here is the plan:\n[{"description":"d","target":"web"}]\nHope it helps.'
    steps = _parse_steps(raw, ("web",), "c")
    assert steps[0].target == "web"


def test_parse_steps_garbage_falls_back_to_echo():
    steps = _parse_steps("no json here", ("web", "echo"), "content")
    assert len(steps) == 1
    assert steps[0].target == "echo"


# --- factory fallback --------------------------------------------------- #
def test_build_core_falls_back_when_deps_missing():
    """A non-mock model id with torch/transformers absent must not crash."""
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present; fallback path not exercised")
    s = Settings()
    s.core.model = "some/tiny-instruct-model"
    core = build_core(s)
    # Fallback yields a working mock core.
    assert core.plan(Request(content="x")).steps[0].target == "echo"


# --- gated integration: real tiny model -------------------------------- #
@pytest.mark.integration
def test_hf_core_real_tiny_model():
    """Loads a very small instruct model; skipped unless deps + env allow."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from majestic.core.hf_core import HFReasoningCore

    core = HFReasoningCore(model_id="sshleifer/tiny-gpt2", max_new_tokens=8)
    plan = core.plan(Request(content="say hi"))
    assert len(plan.steps) >= 1
