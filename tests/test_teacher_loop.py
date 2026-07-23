"""Tests for the teacher -> factory distillation loop."""
from __future__ import annotations

from pathlib import Path

from modelrig.factory import Factory
from modelrig.teacher_loop import DistillationLoop


def _loop(tmp_path: Path) -> DistillationLoop:
    return DistillationLoop(
        factory=Factory(base_path=tmp_path), data_dir=tmp_path / "datasets"
    )


def test_distill_creates_dataset_and_feasible_spec(tmp_path: Path):
    loop = _loop(tmp_path)
    spec = loop.distill("sentiment", "android_midrange")
    assert spec.task == "sentiment"
    assert Path(spec.dataset).exists()
    assert spec.extras["feasibility"]["feasible"] is True
    assert spec.device is not None and spec.device.name == "android_midrange"


def test_distill_and_build_end_to_end(tmp_path: Path):
    loop = _loop(tmp_path)
    result = loop.distill_and_build("sentiment", "laptop_gpu")
    assert result.success is True
    assert result.eval_report["passed"] is True
    assert result.build_id in loop.factory.registry.list()


def test_ingest_telemetry_and_routing_updates(tmp_path: Path):
    loop = _loop(tmp_path)
    loop.ingest_telemetry([
        {"capability": "translate", "query": "..."},
        {"capability": "translate", "query": "..."},
        {"capability": "translate", "query": "..."},
        {"capability": "math", "query": "..."},
    ])
    assert loop.escalation_counts["translate"] == 3
    assert loop.routing_updates(threshold=3) == ["translate"]
    assert loop.routing_updates(threshold=5) == []
