"""Tests for the three-level IR stack (B-03, GAP-01)."""
from __future__ import annotations

from pathlib import Path

import pytest

from modelrig.ir import (
    AbstentionPolicy,
    ArtefactIR,
    BuildPlanIR,
    DataRights,
    SpecIR,
    content_hash,
    load_spec_ir,
    save_spec_ir,
)
from modelrig.primitives import TaskPrimitive


def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT,
        device_target="android_tablet_4gb",
        offline_required=True,
        seed_data_count=200,
        data_rights=DataRights.CUSTOMER_OWNED,
    )
    base.update(over)
    return SpecIR(**base)


def test_spec_hash_is_stable_and_content_addressed():
    a, b = _spec(), _spec()
    assert a.hash == b.hash                       # identical content -> cache hit
    assert _spec(quality_gate=0.5).hash != a.hash  # any change -> new identity
    assert len(a.hash) == 32


def test_hash_ignores_key_order():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_spec_roundtrips_through_dict():
    spec = _spec()
    restored = SpecIR.from_dict(spec.to_dict())
    assert restored.hash == spec.hash
    assert restored.task_primitive is TaskPrimitive.EXTRACT
    assert restored.data_rights is DataRights.CUSTOMER_OWNED


def test_spec_file_roundtrip(tmp_path: Path):
    spec = _spec()
    save_spec_ir(spec, tmp_path / "spec.json")
    assert load_spec_ir(tmp_path / "spec.json").hash == spec.hash


def test_load_spec_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_spec_ir(tmp_path / "nope.json")


def test_plan_and_artefact_bind_to_their_parents():
    spec = _spec()
    plan = BuildPlanIR(spec_hash=spec.hash, base_ref="Qwen/Qwen3-1.7B")
    artefact = ArtefactIR(plan_hash=plan.hash, spec_hash=spec.hash)
    assert plan.spec_hash == spec.hash
    assert artefact.plan_hash == plan.hash
    assert BuildPlanIR.from_dict(plan.to_dict()).hash == plan.hash
    assert ArtefactIR.from_dict(artefact.to_dict()).hash == artefact.hash


def test_abstention_policy_enum():
    assert AbstentionPolicy("flag") is AbstentionPolicy.FLAG
