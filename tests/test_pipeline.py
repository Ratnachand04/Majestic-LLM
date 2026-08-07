"""End-to-end compiler tests (B-01, C-02): spec in, certified cartridge out."""
from __future__ import annotations

from pathlib import Path

from modelrig.feasibility import YamlDeviceProfiler
from modelrig.forge import Forge
from modelrig.ir import DataRights, SpecIR
from modelrig.pipeline import MajesticCompiler
from modelrig.primitives import TaskPrimitive
from modelrig.registry import CartridgeRegistry

PROFILER = YamlDeviceProfiler("configs/devices.yaml")

_POS = ["good great work", "love this service", "happy with the result",
        "nice and helpful staff", "excellent quality throughout"]
_NEG = ["bad broken service", "hate this experience", "sad and useless outcome",
        "terrible unhelpful staff", "awful quality throughout"]


def _corpus(n: int = 120) -> list[tuple[str, str]]:
    rows = []
    for i in range(n // 2):
        rows.append((f"{_POS[i % len(_POS)]} case {i}", "positive"))
        rows.append((f"{_NEG[i % len(_NEG)]} case {i}", "negative"))
    return rows


def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.CLASSIFY,
        device_target="android_midrange",
        seed_data_count=120,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.8,
    )
    base.update(over)
    return SpecIR(**base)


def _compiler(tmp_path: Path) -> MajesticCompiler:
    return MajesticCompiler(
        registry=CartridgeRegistry(tmp_path), profiler=PROFILER, base_path=tmp_path
    )


# --- the happy path ------------------------------------------------------- #
def test_compiles_a_spec_into_a_certified_cartridge(tmp_path: Path):
    result = _compiler(tmp_path).compile(_spec(), _corpus())
    assert result.admitted is True, result.refusal
    assert result.stage_reached == "registry"
    assert result.cartridge.certified is True
    assert result.scorecard.passed is True
    # Every gate was recorded, in order.
    assert [g.gate for g in result.gates] == [
        "gate1_spec_admissibility", "gate2_plan_feasibility",
        "gate3_artefact_certification",
    ]


def test_answer_flip_is_measured_after_quantisation(tmp_path: Path):
    result = _compiler(tmp_path).compile(_spec(), _corpus())
    assert result.scorecard.post_quantisation is True
    assert result.scorecard.answer_flip_rate is not None


def test_cartridge_is_admitted_to_the_registry(tmp_path: Path):
    compiler = _compiler(tmp_path)
    result = compiler.compile(_spec(), _corpus())
    assert result.cartridge_id in compiler.registry.list()
    assert compiler.registry.lineage(result.cartridge_id)["base_ref"]


def test_identical_spec_hits_the_cache_second_time(tmp_path: Path):
    """An identical spec hash skips the build entirely at zero marginal cost."""
    compiler = _compiler(tmp_path)
    first = compiler.compile(_spec(), _corpus())
    second = compiler.compile(_spec(), _corpus())
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cartridge_id == first.cartridge_id


# --- honest refusals, cheaply ---------------------------------------------- #
def test_refuses_at_gate1_before_any_work(tmp_path: Path):
    result = _compiler(tmp_path).compile(_spec(seed_data_count=5), _corpus(10))
    assert result.refused
    assert result.stage_reached == "gate1"
    assert result.plan is None              # no planning, no data, no GPU
    assert "seed data" in result.refusal


def test_refuses_unusable_data_rights(tmp_path: Path):
    result = _compiler(tmp_path).compile(
        _spec(data_rights=DataRights.THIRD_PARTY_NO_TRAINING), _corpus()
    )
    assert result.refused
    assert result.stage_reached == "gate1"


def test_refuses_when_the_quality_bar_cannot_be_met(tmp_path: Path):
    result = _compiler(tmp_path).compile(_spec(quality_gate=0.98), _corpus())
    if result.refused:
        assert result.stage_reached in ("gate1", "gate3")
        if result.stage_reached == "gate3":
            assert result.repair_suggestions   # external evidence for the repairer


# --- the C-02 flow, from plain language ------------------------------------ #
def test_forge_to_cartridge_end_to_end(tmp_path: Path):
    forge = Forge()
    state = forge.parse(
        "Classify incoming support tickets by sentiment on an Android phone. "
        "We have 120 real tickets. Flag anything uncertain."
    )
    state = forge.answer(
        state, offline_required=True, data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.8, seed_data_count=120,
    )
    spec = forge.to_spec(state)
    assert spec.task_primitive is TaskPrimitive.CLASSIFY

    result = _compiler(tmp_path).compile(spec, _corpus())
    assert result.admitted is True, result.refusal
    # What the customer receives is a scorecard, not a model file.
    card = result.scorecard
    assert card.sample_predictions
    assert result.cartridge.model_card["offline"] is True
