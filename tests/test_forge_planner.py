"""Tests for FORGE (B-04) and the PLANNER (B-05)."""
from __future__ import annotations

import pytest

from modelrig.feasibility import YamlDeviceProfiler
from modelrig.forge import Forge
from modelrig.ir import DataRights, SpecIR
from modelrig.planner import Planner
from modelrig.primitives import TaskPrimitive

PROFILER = YamlDeviceProfiler("configs/devices.yaml")

LAB_REQUEST = (
    "Read lab requisition forms into our system on an Android tablet. "
    "Must work when the internet dies. We have 200 real forms. "
    "Flag anything uncertain, do not guess. Hindi and English. 95 percent."
)


# --- FORGE --------------------------------------------------------------- #
def test_forge_extracts_slots_from_plain_language():
    state = Forge().parse(LAB_REQUEST)
    assert state.slots["task_primitive"].value is TaskPrimitive.EXTRACT
    assert state.slots["device_target"].value == "android_tablet_4gb"
    assert state.slots["seed_data_count"].value == 200
    assert state.slots["quality_gate"].value == pytest.approx(0.95)
    assert set(state.slots["languages"].value) == {"hi-IN", "en-IN"}


def test_forge_flags_the_offline_ambiguity():
    """'when the internet dies' is ambiguous — always offline, or just resilient?"""
    state = Forge().parse(LAB_REQUEST)
    slot = state.slots["offline_required"]
    assert slot.ambiguous is True
    assert slot.filled is False          # never silently defaulted
    assert "offline_required" in [s.name for s in state.unfilled()]


def test_forge_asks_only_unfilled_slots_ranked_by_gain():
    state = Forge().parse(LAB_REQUEST)
    questions = state.questions(limit=4)
    assert 0 < len(questions) <= 4       # four questions, not forty
    gains = [s.information_gain for s in state.unfilled()]
    assert gains == sorted(gains, reverse=True)


def test_forge_emits_a_hash_addressed_spec():
    forge = Forge()
    state = forge.answer(
        forge.parse(LAB_REQUEST),
        offline_required=True,
        data_rights=DataRights.CUSTOMER_OWNED,
    )
    spec = forge.to_spec(state)
    assert spec.task_primitive is TaskPrimitive.EXTRACT
    assert spec.offline_required is True
    assert len(spec.hash) == 32


def test_forge_refuses_to_guess_the_primitive():
    forge = Forge()
    with pytest.raises(ValueError, match="task_primitive"):
        forge.to_spec(forge.parse("please do the needful with our stuff"))


def test_forge_identical_descriptions_hash_identically():
    forge = Forge()
    a, _ = forge.interview(LAB_REQUEST, offline_required=True)
    b, _ = forge.interview(LAB_REQUEST, offline_required=True)
    assert a.hash == b.hash          # identical hash -> served from cache


# --- PLANNER ------------------------------------------------------------- #
def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT,
        device_target="android_tablet_4gb",
        seed_data_count=200,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.9,
    )
    base.update(over)
    return SpecIR(**base)


def test_planner_admits_a_plan_deterministically():
    result = Planner(profiler=PROFILER).plan(_spec())
    assert result.admitted is True
    assert result.path == "deterministic"     # no LLM involved on the common path
    assert result.plan.bit_width == "int4"    # optimal accuracy-per-bit (A-01)


def test_planner_caps_the_base_at_the_device_tier():
    """A 4 GB tablet must not be handed an 8B base."""
    result = Planner(profiler=PROFILER).plan(_spec(device_target="android_tablet_4gb"))
    assert result.admitted
    from modelrig.catalogue import DEFAULT_CATALOGUE

    base = DEFAULT_CATALOGUE.base(result.plan.base_ref)
    assert base.params_b <= 2.0


def test_planner_reuses_precedent_after_a_pass():
    planner = Planner(profiler=PROFILER)
    spec = _spec()
    first = planner.plan(spec)
    planner.record_outcome(spec, first.plan, passed=True)
    second = planner.plan(spec)
    assert second.path == "precedent"
    assert second.plan.base_ref == first.plan.base_ref


def test_planner_mutates_an_infeasible_llm_proposal():
    """The LLM proposes; the validator disposes."""
    def bad_proposer(spec, catalogue):
        from modelrig.ir import BuildPlanIR

        return BuildPlanIR(spec_hash=spec.hash, base_ref="Qwen/Qwen3-8B",
                           bit_width="fp16", target="gguf", budget_usd=10.0)

    planner = Planner(profiler=PROFILER, proposer=bad_proposer)
    result = planner.plan(_spec(device_target="android_lowend"))
    assert result.path == "proposed"
    assert result.mutations, "an infeasible proposal must be mutated, not accepted"


def test_planner_defaults_to_sequence_kd_which_crosses_tokenizers():
    result = Planner(profiler=PROFILER).plan(_spec(seed_data_count=200))
    assert result.plan.distil_mode in ("sequence_kd", "none")
    assert result.plan.distil_mode != "logit_kd"


def test_planner_records_outcomes_for_meta_learning():
    planner = Planner(profiler=PROFILER)
    spec = _spec()
    result = planner.plan(spec)
    planner.record_outcome(spec, result.plan, passed=True)
    assert len(planner.outcomes.history) == 1
    # Cold start: no prediction until enough history exists (GAP-02).
    assert planner.outcomes.predict_pass_probability("extract") is None
