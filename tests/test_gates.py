"""Tests for the three verification gates (B-03) — refuse before spending."""
from __future__ import annotations

from modelrig.catalogue import DEFAULT_CATALOGUE
from modelrig.feasibility import YamlDeviceProfiler
from modelrig.gates import (
    gate1_spec_admissibility,
    gate2_plan_feasibility,
    gate3_artefact_certification,
)
from modelrig.ir import AbstentionPolicy, ArtefactIR, BuildPlanIR, DataRights, SpecIR
from modelrig.primitives import TaskPrimitive

PROFILER = YamlDeviceProfiler("configs/devices.yaml")


def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.CLASSIFY,
        device_target="android_midrange",
        seed_data_count=200,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.9,
    )
    base.update(over)
    return SpecIR(**base)


# --- Gate 1 -------------------------------------------------------------- #
def test_gate1_admits_a_sound_spec():
    assert gate1_spec_admissibility(_spec(), PROFILER).passed is True


def test_gate1_refuses_below_the_seed_floor():
    result = gate1_spec_admissibility(_spec(seed_data_count=5), PROFILER)
    assert result.passed is False
    assert any("seed data" in r for r in result.reasons)


def test_gate1_refuses_insufficient_data_rights():
    result = gate1_spec_admissibility(
        _spec(data_rights=DataRights.THIRD_PARTY_NO_TRAINING), PROFILER
    )
    assert result.passed is False
    assert any("data rights" in r for r in result.reasons)


def test_gate1_refuses_offline_plus_escalation():
    result = gate1_spec_admissibility(
        _spec(offline_required=True, abstention_policy=AbstentionPolicy.ESCALATE), PROFILER
    )
    assert result.passed is False
    assert any("offline" in r for r in result.reasons)


def test_gate1_refuses_step_depth_beyond_the_tier():
    result = gate1_spec_admissibility(_spec(max_step_depth=5), PROFILER)
    assert result.passed is False
    assert any("step depth" in r for r in result.reasons)


def test_gate1_warns_on_unmeasured_device():
    result = gate1_spec_admissibility(_spec(), PROFILER)
    assert any("GAP-10" in w for w in result.warnings)


# --- Gate 2 -------------------------------------------------------------- #
def _plan(**over) -> BuildPlanIR:
    base = dict(spec_hash="x", base_ref="Qwen/Qwen3-1.7B", bit_width="int4",
                target="gguf", budget_usd=10.0)
    base.update(over)
    return BuildPlanIR(**base)


def test_gate2_admits_a_fitting_plan():
    assert gate2_plan_feasibility(_spec(), _plan(), PROFILER).passed is True


def test_gate2_refuses_a_base_that_cannot_fit():
    spec = _spec(device_target="android_lowend")
    result = gate2_plan_feasibility(spec, _plan(base_ref="Qwen/Qwen3-8B"), PROFILER)
    assert result.passed is False
    assert any("RAM" in r or "file-size" in r for r in result.reasons)


def test_gate2_refuses_cross_tokenizer_logit_distillation():
    plan = _plan(base_ref="meta-llama/Llama-3.2-1B-Instruct",
                 teacher_ref="Qwen/Qwen3-32B", distil_mode="logit_kd")
    result = gate2_plan_feasibility(_spec(), plan, PROFILER)
    assert result.passed is False
    assert any("identical tokenizer" in r for r in result.reasons)


def test_gate2_refuses_offline_break_from_server_target():
    result = gate2_plan_feasibility(
        _spec(offline_required=True), _plan(target="vllm"), PROFILER
    )
    assert result.passed is False
    assert any("offline closure" in r for r in result.reasons)


def test_gate2_refuses_over_budget_plan():
    result = gate2_plan_feasibility(
        _spec(budget_ceiling_usd=5.0), _plan(budget_usd=200.0), PROFILER
    )
    assert result.passed is False
    assert any("ceiling" in r for r in result.reasons)


def test_gate2_reports_kv_cache_in_evidence():
    result = gate2_plan_feasibility(_spec(), _plan(), PROFILER)
    assert result.evidence["kv_cache_mb"] > 0  # the term nobody budgets for


# --- Gate 3 -------------------------------------------------------------- #
def _artefact(**over) -> ArtefactIR:
    base = dict(
        plan_hash="p", spec_hash="s", quantised_blob_hash="q",
        model_card={"task_primitive": "classify"},
        licence_chain={"permitted": True},
    )
    base.update(over)
    return ArtefactIR(**base)


def _report(**over) -> dict:
    base = dict(
        metric="accuracy", score=0.95, threshold=0.9, passed=True, n_test=50,
        held_out_is_real=True, post_quantisation=True, answer_flip_rate=0.02,
        answer_flip_bound=0.10, regression_passed=True, safety_passed=True,
        privacy_passed=True,
    )
    base.update(over)
    return base


def test_gate3_certifies_a_clean_build():
    assert gate3_artefact_certification(_artefact(), _spec(), _report()).passed is True


def test_gate3_refuses_high_answer_flip_rate():
    result = gate3_artefact_certification(_artefact(), _spec(), _report(answer_flip_rate=0.4))
    assert result.passed is False
    assert any("answer-flip" in r for r in result.reasons)


def test_gate3_refuses_when_quantised_model_was_not_re_evaluated():
    result = gate3_artefact_certification(
        _artefact(), _spec(), _report(post_quantisation=False)
    )
    assert result.passed is False
    assert any("re-evaluated" in r for r in result.reasons)


def test_gate3_refuses_failed_safety_suite():
    result = gate3_artefact_certification(_artefact(), _spec(), _report(safety_passed=False))
    assert result.passed is False
    assert any("safety" in r for r in result.reasons)


def test_gate3_refuses_uncertified_artefact():
    result = gate3_artefact_certification(_artefact(model_card={}), _spec(), _report())
    assert result.passed is False
    assert any("model card" in r for r in result.reasons)


def test_gate3_refuses_licence_that_does_not_permit_release():
    result = gate3_artefact_certification(
        _artefact(licence_chain={"permitted": False}), _spec(), _report()
    )
    assert result.passed is False


def test_catalogue_is_narrow_by_design():
    assert len(DEFAULT_CATALOGUE.bases) == 6
