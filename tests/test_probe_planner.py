"""Probe-before-plan: the reordering that turns a guess into a constraint (§9-§10)."""
from __future__ import annotations

import pytest

from modelrig.ir import DataRights, ProfileSource, SpecIR
from modelrig.planner import default_catalog, plan
from modelrig.planner.predicates import P_lat, P_ram, context_budget
from modelrig.primitives import TaskPrimitive
from modelrig.probe import DeviceProfile
from modelrig.probe import ProfileSource as ProbeSource

CAT = default_catalog()


def _profile(**over) -> dict:
    base = dict(
        device_id="sm-a536b-probe-8f3a", ram_total_mb=4000, ram_free_mb=2600,
        bw_eff_gbps=9.24, overhead_ms_per_token=2.16, thermal_derate_180s=0.6,
        simd=("neon", "dotprod"), probe_lo_mb=200, probe_hi_mb=400,
        source=ProbeSource.PROBE,
    )
    base.update(over)
    return DeviceProfile(**base).to_dict()


def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT,
        device_target="android_tablet_4gb",
        seed_data_count=200,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.9,
        latency_budget_ms=60_000,
        io_schema={"tokens_in": 500, "tokens_out": 60},
    )
    base.update(over)
    return SpecIR(**base)


# =========================================================================== #
# §14 — the schema changes
# =========================================================================== #
def test_spec_carries_a_device_profile_and_its_source():
    spec = _spec(device_profile=_profile(), profile_source=ProfileSource.PROBE)
    assert spec.profile_source.measured is True
    assert spec.device_profile["bw_eff_gbps"] == pytest.approx(9.24)


def test_profile_source_defaults_to_assumed():
    """No probe means the device facts are a guess, and the field says so."""
    assert _spec().profile_source is ProfileSource.ASSUMED
    assert _spec().profile_source.measured is False


def test_schema_round_trips_with_the_new_fields():
    spec = _spec(device_profile=_profile(), profile_source=ProfileSource.PROBE,
                 context_budget=4096)
    restored = SpecIR.from_dict(spec.to_dict())
    assert restored.hash == spec.hash
    assert restored.profile_source is ProfileSource.PROBE
    assert restored.context_budget == 4096


def test_a_probe_changes_the_spec_hash():
    """Different device facts are a different build, and must not share a cache key."""
    assert _spec().hash != _spec(device_profile=_profile()).hash


# =========================================================================== #
# §9 — P_lat moves from Tier 1 to Tier 3
# =========================================================================== #
def test_without_a_probe_the_latency_predicate_refuses():
    result = P_lat(_spec(), _cand(), CAT)
    assert result.ok is False
    assert "unmeasured" in result.reason


def test_with_a_probe_the_latency_predicate_may_promise():
    """The deadlock is resolved by measurement, not by buying phones."""
    spec = _spec(device_profile=_profile(), profile_source=ProfileSource.PROBE)
    result = P_lat(spec, _cand(), CAT)
    assert result.ok is True
    assert result.detail["measured"] is True
    assert result.detail["may_promise"] is True


def test_a_measured_refusal_reports_the_thermal_derate():
    spec = _spec(latency_budget_ms=2000, device_profile=_profile(),
                 profile_source=ProfileSource.PROBE)
    result = P_lat(spec, _cand(), CAT)
    assert result.ok is False
    assert "thermal derate" in result.reason
    assert "measured, so it will not improve" in result.remedy


def test_an_interpolated_profile_cannot_promise():
    """However many observations back it, it is not a measurement of THIS unit."""
    spec = _spec(device_profile=_profile(source=ProbeSource.INTERPOLATED))
    result = P_lat(spec, _cand(), CAT)
    assert result.ok is False
    assert "unmeasured" in result.reason


# =========================================================================== #
# §13 — the probe/production mismatch reaches the memory predicate
# =========================================================================== #
def test_measured_free_ram_supersedes_the_prior():
    spec = _spec(device_profile=_profile(), profile_source=ProfileSource.PROBE)
    result = P_ram(spec, _cand(), CAT)
    assert result.ok is True
    assert "probe" in result.detail["ram_source"]
    # 2600 MB measured, discounted by the 0.85 production headroom factor.
    assert result.detail["free_ram_mb"] == pytest.approx(2210, abs=1)


def test_a_tighter_measured_reading_can_flip_a_passing_plan():
    """The prior said 4000 MB free; the device actually has far less."""
    generous = _spec()
    measured = _spec(device_profile=_profile(ram_free_mb=1400),
                     profile_source=ProfileSource.PROBE)
    assert P_ram(generous, _cand(), CAT).ok is True
    assert P_ram(measured, _cand(), CAT).ok is False


# =========================================================================== #
# §10 — the context budget is derived, not elicited
# =========================================================================== #
def test_context_budget_is_derived_from_the_memory_equation():
    device = CAT.device("android_tablet_4gb")
    budget = context_budget(CAT.model("Qwen/Qwen3-1.7B"), CAT,
                            quantiser="q4_k_m", device=device)
    assert budget > 0
    assert budget <= CAT.model("Qwen/Qwen3-1.7B").max_context


def test_a_bigger_base_leaves_less_room_for_context():
    """Device constraints propagate upward into task design, not only downward."""
    device = CAT.device("android_tablet_4gb")
    small = context_budget(CAT.model("Qwen/Qwen3-0.6B"), CAT,
                           quantiser="q4_k_m", device=device)
    large = context_budget(CAT.model("Qwen/Qwen3-1.7B"), CAT,
                           quantiser="q4_k_m", device=device)
    assert small > large


def test_no_room_for_weights_means_no_context_at_all():
    device = CAT.device("android_lowend")
    assert context_budget(CAT.model("Qwen/Qwen3-8B"), CAT,
                          quantiser="q4_k_m", device=device) == 0


def test_the_passing_plan_reports_its_context_budget():
    spec = _spec(device_profile=_profile(), profile_source=ProfileSource.PROBE)
    result = P_ram(spec, _cand(), CAT)
    assert result.ok is True
    assert result.detail["context_budget"] > 0


# =========================================================================== #
# End to end
# =========================================================================== #
def test_a_probe_can_turn_a_refusal_into_an_admitted_plan():
    """Measured decode is ~12 s plus ~10 s prefill, so 30 s is the honest budget."""
    unmeasured = _spec(latency_budget_ms=30_000)
    assert plan(unmeasured, CAT).admitted is False      # cannot promise at all

    probed = _spec(latency_budget_ms=30_000, device_profile=_profile(),
                   profile_source=ProfileSource.PROBE)
    assert plan(probed, CAT).admitted is True           # measured: may promise


def test_a_probe_can_also_turn_an_estimate_into_an_honest_refusal():
    """Measurement cuts both ways, which is the point of measuring."""
    optimistic = _spec(latency_budget_ms=2000,
                       io_schema={"tokens_in": 500, "tokens_out": 60,
                                  "accept_unmeasured_latency": True})
    slow_device = _spec(latency_budget_ms=2000,
                        device_profile=_profile(bw_eff_gbps=2.0),
                        profile_source=ProfileSource.PROBE)
    assert plan(optimistic, CAT).admitted is False
    outcome = plan(slow_device, CAT)
    assert outcome.admitted is False
    assert "P_lat" in outcome.refusal.witness


def _cand():
    from modelrig.planner.core import PlanCandidate

    return PlanCandidate(
        base=CAT.model("Qwen/Qwen3-1.7B"), teacher=None, distil_mode="none",
        peft_method="lora", rank=16, quantiser="q4_k_m", bit_width="int4",
        target="gguf", data_recipe={},
    )
