"""Tests for the device probe and the verification ladder (Part 3 §6-§9, §12-§13)."""
from __future__ import annotations

import pytest

from modelrig.devicedb import DeviceDB, ProbeObservation, resolve_profile
from modelrig.probe import (
    MB,
    DeviceProfile,
    ProbeBundle,
    ProbeError,
    ProbePoint,
    ProfileSource,
    Tier,
    assess_latency,
    build_profile,
    calibrate,
    emulation_for,
    roofline_bound,
    thermal_derate,
)


def _points(small_toks=42.0, large_toks=22.0):
    return [ProbePoint(200 * MB, small_toks), ProbePoint(400 * MB, large_toks)]


def _profile(**over) -> DeviceProfile:
    base = dict(
        device_id="probe-1", ram_total_mb=4000, ram_free_mb=2600,
        bw_eff_gbps=9.24, overhead_ms_per_token=2.16, thermal_derate_180s=0.6,
        simd=("neon", "dotprod"), probe_lo_mb=200, probe_hi_mb=400,
        source=ProfileSource.PROBE,
    )
    base.update(over)
    return DeviceProfile(**base)


# =========================================================================== #
# §8 — two-point calibration


# =========================================================================== #


def test_two_point_calibration_reproduces_the_worked_example():
    """200 MB at 42 tok/s and 400 MB at 22 tok/s give ~9.2 GB/s and ~2.1 ms."""
    cal = calibrate(_points())
    assert cal.bw_eff_gbps == pytest.approx(9.24, abs=0.05)
    assert cal.overhead_s_per_token * 1000 == pytest.approx(2.16, abs=0.05)


def test_calibration_predicts_the_real_cartridge():
    """A 1.12 GB cartridge lands at ~8 tok/s — without ever shipping it."""
    cal = calibrate(_points())
    assert cal.tokens_per_s(1120 * MB) == pytest.approx(8.1, abs=0.3)


def test_the_affine_model_is_solved_exactly_not_fitted():
    """Two unknowns, two equations. The probe points must lie on the line."""
    cal = calibrate(_points())
    for p in _points():
        assert cal.seconds_per_token(p.size_bytes) == pytest.approx(
            p.seconds_per_token, rel=1e-9
        )


def test_overhead_is_not_assumed_zero():
    """Per-token framework overhead is real; a pure S/BW model would misfit."""
    cal = calibrate(_points())
    assert cal.overhead_s_per_token > 0


def test_a_faster_larger_model_is_rejected_as_a_broken_benchmark():
    """Decode is bandwidth-bound. This is thermal state or caching, not physics."""
    with pytest.raises(ProbeError, match="broken benchmark"):
        calibrate([ProbePoint(200 * MB, 20.0), ProbePoint(400 * MB, 30.0)])


def test_one_point_cannot_calibrate_two_unknowns():
    with pytest.raises(ProbeError, match="at least two"):
        calibrate([ProbePoint(200 * MB, 42.0)])


def test_identical_sizes_cannot_calibrate():
    with pytest.raises(ProbeError, match="distinct"):
        calibrate([ProbePoint(200 * MB, 42.0), ProbePoint(200 * MB, 41.0)])


def test_three_points_use_least_squares():
    cal = calibrate([
        ProbePoint(200 * MB, 42.0), ProbePoint(300 * MB, 30.0), ProbePoint(400 * MB, 22.0),
    ])
    assert cal.bw_eff_gbps > 0
    assert cal.lo_bytes == 200 * MB and cal.hi_bytes == 400 * MB


def test_probe_rejects_nonsense_measurements():
    with pytest.raises(ProbeError):
        ProbePoint(0, 42.0)
    with pytest.raises(ProbeError):
        ProbePoint(200 * MB, 0.0)


# =========================================================================== #
# §13 — extrapolation limits


# =========================================================================== #


def test_predictions_inside_the_bracket_are_trusted():
    cal = calibrate(_points())
    assert cal.extrapolation_factor(300 * MB) == 1.0
    assert cal.trustworthy_for(300 * MB)


def test_extrapolation_reach_is_reported():
    """Probing at 200/400 MB and predicting a 4 GB model is a 10x reach."""
    cal = calibrate(_points())
    assert cal.extrapolation_factor(4000 * MB) == pytest.approx(10.0)
    assert cal.trustworthy_for(4000 * MB, max_reach=10.0) is True
    assert cal.trustworthy_for(8000 * MB, max_reach=10.0) is False


def test_far_extrapolation_reverts_a_measured_profile_to_a_bound():
    """Beyond a decade, even a probe stops licensing a promise."""
    near = assess_latency(_profile(), 1120 * MB, tokens_out=60)
    far = assess_latency(_profile(), 9000 * MB, tokens_out=60)
    assert near.tier is Tier.MEASURED and near.may_promise is True
    assert far.tier is Tier.ANALYTICAL and far.may_promise is False
    assert "one decade" in far.reason


# =========================================================================== #
# §8 — thermal derate


# =========================================================================== #


def test_thermal_derate_is_the_sustained_ratio():
    assert thermal_derate(22.0, 13.2) == pytest.approx(0.6)


def test_derate_cannot_exceed_one():
    """A device cannot sustainably beat its own burst rate."""
    assert thermal_derate(20.0, 25.0) == 1.0


def test_all_promises_use_the_sustained_figure():
    profile = _profile(thermal_derate_180s=0.6)
    burst = profile.tokens_per_s(1120 * MB, sustained=False)
    sustained = profile.tokens_per_s(1120 * MB, sustained=True)
    assert sustained == pytest.approx(burst * 0.6)
    assert profile.tokens_per_s(1120 * MB) == sustained     # sustained is the default


def test_build_profile_wires_the_derate_in():
    profile = build_profile(
        "p", _points(), ram_total_mb=4000, ram_free_mb=2600,
        sustained_tokens_per_s=13.2, burst_reference_tokens_per_s=22.0,
    )
    assert profile.thermal_derate_180s == pytest.approx(0.6)
    assert profile.measured is True


def test_missing_sustained_phase_means_no_derate_and_says_so():
    profile = build_profile("p", _points(), ram_total_mb=4000, ram_free_mb=2600)
    assert profile.thermal_derate_180s == 1.0


# =========================================================================== #
# §7 — the ladder


# =========================================================================== #


def test_only_measured_may_promise():
    assert Tier.MEASURED.may_promise is True
    assert Tier.EMULATED.may_promise is False
    assert Tier.ANALYTICAL.may_promise is False


def test_every_tier_may_refuse():
    """All three only ever subtract from performance, so refusal is always sound."""
    assert all(t.may_refuse for t in Tier)


def test_profile_source_maps_to_a_tier():
    assert ProfileSource.PROBE.tier is Tier.MEASURED
    assert ProfileSource.DEVICE_LAB.tier is Tier.MEASURED
    assert ProfileSource.INTERPOLATED.tier is Tier.ANALYTICAL
    assert ProfileSource.ASSUMED.tier is Tier.ANALYTICAL


def test_roofline_is_an_upper_bound():
    """Real throughput is always at or below it: throttling only subtracts."""
    bound = roofline_bound(1120 * MB, 9.24e9)
    actual = _profile().tokens_per_s(1120 * MB)
    assert actual < bound


def test_emulation_says_what_it_cannot_verify():
    plan = emulation_for(_profile(), "cartridge.gguf")
    assert plan.memory_max_mb == _profile().usable_ram_mb
    assert "MemoryMax" in plan.command()
    assert "real tokens per second" in plan.CANNOT_VERIFY
    assert "does it OOM at the target RAM ceiling" in plan.VERIFIES


# =========================================================================== #
# §13 — probe/production mismatch


# =========================================================================== #


def test_free_ram_is_discounted_for_production():
    """The probe runs idle; production shares RAM with the customer's other apps."""
    profile = _profile(ram_free_mb=2600, headroom_factor=0.85)
    assert profile.usable_ram_mb == 2210
    assert profile.usable_ram_mb < profile.ram_free_mb


def test_probe_bundle_touches_no_customer_data():
    """It measures hardware, not content — which is why it can run on regulated data."""
    assert ProbeBundle().manifest()["touches_customer_data"] is False


def test_profile_round_trips(tmp_path):
    profile = _profile()
    profile.save(tmp_path / "p.json")
    restored = DeviceProfile.load(tmp_path / "p.json")
    assert restored.bw_eff_gbps == profile.bw_eff_gbps
    assert restored.source is ProfileSource.PROBE
    assert restored.simd == profile.simd


# =========================================================================== #
# §12 — the compounding device model


# =========================================================================== #


def _obs(soc="sm8550", bw=9.2, derate=0.6):
    return ProbeObservation(soc=soc, bw_eff_gbps=bw, overhead_ms_per_token=2.1,
                            thermal_derate=derate, ram_total_mb=8000)


def test_interpolation_needs_enough_observations():
    db = DeviceDB()
    db.record(_obs())
    assert db.needs_probe("sm8550") is True
    assert db.interpolate("sm8550") is None

    db.record(_obs(bw=9.6))
    db.record(_obs(bw=8.9))
    assert db.needs_probe("sm8550") is False
    assert db.interpolate("sm8550") is not None


def test_an_interpolated_profile_never_reaches_the_measured_tier():
    """However many observations back it, it is not a measurement of THIS unit."""
    db = DeviceDB()
    for bw in (9.2, 9.6, 8.9, 9.1, 9.4):
        db.record(_obs(bw=bw))
    profile = db.interpolate("sm8550")
    assert profile.source is ProfileSource.INTERPOLATED
    assert profile.measured is False
    assert profile.tier is Tier.ANALYTICAL


def test_summary_uses_the_median_bandwidth_and_the_worst_derate():
    db = DeviceDB()
    db.record(_obs(bw=9.0, derate=0.9))
    db.record(_obs(bw=9.5, derate=0.5))     # one throttled unit
    db.record(_obs(bw=100.0, derate=0.8))   # one outlier
    summary = db.summarise("sm8550")
    assert summary.bw_eff_gbps == 9.5                 # median, not mean
    assert summary.thermal_derate == 0.5              # worst, not typical


def test_high_dispersion_is_surfaced():
    """One SoC name can hide different phones, thermal designs and memory."""
    db = DeviceDB()
    for bw in (4.0, 9.0, 15.0):
        db.record(_obs(bw=bw))
    assert db.summarise("sm8550").dispersion > 0.25
    assert "sm8550" in db.coverage()["high_dispersion"]


def test_only_measured_profiles_enter_the_database():
    """Otherwise the model trains on its own output."""
    db = DeviceDB()
    with pytest.raises(ValueError, match="only measured"):
        db.record_profile(_profile(source=ProfileSource.INTERPOLATED), "sm8550")
    db.record_profile(_profile(), "sm8550")
    assert len(db) == 1


def test_coverage_grows_with_deployments():
    db = DeviceDB()
    assert db.coverage()["socs_confident"] == 0
    for _ in range(3):
        db.record(_obs())
    assert db.coverage()["socs_confident"] == 1
    assert "sm8550" in db.coverage()["probe_skippable"]


def test_resolution_ladder_prefers_probe_then_interpolation_then_prior():
    db = DeviceDB()
    prior = _profile(source=ProfileSource.ASSUMED)

    _, source = resolve_profile("unseen", db, prior=prior)
    assert source is ProfileSource.ASSUMED

    for _ in range(3):
        db.record(_obs(soc="sm8550"))
    _, source = resolve_profile("sm8550", db, prior=prior)
    assert source is ProfileSource.INTERPOLATED

    _, source = resolve_profile("sm8550", db, probed=_profile(), prior=prior)
    assert source is ProfileSource.PROBE


def test_device_db_round_trips(tmp_path):
    db = DeviceDB()
    db.record(_obs())
    db.save(tmp_path / "db.json")
    restored = DeviceDB()
    restored.load(tmp_path / "db.json")
    assert len(restored) == 1
    assert restored.socs == ["sm8550"]
