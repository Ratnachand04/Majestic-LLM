"""Tests for the device-aware feasibility engine."""
from __future__ import annotations

import pytest

from modelrig.buildspec import BuildSpec, DeviceProfile, TrainingMethod
from modelrig.feasibility import (
    HeuristicFeasibilityEngine,
    HeuristicPerfPredictor,
    YamlDeviceProfiler,
)

PHONE = DeviceProfile(name="phone", ram_gb=6, accelerator="cpu",
                      compute_gflops=40, battery_wh=17)


# --- device profiler ---------------------------------------------------- #
def test_profiler_loads_devices():
    prof = YamlDeviceProfiler("configs/devices.yaml")
    dev = prof.profile("android_midrange")
    assert dev.ram_gb == 6
    assert dev.accelerator == "cpu"
    assert len(prof.all()) >= 4


def test_profiler_unknown_device_raises():
    with pytest.raises(KeyError):
        YamlDeviceProfiler("configs/devices.yaml").profile("nonexistent")


# --- perf predictor ----------------------------------------------------- #
def test_quantization_reduces_footprint():
    p = HeuristicPerfPredictor()
    big = p.predict("gpt2", "fp32", "gguf", PHONE)
    small = p.predict("gpt2", "int4", "gguf", PHONE)
    assert small.ram_mb < big.ram_mb


def test_gpu_faster_than_cpu():
    p = HeuristicPerfPredictor()
    gpu = DeviceProfile(name="g", ram_gb=16, accelerator="gpu", battery_wh=80)
    cpu = DeviceProfile(name="c", ram_gb=16, accelerator="cpu", battery_wh=80)
    assert p.predict("gpt2", "int8", "gguf", gpu).latency_ms < \
        p.predict("gpt2", "int8", "gguf", cpu).latency_ms


def test_mains_powered_has_zero_battery_draw():
    p = HeuristicPerfPredictor()
    pi = DeviceProfile(name="pi", ram_gb=4, accelerator="cpu", battery_wh=0)
    assert p.predict("gpt2", "int8", "gguf", pi).battery_pct_per_hr == 0.0


# --- feasibility engine ------------------------------------------------- #
def _spec(**extras) -> BuildSpec:
    return BuildSpec(task="t", base_model=extras.pop("base_model", "centroid"),
                     method=TrainingMethod.CENTROID, quantization=extras.pop("quant", "int8"),
                     device=PHONE, extras=extras)


def test_small_model_is_feasible_with_headroom():
    engine = HeuristicFeasibilityEngine()
    verdict = engine.evaluate(_spec())
    assert verdict.feasible is True
    assert verdict.ram_headroom_mb > 0
    assert verdict.reasons == []
    assert engine.check(_spec()) is True


def test_large_model_exceeds_ram_budget():
    engine = HeuristicFeasibilityEngine()
    verdict = engine.evaluate(_spec(base_model="giant", quant="fp16", params_b=7.0))
    assert verdict.feasible is False
    assert any("RAM" in r for r in verdict.reasons)


def test_latency_budget_can_fail():
    engine = HeuristicFeasibilityEngine()
    verdict = engine.evaluate(_spec(latency_budget_ms=0.0001))
    assert verdict.feasible is False
    assert any("latency" in r for r in verdict.reasons)


def test_device_from_profiler_by_name():
    engine = HeuristicFeasibilityEngine()
    spec = BuildSpec(task="t", base_model="centroid", method=TrainingMethod.CENTROID,
                     extras={"device": "laptop_gpu"})
    assert engine.check(spec) is True


def test_missing_device_raises():
    engine = HeuristicFeasibilityEngine()
    spec = BuildSpec(task="t", base_model="centroid", method=TrainingMethod.CENTROID)
    with pytest.raises(ValueError):
        engine.evaluate(spec)


# --- KV cache accounting (A-01) ------------------------------------------ #
def test_kv_cache_is_counted_in_the_footprint():
    """Deployable size = weights + KV cache + runtime, never weights alone."""
    p = HeuristicPerfPredictor()
    est = p.predict("Qwen/Qwen3-1.7B", "int4", "gguf", PHONE, context_length=2048)
    assert est.kv_cache_mb > 0
    assert est.ram_mb == pytest.approx(
        est.weights_mb + est.kv_cache_mb + est.runtime_mb, rel=1e-3
    )


def test_kv_cache_grows_linearly_with_context():
    """The variable nobody budgets for, and the usual cause of on-device OOM."""
    p = HeuristicPerfPredictor()
    short = p.predict("Qwen/Qwen3-1.7B", "int4", "gguf", PHONE, context_length=1024)
    long = p.predict("Qwen/Qwen3-1.7B", "int4", "gguf", PHONE, context_length=8192)
    assert long.kv_cache_mb == pytest.approx(short.kv_cache_mb * 8, rel=1e-3)
    assert long.ram_mb > short.ram_mb


def test_long_context_can_flip_a_fitting_model_to_infeasible():
    engine = HeuristicFeasibilityEngine()
    fits = BuildSpec(task="t", base_model="Qwen/Qwen3-1.7B",
                     method=TrainingMethod.CENTROID, quantization="int4",
                     device=PHONE, extras={"context_length": 2048})
    huge_ctx = BuildSpec(task="t", base_model="Qwen/Qwen3-1.7B",
                         method=TrainingMethod.CENTROID, quantization="int4",
                         device=PHONE, extras={"context_length": 131072})
    assert engine.evaluate(fits).feasible is True
    assert engine.evaluate(huge_ctx).feasible is False


def test_four_bit_weights_match_the_published_size_ladder():
    """A-01 size ladder: 1.7B at 4-bit is ~1.1 GB on disk.

    Deployed int4 costs ~0.65 GB/B, not the theoretical 0.55: mixed-precision
    schemes keep embeddings and some tensors wider. Budgeting with the
    theoretical figure under-counts every footprint by ~18%.
    """
    p = HeuristicPerfPredictor()
    est = p.predict("Qwen/Qwen3-1.7B", "int4", "gguf", PHONE)
    assert 1050 < est.weights_mb < 1150     # ~1.1 GB, per the ladder


@pytest.mark.parametrize("params_b,disk_gb", [(0.6, 0.4), (1.7, 1.1), (3.4, 2.2)])
def test_size_ladder_is_reproduced_across_the_range(params_b, disk_gb):
    from modelrig.catalogue import BYTES_PER_PARAM

    predicted = params_b * BYTES_PER_PARAM["int4"]
    assert abs(predicted - disk_gb) / disk_gb < 0.20


def test_two_times_file_size_rule_caps_the_device():
    """A 4 GB phone tops out near 1.7B at 4-bit, not the 3.5B file size suggests."""
    engine = HeuristicFeasibilityEngine()
    tablet = DeviceProfile(name="t", ram_gb=4, accelerator="cpu", battery_wh=28)
    max_b = engine.max_base_for(tablet, "int4")
    assert 1.0 < max_b < 2.5     # comfortably below the naive 3.5B


def test_max_base_grows_with_device_ram():
    engine = HeuristicFeasibilityEngine()
    small = DeviceProfile(name="s", ram_gb=3, accelerator="cpu")
    big = DeviceProfile(name="b", ram_gb=16, accelerator="cpu")
    assert engine.max_base_for(big) > engine.max_base_for(small)


def test_estimates_are_flagged_unmeasured():
    """GAP-10: mobile numbers are design targets until a device lab measures them."""
    p = HeuristicPerfPredictor()
    assert p.predict("Qwen/Qwen3-1.7B", "int4", "gguf", PHONE).measured is False


def test_outcome_predictor_cold_starts_then_predicts():
    """GAP-02 scaffold: no prediction until enough history exists."""
    from modelrig.feasibility import OutcomePredictor

    pred = OutcomePredictor(min_history=3)
    assert pred.predict_pass_probability("extract") is None
    for i in range(3):
        pred.record(f"s{i}", f"p{i}", "extract", passed=i > 0)
    assert pred.predict_pass_probability("extract") == pytest.approx(2 / 3, abs=1e-4)
