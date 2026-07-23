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
