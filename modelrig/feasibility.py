"""Device-aware feasibility engine — the core differentiator.

Predicts latency / RAM / battery for a (model x quant x runtime x device) and
provides a *compile-time* feasibility verdict before a build is run: either a
pass with a numeric guarantee (headroom), or a fail with a concrete reason.

The predictor here is a transparent heuristic over a small benchmark table. A
learned predictor (regression over real on-device measurements) is left as a
documented TODO — the interface is ready for it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from majestic.logging_utils import get_logger
from modelrig.buildspec import BuildSpec, DeviceProfile

logger = get_logger(__name__)

# --- small benchmark table (the cost model's constants) ----------------- #
_BYTES_PER_PARAM = {"int4": 0.5, "int8": 1.0, "fp16": 2.0, "fp32": 4.0, "none": 2.0}

# Rough parameter counts (in billions) for known small bases. Unknown models
# fall back to spec.extras['params_b'] or a small default.
_MODEL_PARAMS_B = {
    "sshleifer/tiny-gpt2": 0.001,
    "prajjwal1/bert-tiny": 0.004,
    "distilgpt2": 0.082,
    "gpt2": 0.124,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.1,
    "centroid": 0.0002,
    "knn": 0.0002,
    "hashing-centroid": 0.0002,
}

# Sustained decode throughput baseline (tokens/sec) for a 1B-param int8 model.
_THROUGHPUT_1B_INT8 = {"cpu": 12.0, "gpu": 120.0, "npu": 60.0}
# Relative decode speedups per quantization vs int8.
_QUANT_SPEEDUP = {"int4": 1.6, "int8": 1.0, "fp16": 0.7, "fp32": 0.4, "none": 0.7}
# Typical active power draw (watts) by accelerator, for battery estimates.
_POWER_W = {"cpu": 5.0, "gpu": 30.0, "npu": 3.0}
# Fixed runtime/activation overhead added to the weight footprint (MB).
_RUNTIME_OVERHEAD_MB = 120.0
# Fraction of device RAM usable by a single app before the OS pushes back.
_USABLE_RAM_FRACTION = 0.6


@dataclass
class PerfEstimate:
    latency_ms: float          # per generated token (or per inference for classifiers)
    ram_mb: float
    battery_pct_per_hr: float


@dataclass
class FeasibilityVerdict:
    feasible: bool
    estimate: PerfEstimate
    ram_budget_mb: float
    ram_headroom_mb: float
    latency_budget_ms: float | None = None
    reasons: list[str] = field(default_factory=list)


class PerfPredictor(ABC):
    """Predicts on-device performance from a benchmark-derived cost model."""

    @abstractmethod
    def predict(
        self, model: str, quant: str, runtime: str, device: DeviceProfile
    ) -> PerfEstimate:
        raise NotImplementedError


class DeviceProfiler(ABC):
    @abstractmethod
    def profile(self, device_name: str) -> DeviceProfile:
        raise NotImplementedError


class FeasibilityEngine(ABC):
    @abstractmethod
    def check(self, spec: BuildSpec) -> bool:
        """True (with a guarantee) if the spec fits the device budget, else False."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class YamlDeviceProfiler(DeviceProfiler):
    """Loads device profiles from ``configs/devices.yaml`` (or a given path)."""

    def __init__(self, path: str | Path = "configs/devices.yaml") -> None:
        self._devices: dict[str, DeviceProfile] = {}
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            for entry in raw.get("devices", []):
                profile = DeviceProfile(
                    name=entry["name"],
                    ram_gb=float(entry["ram_gb"]),
                    accelerator=entry.get("accelerator", "cpu"),
                    compute_gflops=float(entry.get("compute_gflops", 10.0)),
                    battery_wh=float(entry.get("battery_wh", 0.0)),
                )
                self._devices[profile.name] = profile

    def profile(self, device_name: str) -> DeviceProfile:
        if device_name not in self._devices:
            raise KeyError(f"unknown device {device_name!r}; have {list(self._devices)}")
        return self._devices[device_name]

    def all(self) -> list[DeviceProfile]:
        return list(self._devices.values())


class HeuristicPerfPredictor(PerfPredictor):
    """Transparent cost model: footprint, latency and battery from constants."""

    def _params_b(self, model: str, params_b: float | None) -> float:
        if params_b is not None:
            return params_b
        return _MODEL_PARAMS_B.get(model, 0.05)

    def predict(
        self,
        model: str,
        quant: str,
        runtime: str,
        device: DeviceProfile,
        params_b: float | None = None,
    ) -> PerfEstimate:
        params = self._params_b(model, params_b)
        bytes_pp = _BYTES_PER_PARAM.get(quant, 2.0)

        ram_mb = params * 1e9 * bytes_pp / 1e6 + _RUNTIME_OVERHEAD_MB

        base = _THROUGHPUT_1B_INT8.get(device.accelerator, 12.0)
        speedup = _QUANT_SPEEDUP.get(quant, 0.7)
        throughput = base * (1.0 / max(params, 1e-3)) * speedup  # tokens/sec
        latency_ms = 1000.0 / max(throughput, 1e-6)

        power_w = _POWER_W.get(device.accelerator, 5.0)
        battery_pct_per_hr = (
            power_w / device.battery_wh * 100.0 if device.battery_wh > 0 else 0.0
        )
        return PerfEstimate(
            latency_ms=round(latency_ms, 3),
            ram_mb=round(ram_mb, 1),
            battery_pct_per_hr=round(battery_pct_per_hr, 2),
        )


class HeuristicFeasibilityEngine(FeasibilityEngine):
    """Compares predicted footprint/latency/battery to the device budget."""

    def __init__(
        self,
        profiler: DeviceProfiler | None = None,
        predictor: PerfPredictor | None = None,
    ) -> None:
        self.profiler = profiler or YamlDeviceProfiler()
        self.predictor = predictor or HeuristicPerfPredictor()

    def _resolve_device(self, spec: BuildSpec) -> DeviceProfile:
        if spec.device is not None:
            return spec.device
        name = spec.extras.get("device")
        if name:
            return self.profiler.profile(name)
        raise ValueError("no device: set spec.device or spec.extras['device']")

    def evaluate(self, spec: BuildSpec) -> FeasibilityVerdict:
        """Full numeric verdict: estimate + budget + headroom + reasons."""
        device = self._resolve_device(spec)
        params_b = spec.extras.get("params_b")
        estimate = self.predictor.predict(
            spec.base_model, spec.quantization, spec.runtime, device,
            **({"params_b": params_b} if isinstance(self.predictor, HeuristicPerfPredictor) else {}),
        )

        ram_budget = device.ram_gb * 1024.0 * _USABLE_RAM_FRACTION
        ram_headroom = ram_budget - estimate.ram_mb
        latency_budget = spec.extras.get("latency_budget_ms")
        battery_budget = spec.extras.get("battery_pct_per_hr_budget")

        reasons: list[str] = []
        if estimate.ram_mb > ram_budget:
            reasons.append(
                f"RAM {estimate.ram_mb:.0f}MB exceeds budget {ram_budget:.0f}MB "
                f"({device.ram_gb}GB * {_USABLE_RAM_FRACTION})"
            )
        if latency_budget is not None and estimate.latency_ms > latency_budget:
            reasons.append(
                f"latency {estimate.latency_ms:.1f}ms exceeds budget {latency_budget}ms"
            )
        if (
            battery_budget is not None
            and estimate.battery_pct_per_hr > battery_budget
        ):
            reasons.append(
                f"battery {estimate.battery_pct_per_hr:.1f}%/hr exceeds "
                f"budget {battery_budget}%/hr"
            )

        return FeasibilityVerdict(
            feasible=not reasons,
            estimate=estimate,
            ram_budget_mb=round(ram_budget, 1),
            ram_headroom_mb=round(ram_headroom, 1),
            latency_budget_ms=latency_budget,
            reasons=reasons,
        )

    def check(self, spec: BuildSpec) -> bool:
        verdict = self.evaluate(spec)
        logger.info(
            "feasibility: %s (ram %.0f/%.0f MB, %.1f ms)",
            "PASS" if verdict.feasible else "FAIL",
            verdict.estimate.ram_mb, verdict.ram_budget_mb, verdict.estimate.latency_ms,
        )
        return verdict.feasible
