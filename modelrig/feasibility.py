"""Device-aware feasibility engine — the core differentiator.

Predicts latency / RAM / battery for a (model x quant x runtime x device) and
provides a *compile-time* verdict before a build is run: either a pass with a
numeric guarantee (headroom), or a fail with a concrete reason.

The memory model follows A-01: the deployable size is bounded by

    weights  +  KV cache  +  runtime/activations   <=   device free RAM

The KV cache is the term nobody budgets for — it grows linearly with context
length and batch, and is the usual cause of on-device OOM. Two independent
checks are applied:

1. **Component budget** — the three terms above must fit the usable RAM.
2. **The 2x working rule** — device free RAM must be at least twice the model
   file size. This is why a 4 GB phone tops out near a 1.7B base at 4-bit, not
   the 3.5B the file size alone would suggest.

GAP-02 (open): predicting whether a build will PASS ITS EVAL GATE before
spending money is not solved here. This engine answers "does it fit and is it
fast enough", not "will it be good enough". :class:`OutcomePredictor` is the
scaffold for that learned predictor, fitted from accumulated
(spec, plan, outcome) triples.

GAP-10 (open): mobile numbers below are heuristics, not measurements. Profiles
carry a ``measured`` flag; until a physical device lab fills it in, treat
on-device latency and battery as design targets. Energy and thermal throttling —
not raw speed — are the binding mobile constraints (MELTing Point 2403.12844).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from majestic.logging_utils import get_logger
from modelrig.buildspec import BuildSpec, DeviceProfile
from modelrig.catalogue import BYTES_PER_PARAM, DEFAULT_CATALOGUE, Catalogue

logger = get_logger(__name__)

# Rough parameter counts (in billions) for models outside the catalogue.
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
# Relative decode speedups per quantisation vs int8.
_QUANT_SPEEDUP = {"int4": 1.6, "int8": 1.0, "fp16": 0.7, "fp32": 0.4, "none": 0.7}
# Typical active power draw (watts) by accelerator, for battery estimates.
_POWER_W = {"cpu": 5.0, "gpu": 30.0, "npu": 3.0}
# Fixed runtime/activation overhead: framework, embedder, grammar state, app.
_RUNTIME_OVERHEAD_MB = 120.0
# Fraction of device RAM usable by a single app before the OS pushes back.
_USABLE_RAM_FRACTION = 0.6
# Default planning context length when the spec does not state one.
_DEFAULT_CONTEXT = 2048
# A-01 working rule: free RAM must be at least this multiple of the file size.
_FILE_SIZE_HEADROOM_MULTIPLE = 2.0
# KV-cache geometry fallback for models outside the catalogue.
_FALLBACK_LAYERS, _FALLBACK_KV_HEADS, _FALLBACK_HEAD_DIM = 28, 8, 128


@dataclass
class PerfEstimate:
    """Predicted on-device cost, broken down so a failure names its cause."""

    latency_ms: float           # per generated token (or per inference for classifiers)
    ram_mb: float               # weights + KV cache + runtime
    battery_pct_per_hr: float
    weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    runtime_mb: float = 0.0
    context_length: int = _DEFAULT_CONTEXT
    measured: bool = False      # False => heuristic, not a device-lab measurement


@dataclass
class FeasibilityVerdict:
    """Numeric verdict with headroom and concrete reasons on failure."""

    feasible: bool
    estimate: PerfEstimate
    ram_budget_mb: float
    ram_headroom_mb: float
    latency_budget_ms: float | None = None
    reasons: list[str] = field(default_factory=list)
    guarantee: dict[str, float] = field(default_factory=dict)


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
                    usable_ram_fraction=float(
                        entry.get("usable_ram_fraction", _USABLE_RAM_FRACTION)
                    ),
                    measured=bool(entry.get("measured", False)),
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

    def __init__(self, catalogue: Catalogue | None = None) -> None:
        self.catalogue = catalogue or DEFAULT_CATALOGUE

    def _params_b(self, model: str, params_b: float | None) -> float:
        if params_b is not None:
            return params_b
        entry = self.catalogue.base(model) or self.catalogue.teacher(model)
        if entry is not None:
            return entry.params_b
        return _MODEL_PARAMS_B.get(model, 0.05)

    def _kv_bytes_per_token(self, model: str, bit_width: str) -> float:
        """KV-cache bytes per token. The cache is held at fp16 even for int4 weights."""
        entry = self.catalogue.base(model) or self.catalogue.teacher(model)
        if entry is not None:
            return entry.kv_bytes_per_token(bytes_per_element=2.0)
        return 2.0 * _FALLBACK_LAYERS * _FALLBACK_KV_HEADS * _FALLBACK_HEAD_DIM * 2.0

    def predict(
        self,
        model: str,
        quant: str,
        runtime: str,
        device: DeviceProfile,
        params_b: float | None = None,
        context_length: int = _DEFAULT_CONTEXT,
        batch: int = 1,
    ) -> PerfEstimate:
        params = self._params_b(model, params_b)
        bytes_pp = BYTES_PER_PARAM.get(quant, 2.0)

        weights_mb = params * 1e9 * bytes_pp / 1e6
        # Tiny non-transformer artefacts (centroid/kNN) carry no KV cache.
        kv_mb = 0.0
        if params >= 0.01:
            kv_mb = (
                self._kv_bytes_per_token(model, quant) * context_length * batch / 1e6
            )
        runtime_mb = _RUNTIME_OVERHEAD_MB
        ram_mb = weights_mb + kv_mb + runtime_mb

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
            weights_mb=round(weights_mb, 1),
            kv_cache_mb=round(kv_mb, 1),
            runtime_mb=round(runtime_mb, 1),
            context_length=context_length,
            measured=device.measured,
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
        extras = spec.extras
        kwargs = {}
        if isinstance(self.predictor, HeuristicPerfPredictor):
            kwargs = {
                "params_b": extras.get("params_b"),
                "context_length": int(extras.get("context_length", _DEFAULT_CONTEXT)),
                "batch": int(extras.get("batch", 1)),
            }
        estimate = self.predictor.predict(
            spec.base_model, spec.quantization, spec.runtime, device, **kwargs
        )

        usable = getattr(device, "usable_ram_fraction", _USABLE_RAM_FRACTION)
        ram_budget = device.ram_gb * 1024.0 * usable
        ram_headroom = ram_budget - estimate.ram_mb
        latency_budget = extras.get("latency_budget_ms")
        battery_budget = extras.get("battery_pct_per_hr_budget")

        reasons: list[str] = []
        if estimate.ram_mb > ram_budget:
            reasons.append(
                f"RAM {estimate.ram_mb:.0f}MB (weights {estimate.weights_mb:.0f} + "
                f"KV {estimate.kv_cache_mb:.0f} + runtime {estimate.runtime_mb:.0f}) "
                f"exceeds budget {ram_budget:.0f}MB ({device.ram_gb}GB * {usable})"
            )

        # A-01 working rule: device FREE RAM >= 2x the model file size. Measured
        # against free RAM, not total: it is what makes a 4 GB phone top out near
        # a 1.7B base at 4-bit rather than the 3.5B the file size alone suggests.
        required_mb = estimate.weights_mb * _FILE_SIZE_HEADROOM_MULTIPLE
        if estimate.weights_mb > 0 and required_mb > ram_budget:
            reasons.append(
                f"file-size rule: {_FILE_SIZE_HEADROOM_MULTIPLE:g}x weights "
                f"({required_mb:.0f}MB) exceeds free RAM {ram_budget:.0f}MB"
            )

        if latency_budget is not None and estimate.latency_ms > latency_budget:
            reasons.append(
                f"latency {estimate.latency_ms:.1f}ms exceeds budget {latency_budget}ms"
            )
        if battery_budget is not None and estimate.battery_pct_per_hr > battery_budget:
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
            guarantee={
                "ram_mb": estimate.ram_mb,
                "ram_budget_mb": round(ram_budget, 1),
                "latency_ms": estimate.latency_ms,
                "battery_pct_per_hr": estimate.battery_pct_per_hr,
            },
        )

    def check(self, spec: BuildSpec) -> bool:
        verdict = self.evaluate(spec)
        logger.info(
            "feasibility: %s (ram %.0f/%.0f MB, kv %.0f MB, %.1f ms)",
            "PASS" if verdict.feasible else "FAIL",
            verdict.estimate.ram_mb, verdict.ram_budget_mb,
            verdict.estimate.kv_cache_mb, verdict.estimate.latency_ms,
        )
        return verdict.feasible

    def max_base_for(
        self, device: DeviceProfile, bit_width: str = "int4",
        context_length: int = _DEFAULT_CONTEXT,
    ) -> float:
        """Largest base (billions of params) that fits this device at ``bit_width``.

        Answers the A-01 size-ladder question directly, and is what the planner
        uses to cap the base before it ever searches the catalogue.
        """
        usable = getattr(device, "usable_ram_fraction", _USABLE_RAM_FRACTION)
        budget_mb = device.ram_gb * 1024.0 * usable
        kv_per_token = (
            2.0 * _FALLBACK_LAYERS * _FALLBACK_KV_HEADS * _FALLBACK_HEAD_DIM * 2.0
        )
        available = budget_mb - _RUNTIME_OVERHEAD_MB - (
            kv_per_token * context_length / 1e6
        )
        if available <= 0:
            return 0.0
        bytes_pp = BYTES_PER_PARAM.get(bit_width, 2.0)
        by_budget = available * 1e6 / (bytes_pp * 1e9)
        # Also respect the 2x file-size rule against FREE RAM (A-01).
        by_rule = (budget_mb / _FILE_SIZE_HEADROOM_MULTIPLE) * 1e6 / (bytes_pp * 1e9)
        return round(max(min(by_budget, by_rule), 0.0), 3)


# --------------------------------------------------------------------------- #
class OutcomePredictor:
    """GAP-02 scaffold: predict gate pass from a spec before spending money.

    Scaling laws predict loss from compute, not task success from a
    specification; AutoML meta-learning predicts configuration quality on
    tabular data, not LLM task outcomes from a natural-language description. The
    real predictor is fitted from accumulated (Spec IR embedding, Build Plan,
    eval outcome) triples — this class records those triples and serves a
    frequency baseline until enough history exists to fit a model.

    Without it, cold-start builds burn money on plans that were never going to
    work and margin collapses.
    """

    def __init__(self, min_history: int = 20) -> None:
        self.history: list[dict] = []
        self.min_history = min_history

    def record(self, spec_hash: str, plan_hash: str, primitive: str, passed: bool) -> None:
        """Append one observed build outcome."""
        self.history.append(
            {
                "spec_hash": spec_hash,
                "plan_hash": plan_hash,
                "primitive": primitive,
                "passed": bool(passed),
            }
        )

    def predict_pass_probability(self, primitive: str) -> float | None:
        """Historical pass rate for a primitive, or ``None`` while cold-starting."""
        rows = [h for h in self.history if h["primitive"] == primitive]
        if len(rows) < self.min_history:
            return None
        return round(sum(1 for r in rows if r["passed"]) / len(rows), 4)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            self.history = json.loads(p.read_text(encoding="utf-8"))
