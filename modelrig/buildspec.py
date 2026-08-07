"""The declarative BuildSpec — the central artifact of the factory.

One BuildSpec fully describes a model build and is the only thing the compiler
consumes. Both user flows (build-your-own, describe-it) produce the same spec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TrainingMethod(str, Enum):
    NONE_RAG = "none_rag"     # no training: nearest-neighbour over embeddings
    CENTROID = "centroid"     # lightweight non-gradient fit (offline-friendly)
    LORA = "lora"             # PEFT LoRA fine-tune (heavy; lazy)
    QLORA = "qlora"           # quantized LoRA (heavy; lazy)
    DISTILL = "distill"       # teacher->student distillation (heavy; lazy)


# Methods that need torch/transformers/peft (imported lazily, integration-gated).
HEAVY_METHODS = {TrainingMethod.LORA, TrainingMethod.QLORA, TrainingMethod.DISTILL}


@dataclass
class DeviceProfile:
    name: str
    ram_gb: float
    accelerator: str = "cpu"        # cpu | gpu | npu
    compute_gflops: float = 10.0    # rough sustained throughput for the predictor
    battery_wh: float = 0.0         # 0 => mains powered (no battery budget)
    usable_ram_fraction: float = 0.6  # share of RAM one app may hold before the OS pushes back
    measured: bool = False          # True only when a device lab measured this profile (GAP-10)


@dataclass
class BuildSpec:
    """Fully describes one model build; consumed by the compiler."""

    task: str
    base_model: str
    method: TrainingMethod = TrainingMethod.LORA
    quantization: str = "int4"      # int4 | int8 | none
    runtime: str = "gguf"           # gguf | onnx | npz | coreml | tflite | executorch | mlc
    device: Optional[DeviceProfile] = None
    target_metric: Optional[str] = None

    # --- data & evaluation ---
    dataset: str = "builtin:sentiment"   # "builtin:<name>" or a path to jsonl/csv
    eval_metric: str = "accuracy"
    target_score: float = 0.6            # the quality gate threshold
    test_split: float = 0.3
    seed: int = 0

    online: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_heavy(self) -> bool:
        return self.method in HEAVY_METHODS


def validate_spec(spec: BuildSpec) -> list[str]:
    """Return a list of human-readable validation errors (empty => valid)."""
    errors: list[str] = []
    if not spec.task or not spec.task.strip():
        errors.append("task must be a non-empty string")
    if not spec.base_model or not spec.base_model.strip():
        errors.append("base_model must be a non-empty string")
    if not isinstance(spec.method, TrainingMethod):
        errors.append(f"method must be a TrainingMethod, got {type(spec.method).__name__}")
    if spec.quantization not in {"int4", "int8", "none"}:
        errors.append(f"quantization must be int4|int8|none, got {spec.quantization!r}")
    if not 0.0 < spec.test_split < 1.0:
        errors.append(f"test_split must be in (0, 1), got {spec.test_split}")
    if not 0.0 <= spec.target_score <= 1.0:
        errors.append(f"target_score must be in [0, 1], got {spec.target_score}")
    if spec.device is not None and spec.device.ram_gb <= 0:
        errors.append("device.ram_gb must be positive")
    return errors


def ensure_valid(spec: BuildSpec) -> None:
    """Raise ``ValueError`` with all problems if the spec is invalid."""
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid BuildSpec:\n  - " + "\n  - ".join(errors))


def spec_from_dict(data: dict[str, Any]) -> BuildSpec:
    """Build a :class:`BuildSpec` from a plain dict (YAML/JSON friendly)."""
    data = dict(data)
    if "method" in data and not isinstance(data["method"], TrainingMethod):
        data["method"] = TrainingMethod(str(data["method"]).lower())
    device = data.get("device")
    if isinstance(device, dict):
        data["device"] = DeviceProfile(
            name=device["name"],
            ram_gb=float(device["ram_gb"]),
            accelerator=device.get("accelerator", "cpu"),
            compute_gflops=float(device.get("compute_gflops", 10.0)),
            battery_wh=float(device.get("battery_wh", 0.0)),
        )
    known = {f for f in BuildSpec.__dataclass_fields__}  # noqa: C416
    return BuildSpec(**{k: v for k, v in data.items() if k in known})


def load_spec(path: str | Path) -> BuildSpec:
    """Load a :class:`BuildSpec` from a ``.json``, ``.yaml`` or ``.yml`` file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"spec file not found: {path}")
    text = p.read_text(encoding="utf-8-sig")  # tolerate a UTF-8 BOM
    if p.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return spec_from_dict(data)
