"""The five build planes: data, training, eval, compression, export.

Each plane is one stage of a reproducible build and returns artifacts (a dict)
that are merged into a shared context for the next plane. The default path is
fully offline (NumPy classifier); the heavy LoRA/QLORA/DISTILL path is imported
lazily and is out of scope for the offline suite.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from majestic.logging_utils import get_logger
from modelrig import classifier
from modelrig.buildspec import BuildSpec, TrainingMethod
from modelrig.datasets import load_dataset, split_dataset

logger = get_logger(__name__)

_NUMPY_KINDS = {"centroid", "knn"}


class Plane(ABC):
    name: str = "plane"

    @abstractmethod
    def run(self, spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute this plane; return artifacts for the next plane."""
        raise NotImplementedError


def _predict(model: dict[str, Any], texts: list[str]) -> list[str]:
    """Dispatch prediction to the right backend."""
    if model.get("kind") in _NUMPY_KINDS:
        return classifier.predict(model, texts)
    from modelrig.training_hf import hf_predict  # lazy heavy path

    return hf_predict(model, texts)


class DataPlane(Plane):
    """Ingest, format, and split the dataset (deterministic)."""

    name = "data"

    def run(self, spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
        rows = load_dataset(spec.dataset)
        train, test = split_dataset(rows, spec.test_split, spec.seed)
        labels = sorted({label for _, label in rows})
        logger.info("data: %d train / %d test / %d labels", len(train), len(test), len(labels))
        return {"train": train, "test": test, "labels": labels,
                "n_train": len(train), "n_test": len(test)}


class TrainingPlane(Plane):
    """Fit the specialist. Offline: centroid / kNN. Heavy: LoRA/QLORA/DISTILL."""

    name = "training"

    def run(self, spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
        if spec.is_heavy:
            from modelrig.training_hf import train_hf  # lazy heavy path

            model = train_hf(spec, ctx)
            return {"model": model, "backend": model.get("kind", "hf")}

        train, labels = ctx["train"], ctx["labels"]
        if spec.method == TrainingMethod.CENTROID:
            model = classifier.fit_centroid(train, labels)
        elif spec.method == TrainingMethod.NONE_RAG:
            model = classifier.fit_knn(train, labels)
        else:  # pragma: no cover - guarded by validate/compile
            raise ValueError(f"unsupported offline method {spec.method}")
        logger.info("training: fit %s model", model["kind"])
        return {"model": model, "backend": model["kind"]}


class EvalPlane(Plane):
    """The quality gate: evaluate on the held-out split and pass/fail the build."""

    name = "eval"

    def run(self, spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
        model, test = ctx["model"], ctx["test"]
        texts = [t for t, _ in test]
        gold = [label for _, label in test]
        preds = _predict(model, texts)
        correct = sum(int(p == g) for p, g in zip(preds, gold))
        score = correct / len(gold) if gold else 0.0
        passed = score >= spec.target_score
        report = {
            "metric": spec.eval_metric,
            "score": round(score, 4),
            "threshold": spec.target_score,
            "passed": passed,
            "n_test": len(gold),
        }
        logger.info("eval: %s=%.3f (gate %.2f) -> %s",
                    spec.eval_metric, score, spec.target_score,
                    "PASS" if passed else "FAIL")
        return {"eval": report, "gate_passed": passed}


class CompressionPlane(Plane):
    """Quantize the model (int8 / packed int4). Skipped when quantization=none."""

    name = "compression"

    def run(self, spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
        model = ctx["model"]
        if model.get("kind") not in _NUMPY_KINDS:
            from modelrig.training_hf import compress_hf  # lazy heavy path

            return compress_hf(spec, model)
        compressed, report = classifier.quantize_model(model, spec.quantization)
        logger.info("compression: %s ratio=%.2fx", report["method"], report["ratio"])
        return {"model": compressed, "compression": report}


class ExportPlane(Plane):
    """Serialize the model + reports to the build's output directory."""

    name = "export"

    def run(self, spec: BuildSpec, ctx: dict[str, Any]) -> dict[str, Any]:
        out_dir = Path(ctx["out_dir"])
        model = ctx["model"]
        if model.get("kind") in _NUMPY_KINDS:
            classifier.save_model(model, out_dir)
            runtime = "npz"
        else:
            from modelrig.training_hf import export_hf  # lazy heavy path

            runtime = export_hf(spec, model, out_dir)

        metadata = {
            "build_id": ctx["build_id"],
            "task": spec.task,
            "base_model": spec.base_model,
            "method": spec.method.value,
            "quantization": spec.quantization,
            "runtime": runtime,
            "labels": ctx.get("labels", []),
            "backend": ctx.get("backend"),
            "compression": ctx.get("compression"),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out_dir / "eval_report.json").write_text(
            json.dumps(ctx["eval"], indent=2), encoding="utf-8"
        )
        logger.info("export: wrote artifact to %s", out_dir)
        return {"artifact_path": str(out_dir), "runtime": runtime, "metadata": metadata}
