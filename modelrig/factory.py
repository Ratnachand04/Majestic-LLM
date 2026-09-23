"""The ModelRig factory: run a BuildSpec through the compiled plane pipeline.

    validate -> compile -> [data -> training -> eval (gate) -> compression -> export]

The eval plane is the quality gate: if the held-out score does not clear
``spec.target_score`` the build is marked failed and no artifact is exported.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from majestic.logging_utils import get_logger
from modelrig.buildspec import BuildSpec, ensure_valid
from modelrig.compiler import Compiler, DefaultCompiler
from modelrig.planes import (
    CompressionPlane,
    DataPlane,
    EvalPlane,
    ExportPlane,
    Plane,
    TrainingPlane,
)
from modelrig.registry import FileSystemRegistry, Registry

logger = get_logger(__name__)

_PLANES: dict[str, type[Plane]] = {
    "data": DataPlane,
    "training": TrainingPlane,
    "eval": EvalPlane,
    "compression": CompressionPlane,
    "export": ExportPlane,
}


@dataclass
class BuildResult:
    build_id: str
    success: bool
    eval_report: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def _build_id(spec: BuildSpec) -> str:
    from modelrig.datasets import load_dataset

    payload = {"spec": asdict(spec), "data": load_dataset(spec.dataset), "format_version": 2}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    task = re.sub(r"[^a-zA-Z0-9_-]", "-", spec.task)[:40].strip("-") or "task"
    return f"{task}-{spec.method.value}-{digest[:24]}"


class Factory:
    """Compiles and runs builds, enforcing the eval gate and registering outputs."""

    def __init__(
        self,
        registry: Registry | None = None,
        compiler: Compiler | None = None,
        base_path: str | Path = "./registry",
    ) -> None:
        self.registry = registry or FileSystemRegistry(base_path)
        self.compiler = compiler or DefaultCompiler()

    def build(self, spec: BuildSpec) -> BuildResult:
        ensure_valid(spec)
        if spec.eval_metric != "accuracy":
            raise ValueError("factory gate currently supports eval_metric=accuracy only")
        if spec.is_heavy:
            from modelrig.training_hf import validate_hf_request

            validate_hf_request(spec)
        build_id = _build_id(spec)
        plane_names = self.compiler.compile(spec)
        logger.info("build %s: planes = %s", build_id, plane_names)

        base = getattr(self.registry, "base_path", Path("./registry"))
        out_dir = Path(base) / build_id
        ctx: dict[str, Any] = {"build_id": build_id, "out_dir": str(out_dir)}

        for name in plane_names:
            plane = _PLANES[name]()
            try:
                ctx.update(plane.run(spec, ctx))
            except ValueError as exc:
                return BuildResult(build_id=build_id, success=False,
                                   reason=f"{name} validation failed: {exc}")

            # Enforce the quality gate right after eval — never export a failure.
            if name == "eval" and not ctx.get("gate_passed", False):
                report = ctx["eval"]
                logger.warning("build %s failed gate: %s", build_id, report)
                return BuildResult(
                    build_id=build_id,
                    success=False,
                    eval_report=report,
                    reason=(
                        f"eval {report['metric']}={report['score']} "
                        f"< target {report['threshold']}"
                    ),
                )

        self.registry.put(build_id, ctx["artifact_path"], ctx["metadata"])
        logger.info("build %s: registered artifact", build_id)
        return BuildResult(
            build_id=build_id,
            success=True,
            eval_report=ctx["eval"],
            artifact_path=ctx["artifact_path"],
            metadata=ctx["metadata"],
        )
