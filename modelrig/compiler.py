"""Deterministic compiler: BuildSpec -> reproducible ordered plane list."""
from __future__ import annotations

from abc import ABC, abstractmethod

from modelrig.buildspec import BuildSpec


class Compiler(ABC):
    @abstractmethod
    def compile(self, spec: BuildSpec) -> list[str]:
        """Return an ordered list of build-plane names (the DAG)."""
        raise NotImplementedError


class DefaultCompiler(Compiler):
    """Standard five-stage pipeline; compression is dropped when disabled."""

    def compile(self, spec: BuildSpec) -> list[str]:
        planes = ["data", "training", "eval"]
        if spec.quantization != "none":
            planes.append("compression")
        planes.append("export")
        return planes
