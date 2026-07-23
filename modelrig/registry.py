"""Model & adapter registry: store build artifacts + metadata."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Registry(ABC):
    @abstractmethod
    def put(self, key: str, artifact: Any, metadata: dict) -> None: ...

    @abstractmethod
    def get(self, key: str) -> Any: ...

    @abstractmethod
    def list(self) -> list[str]: ...


class FileSystemRegistry(Registry):
    """Filesystem-backed registry.

    Artifacts are directories on disk (produced by the ExportPlane). The registry
    records an index of ``key -> {artifact_path, metadata}`` in ``index.json``
    under its base path.
    """

    def __init__(self, base_path: str | Path = "./registry") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_path / "index.json"
        self._index: dict[str, dict] = {}
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text(encoding="utf-8"))

    def _flush(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def put(self, key: str, artifact: Any, metadata: dict) -> None:
        """Register ``artifact`` (a path to the build directory) under ``key``."""
        self._index[key] = {"artifact_path": str(artifact), "metadata": metadata}
        self._flush()

    def get(self, key: str) -> Any:
        if key not in self._index:
            raise KeyError(f"no artifact registered under {key!r}")
        return self._index[key]["artifact_path"]

    def get_metadata(self, key: str) -> dict:
        if key not in self._index:
            raise KeyError(f"no artifact registered under {key!r}")
        return self._index[key]["metadata"]

    def list(self) -> list[str]:
        return list(self._index.keys())
