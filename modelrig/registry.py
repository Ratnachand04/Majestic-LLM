"""Model & adapter registry — content-addressed, deduplicated, lineage-aware (B-08).

The base is stored ONCE and every cartridge is a delta pointing at it. That
single mechanism is why 500 customer models occupy ~16 GB instead of 550 GB.

Deduplication is not only for weights: datasets, eval suites, grammars and
quantised artefacts are all hash-keyed, and an identical spec hash skips the
build entirely and serves the cached cartridge at zero marginal cost. Cache-hit
rate should be tracked as closely as revenue, so
:meth:`CartridgeRegistry.stats` reports it.

Lineage is free once storage is content-addressed: when a defect is found in a
base, the registry can name every cartridge derived from it and rebuild the
affected fleet. Bommasani et al. (2108.07258) call this homogenisation risk;
provenance turns it from an unbounded incident into a query.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from majestic.logging_utils import get_logger
from modelrig.cartridge import Cartridge

logger = get_logger(__name__)


class Registry(ABC):
    @abstractmethod
    def put(self, key: str, artifact: Any, metadata: dict) -> None: ...

    @abstractmethod
    def get(self, key: str) -> Any: ...

    @abstractmethod
    def list(self) -> list[str]: ...


class FileSystemRegistry(Registry):
    """Filesystem-backed registry of build artefacts.

    Artefacts are directories on disk (produced by the ExportPlane). The registry
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


# --------------------------------------------------------------------------- #
@dataclass
class RegistryStats:
    """Storage economics, reported the way B-08 frames them."""

    cartridges: int = 0
    distinct_bases: int = 0
    adapter_bytes: int = 0
    base_bytes: int = 0
    naive_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def stored_bytes(self) -> int:
        return self.base_bytes + self.adapter_bytes

    @property
    def dedup_ratio(self) -> float:
        """How many times smaller the registry is than one-full-model-each."""
        return round(self.naive_bytes / self.stored_bytes, 2) if self.stored_bytes else 1.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 4) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cartridges": self.cartridges,
            "distinct_bases": self.distinct_bases,
            "stored_bytes": self.stored_bytes,
            "naive_bytes": self.naive_bytes,
            "dedup_ratio": self.dedup_ratio,
            "cache_hit_rate": self.cache_hit_rate,
        }


class CartridgeRegistry:
    """Content-addressed cartridge store with dedup, lineage and a spec cache."""

    def __init__(self, base_path: str | Path = "./registry") -> None:
        self.base_path = Path(base_path)
        self.cartridge_dir = self.base_path / "cartridges"
        self.cartridge_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_path / "cartridge_index.json"
        self._index: dict[str, dict] = {}
        #: base_ref -> assumed on-disk size in bytes (stored once for the fleet)
        self._base_sizes: dict[str, int] = {}
        self._hits = 0
        self._misses = 0
        if self._index_path.exists():
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._index = raw.get("cartridges", {})
            self._base_sizes = raw.get("base_sizes", {})

    def _flush(self) -> None:
        self._index_path.write_text(
            json.dumps({"cartridges": self._index, "base_sizes": self._base_sizes}, indent=2),
            encoding="utf-8",
        )

    # -- admission --------------------------------------------------------- #
    def admit(self, cartridge: Cartridge, base_bytes: int = 1_100_000_000) -> str:
        """Admit a CERTIFIED cartridge. Refuses anything uncertified (Gate 3)."""
        if not cartridge.certified:
            raise ValueError(
                "cartridge is not certified: a model card, eval certificate and a "
                "permitting licence chain must be attached before admission (B-03 Gate 3)"
            )
        cid = cartridge.id
        path = self.cartridge_dir / f"{cid}.json"
        cartridge.save(path)
        # The base is stored ONCE, however many cartridges point at it.
        self._base_sizes.setdefault(cartridge.base_ref, base_bytes)
        self._index[cid] = {
            "path": str(path),
            "base_ref": cartridge.base_ref,
            "spec_hash": cartridge.spec_hash,
            "plan_hash": cartridge.plan_hash,
            "adapter_bytes": cartridge.adapter_bytes,
            "version": cartridge.version,
        }
        self._flush()
        logger.info("registry: admitted cartridge %s on base %s", cid[:8], cartridge.base_ref)
        return cid

    def get(self, cartridge_id: str) -> Cartridge:
        if cartridge_id not in self._index:
            raise KeyError(f"no cartridge {cartridge_id!r}")
        return Cartridge.load(self._index[cartridge_id]["path"])

    def list(self) -> list[str]:
        return list(self._index)

    # -- the spec cache ----------------------------------------------------- #
    def by_spec_hash(self, spec_hash: str) -> Cartridge | None:
        """Serve a previously built cartridge for an identical spec, or ``None``.

        An identical spec hash skips the build entirely at zero marginal cost.
        """
        for cid, row in self._index.items():
            if row.get("spec_hash") == spec_hash:
                self._hits += 1
                logger.info("registry: cache HIT for spec %s -> %s", spec_hash[:8], cid[:8])
                return self.get(cid)
        self._misses += 1
        return None

    # -- lineage ------------------------------------------------------------ #
    def derived_from(self, base_ref: str) -> list[str]:
        """Every cartridge derived from a base — the fleet-recall query (B-08)."""
        return [cid for cid, row in self._index.items() if row.get("base_ref") == base_ref]

    def lineage(self, cartridge_id: str) -> dict[str, Any]:
        """Full provenance chain for one cartridge."""
        row = self._index.get(cartridge_id)
        if row is None:
            raise KeyError(f"no cartridge {cartridge_id!r}")
        cart = self.get(cartridge_id)
        return {
            "cartridge_id": cartridge_id,
            "base_ref": row["base_ref"],
            "spec_hash": row["spec_hash"],
            "plan_hash": row["plan_hash"],
            "licence_chain": cart.licence_chain,
            "siblings_on_base": self.derived_from(row["base_ref"]),
        }

    # -- economics ---------------------------------------------------------- #
    def stats(self) -> RegistryStats:
        adapter_bytes = sum(int(r.get("adapter_bytes", 0)) for r in self._index.values())
        base_bytes = sum(self._base_sizes.values())
        naive = sum(
            self._base_sizes.get(r["base_ref"], 0) + int(r.get("adapter_bytes", 0))
            for r in self._index.values()
        )
        return RegistryStats(
            cartridges=len(self._index),
            distinct_bases=len(self._base_sizes),
            adapter_bytes=adapter_bytes,
            base_bytes=base_bytes,
            naive_bytes=naive,
            cache_hits=self._hits,
            cache_misses=self._misses,
        )
