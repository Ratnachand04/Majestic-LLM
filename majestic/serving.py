"""Runtime topology — one architecture, two deployment shapes (A-04, B-09).

The same cartridge format serves a thousand tenants on one GPU or twenty
specialists on one phone. Both shapes rest on the property from A-03: because a
LoRA update is *added*, it can be kept separate so that many adapters share one
resident base.

**Server tier.** One base resident in VRAM, an adapter pool paged in and out on
demand, and requests using different adapters batched together. What the papers
do not address — and what Majestic adds — is a spec-hash-keyed adapter cache with
tier-driven eviction and cross-tenant isolation.

**What breaks it.** Customers on DIFFERENT base models cannot share a GPU. Every
extra base in the catalogue fragments the pool, which is the concrete reason the
catalogue stays narrow (A-04), in tension with the blast-radius argument for base
diversity (B-08). :meth:`AdapterPool.fragmentation_report` makes that trade
visible rather than leaving it as prose.

**Device tier.** The B-09 budget table is computed rather than asserted, so a
change to context length or adapter count immediately shows up as a different
total.

GAP-10: the device numbers are a DESIGN TARGET, not a measurement. Multi-adapter
serving is characterised on datacentre GPUs; the claim that twenty specialists
live in ~1.5 GB on a mid-range phone with ~100 ms swaps is an assumption. MELTing
Point (2403.12844) shows the binding mobile constraints are energy and thermal
throttling, not raw speed. Every value here carries ``measured=False``.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

#: Typical LoRA adapter footprint at rank 16-32 (A-03/B-08).
DEFAULT_ADAPTER_MB = 30.0
#: Assumed adapter swap latency on device. UNMEASURED — see GAP-10.
ASSUMED_SWAP_MS = 100.0


# --------------------------------------------------------------------------- #
@dataclass
class AdapterHandle:
    """One cartridge's adapter as the serving layer sees it."""

    cartridge_id: str
    base_ref: str
    size_mb: float = DEFAULT_ADAPTER_MB
    tier: str = "standard"     # free | standard | premium — drives eviction
    spec_hash: str = ""


class AdapterPool:
    """LRU adapter pool over a shared resident base, with tier-driven eviction.

    Eviction is tier-aware because a free-tier tenant's adapter should be paged
    out before a premium tenant's: within a tier it is least-recently-used, but a
    higher tier is never evicted while a lower-tier candidate exists.
    """

    _TIER_RANK = {"free": 0, "standard": 1, "premium": 2}

    def __init__(self, capacity_mb: float = 8192.0) -> None:
        self.capacity_mb = capacity_mb
        self._resident: OrderedDict[str, AdapterHandle] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def resident_mb(self) -> float:
        return sum(a.size_mb for a in self._resident.values())

    @property
    def resident_ids(self) -> list[str]:
        return list(self._resident)

    def _evict_for(self, incoming: AdapterHandle) -> None:
        while self._resident and self.resident_mb + incoming.size_mb > self.capacity_mb:
            victim_id = min(
                self._resident,
                key=lambda cid: (
                    self._TIER_RANK.get(self._resident[cid].tier, 1),
                    list(self._resident).index(cid),
                ),
            )
            self._resident.pop(victim_id)
            self.evictions += 1
            logger.debug("adapter pool: evicted %s", victim_id[:8])

    def acquire(self, adapter: AdapterHandle) -> bool:
        """Page an adapter in. Returns True on a cache hit (no paging needed)."""
        if adapter.cartridge_id in self._resident:
            self._resident.move_to_end(adapter.cartridge_id)
            self.hits += 1
            return True
        self.misses += 1
        self._evict_for(adapter)
        self._resident[adapter.cartridge_id] = adapter
        return False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def fragmentation_report(self) -> dict[str, Any]:
        """How badly base diversity is fragmenting this pool (A-04).

        Customers on different bases cannot share a GPU, so a pool holding N
        distinct bases needs N resident copies. This is the number that makes the
        narrow-catalogue argument concrete.
        """
        bases: dict[str, int] = {}
        for adapter in self._resident.values():
            bases[adapter.base_ref] = bases.get(adapter.base_ref, 0) + 1
        return {
            "distinct_bases": len(bases),
            "adapters": len(self._resident),
            "adapters_per_base": bases,
            "shareable": len(bases) <= 1,
            "note": (
                "one resident base serves every adapter"
                if len(bases) <= 1
                else f"{len(bases)} bases cannot share a GPU; each needs its own copy"
            ),
        }


# --------------------------------------------------------------------------- #
@dataclass
class ServerTopology:
    """Server tier: shared base + adapter pool + batched heterogeneous kernel."""

    base_ref: str
    vram_gb: float = 80.0            # a single A100
    base_weights_mb: float = 16000.0  # fp16 base resident in VRAM
    kv_cache_mb_per_seq: float = 120.0
    pool: AdapterPool = field(default_factory=lambda: AdapterPool(8192.0))

    @property
    def vram_mb(self) -> float:
        return self.vram_gb * 1024.0

    def max_tenants(self, adapter_mb: float = DEFAULT_ADAPTER_MB) -> int:
        """How many adapters fit once the base and KV cache are accounted for."""
        available = self.vram_mb - self.base_weights_mb - self.kv_cache_mb_per_seq * 8
        return max(int(available // max(adapter_mb, 1e-6)), 0)

    def batch(self, requests: Iterable[AdapterHandle]) -> dict[str, Any]:
        """Admit a heterogeneous batch: different adapters in one execution.

        Requests are grouped by base because only same-base requests can share
        the resident weights — the fragmentation constraint made operational.
        """
        by_base: dict[str, list[AdapterHandle]] = {}
        for req in requests:
            by_base.setdefault(req.base_ref, []).append(req)
            self.pool.acquire(req)

        servable = by_base.get(self.base_ref, [])
        rejected = [r for b, rs in by_base.items() if b != self.base_ref for r in rs]
        return {
            "batched": len(servable),
            "distinct_adapters": len({r.cartridge_id for r in servable}),
            "rejected_wrong_base": len(rejected),
            "pool_hit_rate": self.pool.hit_rate,
            "note": (
                "requests using different adapters execute in one batch"
                if servable else "nothing servable on this base"
            ),
        }

    def unit_economics(self, adapter_mb: float = DEFAULT_ADAPTER_MB) -> dict[str, Any]:
        """Cost per request falls with tenancy — the hosted economics (A-04)."""
        tenants = self.max_tenants(adapter_mb)
        return {
            "tenants_per_gpu": tenants,
            "vram_mb": self.vram_mb,
            "base_weights_mb": self.base_weights_mb,
            "adapter_pool_mb": round(tenants * adapter_mb, 1),
            "measured": False,
        }


# --------------------------------------------------------------------------- #
@dataclass
class RamBudgetLine:
    label: str
    gb: float
    note: str = ""


@dataclass
class DeviceBudget:
    """The B-09 device RAM budget, computed rather than asserted."""

    device: str
    total_gb: float
    lines: list[RamBudgetLine] = field(default_factory=list)
    measured: bool = False

    @property
    def committed_gb(self) -> float:
        return round(sum(line.gb for line in self.lines), 2)

    @property
    def fits(self) -> bool:
        return self.committed_gb <= self.total_gb

    @property
    def headroom_gb(self) -> float:
        return round(self.total_gb - self.committed_gb, 2)

    def as_table(self) -> list[tuple[str, float]]:
        return [(line.label, round(line.gb, 2)) for line in self.lines] + [
            ("Total committed", self.committed_gb)
        ]


def device_budget(
    device_name: str = "android_tablet_4gb",
    total_gb: float = 4.0,
    base_params_b: float = 1.7,
    bit_width: str = "int4",
    context_length: int = 2048,
    n_adapters: int = 20,
    adapter_mb: float = DEFAULT_ADAPTER_MB,
    kv_bytes_per_token: float = 114688.0,
    embedder_gb: float = 0.09,
    runtime_gb: float = 0.06,
    os_headroom_gb: float = 1.40,
) -> DeviceBudget:
    """Compute the on-device RAM budget line by line (B-09).

    Twenty specialists fit in ~0.6 GB because they SHARE the one resident base —
    that is the whole point of keeping adapters separate rather than merging
    them. Change the context length or the adapter count and the total moves,
    which is exactly what a budget should do.
    """
    from modelrig.catalogue import BYTES_PER_PARAM

    bytes_pp = BYTES_PER_PARAM.get(bit_width, 2.0)
    weights_gb = base_params_b * 1e9 * bytes_pp / 1e9
    kv_gb = kv_bytes_per_token * context_length / 1e9
    adapters_gb = n_adapters * adapter_mb / 1024.0

    lines = [
        RamBudgetLine(f"Base model, {base_params_b}B at {bit_width}", round(weights_gb, 2),
                      "resident once, shared by every adapter"),
        RamBudgetLine("KV cache at planned context", round(kv_gb, 2),
                      f"grows linearly with context ({context_length} tokens)"),
        RamBudgetLine("Embedder for the local index", embedder_gb),
        RamBudgetLine("Grammar state and runtime", runtime_gb),
        RamBudgetLine(f"{n_adapters} adapters, resident", round(adapters_gb, 2),
                      "they share the one base, so this is deltas only"),
        RamBudgetLine("Application and OS headroom", os_headroom_gb),
    ]
    budget = DeviceBudget(device=device_name, total_gb=total_gb, lines=lines, measured=False)
    logger.info(
        "device budget %s: %.2f/%.2f GB committed (%s)",
        device_name, budget.committed_gb, total_gb,
        "fits" if budget.fits else "OVER BUDGET",
    )
    return budget


def swap_latency_ms(n_adapters: int = 1, per_swap_ms: float = ASSUMED_SWAP_MS) -> dict[str, Any]:
    """Adapter-swap cost. UNMEASURED — this is the central GAP-10 promise.

    Fast enough that the user experiences a single assistant rather than a model
    chooser IS the product claim. It has not been validated on real hardware.
    """
    return {
        "swaps": n_adapters,
        "assumed_ms_per_swap": per_swap_ms,
        "total_ms": round(n_adapters * per_swap_ms, 1),
        "measured": False,
        "caveat": (
            "assumption, not a measurement: energy and thermal throttling are the "
            "binding mobile constraints (MELTing Point 2403.12844)"
        ),
    }


def plan_device_deployment(
    base_params_b: float,
    total_gb: float,
    context_length: int = 2048,
    n_adapters: int = 20,
    kv_bytes_per_token: float = 114688.0,
) -> tuple[DeviceBudget, Optional[str]]:
    """Budget a deployment and report the first line that breaks it."""
    budget = device_budget(
        total_gb=total_gb, base_params_b=base_params_b,
        context_length=context_length, n_adapters=n_adapters,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    if budget.fits:
        return budget, None
    biggest = max(budget.lines, key=lambda line: line.gb)
    return budget, (
        f"over budget by {-budget.headroom_gb:.2f} GB; the largest line is "
        f"{biggest.label!r} at {biggest.gb:.2f} GB"
    )
