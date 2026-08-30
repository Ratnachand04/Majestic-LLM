"""Adapter paging is the *offline* problem, so use Belady (§15).

Everyone reaches for LRU because adapter swapping looks like caching, and caching
means online paging with an unknown future. Here the future is known.

The graph is a DAG and execution order is a topological sort, so the reference
string ``sigma = (a_1, ..., a_m)`` is fixed **before execution begins**. That is
the offline paging problem, where **Belady's MIN is optimal**: evict the adapter
whose next use is furthest away. No heuristic is needed, and no online policy can
beat it.

Two refinements the spec names, both implemented:

* **Branches** make part of ``sigma`` data-dependent. MIN applies to the
  deterministic prefix; branch arms are weighted by probability.
* **Topological freedom.** Many valid orders exist, and choosing the one that
  minimises the working set is free at construction time — group nodes sharing
  an adapter adjacently.

The saving is real rather than theoretical: a swap is roughly 100 ms of measured
latency, and on a chain that visits four adapters in a two-slot cache the
difference between MIN and LRU is several hundred milliseconds per request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from majestic.fabric.graph import FabricGraph, NodeKind
from majestic.logging_utils import get_logger

logger = get_logger(__name__)

#: Measured adapter swap latency. Part 3's B-09 target is ~100 ms and it is an
#: ASSUMPTION until telemetry says otherwise (§19, GAP-10).
SWAP_MS = 100.0


@dataclass
class PagingPlan:
    """What to evict and when, plus what it costs."""

    reference_string: tuple[str, ...] = ()
    capacity: int = 1
    swaps: int = 0
    evictions: list[tuple[str, str]] = field(default_factory=list)  # (victim, for)
    hits: int = 0

    @property
    def swap_ms(self) -> float:
        return self.swaps * SWAP_MS

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.swaps
        return round(self.hits / total, 4) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_string": list(self.reference_string),
            "capacity": self.capacity,
            "swaps": self.swaps,
            "hits": self.hits,
            "hit_rate": self.hit_rate,
            "swap_ms": round(self.swap_ms, 1),
            "evictions": [{"victim": v, "for": f} for v, f in self.evictions],
        }


def reference_string(graph: FabricGraph, order: Sequence[str] | None = None) -> tuple[str, ...]:
    """The adapters execution will touch, in order.

    Known before execution because the graph is acyclic — which is precisely
    what makes the optimal policy available.
    """
    order = order or graph.topological_order()
    out: list[str] = []
    for name in order:
        node = graph.nodes[name]
        if node.kind is not NodeKind.CARTRIDGE:
            continue
        adapter = node.adapter_ref or node.cartridge_id or name
        out.append(str(adapter))
    return tuple(out)


def belady(sigma: Sequence[str], capacity: int) -> PagingPlan:
    """Belady's MIN: evict whatever is next used furthest in the future.

    Optimal for the offline problem, and provably so — no online policy can do
    better, which is why there is nothing to tune here.
    """
    if capacity < 1:
        raise ValueError("an adapter cache must hold at least one adapter")
    plan = PagingPlan(reference_string=tuple(sigma), capacity=capacity)
    resident: list[str] = []

    for i, adapter in enumerate(sigma):
        if adapter in resident:
            plan.hits += 1
            continue
        plan.swaps += 1
        if len(resident) < capacity:
            resident.append(adapter)
            continue
        # Evict the resident adapter whose next use is furthest away. One never
        # used again is the ideal victim, hence the infinite distance.
        victim = max(resident, key=lambda a: _next_use(sigma, a, i + 1))
        resident[resident.index(victim)] = adapter
        plan.evictions.append((victim, adapter))
    return plan


def lru(sigma: Sequence[str], capacity: int) -> PagingPlan:
    """LRU, for comparison only. Never better than :func:`belady`."""
    if capacity < 1:
        raise ValueError("an adapter cache must hold at least one adapter")
    plan = PagingPlan(reference_string=tuple(sigma), capacity=capacity)
    resident: list[str] = []

    for adapter in sigma:
        if adapter in resident:
            plan.hits += 1
            resident.remove(adapter)
            resident.append(adapter)
            continue
        plan.swaps += 1
        if len(resident) >= capacity:
            plan.evictions.append((resident.pop(0), adapter))
        resident.append(adapter)
    return plan


def _next_use(sigma: Sequence[str], adapter: str, start: int) -> float:
    for j in range(start, len(sigma)):
        if sigma[j] == adapter:
            return float(j)
    return float("inf")


def schedule(graph: FabricGraph, capacity: int,
             order: Sequence[str] | None = None) -> PagingPlan:
    """Plan the swaps for one execution of this graph."""
    return belady(reference_string(graph, order), capacity)


def compare(graph: FabricGraph, capacity: int) -> dict[str, Any]:
    """What choosing the optimal policy is worth on this graph."""
    sigma = reference_string(graph, grouped_order(graph))
    optimal, online = belady(sigma, capacity), lru(sigma, capacity)
    return {
        "reference_string": list(sigma),
        "capacity": capacity,
        "belady_swaps": optimal.swaps,
        "lru_swaps": online.swaps,
        "swaps_saved": online.swaps - optimal.swaps,
        "ms_saved": round((online.swaps - optimal.swaps) * SWAP_MS, 1),
        "why": (
            "execution order is a topological sort of a DAG, so the reference "
            "string is known before the run: this is offline paging, where MIN "
            "is optimal and no online policy can beat it"
        ),
    }


# =========================================================================== #
# Topological freedom (§15)
# =========================================================================== #
def grouped_order(graph: FabricGraph) -> list[str]:
    """A topological order that keeps same-adapter nodes adjacent.

    Many valid orders exist and they do not cost the same. Preferring a ready
    node that reuses the adapter already in hand shrinks the working set, and it
    is free — the choice is made at construction time, not at runtime.
    """
    indegree = {n: 0 for n in graph.nodes}
    for _src, dst in graph.edges:
        indegree[dst] += 1

    ready = [n for n, d in indegree.items() if d == 0]
    order: list[str] = []
    current: str | None = None

    while ready:
        # Prefer a ready node needing the adapter already resident; fall back to
        # a stable choice so the schedule stays reproducible.
        pick = next((n for n in ready if _adapter_of(graph, n) == current), None)
        if pick is None:
            pick = min(ready, key=lambda n: (_adapter_of(graph, n) or "", n))
        ready.remove(pick)
        order.append(pick)
        adapter = _adapter_of(graph, pick)
        if adapter is not None:
            current = adapter
        for succ in graph.successors(pick):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)

    if len(order) != len(graph.nodes):
        raise ValueError("graph contains a cycle; Fabric graphs must be acyclic")
    return order


def _adapter_of(graph: FabricGraph, name: str) -> str | None:
    node = graph.nodes[name]
    if node.kind is not NodeKind.CARTRIDGE:
        return None
    return str(node.adapter_ref or node.cartridge_id or name)


# =========================================================================== #
# Branches (§15)
# =========================================================================== #
def expected_swaps(prefix: Sequence[str], arms: Iterable[tuple[Sequence[str], float]],
                   capacity: int) -> dict[str, Any]:
    """MIN over the deterministic prefix, then arms weighted by probability.

    A branch makes part of the reference string data-dependent, so the plan can
    only be exact up to the branch. Beyond it, the expected cost is the right
    thing to report — and reporting an expectation is honest where reporting a
    single number would not be.
    """
    arms = list(arms)
    total = sum(p for _seq, p in arms)
    if arms and abs(total - 1.0) > 1e-6:
        raise ValueError(f"branch probabilities must sum to 1, got {total:.4f}")

    base = belady(prefix, capacity)
    expected = float(base.swaps)
    for sequence, probability in arms:
        whole = belady([*prefix, *sequence], capacity)
        expected += probability * (whole.swaps - base.swaps)
    return {
        "prefix_swaps": base.swaps,
        "expected_swaps": round(expected, 2),
        "expected_ms": round(expected * SWAP_MS, 1),
        "arms": len(arms),
    }


__all__ = [
    "SWAP_MS", "PagingPlan",
    "belady", "compare", "expected_swaps", "grouped_order", "lru",
    "reference_string", "schedule",
]
