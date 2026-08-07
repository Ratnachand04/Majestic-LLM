"""Static analysis over a Fabric graph (B-10, GAP-04).

No agent framework can statically prove that a graph runs without a network.
Agent frameworks are dynamic and untyped by design; this capability is unique to
a typed DAG and is a genuine differentiator (C-01).

Four properties are proven before the graph ever runs:

1. **Offline closure** — every node is locally resolvable, zero API calls.
   Customers who bought an offline system discovering it silently needs the
   network is the single most damaging possible product failure.
2. **RAM bound** — the peak resident footprint fits the device budget.
3. **Cost bound** — the per-request cost fits the customer's ceiling.
4. **Injection reachability** — no path lets untrusted retrieved content reach a
   privileged tool (2302.12173). Every scraped page is untrusted INPUT, not data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from majestic.fabric.graph import FabricGraph, NodeKind
from majestic.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Everything the analyser proved, and everything it refuses to allow."""

    offline_closed: bool = True
    peak_ram_mb: float = 0.0
    total_cost_usd: float = 0.0
    acyclic: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    network_nodes: list[str] = field(default_factory=list)
    injection_paths: list[list[str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def suggest_split(self) -> str | None:
        """The remedy B-10 offers when closure breaks: split core from tail."""
        if self.offline_closed or not self.network_nodes:
            return None
        return (
            "split into an offline core and an online tail; the network-requiring "
            f"nodes are: {', '.join(self.network_nodes)}"
        )


def analyse(
    graph: FabricGraph,
    *,
    offline_required: bool = False,
    ram_budget_mb: float | None = None,
    cost_ceiling_usd: float | None = None,
) -> AnalysisResult:
    """Statically verify a Fabric graph.

    ``offline_required`` turns a broken closure from a warning into a violation:
    a customer who bought offline gets offline, and escalation is architecturally
    disabled rather than merely discouraged.
    """
    result = AnalysisResult()

    # 0. the graph must be a DAG at all
    try:
        order = graph.topological_order()
    except ValueError as exc:
        result.acyclic = False
        result.violations.append(str(exc))
        return result

    # 1. offline closure
    result.network_nodes = [n for n in order if graph.nodes[n].requires_network]
    result.offline_closed = not result.network_nodes
    if result.network_nodes:
        message = (
            "offline closure BROKEN at "
            + ", ".join(f"{n!r}" for n in result.network_nodes)
        )
        if offline_required:
            result.violations.append(message)
        else:
            result.warnings.append(message)

    # 2. RAM bound. Cartridges sharing one resident base is the whole point of
    #    B-09, so the base is counted once and adapters accumulate.
    seen_bases: set[str] = set()
    ram = 0.0
    for name in order:
        node = graph.nodes[name]
        if node.kind is NodeKind.CARTRIDGE:
            base = str(node.metadata.get("base_ref", node.cartridge_id or name))
            if base not in seen_bases:
                seen_bases.add(base)
                ram += float(node.metadata.get("base_ram_mb", 0.0))
        ram += node.ram_mb
    result.peak_ram_mb = round(ram, 1)
    if ram_budget_mb is not None and ram > ram_budget_mb:
        result.violations.append(
            f"graph needs {ram:.0f}MB but the device budget is {ram_budget_mb:.0f}MB"
        )

    # 3. cost bound
    result.total_cost_usd = round(sum(graph.nodes[n].cost_usd for n in order), 4)
    if cost_ceiling_usd is not None and result.total_cost_usd > cost_ceiling_usd:
        result.violations.append(
            f"graph costs ${result.total_cost_usd:.4f} per request, above the "
            f"ceiling ${cost_ceiling_usd:.4f}"
        )

    # 4. indirect prompt-injection reachability
    untrusted = [n for n in order if graph.nodes[n].produces_untrusted]
    privileged = [n for n in order if graph.nodes[n].privileged]
    for u in untrusted:
        for p in privileged:
            for path in graph.paths_between(u, p):
                result.injection_paths.append(path)
                result.violations.append(
                    "indirect prompt-injection surface: untrusted content from "
                    f"{u!r} can reach privileged node {p!r} via "
                    + " -> ".join(path)
                )

    logger.info(
        "fabric: %s (offline_closed=%s, ram=%.0fMB, cost=$%.4f)",
        "OK" if result.ok else "VIOLATIONS",
        result.offline_closed, result.peak_ram_mb, result.total_cost_usd,
    )
    return result
