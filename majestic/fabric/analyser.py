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
from typing import Any, Sequence

from majestic.fabric.capacity import CapacityReport, analyse_capacity
from majestic.fabric.graph import FabricGraph, NodeKind
from majestic.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Everything the analyser proved, and everything it refuses to allow."""

    offline_closed: bool = True
    #: §11-§12 — how many bits reach each privileged sink, not merely whether.
    capacity: CapacityReport | None = None
    #: §13 — distinct bases the graph holds resident. The RAM trap.
    bases: tuple[str, ...] = ()
    #: §14 — the critical path, thermally derated.
    critical_path: tuple[str, ...] = ()
    critical_path_ms: float = 0.0
    peak_ram_mb: float = 0.0
    total_cost_usd: float = 0.0
    acyclic: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    network_nodes: list[str] = field(default_factory=list)
    injection_paths: list[list[str]] = field(default_factory=list)
    #: What to change. Binary taint says "unsafe"; capacity says which node and
    #: by how much, which is the difference between a verdict and advice.
    remedies: list[str] = field(default_factory=list)

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
    latency_budget_ms: float | None = None,
    capacity_bits: float | None = None,
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

    # 3b. §13 — the mixed-base trap. The RAM sum above already counts each base
    #     once, which is correct; what it does not do is say why a small graph
    #     can cost more than a large one. Three cartridges on three bases hold
    #     ~3.4 GB resident; twenty on one hold ~1.7 GB. Device-targeted graphs
    #     must be base-homogeneous, and that makes base selection a GRAPH-level
    #     decision rather than a per-cartridge one.
    result.bases = tuple(sorted(seen_bases))
    if len(seen_bases) > 1 and ram_budget_mb is not None:
        message = (
            f"graph spans {len(seen_bases)} distinct bases "
            f"({', '.join(sorted(seen_bases))}); each is held resident, so the base "
            "term dominates. Three cartridges on three bases cost more than twenty "
            "on one — device-targeted graphs should be base-homogeneous"
        )
        (result.violations if ram > ram_budget_mb else result.warnings).append(message)

    # 4. §7-§12 — information flow. Not "is there a path" but "how many bits
    #    reach a privileged sink", measured in one topological pass.
    result.capacity = analyse_capacity(graph, c_max=capacity_bits)
    for channel in result.capacity.violations:
        result.injection_paths.append(list(channel.path))
        cap = "unbounded" if channel.unbounded else f"{channel.capacity:.1f} bits"
        result.violations.append(
            "indirect prompt-injection surface: untrusted content from "
            f"{channel.source!r} can reach privileged node {channel.sink!r} via "
            + " -> ".join(channel.path)
            + f" (capacity {cap}, narrowed at {channel.bottleneck!r})"
        )
    result.remedies.extend(result.capacity.remedies())

    # 5. §14 — latency along the critical path, including adapter swaps.
    result.critical_path, result.critical_path_ms = critical_path(graph)
    if latency_budget_ms is not None and result.critical_path_ms > latency_budget_ms:
        result.violations.append(
            f"critical path {result.critical_path_ms:.0f}ms exceeds the "
            f"{latency_budget_ms:.0f}ms budget via "
            + " -> ".join(result.critical_path)
        )

    logger.info(
        "fabric: %s (offline_closed=%s, ram=%.0fMB, cost=$%.4f)",
        "OK" if result.ok else "VIOLATIONS",
        result.offline_closed, result.peak_ram_mb, result.total_cost_usd,
    )
    return result


# =========================================================================== #
# §6 — offline closure by graph rewriting, not by configuration
# =========================================================================== #
@dataclass
class OfflineRewrite:
    """The graph with its network reachable parts removed, and whether it works."""

    graph: FabricGraph
    removed_nodes: tuple[str, ...] = ()
    removed_edges: tuple[tuple[str, str], ...] = ()
    reachable_sinks: tuple[str, ...] = ()
    lost_sinks: tuple[str, ...] = ()

    @property
    def viable(self) -> bool:
        """Whether the residue still does the job."""
        return not self.lost_sinks and bool(self.graph.nodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "removed_nodes": list(self.removed_nodes),
            "removed_edges": [list(e) for e in self.removed_edges],
            "reachable_sinks": list(self.reachable_sinks),
            "lost_sinks": list(self.lost_sinks),
            "viable": self.viable,
        }


def rewrite_offline(graph: FabricGraph) -> OfflineRewrite:
    """``G_offline = G minus {net nodes} minus {edges into them}``, re-verified.

    **This is what "structurally disabled, not configured" means, made precise.**

    Proving offline closure by checking a threshold — ``escalation_threshold =
    inf`` — proves nothing. A threshold is a runtime value: mutable, restorable
    by a bug or a flag flip, and exactly the failure where a customer who bought
    offline discovers a network call at 2 a.m.

    So offline mode *deletes* the escalation edges and the proof is over the
    rewritten graph. There is nothing to misconfigure because there is nothing
    there. If deleting them disconnects the graph, the task genuinely cannot be
    done offline, and the violation names the sink that became unreachable —
    which is the actionable message rather than a bare refusal.
    """
    residue = FabricGraph(
        name=f"{graph.name}:offline",
        graph_version=graph.graph_version,
        owner_id=graph.owner_id,
        max_taint_capacity_bits=graph.max_taint_capacity_bits,
    )
    # A node goes if it needs the network at all — unconditionally, or on a
    # branch. A conditional escalation is removed with its branch, which is the
    # whole point: the condition can no longer be flipped back on.
    removed = {n for n, node in graph.nodes.items() if node.requires_network}
    for name, node in graph.nodes.items():
        if name not in removed:
            residue.nodes[name] = node
    kept_edges = [(a, b) for a, b in graph.edges if a not in removed and b not in removed]
    residue.edges = kept_edges

    # Reachability must be measured from the ORIGINAL inputs. A sink that was
    # fed only by the deleted node becomes a root of the residue, and a naive
    # check would call it reachable when in fact it lost its only input.
    entry = [n for n in graph.roots() if n in residue.nodes]
    live = _reachable_from(residue, entry)
    original_sinks = _sinks(graph)
    reachable = [s for s in original_sinks if s in live]
    lost = [s for s in original_sinks if s not in reachable]

    return OfflineRewrite(
        graph=residue,
        removed_nodes=tuple(sorted(removed)),
        removed_edges=tuple(e for e in graph.edges if e not in kept_edges),
        reachable_sinks=tuple(reachable),
        lost_sinks=tuple(lost),
    )


def _sinks(graph: FabricGraph) -> list[str]:
    sources = {a for a, _ in graph.edges}
    return [n for n in graph.nodes if n not in sources]


def _reachable_from(graph: FabricGraph, entry: Sequence[str]) -> set[str]:
    """Everything still fed from the original entry points."""
    seen: set[str] = set()
    stack = [n for n in entry if n in graph.nodes]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.successors(node))
    return seen


def prove_offline(graph: FabricGraph) -> tuple[bool, list[str]]:
    """§6's predicate: unconditional AND conditional network use, both gone.

    Returns the verdict and the reasons, so a refusal names the node rather than
    asserting that the graph is unsuitable.
    """
    reasons: list[str] = []
    for name, node in graph.nodes.items():
        if node.net_unconditional:
            reasons.append(f"{name!r} needs the network on every path")
        elif node.requires_network:
            reasons.append(
                f"{name!r} escalates to the network when {node.net_condition!r}; a "
                "condition is a runtime value, so offline mode deletes the branch "
                "rather than trusting a threshold"
            )
    if not reasons:
        return True, []

    rewrite = rewrite_offline(graph)
    if rewrite.viable:
        reasons.append(
            "the graph still reaches every sink with those nodes removed, so it "
            "can be made offline by rewriting rather than refused"
        )
        return True, reasons
    reasons.append(
        "removing them disconnects "
        + ", ".join(repr(s) for s in rewrite.lost_sinks)
        + ": this task genuinely cannot be done offline"
    )
    return False, reasons


# =========================================================================== #
# §14 — latency along the critical path
# =========================================================================== #
def critical_path(graph: FabricGraph, swap_ms: float | None = None) -> tuple[tuple[str, ...], float]:
    """The slowest path through the graph, in one topological pass.

    Latency compounds: a downstream node's input includes its predecessor's
    output, so a chatty upstream cartridge inflates everything after it.
    Constraining an intermediate output schema is therefore a latency
    optimisation as well as a security one (§12) — the same lever, two benefits.
    """
    from majestic.fabric.schedule import SWAP_MS

    swap = SWAP_MS if swap_ms is None else swap_ms
    best: dict[str, float] = {}
    came_from: dict[str, str] = {}
    last_adapter: dict[str, str | None] = {}

    for name in graph.topological_order():
        node = graph.nodes[name]
        own = node.p95_ms or node.p50_ms
        adapter = str(node.adapter_ref) if node.adapter_ref else None

        candidates: list[tuple[float, str]] = []
        for pred in graph.predecessors(name):
            cost = best.get(pred, 0.0)
            # A swap is only paid when the adapter actually changes.
            if adapter is not None and last_adapter.get(pred) not in (None, adapter):
                cost += swap
            candidates.append((cost, pred))
        if candidates:
            incoming, pred = max(candidates)
            came_from[name] = pred
        else:
            incoming = 0.0
        best[name] = incoming + own
        last_adapter[name] = adapter or last_adapter.get(came_from.get(name, ""), None)

    if not best:
        return (), 0.0
    end = max(best, key=lambda n: best[n])
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    return tuple(reversed(path)), round(best[end], 1)


# =========================================================================== #
# §17 — the misroute penalty
# =========================================================================== #
def system_accuracy(router_accuracy: float, expert_accuracy: float,
                    fallback_accuracy: float = 0.0) -> float:
    """``A_sys = rho*A_correct + (1-rho)*A_fallback``.

    The earlier claim — a 91% router over 96% experts gives ~87% — silently
    assumed ``A_fallback = 0``, i.e. that a misroute always produces a wrong
    answer. It usually does not: a wrong specialist is often partly useful, and
    one that *detects* it is out of scope can abstain and be rerouted.
    """
    for value in (router_accuracy, expert_accuracy, fallback_accuracy):
        if not 0.0 <= value <= 1.0:
            raise ValueError("accuracies must lie in [0, 1]")
    return router_accuracy * expert_accuracy + (1 - router_accuracy) * fallback_accuracy


def routing_investment(router_accuracy: float = 0.91, expert_accuracy: float = 0.96,
                       improved_router: float = 0.95,
                       graceful_fallback: float = 0.70) -> dict[str, Any]:
    """Where to spend: a better router, or graceful misroutes?

    **Seven points of system accuracy live entirely in what happens on a
    misroute**, and the cheapest way to buy them is not a better router but a
    cartridge that knows when a question is out of scope. Improving the router
    helps only at the margin it improves; abstention helps at *every* router
    accuracy, which is why it is the better first investment.
    """
    base = system_accuracy(router_accuracy, expert_accuracy, 0.0)
    better_router = system_accuracy(improved_router, expert_accuracy, 0.0)
    graceful = system_accuracy(router_accuracy, expert_accuracy, graceful_fallback)
    return {
        "baseline": round(base, 4),
        "better_router": round(better_router, 4),
        "graceful_misroutes": round(graceful, 4),
        "points_from_router": round((better_router - base) * 100, 1),
        "points_from_abstention": round((graceful - base) * 100, 1),
        "verdict": (
            "invest in abstention before the router: it buys more, and it helps "
            "at every router accuracy rather than only at the margin improved"
        ),
    }
