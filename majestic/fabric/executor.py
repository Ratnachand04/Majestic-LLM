"""The Fabric runtime — a typed DAG that actually executes (B-10).

    "Typed DAG runtime. Nodes are cartridges, tools and control flow.
     Statically verified for offline closure, RAM and cost.
     Runs on server, on device, or hybrid."

Analysis alone is not a runtime, so this module runs the graph — but only after
the analyser has cleared it. That ordering is the whole design: a graph is proven
before it executes, and the proof is a precondition rather than a report.

Two enforcement properties carry over from the static pass into execution:

**Offline closure is enforced at runtime too.** A node that requires the network
inside an offline deployment does not "fail at the network call" — it is refused
before its handler is invoked, because a customer who bought offline gets offline.

**The trust taint propagates.** A node consuming untrusted content produces
untrusted output, and a privileged node refuses tainted input. The static
analyser proves no such path exists; the executor makes it unreachable even if
the graph is constructed dynamically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from majestic.fabric.analyser import analyse
from majestic.fabric.graph import FabricGraph
from majestic.logging_utils import get_logger

logger = get_logger(__name__)

Handler = Callable[[Any, dict[str, Any]], Any]


class OfflineViolation(RuntimeError):
    """A network-requiring node was reached inside an offline deployment."""


class TaintViolation(RuntimeError):
    """Untrusted content reached a privileged node."""


@dataclass
class NodeTrace:
    """What one node did."""

    name: str
    kind: str
    output: Any = None
    tainted: bool = False
    skipped: bool = False
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class ExecutionResult:
    """The outcome of running a graph."""

    output: Any = None
    trace: list[NodeTrace] = field(default_factory=list)
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    cost_usd: float = 0.0

    def node(self, name: str) -> Optional[NodeTrace]:
        return next((t for t in self.trace if t.name == name), None)


class FabricRuntime:
    """Executes a verified Fabric graph on server, on device, or hybrid."""

    def __init__(
        self,
        graph: FabricGraph,
        handlers: dict[str, Handler],
        *,
        offline: bool = False,
        ram_budget_mb: float | None = None,
        cost_ceiling_usd: float | None = None,
        verify: bool = True,
    ) -> None:
        self.graph = graph
        self.handlers = handlers
        self.offline = offline
        self.ram_budget_mb = ram_budget_mb
        self.cost_ceiling_usd = cost_ceiling_usd
        self.verify = verify

    # ------------------------------------------------------------------ #
    def preflight(self):
        """Static verification. Executing an unverified graph is not allowed."""
        return analyse(
            self.graph,
            offline_required=self.offline,
            ram_budget_mb=self.ram_budget_mb,
            cost_ceiling_usd=self.cost_ceiling_usd,
        )

    def _static_taint(self) -> dict[str, bool]:
        return _static_taint_for(self.graph)

    def run(self, payload: Any = None) -> ExecutionResult:
        """Verify, then execute in topological order."""
        started = time.perf_counter()
        result = ExecutionResult()

        if self.verify:
            analysis = self.preflight()
            if not analysis.ok:
                result.ok = False
                result.violations = list(analysis.violations)
                logger.warning("fabric: refusing to run an unverified graph: %s",
                               analysis.violations)
                return result

        try:
            order = self.graph.topological_order()
        except ValueError as exc:
            return ExecutionResult(ok=False, violations=[str(exc)])

        values: dict[str, Any] = {}
        tainted: dict[str, bool] = {}
        last_output: Any = payload

        # §18 — the runtime recomputes taint and asserts it matches the static
        # proof. A cheap dynamic check whose only job is to catch drift between
        # the analyser and the executor: two implementations of the same rule
        # that are edited by different people at different times, and where a
        # divergence would otherwise show up as a security failure in the field
        # rather than as a test failure here.
        predicted = self._static_taint()

        for name in order:
            node = self.graph.nodes[name]
            node_started = time.perf_counter()

            # Gather inputs from predecessors; roots take the request payload.
            preds = self.graph.predecessors(name)
            if preds:
                inputs = [values[p] for p in preds if p in values]
                node_input = inputs[0] if len(inputs) == 1 else inputs
                incoming_taint = any(tainted.get(p, False) for p in preds)
            else:
                node_input, incoming_taint = payload, False

            if name in predicted and predicted[name] != incoming_taint:
                # Never silently prefer one over the other. The static proof is
                # what the graph was admitted on, so a mismatch means the
                # admission was made on a different rule than the one running.
                raise TaintViolation(
                    f"taint drift at {name!r}: the static analysis proved "
                    f"incoming taint {predicted[name]} and the runtime computed "
                    f"{incoming_taint}. The analyser and the executor disagree "
                    "about the same graph, so neither verdict can be trusted"
                )

            # Offline closure, enforced rather than merely reported.
            if self.offline and node.requires_network:
                trace = NodeTrace(name, node.kind.value, skipped=True,
                                  error="offline: network-requiring node refused")
                result.trace.append(trace)
                result.ok = False
                result.violations.append(
                    f"node {name!r} requires the network in an offline deployment"
                )
                logger.warning("fabric: refused %s (offline hard-lock)", name)
                break

            # The taint rule: untrusted content must not reach privilege.
            if node.privileged and incoming_taint:
                result.ok = False
                result.violations.append(
                    f"untrusted content reached privileged node {name!r}"
                )
                result.trace.append(
                    NodeTrace(name, node.kind.value, skipped=True,
                              error="refused tainted input")
                )
                logger.warning("fabric: refused %s (tainted input)", name)
                break

            handler = self.handlers.get(name)
            if handler is None:
                result.trace.append(
                    NodeTrace(name, node.kind.value, skipped=True,
                              error="no handler registered")
                )
                values[name] = node_input
                tainted[name] = incoming_taint
                continue

            try:
                output = handler(node_input, {"node": node, "graph": self.graph})
            except Exception as exc:  # noqa: BLE001 - surface, do not crash the graph
                result.ok = False
                result.violations.append(f"node {name!r} raised: {exc}")
                result.trace.append(
                    NodeTrace(name, node.kind.value, error=str(exc),
                              duration_ms=_ms_since(node_started))
                )
                logger.warning("fabric: node %s raised %s", name, exc)
                break

            node_taint = incoming_taint or node.produces_untrusted
            values[name] = output
            tainted[name] = node_taint
            last_output = output
            result.cost_usd += node.cost_usd
            result.trace.append(
                NodeTrace(name, node.kind.value, output=output, tainted=node_taint,
                          duration_ms=_ms_since(node_started))
            )

        result.output = last_output if result.ok else None
        result.total_ms = _ms_since(started)
        result.cost_usd = round(result.cost_usd, 6)
        logger.info(
            "fabric: ran %d/%d nodes in %.1fms (%s)",
            len([t for t in result.trace if not t.skipped]), len(order),
            result.total_ms, "ok" if result.ok else "violations",
        )
        return result


def _static_taint_for(graph: FabricGraph) -> dict[str, bool]:
    """Recompute §7's dataflow in one topological pass, for the §18 check."""
    from majestic.fabric.graph import TaintRole

    incoming: dict[str, bool] = {}
    outgoing: dict[str, bool] = {}
    for name in graph.topological_order():
        node = graph.nodes[name]
        arriving = any(outgoing.get(p, False) for p in graph.predecessors(name))
        incoming[name] = arriving
        role = node.role
        if role is TaintRole.CLEAR:
            outgoing[name] = False
        elif role is TaintRole.SOURCE:
            outgoing[name] = True
        elif role is TaintRole.NONE:
            outgoing[name] = arriving
        else:
            outgoing[name] = arriving
    return incoming


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def split_offline_core(graph: FabricGraph) -> tuple[list[str], list[str]]:
    """Split a graph into its offline core and its online tail (B-10 remedy).

    When static analysis reports broken closure it offers this split. Every node
    that transitively depends on a network-requiring node belongs to the tail;
    everything else can still run with the network switched off.
    """
    online: set[str] = set()
    for name in graph.topological_order():
        node = graph.nodes[name]
        if node.requires_network or any(p in online for p in graph.predecessors(name)):
            online.add(name)
    core = [n for n in graph.topological_order() if n not in online]
    tail = [n for n in graph.topological_order() if n in online]
    return core, tail
