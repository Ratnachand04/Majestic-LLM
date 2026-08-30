"""The typed DAG a customer workflow compiles to (B-10).

Nodes are cartridges, tools or control flow; edges carry typed values. Being a
TYPED DAG rather than free-form agent conversation is what makes static analysis
possible at all — offline closure, RAM bounds, cost bounds and prompt-injection
reachability can all be proven before the graph runs. Free-form multi-agent chat
can prove none of this.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    CARTRIDGE = "cartridge"   # a certified specialist
    TOOL = "tool"             # side effects live here
    CONTROL = "control"       # branch, map, join
    INPUT = "input"
    OUTPUT = "output"
    CONFIRM = "confirm"       # a human interposes judgement — clears taint (§10)


class TaintRole(str, Enum):
    """What a node does to the taint set flowing through it (§7).

    ``PROPAGATE`` is the default for cartridges and it is load-bearing. The
    tempting error is to treat a model as a sanitiser — "it read the scraped
    page and wrote its own summary, so the output is the model's" — which is
    false: a model given injected instructions emits attacker-chosen content.
    §12 gives the one condition under which propagation can be weakened, and it
    is not "we trust the model".
    """

    SOURCE = "source"        # introduces untrusted content
    PROPAGATE = "propagate"  # passes taint through. The default.
    CLEAR = "clear"          # a human approved it
    NONE = "none"            # produces nothing tainted


#: Capacity of an unconstrained output, in bits. Free text is an open channel.
UNBOUNDED_BITS = float("inf")


@dataclass
class Node:
    """One node of a Fabric graph.

    Attributes that drive the static analysis
    ----------------------------------------
    requires_network:
        The node cannot run without a network. One such node anywhere on a path
        breaks offline closure for the whole graph.
    produces_untrusted:
        The node emits content from outside the trust boundary — a scraped page,
        a retrieved document, a third-party API body. Every scraped page is
        untrusted INPUT, not data (2302.12173).
    privileged:
        The node has real-world authority (writes a database, sends a message,
        moves money). An untrusted->privileged path is an exploit waiting to
        happen.
    ram_mb / cost_usd:
        Static budgets summed by the analyser.
    """

    name: str
    kind: NodeKind
    requires_network: bool = False
    produces_untrusted: bool = False
    privileged: bool = False
    ram_mb: float = 0.0
    cost_usd: float = 0.0
    input_type: str = "any"
    output_type: str = "any"
    cartridge_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: §6 — when a node needs the network only under a condition, the condition
    #: is named here. An escalating router is ``requires_network=False`` with
    #: ``net_condition="confidence_below_threshold"``: it does not need the
    #: network to run, only to take one branch.
    #:
    #: This is NOT how offline closure is proven. A condition is a runtime
    #: value, and offline mode deletes the branch instead — see
    #: :func:`~majestic.fabric.analyser.rewrite_offline`. The field exists so
    #: the analyser can tell a conditional escalation apart from an
    #: unconditional network call and report the difference honestly.
    net_condition: str | None = None
    #: §7 — the taint transfer function. Defaults are derived from the other
    #: flags when left unset, so existing graphs keep their meaning.
    taint_role: TaintRole | None = None
    #: §12 — ``log2 |D_out|``: how many bits of the output an attacker upstream
    #: could control. Derived from the compiled decoding grammar at package
    #: time. ``None`` means unconstrained, which is infinite capacity.
    output_domain_bits: float | None = None
    #: §14 — measured or predicted per-node latency, for the critical path.
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    #: §15 — which adapter this node needs resident. Drives the paging schedule.
    adapter_ref: str | None = None

    @property
    def role(self) -> TaintRole:
        """The effective taint role, defaulted from the legacy flags."""
        if self.taint_role is not None:
            return self.taint_role
        if self.kind is NodeKind.CONFIRM:
            return TaintRole.CLEAR
        if self.produces_untrusted:
            return TaintRole.SOURCE
        if self.kind in (NodeKind.CARTRIDGE, NodeKind.CONTROL):
            return TaintRole.PROPAGATE
        return TaintRole.NONE

    @property
    def capacity_bits(self) -> float:
        """Channel capacity of this node's output (§11-§12).

        A node that constrains its output to a finite domain can carry only
        ``log2|D|`` bits of attacker influence, whatever it read. A node with an
        unconstrained output carries everything.
        """
        if self.role is TaintRole.CLEAR:
            return 0.0
        if self.output_domain_bits is None:
            return UNBOUNDED_BITS
        return float(self.output_domain_bits)

    @property
    def net_unconditional(self) -> bool:
        """§6: needs the network on every path, not merely on one branch."""
        return self.requires_network and self.net_condition is None


@dataclass
class FabricGraph:
    """A directed acyclic graph of typed nodes."""

    name: str = "graph"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    graph_version: str = "1.0.0"
    owner_id: str = ""
    #: §11 — bits of attacker influence tolerated at a privileged sink. Zero
    #: recovers the strict "no taint reaches privilege" rule. Set it per sink by
    #: CONSEQUENCE: one bit that flips an authorisation is worse than ten that
    #: pick a label.
    max_taint_capacity_bits: float = 0.0

    @property
    def graph_hash(self) -> str:
        """Content hash over the structure the proofs were established on.

        §3: a verified graph carries its own certificate, and the certificate is
        invalidated by any edit. Hashing the structure is what makes "any edit"
        detectable rather than a matter of trust.
        """
        payload = {
            "version": self.graph_version,
            "nodes": sorted(
                (n.name, n.kind.value, n.requires_network, n.net_condition,
                 n.privileged, n.role.value, n.output_domain_bits, n.adapter_ref)
                for n in self.nodes.values()
            ),
            "edges": sorted(self.edges),
            "max_taint_capacity_bits": self.max_taint_capacity_bits,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:32]

    # -- construction ----------------------------------------------------- #
    def add(self, node: Node) -> Node:
        if node.name in self.nodes:
            raise ValueError(f"duplicate node {node.name!r}")
        self.nodes[node.name] = node
        return node

    def connect(self, src: str, dst: str) -> None:
        for name in (src, dst):
            if name not in self.nodes:
                raise KeyError(f"unknown node {name!r}")
        self.edges.append((src, dst))

    # -- traversal --------------------------------------------------------- #
    def successors(self, name: str) -> list[str]:
        return [d for s, d in self.edges if s == name]

    def predecessors(self, name: str) -> list[str]:
        return [s for s, d in self.edges if d == name]

    def roots(self) -> list[str]:
        targets = {d for _, d in self.edges}
        return [n for n in self.nodes if n not in targets]

    def topological_order(self) -> list[str]:
        """Kahn's algorithm. Raises ``ValueError`` if the graph has a cycle."""
        indegree = {n: 0 for n in self.nodes}
        for _, dst in self.edges:
            indegree[dst] += 1
        queue = [n for n, d in indegree.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for succ in self.successors(node):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self.nodes):
            raise ValueError(f"graph {self.name!r} contains a cycle; Fabric graphs must be acyclic")
        return order

    def sources(self) -> list[str]:
        """Nodes that introduce untrusted content."""
        return [n for n, node in self.nodes.items() if node.role is TaintRole.SOURCE]

    def privileged_sinks(self) -> list[str]:
        """Nodes with real-world authority. The things taint must not reach."""
        return [n for n, node in self.nodes.items() if node.privileged]

    def bases(self) -> set[str]:
        """Distinct model bases the graph would hold resident (§13)."""
        return {
            str(node.metadata.get("base_ref", node.cartridge_id or name))
            for name, node in self.nodes.items()
            if node.kind is NodeKind.CARTRIDGE
        }

    def paths_between(self, src: str, dst: str) -> list[list[str]]:
        """All simple paths from ``src`` to ``dst`` (graph is acyclic)."""
        out: list[list[str]] = []

        def walk(node: str, trail: list[str]) -> None:
            if node == dst:
                out.append([*trail, node])
                return
            for succ in self.successors(node):
                if succ not in trail:
                    walk(succ, [*trail, node])

        walk(src, [])
        return out
