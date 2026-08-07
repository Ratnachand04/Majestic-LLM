"""The typed DAG a customer workflow compiles to (B-10).

Nodes are cartridges, tools or control flow; edges carry typed values. Being a
TYPED DAG rather than free-form agent conversation is what makes static analysis
possible at all — offline closure, RAM bounds, cost bounds and prompt-injection
reachability can all be proven before the graph runs. Free-form multi-agent chat
can prove none of this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    CARTRIDGE = "cartridge"   # a certified specialist
    TOOL = "tool"             # side effects live here
    CONTROL = "control"       # branch, map, join
    INPUT = "input"
    OUTPUT = "output"


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


@dataclass
class FabricGraph:
    """A directed acyclic graph of typed nodes."""

    name: str = "graph"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

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
