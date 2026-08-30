"""Quantitative information flow: how unsafe, not merely whether (§11-§12).

Binary taint is too coarse and it forbids useful designs for no security gain. A
five-way classifier that reads a scraped page and emits one of five labels *is*
tainted — but the attacker controls at most ``log2 5 = 2.3`` bits. Rejecting
that graph buys nothing.

So measure the channel instead of detecting it:

    cap(v) = log2 |D_out(v)|          bits a node can carry
    cap(p) = min over v in p of cap(v)   the bottleneck governs a path

    safe_c(G)  iff  every source-to-privileged-sink path has cap(p) <= c_max

``c_max = 0`` recovers the strict rule. ``c_max = 4`` permits small
enum-constrained mediators.

**The result (§12).** The grammar compiled from a cartridge's ``output_schema``
— which exists for output *validity* — already determines its channel capacity.
The mechanism is a security primitive for free:

    5-way classifier          |D| = 5        2.3 bits   sanitiser at c_max >= 3
    enum {urgent,normal,low}  |D| = 3        1.6 bits   sanitiser
    all-enum extraction       |D| ~ 1e4       13 bits   narrow propagator
    one free-text field       unbounded         inf     propagator
    summarisation             unbounded         inf     propagator

**One free-text field converts a sanitiser into a propagator**, and that is a
design-time fact somebody can act on: constraining ``doctor_name`` to a lookup
over the known referrer list rather than free text collapses the channel from
unbounded to about ten bits, and may be the difference between a graph that
verifies and one that does not.

Novelty, carefully: quantitative information flow is established, and so is
grammar-constrained generation. *Deriving an LLM node's channel capacity from
its decoding grammar and using it as a sanitiser condition in a static analysis*
is the new part.

Two caveats that must travel with any claim made from this:

* **Capacity bounds one pass.** Repeated queries accumulate, so a graph invoked
  ``n`` times has effective capacity ``n * cap`` against a patient attacker.
* **Capacity says nothing about *which* bits.** One bit that flips an
  authorisation decision is worse than ten that select a label, so ``c_max``
  belongs per sink, set by consequence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from majestic.fabric.graph import UNBOUNDED_BITS, FabricGraph, TaintRole
from majestic.logging_utils import get_logger

logger = get_logger(__name__)

#: A single unconstrained string field. Not a large number — an open channel.
FREE_TEXT = UNBOUNDED_BITS

#: Assumed cardinality of a numeric field under a grammar, when no range is
#: declared. A 32-bit-ish quantity.
# HYPOTHESIS — a placeholder until grammars declare numeric ranges.
NUMERIC_BITS = 32.0


def bits_for_domain(size: int) -> float:
    """``log2 |D|``. A one-element domain carries nothing."""
    if size <= 0:
        raise ValueError("a domain must have at least one member")
    return math.log2(size)


def bits_for_schema(schema: Mapping[str, Any]) -> float:
    """Capacity implied by a compiled output schema (§12).

    Fields compose additively in bits, because the attacker may choose each
    independently. One unconstrained field therefore makes the whole schema
    unbounded — which is exactly the asymmetry that makes this actionable.
    """
    total = 0.0
    for name, spec in schema.items():
        if str(name).startswith("_"):
            continue
        field_bits = _field_bits(spec)
        if field_bits == UNBOUNDED_BITS:
            logger.debug("capacity: field %r is unconstrained — channel is open", name)
            return UNBOUNDED_BITS
        total += field_bits
    return total


def _field_bits(spec: Any) -> float:
    """Capacity of one schema field."""
    if isinstance(spec, (list, tuple, set)):          # an explicit enum
        return bits_for_domain(len(spec)) if spec else 0.0
    if isinstance(spec, Mapping):
        if "enum" in spec:
            return bits_for_domain(len(spec["enum"]))
        if "max_cardinality" in spec:
            return bits_for_domain(int(spec["max_cardinality"]))
        kind = str(spec.get("type", "string"))
        return _bits_for_type(kind)
    return _bits_for_type(str(spec))


def _bits_for_type(kind: str) -> float:
    kind = kind.strip().lower()
    if kind in ("bool", "boolean"):
        return 1.0
    if kind in ("int", "integer", "number", "float", "numeric"):
        return NUMERIC_BITS
    if kind in ("date", "datetime", "timestamp"):
        return 25.0        # ~1 day resolution over a century
    return FREE_TEXT       # string and anything unrecognised


@dataclass(frozen=True)
class Channel:
    """One source-to-sink path, with the bottleneck that governs it."""

    source: str
    sink: str
    path: tuple[str, ...]
    capacity: float
    bottleneck: str

    @property
    def unbounded(self) -> bool:
        return self.capacity == UNBOUNDED_BITS

    def within(self, c_max: float) -> bool:
        return self.capacity <= c_max

    def describe(self) -> str:
        cap = "unbounded" if self.unbounded else f"{self.capacity:.1f} bits"
        return (f"{' -> '.join(self.path)}: {cap}, narrowed at "
                f"{self.bottleneck!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sink": self.sink,
            "path": list(self.path),
            "capacity_bits": None if self.unbounded else round(self.capacity, 2),
            "unbounded": self.unbounded,
            "bottleneck": self.bottleneck,
        }


def path_capacity(graph: FabricGraph, path: Sequence[str]) -> tuple[float, str]:
    """``cap(p) = min over the path``, and the node that sets it.

    The bottleneck is the actionable part of the answer: it names the node whose
    output schema to narrow.
    """
    best, at = UNBOUNDED_BITS, ""
    for name in path:
        node = graph.nodes[name]
        # The source itself is what the attacker controls; its own capacity does
        # not constrain what it emits. Downstream nodes are the narrowing ones.
        if node.role is TaintRole.SOURCE:
            continue
        # Name the first node that could be narrowed, so that when nothing
        # constrains the path the remedy points at a schema somebody can
        # actually change. Naming the scraper would be useless advice: its
        # output is the attacker's, and there is no schema to tighten.
        if not at:
            at = name
        cap = node.capacity_bits
        if cap < best:
            best, at = cap, name
    return best, at or (path[0] if path else "")


@dataclass
class CapacityReport:
    """Every channel from an untrusted source to a privileged sink."""

    channels: list[Channel] = field(default_factory=list)
    c_max: float = 0.0

    @property
    def violations(self) -> list[Channel]:
        return [c for c in self.channels if not c.within(self.c_max)]

    @property
    def safe(self) -> bool:
        return not self.violations

    @property
    def worst(self) -> Channel | None:
        return max(self.channels, key=lambda c: c.capacity, default=None)

    def remedies(self) -> list[str]:
        """What to narrow. The reason this beats binary taint.

        Binary taint says "unsafe". This says which node, and by how much.
        """
        out: list[str] = []
        for channel in self.violations:
            if channel.unbounded:
                out.append(
                    f"constrain {channel.bottleneck!r}: its output schema has an "
                    "unconstrained field, so the channel is open. One free-text "
                    "field converts a sanitiser into a propagator — replacing it "
                    "with a lookup over a known set collapses the channel to a "
                    "handful of bits"
                )
            else:
                out.append(
                    f"narrow {channel.bottleneck!r} from {channel.capacity:.1f} bits "
                    f"to at most {self.c_max:.1f}, or interpose a confirm node so a "
                    "human clears the taint"
                )
        return out

    def as_dict(self) -> dict[str, Any]:
        worst = self.worst
        return {
            "c_max_bits": self.c_max,
            "channels": [c.as_dict() for c in self.channels],
            "violations": [c.as_dict() for c in self.violations],
            "worst_capacity_bits": (
                None if worst is None or worst.unbounded else round(worst.capacity, 2)
            ),
            "safe": self.safe,
            "remedies": self.remedies(),
        }


def analyse_capacity(graph: FabricGraph, c_max: float | None = None) -> CapacityReport:
    """§11's threshold safety condition, in **one topological pass**.

    The safety question is "does any path exceed ``c_max``", which is the
    *widest* path — the maximum over paths of the minimum over nodes. Enumerating
    paths to find it is exponential in the worst case and unnecessary: on a DAG
    the widest path to every node is computable in a single sweep, because every
    predecessor is resolved before its successor.

        width(v) = max over predecessors u of min(width(u), cap(v))

    with sources seeded at unbounded, since the attacker controls what a source
    emits. Back-pointers recover the witness path, so the report still names the
    route and the node to narrow.

    **O(|V| + |E|), no iteration.** That is §8's linearity claim, and it holds
    for the quantitative analysis and not only the binary one. The DAG
    restriction — the flexibility Fabric gives up — is what buys it.

    ``c_max = 0`` reproduces binary taint exactly: any surviving path from an
    untrusted source to a privileged sink is a violation unless something on it
    clears or fully constrains the flow.
    """
    limit = graph.max_taint_capacity_bits if c_max is None else c_max
    report = CapacityReport(c_max=limit)

    sources = set(graph.sources())
    sinks = graph.privileged_sinks()
    if not sources or not sinks:
        return report

    # width[v] = capacity of the widest tainted path reaching v, or None when no
    # tainted path reaches it at all.
    width: dict[str, float] = {}
    origin: dict[str, str] = {}       # which source that path started from
    came_from: dict[str, str] = {}

    for name in graph.topological_order():
        node = graph.nodes[name]
        if node.role is TaintRole.CLEAR:
            # A human interposed judgement: nothing tainted continues past here.
            continue
        if name in sources:
            width[name] = UNBOUNDED_BITS
            origin[name] = name
            continue

        best: float | None = None
        for pred in graph.predecessors(name):
            if pred not in width:
                continue
            through = min(width[pred], node.capacity_bits)
            if best is None or through > best:
                best, came_from[name], origin[name] = through, pred, origin[pred]
        if best is not None:
            width[name] = best

    for sink in sinks:
        if sink not in width:
            continue
        path = _witness(sink, came_from)
        capacity = width[sink]
        _, bottleneck = path_capacity(graph, path)
        report.channels.append(Channel(
            source=origin[sink], sink=sink, path=tuple(path),
            capacity=capacity, bottleneck=bottleneck,
        ))

    if report.violations:
        logger.warning(
            "fabric: %d channel(s) exceed %.1f bits at a privileged sink",
            len(report.violations), limit,
        )
    return report


def _witness(sink: str, came_from: dict[str, str]) -> list[str]:
    """Reconstruct the widest path from the back-pointers."""
    path = [sink]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    return list(reversed(path))


def _cleared(graph: FabricGraph, path: Iterable[str]) -> bool:
    """A confirm node anywhere on the path clears it (§10)."""
    return any(graph.nodes[n].role is TaintRole.CLEAR for n in path)


def classify(node_bits: float, c_max: float = 3.0) -> str:
    """What a node is, at a given tolerance. The §12 table, as a function."""
    if node_bits == UNBOUNDED_BITS:
        return "propagator"
    if node_bits <= c_max:
        return "sanitiser"
    return "narrow propagator"


def repeated_capacity(capacity: float, invocations: int) -> float:
    """``n * cap`` — what a patient attacker accumulates over ``n`` passes.

    The single-pass bound is not the whole story, and a graph that is safe once
    is not automatically safe a thousand times.
    """
    if invocations < 1:
        raise ValueError("invocation count must be at least 1")
    if capacity == UNBOUNDED_BITS:
        return UNBOUNDED_BITS
    return capacity * invocations


__all__ = [
    "FREE_TEXT", "NUMERIC_BITS",
    "CapacityReport", "Channel",
    "analyse_capacity", "bits_for_domain", "bits_for_schema", "classify",
    "path_capacity", "repeated_capacity",
]
