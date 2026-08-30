"""Fabric's static analyses (Part 6).

The contribution is not the analyses — taint analysis is fifty years old — it is
the type system that makes a model-and-tool graph analysable at all. These tests
assert the three things that buys, and the one new result: that a cartridge's
decoding grammar already determines its channel capacity.
"""
from __future__ import annotations

import math
import random

import pytest

from majestic.fabric.analyser import (
    analyse,
    critical_path,
    prove_offline,
    rewrite_offline,
    routing_investment,
    system_accuracy,
)
from majestic.fabric.capacity import (
    UNBOUNDED_BITS,
    analyse_capacity,
    bits_for_domain,
    bits_for_schema,
    classify,
    repeated_capacity,
)
from majestic.fabric.executor import FabricRuntime, TaintViolation
from majestic.fabric.graph import FabricGraph, Node, NodeKind, TaintRole
from majestic.fabric.schedule import (
    SWAP_MS,
    belady,
    compare,
    expected_swaps,
    grouped_order,
    lru,
    reference_string,
)


def _chain(*nodes: Node, name: str = "g", **kw) -> FabricGraph:
    g = FabricGraph(name=name, **kw)
    for node in nodes:
        g.add(node)
    for a, b in zip(nodes, nodes[1:]):
        g.connect(a.name, b.name)
    return g


def _scraper(name: str = "scrape") -> Node:
    return Node(name, NodeKind.TOOL, requires_network=True, produces_untrusted=True)


def _sink(name: str = "notify") -> Node:
    return Node(name, NodeKind.TOOL, privileged=True)


# =========================================================================== #
# §12 — capacity from the decoding grammar. The new result.
# =========================================================================== #
def test_the_capacity_table_reproduces():
    """The mechanism Majestic already uses for output *validity* turns out to
    be a security primitive for free."""
    assert bits_for_domain(5) == pytest.approx(2.32, abs=0.01)     # 5-way
    assert bits_for_domain(3) == pytest.approx(1.58, abs=0.01)     # 3-way enum

    all_enum = {"priority": ["urgent", "normal", "low"],
                "code": {"max_cardinality": 2000}, "flag": "bool"}
    assert 10 < bits_for_schema(all_enum) < 16                     # ~13 bits

    assert bits_for_schema({**all_enum, "notes": "string"}) == UNBOUNDED_BITS


def test_one_free_text_field_converts_a_sanitiser_into_a_propagator():
    """The design-time fact somebody can act on."""
    constrained = {"priority": ["urgent", "normal", "low"]}
    assert classify(bits_for_schema(constrained), c_max=3.0) == "sanitiser"
    assert classify(bits_for_schema({**constrained, "doctor_name": "string"})) \
        == "propagator"


def test_narrowing_a_field_to_a_lookup_closes_the_channel():
    """Constraining doctor_name to the known referrer list collapses the channel
    from unbounded to about ten bits."""
    free = {"priority": ["urgent", "normal", "low"], "doctor_name": "string"}
    looked_up = {**free, "doctor_name": {"max_cardinality": 1200}}
    assert bits_for_schema(free) == UNBOUNDED_BITS
    assert bits_for_schema(looked_up) < 30


def test_a_narrow_classifier_is_safe_at_a_tolerance_and_unsafe_at_zero():
    """Binary taint rejects this graph. Capacity says 2.3 bits, which may be
    perfectly acceptable — and rejecting it buys no security."""
    g = _chain(_scraper(),
               Node("classify", NodeKind.CARTRIDGE, output_domain_bits=math.log2(5)),
               _sink())
    assert analyse_capacity(g, c_max=0.0).safe is False      # strict rule
    assert analyse_capacity(g, c_max=3.0).safe is True       # measured channel


def test_an_unconstrained_cartridge_is_a_propagator_at_any_tolerance():
    g = _chain(_scraper(), Node("summarise", NodeKind.CARTRIDGE), _sink())
    for c_max in (0.0, 3.0, 64.0):
        assert analyse_capacity(g, c_max=c_max).safe is False


def test_the_remedy_names_a_node_whose_schema_can_be_narrowed():
    """Naming the scraper would be useless advice: its output is the attacker's
    and there is no schema to tighten."""
    g = _chain(_scraper(), Node("summarise", NodeKind.CARTRIDGE), _sink())
    report = analyse_capacity(g, c_max=3.0)
    assert report.channels[0].bottleneck == "summarise"
    assert "constrain 'summarise'" in report.remedies()[0]


def test_capacity_beats_binary_taint_by_saying_how_much():
    """Binary taint says 'unsafe'. This says which node, and by how much."""
    g = _chain(_scraper(),
               Node("extract", NodeKind.CARTRIDGE, output_domain_bits=13.0),
               _sink())
    report = analyse_capacity(g, c_max=4.0)
    assert report.safe is False
    assert "narrow 'extract' from 13.0 bits" in report.remedies()[0]


def test_repeated_invocation_accumulates_bits():
    """The single-pass bound is not the whole story: a graph safe once is not
    automatically safe a thousand times."""
    assert repeated_capacity(2.3, 100) == pytest.approx(230.0)
    assert repeated_capacity(UNBOUNDED_BITS, 2) == UNBOUNDED_BITS
    with pytest.raises(ValueError):
        repeated_capacity(2.3, 0)


# =========================================================================== #
# §7-§10 — the lattice, and what cartridges do to it
# =========================================================================== #
def test_a_cartridge_propagates_rather_than_sanitising():
    """The tempting error: 'the model read the page and wrote its own summary,
    so the output is the model's'. False — a model given injected instructions
    emits attacker-chosen content."""
    assert Node("c", NodeKind.CARTRIDGE).role is TaintRole.PROPAGATE
    assert Node("c", NodeKind.CARTRIDGE).capacity_bits == UNBOUNDED_BITS


def test_a_confirm_node_clears_taint():
    """A human interposed judgement."""
    confirm = Node("human", NodeKind.CONFIRM)
    assert confirm.role is TaintRole.CLEAR
    assert confirm.capacity_bits == 0.0

    g = _chain(_scraper(), Node("summarise", NodeKind.CARTRIDGE),
               Node("human", NodeKind.CONFIRM), _sink())
    assert analyse_capacity(g, c_max=0.0).safe is True


def test_a_graph_with_no_source_or_no_sink_has_no_channels():
    assert analyse_capacity(_chain(_scraper(), Node("out", NodeKind.OUTPUT))).channels == []
    assert analyse_capacity(_chain(Node("in", NodeKind.INPUT), _sink())).channels == []


def test_the_analysis_is_a_single_pass_over_a_diamond():
    """The widest path is the max over paths of the min over nodes, and it is
    computable in one sweep — no path enumeration, no fixpoint."""
    g = FabricGraph(name="diamond")
    g.add(_scraper())
    g.add(Node("narrow", NodeKind.CARTRIDGE, output_domain_bits=2.0))
    g.add(Node("wide", NodeKind.CARTRIDGE, output_domain_bits=20.0))
    g.add(_sink())
    for a, b in (("scrape", "narrow"), ("scrape", "wide"),
                 ("narrow", "notify"), ("wide", "notify")):
        g.connect(a, b)

    report = analyse_capacity(g, c_max=4.0)
    # The WIDEST path governs safety, so the analysis must find the 20-bit arm.
    assert report.safe is False
    assert report.channels[0].capacity == pytest.approx(20.0)
    assert report.channels[0].bottleneck == "wide"


# =========================================================================== #
# §6 — offline closure by rewriting, not configuration
# =========================================================================== #
def _escalating_graph() -> FabricGraph:
    g = FabricGraph(name="triage")
    g.add(Node("intake", NodeKind.INPUT))
    g.add(Node("classify", NodeKind.CARTRIDGE, output_domain_bits=2.0))
    g.add(Node("escalate", NodeKind.TOOL, requires_network=True,
               net_condition="confidence_below_threshold"))
    g.add(Node("reply", NodeKind.OUTPUT))
    g.connect("intake", "classify")
    g.connect("classify", "escalate")
    g.connect("classify", "reply")
    g.connect("escalate", "reply")
    return g


def test_a_conditional_escalation_is_distinguished_from_an_unconditional_call():
    g = _escalating_graph()
    assert g.nodes["escalate"].net_unconditional is False
    assert Node("api", NodeKind.TOOL, requires_network=True).net_unconditional is True


def test_offline_mode_deletes_the_branch_rather_than_trusting_a_threshold():
    """A threshold is a runtime value: mutable, restorable by a flag flip, and
    exactly the failure where a customer who bought offline finds a network
    call at 2 a.m. So the edge is removed and the proof is over the residue."""
    rewrite = rewrite_offline(_escalating_graph())
    assert "escalate" in rewrite.removed_nodes
    assert "escalate" not in rewrite.graph.nodes
    assert all("escalate" not in edge for edge in rewrite.graph.edges)
    assert rewrite.viable is True


def test_a_graph_that_cannot_lose_its_network_node_is_refused_by_name():
    """If deleting them disconnects the graph the task genuinely cannot be done
    offline, and the violation names the sink — the actionable message."""
    g = _chain(Node("intake", NodeKind.INPUT), _scraper(),
               Node("reply", NodeKind.OUTPUT))
    rewrite = rewrite_offline(g)
    assert rewrite.viable is False
    assert "reply" in rewrite.lost_sinks

    ok, reasons = prove_offline(g)
    assert ok is False
    assert any("cannot be done offline" in r for r in reasons)


def test_a_rewritable_graph_is_provably_offline_capable():
    ok, reasons = prove_offline(_escalating_graph())
    assert ok is True
    assert any("deletes the branch" in r for r in reasons)


def test_a_graph_with_no_network_is_offline_with_no_rewriting():
    ok, reasons = prove_offline(_chain(Node("in", NodeKind.INPUT),
                                       Node("c", NodeKind.CARTRIDGE)))
    assert ok is True and reasons == []


# =========================================================================== #
# §13 — the mixed-base RAM trap
# =========================================================================== #
def _cartridge_chain(bases: list[str]) -> FabricGraph:
    g = FabricGraph(name="fleet")
    g.add(Node("in", NodeKind.INPUT))
    prev = "in"
    for i, base in enumerate(bases):
        name = f"c{i}"
        g.add(Node(name, NodeKind.CARTRIDGE, ram_mb=30,
                   metadata={"base_ref": base, "base_ram_mb": 1120}))
        g.connect(prev, name)
        prev = name
    return g


def test_three_cartridges_can_cost_more_than_twenty():
    """The base term is a sum over DISTINCT bases and it dominates everything."""
    twenty_one_base = analyse(_cartridge_chain(["qwen"] * 20), ram_budget_mb=2600)
    three_three_bases = analyse(_cartridge_chain(["qwen", "llama", "smol"]),
                                ram_budget_mb=2600)
    assert twenty_one_base.peak_ram_mb == pytest.approx(1720, abs=1)
    assert three_three_bases.peak_ram_mb == pytest.approx(3450, abs=1)
    assert twenty_one_base.ok is True
    assert three_three_bases.ok is False


def test_a_mixed_base_graph_is_told_why_it_failed():
    """Not just 'over budget' — device-targeted graphs must be base-homogeneous,
    which makes base selection a graph-level decision."""
    result = analyse(_cartridge_chain(["qwen", "llama", "smol"]), ram_budget_mb=2600)
    assert any("base-homogeneous" in v for v in result.violations)
    assert len(result.bases) == 3


def test_a_mixed_base_graph_within_budget_is_only_warned():
    result = analyse(_cartridge_chain(["qwen", "llama"]), ram_budget_mb=8000)
    assert result.ok is True
    assert any("distinct bases" in w for w in result.warnings)


# =========================================================================== #
# §14 — the critical path
# =========================================================================== #
def test_latency_follows_the_slowest_path():
    g = FabricGraph(name="fork")
    g.add(Node("in", NodeKind.INPUT))
    g.add(Node("fast", NodeKind.CARTRIDGE, p95_ms=100))
    g.add(Node("slow", NodeKind.CARTRIDGE, p95_ms=900))
    g.add(Node("out", NodeKind.OUTPUT, p95_ms=50))
    for a, b in (("in", "fast"), ("in", "slow"), ("fast", "out"), ("slow", "out")):
        g.connect(a, b)
    path, ms = critical_path(g)
    assert "slow" in path and "fast" not in path
    assert ms == pytest.approx(950.0)


def test_a_swap_is_charged_only_when_the_adapter_changes():
    same = _chain(Node("a", NodeKind.CARTRIDGE, p95_ms=100, adapter_ref="x"),
                  Node("b", NodeKind.CARTRIDGE, p95_ms=100, adapter_ref="x"))
    switch = _chain(Node("a", NodeKind.CARTRIDGE, p95_ms=100, adapter_ref="x"),
                    Node("b", NodeKind.CARTRIDGE, p95_ms=100, adapter_ref="y"))
    assert critical_path(switch)[1] - critical_path(same)[1] == pytest.approx(SWAP_MS)


def test_a_latency_budget_is_enforced_with_the_path_named():
    g = _chain(Node("in", NodeKind.INPUT), Node("slow", NodeKind.CARTRIDGE, p95_ms=9000))
    result = analyse(g, latency_budget_ms=2000)
    assert result.ok is False
    assert any("critical path" in v and "slow" in v for v in result.violations)


# =========================================================================== #
# §15 — Belady, because this is offline paging
# =========================================================================== #
def test_belady_is_never_worse_than_lru():
    """Provably optimal for the offline problem, so there is nothing to tune."""
    rng = random.Random(0)
    for _ in range(200):
        sigma = [rng.choice("abcde") for _ in range(12)]
        capacity = rng.randint(1, 4)
        assert belady(sigma, capacity).swaps <= lru(sigma, capacity).swaps


def test_belady_beats_lru_on_the_case_lru_gets_wrong():
    sigma = ["a", "b", "c", "a", "d", "a", "b", "c"]
    assert belady(sigma, 2).swaps < lru(sigma, 2).swaps


def test_the_reference_string_is_known_before_execution():
    """This is what makes the optimal policy available: the graph is a DAG, so
    execution order — and therefore the reference string — is fixed in advance."""
    g = _chain(Node("a", NodeKind.CARTRIDGE, adapter_ref="x"),
               Node("b", NodeKind.CARTRIDGE, adapter_ref="y"),
               Node("c", NodeKind.CARTRIDGE, adapter_ref="x"))
    assert reference_string(g) == ("x", "y", "x")


def test_a_grouped_order_shrinks_the_working_set():
    """Many topological orders exist and they do not cost the same. Grouping
    same-adapter nodes is free at construction time."""
    g = FabricGraph(name="regroupable")
    g.add(Node("root", NodeKind.INPUT))
    for name, adapter in (("a1", "x"), ("b1", "y"), ("a2", "x"), ("b2", "y")):
        g.add(Node(name, NodeKind.CARTRIDGE, adapter_ref=adapter))
        g.connect("root", name)

    grouped = reference_string(g, grouped_order(g))
    assert belady(grouped, 1).swaps <= belady(("x", "y", "x", "y"), 1).swaps


def test_swaps_are_priced_in_measured_milliseconds():
    plan = belady(["a", "b", "a", "b"], 1)
    assert plan.swaps == 4
    assert plan.swap_ms == pytest.approx(4 * SWAP_MS)
    assert plan.hit_rate == 0.0


def test_branch_arms_are_weighted_by_probability():
    """A branch makes part of the reference string data-dependent, so an
    expectation is the honest thing to report."""
    result = expected_swaps(["a", "b"], [(["c"], 0.7), (["a"], 0.3)], capacity=2)
    assert result["prefix_swaps"] == 2
    assert 2.0 <= result["expected_swaps"] <= 3.0

    with pytest.raises(ValueError, match="sum to 1"):
        expected_swaps(["a"], [(["b"], 0.5)], capacity=2)


def test_an_empty_cache_is_refused():
    with pytest.raises(ValueError, match="at least one adapter"):
        belady(["a"], 0)


def test_the_comparison_reports_what_optimality_is_worth():
    g = _chain(Node("a", NodeKind.CARTRIDGE, adapter_ref="x"),
               Node("b", NodeKind.CARTRIDGE, adapter_ref="y"),
               Node("c", NodeKind.CARTRIDGE, adapter_ref="z"),
               Node("d", NodeKind.CARTRIDGE, adapter_ref="x"))
    report = compare(g, capacity=2)
    assert report["belady_swaps"] <= report["lru_swaps"]
    assert "offline paging" in report["why"]


# =========================================================================== #
# §17 — the misroute penalty
# =========================================================================== #
def test_the_misroute_table_reproduces():
    """The earlier '91% router over 96% experts gives 87%' silently assumed a
    misroute always produces a wrong answer."""
    assert system_accuracy(0.91, 0.96, 0.0) == pytest.approx(0.874, abs=0.001)
    assert system_accuracy(0.91, 0.96, 0.70) == pytest.approx(0.937, abs=0.001)


def test_abstention_buys_more_than_a_better_router():
    """And it helps at every router accuracy, not only at the margin improved."""
    report = routing_investment()
    assert report["points_from_abstention"] > report["points_from_router"]
    assert "invest in abstention before the router" in report["verdict"]


def test_accuracies_outside_the_unit_interval_are_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        system_accuracy(1.5, 0.96)


# =========================================================================== #
# §3 / §18 — the certificate, and drift between analyser and executor
# =========================================================================== #
def test_the_graph_hash_changes_when_the_structure_does():
    """A verified graph carries its own certificate, and any edit invalidates
    it. Hashing the structure makes 'any edit' detectable rather than trusted."""
    g = _chain(Node("a", NodeKind.CARTRIDGE), Node("b", NodeKind.OUTPUT))
    before = g.graph_hash
    g.add(Node("c", NodeKind.TOOL, privileged=True))
    assert g.graph_hash != before


def test_the_hash_is_stable_across_equal_graphs():
    build = lambda: _chain(Node("a", NodeKind.CARTRIDGE), Node("b", NodeKind.OUTPUT))
    assert build().graph_hash == build().graph_hash


def test_the_runtime_agrees_with_the_static_proof():
    """§18: a cheap dynamic check whose only job is to catch drift between two
    implementations of the same rule."""
    g = _chain(Node("in", NodeKind.INPUT),
               Node("classify", NodeKind.CARTRIDGE, output_domain_bits=2.0),
               Node("out", NodeKind.OUTPUT))
    handlers = {n: (lambda payload, _n=n: {"node": _n}) for n in g.nodes}
    result = FabricRuntime(g, handlers, offline=True).run({"text": "hello"})
    assert result.ok is True


def test_taint_drift_between_analyser_and_executor_is_fatal():
    """Never silently prefer one verdict: the graph was admitted on the static
    proof, so a mismatch means admission used a different rule than execution."""
    g = _chain(_scraper("feed"), Node("out", NodeKind.OUTPUT))
    handlers = {n: (lambda payload, _n=n: {"node": _n}) for n in g.nodes}
    runtime = FabricRuntime(g, handlers, offline=False)
    runtime._static_taint = lambda: {"out": False, "feed": True}   # forced disagreement
    with pytest.raises(TaintViolation, match="taint drift"):
        runtime.run({"text": "hello"})
