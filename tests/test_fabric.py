"""Tests for FABRIC: the typed DAG and its static analyser (B-10, GAP-04)."""
from __future__ import annotations

import pytest

from majestic.fabric import FabricGraph, Node, NodeKind, analyse


def _customer_graph() -> FabricGraph:
    """The three-node graph from B-10: two offline cartridges, one online tool."""
    g = FabricGraph("requisition-flow")
    g.add(Node("req-extractor-v3", NodeKind.CARTRIDGE, ram_mb=30,
               metadata={"base_ref": "Qwen/Qwen3-1.7B", "base_ram_mb": 1100}))
    g.add(Node("urgency-classifier", NodeKind.CARTRIDGE, ram_mb=30,
               metadata={"base_ref": "Qwen/Qwen3-1.7B", "base_ram_mb": 1100}))
    g.add(Node("doctor-notify", NodeKind.TOOL, requires_network=True, privileged=True))
    g.connect("req-extractor-v3", "urgency-classifier")
    g.connect("urgency-classifier", "doctor-notify")
    return g


# --- graph mechanics ------------------------------------------------------ #
def test_topological_order():
    order = _customer_graph().topological_order()
    assert order.index("req-extractor-v3") < order.index("doctor-notify")


def test_cycles_are_rejected():
    g = FabricGraph()
    g.add(Node("a", NodeKind.CARTRIDGE))
    g.add(Node("b", NodeKind.CARTRIDGE))
    g.connect("a", "b")
    g.connect("b", "a")
    with pytest.raises(ValueError, match="cycle"):
        g.topological_order()


def test_duplicate_node_rejected():
    g = FabricGraph()
    g.add(Node("a", NodeKind.CARTRIDGE))
    with pytest.raises(ValueError, match="duplicate"):
        g.add(Node("a", NodeKind.TOOL))


# --- offline closure (the differentiator) --------------------------------- #
def test_offline_closure_broken_is_flagged():
    result = analyse(_customer_graph(), offline_required=True)
    assert result.offline_closed is False
    assert "doctor-notify" in result.network_nodes
    assert any("offline closure BROKEN" in v for v in result.violations)
    assert "offline core and an online tail" in result.suggest_split()


def test_offline_closure_only_a_warning_when_not_required():
    result = analyse(_customer_graph(), offline_required=False)
    assert result.offline_closed is False
    assert any("offline closure BROKEN" in w for w in result.warnings)


def test_fully_offline_graph_is_closed():
    g = FabricGraph()
    g.add(Node("extract", NodeKind.CARTRIDGE, ram_mb=30))
    g.add(Node("classify", NodeKind.CARTRIDGE, ram_mb=30))
    g.connect("extract", "classify")
    result = analyse(g, offline_required=True)
    assert result.offline_closed is True
    assert result.ok is True
    assert result.suggest_split() is None


# --- RAM and cost bounds --------------------------------------------------- #
def test_shared_base_is_counted_once():
    """Twenty specialists fit because they share one resident base (B-09)."""
    g = FabricGraph()
    for i in range(20):
        g.add(Node(f"c{i}", NodeKind.CARTRIDGE, ram_mb=30,
                   metadata={"base_ref": "shared", "base_ram_mb": 1100}))
    result = analyse(g)
    assert result.peak_ram_mb == pytest.approx(1100 + 20 * 30)


def test_ram_budget_violation():
    g = FabricGraph()
    g.add(Node("big", NodeKind.CARTRIDGE, ram_mb=100,
               metadata={"base_ref": "b", "base_ram_mb": 4500}))
    result = analyse(g, ram_budget_mb=2400)
    assert result.ok is False
    assert any("device budget" in v for v in result.violations)


def test_cost_ceiling_violation():
    g = FabricGraph()
    g.add(Node("cloud", NodeKind.TOOL, cost_usd=0.05))
    result = analyse(g, cost_ceiling_usd=0.01)
    assert result.ok is False
    assert any("ceiling" in v for v in result.violations)


# --- indirect prompt injection --------------------------------------------- #
def test_untrusted_content_reaching_a_privileged_tool_is_a_violation():
    """Every scraped page is untrusted INPUT, not data (2302.12173)."""
    g = FabricGraph()
    g.add(Node("scraper", NodeKind.TOOL, requires_network=True, produces_untrusted=True))
    g.add(Node("summarise", NodeKind.CARTRIDGE, ram_mb=30))
    g.add(Node("send-payment", NodeKind.TOOL, privileged=True))
    g.connect("scraper", "summarise")
    g.connect("summarise", "send-payment")
    result = analyse(g)
    assert result.ok is False
    assert any("prompt-injection" in v for v in result.violations)
    assert result.injection_paths


def test_no_injection_path_when_privileged_node_is_upstream():
    g = FabricGraph()
    g.add(Node("db-write", NodeKind.TOOL, privileged=True))
    g.add(Node("scraper", NodeKind.TOOL, produces_untrusted=True))
    g.connect("db-write", "scraper")   # untrusted content flows AWAY from privilege
    result = analyse(g)
    assert not result.injection_paths
