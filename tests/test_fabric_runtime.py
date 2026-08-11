"""Tests for the Fabric runtime, context assembly and the ReAct loop.

Covers B-10 (the typed DAG actually runs), A-06 (Lost in the Middle) and
A-09 (constrained tool calls).
"""
from __future__ import annotations

import pytest

from majestic.agent import ReActLoop, ToolRegistry, ToolSpec, parse_tool_call
from majestic.fabric import FabricGraph, FabricRuntime, Node, NodeKind, split_offline_core
from majestic.retrieval.context import Passage, assemble, boundary_order


# --- the Fabric runtime (B-10) -------------------------------------------- #
def _pipeline_graph() -> FabricGraph:
    g = FabricGraph("flow")
    g.add(Node("extract", NodeKind.CARTRIDGE, ram_mb=30))
    g.add(Node("classify", NodeKind.CARTRIDGE, ram_mb=30))
    g.connect("extract", "classify")
    return g


def test_graph_executes_in_topological_order():
    graph = _pipeline_graph()
    runtime = FabricRuntime(graph, {
        "extract": lambda x, ctx: f"{x}|extracted",
        "classify": lambda x, ctx: f"{x}|classified",
    })
    result = runtime.run("doc")
    assert result.ok is True
    assert result.output == "doc|extracted|classified"
    assert [t.name for t in result.trace] == ["extract", "classify"]


def test_unverified_graph_is_refused_before_running():
    """Analysis is a precondition for execution, not a report about it."""
    g = _pipeline_graph()
    g.add(Node("notify", NodeKind.TOOL, requires_network=True))
    g.connect("classify", "notify")
    ran = []
    runtime = FabricRuntime(
        g, {"extract": lambda x, c: ran.append("e") or x}, offline=True
    )
    result = runtime.run("doc")
    assert result.ok is False
    assert ran == []                      # nothing executed at all
    assert any("offline closure" in v for v in result.violations)


def test_offline_hard_lock_is_enforced_at_runtime():
    g = _pipeline_graph()
    g.add(Node("notify", NodeKind.TOOL, requires_network=True))
    g.connect("classify", "notify")
    runtime = FabricRuntime(
        g,
        {"extract": lambda x, c: x, "classify": lambda x, c: x,
         "notify": lambda x, c: "SENT"},
        offline=True, verify=False,       # skip preflight to prove runtime enforcement
    )
    result = runtime.run("doc")
    assert result.ok is False
    assert result.node("notify").skipped is True
    assert result.node("notify").output is None


def test_taint_propagates_and_privilege_refuses_it():
    g = FabricGraph()
    g.add(Node("scrape", NodeKind.TOOL, produces_untrusted=True))
    g.add(Node("summarise", NodeKind.CARTRIDGE))
    g.add(Node("pay", NodeKind.TOOL, privileged=True))
    g.connect("scrape", "summarise")
    g.connect("summarise", "pay")
    paid = []
    runtime = FabricRuntime(
        g,
        {"scrape": lambda x, c: "<script>", "summarise": lambda x, c: x,
         "pay": lambda x, c: paid.append(x)},
        verify=False,
    )
    result = runtime.run(None)
    assert result.ok is False
    assert paid == []                     # the privileged tool never fired
    assert any("untrusted" in v for v in result.violations)


def test_node_errors_are_surfaced_not_swallowed():
    graph = _pipeline_graph()
    runtime = FabricRuntime(graph, {
        "extract": lambda x, c: (_ for _ in ()).throw(RuntimeError("boom")),
    })
    result = runtime.run("doc")
    assert result.ok is False
    assert any("boom" in v for v in result.violations)


def test_cost_and_timing_are_accumulated():
    g = FabricGraph()
    g.add(Node("cloud", NodeKind.TOOL, cost_usd=0.002))
    runtime = FabricRuntime(g, {"cloud": lambda x, c: "ok"})
    result = runtime.run(None)
    assert result.cost_usd == pytest.approx(0.002)
    assert result.total_ms >= 0


def test_split_offline_core_names_the_tail():
    g = _pipeline_graph()
    g.add(Node("notify", NodeKind.TOOL, requires_network=True))
    g.connect("classify", "notify")
    core, tail = split_offline_core(g)
    assert core == ["extract", "classify"]
    assert tail == ["notify"]


# --- context assembly (A-06) ---------------------------------------------- #
def _passages(n: int = 5) -> list[Passage]:
    return [Passage(text=f"passage {i}", score=1.0 - i * 0.1, source=f"doc{i}")
            for i in range(n)]


def test_strongest_evidence_sits_at_the_boundaries():
    """Evidence placed mid-context is largely ignored (2307.03172)."""
    ordered = boundary_order(_passages(5))
    scores = [p.score for p in ordered]
    assert scores[0] == max(scores)
    assert scores[-1] == sorted(scores, reverse=True)[1]
    assert scores[len(scores) // 2] == min(scores)


def test_chunks_are_capped_aggressively():
    ctx = assemble(_passages(20), max_chunks=5)
    assert len(ctx.passages) == 5
    assert len(ctx.dropped) == 15


def test_token_budget_is_a_budget_not_a_target():
    ctx = assemble(_passages(10), max_chunks=10, token_budget=6)
    assert ctx.approx_tokens <= 6
    assert ctx.dropped


def test_weak_passages_are_filtered():
    passages = [Passage("strong", 0.9), Passage("weak", 0.01)]
    ctx = assemble(passages, min_score=0.1)
    assert [p.text for p in ctx.passages] == ["strong"]


def test_citations_let_the_customer_audit_the_answer():
    ctx = assemble(_passages(3))
    assert len(ctx.citations) == 3
    assert ctx.citations[0]["source"]
    assert "[1]" in ctx.render()


# --- ReAct with constrained tool calls (A-09) ----------------------------- #
def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec("search", lambda q: f"results for {q}",
                          args={"q": "string"}, requires_network=True,
                          produces_untrusted=True))
    reg.register(ToolSpec("calc", lambda expr: str(eval(expr, {"__builtins__": {}}))),  # noqa: S307
                 )
    reg.register(ToolSpec("pay", lambda amount: f"paid {amount}", privileged=True))
    return reg


def test_free_form_output_is_rejected():
    """A cartridge emits a constrained call, never free-form reasoning."""
    with pytest.raises(ValueError, match="free-form"):
        parse_tool_call("I think I should search for cats")


def test_unregistered_tool_is_rejected():
    with pytest.raises(ValueError, match="unregistered"):
        parse_tool_call('{"tool": "rm_rf", "args": {}}', _registry())


def test_valid_constrained_call_parses():
    call = parse_tool_call('{"tool": "calc", "args": {"expr": "2+2"}}', _registry())
    assert call.tool == "calc" and call.args == {"expr": "2+2"}


def test_loop_runs_until_it_answers():
    reg = _registry()
    script = ['{"tool": "calc", "args": {"expr": "6*7"}}', "ANSWER: 42"]
    loop = ReActLoop(reg, lambda req, steps: script[len(steps)], max_steps=3)
    result = loop.run("what is 6*7")
    assert result.completed is True
    assert result.answer == "42"
    assert result.steps[0].observation == "42"


def test_step_depth_is_capped_by_the_device_tier():
    """Sub-2B models degrade badly across multi-step loops."""
    reg = _registry()
    loop = ReActLoop(reg, lambda req, steps: '{"tool": "calc", "args": {"expr": "1+1"}}',
                     max_steps=2)
    result = loop.run("loop forever")
    assert result.completed is False
    assert result.depth == 2
    assert "step depth" in result.refusal


def test_offline_loop_refuses_a_network_tool():
    reg = _registry()
    loop = ReActLoop(reg, lambda req, steps: '{"tool": "search", "args": {"q": "x"}}',
                     offline=True)
    result = loop.run("find something")
    assert result.completed is False
    assert "requires the network" in result.refusal


def test_untrusted_observation_cannot_reach_a_privileged_tool():
    reg = _registry()
    script = ['{"tool": "search", "args": {"q": "invoice"}}',
              '{"tool": "pay", "args": {"amount": 100}}']
    loop = ReActLoop(reg, lambda req, steps: script[len(steps)], max_steps=3)
    result = loop.run("pay the invoice")
    assert result.completed is False
    assert "privileged" in result.refusal


def test_tool_manifest_feeds_cartridge_slot_three():
    manifest = _registry().manifest()
    assert any(t["requires_network"] for t in manifest)
    assert any(t["privileged"] for t in manifest)
    assert _registry().offline_safe() is False
