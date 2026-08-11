"""Tests for parallel candidates and the k-bit base rule (A-01, B-01, C-02)."""
from __future__ import annotations

from modelrig.candidates import CandidateResult, Selection, build_candidates, select
from modelrig.catalogue import DEFAULT_CATALOGUE, ladder_tier
from modelrig.feasibility import YamlDeviceProfiler
from modelrig.ir import BuildPlanIR, DataRights, SpecIR
from modelrig.planner import Planner
from modelrig.primitives import TaskPrimitive
from modelrig.proving_ground import AxisResult, Scorecard

PROFILER = YamlDeviceProfiler("configs/devices.yaml")


def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT,
        device_target="laptop_cpu",
        seed_data_count=200,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.8,
    )
    base.update(over)
    return SpecIR(**base)


def _card(score: float, passed: bool = True) -> Scorecard:
    return Scorecard(
        axes=[AxisResult("task_metric", score, 0.8, passed)], passed=passed
    )


def _result(base_ref: str, score: float, latency: float, params_b: float,
            passed: bool = True) -> CandidateResult:
    return CandidateResult(
        plan=BuildPlanIR(spec_hash="s", base_ref=base_ref),
        scorecard=_card(score, passed), latency_ms=latency, params_b=params_b,
    )


# --- the k-bit scaling law (A-01) ---------------------------------------- #
def test_planner_picks_the_largest_base_that_fits_not_the_smallest():
    """Under a fixed RAM budget the correct move is the LARGEST base at 4-bit."""
    result = Planner(profiler=PROFILER).plan(_spec(device_target="laptop_cpu"))
    assert result.admitted
    chosen = DEFAULT_CATALOGUE.base(result.plan.base_ref)
    cap = Planner(profiler=PROFILER)._device_cap(_spec(device_target="laptop_cpu"))
    bigger_that_fit = [
        b for b in DEFAULT_CATALOGUE.bases
        if not b.is_moe and b.params_b <= cap
        and TaskPrimitive.EXTRACT in b.good_for
        and b.params_b > chosen.params_b
    ]
    assert bigger_that_fit == [], f"a larger base fitted but was not chosen: {bigger_that_fit}"


def test_catalogue_orders_largest_first_by_default():
    bases = DEFAULT_CATALOGUE.bases_for(TaskPrimitive.EXTRACT)
    sizes = [b.params_b for b in bases]
    assert sizes == sorted(sizes, reverse=True)


def test_plan_records_the_rule_and_ladder_tier():
    plan = Planner(profiler=PROFILER).plan(_spec()).plan
    assert "largest base that fits" in plan.provenance["rule"]
    assert plan.provenance["ladder_tier"]
    assert ladder_tier(1.7) == "4-6 GB RAM tier"


# --- MoE exclusion (A-07) -------------------------------------------------- #
def test_moe_is_excluded_from_memory_constrained_targets():
    """Sparse activation is not sparse residency."""
    result = Planner(profiler=PROFILER).plan(_spec(device_target="android_midrange"))
    base = DEFAULT_CATALOGUE.base(result.plan.base_ref)
    assert base.is_moe is False


def test_moe_entry_declares_the_residency_trap():
    moe = next(b for b in DEFAULT_CATALOGUE.bases if b.is_moe)
    assert moe.active_params_b < moe.params_b
    assert moe.resident_params_b == moe.params_b   # all of it must be held


def test_bases_for_excludes_moe_by_default():
    assert all(not b.is_moe for b in DEFAULT_CATALOGUE.bases_for(TaskPrimitive.EXTRACT))


# --- parallel candidates (B-01, C-02 acts 5-6) --------------------------- #
def test_candidate_plans_are_ordered_largest_first():
    plans = Planner(profiler=PROFILER).candidate_plans(_spec(), limit=2)
    assert len(plans) == 2
    sizes = [DEFAULT_CATALOGUE.base(p.base_ref).params_b for p in plans]
    assert sizes[0] > sizes[1]


def test_build_candidates_runs_every_plan():
    plans = [BuildPlanIR(spec_hash="s", base_ref=f"b{i}") for i in range(3)]
    seen = []

    def build(plan):
        seen.append(plan.base_ref)
        return CandidateResult(plan=plan, scorecard=_card(0.9))

    results = build_candidates(plans, build)
    assert len(results) == 3
    assert sorted(seen) == ["b0", "b1", "b2"]


def test_one_failing_candidate_does_not_sink_the_batch():
    plans = [BuildPlanIR(spec_hash="s", base_ref="ok"),
             BuildPlanIR(spec_hash="s", base_ref="boom")]

    def build(plan):
        if plan.base_ref == "boom":
            raise RuntimeError("out of memory")
        return CandidateResult(plan=plan, scorecard=_card(0.9))

    results = build_candidates(plans, build)
    assert len(results) == 2
    assert any(r.error for r in results)
    assert any(r.passed for r in results)


# --- selection: the C-02 act-6 trade -------------------------------------- #
def test_latency_budget_beats_a_higher_score():
    """C-02: 4B scores 0.95 at 4.1s, 1.7B scores 0.94 at 1.6s — 1.7B wins."""
    results = [
        _result("Qwen/Qwen3-4B", 0.95, 4100, 4.0),
        _result("Qwen/Qwen3-1.7B", 0.94, 1600, 1.7),
    ]
    selection = select(results, _spec(latency_budget_ms=2000))
    assert selection.winner.plan.base_ref == "Qwen/Qwen3-1.7B"
    assert "broke the budget" in selection.rationale


def test_highest_score_wins_when_all_are_within_budget():
    results = [
        _result("Qwen/Qwen3-4B", 0.95, 900, 4.0),
        _result("Qwen/Qwen3-1.7B", 0.90, 600, 1.7),
    ]
    assert select(results, _spec(latency_budget_ms=2000)).winner.params_b == 4.0


def test_ties_break_toward_the_smaller_base():
    results = [
        _result("Qwen/Qwen3-4B", 0.94, 900, 4.0),
        _result("Qwen/Qwen3-1.7B", 0.94, 600, 1.7),
    ]
    selection = select(results, _spec())
    assert selection.winner.params_b == 1.7
    assert "larger" in selection.rationale


def test_failing_candidates_are_never_offered():
    results = [_result("a", 0.99, 100, 1.7, passed=False)]
    selection = select(results, _spec())
    assert selection.chosen is False
    assert "no candidate cleared its gate" in selection.rationale


def test_all_over_budget_is_a_refusal_not_a_compromise():
    results = [_result("a", 0.99, 9000, 1.7)]
    selection = select(results, _spec(latency_budget_ms=1000))
    assert selection.chosen is False
    assert "contract, not a preference" in selection.rationale


def test_comparison_table_is_side_by_side_for_the_scorecard():
    results = [
        _result("Qwen/Qwen3-4B", 0.95, 4100, 4.0),
        _result("Qwen/Qwen3-1.7B", 0.94, 1600, 1.7),
    ]
    table = select(results, _spec(latency_budget_ms=2000)).comparison_table()
    assert len(table) == 2
    assert sum(1 for row in table if row["winner"]) == 1


def test_empty_selection_is_honest():
    assert Selection().chosen is False
    assert select([], _spec()).chosen is False
