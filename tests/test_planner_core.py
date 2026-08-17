"""Tests for base selection, enumeration, refusal and the decision rule.

Covers §2 (enumerability), §5 (objective), §6 (the threshold), §7 (the
algorithm), §12 (max over a chain), §15 (parallel candidates) and the §8 worked
example end to end.
"""
from __future__ import annotations

import pytest

from modelrig.ir import DataRights, SpecIR
from modelrig.planner import (
    OutcomePredictor,
    PrecedentIndex,
    Tier,
    default_catalog,
    enumerate_feasible,
    minimal_cover,
    plan,
    refusal_threshold,
    select_base,
    should_build,
    spec_shape,
    verify_antitone,
)
from modelrig.planner.costmodel import USD
from modelrig.planner.metalearn import Observation, features_of
from modelrig.planner.objective import DecisionParams, quality_prior, utility
from modelrig.planner.predicates import PredicateResult, memory
from modelrig.planner.refusal import WitnessSet, build_refusal
from modelrig.primitives import TaskPrimitive

CAT = default_catalog()


def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT,
        device_target="android_tablet_4gb",
        seed_data_count=200,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.9,
    )
    base.update(over)
    return SpecIR(**base)


# =========================================================================== #
# §12 — base selection is a maximum over a chain
# =========================================================================== #
def test_feasible_bases_form_a_chain():
    """P_ram is antitone in size: if a 4B fits, every smaller base fits."""
    for device in ("android_lowend", "android_tablet_4gb", "laptop_cpu"):
        assert verify_antitone(_spec(device_target=device), CAT), device


def test_select_base_returns_the_maximum_of_that_chain():
    spec = _spec(device_target="android_tablet_4gb")
    chosen = select_base(spec, CAT)
    assert chosen is not None
    device = CAT.device(spec.device_target)
    for model in CAT.bases(TaskPrimitive.EXTRACT):
        fits = memory(model, CAT, quantiser="q4_k_m", context=2048,
                      device=device).total <= device.free_ram
        if model.params > chosen.params:
            assert not fits, f"{model.ref} fits but a smaller base was chosen"


def test_bigger_device_admits_a_bigger_base():
    small = select_base(_spec(device_target="android_lowend"), CAT)
    large = select_base(_spec(device_target="laptop_cpu"), CAT)
    assert large is not None
    assert small is None or large.params > small.params


def test_moe_never_enters_the_chain_by_default():
    """Sparse activation is not sparse residency (A-07)."""
    assert all(not m.is_moe for m in CAT.bases(TaskPrimitive.EXTRACT))
    chosen = select_base(_spec(device_target="laptop_cpu"), CAT)
    assert chosen is not None and chosen.is_moe is False


def test_select_base_returns_none_when_nothing_fits():
    """Returning None is information, not an error."""
    tiny = _spec(device_target="android_lowend",
                 io_schema={"context_length": 32768})
    assert select_base(tiny, CAT) is None


# =========================================================================== #
# §2 — the space is enumerable, and pruning makes it cheap
# =========================================================================== #
def test_enumeration_prunes_hard_and_early():
    enum = enumerate_feasible(_spec(), CAT)
    assert enum.considered > 0
    # Early exit means far fewer evaluations than 7 per candidate.
    assert enum.predicate_evaluations < 7 * enum.considered


def test_enumeration_terminates_quickly():
    """Sub-second on a laptop is the claim; assert the work is bounded."""
    enum = enumerate_feasible(_spec(), CAT)
    assert enum.considered < 5_000


def test_every_feasible_candidate_passes_every_predicate():
    from modelrig.planner.predicates import ordered_predicates

    enum = enumerate_feasible(_spec(), CAT)
    assert enum.feasible
    for candidate in enum.feasible[:20]:
        for p in ordered_predicates():
            assert p(_spec(), candidate, CAT).ok, f"{p.name} on {candidate.ident}"


# =========================================================================== #
# §7 line 16 — minimal cover is the product
# =========================================================================== #
def _w(predicate: str, candidate: str) -> PredicateResult:
    return PredicateResult(False, predicate, reason=f"{predicate} failed", remedy="fix it")


def test_minimal_cover_finds_the_smallest_explanation():
    ws = WitnessSet()
    for c in ("a", "b", "c"):
        ws.add(c, _w("P_ram", c))
    for c in ("d", "e"):
        ws.add(c, _w("P_lat", c))
    # A predicate that explains nothing new must not appear.
    ws.add("a", _w("P_cost", "a"))
    cover = minimal_cover(ws)
    assert set(cover) == {"P_ram", "P_lat"}


def test_cover_collapses_redundant_witnesses():
    """Forty redundant complaints become one."""
    ws = WitnessSet()
    for i in range(200):
        ws.add(f"cand{i}", _w("P_ram", f"cand{i}"))
    assert minimal_cover(ws) == ["P_ram"]


def test_cover_prefers_hard_predicates_on_a_tie():
    ws = WitnessSet()
    for c in ("a", "b"):
        ws.add(c, _w("P_ram", c))     # hard
        ws.add(c, _w("P_seed", c))    # soft, same coverage
    assert minimal_cover(ws)[0] == "P_ram"


def test_empty_witness_set_covers_nothing():
    assert minimal_cover(WitnessSet()) == []


def test_refusal_renders_something_a_customer_can_act_on():
    ws = WitnessSet()
    ws.add("x", _w("P_ram", "x"))
    refusal = build_refusal("hash", ws, considered=12, extra_notes=["unmeasured device"])
    text = refusal.render()
    assert "REFUSED" in text and "P_ram" in text
    assert "Remedies" in text and "unmeasured device" in text
    assert refusal.sound is True                # carried by a hard predicate


# =========================================================================== #
# §6 — the decision threshold
# =========================================================================== #
def test_threshold_matches_the_worked_figures():
    """C=$40, V=$300, kappa=$200 gives theta* = 240/500 = 0.48."""
    params = DecisionParams(value=300 * USD, build_cost=40 * USD, trust_damage=200 * USD)
    assert params.threshold() == pytest.approx(0.48)


def test_trust_damage_drives_the_threshold_toward_one():
    """A system whose failures are catastrophic should refuse what it is unsure of."""
    low = DecisionParams(value=300 * USD, build_cost=40 * USD, trust_damage=10 * USD)
    high = DecisionParams(value=300 * USD, build_cost=40 * USD, trust_damage=100_000 * USD)
    assert low.threshold() < high.threshold()
    assert high.threshold() > 0.99


def test_even_a_free_build_can_be_refused():
    """As C -> 0, theta* -> kappa/(V+kappa), still bounded away from zero.

    This is why "just build it and see" is wrong even where marginal cost is
    zero: the damage of shipping a bad model is not the compute you burned.
    """
    free = DecisionParams(value=300 * USD, build_cost=0, trust_damage=200 * USD)
    assert free.threshold() == pytest.approx(200 / 500)
    assert should_build(0.30, build_cost=0, tier=Tier.COMMERCIAL) is False


def test_expected_value_crosses_zero_exactly_at_the_threshold():
    params = DecisionParams(value=300 * USD, build_cost=40 * USD, trust_damage=200 * USD)
    theta = params.threshold()
    assert params.expected_value(theta) == pytest.approx(0, abs=USD // 100)
    assert params.expected_value(theta + 0.1) > 0
    assert params.expected_value(theta - 0.1) < 0


def test_regulated_tier_is_far_more_conservative():
    cost = 40 * USD
    assert refusal_threshold(cost, Tier.REGULATED) > refusal_threshold(cost, Tier.COMMERCIAL)
    assert refusal_threshold(cost, Tier.COMMERCIAL) > refusal_threshold(cost, Tier.EXPERIMENTAL)


# =========================================================================== #
# §5 — the objective
# =========================================================================== #
def test_quality_dominates_the_utility():
    better = utility(quality=0.9, cost_micro=40 * USD, cost_ceiling_micro=40 * USD,
                     latency_ms=1000, latency_budget_ms=1000,
                     weight_bytes=10, weight_ceiling_bytes=10)
    worse = utility(quality=0.5, cost_micro=0, cost_ceiling_micro=40 * USD,
                    latency_ms=0, latency_budget_ms=1000,
                    weight_bytes=0, weight_ceiling_bytes=10)
    assert better.total > worse.total


def test_size_breaks_ties_toward_the_smaller_artefact():
    big = utility(quality=0.8, cost_micro=0, cost_ceiling_micro=USD, latency_ms=0,
                  latency_budget_ms=None, weight_bytes=10, weight_ceiling_bytes=10)
    small = utility(quality=0.8, cost_micro=0, cost_ceiling_micro=USD, latency_ms=0,
                    latency_budget_ms=None, weight_bytes=1, weight_ceiling_bytes=10)
    assert small.total > big.total


def test_quality_prior_is_monotone_in_the_things_that_matter():
    kw = dict(min_params_b=0.6, seed_floor=100, distil_mode="sequence_kd",
              bit_width_effective=5.2)
    assert quality_prior(params_b=4.0, seed_count=400, **kw) > \
           quality_prior(params_b=0.6, seed_count=400, **kw)
    assert quality_prior(params_b=1.7, seed_count=400, **kw) > \
           quality_prior(params_b=1.7, seed_count=100, **kw)


# =========================================================================== #
# metalearn — shrinkage
# =========================================================================== #
def test_learner_is_inert_with_no_history():
    predictor = OutcomePredictor()
    assert predictor.gamma == 0.0
    assert predictor.active is False
    assert predictor.predict(0.73, (1.0,) * 6) == 0.73     # prior carries it entirely


def test_gamma_rises_with_evidence():
    predictor = OutcomePredictor(h0=50)
    for i in range(50):
        predictor.record(Observation(f"s{i}", f"p{i}", features_of(
            params_b=1.7, seed_ratio=2.0, bits_effective=5.2,
            distil_mode="sequence_kd", context=2048, quality_gate=0.9,
        ), outcome=0.9, passed=True))
    assert predictor.gamma == pytest.approx(0.5)           # |H| = h0
    assert predictor.active is True


def test_learned_prediction_blends_rather_than_replaces():
    predictor = OutcomePredictor(h0=50, min_history=10)
    feats = features_of(params_b=1.7, seed_ratio=2.0, bits_effective=5.2,
                        distil_mode="sequence_kd", context=2048, quality_gate=0.9)
    for i in range(20):
        predictor.record(Observation(f"s{i}", f"p{i}", feats, outcome=0.2, passed=False))
    blended = predictor.predict(1.0, feats)
    assert 0.2 < blended < 1.0          # strictly between prior and learner


def test_history_round_trips(tmp_path):
    predictor = OutcomePredictor()
    predictor.record(Observation("s", "p", (1.0,) * 6, outcome=0.5))
    predictor.save(tmp_path / "h.json")
    restored = OutcomePredictor()
    restored.load(tmp_path / "h.json")
    assert len(restored) == 1


# =========================================================================== #
# §7 — the algorithm end to end
# =========================================================================== #
def test_plan_admits_deterministically_and_reproducibly():
    a = plan(_spec(), CAT)
    b = plan(_spec(), CAT)
    assert a.admitted and b.admitted
    assert a.path == "deterministic"
    assert a.plan.hash == b.plan.hash          # same spec, same catalogue, same plan


def test_precedent_reuse_short_circuits_enumeration():
    spec = _spec()
    index = PrecedentIndex()
    first = plan(spec, CAT, precedents=index)
    index.add(spec_shape(spec), first.plan, passed=True)
    second = plan(spec, CAT, precedents=index)
    assert second.path == "precedent"


def test_exploration_floor_occasionally_ignores_precedent():
    """§11 open question 3: precedent reuse can lock in early mistakes."""
    spec = _spec()
    index = PrecedentIndex(exploration_rate=0.5)
    index.add(spec_shape(spec), plan(spec, CAT).plan, passed=True)
    paths = {plan(spec, CAT, precedents=index).path for _ in range(6)}
    assert "deterministic" in paths            # the floor really fires


def test_refusal_is_a_return_value_not_an_exception():
    outcome = plan(_spec(seed_data_count=3), CAT)
    assert outcome.admitted is False
    assert outcome.refusal is not None
    assert "P_seed" in outcome.refusal.witness


def test_refusal_carries_a_witness_and_remedies():
    outcome = plan(_spec(data_rights=DataRights.UNKNOWN), CAT)
    assert outcome.refusal is not None
    assert outcome.refusal.witness
    assert outcome.refusal.remedies


def test_parallel_candidates_are_not_attached_unless_requested():
    """§15: never spend the customer's money to buy time they did not ask for."""
    assert plan(_spec(), CAT).candidates == []
    opted = plan(_spec(io_schema={"allow_parallel_candidates": True}), CAT)
    assert opted.admitted


# =========================================================================== #
# §8 — the worked example, reproduced
# =========================================================================== #
def _apex(**over) -> SpecIR:
    io = {"tokens_in": 500, "tokens_out": 60, "accept_unmeasured_latency": True}
    io.update(over.pop("io_schema", {}))
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT,
        device_target="android_tablet_4gb",
        offline_required=True,
        seed_data_count=150,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.93,
        latency_budget_ms=2000,
        io_schema=io,
    )
    base.update(over)
    return SpecIR(**base)


def test_apex_memory_matches_the_worked_example():
    """1.7B fits at ~2.9 GB; 4B does not."""
    device = CAT.device("android_tablet_4gb")
    small = memory(CAT.model("Qwen/Qwen3-1.7B"), CAT, quantiser="q4_k_m",
                   context=2048, device=device)
    big = memory(CAT.model("Qwen/Qwen3-4B"), CAT, quantiser="q4_k_m",
                 context=2048, device=device)
    assert small.kv_cache / 1e6 == pytest.approx(235, abs=1)
    assert small.total / 1e6 == pytest.approx(2903, abs=25)
    assert small.total <= device.free_ram
    assert big.total > device.free_ram


def test_apex_is_refused_on_latency_not_admitted():
    """13.8 s against a 2 s budget. Prose said 1.6 s; the arithmetic says otherwise."""
    outcome = plan(_apex(), CAT)
    assert outcome.admitted is False
    assert "P_lat" in outcome.refusal.witness
    assert any("prefill" in r for r in outcome.refusal.reasons)


def test_apex_refusal_names_the_unmeasured_device():
    outcome = plan(_apex(), CAT)
    assert any("unmeasured" in n for n in outcome.refusal.notes)


def test_apex_becomes_feasible_with_a_realistic_budget():
    """Remedy 1: extraction is not interactive; 15 s costs no quality."""
    outcome = plan(_apex(latency_budget_ms=20_000), CAT)
    assert outcome.admitted is True


def test_apex_refusal_costs_nothing():
    outcome = plan(_apex(), CAT)
    assert outcome.refusal.candidates_eliminated > 0
    assert outcome.plan is None                # no GPU was ever allocated
