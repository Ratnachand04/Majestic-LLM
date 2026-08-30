"""Refusal quality, measured without inflating it (Part 2 §16, §2.2).

§16 is a correction to how the flagship claim must be evaluated. The failure it
guards against is silent and flattering: an aggregate precision number that
approaches 1 as you add specs whose refusals were never in question. These tests
assert the guard works, and that the honest number is the one reported.
"""
from __future__ import annotations

import pytest

from modelrig.planner import default_catalog
from modelrig.planner.audit import (
    C2_PRECISION_TARGET,
    PrecisionEstimate,
    RefusalAudit,
    RefusalOutcome,
    enumeration_economy,
    plan_space_size,
    summarise_witnesses,
)
from modelrig.planner.predicates import HARD, SOFT
from modelrig.planner.refusal import Refusal


def _audit(hard: int = 0, soft: int = 0, soft_correct: int = 0,
           predicate: str = "P_seed") -> RefusalAudit:
    a = RefusalAudit()
    for i in range(hard):
        a.add(RefusalOutcome(f"h{i}", ("P_ram",), would_have_failed=True))
    for i in range(soft):
        a.add(RefusalOutcome(f"s{i}", (predicate,), would_have_failed=i < soft_correct))
    return a


# =========================================================================== #
# §16 — the partition
# =========================================================================== #
def test_hard_and_soft_predicates_are_disjoint_and_complete():
    from modelrig.planner.predicates import ALL_PREDICATES

    assert set(HARD) & set(SOFT) == set()
    assert set(HARD) | set(SOFT) == {p.name for p in ALL_PREDICATES}


def test_the_hard_predicates_are_the_ones_sound_by_construction():
    """A model that does not fit does not fit; undefined KL is undefined; an
    illegal licence chain is illegal; reachability over a finite graph is exact."""
    for name in ("P_ram", "P_tok", "P_lic", "P_off"):
        assert name in HARD


def test_a_refusal_any_hard_predicate_carries_is_certain():
    mixed = RefusalOutcome("x", ("P_ram", "P_seed"), would_have_failed=True)
    assert mixed.carried_by_hard is True
    assert mixed.soft_only is False        # the hard one settles it


def test_only_soft_only_refusals_inform_calibration():
    soft = RefusalOutcome("x", ("P_seed", "P_lat"), would_have_failed=True)
    assert soft.soft_only is True
    assert soft.soft_fired == ("P_seed", "P_lat")


# =========================================================================== #
# §16 — the aggregate is inflatable, and the audit shows it
# =========================================================================== #
def test_padding_with_hard_refusals_inflates_the_aggregate():
    """The exact failure §16 exists to prevent: adding memory-infeasible specs
    drives aggregate precision toward 1 while measuring nothing."""
    lean = _audit(hard=0, soft=10, soft_correct=6)
    padded = _audit(hard=40, soft=10, soft_correct=6)

    assert lean.aggregate_precision().precision == pytest.approx(0.60)
    assert padded.aggregate_precision().precision == pytest.approx(0.92)
    # ...while the honest number is unmoved, because nothing new was measured.
    assert padded.soft_precision().precision == pytest.approx(0.60)
    assert padded.inflation() == pytest.approx(0.32)


def test_the_headline_is_the_soft_number():
    audit = _audit(hard=40, soft=10, soft_correct=6)
    report = audit.report()
    assert report["primary"]["label"] == "soft-only (primary)"
    assert report["primary"]["precision"] == pytest.approx(0.60)
    # The aggregate is still computed — so it can be shown to be inflated next
    # to the honest figure — but both its key and its label say not to quote it.
    assert "do_not_report" in "".join(report)
    assert "INFLATED" in report["aggregate_do_not_report"]["label"]


def test_hard_precision_is_stated_never_computed():
    audit = _audit(hard=10, soft=10, soft_correct=5)
    hard = audit.hard_precision_is_unmeasurable()
    assert hard["precision"] == 1.0
    assert hard["measured"] is False
    assert "No experiment can inform this number" in hard["why"]
    assert hard["fired"]["P_ram"] == 10


# =========================================================================== #
# §16 step 3 — the corpus-composition guard
# =========================================================================== #
def test_a_hard_dominated_corpus_is_refused_as_unusable():
    audit = _audit(hard=40, soft=10, soft_correct=6)
    warnings = audit.corpus_warnings()
    assert audit.report()["usable"] is False
    assert any("measures corpus composition rather than calibration" in w
               for w in warnings)


def test_a_corpus_with_no_soft_refusals_measures_nothing():
    audit = _audit(hard=25, soft=0)
    assert audit.soft_precision().precision is None
    assert any("no soft predicate fired anywhere" in w
               for w in audit.corpus_warnings())


def test_an_empty_corpus_concludes_nothing():
    assert RefusalAudit().corpus_warnings() == [
        "the corpus is empty: nothing can be concluded"
    ]


def test_a_well_designed_corpus_passes_the_guard():
    """§16 step 3: specs that are memory-feasible, licence-clean and
    tokenizer-consistent, but marginal on seed sufficiency or quality."""
    audit = _audit(hard=10, soft=60, soft_correct=45)
    assert audit.soft_share > 0.5
    assert audit.report()["usable"] is True
    assert audit.report()["corpus_warnings"] == []


def test_a_thin_soft_corpus_is_flagged_even_when_it_dominates():
    audit = _audit(hard=1, soft=12, soft_correct=9)
    assert audit.soft_share > 0.5
    assert any("too few to separate" in w for w in audit.corpus_warnings())


# =========================================================================== #
# §16 step 4 — stratified by which predicate fired
# =========================================================================== #
def test_precision_is_reported_per_predicate():
    audit = RefusalAudit()
    for i in range(20):
        audit.add(RefusalOutcome(f"a{i}", ("P_seed",), would_have_failed=i < 16))
    for i in range(20):
        audit.add(RefusalOutcome(f"b{i}", ("P_lat",), would_have_failed=i < 8))

    by_pred = audit.by_predicate()
    assert by_pred["P_seed"].precision == pytest.approx(0.80)
    assert by_pred["P_lat"].precision == pytest.approx(0.40)
    # The aggregate hides the fact that one of them is barely better than chance.
    assert audit.soft_precision().precision == pytest.approx(0.60)


def test_hard_predicates_never_appear_in_the_stratified_table():
    audit = _audit(hard=20, soft=20, soft_correct=12)
    assert set(audit.by_predicate()) <= set(SOFT)


def test_a_refusal_carried_by_both_is_excluded_from_the_soft_table():
    """Its refusal was certain regardless, so it says nothing about the soft
    predicate that happened to fire alongside."""
    audit = RefusalAudit()
    audit.add(RefusalOutcome("x", ("P_ram", "P_seed"), would_have_failed=True))
    assert audit.by_predicate() == {}
    assert audit.soft_precision().total == 0


# =========================================================================== #
# Intervals — a point estimate is not evidence
# =========================================================================== #
def test_a_point_estimate_at_the_bar_does_not_clear_it():
    """0.60 over n=10 is not evidence of clearing a 0.60 bar."""
    thin = PrecisionEstimate(6, 10)
    assert thin.precision == pytest.approx(C2_PRECISION_TARGET)
    assert thin.clears() is False
    lo, _hi = thin.wilson()
    assert lo < C2_PRECISION_TARGET


def test_enough_evidence_does_clear_the_bar():
    strong = PrecisionEstimate(160, 200)
    assert strong.clears() is True
    assert strong.wilson()[0] > C2_PRECISION_TARGET


def test_the_interval_is_well_behaved_at_the_extremes():
    lo, hi = PrecisionEstimate(5, 5).wilson()
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 1.0                       # five for five is not certainty
    assert PrecisionEstimate(0, 0).wilson() is None


def test_the_rendered_report_keeps_the_two_claims_apart():
    text = _audit(hard=40, soft=10, soft_correct=6).render()
    assert "the hard predicates are checked at all" in text
    assert "sound by construction, precision 1, unmeasurable" in text
    assert "must not be reported as the headline" in text
    assert "CORPUS WARNINGS" in text


# =========================================================================== #
# §2.2 — the structural claim, computed rather than asserted
# =========================================================================== #
def test_the_plan_space_is_enumerable():
    """The single fact the whole algorithm rests on. Computed against the live
    catalogue, because a catalogue that grew by orders of magnitude would
    quietly invalidate the argument for rejecting search."""
    space = plan_space_size()
    assert space["enumerable"] is True
    assert 5.0 <= space["log10"] <= 7.0        # the 1e6 regime, not NAS's 1e18
    assert space["nas_log10"] - space["log10"] > 10


def test_every_plan_dimension_is_non_empty():
    for name, size in plan_space_size()["dimensions"].items():
        assert size >= 1, name


def test_the_ordering_saves_most_of_the_predicate_evaluations():
    """§4: early exit under c/(1-rho) ordering. The saving is the gap between
    evaluations actually performed and checking every predicate."""
    economy = enumeration_economy(considered=144, predicate_evaluations=480)
    assert economy["evaluations_per_candidate"] == pytest.approx(3.33, abs=0.01)
    assert economy["evaluations_per_candidate"] < economy["without_early_exit"]
    assert economy["saving"] > 0.3


def test_the_economy_is_undefined_with_nothing_considered():
    assert enumeration_economy(0, 0)["evaluations_per_candidate"] is None


# =========================================================================== #
# Corpus design support
# =========================================================================== #
def test_witnesses_are_summarised_by_soundness():
    refusals = [
        Refusal(spec_hash="a", witness=["P_ram"], hard_failures=["P_ram"]),
        Refusal(spec_hash="b", witness=["P_seed"], soft_failures=["P_seed"]),
        Refusal(spec_hash="c", witness=["P_seed"], soft_failures=["P_seed"]),
    ]
    summary = summarise_witnesses(refusals)
    assert summary["hard"] == {"P_ram": 1}
    assert summary["soft"] == {"P_seed": 2}
    assert summary["soft_only"] == 2


def test_an_audit_can_be_built_straight_from_planner_refusals():
    refusal = Refusal(spec_hash="abc", witness=["P_seed"], soft_failures=["P_seed"])
    audit = RefusalAudit()
    audit.record(refusal, would_have_failed=True)
    assert audit.soft_only[0].spec_hash == "abc"
    assert audit.soft_precision().precision == pytest.approx(1.0)


def test_the_catalogue_is_narrow_enough_to_enumerate():
    """A-04's narrow catalogue is also what keeps §2.2 true."""
    assert len(default_catalog().bases()) <= 8
