"""Tests for learned deferral (A-08) and the flywheel (B-10)."""
from __future__ import annotations

from pathlib import Path

from majestic.flywheel import Correction, Flywheel
from majestic.router.deferral import DeferralRouter, LearnedDeferralRule

VOCAB = ["invoice", "total", "amount", "date", "vendor", "tax"]


# --- the learned deferral rule -------------------------------------------- #
def test_low_confidence_escalates_high_confidence_does_not():
    router = DeferralRouter(vocabulary=VOCAB)
    confident = router.decide("invoice total amount", confidence=0.95)
    unsure = router.decide("wholly unrelated gibberish xyzzy", confidence=0.05)
    assert confident.escalate is False
    assert unsure.escalate is True
    assert unsure.predicted_quality_gap > confident.predicted_quality_gap


def test_distribution_shift_raises_the_predicted_gap():
    """OOV ratio is the local distribution-shift signal, not raw confidence."""
    rule = LearnedDeferralRule()
    in_dist = rule.features("invoice total amount", 0.6, VOCAB)
    out_dist = rule.features("zzz qqq vvv", 0.6, VOCAB)
    assert rule.predict_gap(out_dist) > rule.predict_gap(in_dist)


def test_offline_mode_hard_locks_escalation():
    """A customer who bought offline gets offline — architecturally."""
    router = DeferralRouter(offline_required=True, vocabulary=VOCAB)
    decision = router.decide("total gibberish xyzzy", confidence=0.01)
    assert decision.escalate is False
    assert decision.locked_offline is True
    assert "architecturally disabled" in decision.reason


def test_escalation_budget_is_a_business_control():
    router = DeferralRouter(vocabulary=VOCAB, max_escalation_rate=0.2)
    for _ in range(10):
        router.decide("zzz qqq vvv unknown", confidence=0.01)
    assert router.escalation_rate <= 0.35   # the knob caps cloud spend


def test_every_escalation_is_logged_as_a_hard_case():
    router = DeferralRouter(vocabulary=VOCAB)
    router.decide("zzz qqq vvv", confidence=0.01, capability="extract")
    cases = router.drain_hard_cases()
    assert len(cases) == 1
    assert cases[0].capability == "extract"
    assert router.hard_cases == []          # drained into the flywheel


def test_rule_learns_from_observed_outcomes():
    router = DeferralRouter(vocabulary=VOCAB)
    before = router.rule.predict_gap(router.rule.features("invoice total", 0.9, VOCAB))
    for _ in range(30):
        router.observe("invoice total", confidence=0.9, cloud_was_better=True)
    after = router.rule.predict_gap(router.rule.features("invoice total", 0.9, VOCAB))
    assert after > before
    assert router.rule.n_updates == 30


def test_rule_persists(tmp_path: Path):
    rule = LearnedDeferralRule()
    rule.update(rule.features("x", 0.5), True)
    rule.save(tmp_path / "rule.json")
    assert LearnedDeferralRule.load(tmp_path / "rule.json").n_updates == 1


# --- the flywheel ---------------------------------------------------------- #
def _flywheel_with(n: int = 25) -> Flywheel:
    fly = Flywheel(min_corrections=20)
    for i in range(n):
        fly.record(
            Correction(query=f"q{i}", model_output="wrong", approved=False,
                       corrected_output=f"right{i}")
        )
    return fly


def test_corrections_queue_locally_then_sync():
    fly = _flywheel_with()
    assert fly.ready is False               # nothing synced yet
    assert fly.sync() == 25
    assert fly.ready is True


def test_kto_dataset_splits_binary_signal():
    fly = _flywheel_with(4)
    fly.record(Correction("good", "fine", approved=True))
    fly.sync()
    desirable, undesirable = fly.kto_dataset()
    assert len(desirable) == 5              # corrections + approvals
    assert len(undesirable) == 4


def test_held_out_grows_with_real_corrections():
    fly = _flywheel_with()
    fly.sync()
    grown = fly.grow_held_out([("original", "gold")])
    assert len(grown) == 26                 # the bar gets harder each generation


def test_new_version_offered_only_if_it_wins():
    fly = _flywheel_with()
    fly.sync()
    fly.current_score = 0.90

    worse = fly.propose_next_version(lambda train, held: 0.85, [("a", "b")])
    assert worse.accepted is False
    assert fly.current_version == 1         # customer never gets a regression

    better = fly.propose_next_version(lambda train, held: 0.95, [("a", "b")])
    assert better.accepted is True
    assert fly.current_version == 2
    assert fly.current_score == 0.95


def test_degradation_across_generations_is_detected():
    """GAP-05: the flywheel could silently degrade over years."""
    fly = Flywheel(min_corrections=1)
    for score in (0.9, 0.8, 0.7):
        fly.current_score = 0.0             # force acceptance to log the trend
        fly.propose_next_version(lambda t, h, s=score: s, [("a", "b")])
    assert fly.log.degrading() is True


def test_flywheel_persists(tmp_path: Path):
    fly = _flywheel_with()
    fly.sync()
    assert fly.save(tmp_path / "fly.json").exists()
