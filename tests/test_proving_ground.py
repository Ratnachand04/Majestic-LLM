"""Tests for the PROVING GROUND — the gate and the repair loop (B-07)."""
from __future__ import annotations

import pytest

from modelrig.proving_ground import (
    FailureReport,
    ProvingGround,
    Repairer,
    answer_flip_rate,
    expected_calibration_error,
)

HELD_OUT = [("good great work", "positive"), ("bad awful thing", "negative"),
            ("love this a lot", "positive"), ("hate this a lot", "negative")]


def _perfect(texts):
    return ["positive" if "good" in t.lower() or "love" in t.lower() else "negative"
            for t in texts]


def _broken(texts):
    return ["positive" for _ in texts]


# --- answer-flip rate (A-05) --------------------------------------------- #
def test_answer_flip_rate_catches_changed_answers_at_equal_accuracy():
    """Aggregate parity is an illusion: same score, different answers."""
    reference = ["a", "b", "c", "d"]
    candidate = ["b", "a", "d", "c"]
    assert answer_flip_rate(reference, candidate) == 1.0
    assert answer_flip_rate(reference, reference) == 0.0


def test_expected_calibration_error():
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert expected_calibration_error([1.0, 1.0], [False, False]) == 1.0


# --- the seven axes ------------------------------------------------------- #
def test_all_seven_axes_run():
    card = ProvingGround(quality_gate=0.9).evaluate(_perfect, HELD_OUT)
    names = {a.name for a in card.axes}
    assert names == {
        "task_metric", "calibrated_judge", "behavioural", "regression",
        "safety", "privacy", "calibration",
    }
    assert len(card.axes) == 7


def test_gate_passes_a_good_model():
    card = ProvingGround(quality_gate=0.9).evaluate(_perfect, HELD_OUT)
    assert card.passed is True
    assert card.failure_report is None


def test_gate_fails_a_bad_model_and_reports_honest_failures():
    card = ProvingGround(quality_gate=0.9).evaluate(_broken, HELD_OUT)
    assert card.passed is False
    assert "task_metric" in card.failure_report.failed_axes
    assert card.honest_failures, "the scorecard must show honest failure cases"


def test_gate_fails_on_excessive_answer_flip():
    reference = ["negative", "positive", "negative", "positive"]  # all wrong-way
    card = ProvingGround(quality_gate=0.5).evaluate(
        _perfect, HELD_OUT, reference_predictions=reference, post_quantisation=True
    )
    assert card.answer_flip_rate == 1.0
    assert card.passed is False


def test_scorecard_flattens_into_the_gate3_report():
    card = ProvingGround(quality_gate=0.9).evaluate(_perfect, HELD_OUT)
    report = card.to_eval_report()
    for key in ("regression_passed", "safety_passed", "privacy_passed",
                "post_quantisation", "held_out_is_real", "ece"):
        assert key in report


def test_scorecard_carries_samples_for_the_customer():
    card = ProvingGround(quality_gate=0.9).evaluate(_perfect, HELD_OUT)
    assert card.sample_predictions          # what the customer actually reads
    assert card.n_held_out == len(HELD_OUT)


# --- the repairer's architectural constraint ------------------------------ #
def test_repairer_refuses_without_external_evidence():
    """Unaided self-reflection is forbidden (2310.01798)."""
    with pytest.raises(ValueError, match="structured failure report"):
        Repairer().repair(None)
    with pytest.raises(ValueError):
        Repairer().repair(FailureReport())


def test_repairer_suggests_mutations_from_a_failure_report():
    card = ProvingGround(quality_gate=0.9).evaluate(_broken, HELD_OUT)
    mutations = Repairer().repair(card.failure_report)
    assert mutations
