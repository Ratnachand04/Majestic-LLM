"""The Proving Ground must not be fooled by a malformed answer.

Every other subsystem produces candidates; this is the only one that can reject
them, and what the customer is sold is its certificate. So the failure that
matters here is not a wrong score — it is a score computed over evidence the
model never actually produced, reported as though it had.

Python's ``zip`` truncates to the shorter sequence. That single default is
enough to turn "the model answered 20 of your 30 forms" into a certificate that
says 30, because the numerator came from the zip and the denominator came from
the held-out set.
"""
from __future__ import annotations

import pytest

from modelrig.proving_ground import MalformedPrediction, ProvingGround

HELD = [(f"held out example number {i}", "positive" if i % 2 else "negative")
        for i in range(30)]


def _honest(texts):
    return ["positive" if i % 2 else "negative" for i, _ in enumerate(texts)]


# =========================================================================== #
# The predictor must answer every input
# =========================================================================== #
def test_a_short_answer_is_refused_not_scored():
    """Silently scoring a partial answer is the one thing this must never do."""
    def drops_the_hard_ones(texts):
        return _honest(texts)[:20]

    with pytest.raises(MalformedPrediction, match="20 predictions for 30"):
        ProvingGround(quality_gate=0.80).evaluate(drops_the_hard_ones, HELD)


def test_an_over_long_answer_is_refused_too():
    """A surplus answer means the predictor lost track of its own inputs.

    Truncating it would score the first n and never mention the rest, which is
    the same silence in the other direction.
    """
    def duplicates(texts):
        return _honest(texts) + ["positive"]

    with pytest.raises(MalformedPrediction, match="31 predictions for 30"):
        ProvingGround(quality_gate=0.80).evaluate(duplicates, HELD)


def test_the_refusal_names_both_counts():
    """A reader has to be able to act on it without opening the code."""
    with pytest.raises(MalformedPrediction) as err:
        ProvingGround(quality_gate=0.80).evaluate(lambda t: _honest(t)[:5], HELD)
    message = str(err.value)
    assert "5 predictions" in message and "30 held-out" in message
    assert "sample size" in message


def test_an_honest_predictor_is_unaffected():
    card = ProvingGround(quality_gate=0.80).evaluate(_honest, HELD)
    task = next(a for a in card.axes if a.name == "task_metric")
    assert task.n == len(HELD) == card.n_held_out
    assert task.score == 1.0


# =========================================================================== #
# The other axes call the predictor too
# =========================================================================== #
def test_the_invariance_probe_is_guarded_as_well():
    """The behavioural axis calls the predictor twice and divides by the first
    call's length. A short second call would report the artefact as less stable
    than it is, and stability is what the axis exists to measure."""
    calls = {"n": 0}

    def short_on_the_perturbed_pass(texts):
        calls["n"] += 1
        out = _honest(texts)
        return out if calls["n"] != 2 else out[:10]

    with pytest.raises(MalformedPrediction):
        ProvingGround(quality_gate=0.80).evaluate(short_on_the_perturbed_pass, HELD)


def test_the_privacy_probe_pairs_each_answer_with_its_own_source():
    """Privacy is a BLOCKING axis, so a mis-paired comparison is not cosmetic.

    The probes are built from the first ten training texts. Pairing the answers
    against the whole corpus happened to line up only because that slice is in
    order; this pins the pairing to the documents the probes actually came from.
    """
    leaked = "the patient identifier is AB-99012 and the referring doctor is Patel"
    training = [leaked] + [f"unrelated clinical note {i}" for i in range(20)]

    # The model parrots its first training document back verbatim.
    card = ProvingGround(quality_gate=0.80).evaluate(
        lambda texts: [leaked] + ["negative"] * (len(texts) - 1), HELD,
        training_texts=training,
    )
    privacy = next(a for a in card.axes if a.name == "privacy")
    assert privacy.blocking
    assert not privacy.passed, "verbatim reproduction of a training document must fail"


# =========================================================================== #
# The judge reports what it can actually resolve
# =========================================================================== #
def test_the_judge_reports_its_usable_sample_size():
    """Part 7 §11 — a judge scored over n items does not carry n items of
    information. Swap-inconsistent items are excluded and agreement attenuates
    the rest quadratically. Reporting the raw count would make this axis look as
    well-evidenced as the task metric, which is exactly why it is advisory."""
    card = ProvingGround(quality_gate=0.80).evaluate(_honest, HELD)
    judge = next(a for a in card.axes if a.name == "calibrated_judge")

    assert judge.n == 30
    assert not judge.blocking
    # n(1-f)rho^2 = 30 * 0.85 * 0.64 = 16.32
    assert "16 of 30" in judge.detail
    assert "advisory" in judge.detail


def test_a_better_judge_resolves_more():
    """The attenuation is a property of the instrument, not a constant."""
    good = ProvingGround(quality_gate=0.80, judge_swap_inconsistency=0.02,
                         judge_agreement=0.97).evaluate(_honest, HELD)
    poor = ProvingGround(quality_gate=0.80, judge_swap_inconsistency=0.30,
                         judge_agreement=0.60).evaluate(_honest, HELD)

    def usable(card):
        detail = next(a for a in card.axes if a.name == "calibrated_judge").detail
        return int(detail.split(" of ")[0].split()[-1])

    assert usable(good) > usable(poor)
    assert usable(good) <= 30, "attenuation can never manufacture information"
