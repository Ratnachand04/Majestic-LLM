"""The statistics behind the certificate (Part 7).

Majestic does not sell a model file, it sells a certificate — so if these are
wrong the product is a lie with good typography. Two corrections are under test:
the gate is a hypothesis test rather than a comparison (§2-§5), and seven
blocking axes would reject four good models in five (§8-§9).
"""
from __future__ import annotations

import pytest

from modelrig.proving_ground import BLOCKING_AXES, ProvingGround, Status
from modelrig.stats import (
    beta_posterior,
    bootstrap_variance,
    certifiable_threshold,
    certifies,
    clopper_pearson_lcb,
    days_to_certify,
    decompose_flips,
    effective_n,
    judge_usable_n,
    mcnemar,
    power_report,
    required_n,
    resolvable_difference,
    wilson,
)


# =========================================================================== #
# §3 — the small-holdout problem
# =========================================================================== #
def test_the_wilson_interval_at_fifty_spans_fourteen_points():
    """"0.937 versus 0.93" is not a comparison — it is noise."""
    interval = wilson(47, 50)
    assert interval.low == pytest.approx(0.838, abs=0.002)
    assert interval.high == pytest.approx(0.979, abs=0.002)
    assert interval.width > 0.13


def test_one_error_in_fifty_makes_a_93_percent_claim_uncertifiable():
    """§3's table, and the finding it forces: the Apex build every document in
    this series treated as a success cannot honestly claim its stated gate."""
    assert clopper_pearson_lcb(50, 50) == pytest.approx(0.942, abs=0.002)
    assert certifies(50, 50, 0.93) is True          # a perfect sweep, barely

    assert clopper_pearson_lcb(49, 50) == pytest.approx(0.906, abs=0.005)
    assert certifies(49, 50, 0.93) is False         # a single error

    assert certifies(47, 50, 0.93) is False         # the Apex build


def test_the_interval_is_honest_at_a_clean_sweep_of_five():
    """The normal approximation gives a zero-width interval at 1.0, which would
    certify anything on five examples. Wilson does not."""
    interval = wilson(5, 5)
    assert interval.low < 0.6
    assert interval.high == pytest.approx(1.0, abs=0.01)


def test_impossible_counts_are_refused():
    with pytest.raises(ValueError, match="0 <= successes <= n"):
        wilson(51, 50)


# =========================================================================== #
# §4 — how much data a gate actually needs
# =========================================================================== #
def test_the_sample_size_formula_reproduces():
    """Three of §4's four rows reproduce exactly from the one-sided formula the
    section header specifies. The q1=0.98 row gives 116, not 150 — it is the one
    row that would need a two-sided z, and is treated as a slip."""
    assert required_n(0.93, 0.96) == pytest.approx(380, abs=2)
    assert required_n(0.93, 0.85) == pytest.approx(81, abs=2)
    assert required_n(0.93, 0.83) == pytest.approx(53, abs=3)
    assert required_n(0.93, 0.98) == pytest.approx(116, abs=3)      # not 150


def test_fifty_examples_resolves_about_ten_points():
    """It cannot resolve the three-point differences gate thresholds are
    typically written at."""
    assert resolvable_difference(50, 0.93) == pytest.approx(0.105, abs=0.02)
    assert resolvable_difference(380, 0.93) < 0.04


def test_twenty_examples_resolves_nothing_at_all():
    """The schema's holdout_count >= 20 minimum is far too permissive."""
    assert resolvable_difference(20, 0.93) == float("inf")


def test_the_power_report_says_what_is_needed():
    report = power_report(50, 0.93)
    assert report.adequate is False
    assert report.needed_for_3pt > 300
    assert report.needed_for_5pt > 50


# =========================================================================== #
# §5 — gate on the lower bound
# =========================================================================== #
def test_the_gate_is_a_hypothesis_test_not_a_comparison():
    """The burden of proof sits on the model: a certificate asserts something,
    and an assertion needs evidence."""
    # Point estimate clears the gate; the evidence does not carry it.
    assert 47 / 50 > 0.93
    assert certifies(47, 50, 0.93) is False


def test_the_honest_escape_is_to_certify_what_n_supports():
    """Rather than refusing a build whose data cannot carry 0.93, offer the
    certificate the data does carry. A true statement beats no statement."""
    assert certifiable_threshold(47, 50) == pytest.approx(0.85, abs=0.01)
    assert certifies(47, 50, 0.85) is True


def test_more_data_certifies_where_less_could_not():
    """A 0.94 point estimate is only one point above a 0.93 gate, so even 500
    examples do not carry it — which is the honest shape of the problem, not a
    quirk. Certification needs either far more data or a wider margin."""
    assert certifies(47, 50, 0.93) is False
    assert certifies(470, 500, 0.93) is False       # same score, 10x the data
    assert certifies(4700, 5000, 0.93) is True      # 100x finally carries it
    assert certifies(490, 500, 0.93) is True        # or a wider margin instead


# =========================================================================== #
# §6 — Bayesian narrowing, and its trap
# =========================================================================== #
def test_a_prior_narrows_the_interval():
    weak = beta_posterior(47, 50, a0=1, b0=1)
    informed = beta_posterior(47, 50, a0=40, b0=3)
    assert (informed.high - informed.low) < (weak.high - weak.low)


def test_the_prior_may_never_outvote_the_customers_own_data():
    """Without the cap, a strong prior from successful ancestors certifies a
    genuinely broken build."""
    result = beta_posterior(5, 10, a0=900, b0=10)
    assert result.prior_capped is True
    assert result.prior_strength <= 10
    assert result.posterior_mean < 0.95        # not dragged to the prior


def test_both_numbers_are_reported():
    """A customer entitled to know their model works is entitled to know which
    number came from their data."""
    result = beta_posterior(47, 50, a0=20, b0=2).as_dict()
    assert "posterior_mean" in result and "frequentist_lcb" in result
    assert "which came from their data" in result["note"]


# =========================================================================== #
# §7 — provisional certification
# =========================================================================== #
def test_the_certificate_strengthens_with_deployment():
    """Corrections are ground truth, so n_eff grows and the interval narrows."""
    assert effective_n(50, 200, 0.06, 0) == 50
    assert effective_n(50, 200, 0.06, 30) == 410
    assert effective_n(50, 200, 0.06, 30) > effective_n(50, 200, 0.06, 10)


def test_the_customer_is_told_when_the_claim_will_be_supported():
    """About a month of use at 200/day and a 6% correction rate."""
    days = days_to_certify(n0=50, target_n=380, daily_volume=200, correction_rate=0.06)
    assert 25 <= days <= 30


def test_a_deployment_with_no_corrections_never_certifies():
    assert days_to_certify(50, 380, 200, 0.0) is None
    assert days_to_certify(400, 380, 200, 0.06) == 0.0


# =========================================================================== #
# §9 — seven axes run, four block
# =========================================================================== #
def test_exactly_four_axes_block():
    """Seven blocking gates is not conservatism, it is a broken gate: at 80%
    power per axis it rejects four good models in five."""
    assert len(BLOCKING_AXES) == 4
    assert BLOCKING_AXES == {"task_metric", "safety", "privacy", "contamination"}


def test_the_blocking_set_is_chosen_by_consequence():
    """Task metric is the job; safety and privacy have unbounded loss;
    contamination is deterministic so it blocks for free."""
    for name in ("judge", "calibrated_judge", "behavioural", "regression",
                 "calibration"):
        assert name not in BLOCKING_AXES


def _held_out(n: int, correct: int):
    return [(f"t{i}", "GOLD") for i in range(n)], correct


def _predictor(correct: int):
    def predict(texts):
        return ["GOLD" if i < correct else "WRONG" for i in range(len(texts))]
    return predict


def test_an_advisory_failure_does_not_halt_the_build():
    """All seven run and every result appears on the scorecard. What changes is
    only which failures stop the pipeline."""
    held = [(f"t{i}", "GOLD") for i in range(50)]

    def brittle(texts):
        # Correct on the held-out set, but a cosmetic edit changes the answer —
        # so the behavioural (invariance) axis fails while the task metric does
        # not. Exactly the case that should be reported and not halt the build.
        return ["GOLD" if t.startswith("t") else "DIFFERENT" for t in texts]

    card = ProvingGround(quality_gate=0.9).evaluate(brittle, held)
    assert card.passed is True
    assert card.advisory_failures
    assert all(a.blocking is False or a.passed for a in card.axes)


def test_no_axis_is_ever_skipped():
    """The invariant that survives the correction: every axis runs, every
    result is shown, advisory failures are shown prominently."""
    held, correct = _held_out(20, 20)
    card = ProvingGround().evaluate(_predictor(correct), held)
    names = {a.name for a in card.axes}
    assert len(card.axes) == 7
    assert {"task_metric", "safety", "privacy", "calibration"} <= names


def test_a_blocking_failure_still_halts():
    held, correct = _held_out(50, 20)          # 40% — far below the gate
    card = ProvingGround(quality_gate=0.9).evaluate(_predictor(correct), held)
    assert card.passed is False
    assert card.status is Status.REFUSED
    assert card.failure_report is not None
    assert "task_metric" in card.failure_report.failed_axes


# =========================================================================== #
# §7 / §19 — the scorecard
# =========================================================================== #
def test_the_apex_build_ships_provisional_not_certified():
    """The estimate clears the target; this much data cannot yet prove it."""
    held, correct = _held_out(50, 47)
    card = ProvingGround(quality_gate=0.93).evaluate(_predictor(correct), held)
    assert card.passed is True
    assert card.status is Status.PROVISIONAL


def test_enough_data_upgrades_the_status_to_certified():
    held, correct = _held_out(500, 490)
    card = ProvingGround(quality_gate=0.93).evaluate(_predictor(correct), held)
    assert card.status is Status.CERTIFIED


def test_the_summary_is_written_for_a_clinic_owner():
    """"93.7% (between 84% and 98% — based on 50 of your forms)" is a sentence a
    non-technical reader can act on. "F1 = 0.937" is not."""
    held, correct = _held_out(50, 47)
    card = ProvingGround(quality_gate=0.93).evaluate(_predictor(correct), held)
    summary = card.plain_summary()
    assert "%" in summary and "between" in summary and "50 of your examples" in summary
    assert "PROVISIONAL" in summary


def test_the_scorecard_offers_the_claim_the_data_supports():
    held, correct = _held_out(50, 47)
    card = ProvingGround(quality_gate=0.93).evaluate(_predictor(correct), held)
    assert card.certifiable_claim() == pytest.approx(0.85, abs=0.02)
    assert card.power()["adequate"] is False


def test_the_eval_report_carries_the_interval_and_status():
    held, correct = _held_out(50, 47)
    report = ProvingGround(quality_gate=0.93).evaluate(
        _predictor(correct), held).to_eval_report()
    assert report["status"] == "provisional"
    assert report["interval"]["n"] == 50
    assert report["power"]["n_for_3_points"] > 300


# =========================================================================== #
# §11 — judge attenuation
# =========================================================================== #
def test_instrument_noise_costs_sample_size_twice_over():
    """rho^2 is classical attenuation. At n=50, f=0.15, rho=0.80 about half the
    data is gone — the quantitative reason the Judge is advisory."""
    assert judge_usable_n(50, 0.15, 0.80) == pytest.approx(27.2, abs=0.5)
    assert judge_usable_n(50, 0.0, 1.0) == 50.0


def test_a_worse_judge_costs_more_than_linearly():
    assert judge_usable_n(100, 0.0, 0.5) == pytest.approx(25.0)   # not 50


# =========================================================================== #
# §12 — the flip decomposition and its mechanism
# =========================================================================== #
def test_identical_deltas_can_hide_very_different_churn():
    """§12's table. Model B has rewritten a third of its answers."""
    a = decompose_flips([False] * 1 + [True] * 3 + [True] * 96,
                        [True] * 1 + [False] * 3 + [True] * 96)
    assert a.delta == pytest.approx(0.02, abs=0.001)
    assert a.phi == pytest.approx(0.04, abs=0.001)

    b = decompose_flips([False] * 15 + [True] * 17 + [True] * 68,
                        [True] * 15 + [False] * 17 + [True] * 68)
    assert b.delta == pytest.approx(0.02, abs=0.001)
    assert b.phi == pytest.approx(0.32, abs=0.001)

    assert a.delta == pytest.approx(b.delta)
    assert b.phi == pytest.approx(8 * a.phi)     # same net change, 8x the churn


def test_phi_is_an_estimate_of_boundary_mass():
    """If quantisation perturbs the decision function by epsilon, the answers
    that flip are exactly those whose margin was smaller than it."""
    churny = decompose_flips([True] * 50, [False] * 25 + [True] * 25)
    assert churny.boundary_mass == churny.phi
    assert churny.phi == pytest.approx(0.5)


def test_the_bootstrap_experiment_is_available_without_new_models():
    """§12's second, cheaper experiment for C4: high-phi models should show
    higher bootstrap variance, testable with no field data."""
    certain = bootstrap_variance([True] * 100)
    uncertain = bootstrap_variance([True] * 50 + [False] * 50)
    assert uncertain > certain
    assert bootstrap_variance([]) == 0.0


def test_unpaired_sequences_are_refused():
    with pytest.raises(ValueError, match="equal-length"):
        decompose_flips([True, False], [True])


# =========================================================================== #
# §20 — paired comparison, honestly read
# =========================================================================== #
def test_at_fifty_examples_the_honest_headline_is_could_not_detect():
    """"We could not detect a difference" and "the student matches the teacher"
    are different claims, and only the first is supported."""
    student = [True] * 45 + [False] * 5
    teacher = [True] * 43 + [False] * 2 + [True] * 5
    result = mcnemar(student, teacher)
    assert result.discordant < 10
    assert "could not detect" in result.headline()


def test_enough_discordant_pairs_supports_a_real_claim():
    student = [True] * 40 + [False] * 10
    teacher = [False] * 40 + [True] * 10
    result = mcnemar(student, teacher)
    assert result.discordant == 50
    assert result.significant is True
    assert "better" in result.headline()


def test_only_discordant_pairs_carry_information():
    identical = mcnemar([True] * 50, [True] * 50)
    assert identical.discordant == 0
    assert identical.p_value == 1.0
