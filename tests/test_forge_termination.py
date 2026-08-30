"""Termination, the probe hand-off, and learning (Part 4 §3, §5-§7, §17).

§8's ask set is the union of three criteria — required, decision-relevant and
ambiguous — and §5 adds two termination conditions beyond "the marginal rule
stopped": no unresolved ambiguity, and Gate 1 passes. These tests cover the
parts of the interview that decide when to *stop*, and the §6 diagnostic that
says whether the question policy is improving or merely getting longer.
"""
from __future__ import annotations

import pytest

from modelrig.forge import Forge
from modelrig.forge.core import (
    ATTRITION_GAMMA,
    MAX_AMBIGUITY,
    AskReason,
    Interviewer,
    Stop,
)
from modelrig.forge.infogain import marginal_information_gain, plan_diversity
from modelrig.forge.outcomes import InterviewOutcome, OutcomeLog
from modelrig.forge.posterior import parse_K
from modelrig.ir import ProfileSource, load_spec_ir
from modelrig.probe import DeviceProfile
from modelrig.probe import ProfileSource as ProbeSource

LAB = (
    "Read lab requisition forms into our system on an Android tablet. "
    "Must work when the internet dies. We have 200 real forms. "
    "Flag anything uncertain, do not guess. Hindi and English. 95 percent."
)

ANSWERS = {
    "offline_required": True, "data_rights": "customer_owned",
    "seed_data_ref": "s3://forms", "io_schema": {"patient_id": "str"},
    "latency_budget_ms": 60_000, "expected_input_tokens": 500, "quality_gate": 0.95,
}


def _probe(**over) -> dict:
    base = dict(
        device_id="apex-tablet-01", ram_total_mb=4000, ram_free_mb=2600,
        bw_eff_gbps=9.24, overhead_ms_per_token=2.16, thermal_derate_180s=0.68,
        prefill_ref_tok_s=50.0, reference_params=1_720_000_000,
        probe_lo_mb=400, probe_hi_mb=1_200, storage_free_mb=20_000,
        power_draw_w=3.5, source=ProbeSource.PROBE,
    )
    base.update(over)
    return DeviceProfile(**base).to_dict()


# =========================================================================== #
# §4 — gamma is pinned by three stated points
# =========================================================================== #
def test_gamma_is_the_specified_value():
    """§4 states 82% / 61% / 37% at four, ten and twenty questions. Those three
    points determine gamma, so they are the thing to assert."""
    assert ATTRITION_GAMMA == pytest.approx(0.05)


# =========================================================================== #
# §3 — the marginalised estimator
# =========================================================================== #
def _scored(description: str, slot: str, **kw):
    iv = Interviewer()
    post = iv.observe(description)
    known = {n: e.mode for n, e in post.slots.items()
             if not e.is_empty and not e.is_ambiguous and e.confidence >= 0.6}
    builder = iv._scoring_builder(known, description)
    return marginal_information_gain(slot, post, builder, iv.oracle, samples=12, **kw)


def test_the_marginal_estimator_still_zeroes_an_irrelevant_slot():
    """The property that must survive marginalising: a slot whose every answer
    yields the same plan is worth no questions, however uncertain it is."""
    assert _scored(LAB, "budget_ceiling_usd").gain == 0.0


def test_the_marginal_estimator_ranks_rather_than_saturating():
    """Pinning the other slots normalises by log2(n) and ties slots at the
    ceiling. Marginalising over their posteriors discriminates instead."""
    gains = {s: _scored(LAB, s).gain for s in
             ("latency_budget_ms", "expected_input_tokens", "device_target")}
    assert all(0.0 <= g <= 1.0 for g in gains.values())
    assert len(set(round(g, 3) for g in gains.values())) > 1


def test_plan_diversity_is_zero_when_every_plan_agrees():
    from modelrig.forge.infogain import PlanSignature

    same = PlanSignature(admitted=True, base_ref="b", quantiser="q4_k_m")
    other = PlanSignature(admitted=False, witness=("P_ram",))
    assert plan_diversity([same, same, same]) == 0.0
    assert plan_diversity([same]) == 0.0            # one plan is not a spread
    assert plan_diversity([same, other]) > 0.0


def test_the_marginal_estimate_is_reproducible():
    assert _scored(LAB, "latency_budget_ms").gain == \
        pytest.approx(_scored(LAB, "latency_budget_ms").gain)


# =========================================================================== #
# §8 lines 8-9 — three reasons to ask, and they are different
# =========================================================================== #
def test_ambiguity_is_an_independent_reason_to_ask():
    """§5 condition 3. A slot the description reads two ways gets asked whatever
    the plan says, because the alternative is silently picking one reading."""
    iv = Interviewer().conduct(LAB)
    reasons = {q.reason for q in (*iv.asked, *iv.pending)}
    assert reasons <= {AskReason.REQUIRED, AskReason.DECISION, AskReason.AMBIGUOUS}
    assert AskReason.REQUIRED in reasons


def test_a_question_records_which_of_the_three_criteria_put_it_there():
    iv = Interviewer().conduct(LAB)
    for q in iv.pending:
        assert q.as_dict()["reason"] in {"required", "decision", "ambiguous"}
        assert q.rationale


def test_a_declared_ambiguity_counts_even_when_every_resample_agrees():
    """The canonical case. "Must work when the internet dies" supports two
    readings, and the parser recognises it outright — stronger evidence than a
    vote, not weaker. Resampling alone would score it unambiguous."""
    post = Interviewer().observe(LAB)
    offline = post["offline_required"]
    assert offline.declared_ambiguous is True
    assert offline.ambiguity == 0.0              # every resample agreed
    assert offline.effective_ambiguity >= 0.5    # ...and it is still contested
    assert offline.is_ambiguous is True


def test_an_ambiguous_slot_is_listed_as_unresolved_when_unanswered():
    iv = Interviewer().conduct(LAB)
    assert "offline_required" in iv.unresolved_ambiguities
    assert iv.complete is False


def test_the_ambiguity_ceiling_is_a_real_threshold():
    assert 0.0 < MAX_AMBIGUITY < 1.0


# =========================================================================== #
# §5 — termination has four conditions, not one
# =========================================================================== #
def test_a_spec_that_type_checks_is_not_yet_complete():
    """Gate 1 is condition 4. A spec can be well-formed and inadmissible."""
    iv = Interviewer().conduct(
        "Classify support tickets by sentiment on an android phone",
        answers={"offline_required": True, "seed_data_count": 150},
    )
    assert iv.spec is not None            # it type-checked
    assert iv.gate1.passed is False       # ...and is not admissible
    assert iv.complete is False
    assert iv.stopped_because is Stop.GATE_1_FAILED


def test_a_gate_1_refusal_names_what_the_user_can_fix():
    """§11: Gate 1 refusals are FORGE's to explain, being remediable by the
    user. Gate 2 refusals belong to the Planner."""
    iv = Interviewer().conduct(LAB, answers={k: v for k, v in ANSWERS.items()
                                             if k != "data_rights"})
    assert iv.gate1.passed is False
    assert any("data rights" in r for r in iv.gate1.reasons)


def test_answering_everything_completes_the_interview():
    iv = Interviewer().conduct(LAB, answers=ANSWERS)
    assert iv.gate1.passed is True
    assert iv.unresolved_ambiguities == []
    assert iv.complete is True
    assert iv.needs_answers is False


def test_unresolved_ambiguities_are_emitted_rather_than_guessed():
    """§5: at the cap FORGE emits a partial spec plus an explicit list. It
    never guesses to complete."""
    iv = Interviewer(max_ambiguity=0.0).conduct(LAB, answers=ANSWERS)
    if iv.unresolved_ambiguities:
        assert iv.complete is False
        assert iv.stopped_because is Stop.UNRESOLVED_AMBIGUITY
        assert iv.spec is not None        # the partial spec is still emitted
    assert "unresolved_ambiguities" in iv.report()


# =========================================================================== #
# §7 — the probe hand-off
# =========================================================================== #
def test_an_unprobed_interview_emits_a_probe_token_not_a_question():
    """Never ask a non-technical user for RAM."""
    iv = Interviewer().conduct(LAB, answers=ANSWERS)
    assert iv.needs_probe is True
    token = iv.probe_request
    assert token["costs_questions"] == 0
    assert "target.device_profile" in token["collapses"]
    assert "ram_free" in token["measures"]


def test_a_probed_interview_asks_for_no_probe():
    iv = Interviewer().conduct(LAB, answers=ANSWERS, device_profile=_probe())
    assert iv.needs_probe is False
    assert iv.probe_request is None
    assert iv.spec.profile_source is ProfileSource.PROBE


def test_the_probe_settles_the_device_group_without_a_question():
    iv = Interviewer().conduct(LAB, answers=ANSWERS, device_profile=_probe())
    asked = {q.slot for q in (*iv.asked, *iv.pending)}
    assert "device_profile" not in asked
    assert iv.probed                      # settled, and recorded as settled


# =========================================================================== #
# §6 — learning, or just asking more?
# =========================================================================== #
def _log(early_q, early_pass, late_q, late_pass, n=10) -> OutcomeLog:
    log = OutcomeLog()
    for i in range(n):
        log.add(InterviewOutcome(f"e{i}", questions_asked=early_q,
                                 gate_passed_first_attempt=i < early_pass))
    for i in range(n):
        log.add(InterviewOutcome(f"l{i}", questions_asked=late_q,
                                 gate_passed_first_attempt=i < late_pass))
    return log


def test_the_reward_is_observed_never_judged():
    """§6: gate outcome, question count, customer acceptance. No proxy."""
    good = InterviewOutcome("a", questions_asked=4, gate_passed_first_attempt=True)
    rejected = InterviewOutcome("b", questions_asked=4,
                                gate_passed_first_attempt=True, customer_rejected=True)
    assert good.reward() == pytest.approx(1 - 4 * ATTRITION_GAMMA)
    assert rejected.reward() < good.reward()
    assert good.succeeded is True and rejected.succeeded is False


def test_an_abandoned_interview_pays_attrition_and_earns_nothing():
    """No gate was ever attempted — the outcome the question budget exists to
    avoid, and it must not score as a neutral result."""
    abandoned = InterviewOutcome("x", questions_asked=12, abandoned=True)
    assert abandoned.reward() == pytest.approx(-12 * ATTRITION_GAMMA)
    assert abandoned.succeeded is False


def test_a_longer_interview_that_passes_more_is_not_learning():
    """§6's diagnostic. Pass rate rose, but only because q_bar rose — the
    attrition cost is being paid and not counted."""
    diagnosis = _log(4, 5, 14, 7).diagnose()
    assert diagnosis["verdict"] == "asking_more"
    assert "This is not learning" in diagnosis["detail"]
    assert diagnosis["delta"]["mean_questions"] > 0


def test_a_shorter_interview_that_passes_more_is_learning():
    diagnosis = _log(7, 5, 5, 8).diagnose()
    assert diagnosis["verdict"] == "learning"
    assert diagnosis["delta"]["pass_rate"] > 0
    assert diagnosis["delta"]["mean_questions"] < 0


def test_the_diagnostic_reports_both_rates_or_neither():
    """A pass rate without a question count is indistinguishable from a policy
    that is simply asking more."""
    for key in ("pass_rate", "mean_questions", "mean_reward"):
        assert key in _log(7, 5, 5, 8).diagnose()["delta"]


def test_a_thin_history_refuses_to_diagnose():
    log = OutcomeLog()
    log.add(InterviewOutcome("a", questions_asked=4))
    assert log.diagnose()["verdict"] == "insufficient_history"


def test_outcomes_can_be_recorded_straight_from_an_interview():
    iv = Interviewer().conduct(LAB, answers=ANSWERS)
    log = OutcomeLog()
    outcome = log.record(iv, gate_passed_first_attempt=True)
    assert outcome.spec_hash == iv.spec.hash
    assert outcome.questions_asked == len(iv.asked) + len(iv.pending)


def test_the_outcome_log_round_trips(tmp_path):
    log = _log(7, 5, 5, 8)
    restored = OutcomeLog.load(log.save(tmp_path / "outcomes.json"))
    assert restored.to_list() == log.to_list()
    assert restored.diagnose()["verdict"] == log.diagnose()["verdict"]


def test_reward_is_attributed_per_slot():
    log = OutcomeLog()
    log.add(InterviewOutcome("a", questions_asked=3, slots_asked=("data_rights",),
                             gate_passed_first_attempt=True))
    log.add(InterviewOutcome("b", questions_asked=3, slots_asked=("behaviour.tone",),
                             gate_passed_first_attempt=False))
    by_slot = log.by_slot()
    assert by_slot["data_rights"]["mean_reward"] > by_slot["behaviour.tone"]["mean_reward"]


# =========================================================================== #
# §17 — the schema gap, and the fixture that pins the correction
# =========================================================================== #
APEX = "configs/specs/apex_diagnostics.yaml"


def test_the_apex_spec_carries_the_first_order_latency_drivers():
    """§17: expected_input_tokens did not exist, and §13.3 shows it is a
    first-order driver. Without it latency is costed on decode alone."""
    spec = load_spec_ir(APEX)
    assert spec.expected_input_tokens == 1000
    assert spec.expected_output_tokens == 80
    assert spec.expected_daily_volume == 200


def test_the_apex_latency_budget_reflects_the_corrected_arithmetic():
    """1.6 s asserted, 13.8 s at 500 tokens, ~29 s at 1000. The budget carries
    the corrected figure, not the one prose accepted without checking."""
    spec = load_spec_ir(APEX)
    assert spec.latency_budget_ms >= 29_000
    assert spec.latency_budget_ms < 120_000      # realistic, not merely generous


def test_the_apex_workload_is_past_the_prefill_crossover():
    """n_in/n_out = 12.5, far above the 4-6 crossover. Prefill is the dominant
    term, not a correction to it."""
    from modelrig.resources import prefill_crossover

    spec = load_spec_ir(APEX)
    ratio = spec.expected_input_tokens / spec.expected_output_tokens
    assert ratio > prefill_crossover(140.0, 26.0)


def test_the_apex_spec_is_admissible_but_needs_a_probe_to_be_planned():
    """The Part 3 / Part 4 seam: Gate 1 passes on the elicited spec, and Gate 2
    refuses until the device is measured, because an unmeasured profile cannot
    license a latency promise."""
    from modelrig.gates import gate1_spec_admissibility
    from modelrig.planner import default_catalog, plan

    spec = load_spec_ir(APEX)
    assert gate1_spec_admissibility(spec).passed is True
    assert plan(spec, default_catalog()).admitted is False

    spec.device_profile, spec.profile_source = _probe(), ProfileSource.PROBE
    assert plan(spec, default_catalog()).admitted is True


def test_the_apex_description_parses_to_the_fixture_primitive():
    spec = load_spec_ir(APEX)
    post = parse_K(Forge(), spec.notes, k=6)
    assert post["task_primitive"].mode is spec.task_primitive
