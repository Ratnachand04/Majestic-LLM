"""FORGE as an information-gathering problem (Part 4 §1-§11).

The claims under test, in the order the spec makes them:

* a parse is a *posterior*, and absence is not ambiguity (§2);
* information gain is measured by pushing candidate answers through the
  **Planner**, not by the parser's own entropy (§3);
* every question costs attrition, so the marginal rule decides (§4);
* the slot table's provenance — elicited, probed, derived — is the economy that
  makes four questions enough (§9-§11).
"""
from __future__ import annotations

import math

import pytest

from modelrig.forge import Forge, slots as slot_table
from modelrig.forge.core import (
    ATTRITION_GAMMA,
    Interviewer,
    Stop,
    completion_probability,
    economy,
    unasked_because_irrelevant,
    value_ratio,
    worth_asking,
)
from modelrig.forge.infogain import PlanSignature, information_gain
from modelrig.forge.posterior import canonical_key, candidate_values, parse_K
from modelrig.forge.slots import AskPolicy, IGClass, Source
from modelrig.planner.objective import Tier
from modelrig.primitives import TaskPrimitive

LAB = (
    "Read lab requisition forms into our system on an Android tablet. "
    "Must work when the internet dies. We have 200 real forms. "
    "Flag anything uncertain, do not guess. Hindi and English. 95 percent."
)
DRAFT = (
    "Draft polite replies to customer emails on a laptop. "
    "We have 150 real messages to learn from."
)


# =========================================================================== #
# §9-§11 — the slot table is the economy
# =========================================================================== #
def test_the_table_is_internally_consistent():
    assert slot_table.validate_table() == []


def test_nothing_probed_or_derived_is_ever_asked():
    """The design principle, enforced rather than merely stated: a field is
    elicited only if it is a human preference that cannot be measured."""
    for s in slot_table.SLOTS:
        if s.source is not Source.ELICITED:
            assert s.ask is AskPolicy.NEVER, s.name
            assert not s.costs_attrition


def test_derived_slots_declare_what_they_are_derived_from():
    for s in slot_table.SLOTS:
        if s.source is Source.DERIVED:
            assert s.derived_from, s.name


def test_the_device_group_is_probed_not_elicited():
    """Never ask a non-technical user for RAM. One probe collapses the whole
    group and costs no questions at all."""
    profile = slot_table.get("target.device_profile")
    assert profile.source is Source.PROBED
    assert profile.ig_class is IGClass.HIGHEST      # high value, zero questions


def test_context_budget_is_derived_from_the_device_and_never_asked():
    ctx = slot_table.get("target.context_budget")
    assert ctx.source is Source.DERIVED
    assert "target.device_profile" in ctx.derived_from


def test_the_table_maps_cleanly_onto_the_spec_ir():
    from modelrig.ir import SpecIR

    fields = set(SpecIR.__dataclass_fields__)
    for s in slot_table.SLOTS:
        if s.spec_field:
            assert s.spec_field in fields, s.name


def test_tone_is_conditional_on_the_primitive():
    """§10's cleanest case for why entropy is the wrong criterion: tone can be
    maximally uncertain in an extraction description and still be worthless."""
    tone = slot_table.get("behaviour.tone")
    assert tone.ig_class is IGClass.ZERO
    assert tone.applicable({"task.primitive": "extract"}) is False
    assert tone.applicable({"task.primitive": "generate"}) is True


def test_askable_shrinks_when_the_primitive_rules_a_slot_out():
    extract = {s.name for s in slot_table.askable({"task.primitive": "extract"})}
    generate = {s.name for s in slot_table.askable({"task.primitive": "generate"})}
    assert "behaviour.tone" in generate
    assert "behaviour.tone" not in extract


# =========================================================================== #
# §2 — a parse is a posterior, and absence is not ambiguity
# =========================================================================== #
def test_parse_K_returns_a_distribution_not_a_point_estimate():
    post = parse_K(Forge(), LAB, k=8)
    assert post.k == 8
    assert post["task_primitive"].samples == 8
    assert post["task_primitive"].mode is TaskPrimitive.EXTRACT


def test_the_first_sample_is_always_the_full_description():
    """So the posterior's mode can never disagree with an honest single parse."""
    post = parse_K(Forge(), LAB, k=6)
    single = Forge().parse(LAB)
    assert post["task_primitive"].mode is single.slots["task_primitive"].value


def test_absence_and_ambiguity_are_different_measurements():
    """The distinction a point estimate destroys. One needs supplying, the other
    needs disambiguating — and the fixes are not interchangeable."""
    post = parse_K(Forge(), LAB, k=8)
    silent = post["seed_data_ref"]               # never mentioned
    contested = post["offline_required"]         # "when the internet dies"

    assert silent.is_empty is True
    assert silent.ambiguity == 0.0               # nothing to disambiguate
    assert contested.is_empty is False
    assert contested.is_ambiguous is True


def test_a_redundantly_evidenced_slot_survives_ablation():
    """The resampling scheme measures textual support, so a slot resting on one
    phrase flips where a well-evidenced one does not."""
    post = parse_K(Forge(), LAB, k=12)
    assert post["task_primitive"].stable is True
    assert "task_primitive" in post.stable()


def test_entropy_is_zero_when_every_sample_agrees():
    post = parse_K(Forge(), LAB, k=8)
    assert post["task_primitive"].entropy == pytest.approx(0.0)


def test_entropy_is_maximal_on_an_even_split():
    from modelrig.forge.posterior import SlotPosterior

    p = SlotPosterior(name="x", counts={"a": 4, "b": 4}, samples=8,
                      representative={"a": "a", "b": "b"})
    assert p.entropy == pytest.approx(1.0)
    assert p.normalised_entropy == pytest.approx(1.0)
    assert p.ambiguity == pytest.approx(0.5)


def test_constrained_decoding_discards_out_of_domain_values():
    """A hallucinated value must never be able to win a vote."""
    from modelrig.forge.posterior import project_onto_domain

    assert project_onto_domain("data_rights", "customer_owned") == "customer_owned"
    assert project_onto_domain("data_rights", "whatever_we_like") is None
    assert project_onto_domain("latency_budget_ms", 30_000) == 30_000  # no domain


def test_continuous_slots_are_bucketed_at_a_width_that_suits_them():
    """A single bucket for every plausible quality gate would erase the
    difference between a lenient bar and one nothing can meet."""
    assert canonical_key("latency_budget_ms", 30_000) == \
           canonical_key("latency_budget_ms", 31_000)
    assert canonical_key("quality_gate", 0.80) != canonical_key("quality_gate", 0.98)


def test_the_posterior_is_reproducible():
    """A build that cannot be replayed cannot be audited."""
    a = parse_K(Forge(), LAB, k=8, seed=7)
    b = parse_K(Forge(), LAB, k=8, seed=7)
    assert a.as_dict() == b.as_dict()


def test_a_supplied_answer_is_ground_truth_not_a_sample():
    post = parse_K(Forge(), LAB, k=6, known={"offline_required": True})
    assert post["offline_required"].is_ambiguous is False
    assert post["offline_required"].mode is True


# =========================================================================== #
# §3 — the Planner is the information-gain oracle
# =========================================================================== #
def _scored(description: str, slot: str, **over):
    iv = Interviewer(**over)
    post = iv.observe(description)
    known = {n: e.mode for n, e in post.slots.items()
             if not e.is_empty and not e.is_ambiguous and e.confidence >= 0.6}
    builder = iv._scoring_builder(known, description)
    return information_gain(slot, post, builder, iv.oracle)


def test_information_gain_is_the_entropy_of_the_induced_plans():
    """IG = H(Pi | sigma) exactly, because the plan is a deterministic function
    of the answer: H(Pi | A) = 0, so I(A; Pi) = H(Pi)."""
    g = _scored(LAB, "data_rights")
    assert g.gain > 0
    assert g.distinct_plans > 1
    # Two outcomes over three equally weighted values: -(2/3 lg 2/3 + 1/3 lg 1/3).
    expected = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert g.entropy_bits == pytest.approx(expected, abs=1e-9)
    assert g.gain == pytest.approx(expected / math.log2(3), abs=1e-9)


def test_a_slot_that_cannot_move_the_plan_scores_exactly_zero():
    """However uncertain it is. This is a proof, not a rule of thumb: if every
    candidate answer yields the same plan, the answer carries no information
    about the plan."""
    g = _scored(LAB, "budget_ceiling_usd")
    assert g.gain == 0.0
    assert g.decision_relevant is False
    assert "cannot change the build" in g.reason()


def test_an_answer_that_straddles_admitted_and_refused_carries_the_whole_stake():
    g = _scored(LAB, "data_rights")
    assert g.flips_admissibility is True
    assert g.stake == 1.0
    # A plan change between two working models is worth a fraction of that.
    assert _scored(LAB, "device_target").stake < 1.0


def test_the_input_length_is_decision_relevant_because_prefill_is(
):
    """Part 4 §17: costing decode alone understates a document task ~3x, so the
    input length moves the plan and is worth a question."""
    g = _scored(LAB, "expected_input_tokens")
    assert g.gain > 0
    assert g.flips_admissibility is True


def test_the_oracle_is_memoised_because_the_planner_is_deterministic():
    iv = Interviewer()
    post = iv.observe(LAB)
    known = {n: e.mode for n, e in post.slots.items()
             if not e.is_empty and not e.is_ambiguous and e.confidence >= 0.6}
    builder = iv._scoring_builder(known, LAB)
    information_gain("data_rights", post, builder, iv.oracle)
    calls = iv.oracle.calls
    information_gain("data_rights", post, builder, iv.oracle)
    assert iv.oracle.calls == calls          # every repeat served from cache
    assert iv.oracle.hits > 0


def test_a_value_that_cannot_be_lowered_into_a_spec_counts_as_an_outcome():
    """Not a silent skip: an unbuildable answer is a distinct, and maximally
    bad, result."""
    from modelrig.forge.infogain import induced_plans

    def explode(_overrides):
        raise ValueError("no primitive")

    outcomes, error = induced_plans("task_primitive", ["extract"], explode, lambda s: None)
    assert error is not None
    assert outcomes["extract"] == PlanSignature(admitted=False, witness=("spec_invalid",))


def test_the_plan_signature_ignores_detail_the_customer_cannot_see():
    """Two plans differing only in rank are the same answer; counting them apart
    would manufacture gain out of implementation detail."""
    a = PlanSignature(admitted=True, base_ref="b", quantiser="q4_k_m", target="gguf")
    b = PlanSignature(admitted=True, base_ref="b", quantiser="q4_k_m", target="gguf")
    assert a == b and hash(a) == hash(b)


def test_the_plan_signature_keeps_the_detail_the_customer_can_see():
    """Grammar and eval suite decide the output shape and what is certified.
    Drop them and the task primitive measures as worthless to ask."""
    extract = PlanSignature(admitted=True, base_ref="b", eval_suite_ref="eval:extract")
    classify = PlanSignature(admitted=True, base_ref="b", eval_suite_ref="eval:classify")
    assert extract != classify


# =========================================================================== #
# §4 — every question costs attrition
# =========================================================================== #
def test_completion_decays_exponentially_in_the_question_count():
    """§4 pins gamma by three stated points, so assert the curve through them."""
    assert completion_probability(0) == 1.0
    assert completion_probability(4) == pytest.approx(math.exp(-4 * ATTRITION_GAMMA))
    assert completion_probability(4) == pytest.approx(0.82, abs=0.01)
    assert completion_probability(10) == pytest.approx(0.61, abs=0.01)
    assert completion_probability(20) == pytest.approx(0.37, abs=0.01)


def test_lambda_rises_with_trust_damage():
    """Lambda = (V + kappa)/V. Attrition costs a share of the deal; bad
    information costs the deal and the trust damage on top."""
    assert value_ratio(Tier.EXPERIMENTAL) < value_ratio(Tier.COMMERCIAL)
    assert value_ratio(Tier.COMMERCIAL) < value_ratio(Tier.REGULATED)
    assert value_ratio(Tier.REGULATED) > 10


def test_a_regulated_build_asks_questions_an_experimental_one_will_not():
    """The coupling worth having: raising kappa raises theta* AND lambda, so the
    system refuses more and asks more. Both are caution, moving together."""
    marginal = 0.02
    assert worth_asking(marginal, tier=Tier.REGULATED) is True
    assert worth_asking(marginal, tier=Tier.COMMERCIAL) is False
    assert worth_asking(marginal, tier=Tier.EXPERIMENTAL) is False


def test_a_zero_gain_slot_is_never_worth_asking_at_any_tier():
    for tier in Tier:
        assert worth_asking(0.0, tier=tier) is False


def test_the_stake_multiplier_enters_the_rule():
    """An answer that only swaps one feasible base for another is worth a
    fraction of one that decides whether there is a model at all."""
    gain = 0.06
    assert worth_asking(gain, stake=1.0, tier=Tier.COMMERCIAL) is True
    assert worth_asking(gain, stake=0.25, tier=Tier.COMMERCIAL) is False


# =========================================================================== #
# §8 — the algorithm end to end
# =========================================================================== #
def test_the_interview_asks_few_questions_not_forty():
    iv = Interviewer().conduct(LAB)
    assert 0 < len(iv.pending) <= 8
    assert iv.completion_probability > 0.35


def test_questions_are_ordered_by_measured_gain_not_by_the_table_prior():
    """Attrition means order matters: a customer who abandons halfway should
    have answered the questions that mattered most."""
    iv = Interviewer().conduct(LAB)
    values = [q.gain for q in iv.pending]
    assert values == sorted(values, reverse=True)
    assert values[0] > 0


def test_the_planner_proves_some_questions_are_not_worth_asking():
    iv = Interviewer().conduct(LAB)
    skipped = unasked_because_irrelevant(iv)
    assert skipped
    for _slot, why in skipped:
        assert "cannot change the build" in why


def test_a_must_ask_slot_is_asked_even_at_zero_gain():
    """quality_gate moves no plan — the Planner ranks against theta*, and the
    customer's bar is what Gate 3 certifies against. A build with no agreed
    acceptance bar cannot be certified, so it is asked anyway."""
    iv = Interviewer().conduct(DRAFT)
    asked = {q.slot for q in (*iv.asked, *iv.pending)}
    assert "quality_gate" in asked
    zero_gain_must_asks = [q for q in iv.pending if q.must_ask and q.gain == 0.0]
    assert zero_gain_must_asks


def test_a_probe_costs_no_questions_and_settles_the_device_group():
    known: dict = {}
    collapsed = Interviewer.apply_probe(known, {"device_id": "x", "ram_free_mb": 2600})
    assert known["profile_source"].measured is True
    assert collapsed                      # slots settled without a question


def test_derivation_fills_slots_for_free():
    known = {"seed_data_count": 200, "io_schema": {"patient": "str", "test": "str"},
             "offline_required": True}
    derived = Interviewer.derive(known)
    assert derived["holdout_count"] == 50
    assert derived["expected_output_tokens"] == 24
    assert derived["runtime_mode"] == "on_device"


def test_answering_the_questions_produces_a_planned_spec():
    iv = Interviewer().conduct(LAB, answers={
        "offline_required": True,
        "data_rights": "customer_owned",
        "seed_data_ref": "s3://forms",
        "io_schema": {"patient_id": "str", "test_code": "str"},
        "latency_budget_ms": 60_000,
        "expected_input_tokens": 500,
        "quality_gate": 0.95,
    })
    assert iv.complete is True
    assert iv.spec.task_primitive is TaskPrimitive.EXTRACT
    assert iv.spec.offline_required is True
    assert iv.outcome is not None
    assert iv.pending == []


def test_the_transcript_explains_every_question_and_every_omission():
    iv = Interviewer().conduct(LAB)
    report = iv.report()
    assert report["gamma"] == ATTRITION_GAMMA
    assert report["lambda"] == pytest.approx(value_ratio(Tier.COMMERCIAL), abs=1e-4)
    for q in report["pending"]:
        assert q["question"] or q["must_ask"]
        assert q["rationale"]
    for g in report["not_worth_asking"]:
        assert g["reason"]


def test_the_interview_reports_its_own_economy():
    iv = Interviewer().conduct(LAB)
    acct = economy(iv)
    assert acct["probed"] + acct["derived"] > 0
    assert acct["questions_pending"] <= acct["elicited"]
    assert acct["slots_proved_irrelevant"] > 0


def test_the_interview_is_reproducible_question_for_question():
    a = Interviewer().conduct(LAB)
    b = Interviewer().conduct(LAB)
    assert [q.slot for q in a.pending] == [q.slot for q in b.pending]
    assert a.stopped_because is b.stopped_because


def test_stopping_is_recorded_as_a_reason_not_a_silence():
    iv = Interviewer().conduct(LAB)
    assert isinstance(iv.stopped_because, Stop)
    assert iv.report()["stopped_because"] in {s.value for s in Stop}


def test_forge_still_refuses_to_guess_rather_than_defaulting():
    """The scoring defaults are a counterfactual used inside the oracle. They
    must never leak into an emitted spec."""
    iv = Interviewer().conduct("please do the needful with our stuff")
    assert iv.complete is False
    assert iv.spec is None


def test_candidate_values_fall_back_to_the_table_for_an_unmentioned_slot():
    """Otherwise an empty slot would score zero gain for the circular reason
    that nothing was sampled from it."""
    post = parse_K(Forge(), LAB, k=4)
    assert len(candidate_values("latency_budget_ms", post.get("latency_budget_ms"))) >= 2
    assert len(candidate_values("data_rights", post.get("data_rights"))) >= 2
