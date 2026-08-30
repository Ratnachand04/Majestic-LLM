"""The interview: when to ask, and when to stop (Part 4 §4-§8).

Every question has a cost that no elicitation paper prices, because their
subjects are paid to finish. Ours are not:

    P(complete | q questions) = e^(-gamma * q)

At ``gamma = 0.05`` roughly 82% of customers finish a four-question interview,
61% finish ten, and 37% finish twenty. That is why "just ask everything" is not
a conservative choice — it is the choice that guarantees no spec at all.

**The marginal rule.** Ask slot ``i`` only when what the answer is worth beats
what the question costs:

    IG(sigma_i) * stake_i * Lambda  >  gamma,      Lambda = (V + kappa) / V

The asymmetry in ``Lambda`` is the whole point. Attrition costs a share of the
*deal* — the customer walks, and you lose ``V``. Bad information costs the deal
*and* the trust damage of shipping a failure — ``V + kappa``. So the ratio is
what decides, and it couples the interview to the refusal threshold
``theta* = (C + kappa)/(V + kappa)`` through the same two parameters. Raise
``kappa`` and the system both refuses more *and* asks more. Both are caution,
and they move together rather than trading off, which is the property you want
of a regulated-domain setting.

**Termination.** Not a question cap. The interview stops when no remaining slot
clears the marginal rule — and the strongest case of that is worth naming: when
every candidate answer to every remaining uncertainty yields the *same plan*,
the remaining uncertainty is provably irrelevant and stopping is not a heuristic
but a proof. ``max_questions`` exists as a policy backstop, not as the
mechanism.

The order of §8: parse K times, probe (which collapses the device group without
spending a question), derive everything derivable, ask the must-ask slots that
are still empty, then run the marginal rule until it fails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from majestic.logging_utils import get_logger
from modelrig.forge import slots as slot_table
from modelrig.forge.infogain import InfoGain, PlanOracle, PlanSignature, rank
from modelrig.forge.parser import Forge, InterviewState
from modelrig.forge.posterior import DEFAULT_K, ParsePosterior, parse_K
from modelrig.forge.slots import AskPolicy
from modelrig.ir import ProfileSource, SpecIR
from modelrig.planner.objective import Tier, decision_params

logger = get_logger(__name__)

#: Attrition per question, in the exponent of ``P(complete) = e^(-gamma q)``.
#: HYPOTHESIS — no published elicitation study measures this, because their
#: subjects are paid to finish. 0.05 is the specified value, and it is pinned by
#: three stated points: four questions complete 82% of the time, ten 61%, and
#: twenty 37%. :func:`completion_probability` reproduces all three.
ATTRITION_GAMMA = 0.05

#: A backstop, not the termination condition. The marginal rule should stop
#: first; if it does not, something upstream is miscalibrated and the interview
#: must not run away.
MAX_QUESTIONS = 8

#: ``A_max`` — the ambiguity a spec may carry unresolved (§5 condition 3).
#: Above it the description supports two readings and FORGE asks rather than
#: picking one, *whatever the information gain says*. Silently defaulting an
#: ambiguous slot is the single most expensive failure available here.
MAX_AMBIGUITY = 0.25


class AskReason(str, Enum):
    """Why a slot entered the ask set. §8 lines 8-9 union three criteria.

    They are genuinely different, and all three are needed. A required slot may
    be unambiguous and score zero gain; an ambiguous slot may be optional; a
    decision-relevant slot may be neither.
    """

    REQUIRED = "required"        # must_ask: bypasses the marginal rule
    DECISION = "decision"        # IG * stake * Lambda > gamma
    AMBIGUOUS = "ambiguous"      # A_i > A_max: the text supports two readings


def completion_probability(questions: int, gamma: float = ATTRITION_GAMMA) -> float:
    """``P(complete | q)``. Monotone decreasing, and that is the whole problem."""
    if questions < 0:
        raise ValueError("question count must be non-negative")
    return math.exp(-gamma * questions)


def value_ratio(tier: Tier = Tier.COMMERCIAL, build_cost_micro: int = 40_000_000) -> float:
    """``Lambda = (V + kappa) / V`` — what information is worth per unit of deal.

    Always at least 1: even with no trust damage at all, information is worth
    what the deal is worth. The tiers spread it widely — about 1.03 for
    experimental work, 1.67 commercial, 11 regulated — which is exactly the
    intended behaviour. A regulated build should tolerate a long interview and a
    throwaway experiment should not.
    """
    params = decision_params(build_cost_micro, tier)
    if params.value <= 0:
        return 1.0
    return (params.value + params.trust_damage) / params.value


def worth_asking(
    gain: float, *, stake: float = 1.0, tier: Tier = Tier.COMMERCIAL,
    gamma: float = ATTRITION_GAMMA, build_cost_micro: int = 40_000_000,
) -> bool:
    """The §4 marginal rule: ``IG * stake * Lambda > gamma``."""
    return gain * stake * value_ratio(tier, build_cost_micro) > gamma


# --------------------------------------------------------------------------- #
@dataclass
class Question:
    """One question, with the arithmetic that justified asking it."""

    slot: str
    text: str
    gain: float
    stake: float
    value: float                       # IG * stake * Lambda
    threshold: float                   # gamma
    must_ask: bool = False
    rationale: str = ""
    candidates: list[Any] = field(default_factory=list)
    answer: Any = None
    answered: bool = False
    reason: AskReason = AskReason.DECISION
    ambiguity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot, "question": self.text,
            "gain": round(self.gain, 4), "stake": self.stake,
            "value": round(self.value, 4), "threshold": self.threshold,
            "must_ask": self.must_ask, "rationale": self.rationale,
            "reason": self.reason.value, "ambiguity": round(self.ambiguity, 4),
            "answered": self.answered,
        }


class Stop(str, Enum):
    """Why the interview ended. Only the first two are healthy."""

    PLAN_INVARIANT = "plan_invariant"      # every remaining answer gives the same plan
    NOT_WORTH_ASKING = "not_worth_asking"  # the marginal rule failed
    BACKSTOP = "question_cap"              # MAX_QUESTIONS hit — investigate
    NO_ANSWERS = "no_answers"              # the customer stopped answering
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"   # A_i > A_max at the cap
    GATE_1_FAILED = "gate_1_failed"        # admissible spec, inadmissible request
    NEEDS_PROBE = "needs_probe"            # the device slots dominate; §7


@dataclass
class Interview:
    """The transcript, the spec it produced, and the reasoning behind both."""

    description: str
    posterior: ParsePosterior
    state: InterviewState
    asked: list[Question] = field(default_factory=list)
    pending: list[Question] = field(default_factory=list)
    skipped: list[InfoGain] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)
    probed: list[str] = field(default_factory=list)
    spec: SpecIR | None = None
    outcome: Any = None
    stopped_because: Stop = Stop.NOT_WORTH_ASKING
    tier: Tier = Tier.COMMERCIAL
    oracle_stats: dict[str, int] = field(default_factory=dict)
    #: §5 — slots the description read two ways and nobody resolved. Emitted
    #: explicitly alongside a partial spec; FORGE never guesses to complete.
    unresolved_ambiguities: list[str] = field(default_factory=list)
    #: §7 — a probe token, emitted INSTEAD of a question when the device slots
    #: dominate. One round trip collapses four of them; asking cannot.
    probe_request: dict[str, Any] | None = None
    #: §5 condition 4 — Gate 1 must pass before a spec is emitted.
    gate1: Any = None

    @property
    def questions_asked(self) -> int:
        return len(self.asked)

    @property
    def questions(self) -> list[str]:
        """The questions still to put to the customer, best first."""
        return [q.text for q in self.pending]

    @property
    def completion_probability(self) -> float:
        return completion_probability(self.questions_asked + len(self.pending))

    @property
    def complete(self) -> bool:
        """§5: all four conditions, not just a spec that type-checks."""
        return (
            self.spec is not None
            and not self.unresolved_ambiguities
            and (self.gate1 is None or self.gate1.passed)
        )

    @property
    def needs_answers(self) -> bool:
        """The other half of forge's return type (§1)."""
        return not self.complete

    @property
    def needs_probe(self) -> bool:
        return self.probe_request is not None

    def report(self) -> dict[str, Any]:
        return {
            "questions_asked": self.questions_asked,
            "questions_pending": len(self.pending),
            "completion_probability": round(self.completion_probability, 4),
            "stopped_because": self.stopped_because.value,
            "tier": self.tier.value,
            "lambda": round(value_ratio(self.tier), 4),
            "gamma": ATTRITION_GAMMA,
            "asked": [q.as_dict() for q in self.asked],
            "pending": [q.as_dict() for q in self.pending],
            "not_worth_asking": [g.as_dict() for g in self.skipped],
            "derived": {k: _plain(v) for k, v in self.derived.items()},
            "probed": list(self.probed),
            "spec_hash": self.spec.hash if self.spec else None,
            "oracle": dict(self.oracle_stats),
            "unresolved_ambiguities": list(self.unresolved_ambiguities),
            "probe_request": dict(self.probe_request) if self.probe_request else None,
            "gate1_passed": self.gate1.passed if self.gate1 is not None else None,
            "complete": self.complete,
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


# --------------------------------------------------------------------------- #
class Interviewer:
    """Runs the §8 algorithm over a parser, a probe and the Planner."""

    def __init__(
        self,
        forge: Forge | None = None,
        *,
        catalog: Any = None,
        tier: Tier = Tier.COMMERCIAL,
        gamma: float = ATTRITION_GAMMA,
        k: int = DEFAULT_K,
        max_questions: int = MAX_QUESTIONS,
        max_values: int = 3,
        max_ambiguity: float = MAX_AMBIGUITY,
        marginal: bool = True,
        samples: int = 12,
        planner: Callable[..., Any] | None = None,
    ) -> None:
        self.forge = forge or Forge()
        self.tier = tier
        self.gamma = gamma
        self.k = k
        self.max_questions = max_questions
        self.max_values = max_values
        self.max_ambiguity = max_ambiguity
        self.marginal = marginal
        self.samples = samples
        self.oracle = PlanOracle(catalog=catalog, planner=planner)

    # -- §8 step 1: parse K times ------------------------------------------ #
    def observe(
        self, description: str, *, known: dict[str, Any] | None = None, seed: int = 0
    ) -> ParsePosterior:
        return parse_K(self.forge, description, k=self.k, seed=seed, known=known)

    # -- §8 step 2: probe, which costs no questions ------------------------ #
    @staticmethod
    def apply_probe(known: dict[str, Any], profile: dict[str, Any] | None) -> list[str]:
        """Fold a device probe into the known answers.

        One round trip collapses the whole device group from a posterior to a
        point mass. Nothing else in the interview buys that much for that
        little, which is why the probe is worth engineering even though it is
        harder than asking.
        """
        if not profile:
            return []
        known["device_profile"] = profile
        known["profile_source"] = ProfileSource.PROBE
        return list(slot_table.PROBED) + ["target.latency_budget_ms(bounded)"]

    # -- §8 step 3: derive everything derivable ---------------------------- #
    @staticmethod
    def derive(known: dict[str, Any]) -> dict[str, Any]:
        """Fill the derived slots. Free, and every one is a question not asked."""
        out: dict[str, Any] = {}
        seeds = known.get("seed_data_count")
        if seeds:
            # A locked holdout, never trained on: the only honest basis for the
            # certificate the customer is shown.
            out["holdout_count"] = max(int(int(seeds) * 0.25), 1)
        if known.get("expected_output_tokens") is None:
            schema = known.get("io_schema")
            if isinstance(schema, dict) and schema:
                fields = [k for k in schema if not str(k).startswith("_")]
                # ~12 tokens of JSON per emitted field, floored so a one-field
                # schema does not imply a zero-length generation.
                out["expected_output_tokens"] = max(16, 12 * len(fields))
        if known.get("offline_required") is not None:
            out["runtime_mode"] = "on_device" if known["offline_required"] else "hybrid"
        known.update({k: v for k, v in out.items() if k in _SPEC_FIELDS})
        return out

    # -- §8 steps 4-6: ask, in the order the Planner says matters ---------- #
    def next_questions(
        self,
        posterior: ParsePosterior,
        known: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> tuple[list[Question], list[InfoGain], Stop]:
        """Score every candidate slot and return the ones worth asking."""
        limit = self.max_questions if limit is None else limit
        lam = value_ratio(self.tier)
        partial = _dotted(known)

        candidates = [
            s for s in slot_table.askable(partial)
            if s.spec_field and known.get(s.spec_field) is None
        ]
        must, optional = [], []
        for s in candidates:
            (must if s.ask is AskPolicy.MUST_ASK else optional).append(s)

        builder = self._scoring_builder(known, posterior.description)
        scored = {
            g.slot: g for g in rank(
                [s.spec_field for s in candidates], posterior, builder, self.oracle,
                max_values=self.max_values, marginal=self.marginal,
                samples=self.samples,
            )
        }

        def ambiguity(slot: slot_table.Slot) -> float:
            entry = posterior.get(slot.spec_field)
            return entry.effective_ambiguity if entry is not None else 0.0

        asked: list[Question] = []
        skipped: list[InfoGain] = []
        # Must-ask slots bypass the marginal rule, but they are still *ordered*
        # by measured gain: attrition means a customer who abandons halfway
        # should have answered the questions that mattered most.
        for s in sorted(must, key=lambda s: -scored[s.spec_field].gain):
            g = scored[s.spec_field]
            asked.append(_question(s, g, lam, self.gamma, must_ask=True,
                                   reason=AskReason.REQUIRED, ambiguity=ambiguity(s)))

        # §8 lines 8-9 union THREE reasons to ask, and they are different
        # criteria. A slot can be decision-relevant without being ambiguous, and
        # ambiguous without being decision-relevant — and an ambiguous slot must
        # be asked either way, because the alternative is picking one reading of
        # a sentence that supports two.
        for s in sorted(optional, key=lambda s: (-scored[s.spec_field].gain
                                                 * scored[s.spec_field].stake)):
            g = scored[s.spec_field]
            amb = ambiguity(s)
            contested = amb > self.max_ambiguity or (
                s.ask is AskPolicy.IF_AMBIGUOUS and _is_ambiguous(posterior, s)
            )
            relevant = worth_asking(g.gain, stake=g.stake, tier=self.tier,
                                    gamma=self.gamma)
            if contested:
                asked.append(_question(s, g, lam, self.gamma,
                                       reason=AskReason.AMBIGUOUS, ambiguity=amb))
            elif relevant and s.ask is not AskPolicy.IF_AMBIGUOUS:
                asked.append(_question(s, g, lam, self.gamma,
                                       reason=AskReason.DECISION, ambiguity=amb))
            else:
                skipped.append(g)

        stop = Stop.NOT_WORTH_ASKING
        if optional and all(not scored[s.spec_field].decision_relevant for s in optional):
            # Nothing left that we do not know can move the plan.
            stop = Stop.PLAN_INVARIANT
        if any(q.reason is AskReason.AMBIGUOUS for q in asked):
            stop = Stop.UNRESOLVED_AMBIGUITY
        if len(asked) > limit:
            asked, stop = asked[:limit], Stop.BACKSTOP
        return asked, skipped, stop

    # -- the whole thing ---------------------------------------------------- #
    def conduct(
        self,
        description: str,
        *,
        answers: dict[str, Any] | None = None,
        device_profile: dict[str, Any] | None = None,
        seed: int = 0,
        plan_it: bool = True,
    ) -> Interview:
        """Run the interview end to end and, where possible, emit a Spec IR.

        ``answers`` stands in for the customer: any slot supplied here is
        treated as answered, so a caller can replay an interview or drive it one
        round at a time. Whatever is still unanswered comes back in
        :attr:`Interview.pending` rather than being guessed.
        """
        known: dict[str, Any] = {k: v for k, v in (answers or {}).items() if v is not None}
        probed = self.apply_probe(known, device_profile)

        posterior = self.observe(description, known=known, seed=seed)
        # Anything the description settled on its own is an answer we did not
        # have to ask for.
        for name, entry in posterior.slots.items():
            if name in known or entry.is_empty or entry.is_ambiguous:
                continue
            if entry.confidence >= 0.6:
                known.setdefault(name, entry.mode)

        derived = self.derive(known)
        questions, skipped, stop = self.next_questions(posterior, known)

        asked = [q for q in questions if q.slot in known]
        pending = [q for q in questions if q.slot not in known]
        for q in asked:
            q.answer, q.answered = known[q.slot], True
        if pending and not answers:
            stop = Stop.NO_ANSWERS

        unresolved = sorted(
            name for name, entry in posterior.slots.items()
            if entry.effective_ambiguity > self.max_ambiguity
            and known.get(name) is None
        )
        probe_request = self.probe_token(known, posterior)
        if probe_request is not None and not pending:
            stop = Stop.NEEDS_PROBE

        interview = Interview(
            description=description, posterior=posterior,
            state=self.forge.parse(description, **_parser_known(known)),
            asked=asked, pending=pending, skipped=skipped, derived=derived,
            probed=probed, stopped_because=stop, tier=self.tier,
            oracle_stats=dict(self.oracle.stats),
            unresolved_ambiguities=unresolved, probe_request=probe_request,
        )

        try:
            interview.spec = self._spec_builder(known, description)({})
        except ValueError as exc:
            # A must-ask slot is still empty. That is not a failure of the
            # interview; it is the interview working.
            logger.info("forge: no spec yet — %s", exc)
            return interview

        # §5 condition 4: a spec that type-checks is not yet admissible.
        interview.gate1 = self.check_gate1(interview.spec)
        if interview.gate1 is not None and not interview.gate1.passed:
            # A Gate 1 refusal is FORGE's to explain, not the Planner's (§11):
            # it is remediable by the user, and the interview must say so.
            stop = Stop.GATE_1_FAILED
            interview.stopped_because = stop
            logger.info("forge: gate 1 refused %s", interview.spec.hash[:8])
            return interview
        if unresolved:
            interview.stopped_because = Stop.UNRESOLVED_AMBIGUITY

        if plan_it and interview.spec is not None:
            interview.outcome = self.oracle(interview.spec)
            interview.oracle_stats = dict(self.oracle.stats)
        return interview

    # -- §5 condition 4 and §7's probe token -------------------------------- #
    @staticmethod
    def check_gate1(spec: SpecIR) -> Any:
        """Gate 1: is this request admissible at all?

        §11 splits the two refusals by whose fault they are. A Gate 1 failure —
        seed floor, data rights, unsupported primitive — is **remediable by the
        user**, so FORGE owns explaining it. A Gate 2 failure is about physics
        and budget, and belongs to the Planner.
        """
        try:
            from modelrig.gates import gate1_spec_admissibility

            return gate1_spec_admissibility(spec)
        except Exception as exc:  # noqa: BLE001 - never let the gate break the interview
            logger.warning("forge: gate 1 could not be evaluated: %s", exc)
            return None

    def probe_token(
        self, known: dict[str, Any], posterior: ParsePosterior
    ) -> dict[str, Any] | None:
        """§7: emit a PROBE token instead of a question, when one is warranted.

        Never ask a non-technical user for RAM. The probe collapses the whole
        device group from a posterior to a point mass in one round trip, and
        costs zero questions — so while the device is unprobed it is strictly the
        highest-value next action, and it is not an interview turn.
        """
        if _is_measured(known.get("profile_source")):
            return None
        return {
            "token": f"probe:{posterior.description[:40]!r}",
            "collapses": list(slot_table.PROBED),
            "measures": ["ram_free", "bw_eff", "flops_eff", "thermal_derate",
                         "simd", "storage_free"],
            "costs_questions": 0,
            "why": (
                "these slots are physical, so they are measured rather than "
                "asked; one round trip settles the whole group and licenses a "
                "latency promise the analytical bound cannot make"
            ),
        }

    # -- internals ---------------------------------------------------------- #
    def _spec_builder(
        self, known: dict[str, Any], description: str = ""
    ) -> Callable[[dict[str, Any]], SpecIR]:
        """A closure the IG oracle calls once per candidate answer.

        It raises ``ValueError`` while a must-ask slot is still empty, and
        :func:`~modelrig.forge.infogain.induced_plans` counts that as an outcome
        rather than swallowing it. So before the primitive is known every other
        slot scores zero gain — correctly, since nothing can be planned yet — and
        the interview naturally opens with the must-ask questions.
        """
        base = dict(known)

        def build(overrides: dict[str, Any]) -> SpecIR:
            merged = _parser_known({**base, **overrides})
            return self.forge.to_spec(self.forge.parse(description, **merged), **merged)

        return build

    def _scoring_builder(
        self, known: dict[str, Any], description: str = ""
    ) -> Callable[[dict[str, Any]], SpecIR]:
        """As :meth:`_spec_builder`, but with the *other* unknowns stood in for.

        Measuring ``IG(sigma_i)`` means varying slot ``i`` and watching the plan.
        If some *other* unknown refuses every plan on its own — unstated data
        rights fail ``P_lic`` for every candidate — then every plan is identical,
        every gain is zero, and the interview concludes it should ask nothing at
        all. The masking unknown hides the informative one.

        Two stand-ins fix that, and neither ever reaches an emitted spec:

        * the other slots take their
          :attr:`~modelrig.forge.slots.Slot.scoring_default`;
        * an unprobed device is scored under the analytical latency bound,
          because ``P_lat`` refuses outright on an unmeasured profile (Part 3
          §9) and would otherwise flatten every gain to zero.

        The second is a finding in its own right rather than a workaround: while
        the device is unprobed, *no answer the customer can give* changes the
        outcome. The probe is the highest-value next action and it is not a
        question — which is exactly §10's argument for building one.
        """
        base = {**_scoring_defaults(known), **known}
        if not _is_measured(base.get("profile_source")):
            io = dict(base.get("io_schema") or {})
            io["accept_unmeasured_latency"] = True
            base["io_schema"] = io
        return self._spec_builder(base, description)


def _is_ambiguous(posterior: ParsePosterior, slot: slot_table.Slot) -> bool:
    entry = posterior.get(slot.spec_field)
    return entry is not None and entry.is_ambiguous


def _question(slot: slot_table.Slot, gain: InfoGain, lam: float, gamma: float,
              *, must_ask: bool = False, reason: AskReason = AskReason.DECISION,
              ambiguity: float = 0.0) -> Question:
    if must_ask and not gain.decision_relevant:
        # Not a contradiction: the slot's value shows up after the plan rather
        # than in it, so the marginal rule cannot see it. A build with no agreed
        # acceptance bar and no seed corpus cannot be certified however good the
        # plan looks.
        rationale = (
            "required regardless: the answer does not move the plan, but the "
            "build cannot be certified without it"
        )
    else:
        rationale = gain.reason()
    if reason is AskReason.AMBIGUOUS:
        rationale = (
            f"the description supports more than one reading here (A={ambiguity:.2f}) "
            "— resolving it is worth a question whatever the plan says, because the "
            "alternative is silently picking one"
        )
    return Question(
        slot=slot.spec_field or slot.name, text=slot.question, gain=gain.gain,
        stake=gain.stake, value=gain.gain * gain.stake * lam, threshold=gamma,
        must_ask=must_ask, rationale=rationale, candidates=list(gain.values_tried),
        reason=reason, ambiguity=ambiguity,
    )


_SPEC_FIELDS = set(SpecIR.__dataclass_fields__)


def _scoring_defaults(known: dict[str, Any]) -> dict[str, Any]:
    """Stand-ins for the slots still unknown. Oracle-only — never emitted."""
    return {
        s.spec_field: s.scoring_default
        for s in slot_table.SLOTS
        if s.spec_field and s.scoring_default is not None and known.get(s.spec_field) is None
    }


def _is_measured(source: Any) -> bool:
    if isinstance(source, ProfileSource):
        return source.measured
    if isinstance(source, str):
        try:
            return ProfileSource(source).measured
        except ValueError:
            return False
    return False


def _parser_known(known: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the Spec IR does not carry, so ``to_spec`` never sees noise."""
    return {k: v for k, v in known.items() if k in _SPEC_FIELDS}


def _dotted(known: dict[str, Any]) -> dict[str, Any]:
    """Re-key known answers by dotted slot name, for the ``applies_when`` guards."""
    out: dict[str, Any] = {}
    for field_name, value in known.items():
        slot = slot_table.find(field_name)
        key = slot.name if slot is not None else field_name
        out[key] = value.value if isinstance(value, Enum) else value
    return out


# --------------------------------------------------------------------------- #
def interview(
    description: str,
    *,
    answers: dict[str, Any] | None = None,
    device_profile: dict[str, Any] | None = None,
    tier: Tier = Tier.COMMERCIAL,
    catalog: Any = None,
    k: int = DEFAULT_K,
    seed: int = 0,
) -> Interview:
    """Convenience entry point: one call, one transcript."""
    return Interviewer(catalog=catalog, tier=tier, k=k).conduct(
        description, answers=answers, device_profile=device_profile, seed=seed,
    )


def ask_order(interview_: Interview) -> list[str]:
    """The slots that were or will be asked, in order."""
    return [q.slot for q in (*interview_.asked, *interview_.pending)]


def unasked_because_irrelevant(interview_: Interview) -> list[tuple[str, str]]:
    """``(slot, why)`` for every question the Planner proved was not worth asking."""
    return [(g.slot, g.reason()) for g in interview_.skipped]


def economy(interview_: Interview) -> dict[str, Any]:
    """The §9 accounting, instantiated on a real interview."""
    table = slot_table.economy(_dotted({q.slot: True for q in interview_.asked}))
    table.update({
        "questions_asked": interview_.questions_asked,
        "questions_pending": len(interview_.pending),
        "slots_probed": len(interview_.probed),
        "slots_derived": len(interview_.derived),
        "slots_proved_irrelevant": len(interview_.skipped),
        "completion_probability": round(interview_.completion_probability, 4),
    })
    return table


__all__ = [
    "ATTRITION_GAMMA", "MAX_QUESTIONS",
    "Interview", "Interviewer", "PlanSignature", "Question", "Stop",
    "ask_order", "completion_probability", "economy", "interview",
    "unasked_because_irrelevant", "value_ratio", "worth_asking",
]
