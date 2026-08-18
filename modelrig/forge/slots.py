"""The Spec IR slot table (Part 4 §9-§11).

**The design principle:** a field is elicited only if it encodes a human
preference that cannot be measured or computed. Everything physical is probed.
Everything implied is derived.

    elicited   asked of the human      costs attrition      MINIMISE
    probed     measured on the device  one round trip       prefer
    derived    computed from others    free                 MAXIMISE

That ordering is the whole economy of the interview. Nine slots are `must_ask`,
but the probe collapses the device group at once and derivation handles six
more — so a well-parsed description typically leaves **three to five** questions
to actually put to a human, which is inside the attrition budget of §4.

The table is a data structure, not a schema definition: it says where each field
comes from and whether it is worth a question. The types live in
:mod:`modelrig.ir`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

from modelrig.licence import DataRights
from modelrig.primitives import TaskPrimitive


class Source(str, Enum):
    """Where a slot's value comes from. Exactly one per field."""

    ELICITED = "elicited"   # asked of the human — the only one that costs attrition
    PROBED = "probed"       # measured on the device
    DERIVED = "derived"     # computed from other fields


class AskPolicy(str, Enum):
    """When, if ever, this slot becomes a question.

    ``MUST_ASK`` exists for a specific reason: some slots are indispensable and
    still score *zero* information gain, because their value shows up after the
    plan rather than in it. ``quality.gate_threshold`` is the clean case — the
    Planner ranks on predicted quality against ``theta*``, and the customer's
    gate is what Gate 3 certifies against, so varying it moves no plan at all
    and the marginal rule would drop it. A build with no agreed acceptance bar
    cannot be certified, so it is asked regardless.
    """

    MUST_ASK = "must_ask"           # bypasses the marginal rule entirely
    IF_AMBIGUOUS = "if_ambiguous"   # only when resampling disagrees
    CONDITIONAL = "conditional"     # only when a predicate over the spec holds
    CONFIRM_ONLY = "confirm_only"   # a default is offered; confirm rather than elicit
    NEVER = "never"                 # probed or derived


class IGClass(str, Enum):
    """A coarse prior on decision-relevance, before the Planner is consulted.

    Used to order candidates cheaply and to sanity-check the sampled estimate.
    The real number comes from :mod:`modelrig.forge.infogain`.
    """

    ZERO = "zero"           # cannot change the plan; never ask
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    HIGHEST = "highest"

    @property
    def prior(self) -> float:
        return {"zero": 0.0, "low": 0.15, "moderate": 0.4,
                "high": 0.7, "highest": 1.0}[self.value]


@dataclass(frozen=True)
class Slot:
    """One field of the Spec IR, with its provenance and ask policy."""

    name: str
    kind: str                       # a short type tag, for the parser's domain
    source: Source
    ask: AskPolicy
    ig_class: IGClass
    #: The values this slot may take. A *constraint*: constrained decoding
    #: projects every parse onto it, so an out-of-domain sample is discarded
    #: rather than allowed to win a vote. Empty where the slot is continuous.
    domain: tuple[Any, ...] = ()
    #: Representative points for the information-gain oracle on a continuous
    #: slot. NOT a constraint — the customer may say anything; these are simply
    #: the values pushed through the Planner to see whether the answer moves the
    #: plan. Without them a continuous slot would score zero gain for the
    #: circular reason that nothing was enumerated from it.
    probe_values: tuple[Any, ...] = ()
    #: A stand-in used **only inside the information-gain oracle**, never to
    #: emit a spec. Scoring slot ``i`` means asking "if everything else were
    #: settled, would this answer move the plan?" — and without a stand-in a
    #: single blocking unknown (unstated data rights refuse every plan on
    #: ``P_lic``) drives every *other* slot's gain to zero and the interview
    #: concludes it should ask nothing. That is the precise opposite of correct.
    #: This is a counterfactual, not a commitment: :meth:`Forge.to_spec` still
    #: refuses rather than defaulting.
    scoring_default: Any = None
    question: str = ""
    note: str = ""
    #: The :class:`~modelrig.ir.SpecIR` attribute this slot lands in, where one
    #: exists. Empty for slots that live inside ``io_schema`` or that shape the
    #: build without being carried on the spec (tone, licence allowlist, tier).
    #: The table is the single source of truth for this mapping.
    spec_field: str = ""
    #: For CONDITIONAL slots: when does this become worth asking at all?
    applies_when: Callable[[dict[str, Any]], bool] | None = None
    #: For DERIVED slots: what it is computed from, for the dependency graph.
    derived_from: tuple[str, ...] = ()

    @property
    def costs_attrition(self) -> bool:
        return self.source is Source.ELICITED and self.ask is not AskPolicy.NEVER

    @property
    def must_ask(self) -> bool:
        return self.ask is AskPolicy.MUST_ASK

    def applicable(self, partial: dict[str, Any]) -> bool:
        if self.applies_when is None:
            return True
        try:
            return bool(self.applies_when(partial))
        except Exception:  # noqa: BLE001 - a broken guard must not block the interview
            return True


_PRIMITIVES = tuple(p.value for p in TaskPrimitive)
#: Ordered so the *extremes* come first. The oracle pushes only the first few
#: values through the Planner, and omitting the absorbing element
#: (``no_training``, which fails ``P_lic`` outright) would make the whole slot
#: read as zero-gain — the one reading where the answer matters most.
_DATA_RIGHTS = ("customer_owned", "no_training", "licensed", "public_domain", "unknown")
_TONE_PRIMITIVES = {TaskPrimitive.REWRITE.value, TaskPrimitive.GENERATE.value,
                    TaskPrimitive.SUMMARISE.value}


def _tone_applies(partial: dict[str, Any]) -> bool:
    """Tone enters a drafting plan and not an extraction one (§10).

    This is the clearest example of why entropy is the wrong ask-criterion: tone
    can be maximally uncertain and still have exactly zero information gain,
    because no coordinate of an extraction plan reads it.
    """
    return str(partial.get("task.primitive", "")) in _TONE_PRIMITIVES


#: The table. Order is presentation order, not ask order.
SLOTS: tuple[Slot, ...] = (
    # --- task ---------------------------------------------------------- #
    Slot("task.primitive", "enum", Source.ELICITED, AskPolicy.MUST_ASK,
         IGClass.HIGHEST, domain=_PRIMITIVES,
         question="What should the model DO with the input — extract fields, "
                  "classify it, answer questions about it, or draft a reply?",
         spec_field="task_primitive"),
    Slot("task.io_contract.output_schema", "json_schema", Source.ELICITED,
         AskPolicy.MUST_ASK, IGClass.HIGH, scoring_default={"field": "string"},
         question="What fields or labels must come out, exactly?",
         spec_field="io_schema"),
    Slot("task.languages", "list", Source.ELICITED, AskPolicy.IF_AMBIGUOUS,
         IGClass.MODERATE,
         question="Which languages must it handle?",
         spec_field="languages"),

    # --- data ----------------------------------------------------------- #
    Slot("data.seed_ref", "ref", Source.ELICITED, AskPolicy.MUST_ASK, IGClass.HIGHEST,
         scoring_default="seed://not-yet-supplied",
         question="Can you upload real examples? We need them to build and, "
                  "separately, to prove the result.",
         spec_field="seed_data_ref"),
    Slot("data.seed_count", "int", Source.ELICITED, AskPolicy.MUST_ASK, IGClass.HIGHEST,
         probe_values=(25, 200, 1000), scoring_default=200,
         question="How many real examples do you have?",
         spec_field="seed_data_count"),
    Slot("data.holdout_count", "int", Source.DERIVED, AskPolicy.NEVER, IGClass.ZERO,
         derived_from=("data.seed_count",),
         note="a fixed share of the seed corpus, locked and never trained on"),
    Slot("data.rights", "enum", Source.ELICITED, AskPolicy.MUST_ASK, IGClass.HIGH,
         domain=_DATA_RIGHTS, scoring_default="customer_owned",
         question="Is the example data yours to train on?",
         note="gates the licence lattice; an absorbing restriction here is fatal",
         spec_field="data_rights"),

    # --- target: probed, not asked -------------------------------------- #
    Slot("target.device_profile", "object", Source.PROBED, AskPolicy.NEVER,
         IGClass.HIGHEST, spec_field="device_profile",
         note="never ask a non-technical user for RAM — the probe collapses four "
              "high-IG slots at once and costs zero questions"),
    Slot("target.device_class", "enum", Source.ELICITED, AskPolicy.CONFIRM_ONLY,
         IGClass.HIGH, spec_field="device_target",
         probe_values=("android_lowend", "android_tablet_4gb", "laptop_cpu"),
         question="Which device will run this? (Scan the QR probe and we will "
                  "detect RAM and chip automatically.)",
         note="a prior over the real thing. A probe supersedes it entirely and "
              "removes the question, which is why the probe is worth a round trip"),
    Slot("target.offline_required", "bool", Source.ELICITED, AskPolicy.MUST_ASK,
         IGClass.HIGHEST, domain=(True, False), spec_field="offline_required",
         scoring_default=False,
         question="Must it work with no internet ALWAYS, or only survive the "
                  "network dropping occasionally?",
         note="partitions the feasible set; the canonical ambiguity"),
    Slot("target.latency_budget_ms", "int", Source.ELICITED, AskPolicy.MUST_ASK,
         IGClass.HIGH, spec_field="latency_budget_ms",
         probe_values=(1_000, 30_000, 300_000), scoring_default=30_000,
         question="How fast must a single result come back?",
         note="elicited then sanity-checked against the roofline"),
    Slot("target.expected_input_tokens", "int", Source.ELICITED, AskPolicy.MUST_ASK,
         IGClass.HIGHEST, spec_field="expected_input_tokens",
         probe_values=(20, 500, 4_000), scoring_default=500,
         question="Roughly how long is a typical input — a page, a paragraph, a line?",
         note="§17: a FIRST-ORDER latency driver. Derive it from sample documents "
              "where possible (95th percentile) and save the question"),
    Slot("target.expected_output_tokens", "int", Source.DERIVED, AskPolicy.NEVER,
         IGClass.MODERATE, derived_from=("task.io_contract.output_schema",),
         spec_field="expected_output_tokens",
         note="derivable from the size of the output schema"),
    Slot("target.expected_daily_volume", "int", Source.ELICITED, AskPolicy.CONFIRM_ONLY,
         IGClass.MODERATE, spec_field="expected_daily_volume",
         probe_values=(10, 200, 2_000),
         question="Roughly how many of these per day?",
         note="needed for the energy predicate"),
    Slot("target.context_budget", "int", Source.DERIVED, AskPolicy.NEVER, IGClass.ZERO,
         derived_from=("target.device_profile", "task.primitive"),
         spec_field="context_budget",
         note="§13.1 inverted for context; fully determined by device and base"),

    # --- quality --------------------------------------------------------- #
    Slot("quality.gate_metric", "enum", Source.DERIVED, AskPolicy.NEVER, IGClass.ZERO,
         derived_from=("task.primitive",),
         note="each primitive has one natural metric; asking would be noise"),
    Slot("quality.gate_threshold", "float", Source.ELICITED, AskPolicy.MUST_ASK,
         IGClass.HIGH, spec_field="quality_gate", probe_values=(0.80, 0.90, 0.98),
         scoring_default=0.90,
         question="What accuracy would make this worth deploying?"),
    Slot("quality.flip_rate_max", "float", Source.DERIVED, AskPolicy.NEVER, IGClass.LOW,
         derived_from=("task.primitive",),
         note="a policy default; the customer has no basis to set it"),

    # --- behaviour -------------------------------------------------------- #
    Slot("behaviour.abstention_policy", "enum", Source.ELICITED, AskPolicy.MUST_ASK,
         IGClass.MODERATE, domain=("flag", "abstain", "escalate", "guess"),
         spec_field="abstention_policy", scoring_default="flag",
         question="When the model is unsure, should it flag for review or answer anyway?"),
    Slot("behaviour.tone", "string", Source.ELICITED, AskPolicy.CONDITIONAL,
         IGClass.ZERO, applies_when=_tone_applies,
         question="What voice should it write in?",
         note="ZERO information gain for extraction: no plan coordinate reads it"),

    # --- policy ----------------------------------------------------------- #
    Slot("policy.regulated_domain", "enum", Source.ELICITED, AskPolicy.MUST_ASK,
         IGClass.HIGH, domain=("none", "medical", "financial", "legal"),
         question="Is this a regulated domain — medical, financial, legal?",
         note="sets kappa, which governs BOTH the refusal threshold and how many "
              "questions are worth asking"),

    # --- runtime, licence, budget ------------------------------------------ #
    Slot("runtime.mode", "enum", Source.DERIVED, AskPolicy.NEVER, IGClass.ZERO,
         derived_from=("target.offline_required",),
         note="offline implies the mode; asking both invites contradiction"),
    Slot("licence.allowlist", "list", Source.ELICITED, AskPolicy.CONFIRM_ONLY,
         IGClass.MODERATE, spec_field="policy_rules",
         question="Any licence restrictions we should honour beyond the defaults?"),
    Slot("budget.max_build_micro_usd", "int", Source.ELICITED, AskPolicy.CONFIRM_ONLY,
         IGClass.MODERATE, spec_field="budget_ceiling_usd",
         probe_values=(10.0, 40.0, 200.0),
         question="What is your budget ceiling for this build?",
         note="a tier default; confirm rather than elicit"),
)

BY_NAME: dict[str, Slot] = {s.name: s for s in SLOTS}
BY_FIELD: dict[str, Slot] = {s.spec_field: s for s in SLOTS if s.spec_field}

#: Slots whose default would be the most expensive thing FORGE can get wrong.
#: Their IG is high by construction and they bypass the marginal rule entirely.
MUST_ASK: tuple[str, ...] = tuple(s.name for s in SLOTS if s.must_ask)

#: The group a single probe collapses from posterior to point mass (§7).
PROBED: tuple[str, ...] = tuple(s.name for s in SLOTS if s.source is Source.PROBED)

DERIVED: tuple[str, ...] = tuple(s.name for s in SLOTS if s.source is Source.DERIVED)
ELICITED: tuple[str, ...] = tuple(s.name for s in SLOTS if s.source is Source.ELICITED)


def get(name: str) -> Slot:
    """Look a slot up by dotted name or by its ``SpecIR`` field name."""
    if name in BY_NAME:
        return BY_NAME[name]
    if name in BY_FIELD:
        return BY_FIELD[name]
    raise KeyError(f"unknown slot {name!r}")


def find(name: str) -> Slot | None:
    """As :func:`get`, but ``None`` for names the table does not carry."""
    return BY_NAME.get(name) or BY_FIELD.get(name)


def askable(partial: dict[str, Any] | None = None) -> list[Slot]:
    """Slots that could become a question given what is known so far."""
    partial = partial or {}
    return [
        s for s in SLOTS
        if s.costs_attrition and s.ask is not AskPolicy.NEVER and s.applicable(partial)
    ]


def unfilled_must_ask(partial: dict[str, Any]) -> list[str]:
    """Which required slots are still empty."""
    return [n for n in MUST_ASK if partial.get(n) is None]


def economy(partial: dict[str, Any] | None = None) -> dict[str, Any]:
    """The §9 accounting: how many questions this table actually implies."""
    partial = partial or {}
    return {
        "total_slots": len(SLOTS),
        "elicited": len(ELICITED),
        "probed": len(PROBED),
        "derived": len(DERIVED),
        "must_ask": len(MUST_ASK),
        "askable_now": len(askable(partial)),
        "principle": (
            "a field is elicited only if it encodes a human preference that "
            "cannot be measured or computed"
        ),
    }


def dependency_edges() -> list[tuple[str, str]]:
    """``(source, derived)`` pairs — the derivation graph of §14."""
    return [(src, s.name) for s in SLOTS for src in s.derived_from]


def validate_table(slots: Iterable[Slot] = SLOTS) -> list[str]:
    """Invariants the table itself must satisfy. Empty means sound."""
    problems: list[str] = []
    seen: set[str] = set()
    for s in slots:
        if s.name in seen:
            problems.append(f"duplicate slot {s.name}")
        seen.add(s.name)
        if s.source is not Source.ELICITED and s.ask is not AskPolicy.NEVER:
            problems.append(
                f"{s.name}: {s.source.value} slots must never be asked — "
                "probing or deriving is the whole point"
            )
        if s.source is Source.DERIVED and not s.derived_from:
            problems.append(f"{s.name}: derived but declares no source fields")
        if s.must_ask and s.ig_class is IGClass.ZERO:
            problems.append(
                f"{s.name}: must_ask with zero information gain is a contradiction"
            )
        if s.costs_attrition and not s.question:
            problems.append(f"{s.name}: elicited but carries no question text")
        if s.domain and s.probe_values:
            problems.append(
                f"{s.name}: declares both a domain and probe values — a slot with "
                "an enumerable domain has nothing left to probe"
            )
        if s.ask is AskPolicy.CONDITIONAL and s.applies_when is None:
            problems.append(f"{s.name}: conditional but declares no guard")
    if set(_DATA_RIGHTS) != {d.value for d in DataRights}:
        problems.append(
            "data.rights: the hand-ordered domain has drifted from DataRights — "
            "a rights value the oracle never tries is a rights value it will "
            "report as costing nothing to leave unanswered"
        )
    return problems
