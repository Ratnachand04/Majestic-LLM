"""FORGE — from plain language to a typed specification (B-04, Part 4).

The non-technical front door, and the module with the least prior art to borrow
from. Slot-filling dialogue systems are a solved problem for *closed* domains
where every slot is enumerable; this domain is not closed, and the schema
mixes preference, physics, law and statistics in one artefact.

Four ideas hold it together, one per module:

``slots``      the schema, annotated with where each value comes from. Elicited
               only if it is a human preference that cannot be measured or
               computed; probed if it is physical; derived if it is implied.
               That ordering is the entire economy of the interview (§9-§11).
``posterior``  ``parse_K`` — parse repeatedly and keep the *distribution*, which
               separates a slot the description never mentioned from one it
               mentioned ambiguously. Different problems, different fixes (§2).
``infogain``   score each candidate question by pushing its answers through the
               **Planner** and measuring how far the plan moves. The compiler
               is the information-gain oracle, and no other module in the
               pipeline gets reused this way (§3).
``core``       the marginal rule ``IG * stake * Lambda > gamma`` and the §8
               algorithm. Every question costs attrition, so the interview stops
               when the next question is worth less than the customer it loses.

The parser in :mod:`modelrig.forge.parser` is deterministic and offline. An LLM
front end can replace :meth:`Forge.parse` without disturbing anything
downstream, because the output contract is the Spec IR — and because everything
above reads the posterior, not the parser.
"""
from modelrig.forge.core import (
    ATTRITION_GAMMA,
    MAX_QUESTIONS,
    Interview,
    Interviewer,
    Question,
    Stop,
    ask_order,
    completion_probability,
    economy,
    interview,
    unasked_because_irrelevant,
    value_ratio,
    worth_asking,
)
from modelrig.forge.infogain import (
    DELTA_SHARE,
    InfoGain,
    PlanOracle,
    PlanSignature,
    induced_plans,
    information_gain,
    rank,
    zero_gain,
)
from modelrig.forge.parser import Forge, InterviewState, Slot
from modelrig.forge.posterior import (
    AMBIGUITY_THRESHOLD,
    DEFAULT_K,
    ParsePosterior,
    SlotPosterior,
    candidate_values,
    parse_K,
    project_onto_domain,
    summarise,
)
from modelrig.forge.slots import (
    ELICITED,
    MUST_ASK,
    PROBED,
    SLOTS,
    AskPolicy,
    IGClass,
    Source,
    askable,
    validate_table,
)

__all__ = [
    # the parser and its state — the surface the rest of the pipeline imports
    "Forge", "InterviewState", "Slot",
    # the slot table
    "SLOTS", "MUST_ASK", "PROBED", "ELICITED", "AskPolicy", "IGClass", "Source",
    "askable", "validate_table",
    # the posterior
    "AMBIGUITY_THRESHOLD", "DEFAULT_K", "ParsePosterior", "SlotPosterior",
    "candidate_values", "parse_K", "project_onto_domain", "summarise",
    # information gain, measured through the Planner
    "DELTA_SHARE", "InfoGain", "PlanOracle", "PlanSignature",
    "induced_plans", "information_gain", "rank", "zero_gain",
    # the interview
    "ATTRITION_GAMMA", "MAX_QUESTIONS", "Interview", "Interviewer", "Question",
    "Stop", "ask_order", "completion_probability", "economy", "interview",
    "unasked_because_irrelevant", "value_ratio", "worth_asking",
]
