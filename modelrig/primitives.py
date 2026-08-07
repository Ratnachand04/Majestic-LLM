"""The eight task primitives — the closed set the compiler supports.

Per B-05 ("the eight structural task types") and B-03 Gate 1 ("is the primitive
inside the supported set of eight?"). The taxonomy is organised by **shared
training recipe and I/O contract**, not by semantics — two requests belong to
the same primitive when they compile to the same data recipe, grammar shape and
eval suite.

GAP-06 (open): the eight-primitive compression has NOT yet been validated
against real incoming specifications, and the published coverage rate is
unknown. If coverage turns out low the taxonomy is wrong and the planner must be
redesigned. Treat this set as provisional and measure
:func:`coverage_report` against real traffic before relying on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskPrimitive(str, Enum):
    """The supported structural task types. Anything outside this set is refused."""

    EXTRACT = "extract"        # document -> typed fields
    CLASSIFY = "classify"      # text -> label(s) from a closed set
    SUMMARISE = "summarise"    # long text -> short text, same language
    REWRITE = "rewrite"        # text -> transformed text (translate, restyle, reformat)
    GENERATE = "generate"      # brief -> drafted text
    ANSWER = "answer"          # question + retrieved context -> grounded answer
    ROUTE = "route"            # input -> routing decision / triage label + reason
    TOOLCALL = "toolcall"      # request -> constrained tool invocation


@dataclass(frozen=True)
class PrimitiveSpec:
    """Compile-time properties of one primitive.

    Attributes
    ----------
    seed_floor:
        Minimum number of REAL customer examples required before a build is
        admissible. Below this the Data Factory refuses rather than amplifying
        into model collapse (B-06, Curse of Recursion 2305.17493).
    default_metric:
        Metric the Proving Ground uses when the spec does not name one.
    structured_output:
        Whether the primitive emits a schema-constrained output, in which case a
        grammar is compiled into the cartridge's I/O contract (B-08 slot 4).
    min_params_b:
        Smallest base (billions of parameters) at which this primitive has been
        observed to clear a commercial gate. Used by Gate 1 to reject a quality
        bar that is unachievable at the requested device tier.
    max_step_depth:
        Maximum agentic step depth the primitive may request. Sub-2B models
        degrade badly across multi-step loops (A-09), so the planner refuses
        specs whose depth exceeds what the device tier sustains.
    """

    primitive: TaskPrimitive
    seed_floor: int
    default_metric: str
    structured_output: bool
    min_params_b: float
    max_step_depth: int
    description: str


_SPECS: dict[TaskPrimitive, PrimitiveSpec] = {
    TaskPrimitive.EXTRACT: PrimitiveSpec(
        TaskPrimitive.EXTRACT, 120, "field_f1", True, 0.6, 1,
        "Pull typed fields out of a document into a fixed schema.",
    ),
    TaskPrimitive.CLASSIFY: PrimitiveSpec(
        TaskPrimitive.CLASSIFY, 80, "accuracy", True, 0.35, 1,
        "Assign one or more labels from a closed set.",
    ),
    TaskPrimitive.SUMMARISE: PrimitiveSpec(
        TaskPrimitive.SUMMARISE, 150, "rouge_l", False, 1.5, 1,
        "Compress a long document into a short one in the same language.",
    ),
    TaskPrimitive.REWRITE: PrimitiveSpec(
        TaskPrimitive.REWRITE, 150, "chrf", False, 1.5, 1,
        "Transform text: translate, restyle, or reformat while preserving meaning.",
    ),
    TaskPrimitive.GENERATE: PrimitiveSpec(
        TaskPrimitive.GENERATE, 200, "judge_score", False, 1.7, 1,
        "Draft new text from a brief, in the customer's voice.",
    ),
    TaskPrimitive.ANSWER: PrimitiveSpec(
        TaskPrimitive.ANSWER, 150, "groundedness", False, 1.7, 2,
        "Answer a question from retrieved context, with citations.",
    ),
    TaskPrimitive.ROUTE: PrimitiveSpec(
        TaskPrimitive.ROUTE, 80, "accuracy", True, 0.35, 1,
        "Decide where a request goes and why: triage, dispatch, escalation.",
    ),
    TaskPrimitive.TOOLCALL: PrimitiveSpec(
        TaskPrimitive.TOOLCALL, 200, "call_exact_match", True, 1.7, 3,
        "Emit a schema-constrained call against a registered tool.",
    ),
}

assert len(_SPECS) == 8, "the supported set is exactly eight primitives"


def spec_for(primitive: TaskPrimitive | str) -> PrimitiveSpec:
    """Return the :class:`PrimitiveSpec` for a primitive.

    Raises ``ValueError`` for anything outside the supported set of eight — this
    is the check behind Gate 1 admissibility.
    """
    if isinstance(primitive, str):
        try:
            primitive = TaskPrimitive(primitive.lower())
        except ValueError as exc:
            raise ValueError(
                f"{primitive!r} is not one of the supported primitives: "
                f"{[p.value for p in TaskPrimitive]}"
            ) from exc
    return _SPECS[primitive]


def all_primitives() -> list[PrimitiveSpec]:
    """Every supported primitive, in declaration order."""
    return list(_SPECS.values())


def coverage_report(requests: list[str | None]) -> dict[str, float | int]:
    """Measure what fraction of real requests map onto the closed set (GAP-06).

    ``requests`` is a list of primitive names as classified by FORGE, with
    ``None`` for requests that could not be mapped. Publishing this number is the
    validation GAP-06 demands; a low rate means the taxonomy is wrong.
    """
    total = len(requests)
    covered = sum(1 for r in requests if r and _is_supported(r))
    return {
        "total": total,
        "covered": covered,
        "uncovered": total - covered,
        "coverage_rate": round(covered / total, 4) if total else 0.0,
    }


def _is_supported(name: str) -> bool:
    try:
        TaskPrimitive(name.lower())
    except ValueError:
        return False
    return True
