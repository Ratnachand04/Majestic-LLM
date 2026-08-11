"""Context assembly — more context is not better (A-06).

Lost in the Middle (2307.03172) shows a U-shaped positional curve: evidence
placed mid-context is largely ignored. Small models are WORSE at this than the
models studied, and Majestic's whole operating range is small models. Two rules
follow, and both cut against the instinct to stuff the window:

**Cap chunks aggressively.** Retrieving twenty passages and hoping does not
work; the middle ones are dead weight that also cost KV cache — which is the
term that breaks the device memory budget (A-01).

**Place the strongest evidence at the BOUNDARIES.** Rank by score, then
interleave so the best-ranked passages sit at the head and tail of the context
and the weakest fall in the middle where they would be ignored anyway.

Citations are emitted alongside, because a grounded answer the customer cannot
audit is not evidence of anything (GAP-09: trust is the deliverable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

#: Small models degrade fast with context; cap hard rather than maximising.
DEFAULT_MAX_CHUNKS = 5
#: Rough characters-per-token used to budget the window without a tokenizer.
CHARS_PER_TOKEN = 4.0


@dataclass
class Passage:
    """One retrieved chunk with its provenance."""

    text: str
    score: float = 0.0
    source: str = ""
    chunk_id: str = ""

    @property
    def approx_tokens(self) -> int:
        return max(1, int(len(self.text) / CHARS_PER_TOKEN))


@dataclass
class AssembledContext:
    """The context actually handed to the model, plus what was dropped and why."""

    passages: list[Passage] = field(default_factory=list)
    dropped: list[Passage] = field(default_factory=list)
    approx_tokens: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)

    def render(self, separator: str = "\n\n") -> str:
        """Render with citation markers the customer can follow back."""
        return separator.join(
            f"[{i + 1}] {p.text}" for i, p in enumerate(self.passages)
        )

    @property
    def order_explanation(self) -> str:
        if len(self.passages) <= 2:
            return "too few passages for boundary placement to matter"
        return (
            "strongest evidence placed at the context boundaries; weakest in the "
            "middle, where a U-shaped positional curve means it is ignored anyway"
        )


def boundary_order(passages: Sequence[Passage]) -> list[Passage]:
    """Reorder so the strongest evidence sits at the head and tail.

    Ranked ``[a, b, c, d, e]`` (a strongest) becomes ``[a, c, e, d, b]``: the top
    two occupy the first and last positions, and the weakest sink to the middle.
    """
    ranked = sorted(passages, key=lambda p: -p.score)
    head: list[Passage] = []
    tail: list[Passage] = []
    for i, passage in enumerate(ranked):
        (head if i % 2 == 0 else tail).append(passage)
    return head + list(reversed(tail))


def assemble(
    passages: Sequence[Passage],
    *,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    token_budget: int | None = None,
    min_score: float = 0.0,
) -> AssembledContext:
    """Cap, filter, order for the boundaries, and emit citations.

    ``token_budget`` is a real budget, not a target: context competes with the KV
    cache for the same device RAM, so unused budget is a saving rather than
    waste.
    """
    kept = [p for p in passages if p.score >= min_score]
    dropped = [p for p in passages if p.score < min_score]

    ranked = sorted(kept, key=lambda p: -p.score)
    if len(ranked) > max_chunks:
        dropped.extend(ranked[max_chunks:])
        ranked = ranked[:max_chunks]

    if token_budget is not None:
        budgeted: list[Passage] = []
        used = 0
        for passage in ranked:
            if used + passage.approx_tokens > token_budget:
                dropped.append(passage)
                continue
            budgeted.append(passage)
            used += passage.approx_tokens
        ranked = budgeted

    ordered = boundary_order(ranked)
    context = AssembledContext(
        passages=ordered,
        dropped=dropped,
        approx_tokens=sum(p.approx_tokens for p in ordered),
        citations=[
            {"marker": i + 1, "source": p.source or p.chunk_id or "unknown",
             "score": round(p.score, 4)}
            for i, p in enumerate(ordered)
        ],
    )
    logger.info(
        "context: kept %d/%d passages (~%d tokens), dropped %d",
        len(ordered), len(passages), context.approx_tokens, len(dropped),
    )
    return context
