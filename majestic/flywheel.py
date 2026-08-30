"""The flywheel — field corrections become the next generation (B-10).

    01 field correction   staff press thumbs-down or fix a field; no labelling effort
    02 queue locally      sync when online; works inside the offline guarantee
    03 KTO on binary      thumbs up/down is exactly the unpaired signal KTO consumes
    04 rebuild + re-eval  against a held-out set that has GROWN with real data
    05 offer v(n+1)       only if it WINS; the customer is never given a regression

Why this loop is the strategy, not a feature: escalated queries and corrected
fields are the hard cases the local model could not handle. Accumulated across
years and customers, that corpus is the one asset a competitor cannot clone. The
interface is a month of work, the compiler six months of lead, the per-vertical
eval library eighteen months — the failure corpus is permanent. Every other part
of Majestic is a race; this is the only part that compounds.

GAP-05 (open): nobody has characterised narrow-cartridge quality across twenty
retraining generations on accumulating corrections. The flywheel could silently
degrade models over years before anyone notices, which is why
:meth:`Flywheel.propose_next_version` refuses a regression and
:class:`GenerationLog` keeps the whole history auditable.


**Why retraining uses human corrections and never model outputs (I-03).**

This is where the Curse of Recursion (2305.17493) actually applies. If
generation ``g+1`` trained on generation ``g``'s *outputs*, that is exactly the
recursive structure the paper describes: distribution tails lost, compounding
across generations, and the damage is not recoverable by generating more.

Part 8 §19 corrects the rest of the series on this point — single-build
amplification is anchored on real seeds and is therefore subject to saturation
rather than collapse. The flywheel is the one place where the feedback loop is
real, which is why the invariant forbidding it belongs here and the citation is
precisely on point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

Example = tuple[str, str]


@dataclass
class Correction:
    """One field correction or thumbs signal from the field.

    ``approved`` is the unpaired binary signal: thumbs up/down is exactly the
    data KTO (2402.01306) consumes, and it costs staff no labelling effort.
    """

    query: str
    model_output: str
    approved: bool
    corrected_output: str | None = None
    capability: str = "unknown"
    synced: bool = False

    @property
    def is_labelled(self) -> bool:
        """A correction with a fixed value is a training pair, not just a signal."""
        return self.corrected_output is not None


@dataclass
class Generation:
    """One turn of the wheel."""

    version: int
    score: float
    held_out_size: int
    accepted: bool
    reason: str = ""


@dataclass
class GenerationLog:
    """The audit trail across generations — the GAP-05 early-warning system."""

    generations: list[Generation] = field(default_factory=list)

    def add(self, generation: Generation) -> None:
        self.generations.append(generation)

    @property
    def best_score(self) -> float:
        accepted = [g.score for g in self.generations if g.accepted]
        return max(accepted) if accepted else 0.0

    def degrading(self, window: int = 3) -> bool:
        """True when accepted scores have trended down across ``window`` turns."""
        accepted = [g.score for g in self.generations if g.accepted]
        if len(accepted) < window:
            return False
        recent = accepted[-window:]
        return all(b < a for a, b in zip(recent, recent[1:]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "generations": [
                {"version": g.version, "score": g.score, "held_out_size": g.held_out_size,
                 "accepted": g.accepted, "reason": g.reason}
                for g in self.generations
            ],
            "best_score": self.best_score,
            "degrading": self.degrading(),
        }


class Flywheel:
    """Collects corrections, grows the held-out set, and gates every new version."""

    def __init__(self, min_corrections: int = 20) -> None:
        self.pending: list[Correction] = []
        self.synced: list[Correction] = []
        self.log = GenerationLog()
        self.min_corrections = min_corrections
        self.current_version = 1
        self.current_score = 0.0

    # -- 01/02: capture and sync ------------------------------------------ #
    def record(self, correction: Correction) -> None:
        """Queue a correction locally. Works inside the offline guarantee."""
        self.pending.append(correction)

    def sync(self) -> int:
        """Flush the local queue when the device next has connectivity."""
        for c in self.pending:
            c.synced = True
        self.synced.extend(self.pending)
        count, self.pending = len(self.pending), []
        logger.info("flywheel: synced %d corrections", count)
        return count

    @property
    def ready(self) -> bool:
        """Enough signal accumulated to justify a rebuild."""
        return len(self.synced) >= self.min_corrections

    # -- 03: binary-signal dataset ---------------------------------------- #
    def kto_dataset(self) -> tuple[list[Example], list[Example]]:
        """Split synced corrections into (desirable, undesirable) sets.

        KTO consumes unpaired binary feedback directly — no preference pairs and
        no labelling effort are required.
        """
        desirable: list[Example] = []
        undesirable: list[Example] = []
        for c in self.synced:
            if c.approved:
                desirable.append((c.query, c.model_output))
            elif c.is_labelled:
                # A corrected field is a positive example of the RIGHT answer.
                desirable.append((c.query, c.corrected_output or ""))
                undesirable.append((c.query, c.model_output))
            else:
                undesirable.append((c.query, c.model_output))
        return desirable, undesirable

    def grow_held_out(self, held_out: Sequence[Example]) -> list[Example]:
        """Grow the evaluation set with REAL corrected data.

        Each generation is judged against a harder, more realistic bar than the
        last — which is what stops the flywheel from congratulating itself.
        """
        grown = list(held_out)
        seen = {q for q, _ in grown}
        for c in self.synced:
            if c.is_labelled and c.query not in seen:
                grown.append((c.query, c.corrected_output or ""))
                seen.add(c.query)
        return grown

    # -- 04/05: rebuild and offer only if it wins -------------------------- #
    def propose_next_version(
        self,
        rebuild: Callable[[list[Example], list[Example]], float],
        held_out: Sequence[Example],
        margin: float = 0.0,
    ) -> Generation:
        """Rebuild, re-evaluate on the GROWN held-out set, and gate the result.

        ``rebuild`` receives ``(training_examples, grown_held_out)`` and returns
        the new version's score. The new version is offered ONLY if it beats the
        incumbent by ``margin``; the customer is never given a regression.
        """
        desirable, undesirable = self.kto_dataset()
        grown = self.grow_held_out(held_out)
        score = float(rebuild(desirable, grown))

        wins = score > self.current_score + margin
        generation = Generation(
            version=self.current_version + 1,
            score=round(score, 4),
            held_out_size=len(grown),
            accepted=wins,
            reason=(
                f"v{self.current_version + 1} scored {score:.4f} vs incumbent "
                f"{self.current_score:.4f}"
                + ("" if wins else " — withheld, the customer is never given a regression")
            ),
        )
        self.log.add(generation)

        if wins:
            self.current_version += 1
            self.current_score = score
            self.synced = []  # consumed into this generation
            logger.info("flywheel: accepted v%d at %.4f", self.current_version, score)
        else:
            logger.info("flywheel: withheld v%d (%.4f <= %.4f)",
                        generation.version, score, self.current_score)

        if self.log.degrading():
            logger.warning(
                "flywheel: accepted scores trending DOWN across generations — the "
                "multi-generation degradation GAP-05 warns about"
            )
        _ = undesirable  # retained for a real KTO trainer; unused offline
        return generation

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "current_version": self.current_version,
                    "current_score": self.current_score,
                    "synced": [
                        {"query": c.query, "model_output": c.model_output,
                         "approved": c.approved, "corrected_output": c.corrected_output,
                         "capability": c.capability}
                        for c in self.synced
                    ],
                    "log": self.log.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return p
