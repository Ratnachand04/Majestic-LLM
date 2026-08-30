"""Learning which questions were worth asking (Part 4 §6).

STaR-GATE's loop with a much better reward. Elicitation research optimises
against proxy rewards — a judge's opinion of whether the interview *seemed*
thorough — because nothing downstream can settle it. Here something can:

    R(question sequence) = 1[gate passed first attempt] - gamma * q
                           - 1[customer rejected]

The reward is **objective and delayed**. Did the model this interview produced
pass its gate on the first attempt, and did the customer accept it. No judge is
involved and no proxy is needed.

**The diagnostic that matters.** Track first-attempt pass rate against mean
question count, and watch them together:

    if pass rate rises AND q_bar falls   -> the policy is learning
    if pass rate rises AND q_bar rises   -> it is just asking more

The second is not learning. It is paying the attrition cost without counting it,
and it looks like success on every dashboard that plots pass rate alone. That is
why :meth:`OutcomeLog.trend` refuses to report one without the other.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from majestic.logging_utils import get_logger
from modelrig.forge.core import ATTRITION_GAMMA

logger = get_logger(__name__)

#: Interviews per window when reporting a trend. Below this the two rates are
#: too noisy to compare and the diagnostic says so rather than guessing.
MIN_WINDOW = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class InterviewOutcome:
    """One completed interview and what became of the build it specified."""

    spec_hash: str
    questions_asked: int
    slots_asked: tuple[str, ...] = ()
    gate_passed_first_attempt: bool = False
    customer_rejected: bool = False
    abandoned: bool = False          # the customer never finished the interview
    recorded_at: str = field(default_factory=_now)
    note: str = ""

    def reward(self, gamma: float = ATTRITION_GAMMA) -> float:
        """§6's reward. Every term is observed, none is judged.

        An abandoned interview scores the attrition cost and nothing else: no
        gate was ever attempted, which is exactly the outcome the question
        budget exists to avoid.
        """
        if self.abandoned:
            return -gamma * self.questions_asked
        return (
            float(self.gate_passed_first_attempt)
            - gamma * self.questions_asked
            - float(self.customer_rejected)
        )

    @property
    def succeeded(self) -> bool:
        return (self.gate_passed_first_attempt and not self.customer_rejected
                and not self.abandoned)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slots_asked"] = list(self.slots_asked)
        data["reward"] = round(self.reward(), 4)
        return data


@dataclass(frozen=True)
class Trend:
    """Pass rate and question count, reported together and never apart."""

    pass_rate: float
    mean_questions: float
    abandon_rate: float
    mean_reward: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_rate": round(self.pass_rate, 4),
            "mean_questions": round(self.mean_questions, 2),
            "abandon_rate": round(self.abandon_rate, 4),
            "mean_reward": round(self.mean_reward, 4),
            "n": self.n,
        }


@dataclass
class OutcomeLog:
    """The history the question policy learns from."""

    outcomes: list[InterviewOutcome] = field(default_factory=list)

    def add(self, outcome: InterviewOutcome) -> InterviewOutcome:
        self.outcomes.append(outcome)
        return outcome

    def record(self, interview: Any, *, gate_passed_first_attempt: bool = False,
               customer_rejected: bool = False, abandoned: bool = False,
               note: str = "") -> InterviewOutcome:
        """Log what became of the build an :class:`Interview` specified."""
        asked = [*getattr(interview, "asked", []), *getattr(interview, "pending", [])]
        spec = getattr(interview, "spec", None)
        return self.add(InterviewOutcome(
            spec_hash=spec.hash if spec is not None else "",
            questions_asked=len(asked),
            slots_asked=tuple(q.slot for q in asked),
            gate_passed_first_attempt=gate_passed_first_attempt,
            customer_rejected=customer_rejected,
            abandoned=abandoned,
            note=note,
        ))

    # -- the numbers ------------------------------------------------------ #
    def trend(self, outcomes: Sequence[InterviewOutcome] | None = None,
              gamma: float = ATTRITION_GAMMA) -> Trend | None:
        rows = list(outcomes if outcomes is not None else self.outcomes)
        if not rows:
            return None
        n = len(rows)
        return Trend(
            pass_rate=sum(o.gate_passed_first_attempt for o in rows) / n,
            mean_questions=sum(o.questions_asked for o in rows) / n,
            abandon_rate=sum(o.abandoned for o in rows) / n,
            mean_reward=sum(o.reward(gamma) for o in rows) / n,
            n=n,
        )

    def windows(self, size: int = MIN_WINDOW) -> tuple[Trend | None, Trend | None]:
        """The earliest and latest windows, for a before/after comparison."""
        if len(self.outcomes) < 2 * size:
            return None, None
        return self.trend(self.outcomes[:size]), self.trend(self.outcomes[-size:])

    def diagnose(self, size: int = MIN_WINDOW) -> dict[str, Any]:
        """§6's diagnostic: is the policy learning, or just asking more?

        Reports both rates or neither. A pass rate that rose because the
        interview got longer is not an improvement — the attrition term was paid
        and simply not counted — and it is indistinguishable from real learning
        on any dashboard that plots pass rate alone.
        """
        early, late = self.windows(size)
        if early is None or late is None:
            return {
                "verdict": "insufficient_history",
                "detail": (f"need {2 * size} interviews to compare windows, have "
                           f"{len(self.outcomes)}"),
                "n": len(self.outcomes),
            }

        d_pass = late.pass_rate - early.pass_rate
        d_questions = late.mean_questions - early.mean_questions
        d_reward = late.mean_reward - early.mean_reward

        if d_pass > 0 and d_questions <= 0:
            verdict, detail = "learning", (
                "first-attempt pass rate rose while the interview got no longer: "
                "the policy is choosing better questions, not more of them"
            )
        elif d_pass > 0 and d_reward <= 0:
            verdict, detail = "asking_more", (
                f"pass rate rose {d_pass:+.1%} but mean questions rose "
                f"{d_questions:+.1f} and mean reward did not improve — the "
                "attrition cost is being paid and not counted. This is not learning"
            )
        elif d_pass > 0:
            verdict, detail = "learning_at_a_cost", (
                f"pass rate rose {d_pass:+.1%} at {d_questions:+.1f} more questions; "
                "reward still improved, so the trade is paying, but watch the "
                "abandon rate"
            )
        elif d_reward > 0:
            verdict, detail = "shorter_interviews", (
                f"pass rate did not rise, but the interview shortened "
                f"{d_questions:+.1f} questions and reward improved"
            )
        else:
            verdict, detail = "not_improving", (
                f"pass rate {d_pass:+.1%}, questions {d_questions:+.1f}, reward "
                f"{d_reward:+.3f} — no evidence the policy is improving"
            )

        return {
            "verdict": verdict,
            "detail": detail,
            "early": early.as_dict(),
            "late": late.as_dict(),
            "delta": {
                "pass_rate": round(d_pass, 4),
                "mean_questions": round(d_questions, 2),
                "mean_reward": round(d_reward, 4),
                "abandon_rate": round(late.abandon_rate - early.abandon_rate, 4),
            },
        }

    def by_slot(self, gamma: float = ATTRITION_GAMMA) -> dict[str, dict[str, Any]]:
        """Mean reward of interviews that asked each slot.

        Not a causal estimate — a slot asked only on hard specs will look bad
        whatever it contributed. It is a place to start looking, and it is what
        a question policy would be fitted against.
        """
        totals: dict[str, list[float]] = {}
        for outcome in self.outcomes:
            for slot in outcome.slots_asked:
                totals.setdefault(slot, []).append(outcome.reward(gamma))
        return {
            slot: {"n": len(rewards),
                   "mean_reward": round(sum(rewards) / len(rewards), 4)}
            for slot, rewards in sorted(totals.items())
        }

    # -- persistence ------------------------------------------------------- #
    def to_list(self) -> list[dict[str, Any]]:
        return [o.as_dict() for o in self.outcomes]

    @classmethod
    def from_list(cls, rows: Iterable[dict[str, Any]] | None) -> OutcomeLog:
        log = cls()
        for row in rows or []:
            data = {k: v for k, v in dict(row).items()
                    if k in InterviewOutcome.__dataclass_fields__}
            data["slots_asked"] = tuple(data.get("slots_asked", ()))
            log.add(InterviewOutcome(**data))
        return log

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_list(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> OutcomeLog:
        return cls.from_list(json.loads(Path(path).read_text(encoding="utf-8")))

    def render(self) -> str:
        d = self.diagnose()
        if d["verdict"] == "insufficient_history":
            return f"FORGE OUTCOMES — {d['detail']}"
        early, late = d["early"], d["late"]
        return "\n".join([
            "FORGE OUTCOMES (§6)",
            "",
            f"  first {early['n']:>3}   pass {early['pass_rate']:.2f}   "
            f"q_bar {early['mean_questions']:.1f}   reward {early['mean_reward']:+.3f}",
            f"  last  {late['n']:>3}   pass {late['pass_rate']:.2f}   "
            f"q_bar {late['mean_questions']:.1f}   reward {late['mean_reward']:+.3f}",
            "",
            f"  {d['verdict'].upper()}: {d['detail']}",
        ])


__all__ = ["MIN_WINDOW", "InterviewOutcome", "OutcomeLog", "Trend"]
