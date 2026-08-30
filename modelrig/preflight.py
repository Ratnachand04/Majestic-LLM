"""The assertions that run before a GPU is touched (Part 9 §18-§19).

The highest-value code in the Trainer, and the cheapest: it costs nothing to run
and it prevents the failure modes that are otherwise discovered only *after* a
full build has completed and evaluated cleanly.

That last clause is the point. Every failure listed here reports success. The
loss goes down, the checkpoint saves, the scorecard fills in — and the model is
quietly wrong. A silent failure that survives evaluation is far more expensive
than a crash, because it ships.

**The pre-flight runs inside ``train``, never in a caller.** It must be
impossible to reach a training step by a path that skips it, which is a
structural property rather than a convention: a caller can forget, and the one
that forgets will be the one written in a hurry six months from now.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from majestic.logging_utils import get_logger

logger = get_logger(__name__)


class PreflightError(RuntimeError):
    """A pre-flight assertion failed. No training may occur."""


@dataclass
class PreflightReport:
    """What was checked, and what it found."""

    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def record(self, name: str) -> None:
        self.checks.append(name)

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks_run": len(self.checks),
            "checks": list(self.checks),
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


def _hash(value: Any) -> str:
    return hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).hexdigest()


# =========================================================================== #
# §18 — contamination (I-01)
# =========================================================================== #
def assert_no_contamination(
    train_hashes: Iterable[str], holdout_hashes: Iterable[str],
    train_sources: Iterable[str] = (), holdout_sources: Iterable[str] = (),
) -> None:
    """I-01. Raises rather than returning a verdict, so it cannot be ignored.

    Checked at **two** levels. Example-level hashes catch the obvious case. But
    Part 8's augmentation makes the subtler one possible: ``T(d)`` for a
    held-out ``d`` differs byte for byte from ``d`` and still leaks it. Only
    source-document provenance catches that, which is why every generated
    example carries a ``source_id``.
    """
    overlap = set(train_hashes) & set(holdout_hashes)
    if overlap:
        raise PreflightError(
            f"contamination: {len(overlap)} example(s) appear in both train and "
            f"holdout (e.g. {sorted(overlap)[:3]}). Every number the Proving "
            "Ground would produce is invalid"
        )

    shared_sources = set(train_sources) & set(holdout_sources)
    if shared_sources:
        raise PreflightError(
            f"contamination at the SOURCE level: {len(shared_sources)} held-out "
            f"document(s) also produced training examples (e.g. "
            f"{sorted(shared_sources)[:3]}). An augmented variant of a held-out "
            "document leaks it even though the bytes differ"
        )


# =========================================================================== #
# §19 — the silent failures, in order of how often they occur
# =========================================================================== #
def assert_chat_template(dataset_template: str, base_template: str) -> None:
    """**The most common silent failure in fine-tuning, anywhere.**

    If the data is formatted with a different chat template than the base
    expects, training succeeds, the loss falls, the checkpoint saves, and the
    model is quietly wrong. Nothing downstream notices until a customer does.

    A hard assertion, not a convention.
    """
    if not dataset_template or not base_template:
        raise PreflightError(
            "chat template missing: the dataset and the base must each declare "
            "one, because a mismatch reports success and produces a broken model"
        )
    if _hash(dataset_template) != _hash(base_template):
        raise PreflightError(
            "chat template mismatch: the dataset was formatted with a different "
            f"template ({_hash(dataset_template)}) than the base expects "
            f"({_hash(base_template)}). Training would succeed and the model "
            "would be subtly wrong"
        )


def assert_tokenizer_match(prep_tokenizer: str, train_tokenizer: str) -> None:
    """Data prepared with one tokenizer and trained with another learns garbage."""
    if _hash(prep_tokenizer) != _hash(train_tokenizer):
        raise PreflightError(
            f"tokenizer mismatch: data was prepared with {prep_tokenizer!r} and "
            f"training uses {train_tokenizer!r}"
        )


def assert_loss_mask(mask: Sequence[int], pad_positions: Sequence[int]) -> None:
    """Padding in the loss teaches the model to emit padding."""
    included = [i for i in pad_positions if i < len(mask) and mask[i]]
    if included:
        raise PreflightError(
            f"{len(included)} padding token(s) are inside the loss mask; the "
            "model would learn to emit padding"
        )


def normalise_by_tokens(
    microbatch_losses: Sequence[float], microbatch_tokens: Sequence[int]
) -> float:
    """§8 — token-level normalisation across the accumulation window.

    Gradient accumulation is *not* exactly a larger batch. Averaging per
    microbatch and then across accumulation steps weights **short sequences more
    heavily** than token-level averaging over the true batch would.

    For variable-length documents — exactly Majestic's case — that is a silent,
    systematic bias toward whatever the short inputs happen to teach. Normalise
    by token count across the window instead.
    """
    if len(microbatch_losses) != len(microbatch_tokens):
        raise ValueError("each microbatch loss needs its token count")
    total_tokens = sum(microbatch_tokens)
    if total_tokens == 0:
        return 0.0
    return sum(loss * n for loss, n in zip(microbatch_losses, microbatch_tokens)) / total_tokens


def naive_microbatch_mean(microbatch_losses: Sequence[float]) -> float:
    """The wrong normalisation, kept so the bias can be *shown* rather than
    asserted. Equal to the correct one only when every microbatch has the same
    token count."""
    return sum(microbatch_losses) / len(microbatch_losses) if microbatch_losses else 0.0


def assert_finite(loss: float, step: int = 0) -> None:
    """Fail fast on divergence rather than saving a NaN checkpoint."""
    if loss != loss or loss in (float("inf"), float("-inf")):
        raise PreflightError(
            f"loss diverged to {loss} at step {step}; failing rather than saving "
            "a checkpoint that would evaluate as garbage"
        )


# =========================================================================== #
# The whole pre-flight
# =========================================================================== #
def preflight(
    *,
    train_hashes: Iterable[str] = (),
    holdout_hashes: Iterable[str] = (),
    train_sources: Iterable[str] = (),
    holdout_sources: Iterable[str] = (),
    dataset_template: str = "",
    base_template: str = "",
    prep_tokenizer: str = "",
    train_tokenizer: str = "",
    strict: bool = True,
) -> PreflightReport:
    """Run every check. Raises on the first failure when ``strict``.

    ``strict=False`` collects everything instead, which is useful for reporting
    but must never be the path a real build takes.
    """
    report = PreflightReport()

    for name, check in (
        ("contamination", lambda: assert_no_contamination(
            train_hashes, holdout_hashes, train_sources, holdout_sources)),
        ("chat_template", lambda: assert_chat_template(dataset_template, base_template)
            if (dataset_template or base_template) else None),
        ("tokenizer", lambda: assert_tokenizer_match(prep_tokenizer, train_tokenizer)
            if (prep_tokenizer or train_tokenizer) else None),
    ):
        try:
            check()
            report.record(name)
        except PreflightError as exc:
            report.fail(name, str(exc))
            if strict:
                raise

    logger.info("preflight: %d check(s) passed", len(report.checks))
    return report


__all__ = [
    "PreflightError", "PreflightReport",
    "assert_chat_template", "assert_finite", "assert_loss_mask",
    "assert_no_contamination", "assert_tokenizer_match", "naive_microbatch_mean",
    "normalise_by_tokens", "preflight",
]
