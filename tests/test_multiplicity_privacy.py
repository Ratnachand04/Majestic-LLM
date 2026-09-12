"""Duplication is a privacy control, and it has to be measured to be one.

Part 8 §21: memorisation probability rises sharply with duplication count, and
Majestic ships weights to devices, so the attacker has white-box access. Naive
augmentation increases duplication *by design* — twenty transforms of one scan
are twenty near-copies of its content.

The privacy axis in the Proving Ground measures verbatim extractability. This
figure is its stated prior: a build holding forty copies of one document should
expect a worse result than one holding five, and knowing that beforehand turns
a bad audit from a surprise into a prediction.
"""
from __future__ import annotations

import pytest

from modelrig.augment import MAX_MULTIPLICITY
from modelrig.data_factory import DataFactory, DataRefusal
from modelrig.primitives import TaskPrimitive


def _distinct(n: int = 120) -> list[tuple[str, str]]:
    return [(f"patient {i} referred by doctor Patel for a blood panel",
             "a" if i % 2 else "b") for i in range(n)]


# =========================================================================== #
# The prior is measured and reported
# =========================================================================== #
def test_the_report_carries_the_duplication_figure():
    """Without it the privacy audit runs blind to the thing that predicts it."""
    report = DataFactory().build(_distinct(), TaskPrimitive.CLASSIFY).report

    assert report.max_source_multiplicity > 0
    assert report.max_source_multiplicity <= MAX_MULTIPLICITY
    assert report.sources_capped == 0, "a distinct corpus should need no trimming"


def test_the_cap_bounds_what_reaches_training():
    """§21: bound k_i per source_id. Twenty copies of one document is a choice,
    and it is not one the amplifier gets to make silently."""
    tight = DataFactory(max_multiplicity=2).build(_distinct(), TaskPrimitive.CLASSIFY)
    loose = DataFactory(max_multiplicity=8).build(_distinct(), TaskPrimitive.CLASSIFY)

    assert tight.report.max_source_multiplicity <= 2
    assert tight.report.sources_capped > 0, "the trim must be reported, not silent"
    assert len(tight.train) < len(loose.train)


def test_the_default_cap_does_not_disturb_a_healthy_build():
    """The control is a bound, not a behaviour change. A corpus that never
    approaches the cap must be amplified exactly as before."""
    capped = DataFactory().build(_distinct(), TaskPrimitive.CLASSIFY)
    huge = DataFactory(max_multiplicity=10**9).build(_distinct(), TaskPrimitive.CLASSIFY)

    assert capped.report.sources_capped == 0
    assert len(capped.train) == len(huge.train)


# =========================================================================== #
# The interaction that matters
# =========================================================================== #
def test_the_cap_does_not_mask_generator_collapse():
    """The regression this control can cause, pinned.

    A hundred copies of one document is the degenerate corpus the collapse
    guard exists to refuse. That guard works by survival through dedup: a
    generator emitting one row five hundred ways produces almost nothing that
    survives. Trimming those copies BEFORE dedup removes them from the
    denominator and makes the degeneracy look like health — the corpus is then
    admitted, and a model trains on a single document.

    So rows the cap removes count as non-survivors, which is what they are.
    """
    degenerate = [("ok", "a" if i % 2 else "b") for i in range(100)]
    with pytest.raises(DataRefusal, match="collapse"):
        DataFactory().build(degenerate, TaskPrimitive.CLASSIFY)


def test_survival_is_measured_against_what_the_generator_produced():
    """The denominator is the generator's output, not the post-cap remainder."""
    bundle = DataFactory().build(_distinct(), TaskPrimitive.CLASSIFY)
    report = bundle.report

    assert report.raw_count >= len(bundle.train) - bundle.real_seed_count
    assert 0.0 < report.survival_rate <= 1.0


def test_a_degenerate_corpus_is_refused_at_every_cap():
    """Whatever the cap is set to, degeneracy is still degeneracy."""
    degenerate = [("ok", "a" if i % 2 else "b") for i in range(100)]
    for cap in (1, 2, 8, 10**9):
        with pytest.raises(DataRefusal):
            DataFactory(max_multiplicity=cap).build(degenerate, TaskPrimitive.CLASSIFY)
