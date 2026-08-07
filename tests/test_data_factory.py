"""Tests for the DATA FACTORY and its blocking guardrails (B-06)."""
from __future__ import annotations

import pytest

from modelrig.data_factory import (
    DataFactory,
    DataRefusal,
    diversity_entropy,
    scrub_pii,
)
from modelrig.primitives import TaskPrimitive

_POS = ["good great work", "love this service", "happy with the result",
        "nice and helpful staff", "excellent quality throughout"]
_NEG = ["bad broken service", "hate this experience", "sad and useless outcome",
        "terrible unhelpful staff", "awful quality throughout"]


def _corpus(n: int = 100) -> list[tuple[str, str]]:
    rows = []
    for i in range(n // 2):
        rows.append((f"{_POS[i % len(_POS)]} case {i}", "positive"))
        rows.append((f"{_NEG[i % len(_NEG)]} case {i}", "negative"))
    return rows


# --- the real-seed floor -------------------------------------------------- #
def test_refuses_below_the_seed_floor():
    """Below the floor the factory refuses rather than degrading quietly."""
    with pytest.raises(DataRefusal, match="floor"):
        DataFactory().build(_corpus(10), TaskPrimitive.CLASSIFY)


def test_builds_above_the_floor():
    bundle = DataFactory().build(_corpus(100), TaskPrimitive.CLASSIFY)
    assert bundle.train and bundle.held_out
    assert bundle.report.passed


# --- the immutable split -------------------------------------------------- #
def test_held_out_is_locked_and_never_trained_on():
    bundle = DataFactory().build(_corpus(100), TaskPrimitive.CLASSIFY)
    held_texts = {t for t, _ in bundle.held_out}
    train_texts = {t for t, _ in bundle.train}
    assert held_texts and not (held_texts & train_texts)


def test_split_is_deterministic():
    a = DataFactory(seed=7).build(_corpus(100), TaskPrimitive.CLASSIFY)
    b = DataFactory(seed=7).build(_corpus(100), TaskPrimitive.CLASSIFY)
    assert a.held_out == b.held_out


# --- amplification and accumulation --------------------------------------- #
def test_amplifies_beyond_the_real_corpus():
    bundle = DataFactory().build(_corpus(100), TaskPrimitive.CLASSIFY)
    assert bundle.synthetic_count > 0
    assert len(bundle.train) > bundle.real_seed_count


def test_accumulate_do_not_replace():
    """Earlier generations' real data is mixed in, never discarded."""
    previous = _corpus(60)
    bundle = DataFactory().build(_corpus(60), TaskPrimitive.CLASSIFY,
                                 previous_real=previous)
    assert bundle.real_seed_count + len(bundle.held_out) == 120


# --- the blocking QA gates ------------------------------------------------- #
def test_pii_is_scrubbed_before_training():
    text = "contact raj at raj@example.com or 9876543210, card 4111 1111 1111 1111"
    clean, n = scrub_pii(text)
    assert n >= 3
    assert "raj@example.com" not in clean
    assert "[EMAIL]" in clean


def test_pii_scrub_runs_inside_the_pipeline():
    rows = _corpus(100)
    rows[0] = ("good great work contact a@b.com", "positive")
    bundle = DataFactory().build(rows, TaskPrimitive.CLASSIFY)
    assert "@b.com" not in " ".join(t for t, _ in bundle.train)


def test_dedup_removes_near_duplicates():
    factory = DataFactory()
    bundle = factory.build(_corpus(100), TaskPrimitive.CLASSIFY)
    r = bundle.report
    assert r.after_minhash_dedup <= r.after_exact_dedup <= r.raw_count
    assert r.after_semantic_dedup <= r.after_minhash_dedup


def test_diversity_entropy_detects_collapse():
    collapsed = [("same same same", "a")] * 50
    varied = _corpus(50)
    assert diversity_entropy(collapsed) < diversity_entropy(varied)


def test_collapsed_generation_blocks_the_build():
    """A degenerate corpus must kill the run, not warn."""
    rows = [("ok", "a" if i % 2 else "b") for i in range(100)]
    with pytest.raises(DataRefusal):
        DataFactory().build(rows, TaskPrimitive.CLASSIFY)


def test_rationale_traces_are_training_time_only():
    factory = DataFactory()
    trace = factory.rationale_trace("some form text", "positive")
    bundle = factory.build(_corpus(100), TaskPrimitive.CLASSIFY, with_rationales=True)
    assert "Therefore" in trace
    # Traces live in training data; the held-out contract stays clean.
    assert all("Therefore:" not in t for t, _ in bundle.held_out)
