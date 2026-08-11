"""Model-compatibility and architecture-conformance checks.

These are executable versions of the architecture's own rules. A rule that lives
only in documentation drifts; a rule with a test does not.
"""
from __future__ import annotations

import pytest

from majestic.serving import device_budget
from modelrig.catalogue import DEFAULT_CATALOGUE
from modelrig.conformance import (
    PUBLISHED_CONFIGS,
    check_architecture_conformance,
    check_model_compatibility,
    run_all,
)


# --- model compatibility -------------------------------------------------- #
def test_catalogue_matches_published_model_configs():
    """A wrong n_kv_heads silently corrupts every device feasibility verdict."""
    report = check_model_compatibility()
    assert report.errors == [], [f.detail for f in report.errors]


def test_every_catalogue_base_has_a_published_reference():
    refs = {b.ref for b in DEFAULT_CATALOGUE.bases}
    missing = refs - set(PUBLISHED_CONFIGS)
    assert missing == set(), f"unverified KV geometry for: {missing}"


def test_teachers_are_permissively_licensed():
    """Closed or restrictive teachers cannot train a competing model (A-02)."""
    report = check_model_compatibility()
    assert not [f for f in report.errors if f.check == "teacher_licence"]


def test_missing_same_family_teacher_is_reported_as_a_warning():
    """Logit KD needs an identical tokenizer; without one only sequence KD works."""
    report = check_model_compatibility()
    families = {f.subject for f in report.warnings if f.check == "logit_kd_reachable"}
    assert "llama" in families      # no llama teacher in a Qwen-centred catalogue


# --- architecture conformance --------------------------------------------- #
def test_architecture_is_conformant():
    report = check_architecture_conformance()
    assert report.errors == [], [f.detail for f in report.errors]


def test_full_report_is_clean():
    report = run_all()
    assert report.ok is True
    assert report.checks_run > 30


def test_findings_cite_their_source_diagram():
    report = run_all()
    assert all(f.source for f in report.findings)


# --- the numbers the architecture states ---------------------------------- #
def test_device_budget_reproduces_the_b09_table_exactly():
    """B-09's 4 GB tablet: 1.10 + 0.22 + 0.09 + 0.06 + 0.60 + 1.40 = 3.47 GB."""
    budget = device_budget(
        total_gb=4.0, base_params_b=1.7, context_length=2048, n_adapters=20
    )
    assert budget.committed_gb == pytest.approx(3.47, abs=0.05)
    assert budget.fits is True

    weights = next(line for line in budget.lines if "Base model" in line.label)
    assert weights.gb == pytest.approx(1.10, abs=0.05)


def test_moe_residency_trap_is_encoded():
    """30B resident for 3B active is the A-07 trap, and it must be visible."""
    moe = next(b for b in DEFAULT_CATALOGUE.bases if b.is_moe)
    assert moe.resident_params_b > 9 * (moe.active_params_b or 1)
