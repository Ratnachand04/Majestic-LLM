"""Tests for the primitive taxonomy (GAP-06) and licence chain (GAP-08)."""
from __future__ import annotations

import pytest

from modelrig.licence import DataRights, Licence, resolve_licence_chain
from modelrig.primitives import (
    TaskPrimitive,
    all_primitives,
    coverage_report,
    spec_for,
)


# --- the closed set of eight --------------------------------------------- #
def test_exactly_eight_primitives():
    assert len(all_primitives()) == 8
    assert len(TaskPrimitive) == 8


def test_spec_for_accepts_names_and_enums():
    assert spec_for("extract").primitive is TaskPrimitive.EXTRACT
    assert spec_for(TaskPrimitive.CLASSIFY).default_metric == "accuracy"


def test_unsupported_primitive_is_refused():
    """Anything outside the set must be an honest refusal, not confident garbage."""
    with pytest.raises(ValueError, match="not one of the supported primitives"):
        spec_for("choreograph_a_ballet")


def test_every_primitive_declares_a_seed_floor():
    assert all(p.seed_floor > 0 for p in all_primitives())


def test_coverage_report_measures_the_taxonomy(tmp_path=None):
    """GAP-06 demands this number be published against real traffic."""
    report = coverage_report(["extract", "classify", None, "choreograph"])
    assert report["total"] == 4
    assert report["covered"] == 2
    assert report["coverage_rate"] == 0.5


# --- the licence chain ---------------------------------------------------- #
def test_permissive_chain_resolves():
    chain = resolve_licence_chain(
        Licence.APACHE_2_0, Licence.APACHE_2_0, DataRights.CUSTOMER_OWNED, "IN"
    )
    assert chain.permitted is True
    assert chain.resolved_licence is Licence.APACHE_2_0


def test_closed_api_teacher_is_the_hard_boundary():
    """Distilling from a closed commercial API breaches its terms (A-02)."""
    chain = resolve_licence_chain(
        Licence.APACHE_2_0, Licence.CLOSED_API, DataRights.CUSTOMER_OWNED
    )
    assert chain.permitted is False
    assert any("closed commercial API" in r for r in chain.reasons)


def test_insufficient_data_rights_refuse():
    chain = resolve_licence_chain(
        Licence.APACHE_2_0, None, DataRights.THIRD_PARTY_NO_TRAINING
    )
    assert chain.permitted is False
    assert any("data rights" in r for r in chain.reasons)


def test_unknown_base_licence_refuses():
    chain = resolve_licence_chain(
        Licence.UNKNOWN, None, DataRights.CUSTOMER_OWNED
    )
    assert chain.permitted is False


def test_non_commercial_blocks_commercial_redistribution():
    chain = resolve_licence_chain(
        Licence.NON_COMMERCIAL, None, DataRights.CUSTOMER_OWNED,
        commercial_redistribution=True,
    )
    assert chain.permitted is False


def test_conditional_base_attaches_obligations():
    chain = resolve_licence_chain(
        Licence.LLAMA_COMMUNITY, None, DataRights.CUSTOMER_OWNED, "IN"
    )
    assert chain.permitted is True
    assert any("llama-community" in o for o in chain.obligations)


def test_jurisdiction_obligations_are_recorded():
    eu = resolve_licence_chain(Licence.MIT, None, DataRights.CUSTOMER_OWNED, "DE")
    ind = resolve_licence_chain(Licence.MIT, None, DataRights.CUSTOMER_OWNED, "IN")
    assert any("EU AI Act" in o for o in eu.obligations)
    assert any("DPDP" in o for o in ind.obligations)


def test_provenance_record_is_attached():
    chain = resolve_licence_chain(
        Licence.APACHE_2_0, Licence.APACHE_2_0, DataRights.CUSTOMER_OWNED, "IN"
    )
    record = chain.as_record()
    assert record["provenance"]["base"] == "apache-2.0"
    assert record["permitted"] is True
