"""Tests for the verifier and its pluggable checks."""
from __future__ import annotations

from majestic.types import Response
from majestic.verification.checks import (
    CodeExecutionCheck,
    FactualityCheck,
    MathCheck,
    NonEmptyCheck,
    SchemaCheck,
    _safe_arith,
)
from majestic.verification.verifier import PipelineVerifier


def _resp(content, **meta) -> Response:
    return Response(content=content, metadata=dict(meta))


# --- math --------------------------------------------------------------- #
def test_safe_arith():
    assert _safe_arith("2 + 2") == 4
    assert _safe_arith("10 / 4") == 2.5
    assert _safe_arith("__import__('os')") is None  # not arithmetic


def test_math_check_catches_wrong_equation():
    check = MathCheck()
    wrong = _resp("The result is 2 + 2 = 5.")
    assert check.applies(wrong)
    assert check.run(wrong).passed is False


def test_math_check_passes_correct_equation():
    assert MathCheck().run(_resp("Note that 3 * 4 = 12 exactly.")).passed is True


# --- non-empty ---------------------------------------------------------- #
def test_non_empty():
    assert NonEmptyCheck().run(_resp("hi")).passed is True
    assert NonEmptyCheck().run(_resp("")).passed is False
    assert NonEmptyCheck().run(_resp(None)).passed is False


# --- schema ------------------------------------------------------------- #
def test_schema_check():
    ok = _resp('{"name": "x", "value": 1}', schema=["name", "value"])
    bad = _resp('{"name": "x"}', schema=["name", "value"])
    not_json = _resp("just text", schema=["name"])
    assert SchemaCheck().run(ok).passed is True
    assert SchemaCheck().run(bad).passed is False
    assert SchemaCheck().run(not_json).passed is False


# --- code execution ----------------------------------------------------- #
def test_code_execution_check():
    check = CodeExecutionCheck()
    ok = _resp("6*7 is 42", verify_code={"code": "print(6*7)", "expected": "42"})
    bad = _resp("6*7 is 43", verify_code={"code": "print(6*7)", "expected": "43"})
    assert check.applies(ok)
    assert check.run(ok).passed is True
    assert check.run(bad).passed is False


# --- factuality --------------------------------------------------------- #
def test_factuality_catches_swapped_entity():
    grounding = ["Paris is the capital of France."]
    wrong = _resp("London is the capital of France.", grounding=grounding)
    right = _resp("Paris is the capital of France.", grounding=grounding)
    assert FactualityCheck().run(wrong).passed is False
    assert FactualityCheck().run(right).passed is True


def test_factuality_only_applies_with_grounding():
    assert FactualityCheck().applies(_resp("anything")) is False


# --- full pipeline ------------------------------------------------------ #
def test_pipeline_verifier_records_results_and_gates():
    verifier = PipelineVerifier()
    bad = _resp("2 + 2 = 5")
    assert verifier.verify(bad) is False
    names = {r["name"] for r in bad.metadata["verification"]}
    assert "math" in names

    good = _resp("Everything checks out: 2 + 2 = 4.")
    assert verifier.verify(good) is True
