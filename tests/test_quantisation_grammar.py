"""Tests for quantisation (A-05) and compiled grammars (B-08 slot 4, A-09)."""
from __future__ import annotations

import pytest

from modelrig.grammar import (
    compile_enum_grammar,
    compile_for_spec,
    compile_object_grammar,
    compile_toolcall_grammar,
)
from modelrig.primitives import TaskPrimitive
from modelrig.quantisation import (
    OUTLIER_THRESHOLD_B,
    TemperatureScaler,
    build_calibration_set,
    evaluate_quantisation,
    next_escalation,
)

TRAIN = [(f"positive sample {i}", "positive") for i in range(40)] + \
        [(f"negative sample {i}", "negative") for i in range(40)] + \
        [("a single rare case", "rare")]


def _calib(**over):
    return build_calibration_set(TRAIN, **over)


# --- calibration from the customer distribution (A-05) ------------------- #
def test_calibration_is_drawn_from_customer_data():
    calib = _calib(size=30)
    assert calib.on_distribution is True
    assert calib.source == "customer_task_distribution"
    assert all(any(s == t for t, _ in TRAIN) for s in calib.samples)


def test_calibration_is_stratified_so_rare_classes_survive():
    """An unstratified draw is how a rare class silently degrades at 4-bit."""
    calib = _calib(size=9)
    assert "a single rare case" in calib.samples


def test_generic_fallback_is_flagged_not_hidden():
    calib = build_calibration_set([], generic_fallback=["the quick brown fox"])
    assert calib.on_distribution is False
    assert calib.source == "generic_text"


def test_off_distribution_calibration_fails_the_gate():
    calib = build_calibration_set([], generic_fallback=["generic text"])
    outcome = evaluate_quantisation(
        quantiser="awq", bit_width="int4", params_b=1.7,
        reference_predictions=["a"], quantised_predictions=["a"], calibration=calib,
    )
    assert outcome.passed is False
    assert any("generic text" in r for r in outcome.reasons)


# --- the mandatory gate --------------------------------------------------- #
def test_answer_flip_within_bound_passes():
    outcome = evaluate_quantisation(
        quantiser="awq", bit_width="int4", params_b=1.7,
        reference_predictions=["a", "b", "c", "d"],
        quantised_predictions=["a", "b", "c", "d"],
        calibration=_calib(),
    )
    assert outcome.passed is True
    assert outcome.answer_flip_rate == 0.0


def test_excessive_flip_fails_even_at_equal_accuracy():
    outcome = evaluate_quantisation(
        quantiser="awq", bit_width="int4", params_b=1.7,
        reference_predictions=["a", "b", "c", "d"],
        quantised_predictions=["b", "a", "d", "c"],
        calibration=_calib(),
    )
    assert outcome.answer_flip_rate == 1.0
    assert outcome.passed is False
    assert outcome.escalation is not None


def test_large_models_flag_outlier_risk():
    outcome = evaluate_quantisation(
        quantiser="awq", bit_width="int4", params_b=8.0,
        reference_predictions=["a"], quantised_predictions=["a"], calibration=_calib(),
    )
    assert outcome.outlier_risk is True
    assert outcome.passed is False


def test_escalation_ladder_prefers_spqr_for_large_models():
    assert next_escalation("awq", params_b=OUTLIER_THRESHOLD_B + 1) == "spqr"
    assert next_escalation("awq", params_b=1.7) == "gptq"
    assert next_escalation("larger_base", params_b=1.7) is None


# --- confidence recalibration --------------------------------------------- #
def test_temperature_is_refitted_after_quantisation():
    """Quantisation shifts the probability distribution; T must be refitted."""
    scaler = TemperatureScaler()
    # Wildly overconfident and mostly wrong -> needs a temperature above 1.
    scaler.fit([0.99] * 10, [False] * 8 + [True] * 2)
    assert scaler.fitted is True
    assert scaler.temperature > 1.0
    assert scaler(0.99) < 0.99      # scaling pulls confidence down


def test_recalibration_runs_inside_the_gate():
    outcome = evaluate_quantisation(
        quantiser="awq", bit_width="int4", params_b=1.7,
        reference_predictions=["a", "b"], quantised_predictions=["a", "b"],
        calibration=_calib(), confidences=[0.9, 0.9], correct=[True, True],
    )
    assert outcome.temperature > 0


# --- grammars: schema violation is impossible ----------------------------- #
def test_enum_grammar_admits_only_its_labels():
    g = compile_enum_grammar(["positive", "negative"])
    assert g.validate("positive")[0] is True
    assert g.validate("maybe")[0] is False
    assert '"positive"' in g.gbnf


def test_object_grammar_requires_every_field():
    g = compile_object_grammar({"name": "string", "total": "number"})
    assert g.validate('{"name": "x", "total": 5}')[0] is True
    ok, reason = g.validate('{"name": "x"}')
    assert ok is False and "missing" in reason
    assert g.validate("not json")[0] is False


def test_object_grammar_emits_gbnf_rules():
    g = compile_object_grammar({"total": "number"})
    assert "root ::=" in g.gbnf and "number" in g.gbnf


def test_toolcall_grammar_constrains_the_call():
    """A model that cannot express a malformed call cannot make one (A-09)."""
    g = compile_toolcall_grammar([{"name": "search"}, {"name": "fetch"}])
    assert g.validate('{"tool": "search", "args": {"q": "x"}}')[0] is True
    assert g.validate('{"tool": "search"}')[0] is False   # args required
    assert "toolname ::=" in g.gbnf


def test_compile_for_spec_dispatches_by_primitive():
    assert compile_for_spec(TaskPrimitive.CLASSIFY, {"labels": ["a", "b"]}).kind == "enum"
    assert compile_for_spec(TaskPrimitive.EXTRACT, {"fields": {"a": "string"}}).kind == "object"
    assert compile_for_spec(TaskPrimitive.SUMMARISE, {}) is None   # free text by contract


def test_compile_for_spec_accepts_enum_and_string():
    """TaskPrimitive mixes in str; both spellings must work."""
    a = compile_for_spec(TaskPrimitive.CLASSIFY, {"labels": ["x"]})
    b = compile_for_spec("classify", {"labels": ["x"]})
    assert a.gbnf == b.gbnf


def test_empty_grammar_inputs_are_rejected():
    with pytest.raises(ValueError):
        compile_enum_grammar([])
    with pytest.raises(ValueError):
        compile_object_grammar({})
