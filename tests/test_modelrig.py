"""Tests for the ModelRig factory: spec, compiler, classifier, planes, build."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelrig import classifier
from modelrig.buildspec import BuildSpec, TrainingMethod, ensure_valid, validate_spec
from modelrig.compiler import DefaultCompiler
from modelrig.datasets import load_dataset, split_dataset
from modelrig.factory import Factory
from modelrig.registry import FileSystemRegistry


# --- BuildSpec validation ---------------------------------------------- #
def test_validate_spec_ok():
    spec = BuildSpec(task="sentiment", base_model="tiny", method=TrainingMethod.CENTROID)
    assert validate_spec(spec) == []


def test_validate_spec_errors():
    spec = BuildSpec(task="", base_model="", quantization="int7", test_split=2.0)
    errors = validate_spec(spec)
    assert len(errors) >= 3
    with pytest.raises(ValueError):
        ensure_valid(spec)


# --- compiler ----------------------------------------------------------- #
def test_compiler_full_pipeline():
    spec = BuildSpec(task="t", base_model="b", method=TrainingMethod.CENTROID)
    assert DefaultCompiler().compile(spec) == ["data", "training", "eval", "compression", "export"]


def test_compiler_skips_compression_when_disabled():
    spec = BuildSpec(task="t", base_model="b", quantization="none")
    assert "compression" not in DefaultCompiler().compile(spec)


# --- dataset ------------------------------------------------------------ #
def test_dataset_split_is_deterministic():
    rows = load_dataset("builtin:sentiment")
    a = split_dataset(rows, 0.3, seed=0)
    b = split_dataset(rows, 0.3, seed=0)
    assert a == b
    assert len(a[0]) + len(a[1]) == len(rows)


# --- classifier + quantization ----------------------------------------- #
def test_centroid_fit_predict():
    rows = load_dataset("builtin:sentiment")
    model = classifier.fit_centroid(rows, ["negative", "positive"], dim=256)
    preds = classifier.predict(model, ["i love this, wonderful", "awful and broken, i hate it"])
    assert preds == ["positive", "negative"]


@pytest.mark.parametrize("quant", ["int8", "int4"])
def test_quantize_roundtrip_preserves_predictions(quant):
    rows = load_dataset("builtin:sentiment")
    model = classifier.fit_centroid(rows, ["negative", "positive"], dim=256)
    texts = [t for t, _ in rows]
    before = classifier.predict(model, texts)
    q_model, report = classifier.quantize_model(model, quant)
    after = classifier.predict(q_model, texts)
    assert report["ratio"] > 1.0
    # Lossy but should agree on clearly separable data.
    agree = sum(int(a == b) for a, b in zip(before, after)) / len(before)
    assert agree >= 0.9


def test_save_load_roundtrip(tmp_path: Path):
    rows = load_dataset("builtin:sentiment")
    model = classifier.fit_centroid(rows, ["negative", "positive"], dim=128)
    q_model, _ = classifier.quantize_model(model, "int4")
    classifier.save_model(q_model, tmp_path)
    loaded = classifier.load_model(tmp_path)
    assert classifier.predict(loaded, ["i love it"]) == classifier.predict(q_model, ["i love it"])


# --- registry ----------------------------------------------------------- #
def test_registry_put_get_list(tmp_path: Path):
    reg = FileSystemRegistry(tmp_path)
    reg.put("k1", "/some/path", {"task": "x"})
    assert reg.get("k1") == "/some/path"
    assert reg.get_metadata("k1") == {"task": "x"}
    assert "k1" in reg.list()
    with pytest.raises(KeyError):
        reg.get("missing")


# --- end-to-end factory build ------------------------------------------ #
def test_factory_build_produces_quantized_artifact(tmp_path: Path):
    factory = Factory(base_path=tmp_path)
    spec = BuildSpec(
        task="sentiment", base_model="hashing-centroid",
        method=TrainingMethod.CENTROID, quantization="int8", target_score=0.6,
    )
    result = factory.build(spec)
    assert result.success is True
    assert result.eval_report["passed"] is True
    art = Path(result.artifact_path)
    assert (art / "weights.npz").exists()
    assert (art / "metadata.json").exists()
    assert (art / "eval_report.json").exists()
    # Registered and reloadable.
    assert result.build_id in factory.registry.list()
    meta = json.loads((art / "metadata.json").read_text())
    assert meta["quantization"] == "int8"
    reloaded = classifier.load_model(art)
    assert classifier.predict(reloaded, ["wonderful and great"]) == ["positive"]


def test_factory_build_knn_none_rag(tmp_path: Path):
    factory = Factory(base_path=tmp_path)
    spec = BuildSpec(task="sentiment", base_model="knn", method=TrainingMethod.NONE_RAG,
                     quantization="none", target_score=0.6)
    result = factory.build(spec)
    assert result.success is True
    assert result.metadata["runtime"] == "npz"


@pytest.mark.integration
def test_factory_lora_build(tmp_path: Path):
    """Heavy LoRA path — skipped unless torch/transformers/peft are installed."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    factory = Factory(base_path=tmp_path)
    spec = BuildSpec(task="sentiment", base_model="prajjwal1/bert-tiny",
                     method=TrainingMethod.LORA, quantization="none", target_score=0.0)
    result = factory.build(spec)
    assert result.build_id
    assert result.eval_report["metric"] == "accuracy"


def test_factory_gate_fails_on_unlearnable_data(tmp_path: Path):
    # Contradictory dataset: identical texts carry both labels, so no model can
    # separate them -> the eval gate must fail and no artifact is exported.
    ds = tmp_path / "ambiguous.jsonl"
    ds.write_text(
        "\n".join(
            json.dumps({"text": t, "label": lab})
            for t, lab in [
                ("ok", "positive"), ("ok", "negative"),
                ("fine", "positive"), ("fine", "negative"),
                ("sure", "positive"), ("sure", "negative"),
                ("maybe", "positive"), ("maybe", "negative"),
                ("well", "positive"), ("well", "negative"),
            ]
        ),
        encoding="utf-8",
    )
    factory = Factory(base_path=tmp_path)
    spec = BuildSpec(task="ambiguous", base_model="centroid",
                     method=TrainingMethod.CENTROID, dataset=str(ds), target_score=0.95)
    result = factory.build(spec)
    assert result.success is False
    assert result.artifact_path is None
    assert "eval" in result.reason
