"""Regressions for evidence integrity and usable deployment, not implementation mirrors."""
import json

import pytest

from api import studio
from majestic.fabric.graph import FabricGraph, Node, NodeKind
from modelrig import classifier
from modelrig.buildspec import BuildSpec, TrainingMethod
from modelrig.datasets import split_dataset
from modelrig.factory import _build_id
from modelrig.forge import Interviewer
from modelrig.pipeline import load_cartridge_model
from modelrig.proving_ground import ProvingGround
from modelrig.registry import CartridgeRegistry


def test_no_oracle_confidence():
    held = [(str(i), "yes") for i in range(30)]
    card = ProvingGround().evaluate(lambda texts: ["yes"] * len(texts), held)
    axis = next(a for a in card.axes if a.name == "calibration")
    assert axis.n == 0 and not axis.passed


@pytest.mark.parametrize("conf", [[1.0], [float("nan")] * 30, [2.0] * 30])
def test_invalid_confidence_cannot_be_evidence(conf):
    with pytest.raises(ValueError, match="confidence"):
        ProvingGround().evaluate(lambda texts: ["yes"] * len(texts),
                                [(str(i), "yes") for i in range(30)], confidences=conf)


def test_assumptions_never_become_measurements():
    known = {}
    Interviewer.apply_probe(known, studio._probe_profile())
    assert not known["profile_source"].measured


def test_duplicates_are_not_independent_test_evidence():
    rows = [(" a ", "one"), ("A", "one"), ("b", "one"),
            ("c", "two"), ("d", "two")]
    train, test = split_dataset(rows, 0.5, 0)
    assert len(train) + len(test) == 4
    assert {t.strip().lower() for t, _ in train}.isdisjoint(
        {t.strip().lower() for t, _ in test})


def test_build_identity_changes_when_file_contents_change(tmp_path):
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps({"text": "old", "label": "one"}), encoding="utf-8")
    spec = BuildSpec(task="../../unsafe", base_model="centroid",
                     method=TrainingMethod.CENTROID, dataset=str(data))
    original = _build_id(spec)
    data.write_text(json.dumps({"text": "new", "label": "one"}), encoding="utf-8")
    assert _build_id(spec) != original
    assert "/" not in original and "\\" not in original


@pytest.fixture
def built(tmp_path):
    rows = [(row["text"], row["label"]) for row in studio.sample_dataset(120)]
    outcome = studio.build("Classify support tickets on android offline", rows,
                           registry_path=tmp_path)
    assert outcome.admitted, outcome.refusal
    return tmp_path, outcome


def test_recalled_weights_cannot_be_served_directly(built):
    path, out = built
    CartridgeRegistry(path).recall(out.cartridge_id, "test recall")
    with pytest.raises(ValueError, match="recalled"):
        load_cartridge_model(out.cartridge_id, path)


def test_tampered_weights_are_rejected(built):
    path, out = built
    weights = path / "weights" / out.cartridge_id
    model = classifier.load_model(weights)
    model["idf"][0] += 1
    classifier.save_model(model, weights)
    with pytest.raises(ValueError, match="integrity"):
        load_cartridge_model(out.cartridge_id, path)


def test_executed_backend_is_not_the_analytical_plan(built):
    path, out = built
    cart = CartridgeRegistry(path).get(out.cartridge_id)
    assert cart.base_ref == "majestic/tfidf-centroid-v1"
    assert cart.model_card["execution"]["planned_base_executed"] is False
    assert cart.provenance["weights_digest"]


def test_resource_edits_invalidate_graph_identity():
    graph = FabricGraph()
    node = graph.add(Node("x", NodeKind.CARTRIDGE, ram_mb=10))
    previous = graph.graph_hash
    node.ram_mb = 100
    assert graph.graph_hash != previous
    previous = graph.graph_hash
    node.metadata["base_ref"] = "another/base"
    assert graph.graph_hash != previous


def test_integrity_rejects_unlisted_files(tmp_path):
    from modelrig.integrity import verify_manifest, write_manifest

    (tmp_path / "weights").write_bytes(b"original")
    write_manifest(tmp_path)
    verify_manifest(tmp_path)
    (tmp_path / "unexpected.safetensors").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="differ"):
        verify_manifest(tmp_path)


def test_portable_import_and_collision(tmp_path):
    import json

    from modelrig.integrity import write_manifest
    from modelrig.review import import_artifact

    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps(
        {"build_id": "demo-123", "backend": "hf_seqcls"}), encoding="utf-8")
    (source / "weights").write_bytes(b"original")
    write_manifest(source)
    registry = tmp_path / "registry"
    first = import_artifact(source, registry)
    assert import_artifact(source, registry) == first
    (source / "weights").write_bytes(b"changed")
    write_manifest(source)
    with pytest.raises(ValueError, match="different artifact"):
        import_artifact(source, registry)
