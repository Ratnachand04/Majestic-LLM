"""The build-and-ship path: real weights, real inference, real binaries.

The gap these cover is the one that made the system undemonstrable: the
pipeline trained a model, certified it, and then discarded it, leaving a
manifest that pointed at no weights. A certificate about an artefact nobody
kept is not a deliverable.
"""
from __future__ import annotations

import importlib.util
import tempfile

import pytest

from api import studio
from modelrig import classifier
from modelrig.package_exe import bundle_spec, package
from modelrig.pipeline import load_cartridge_model, predict_with_cartridge


def _examples(n: int = 120) -> list[tuple[str, str]]:
    return [(r["text"], r["label"]) for r in studio.sample_dataset(n)]


@pytest.fixture
def built(tmp_path):
    """One real build, into a temporary registry."""
    out = studio.build(
        "Classify support tickets by sentiment on an android phone, offline",
        _examples(),
        registry_path=tmp_path / "registry",
    )
    assert out.admitted, f"fixture build refused: {out.refusal}"
    return out, tmp_path / "registry"


# =========================================================================== #
# The weights actually exist
# =========================================================================== #
def test_a_built_cartridge_has_weights_on_disk(built):
    """Previously the pipeline proved a model was good and threw it away."""
    out, registry = built
    weights = registry / "weights" / out.cartridge_id
    assert weights.is_dir()
    assert (weights / "model.json").exists()
    assert (weights / "weights.npz").exists()
    assert out.weights_bytes > 0


def test_the_built_model_loads_and_predicts(built):
    out, registry = built
    model = load_cartridge_model(out.cartridge_id, registry)
    assert model["labels"]

    preds = predict_with_cartridge(
        out.cartridge_id,
        ["wonderful helpful team, very happy", "broken terrible awful service"],
        registry,
    )
    assert preds == ["positive", "negative"]


def test_serving_a_cartridge_with_no_weights_says_so(tmp_path):
    """The honest error, not an AttributeError three frames down."""
    with pytest.raises(FileNotFoundError, match="no weights for cartridge"):
        load_cartridge_model("deadbeef", tmp_path)


def test_what_ships_is_what_was_certified(built):
    """The scorecard measures the QUANTISED model, so that is the one persisted.

    Keeping the pre-quantisation reference instead would mean the certificate
    describes one artefact and the customer receives another.
    """
    out, registry = built
    model = load_cartridge_model(out.cartridge_id, registry)
    assert model.get("quantized") is True


# =========================================================================== #
# The studio service
# =========================================================================== #
def test_the_studio_builds_certifies_and_serves(built):
    out, registry = built
    assert out.cartridge_id
    assert out.status in ("certified", "provisional")
    assert len(out.axes) == 7
    assert all(g["passed"] for g in out.gates)
    assert out.n_train > 0 and out.n_holdout > 0

    served = studio.predict(out.cartridge_id, ["excellent work"], registry)
    assert served["error"] == ""
    assert served["predictions"][0]["label"] in out.labels


def test_a_cached_rebuild_still_shows_its_evidence(built):
    """The second identical build is a cache hit, so the Proving Ground never
    runs — but the certificate was stored for exactly this reason. Reuse that
    drops the evidence looks like a build that skipped certification."""
    out, registry = built
    again = studio.build(
        "Classify support tickets by sentiment on an android phone, offline",
        _examples(),
        registry_path=registry,
    )
    assert again.admitted
    assert again.stage_reached == "cache"
    assert again.cartridge_id == out.cartridge_id

    assert again.plain_summary == out.plain_summary
    assert again.status == out.status
    assert again.n_holdout == out.n_holdout
    assert again.weights_bytes == out.weights_bytes
    assert [(a["name"], a["passed"], a["blocking"]) for a in again.axes] == \
           [(a["name"], a["passed"], a["blocking"]) for a in out.axes]
    assert again.plan["base_ref"] == out.plan["base_ref"]


def test_a_cached_axis_carries_the_bar_it_cleared(built):
    """A score with no threshold beside it is not a certificate."""
    out, registry = built
    again = studio.build(
        "Classify support tickets by sentiment on an android phone, offline",
        _examples(),
        registry_path=registry,
    )
    task = next(a for a in again.axes if a["name"] == "task_metric")
    assert task["threshold"] == pytest.approx(0.80)
    assert task["score"] >= task["threshold"]


def test_listed_labels_are_the_classes_not_the_schema(built):
    """``io_contract.output_schema`` is ``{"label": "str"}`` — the shape of a
    response. Reporting its keys would tell a user their sentiment model emits
    one class called "label"."""
    _out, registry = built
    row = studio.list_models(registry)[0]
    assert row["labels"] == ["negative", "positive"]


def test_too_few_examples_is_refused_not_crashed(tmp_path):
    """Below the seed floor the Data Factory refuses. That is the product
    working, and the UI renders it as an outcome rather than a failure."""
    out = studio.build(
        "Classify tickets by sentiment",
        _examples(10),
        registry_path=tmp_path / "registry",
    )
    assert out.admitted is False
    assert "floor" in out.refusal.lower() or "seed" in out.refusal.lower()


def test_a_single_label_cannot_train_a_classifier(tmp_path):
    out = studio.build(
        "Classify tickets",
        [("all of these are positive", "positive")] * 100,
        registry_path=tmp_path / "registry",
    )
    assert out.admitted is False
    assert "two distinct labels" in out.refusal


def test_an_empty_description_is_refused(tmp_path):
    assert studio.build("   ", _examples(), registry_path=tmp_path).admitted is False


def test_listing_models_reports_servability(built):
    out, registry = built
    rows = studio.list_models(registry)
    assert len(rows) == 1
    row = rows[0]
    assert row["cartridge_id"] == out.cartridge_id
    assert row["certified"] is True
    assert row["servable"] is True
    assert row["has_weights"] is True


def test_listing_an_absent_registry_is_empty_not_an_error(tmp_path):
    assert studio.list_models(tmp_path / "nope") == []


def test_predicting_with_no_text_is_reported(built):
    out, registry = built
    assert studio.predict(out.cartridge_id, ["   ", ""], registry)["error"]


# =========================================================================== #
# Packaging
# =========================================================================== #
def test_the_bundle_is_weights_not_runtime(built):
    """The model is kilobytes; the binary is mostly Python and numpy. That is
    the fact that decides whether one binary per cartridge is affordable."""
    out, registry = built
    spec = bundle_spec(out.cartridge_id, registry)
    assert spec["weight_bytes"] < 100_000
    assert set(spec["weight_files"]) == {"model.json", "weights.npz"}


def test_packaging_an_unbuilt_cartridge_refuses_cleanly(tmp_path):
    result = package("deadbeef", registry_path=tmp_path, dist_dir=tmp_path / "dist")
    assert result.ok is False
    assert "no weights" in result.error


@pytest.mark.skipif(
    importlib.util.find_spec("PyInstaller") is None, reason="PyInstaller not installed"
)
@pytest.mark.integration
def test_a_cartridge_freezes_into_a_runnable_binary(built, tmp_path):
    """The full shipping path. Deselected by default — it freezes an entire
    interpreter and takes tens of seconds. Run with ``-m integration``."""
    out, registry = built
    result = package(
        out.cartridge_id, registry_path=registry, dist_dir=tmp_path / "dist",
    )
    assert result.ok, result.error
    assert result.exe_path is not None and result.exe_path.exists()
    assert result.size_bytes > 1_000_000        # a real frozen runtime


# =========================================================================== #
# The runner shares the certified inference path
# =========================================================================== #
def test_the_packaged_runner_imports_the_real_classifier():
    """It must not reimplement inference. A second copy of the dequantisation
    logic is free to drift from the one the Proving Ground measured, and a
    shipped model that predicts differently from the certified one is the worst
    defect available here."""
    from modelrig.package_exe import _RUNNER_TEMPLATE

    assert "import classifier" in _RUNNER_TEMPLATE
    assert "classifier.predict" in _RUNNER_TEMPLATE
    assert "classifier.load_model" in _RUNNER_TEMPLATE


def test_quantisation_survives_a_disk_round_trip():
    """What the runner relies on: save -> load -> predict is stable."""
    train = _examples(40)
    labels = sorted({label for _t, label in train})
    model = classifier.fit_centroid(train, labels)
    quantised, _stats = classifier.quantize_model(model, "int8")

    probes = ["helpful and wonderful", "broken and terrible"]
    before = classifier.predict(quantised, probes)

    with tempfile.TemporaryDirectory() as d:
        classifier.save_model(quantised, d)
        after = classifier.predict(classifier.load_model(d), probes)

    assert before == after
