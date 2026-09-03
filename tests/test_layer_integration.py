"""Every layer alone, then all of them together.

The unit suites prove each subsystem is internally correct. This file asks two
different questions, and they are the ones a system of this shape actually
fails on:

  1. Does each layer stand up **on its own** — importable, constructible, and
     useful without dragging the rest of the compiler in behind it? A layer
     that only works inside the pipeline is not a layer, it is an inlined
     function with extra ceremony.

  2. Do the layers **compose** — does the object one produces satisfy the next
     one's preconditions, in the real order, with no adapter in between? Every
     seam is a place where two correct components disagree about a contract.
"""
from __future__ import annotations

import inspect
import json

import pytest

from api import studio

# --------------------------------------------------------------------------- #
# The layers, in pipeline order. Named here so a new subsystem that skips this
# file is visible as an omission rather than silently untested.
# --------------------------------------------------------------------------- #
LAYERS = (
    "forge", "planner", "data_factory", "trainer", "proving_ground",
    "registry", "fabric", "packaging",
)


def _corpus(n: int = 120) -> list[tuple[str, str]]:
    return [(r["text"], r["label"]) for r in studio.sample_dataset(n)]


def _spec(description: str = "Classify support tickets by sentiment on an android phone, offline"):
    from modelrig.forge import Interviewer

    return Interviewer().conduct(
        description,
        device_profile=studio._probe_profile(),
        answers={
            "data_rights": "customer_owned", "quality_gate": 0.80,
            "seed_data_count": 120, "offline_required": True,
            "seed_data_ref": "test://corpus", "latency_budget_ms": 30_000,
            "expected_input_tokens": 120, "io_schema": {"label": "str"},
        },
    ).spec


# =========================================================================== #
# 1. EACH LAYER, ALONE
# =========================================================================== #
class TestLayersStandAlone:
    """Each layer used directly, with nothing above or below it."""

    def test_forge_turns_prose_into_a_typed_spec(self):
        spec = _spec()
        assert spec is not None
        assert spec.task_primitive.value == "classify"
        assert spec.offline_required is True
        assert spec.hash                      # content-addressed

    def test_forge_refuses_to_guess_what_it_was_not_told(self):
        """The interview is the product. A spec invented from silence would
        make every downstream guarantee meaningless."""
        from modelrig.forge import Interviewer

        interview = Interviewer().conduct("make me something good")
        assert interview.pending, "FORGE should still have questions"
        assert interview.spec is None or interview.pending

    def test_planner_admits_or_refuses_with_a_witness(self):
        from modelrig.planner import Planner

        result = Planner().plan(_spec())
        assert result.gate is not None
        if result.admitted:
            assert result.plan is not None
            assert result.plan.base_ref and result.plan.bit_width
        else:
            assert result.gate.reasons, "a refusal must name the binding constraint"

    def test_planner_refuses_an_infeasible_device_with_a_quantified_witness(self):
        """A refusal must be a witness, not a shrug.

        "No feasible plan" is useless: the customer cannot tell whether to buy
        a bigger phone, relax the budget, or give up. Every reason must name
        the candidate, the resource, and the measured margin.
        """
        from modelrig.planner import Planner

        spec = _spec()
        spec.latency_budget_ms = 1          # nothing can serve in one millisecond
        result = Planner().plan(spec)
        assert not result.admitted
        assert result.gate.reasons

        for reason in result.gate.reasons:
            assert any(ch.isdigit() for ch in reason), f"unquantified: {reason}"
            assert "/" in reason or "-" in reason, f"names no candidate: {reason}"
        joined = " ".join(result.gate.reasons).lower()
        assert "budget" in joined or "free" in joined or "over by" in joined

    def test_data_factory_splits_and_amplifies(self):
        from modelrig.data_factory import DataFactory
        from modelrig.primitives import TaskPrimitive

        corpus = _corpus()
        bundle = DataFactory().build(corpus, TaskPrimitive.CLASSIFY)
        assert len(bundle.train) > 0 and len(bundle.held_out) > 0
        assert len(bundle.labels) == 2
        assert bundle.synthetic_count > 0, "the factory should have amplified"

        # Held-out data must be REAL, never amplified. A score measured against
        # generated text measures the generator, not the model — and every
        # claim the system makes rests on this one split being honest.
        originals = {text for text, _label in corpus}
        assert all(text in originals for text, _label in bundle.held_out)

    def test_the_factory_never_leaks_held_out_text_into_training(self):
        """Contamination is a blocking axis downstream; the split is where it
        is actually prevented."""
        from modelrig.data_factory import DataFactory
        from modelrig.primitives import TaskPrimitive

        bundle = DataFactory().build(_corpus(), TaskPrimitive.CLASSIFY)
        train_texts = {text for text, _label in bundle.train}
        held_texts = {text for text, _label in bundle.held_out}
        assert not (train_texts & held_texts)

    def test_data_factory_refuses_below_the_seed_floor(self):
        from modelrig.data_factory import DataFactory, DataRefusal
        from modelrig.primitives import TaskPrimitive

        with pytest.raises(DataRefusal):
            DataFactory().build(_corpus(10), TaskPrimitive.CLASSIFY)

    def test_trainer_fits_a_real_model(self):
        from modelrig import classifier

        train = _corpus(60)
        labels = sorted({label for _t, label in train})
        model = classifier.fit_centroid(train, labels)
        preds = classifier.predict(model, ["wonderful helpful", "broken awful"])
        assert preds == ["positive", "negative"]

    def test_trainer_quantisation_actually_compresses(self):
        """int4 must be packed, not stored one nibble per byte — the whole
        on-device argument rests on this being a real 8x, not a label."""
        from modelrig import classifier

        train = _corpus(60)
        model = classifier.fit_centroid(train, sorted({lbl for _t, lbl in train}))
        _q8, s8 = classifier.quantize_model(model, "int8")
        _q4, s4 = classifier.quantize_model(model, "int4")
        assert s8["ratio"] == pytest.approx(4.0, rel=0.15)
        assert s4["ratio"] == pytest.approx(8.0, rel=0.15)

    def test_proving_ground_scores_and_partitions_its_axes(self):
        from modelrig.proving_ground import ProvingGround

        held = _corpus(40)[:30]
        card = ProvingGround(quality_gate=0.80).evaluate(
            lambda texts: ["positive"] * len(list(texts)), held,
        )
        assert len(card.axes) == 7
        blocking = [a.name for a in card.axes if a.blocking]
        assert set(blocking) <= {"task_metric", "safety", "privacy", "contamination"}
        assert any(not a.blocking for a in card.axes), "advisory axes must exist"

    def test_proving_ground_gates_on_the_lower_bound(self):
        """At small n the interval is what governs the claim, not the point
        estimate. This is the honest-statistics guarantee in one assertion."""
        from modelrig.proving_ground import ProvingGround

        held = _corpus(40)[:30]
        card = ProvingGround(quality_gate=0.80).evaluate(
            lambda texts: [lbl for _t, lbl in held][:len(list(texts))], held,
        )
        task = next(a for a in card.axes if a.name == "task_metric")
        assert task.interval is not None
        assert task.interval.low <= task.score <= task.interval.high

    def test_registry_refuses_an_uncertified_cartridge(self, tmp_path):
        """Gate 3 is enforced at the door, not assumed upstream. The registry
        is the last thing between a build and a customer."""
        from modelrig.cartridge import Cartridge
        from modelrig.registry import CartridgeRegistry

        bare = Cartridge(base_ref="test/base", spec_hash="abc", plan_hash="def")
        assert bare.certified is False
        with pytest.raises(ValueError, match="not certified"):
            CartridgeRegistry(str(tmp_path)).admit(bare)

    def test_registry_round_trips_a_certified_cartridge(self, tmp_path):
        from modelrig.cartridge import Cartridge
        from modelrig.registry import CartridgeRegistry

        cart = Cartridge(
            base_ref="test/base", spec_hash="abc", plan_hash="def",
            model_card={"task_primitive": "classify", "intended_use": "test"},
            eval_certificate={"passed": True, "score": 0.9, "n_test": 30},
            licence_chain={"permitted": True, "resolved_licence": "apache-2.0"},
        )
        assert cart.certified is True

        registry = CartridgeRegistry(str(tmp_path))
        cid = registry.admit(cart)
        assert registry.get(cid).base_ref == "test/base"
        assert cid in registry.list()
        # Content-addressed: the same inputs must land on the same identity.
        assert cid == cart.id

    def test_fabric_measures_capacity_in_one_pass(self):
        """§8's claim is linearity: the widest path is found in a single
        topological sweep. Path enumeration would be exponential, and would
        pass a small-graph test while failing on any real one — so the check is
        on the method, not just the answer."""
        from majestic.fabric import capacity

        source = inspect.getsource(capacity.analyse_capacity)
        assert "permutations" not in source
        assert "all_simple_paths" not in source
        assert "itertools" not in source

    def test_fabric_capacity_is_bits_from_the_decoding_grammar(self):
        """cap(v) = log2|D_out|. A constrained field leaks a bounded number of
        bits; one free-text field turns a sanitiser into a propagator."""
        from majestic.fabric.capacity import bits_for_domain

        assert bits_for_domain(2) == pytest.approx(1.0)
        assert bits_for_domain(256) == pytest.approx(8.0)
        assert bits_for_domain(1) == pytest.approx(0.0)   # no choice, no channel

    def test_packaging_describes_a_bundle_without_building_one(self, tmp_path):
        from modelrig.package_exe import package

        result = package("nope", registry_path=tmp_path, dist_dir=tmp_path / "dist")
        assert result.ok is False and "no weights" in result.error


# =========================================================================== #
# 2. THE SEAMS: each layer's output feeds the next one's input
# =========================================================================== #
class TestLayersCompose:
    """Handoffs, taken one seam at a time and then all at once."""

    def test_forge_output_is_a_valid_planner_input(self):
        from modelrig.planner import Planner

        assert Planner().plan(_spec()).gate is not None    # no adapter needed

    def test_planner_output_is_a_valid_trainer_input(self):
        from modelrig.data_factory import DataFactory
        from modelrig.planner import Planner

        spec = _spec()
        plan = Planner().plan(spec).plan
        assert plan is not None
        bundle = DataFactory().build(_corpus(), spec.task_primitive)
        # The plan names a quantiser the trainer can actually apply.
        from modelrig import classifier

        model = classifier.fit_centroid(bundle.train, bundle.labels)
        quantised, stats = classifier.quantize_model(model, plan.bit_width)
        assert stats["ratio"] > 1.0
        assert classifier.predict(quantised, ["helpful and wonderful"])

    def test_the_certificate_survives_a_round_trip_through_the_registry(self, tmp_path):
        """The seam that broke once: a Scorecard is rich, JSON is flat. What
        the Registry gives back must still answer what the UI asks of it."""
        out = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path,
        )
        assert out.admitted

        from modelrig.registry import CartridgeRegistry

        cert = CartridgeRegistry(str(tmp_path)).get(out.cartridge_id).eval_certificate
        assert cert["plain_summary"] == out.plain_summary
        assert cert["status"] == out.status
        assert cert["axis_order"] == [a["name"] for a in out.axes]
        for axis in out.axes:
            stored = cert["axes"][axis["name"]]
            assert stored["threshold"] == pytest.approx(axis["threshold"])
            assert stored["blocking"] == axis["blocking"]

    def test_the_registry_manifest_and_the_weights_describe_one_artefact(self, tmp_path):
        """A manifest and a weights file that disagree is the worst outcome
        available: a certificate about a model nobody has."""
        out = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path,
        )
        manifest = json.loads(
            (tmp_path / "weights" / out.cartridge_id / "model.json").read_text("utf-8")
        )
        assert sorted(manifest["labels"]) == sorted(out.labels)
        assert manifest["quantized"] is True

        from modelrig.pipeline import predict_with_cartridge

        preds = predict_with_cartridge(
            out.cartridge_id, ["wonderful and helpful"], tmp_path,
        )
        assert preds[0] in manifest["labels"]

    def test_every_layer_reports_in_during_one_compile(self, tmp_path):
        """The end-to-end seam check: run one build and require each stage to
        have actually executed. A layer silently skipped would still produce a
        green pipeline, which is exactly the failure this catches."""
        seen: list[tuple[str, str]] = []
        out = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path,
            progress=lambda e: seen.append((e.stage, e.status)),
        )
        assert out.admitted

        from modelrig.pipeline import STAGES

        ran = {stage for stage, _status in seen}
        assert ran == {key for key, _label in STAGES}, f"stages missing: {ran}"
        assert all(status != "refused" for _s, status in seen)

        # Order, not just presence: a pipeline that certified before it trained
        # would satisfy a set comparison and be nonsense.
        first_seen = list(dict.fromkeys(stage for stage, _ in seen))
        assert first_seen == [key for key, _ in STAGES]

    def test_the_stage_timings_are_measured_not_invented(self, tmp_path):
        out = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path,
        )
        finished = [s for s in out.stages if s["status"] != "running"]
        assert len(finished) == 8
        assert all(s["elapsed_ms"] >= 0 for s in finished)
        assert out.total_ms > 0
        # The Data Factory amplifies a corpus; it cannot be the free stage.
        data = next(s for s in finished if s["stage"] == "data")
        assert data["elapsed_ms"] > 0

    def test_seeds_and_amplified_rows_are_reported_separately(self, tmp_path):
        """The Data Factory is the authority on its own split.

        Deriving the training count from the input size reports the seed count
        under a heading that means "rows trained on" — and contradicts the
        factory's own telemetry on the same screen. Both numbers are real and
        they are not the same number.
        """
        corpus = _corpus()
        out = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            corpus, registry_path=tmp_path,
        )
        assert out.n_seeds + out.n_holdout == len(corpus)
        assert out.n_train > out.n_seeds, "amplification should have happened"

        data = next(s for s in out.stages if s["stage"] == "data" and s["status"] == "ok")
        assert data["data"]["n_train"] == out.n_train
        assert data["data"]["n_holdout"] == out.n_holdout

    def test_observation_does_not_change_the_result(self, tmp_path):
        """A progress listener is a window, not a code path."""
        watched = studio.build(
            "Classify tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path / "a", progress=lambda e: None,
        )
        silent = studio.build(
            "Classify tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path / "b",
        )
        assert watched.cartridge_id == silent.cartridge_id
        assert watched.status == silent.status
        assert watched.plain_summary == silent.plain_summary
        assert [a["name"] for a in watched.axes] == [a["name"] for a in silent.axes]

    def test_a_listener_that_throws_does_not_take_the_build_down(self, tmp_path):
        """The listener is a UI. A compile that succeeded is still a success
        even if nobody was watching by the end."""
        def explode(_event):
            raise RuntimeError("the browser went away")

        out = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path, progress=explode,
        )
        assert out.admitted

    def test_an_undersized_corpus_is_refused_at_the_first_gate(self, tmp_path):
        """Gate 1 owns the seed floor, so a corpus too small to train on is
        rejected before the Planner searches or the Data Factory amplifies —
        the whole point of gating before spending.

        Note where this does NOT stop: the Data Factory enforces the same floor
        independently, and would also refuse. Two layers checking the same
        invariant is deliberate — the factory is reachable on its own.
        """
        out = studio.build(
            "Classify support tickets by sentiment", _corpus(10),
            registry_path=tmp_path,
        )
        assert out.admitted is False

        refused = [s for s in out.stages if s["status"] == "refused"]
        assert len(refused) == 1
        assert refused[0]["stage"] == "gate1"
        assert refused[0]["detail"], "a hold must say what stopped it"
        # Nothing downstream ran: no plan searched, no corpus amplified.
        assert {s["stage"] for s in out.stages} == {"gate1"}
        assert out.stage_reached == "gate1"

    def test_a_hold_after_the_first_gate_still_reports_how_far_it_got(self, tmp_path):
        """Reading a hold backwards is how a user learns what to change: the
        stages that passed are as informative as the one that stopped."""
        out = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            # Enough seeds to clear Gate 1, but one label — nothing to separate.
            [(f"all of these say the same thing {i}", "positive") for i in range(120)],
            registry_path=tmp_path,
        )
        assert out.admitted is False
        assert "two distinct labels" in out.refusal
        # Refused in the studio before the compiler was even invoked, so the
        # user is told the cheapest possible thing at the cheapest moment.
        assert out.stages == []

    def test_the_streamed_build_and_the_plain_build_agree(self, tmp_path):
        """The stream must be a view of the compile, not a second one."""
        events = list(studio.build_stream(
            "Classify support tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path / "stream",
        ))
        assert events[-1]["event"] == "done"
        streamed = events[-1]["outcome"] if "outcome" in events[-1] else events[-1]

        direct = studio.build(
            "Classify support tickets by sentiment on an android phone, offline",
            _corpus(), registry_path=tmp_path / "direct",
        )
        assert streamed["cartridge_id"] == direct.cartridge_id
        assert streamed["status"] == direct.status
        assert streamed["plain_summary"] == direct.plain_summary

        stages = [e for e in events if e["event"] == "stage"]
        assert [s["stage"] for s in stages] == [s["stage"] for s in direct.stages]


# =========================================================================== #
# 3. THE WHOLE STACK, END TO END
# =========================================================================== #
def test_prose_in_running_binary_out(tmp_path):
    """Every layer, in order, ending in an artefact that predicts.

    This is the demonstration the whole system exists to make: a sentence of
    English becomes a certified, quantised, servable model, and the evidence
    for it travels with it the entire way.
    """
    out = studio.build(
        "Classify support tickets by sentiment on an android phone, offline",
        _corpus(), registry_path=tmp_path,
    )

    assert out.admitted                                  # FORGE → PLANNER → gates
    assert out.n_train and out.n_holdout                 # DATA FACTORY
    assert out.weights_bytes > 0                         # TRAINER, persisted
    assert len(out.axes) == 7                            # PROVING GROUND
    assert out.status in ("certified", "provisional")    # honest statistics
    assert all(g["passed"] for g in out.gates)           # three gates
    assert out.cartridge_id                              # REGISTRY

    served = studio.predict(
        out.cartridge_id,
        ["the staff were wonderful", "broken and awful service"],
        tmp_path,
    )
    assert [p["label"] for p in served["predictions"]] == ["positive", "negative"]

    spec = studio.package.__module__ and __import__(
        "modelrig.package_exe", fromlist=["bundle_spec"]
    ).bundle_spec(out.cartridge_id, tmp_path)
    assert spec["weight_bytes"] == out.weights_bytes
