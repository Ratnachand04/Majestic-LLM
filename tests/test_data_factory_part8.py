"""The Data Factory (Part 8): three operations, saturation, and the ceiling.

The reframe under test is §1's: "synthetic data generation" was conflating three
operations with radically different risk, and the conflation hid the fact that
the most common primitive barely needs synthesis at all.
"""
from __future__ import annotations

import random

import pytest

from modelrig.augment import (
    DOMINANT_OPERATION,
    MAX_DEPTH,
    MAX_MULTIPLICITY,
    AugmentedExample,
    Operation,
    Pseudonymiser,
    augment,
    compose,
    cost_estimate,
    multiplicity_report,
    operators_for,
    source_id_of,
    spot_check_sample,
)
from modelrig.diversity import (
    ETA_0,
    amplification_ceiling,
    coverage,
    effective_modes,
    effective_sample_size,
    explain_refusal,
    generate_until_saturated,
    kfold_coverage,
    seed_predicate,
    seeds_required,
    vendi_score,
)
from modelrig.primitives import TaskPrimitive

FORM = "Patient: Alice Smith\nTest: CBC\nDoctor: Doctor Jones\nDate: 2026-01-04"


def _seeds(n: int = 20) -> list[tuple[str, dict]]:
    return [
        (f"{FORM}\nRef: {i}", {"patient": "Alice Smith", "ref": str(i)})
        for i in range(n)
    ]


# =========================================================================== #
# §1 — three operations, not one
# =========================================================================== #
def test_the_three_operations_differ_in_label_provenance():
    """Which is what makes their risk differ, and it is a mechanism rather than
    an observation."""
    assert Operation.AUGMENTATION.label_is_real is True
    assert Operation.BACKTRANSLATION.label_is_real is True     # output is real
    assert Operation.SYNTHESIS.label_is_real is False

    assert Operation.AUGMENTATION.collapse_risk == "very low"
    assert Operation.SYNTHESIS.collapse_risk == "high"


def test_augmentation_needs_no_teacher_at_all():
    """The economic consequence: for augmentation-dominated primitives the Data
    Factory costs close to nothing. The dollar figures quoted for generation
    apply to synthesis-dominated primitives."""
    assert Operation.AUGMENTATION.teacher_tokens_per_example == 0
    assert Operation.SYNTHESIS.teacher_tokens_per_example > 0

    aug = augment(_seeds(), TaskPrimitive.EXTRACT, per_seed=4)
    cost = cost_estimate(aug)
    assert cost["usd"] == 0.0
    assert cost["free_share"] == 1.0


def test_extraction_is_augmentation_dominated():
    """The customer usually already has (document, label) pairs and does not
    know it — they are in the operational database."""
    assert DOMINANT_OPERATION[TaskPrimitive.EXTRACT] is Operation.AUGMENTATION
    assert DOMINANT_OPERATION[TaskPrimitive.CLASSIFY] is Operation.AUGMENTATION
    assert DOMINANT_OPERATION[TaskPrimitive.GENERATE] is Operation.SYNTHESIS


def test_augmented_data_needs_no_quality_filter():
    """Its labels are correct by construction, so a filter can only discard
    valid examples."""
    assert Operation.AUGMENTATION.needs_quality_filter is False
    assert Operation.SYNTHESIS.needs_quality_filter is True


# =========================================================================== #
# §2 — the operator monoid
# =========================================================================== #
def test_operators_compose_and_stay_admissible():
    ops = operators_for(TaskPrimitive.EXTRACT)[:2]
    composed = compose(ops)
    assert " o " in composed.name
    assert isinstance(composed.apply(FORM, random.Random(0)), str)


def test_composition_depth_is_capped():
    """Past about three, a document blurred, rotated, occluded and re-encoded is
    not a document anyone will ever scan."""
    ops = operators_for(TaskPrimitive.EXTRACT)
    with pytest.raises(ValueError, match="drifts off the manifold"):
        compose(list(ops) * 2, max_depth=MAX_DEPTH)


def test_augmentation_preserves_the_label():
    """The admissibility condition: y(T(d)) = y(d)."""
    seeds = _seeds(5)
    for example in augment(seeds, TaskPrimitive.EXTRACT, per_seed=3):
        assert example.label in [label for _text, label in seeds]
        assert example.label_is_real is True


def test_format_operators_apply_to_every_primitive():
    """Whitespace and abbreviation drift are universal; occlusion and OCR noise
    are not, because they describe a scanner rather than a task."""
    everywhere = {op.name for op in operators_for(TaskPrimitive.TOOLCALL)}
    documents = {op.name for op in operators_for(TaskPrimitive.EXTRACT)}
    assert everywhere == {"whitespace_drift", "abbreviation"}
    assert {"occlusion", "ocr_noise"} <= documents


# =========================================================================== #
# §4 — pseudonymisation is privacy AND augmentation
# =========================================================================== #
def test_pseudonymisation_updates_the_label_in_step():
    """For extraction the PII *is* the label: you cannot redact patient
    identifiers from a patient-identifier extractor's training data."""
    p = Pseudonymiser()
    text, label = p.apply(
        "Patient Alice Smith seen today. Alice Smith is 40.",
        {"patient": "Alice Smith"}, ["Alice Smith"],
    )
    assert "Alice Smith" not in text
    assert label["patient"] != "Alice Smith"
    assert label["patient"] in text          # consistent, and still the label


def test_one_document_yields_many_correctly_labelled_variants():
    """Privacy mitigation and data amplification are the same operation."""
    p = Pseudonymiser()
    names = {
        p.apply(FORM, {"patient": "Alice Smith"}, ["Alice Smith"], variant=v)[1]["patient"]
        for v in range(5)
    }
    assert len(names) > 1                    # genuinely different identities


def test_replacement_is_consistent_within_a_variant():
    """Inconsistent replacement would break the task rather than protect it."""
    p = Pseudonymiser()
    text, _ = p.apply("Alice Smith called. Ask for Alice Smith.",
                      {}, ["Alice Smith"], variant=0)
    assert text.count(text.split()[0]) == 2


# =========================================================================== #
# §14, §21 — provenance, and multiplicity as a privacy control
# =========================================================================== #
def test_every_generated_example_carries_its_source():
    """Contamination must be checked at the SOURCE-DOCUMENT level: an augmented
    variant of a held-out document leaks even though the bytes differ."""
    for example in augment(_seeds(4), TaskPrimitive.EXTRACT, per_seed=2):
        assert example.source_id
        assert example.operation is Operation.AUGMENTATION


def test_identical_documents_share_one_source_id():
    assert source_id_of(FORM) == source_id_of(FORM)
    assert source_id_of(FORM) != source_id_of(FORM + " extra")


def test_per_source_multiplicity_is_capped():
    """Twenty transforms of one seed are twenty near-copies of its content —
    a memorisation risk rather than data, and the weights ship to devices."""
    aug = augment(_seeds(5), TaskPrimitive.EXTRACT, per_seed=40)
    report = multiplicity_report(aug)
    assert report["max_multiplicity"] <= MAX_MULTIPLICITY
    assert report["within_cap"] is True


def test_multiplicity_is_reported_as_a_privacy_prior():
    """A build at k_max = 40 should expect a worse extraction rate than one at
    k_max = 5. Knowing that before the audit turns a surprise into a prediction."""
    report = multiplicity_report(augment(_seeds(10), TaskPrimitive.EXTRACT, per_seed=3))
    assert report["sources"] == 10
    assert "white-box access" in report["why"]


def test_the_spot_check_sample_is_bounded():
    """The one place human judgement is not replaceable, and it takes ten
    minutes."""
    aug = augment(_seeds(60), TaskPrimitive.EXTRACT, per_seed=4)
    assert len(spot_check_sample(aug, n=100)) == 100
    assert len(spot_check_sample(aug[:20], n=100)) == 20


# =========================================================================== #
# §7 — effective modes, not example count
# =========================================================================== #
def test_the_hill_number_counts_modes_not_rows():
    """Sixty thousand near-identical rows is one example repeated."""
    assert effective_modes(list(range(8)) * 10) == pytest.approx(8.0, abs=0.01)
    assert effective_modes(["a"] * 80) == pytest.approx(1.0, abs=0.01)
    assert effective_modes([i % 100 for i in range(60_000)]) == pytest.approx(100, abs=1)


def test_the_vendi_score_avoids_the_clustering_step():
    identical = [[1.0, 1.0], [1.0, 1.0]]
    orthogonal = [[1.0, 0.0], [0.0, 1.0]]
    assert vendi_score(identical) < vendi_score(orthogonal)
    assert vendi_score([]) == 0.0


def test_an_empty_set_has_no_modes():
    assert effective_modes([]) == 0.0


# =========================================================================== #
# §8 — the diversity ratio IS the discount factor
# =========================================================================== #
def test_collapsed_and_diverse_sets_contribute_wildly_differently():
    """Same row count, eighty times the contribution. Counting rows cannot see
    this; N_eff can."""
    assert effective_sample_size(0, 8_000) == pytest.approx(2_400)
    assert effective_sample_size(0, 100) == pytest.approx(30)


def test_only_one_constant_remains_to_be_estimated():
    """The old form needed eta_0 and rho_0; N_eff is MEASURED per build, so the
    more speculative constant disappears."""
    assert 0.0 < ETA_0 < 1.0


def test_the_seed_predicate_uses_effective_not_raw_counts():
    # 60k collapsed rows do not rescue 40 seeds.
    assert seed_predicate(n_real=40, n_eff_synthetic=100, floor=200) is False
    assert seed_predicate(n_real=40, n_eff_synthetic=8_000, floor=200) is True


def test_negative_counts_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        effective_sample_size(-1, 100)


# =========================================================================== #
# §9 — saturation stopping
# =========================================================================== #
def test_generation_stops_when_diversity_stops_growing():
    """`synthetic_target` was a number somebody guessed. The count should be a
    reported measurement, not an input."""
    rng = random.Random(0)

    def batch(n):
        return [f"mode{rng.randrange(40)} v{rng.random()}" for _ in range(n)]

    pool, trace = generate_until_saturated(batch, batch_size=2_000, max_examples=60_000)
    assert trace.saturated is True
    assert trace.stopped_at < 60_000               # far short of the cap
    assert len(pool) == trace.stopped_at


def test_saturation_cuts_the_binding_constraint():
    """Wall clock, not dollars, is what binds: generation dominates build
    latency, and stopping early cuts it by most of its length."""
    rng = random.Random(1)

    def batch(n):
        return [f"mode{rng.randrange(30)} v{rng.random()}" for _ in range(n)]

    _pool, trace = generate_until_saturated(batch, batch_size=1_000, max_examples=60_000)
    assert trace.stopped_at / 60_000 < 0.5


def test_a_genuinely_diverse_generator_runs_to_the_cap():
    rng = random.Random(2)

    def batch(n):
        return [f"unique{rng.random()}{i}" for i in range(n)]

    _pool, trace = generate_until_saturated(batch, batch_size=500, max_examples=2_000)
    assert trace.stopped_at is None or trace.stopped_at >= 1_000


def test_collapse_is_simply_saturation_at_a_low_n_eff():
    """§19: the response is to stop generating, not to abort. Under a collapse
    model over-generating damages the build; under saturation it wastes money."""
    def batch(n):
        return ["the same row"] * n

    pool, trace = generate_until_saturated(batch, batch_size=500, max_examples=10_000)
    assert trace.final == pytest.approx(1.0, abs=0.01)
    assert trace.saturated is True
    assert seed_predicate(40, trace.final, floor=200) is False     # then refused


# =========================================================================== #
# §10 — coverage, measured without leaking
# =========================================================================== #
def test_coverage_measures_against_reality_not_internal_spread():
    """A generator can produce 8000 diverse modes all unlike the deployment
    distribution. N_eff cannot see that; coverage can."""
    reference = [[0.0, 0.0], [1.0, 1.0]]
    near = [[0.05, 0.05], [0.95, 0.95]]
    far = [[9.0, 9.0], [8.0, 8.0]]
    assert coverage(reference, near) == 1.0
    assert coverage(reference, far) == 0.0


def test_coverage_is_measured_inside_the_seed_set():
    """The leakage trap: measuring against the holdout and regenerating until it
    improves is fitting to the holdout, and would silently invalidate every
    number the Proving Ground produces."""
    rng = random.Random(0)
    seeds = [[rng.random(), rng.random()] for _ in range(50)]

    def generate(rest):
        return [[x + rng.gauss(0, 0.02), y + rng.gauss(0, 0.02)]
                for x, y in rest for _ in range(2)]

    report = kfold_coverage(seeds, generate, k=5)
    assert len(report.folds) == 5
    assert report.mean > 0.8
    assert report.as_dict()["measured_against"] == "seed folds"
    assert "invalidate" in report.as_dict()["why"]


def test_kfold_needs_enough_seeds_to_fold():
    with pytest.raises(ValueError, match="at least k >= 2"):
        kfold_coverage([[0.0]], lambda rest: [], k=5)


# =========================================================================== #
# §20 — the amplification ceiling
# =========================================================================== #
def test_amplification_multiplies_by_at_most_one_plus_kappa():
    """Fifty seeds cannot become five thousand examples' worth of information."""
    assert amplification_ceiling(50, kappa=1.0) == 100
    assert amplification_ceiling(50, kappa=5.0) == 300


def test_the_seed_floor_becomes_a_derivation():
    """Not a table lookup — which is what makes a refusal explainable."""
    assert seeds_required(floor=200, kappa=1.0) == 100
    assert seeds_required(floor=200, kappa=0.0) == 200      # no amplification


def test_the_refusal_says_more_seeds_not_more_generation():
    """The most useful sentence the system can produce on a failed build."""
    result = explain_refusal(n_real=40, floor=200, kappa=1.2)
    assert result["sufficient"] is False
    assert result["seeds_required"] == 91
    assert "more seeds, not more generation" in result["message"]


def test_a_sufficient_seed_set_is_told_so():
    result = explain_refusal(n_real=150, floor=200, kappa=1.0)
    assert result["sufficient"] is True
    assert "clears" in result["message"]


def test_kappa_is_higher_where_labels_are_real():
    """§20: augmentation-dominated primitives have high kappa because labels are
    real and transforms are many; synthesis-dominated ones have low kappa."""
    assert seeds_required(200, kappa=4.0) < seeds_required(200, kappa=0.5)


# =========================================================================== #
# §19 — the citation was at the wrong subsystem
# =========================================================================== #
def test_the_data_factory_justifies_its_floors_by_saturation():
    """The guardrails do not change; the reasoning does, and the reasoning
    determines what happens when a guardrail fires."""
    import modelrig.data_factory as df

    assert "not recursive" in df.__doc__
    assert "saturation" in df.__doc__.lower()


def test_the_flywheel_keeps_the_collapse_citation():
    """That IS the recursive structure, which is why I-03 forbids it."""
    import majestic.flywheel as fw

    assert "2305.17493" in fw.__doc__
    assert "recursive" in fw.__doc__


def test_the_seed_floor_refusal_no_longer_blames_recursive_collapse():
    from modelrig.gates import gate1_spec_admissibility
    from modelrig.ir import SpecIR
    from modelrig.licence import DataRights

    spec = SpecIR(task_primitive=TaskPrimitive.GENERATE, seed_data_count=5,
                  data_rights=DataRights.CUSTOMER_OWNED)
    result = gate1_spec_admissibility(spec)
    reasons = " ".join(result.reasons)
    assert "2305.17493" not in reasons
    assert "more generation will not substitute" in reasons


def test_an_augmented_example_records_which_operation_made_it():
    example = AugmentedExample(text="x", label={}, source_id="s")
    assert example.operation is Operation.AUGMENTATION
    assert example.label_is_real is True
