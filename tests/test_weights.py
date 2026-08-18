"""Tests for weight compilation and format selection (Part 3 §1-§5, §10)."""
from __future__ import annotations

import pytest

from modelrig.quantformat import (
    FORMATS,
    measured_ranking,
    select_format,
    supported_formats,
    sweep_plan,
)
from modelrig.weights import (
    GB,
    MB,
    NF4_TRAINING,
    AdapterSpec,
    CompilationPlan,
    MergePlan,
    MergeStrategy,
    PinnedWeights,
    QuantRole,
    QuantSetting,
    WeightError,
    WeightMirror,
    choose_merge_strategy,
    compilation_report,
)

SHA = "a" * 64
DEPLOY = QuantSetting(QuantRole.DEPLOYMENT, "q4_k_m", calibrated=True)


def _pinned(ref="Qwen/Qwen3-1.7B", sha=SHA, size=3_400_000_000) -> PinnedWeights:
    return PinnedWeights(ref=ref, sha256=sha, size_bytes=size)


# =========================================================================== #
# §2 — acquisition and pinning
# =========================================================================== #
def test_weights_are_pinned_by_digest_not_by_name():
    w = _pinned()
    assert w.sha256 == SHA
    assert w.short == "a" * 12


def test_a_malformed_digest_is_rejected():
    for bad in ("abc", "A" * 64, "z" * 64):
        with pytest.raises(WeightError, match="sha256"):
            PinnedWeights(ref="x", sha256=bad, size_bytes=1)


def test_repinning_the_same_name_at_a_new_digest_is_refused():
    """A checkpoint changing under a stable name is what pinning exists to catch."""
    mirror = WeightMirror()
    mirror.pin(_pinned())
    with pytest.raises(WeightError, match="already pinned"):
        mirror.pin(_pinned(sha="b" * 64))


def test_repinning_the_same_digest_is_idempotent():
    mirror = WeightMirror()
    mirror.pin(_pinned())
    mirror.pin(_pinned())
    assert len(mirror.entries) == 1


def test_mirror_is_content_addressed():
    mirror = WeightMirror()
    mirror.pin(_pinned())
    assert SHA in str(mirror.path_for("Qwen/Qwen3-1.7B"))


def test_verify_rehashes_the_local_file(tmp_path):
    import hashlib

    blob = tmp_path / "w.bin"
    blob.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    assert PinnedWeights("x", digest, 7).verify(blob) is True
    assert PinnedWeights("x", "c" * 64, 7).verify(blob) is False


def test_a_single_device_tier_needs_far_less_than_the_full_mirror():
    """android_mid only ever touches the 0.6B and 1.7B bases."""
    mirror = WeightMirror()
    mirror.pin(_pinned("Qwen/Qwen3-0.6B", "1" * 64, 1_200_000_000))
    mirror.pin(_pinned("Qwen/Qwen3-1.7B", "2" * 64, 3_400_000_000))
    mirror.pin(_pinned("Qwen/Qwen3-8B", "3" * 64, 16_000_000_000))
    tier = mirror.subset_for(["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"])
    assert tier < mirror.total_bytes / 2
    assert mirror.footprint()["checkpoints"] == 3


# =========================================================================== #
# §3 — the two quantisations
# =========================================================================== #
def test_the_training_quantisation_does_not_reach_the_device():
    assert NF4_TRAINING.role is QuantRole.TRAINING
    assert NF4_TRAINING.survives_to_device is False
    assert DEPLOY.survives_to_device is True


def test_the_training_quantisation_cannot_be_marked_calibrated():
    """It is data-free by construction; claiming otherwise confuses the two."""
    with pytest.raises(WeightError, match="data-free"):
        QuantSetting(QuantRole.TRAINING, "nf4", calibrated=True)


def test_the_deployment_quantisation_must_be_calibrated():
    with pytest.raises(WeightError, match="task distribution"):
        QuantSetting(QuantRole.DEPLOYMENT, "q4_k_m", calibrated=False)


def test_b_must_be_initialised_to_zero():
    """dW = 0 at step 0, so training begins exactly at the base model's behaviour."""
    AdapterSpec(rank=16, alpha=32)                       # default is zeros
    with pytest.raises(WeightError, match="zeros"):
        AdapterSpec(rank=16, alpha=32, b_init="normal")


def test_adapter_scaling_is_alpha_over_r():
    assert AdapterSpec(rank=16, alpha=32).scaling == 2.0
    assert AdapterSpec(rank=64, alpha=64).scaling == 1.0


def test_adapter_is_small_relative_to_the_base():
    """The learned correction is tens of megabytes, not gigabytes."""
    size = AdapterSpec(rank=16, alpha=32).size_bytes(d_model=2048, n_layers=28)
    assert 5 * MB < size < 100 * MB


# =========================================================================== #
# §4 — merging in the right precision
# =========================================================================== #
def test_merging_must_happen_in_bf16():
    """Merging into the 4-bit training base compounds two quantisation errors."""
    MergePlan(MergeStrategy.MERGED, merge_precision="bf16")
    with pytest.raises(WeightError, match="bf16"):
        MergePlan(MergeStrategy.MERGED, merge_precision="nf4")


def test_one_cartridge_merges():
    plan = choose_merge_strategy(
        cartridges_on_device=1, storage_free_bytes=8 * GB,
        base_bytes=1120 * MB, adapter_bytes=30 * MB,
    )
    assert plan.strategy is MergeStrategy.MERGED
    assert "nothing to share" in plan.reason


def test_many_cartridges_share_one_base():
    """N merged models cost N x full; N adapters cost base + N x 30 MB."""
    plan = choose_merge_strategy(
        cartridges_on_device=20, storage_free_bytes=8 * GB,
        base_bytes=1120 * MB, adapter_bytes=30 * MB,
    )
    assert plan.strategy is MergeStrategy.SEPARATE
    assert "share one base" in plan.reason


def test_a_runtime_without_adapter_support_forces_a_merge():
    plan = choose_merge_strategy(
        cartridges_on_device=20, storage_free_bytes=64 * GB,
        base_bytes=1120 * MB, adapter_bytes=30 * MB,
        runtime_supports_adapters=False,
    )
    assert plan.strategy is MergeStrategy.MERGED


def test_sharing_does_not_help_when_even_the_base_will_not_fit():
    plan = choose_merge_strategy(
        cartridges_on_device=20, storage_free_bytes=500 * MB,
        base_bytes=1120 * MB, adapter_bytes=30 * MB,
    )
    assert plan.strategy is MergeStrategy.MERGED
    assert "does not help" in plan.reason


# =========================================================================== #
# The compilation recipe
# =========================================================================== #
def _plan(**over) -> CompilationPlan:
    base = dict(
        base=_pinned(), adapter=AdapterSpec(rank=16, alpha=32),
        merge=MergePlan(MergeStrategy.MERGED), deployment_quant=DEPLOY,
    )
    base.update(over)
    return CompilationPlan(**base)


def test_the_recipe_is_five_ordered_stages():
    stages = [s["stage"] for s in _plan().stages()]
    assert stages == ["acquire", "prepare", "train", "merge", "quantise", "package"]


def test_the_recipe_audits_clean():
    assert _plan().audit() == []


def test_the_nf4_stage_is_marked_as_not_shipping():
    prepare = next(s for s in _plan().stages() if s["stage"] == "prepare")
    assert "does not reach the device" in prepare["note"]


def test_the_report_carries_the_pinned_digest():
    report = compilation_report(_plan(), d_model=2048, n_layers=28)
    assert report["base_sha256"] == "a" * 12
    assert report["training_quant_ships"] is False
    assert report["adapter_mb"] > 0


def test_roles_cannot_be_swapped():
    with pytest.raises(WeightError, match="DEPLOYMENT role"):
        _plan(deployment_quant=NF4_TRAINING)


# =========================================================================== #
# §5, §10 — the format is a device choice, not just a bit width
# =========================================================================== #
def test_i8mm_unlocks_a_faster_format_at_the_same_bit_width():
    without = select_format(["neon", "dotprod"])
    with_i8mm = select_format(["neon", "dotprod", "i8mm"])
    assert with_i8mm.name == "q4_0_4_8"
    assert without.name != "q4_0_4_8"
    assert FORMATS[with_i8mm.name].bits_effective == FORMATS["q4_0_4_4"].bits_effective


def test_formats_needing_absent_instructions_are_excluded():
    choice = select_format(["neon"])
    assert any("i8mm" in r for r in choice.rejected)
    assert choice.name in FORMATS


def test_a_hypothesis_choice_says_it_is_a_hypothesis():
    """§13: the SIMD-to-format mapping is folklore until the probe measures it."""
    choice = select_format(["neon", "dotprod"])
    assert choice.evidence == "hypothesis"
    assert choice.measured is False
    assert "NOT measured" in choice.reason


def test_a_measured_sweep_overrides_the_hypothesis():
    choice = select_format(
        ["neon", "dotprod", "i8mm"],
        measured={"q4_k_m": 30.0, "q4_0_4_8": 12.0},   # folklore says 4_0_4_8 wins
    )
    assert choice.name == "q4_k_m"                      # measurement disagrees, and wins
    assert choice.evidence == "measured"
    assert choice.measured is True


def test_a_bit_ceiling_narrows_the_choice():
    narrow = select_format(["neon", "dotprod"], max_bits=5.0)
    assert FORMATS[narrow.name].bits_effective <= 5.0


def test_no_supported_format_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="no deployment format"):
        select_format(["neon"], accelerator="npu", max_bits=1.0)


def test_sweep_plan_lists_what_the_probe_should_measure():
    plan = sweep_plan(["neon", "dotprod", "i8mm"])
    assert "q4_0_4_8" in plan["formats"]
    assert plan["current_evidence"] == "hypothesis"


def test_measured_ranking_rejects_unknown_formats():
    with pytest.raises(ValueError, match="unknown formats"):
        measured_ranking({"telepathy_int2": 99.0})


def test_supported_formats_are_ordered_best_first():
    ordered = supported_formats(["neon", "dotprod", "i8mm"])
    hints = [f.speed_hint for f in ordered]
    assert hints == sorted(hints, reverse=True)
