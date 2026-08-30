"""The Trainer (Part 9): arithmetic, and the assertions that prevent silence.

The subsystem contributes no novelty by design — every component is published
and the implementation is borrowed. What it contributes to the *system* is that
it makes every decision auditable by making none of them.

So these tests cover two things: the arithmetic the Planner needs (which was
quoted throughout the series without ever being derived), and the pre-flight
assertions, which are the cheapest code here and prevent the failures that are
otherwise discovered only after a build evaluates cleanly.
"""
from __future__ import annotations

import pytest

from modelrig.preflight import (
    PreflightError,
    assert_chat_template,
    assert_finite,
    assert_loss_mask,
    assert_no_contamination,
    assert_tokenizer_match,
    naive_microbatch_mean,
    normalise_by_tokens,
    preflight,
)
from modelrig.trainer import (
    LORA_PLUS_LAMBDA,
    QWEN_1_7B,
    CandidateMode,
    PreferenceMethod,
    activation_bytes,
    adapter_bytes,
    alpha_over_rank,
    build_cost_breakdown,
    compression_ratio,
    effective_batch,
    full_finetune_bytes,
    kto_weights,
    learning_rates,
    memory_plan,
    preference_for,
    sweep_cost,
    trainable_parameters,
    training_cost,
    training_flops,
)

GB = 1_000_000_000
MB = 1_000_000


# =========================================================================== #
# §2 — where "~30 MB" actually comes from
# =========================================================================== #
def test_the_per_layer_fan_reproduces():
    """q + k + v + o + gate + up + down over the 1.7B geometry."""
    assert QWEN_1_7B.per_layer_fan() == 36_992


def test_the_adapter_size_table_reproduces():
    """Quoted for eight documents without derivation. Here is the derivation."""
    assert trainable_parameters(1) == pytest.approx(1.04e6, rel=0.01)
    for rank, params_m, size_mb in ((8, 8.3, 17), (16, 16.6, 33),
                                    (32, 33.1, 66), (64, 66.3, 133)):
        assert trainable_parameters(rank) / 1e6 == pytest.approx(params_m, abs=0.1)
        assert adapter_bytes(rank) / MB == pytest.approx(size_mb, abs=1)


def test_the_thirty_megabyte_figure_is_rank_sixteen():
    """The series-wide adapter size, finally attributable."""
    assert 30 <= adapter_bytes(16) / MB <= 35


def test_adapter_size_is_linear_in_rank():
    assert adapter_bytes(32) == 2 * adapter_bytes(16)


def test_the_compression_ratio_is_tiny():
    """r(d+k)/dk — the reason optimiser state collapses."""
    assert compression_ratio(16, 2048, 2048) < 0.02


def test_rank_zero_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        trainable_parameters(0)


# =========================================================================== #
# §5-§7 — memory is the binding constraint
# =========================================================================== #
def test_full_finetuning_does_not_fit_a_24gb_card():
    """16 bytes per parameter: bf16 weights + grads + fp32 master + Adam m, v."""
    assert full_finetune_bytes(1_700_000_000) / GB == pytest.approx(27.2, abs=0.1)
    assert full_finetune_bytes(1_700_000_000) > 24 * GB


def test_qlora_reduces_memory_by_an_order_of_magnitude():
    """The difference between "needs an A100" and "runs on a laptop GPU"."""
    plan = memory_plan(1_720_000_000, rank=16, batch=4)
    assert plan.frozen_base / GB == pytest.approx(0.97, abs=0.05)
    assert plan.adapter_and_optimiser / GB == pytest.approx(0.27, abs=0.02)
    assert 2.0 <= plan.total / GB <= 4.0            # §6's quoted band
    assert 7 <= plan.reduction <= 11                # §6's quoted reduction


def test_the_optimiser_term_collapses_because_adam_scales_with_trainables():
    """Adam states track TRAINABLE parameters, which are about 1% of the total.
    That is the whole trick."""
    plan = memory_plan(1_720_000_000, rank=16)
    assert plan.adapter_and_optimiser < plan.frozen_base / 3


def test_checkpointing_trades_compute_for_activation_memory():
    """Roughly 5x less memory for about 30% more compute — and Majestic's builds
    are compute-cheap and memory-constrained."""
    plain = activation_bytes(4, 2048, checkpointing=False)
    checkpointed = activation_bytes(4, 2048, checkpointing=True)
    assert plain / checkpointed == pytest.approx(5.3, abs=0.5)


def test_a_bigger_rank_costs_memory():
    small = memory_plan(1_720_000_000, rank=8)
    large = memory_plan(1_720_000_000, rank=64)
    assert large.total > small.total
    assert small.fits(8 * GB) and large.fits(8 * GB)


# =========================================================================== #
# §8 — the batch-size identity, and its trap
# =========================================================================== #
def test_gradient_accumulation_buys_batch_at_no_memory_cost():
    assert effective_batch(micro_batch=2, grad_accum=8) == 16
    assert effective_batch(2, 8, devices=4) == 64


def test_microbatch_averaging_overweights_short_sequences():
    """The silent, systematic bias. Averaging per microbatch and then across
    accumulation steps is NOT token-level averaging over the true batch — and
    Majestic's inputs are variable-length documents."""
    losses = [1.0, 3.0]
    tokens = [900, 100]                  # one long microbatch, one short

    naive = naive_microbatch_mean(losses)
    correct = normalise_by_tokens(losses, tokens)

    assert naive == pytest.approx(2.0)          # the short one counts equally
    assert correct == pytest.approx(1.2)        # ...it should count for a tenth
    assert correct < naive


def test_the_two_agree_when_lengths_are_equal():
    """Which is why the bug is invisible on fixed-length data."""
    assert normalise_by_tokens([1.0, 3.0], [500, 500]) == \
        pytest.approx(naive_microbatch_mean([1.0, 3.0]))


def test_mismatched_loss_and_token_counts_are_refused():
    with pytest.raises(ValueError, match="token count"):
        normalise_by_tokens([1.0, 2.0], [100])


# =========================================================================== #
# §9-§10 — the cost correction
# =========================================================================== #
def test_lora_skips_one_of_the_three_flop_terms():
    """Gradients w.r.t. the frozen base are never materialised: 4PN, not 6PN."""
    lora = training_flops(1_700_000_000, 1_000_000)
    full = training_flops(1_700_000_000, 1_000_000, full_finetune=True)
    assert full / lora == pytest.approx(1.5)


def test_an_extraction_build_trains_for_about_ten_cents():
    """§10: the series quoted "$5-40 for training". This is wrong at the low end
    by a factor of roughly fifty."""
    cost = training_cost(1_700_000_000, examples=2_100, mean_tokens=500, epochs=3)
    assert cost.tokens == pytest.approx(3.2e6, rel=0.05)
    assert cost.usd < 0.20
    assert cost.seconds < 300


def test_a_synthesis_heavy_build_costs_two_orders_more():
    cost = training_cost(4_000_000_000, examples=60_000, mean_tokens=500, epochs=3)
    assert cost.tokens == pytest.approx(90e6, rel=0.01)
    assert 4.0 <= cost.usd <= 10.0
    assert 3.0 <= cost.hours <= 5.0


def test_the_checkpointing_factor_explains_the_gap_to_the_spec():
    """§10's 170 s and 3.2 h omit the 1.3x checkpointing cost; adding it back
    reproduces them."""
    with_ckpt = training_cost(1_700_000_000, 2_100, 500, 3)
    without = training_cost(1_700_000_000, 2_100, 500, 3, checkpointing=False)
    assert without.seconds == pytest.approx(170, abs=15)
    assert with_ckpt.seconds / without.seconds == pytest.approx(1.3, abs=0.01)


def test_an_extraction_build_costs_under_a_dollar():
    """Compounding with Part 8's teacher-cost correction: an extraction build
    with labelled seeds moves from about $40 to under $1."""
    train = training_cost(1_700_000_000, 2_100, 500, 3)
    breakdown = build_cost_breakdown(data_usd=0.0, train_usd=train.usd)
    assert breakdown["under_one_dollar"] is True
    assert breakdown["total"] == pytest.approx(0.45, abs=0.1)


def test_a_synthesis_build_is_still_expensive():
    train = training_cost(4_000_000_000, 60_000, 500, 3)
    breakdown = build_cost_breakdown(data_usd=10.0, train_usd=train.usd, eval_usd=3.0)
    assert breakdown["under_one_dollar"] is False
    assert 20 <= breakdown["total"] <= 70


# =========================================================================== #
# §11-§12 — LoRA+, and hedge versus sweep
# =========================================================================== #
def test_a_and_b_do_not_share_a_learning_rate():
    """B starts at zero and must travel; A starts at scale and must not."""
    rates = learning_rates(2e-4)
    assert rates["lr_B"] == pytest.approx(rates["lr_A"] * LORA_PLUS_LAMBDA)
    assert LORA_PLUS_LAMBDA == 16.0


def test_only_the_alpha_over_rank_ratio_matters():
    """Which is why the plan should record the ratio: changing r then needs no
    learning-rate retuning."""
    assert alpha_over_rank(16, 16) == alpha_over_rank(32, 32) == 1.0
    assert alpha_over_rank(32, 16) == 2.0


def test_hedge_and_sweep_are_governed_by_opposite_results():
    """One flag was making two opposite decisions. Redundancy buys a second draw
    from one distribution; search buys the maximum over several."""
    assert CandidateMode.HEDGE.default_enabled is False
    assert CandidateMode.SWEEP.default_enabled is True
    assert "never expected cost" in CandidateMode.HEDGE.rationale
    assert "model selection" in CandidateMode.SWEEP.rationale


def test_the_legacy_flag_reads_as_hedge():
    """The conservative reading: it is the mode §15 argues against, so treating
    it as a sweep would silently enable something nobody asked for."""
    from modelrig.ir import SpecIR
    from modelrig.planner.core import candidate_mode
    from modelrig.primitives import TaskPrimitive

    def spec(io):
        return SpecIR(task_primitive=TaskPrimitive.EXTRACT, io_schema=io)

    assert candidate_mode(spec({})) is CandidateMode.NONE
    assert candidate_mode(spec({"allow_parallel_candidates": True})) is CandidateMode.HEDGE
    assert candidate_mode(spec({"candidate_mode": "sweep"})) is CandidateMode.SWEEP


def test_a_rank_sweep_is_nearly_free():
    """Which is the argument for doing one instead of guessing."""
    report = sweep_cost(1_700_000_000, examples=2_100)
    assert report["runs"] == 4
    assert report["usd_total"] < 1.0
    assert "E[max q_i] > E[q]" in report["why"]


# =========================================================================== #
# §16-§17 — the preference stage
# =========================================================================== #
def test_kto_is_the_flywheel_method_because_the_data_is_unpaired():
    """Field feedback is a thumbs-up on ONE output. Building DPO pairs from it
    means generating a counterfactual or discarding signal — both lossy."""
    assert PreferenceMethod.KTO.accepts_unpaired is True
    assert PreferenceMethod.DPO.accepts_unpaired is False
    assert preference_for(has_pairs=False, from_field_feedback=True) \
        is PreferenceMethod.KTO


def test_orpo_needs_no_reference_model():
    """A reference model doubles resident memory, which matters given §6."""
    assert PreferenceMethod.ORPO.needs_reference_model is False
    assert PreferenceMethod.DPO.needs_reference_model is True
    assert preference_for(has_pairs=True, from_field_feedback=False) \
        is PreferenceMethod.ORPO


def test_kto_weights_come_from_the_observed_ratio():
    """Field feedback skews negative — people report failures, not successes —
    so defaults tuned on balanced data over-weight the complaints."""
    skewed = kto_weights(desirable=20, undesirable=180)
    assert skewed["balanced"] is False
    assert skewed["desirable"] > skewed["undesirable"]     # upweight the rarer class

    balanced = kto_weights(desirable=100, undesirable=100)
    assert balanced["balanced"] is True


def test_no_feedback_means_no_preference_stage():
    assert preference_for(has_pairs=False, from_field_feedback=False) \
        is PreferenceMethod.NONE


# =========================================================================== #
# §18-§19 — the pre-flight
# =========================================================================== #
def test_contamination_raises_rather_than_reporting():
    """I-01. A verdict can be ignored; an exception cannot."""
    with pytest.raises(PreflightError, match="contamination"):
        assert_no_contamination(["a", "b"], ["b", "c"])
    assert_no_contamination(["a"], ["b"])       # clean: returns


def test_contamination_is_checked_at_the_source_level_too():
    """An augmented variant of a held-out document leaks it even though the
    bytes differ. Only provenance catches that."""
    with pytest.raises(PreflightError, match="SOURCE level"):
        assert_no_contamination(["x"], ["y"], train_sources=["doc1"],
                                holdout_sources=["doc1"])


def test_a_chat_template_mismatch_is_a_hard_assertion():
    """The most common silent failure in fine-tuning anywhere: training
    succeeds, loss falls, checkpoint saves, model is quietly wrong."""
    assert_chat_template("<|im_start|>{role}", "<|im_start|>{role}")
    with pytest.raises(PreflightError, match="chat template mismatch"):
        assert_chat_template("### {role}:", "<|im_start|>{role}")


def test_a_missing_template_is_also_refused():
    with pytest.raises(PreflightError, match="chat template missing"):
        assert_chat_template("", "<|im_start|>")


def test_a_tokenizer_mismatch_between_prep_and_training_is_caught():
    with pytest.raises(PreflightError, match="tokenizer mismatch"):
        assert_tokenizer_match("qwen-v1", "llama-v3")


def test_padding_inside_the_loss_mask_is_refused():
    """Otherwise the model learns to emit padding."""
    with pytest.raises(PreflightError, match="padding token"):
        assert_loss_mask([1, 1, 1, 1], pad_positions=[2, 3])
    assert_loss_mask([1, 1, 0, 0], pad_positions=[2, 3])


def test_divergence_fails_fast_rather_than_saving():
    with pytest.raises(PreflightError, match="diverged"):
        assert_finite(float("nan"), step=42)
    with pytest.raises(PreflightError, match="diverged"):
        assert_finite(float("inf"))
    assert_finite(0.7)


def test_the_whole_preflight_runs_every_check():
    report = preflight(
        train_hashes=["a"], holdout_hashes=["b"],
        train_sources=["d1"], holdout_sources=["d2"],
        dataset_template="<|im_start|>", base_template="<|im_start|>",
        prep_tokenizer="qwen", train_tokenizer="qwen",
    )
    assert report.ok is True
    assert set(report.checks) == {"contamination", "chat_template", "tokenizer"}


def test_a_non_strict_preflight_collects_instead_of_raising():
    report = preflight(train_hashes=["a"], holdout_hashes=["a"], strict=False)
    assert report.ok is False
    assert any("contamination" in f for f in report.failures)
