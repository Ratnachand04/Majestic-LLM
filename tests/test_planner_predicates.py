"""Tests for the seven predicates, the catalogue and the cost model (§3, §4, §13, §14)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from modelrig.ir import AbstentionPolicy, DataRights, SpecIR
from modelrig.planner import costmodel
from modelrig.planner.catalog import (
    CatalogError,
    DeviceSpec,
    QuantiserSpec,
    default_catalog,
    load_devices,
)
from modelrig.planner.core import PlanCandidate
from modelrig.planner.licence_lattice import (
    BOTTOM,
    Restriction,
    atoms_of,
    compose,
    position_of,
    resolve,
)
from modelrig.planner.predicates import (
    ETA,
    HARD,
    MU_MIN,
    SOFT,
    P_cost,
    P_lat,
    P_lic,
    P_off,
    P_ram,
    P_seed,
    P_tok,
    derive_distil_mode,
    expected_cost,
    kv_bytes,
    latency,
    memory,
    ordered_predicates,
    seed_floor,
    weight_bytes,
)
from modelrig.licence import Licence
from modelrig.primitives import TaskPrimitive

CAT = default_catalog()


def _spec(**over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT,
        device_target="android_tablet_4gb",
        seed_data_count=200,
        data_rights=DataRights.CUSTOMER_OWNED,
        quality_gate=0.9,
    )
    base.update(over)
    return SpecIR(**base)


def _cand(base="Qwen/Qwen3-1.7B", teacher=None, **over) -> PlanCandidate:
    b = CAT.model(base)
    t = CAT.model(teacher) if teacher else None
    kw = dict(
        base=b, teacher=t, distil_mode=derive_distil_mode(b, t),
        peft_method="lora", rank=16, quantiser="q4_k_m", bit_width="int4",
        target="gguf", data_recipe={},
    )
    kw.update(over)
    return PlanCandidate(**kw)


# =========================================================================== #
# catalogue — malformed input is a startup failure
# =========================================================================== #
def test_catalogue_loads_and_validates():
    assert len(CAT.models) >= 6
    assert CAT.quantiser("q4_k_m").bits_effective > 4.0     # §14
    assert CAT.device("android_tablet_4gb").free_ram > 0


def test_unknown_lookups_raise_rather_than_return_none():
    for call in (lambda: CAT.model("nope"), lambda: CAT.quantiser("nope"),
                 lambda: CAT.device("nope")):
        with pytest.raises(CatalogError):
            call()


def test_effective_bits_cannot_be_below_nominal():
    """Metadata only ever adds; the reverse is a data-entry error."""
    with pytest.raises(CatalogError):
        QuantiserSpec("bad", bits_nominal=8.0, bits_effective=4.0)


def test_device_reserving_all_its_ram_is_rejected():
    with pytest.raises(CatalogError):
        DeviceSpec(name="x", free_ram=1000, app_reserve=1000)


def test_malformed_device_file_fails_at_load(tmp_path: Path):
    bad = tmp_path / "devices.yaml"
    bad.write_text("devices:\n  - name: x\n", encoding="utf-8")   # no ram_gb
    with pytest.raises(CatalogError, match="ram_gb"):
        load_devices(bad)


def test_throughput_without_reference_size_is_rejected(tmp_path: Path):
    """A token rate is meaningless without the model size it was measured at."""
    bad = tmp_path / "devices.yaml"
    bad.write_text(
        "devices:\n  - name: x\n    ram_gb: 4\n    prefill_tok_s: 50\n", encoding="utf-8"
    )
    with pytest.raises(CatalogError, match="reference_params_b"):
        load_devices(bad)


def test_missing_device_file_fails_loudly(tmp_path: Path):
    with pytest.raises(CatalogError):
        load_devices(tmp_path / "absent.yaml")


# =========================================================================== #
# P_ram (§3.1, §14)
# =========================================================================== #
def test_kv_cache_uses_the_exact_gqa_formula():
    """M_kv = 2 * L * H_kv * d_h * C * B * p_kv. §8: 235 MB for Qwen3-1.7B."""
    m = CAT.model("Qwen/Qwen3-1.7B")
    got = kv_bytes(m, context=2048, batch=1, elem_bytes=2)
    expected = 2 * 28 * 8 * 128 * 2048 * 1 * 2
    assert got == expected
    assert got / 1e6 == pytest.approx(235, abs=1)


def test_kv_uses_kv_heads_not_model_dimension():
    """§14: substituting d_model overestimates by the GQA group factor."""
    m = CAT.model("Qwen/Qwen3-1.7B")
    assert m.d_kv == 8 * 128
    d_model_wrong = 2048                                   # Qwen3-1.7B hidden size
    assert m.d_kv < d_model_wrong                          # GQA really does shrink it
    assert kv_bytes(m, 2048) < 2 * 28 * d_model_wrong * 2048 * 2


def test_kv_grows_linearly_in_context_and_batch():
    m = CAT.model("Qwen/Qwen3-1.7B")
    assert kv_bytes(m, 4096) == 2 * kv_bytes(m, 2048)
    assert kv_bytes(m, 2048, batch=4) == 4 * kv_bytes(m, 2048)


def test_weights_use_effective_bits_not_nominal():
    """§14: a '4-bit' k-quant is not 4 bits per parameter."""
    m = CAT.model("Qwen/Qwen3-1.7B")
    naive = m.params * 4 / 8
    actual = weight_bytes(m, CAT, "q4_k_m")
    assert actual > naive
    assert 1.15 <= actual / naive <= 1.40                  # the 15-25% the spec warns of


def test_moe_residency_counts_every_parameter():
    """Sparse activation is not sparse residency (A-07)."""
    moe = CAT.model("Qwen/Qwen3-30B-A3B")
    assert moe.active_params < moe.params
    assert weight_bytes(moe, CAT, "q4_k_m") == pytest.approx(
        moe.params * 5.2 / 8, rel=1e-6
    )


def test_memory_decomposes_exactly():
    d = CAT.device("android_tablet_4gb")
    b = memory(CAT.model("Qwen/Qwen3-1.7B"), CAT, quantiser="q4_k_m", context=2048, device=d)
    assert b.model_total == b.weights + b.kv_cache + b.embedder + b.runtime
    assert b.total == b.model_total + b.app_reserve


def test_p_ram_passes_and_fails_with_a_numeric_reason():
    assert P_ram(_spec(), _cand("Qwen/Qwen3-1.7B"), CAT).ok is True
    bad = P_ram(_spec(), _cand("Qwen/Qwen3-8B"), CAT)
    assert bad.ok is False
    assert "over by" in bad.reason and "MB" in bad.reason


def test_long_context_flips_a_fitting_plan():
    """The KV term is the variable nobody budgets for."""
    ok = _spec(io_schema={"context_length": 2048})
    huge = _spec(io_schema={"context_length": 32768})
    assert P_ram(ok, _cand(), CAT).ok is True
    assert P_ram(huge, _cand(), CAT).ok is False


# =========================================================================== #
# P_tok (§3.3)
# =========================================================================== #
def test_distil_mode_is_derived_from_tokenizer_identity():
    qwen_base = CAT.model("Qwen/Qwen3-1.7B")
    qwen_teacher = CAT.model("Qwen/Qwen3-32B")
    llama_base = CAT.model("meta-llama/Llama-3.2-1B-Instruct")
    assert derive_distil_mode(qwen_base, qwen_teacher) == "logit_kd"
    assert derive_distil_mode(llama_base, qwen_teacher) == "sequence_kd"
    assert derive_distil_mode(qwen_base, None) == "none"


def test_p_tok_refuses_cross_tokenizer_logit_kd():
    """KL requires one sample space; different vocabularies make it undefined."""
    bad = _cand("meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3-32B",
                distil_mode="logit_kd")
    result = P_tok(_spec(), bad, CAT)
    assert result.ok is False
    assert "one sample space" in result.reason
    assert "sequence-level" in result.remedy


def test_p_tok_allows_same_family_logit_kd():
    ok = _cand("Qwen/Qwen3-1.7B", "Qwen/Qwen3-32B", distil_mode="logit_kd")
    assert P_tok(_spec(), ok, CAT).ok is True


# =========================================================================== #
# P_seed (§3.4)
# =========================================================================== #
@pytest.mark.parametrize("primitive,expected", [
    (TaskPrimitive.CLASSIFY, 60),
    (TaskPrimitive.EXTRACT, 100),
    (TaskPrimitive.REWRITE, 120),
    (TaskPrimitive.ANSWER, 100),
    (TaskPrimitive.ROUTE, 150),
    (TaskPrimitive.TOOLCALL, 250),
])
def test_seed_floors_are_derived_not_asserted(primitive, expected):
    """n_min >= -ln(eta) / mu_min reproduces the spec's table exactly."""
    assert seed_floor(primitive) == expected
    assert seed_floor(primitive) == math.ceil(-math.log(ETA) / MU_MIN[primitive])


def test_seed_floor_is_monotone_in_rarity():
    """Rarer modes need more seeds. Nothing else should reorder them."""
    floors = [(MU_MIN[p], seed_floor(p)) for p in TaskPrimitive]
    for mu_a, n_a in floors:
        for mu_b, n_b in floors:
            if mu_a < mu_b:
                assert n_a >= n_b


def test_p_seed_reports_the_miss_probability():
    result = P_seed(_spec(seed_data_count=10), _cand(), CAT)
    assert result.ok is False
    assert result.detail["miss_probability"] > 0.5      # a 3% mode is likely absent
    assert "amplification cannot recover it" in result.reason


def test_p_seed_passes_above_the_floor():
    assert P_seed(_spec(seed_data_count=100), _cand(), CAT).ok is True


# =========================================================================== #
# P_lic — the join-semilattice (§13)
# =========================================================================== #
def test_join_is_associative():
    """Associativity means the solver needs no canonical component ordering."""
    a = position_of(Licence.APACHE_2_0)
    b = position_of(Licence.LLAMA_COMMUNITY)
    c = position_of(Licence.CC_BY_SA_4)
    assert ((a | b) | c).atoms == (a | (b | c)).atoms


def test_join_is_monotone_so_failures_cannot_be_rescued():
    """Adding a component can only tighten the result."""
    a = position_of(Licence.APACHE_2_0)
    b = position_of(Licence.CLOSED_API)
    assert a.atoms <= (a | b).atoms
    assert (a | b).fatal                     # absorbing, and it stays absorbing
    assert ((a | b) | position_of(Licence.MIT)).fatal


def test_restrictions_accumulate_and_never_cancel():
    joined = compose(position_of(Licence.APACHE_2_0), position_of(Licence.GEMMA_TERMS))
    assert Restriction.USE_POLICY in joined.atoms
    assert atoms_of(Licence.APACHE_2_0) <= joined.atoms


def test_bottom_is_the_identity():
    a = position_of(Licence.MIT)
    assert (a | BOTTOM).atoms == a.atoms


def test_closed_api_teacher_is_absorbing():
    """A-02's hard legal boundary, as a lattice fact."""
    out = resolve(Licence.APACHE_2_0, Licence.CLOSED_API, DataRights.CUSTOMER_OWNED)
    assert out.permitted is False
    assert "absorbing" in out.reason


def test_untrainable_data_refuses():
    out = resolve(Licence.APACHE_2_0, None, DataRights.THIRD_PARTY_NO_TRAINING)
    assert out.permitted is False
    assert Restriction.NO_TRAINING in out.position.atoms


def test_conditional_base_is_permitted_but_carries_obligations():
    out = resolve(Licence.LLAMA_COMMUNITY, None, DataRights.CUSTOMER_OWNED)
    assert out.permitted is True
    assert out.obligations


def test_audit_tier_refuses_terms_that_bind_downstream():
    ok = resolve(Licence.LLAMA_COMMUNITY, None, DataRights.CUSTOMER_OWNED)
    audited = resolve(Licence.LLAMA_COMMUNITY, None, DataRights.CUSTOMER_OWNED,
                      audit_tier=True)
    assert ok.permitted is True
    assert audited.permitted is False


def test_p_lic_wraps_the_lattice():
    assert P_lic(_spec(), _cand(), CAT).ok is True
    assert P_lic(_spec(data_rights=DataRights.UNKNOWN), _cand(), CAT).ok is False


# =========================================================================== #
# P_off (§3.6)
# =========================================================================== #
def test_offline_refuses_a_server_target():
    result = P_off(_spec(offline_required=True), _cand(target="vllm"), CAT)
    assert result.ok is False
    assert "vllm" in result.reason


def test_offline_refuses_cloud_escalation():
    spec = _spec(offline_required=True, abstention_policy=AbstentionPolicy.ESCALATE)
    assert P_off(spec, _cand(), CAT).ok is False


def test_data_residency_forces_a_local_teacher():
    """Even TRAINING-time generation must be local when data cannot leave."""
    spec = _spec(offline_required=True, io_schema={"may_leave_jurisdiction": False})
    remote = _cand(teacher="Qwen/Qwen3-32B", data_recipe={})
    local = _cand(teacher="Qwen/Qwen3-32B", data_recipe={"local_teacher": True})
    assert P_off(spec, remote, CAT).ok is False
    assert P_off(spec, local, CAT).ok is True


def test_offline_check_is_skipped_when_not_required():
    assert P_off(_spec(offline_required=False), _cand(target="vllm"), CAT).ok is True


# =========================================================================== #
# P_lat (§3.2)
# =========================================================================== #
def test_unmeasured_device_refuses_a_latency_commitment():
    """The refusal that matters most: a planner must not fabricate a promise."""
    spec = _spec(latency_budget_ms=60_000)      # generous enough to pass on merit
    result = P_lat(spec, _cand(), CAT)
    assert result.ok is False
    assert "unmeasured" in result.reason
    assert "device lab" in result.remedy


def test_estimates_are_allowed_only_when_explicitly_accepted():
    spec = _spec(latency_budget_ms=60_000,
                 io_schema={"accept_unmeasured_latency": True})
    assert P_lat(spec, _cand(), CAT).ok is True


def test_no_budget_means_no_latency_check():
    assert P_lat(_spec(latency_budget_ms=None), _cand(), CAT).ok is True


def test_prefill_dominates_on_long_inputs():
    """§8: a 500-token form is prefill-bound, not decode-bound."""
    d = CAT.device("android_tablet_4gb")
    est = latency(CAT.model("Qwen/Qwen3-1.7B"), CAT, quantiser="q4_k_m",
                  device=d, tokens_in=500, tokens_out=60)
    assert est.prefill_share > 0.6
    assert est.total_ms / 1000 == pytest.approx(13.8, abs=1.0)


def test_quantisation_buys_decode_latency_roughly_linearly():
    """Decode is bandwidth-bound, so it scales with weight bytes."""
    d = CAT.device("android_tablet_4gb")
    m = CAT.model("Qwen/Qwen3-1.7B")
    narrow = latency(m, CAT, quantiser="q4_k_m", device=d, tokens_in=1, tokens_out=100)
    wide = latency(m, CAT, quantiser="int8", device=d, tokens_in=1, tokens_out=100)
    ratio_bytes = weight_bytes(m, CAT, "int8") / weight_bytes(m, CAT, "q4_k_m")
    assert wide.decode_ms / narrow.decode_ms == pytest.approx(ratio_bytes, rel=0.05)


# =========================================================================== #
# P_cost (§3.7) — integer micro-USD
# =========================================================================== #
def test_money_is_integer_micro_usd():
    out = costmodel.estimate(CAT.model("Qwen/Qwen3-1.7B"), method="lora",
                             n_synthetic=10_000, n_train_tokens=3_200_000,
                             n_eval_examples=50)
    for value in (out.generation, out.training, out.evaluation, out.total):
        assert isinstance(value, int)


def test_lora_is_cheaper_than_full_fine_tuning():
    """zeta(m): no gradients through frozen weights."""
    m = CAT.model("Qwen/Qwen3-1.7B")
    lora = costmodel.training_cost(m.params, 1_000_000, "lora")
    full = costmodel.training_cost(m.params, 1_000_000, "full_ft")
    assert lora < full
    assert lora / full == pytest.approx(costmodel.ZETA["lora"], rel=0.01)


def test_moe_trains_on_active_parameters_only():
    """Cost follows the forward pass; RAM follows residency. They differ."""
    moe = CAT.model("Qwen/Qwen3-30B-A3B")
    out = costmodel.estimate(moe, method="lora", n_synthetic=0,
                             n_train_tokens=1_000_000, n_eval_examples=0)
    dense = costmodel.training_cost(moe.params, 1_000_000, "lora")
    assert out.training < dense


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        costmodel.training_cost(1_000_000, 1000, "telepathy")


def test_p_cost_names_the_dominant_term():
    spec = _spec(budget_ceiling_usd=0.01)
    result = P_cost(spec, _cand(data_recipe={"n_synthetic": 50_000,
                                             "n_train_tokens": 16_000_000,
                                             "n_eval_examples": 50}), CAT)
    assert result.ok is False
    assert result.detail["dominant"] in {"teacher generation", "training", "evaluation"}


# =========================================================================== #
# Ordering (§4) and the hard/soft partition (§16)
# =========================================================================== #
def test_predicates_are_ordered_cheapest_and_most_discriminating_first():
    order = ordered_predicates()
    keys = [p.selectivity for p in order]
    assert keys == sorted(keys)
    assert order[0].name == "P_tok"          # a lookup that halves the space


def test_the_derived_order_beats_the_naive_one():
    """Ordering cannot affect correctness, only cost — and it does affect cost."""
    from modelrig.planner.predicates import ALL_PREDICATES

    optimal = expected_cost(ordered_predicates())
    worst = expected_cost(sorted(ALL_PREDICATES, key=lambda p: -p.selectivity))
    assert optimal < worst


def test_hard_and_soft_partition_matches_the_spec():
    """§16: hard predicates are sound by construction and must be reported apart."""
    assert set(HARD) == {"P_ram", "P_tok", "P_lic", "P_off"}
    assert set(SOFT) == {"P_seed", "P_lat", "P_cost"}
    assert not set(HARD) & set(SOFT)
