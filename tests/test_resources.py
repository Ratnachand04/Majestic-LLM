"""The seven resource dimensions (Part 4 §12-§15).

Every test here is arithmetic. No LLM, no device, no GPU — which is exactly why
§18 puts this module first: it is the layer that can be verified before anything
expensive exists, and the layer everything expensive depends on.
"""
from __future__ import annotations

import pytest

from modelrig import resources as R

QWEN_1_7B = 1_720_000_000
QWEN_L, QWEN_D_KV = 28, 1024          # n_kv_heads * head_dim under GQA


# =========================================================================== #
# 1 — Memory (§13.1)
# =========================================================================== #
def test_effective_bit_width_is_not_the_nominal_one():
    """beta_eff is 4.5-5.5 for a nominal 4-bit k-quant, not 4.0.

    Scales, zero points and unquantised layers are real bytes. Taking the
    nominal width under-counts by 15-25% — enough to admit a plan that OOMs.
    """
    nominal = R.weight_bytes(QWEN_1_7B, 4.0)
    effective = R.weight_bytes(QWEN_1_7B, 4.8)
    assert effective > nominal
    assert (effective - nominal) / nominal == pytest.approx(0.20, abs=0.01)


def test_kv_cache_uses_the_gqa_geometry_not_the_model_dimension():
    """d_kv = n_kv_heads * head_dim. Substituting d_model overestimates 4-8x."""
    correct = R.kv_bytes(QWEN_L, QWEN_D_KV, 4096)
    with_d_model = R.kv_bytes(QWEN_L, 2048, 4096)
    assert with_d_model == 2 * correct        # a spurious refusal, manufactured


def test_kv_cache_is_linear_in_context():
    assert R.kv_bytes(QWEN_L, QWEN_D_KV, 8192) == 2 * R.kv_bytes(QWEN_L, QWEN_D_KV, 4096)


def test_context_is_derived_by_inverting_the_memory_equation():
    """Never elicited: device and base determine it completely (§10)."""
    c = R.max_context(
        free_ram=2_600 * R.MB, params=QWEN_1_7B, bits_effective=4.8,
        n_layers=QWEN_L, d_kv=QWEN_D_KV, embedder=90 * R.MB, runtime=180 * R.MB,
    )
    assert c > 0
    m = R.memory(
        params=QWEN_1_7B, bits_effective=4.8, n_layers=QWEN_L, d_kv=QWEN_D_KV,
        context=c, embedder=90 * R.MB, runtime=180 * R.MB,
    )
    assert R.memory_fits(m, 2_600 * R.MB)
    # And it is the *maximum*: one more token does not fit.
    over = R.memory(
        params=QWEN_1_7B, bits_effective=4.8, n_layers=QWEN_L, d_kv=QWEN_D_KV,
        context=c + 1, embedder=90 * R.MB, runtime=180 * R.MB,
    )
    assert not R.memory_fits(over, 2_600 * R.MB)


def test_a_bigger_base_leaves_less_context():
    """The trade §14 insists is joint: size and context share one budget."""
    common = dict(free_ram=3_000 * R.MB, bits_effective=4.8, n_layers=QWEN_L,
                  d_kv=QWEN_D_KV, embedder=90 * R.MB, runtime=180 * R.MB)
    assert R.max_context(params=600_000_000, **common) > \
           R.max_context(params=QWEN_1_7B, **common)


def test_no_room_for_weights_means_no_context_at_all():
    assert R.max_context(
        free_ram=1_000 * R.MB, params=8_000_000_000, bits_effective=4.8,
        n_layers=36, d_kv=1024, embedder=90 * R.MB, runtime=180 * R.MB,
    ) == 0


# =========================================================================== #
# 2 — Storage (§13.2)
# =========================================================================== #
def test_the_index_can_exceed_the_model():
    """50k chunks at 768 dims in fp32 is ~150 MB — the forgotten term."""
    assert R.index_bytes(50_000) == pytest.approx(153.6 * R.MB, rel=1e-6)


def test_quantising_the_index_is_a_four_fold_storage_win():
    assert R.index_bytes(50_000) / R.index_bytes(50_000, quantised=True) == 4


def test_storage_and_memory_bind_independently():
    """A phone can hold an artefact it cannot load, and vice versa.

    Neither predicate implies the other, which is why storage is its own
    dimension rather than a footnote to memory.
    """
    s = R.storage(artefact=1_100 * R.MB, n_chunks=50_000)
    # Plenty of RAM, not enough disk.
    assert not R.storage_fits(s, free_storage=1_500 * R.MB)
    assert R.storage_fits(s, free_storage=4_000 * R.MB)


def test_the_safety_margin_is_applied_not_merely_documented():
    s = R.storage(artefact=800 * R.MB, grammar=0, runtime=0)
    assert R.storage_fits(s, free_storage=1_000 * R.MB) is True     # 800 <= 800
    assert R.storage_fits(s, free_storage=999 * R.MB) is False


def test_storage_names_its_dominant_term():
    assert R.storage(artefact=100 * R.MB, n_chunks=200_000).dominant == "index"
    assert R.storage(artefact=1_100 * R.MB, n_chunks=1_000).dominant == "artefact"


# =========================================================================== #
# 3 & 4 — Latency and compute (§13.3, §13.4)
# =========================================================================== #
def test_prefill_dominates_a_document_workload():
    """§13.3's correction, on §13.3's own numbers.

    A 1000-token requisition producing 80 tokens of JSON: prefill ~20 s against
    decode ~10 s. Costing decode alone understates the true figure by ~3x, which
    is precisely how a build passes Gate 2 and disappoints in the field.
    """
    lat = R.latency(
        params=QWEN_1_7B, weight_bytes_=1_100_000_000, tokens_in=1_000,
        tokens_out=80, prefill_tok_s=50.0, bw_eff=9.24e9, overhead_s=0.00216,
    )
    assert lat.prefill_s == pytest.approx(20.0, abs=0.1)
    assert lat.decode_s == pytest.approx(9.7, abs=0.5)
    assert lat.total_s == pytest.approx(29.7, abs=0.5)
    assert lat.prefill_dominates is True
    assert lat.prefill_share == pytest.approx(0.67, abs=0.02)


def test_the_crossover_ratio_lands_where_the_spec_says():
    """n_in/n_out > tok/s_prefill / tok/s_decode, about 4-6 in practice."""
    assert R.prefill_crossover(50.0, 16.0) == pytest.approx(3.125, abs=0.01)
    assert R.prefill_crossover(140.0, 26.0) == pytest.approx(5.4, abs=0.1)
    # 1000 in / 80 out is 12.5 — far past every crossover in the catalogue.
    assert 1_000 / 80 > R.prefill_crossover(140.0, 26.0)


def test_prefill_is_compute_bound_and_decode_is_bandwidth_bound():
    """The two terms respond to different hardware, which is why one probe of
    one of them cannot license a promise about the other."""
    base = dict(params=QWEN_1_7B, weight_bytes_=1_100_000_000, tokens_in=1_000,
                tokens_out=80, flops_eff=2e11, bw_eff=9.24e9)
    faster_compute = R.latency(**{**base, "flops_eff": 4e11})
    faster_memory = R.latency(**{**base, "bw_eff": 18.5e9})
    reference = R.latency(**base)
    assert faster_compute.prefill_s < reference.prefill_s
    assert faster_compute.decode_s == reference.decode_s
    assert faster_memory.decode_s < reference.decode_s
    assert faster_memory.prefill_s == reference.prefill_s


def test_effective_flops_are_measured_not_quoted():
    """F_eff = 2 P n_in / t_prefill. Vendor TOPS describe NPU int8 peak and are
    unreachable from a CPU runtime, often by 10-50x."""
    f = R.effective_flops(QWEN_1_7B, 1_000, 20.0)
    assert f == pytest.approx(1.72e11, rel=1e-3)
    # Round trip: the measured figure reproduces the time it was measured from.
    assert R.prefill_seconds(QWEN_1_7B, 1_000, f) == pytest.approx(20.0)


def test_shortening_the_input_is_the_derived_lever():
    """When prefill dominates, crop the input rather than drop a base tier.

    Prefill is linear in n_in and costs no quality on the task itself, so this
    is usually a better trade than a smaller model.
    """
    lat = R.latency(params=QWEN_1_7B, weight_bytes_=1_100_000_000, tokens_in=1_000,
                    tokens_out=80, prefill_tok_s=50.0, bw_eff=9.24e9)
    assert R.shorten_input_saving(lat, 400) == pytest.approx(12.0, abs=0.1)
    assert R.shorten_input_saving(lat, 1_000) == 0.0


def test_the_thermal_derate_is_applied_to_the_promise():
    lat = R.latency(params=QWEN_1_7B, weight_bytes_=1_100_000_000, tokens_in=100,
                    tokens_out=80, prefill_tok_s=50.0, bw_eff=9.24e9)
    assert lat.derated(0.6) == pytest.approx(lat.total_s / 0.6)
    with pytest.raises(ValueError):
        lat.derated(0.0)


def test_thermal_ratio_is_capped_at_one():
    """A device cannot get faster as it heats up; a ratio above 1 is noise."""
    assert R.thermal_ratio(6.0, 10.0) == pytest.approx(0.6)
    assert R.thermal_ratio(11.0, 10.0) == 1.0


# =========================================================================== #
# 6 — Energy (§13.6)
# =========================================================================== #
def test_energy_reproduces_the_front_desk_tablet_figure():
    """3.5 W over a 29 s request is ~100 J, so ~480 requests per charge."""
    e = R.energy(29.0)
    assert e.joules_per_request == pytest.approx(101.5, abs=0.5)
    assert e.requests_per_charge == pytest.approx(478, abs=2)
    # 200 forms a day is roughly 40% of a charge spent on inference.
    assert e.battery_share(200) == pytest.approx(0.42, abs=0.02)


def test_energy_refuses_a_volume_that_cannot_fit_in_a_charge():
    e = R.energy(29.0)
    assert R.energy_fits(e, daily_volume=200) is True
    assert R.energy_fits(e, daily_volume=400) is False   # 480 < 400 * 1.5


def test_a_faster_model_is_a_longer_battery():
    """Energy is downstream of latency, so the prefill correction propagates."""
    assert R.energy(10.0).requests_per_charge > R.energy(29.0).requests_per_charge


# =========================================================================== #
# 7 — Network (§13.7)
# =========================================================================== #
def test_offline_zeroes_recurring_traffic_by_construction():
    """Not a setting that could drift — a structural consequence of the mode."""
    n = R.network(artefact_bytes=1_100 * R.MB, offline=True,
                  escalation_rate=0.3, daily_requests=200)
    assert n.recurring_per_day == 0
    online = R.network(artefact_bytes=1_100 * R.MB, offline=False,
                       escalation_rate=0.3, daily_requests=200)
    assert online.recurring_per_day > 0


def test_a_delta_update_ships_the_adapter_not_the_base():
    """~30 MB against ~1.1 GB: a 97% cut, and the practical argument for
    keeping the adapter separable."""
    full, delta = 1_100 * R.MB, R.delta_update_bytes(30 * R.MB, base_unchanged=True)
    assert delta / full < 0.03
    assert R.delta_update_bytes(30 * R.MB, base_unchanged=False,
                                artefact_bytes=full) == full


# =========================================================================== #
# The joint view (§14) — resources are not independent
# =========================================================================== #
def _envelope(**over):
    args = dict(
        memory_=R.memory(params=QWEN_1_7B, bits_effective=4.8, n_layers=QWEN_L,
                         d_kv=QWEN_D_KV, context=4096, embedder=90 * R.MB,
                         runtime=180 * R.MB),
        storage_=R.storage(artefact=1_100 * R.MB, n_chunks=20_000),
        latency_=R.latency(params=QWEN_1_7B, weight_bytes_=1_100_000_000,
                           tokens_in=1_000, tokens_out=80, prefill_tok_s=50.0,
                           bw_eff=9.24e9),
        thermal=0.6, free_ram=2_600 * R.MB, free_storage=8_000 * R.MB,
        latency_budget_s=60.0, daily_volume=200,
    )
    args.update(over)
    return R.envelope(**args)


def test_the_envelope_names_the_binding_dimension():
    env = _envelope()
    assert env.binding in {"memory", "storage", "latency", "energy"}
    assert "prefill_share" in env.as_dict()


def test_a_full_disk_binds_even_with_ample_ram():
    env = _envelope(free_ram=8_000 * R.MB, free_storage=1_400 * R.MB)
    assert env.fits is False
    assert env.binding == "storage"
    assert any("storage" in v for v in env.violations)


def test_a_latency_violation_names_which_term_dominates():
    env = _envelope(latency_budget_s=10.0)
    assert env.fits is False
    assert any("prefill dominates" in v for v in env.violations)


def test_energy_can_bind_when_nothing_else_does():
    env = _envelope(free_ram=8_000 * R.MB, free_storage=64_000 * R.MB,
                    latency_budget_s=600.0, daily_volume=3_000)
    assert env.fits is False
    assert env.binding == "energy"
    assert any("requests per charge" in v for v in env.violations)


def test_a_comfortable_envelope_reports_no_violations():
    env = _envelope(free_ram=8_000 * R.MB, free_storage=64_000 * R.MB,
                    latency_budget_s=600.0, daily_volume=50)
    assert env.fits is True
    assert env.violations == []


def test_retrieving_more_evidence_makes_the_model_slower():
    """§14's coupling, made concrete: context caps chunks, chunks set n_in, n_in
    drives prefill. More retrieved evidence is not free."""
    few = _envelope(latency_=R.latency(
        params=QWEN_1_7B, weight_bytes_=1_100_000_000, tokens_in=300,
        tokens_out=80, prefill_tok_s=50.0, bw_eff=9.24e9))
    many = _envelope(latency_=R.latency(
        params=QWEN_1_7B, weight_bytes_=1_100_000_000, tokens_in=2_000,
        tokens_out=80, prefill_tok_s=50.0, bw_eff=9.24e9))
    assert many.latency.total_s > few.latency.total_s
    assert many.energy.joules_per_request > few.energy.joules_per_request


# =========================================================================== #
# Guards
# =========================================================================== #
@pytest.mark.parametrize("call", [
    lambda: R.weight_bytes(0, 4.8),
    lambda: R.weight_bytes(QWEN_1_7B, 0),
    lambda: R.kv_bytes(0, 1024, 4096),
    lambda: R.index_bytes(-1),
    lambda: R.prefill_seconds(QWEN_1_7B, 100, 0.0),
    lambda: R.decode_seconds(1_000, 10, 0.0),
    lambda: R.effective_flops(QWEN_1_7B, 100, 0.0),
    lambda: R.energy(0.0),
    lambda: R.thermal_ratio(1.0, 0.0),
])
def test_impossible_inputs_raise_rather_than_returning_a_plausible_number(call):
    with pytest.raises(ValueError):
        call()


def test_latency_needs_one_of_the_two_prefill_measurements():
    with pytest.raises(ValueError, match="flops_eff or prefill_tok_s"):
        R.latency(params=QWEN_1_7B, weight_bytes_=1_000, tokens_in=10,
                  tokens_out=10, bw_eff=9.24e9)
