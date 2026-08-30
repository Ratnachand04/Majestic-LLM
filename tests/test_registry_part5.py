"""The Registry (Part 5): cache privacy, recall, economics and the corpus.

Two things here are corrections rather than features, and they get the most
attention: §8's cache leak — which is a data-protection bug arising from a
plausible optimisation — and §13's missing recall mechanism, without which a
defective cartridge stays certified forever.
"""
from __future__ import annotations

import pytest

from modelrig.cachekey import (
    CACHE_EXCLUDED,
    CACHE_REQUIRED,
    data_is_public,
    h_cache,
    h_ident,
    lookup_allowed,
)
from modelrig.cartridge import Cartridge, Status
from modelrig.corpus import (
    FORBIDDEN_MITIGATIONS,
    PERMITTED_MITIGATIONS,
    CorpusError,
    Generation,
    SyntheticRecipe,
    append,
    apply_mitigation,
    check_monotone,
    effective_size,
    growth,
    storage_plan,
)
from modelrig.economics import (
    CacheMix,
    DedupReport,
    break_even_k,
    dedup_ceiling,
    dedup_ratio,
    diversity_tradeoff,
    expected_affected,
    expected_cost,
    expected_loss,
    gross_margin,
)
from modelrig.ir import SpecIR
from modelrig.licence import DataRights
from modelrig.primitives import TaskPrimitive
from modelrig.registry import CartridgeRegistry

MB = 1_000_000


def _spec(owner: str = "acme", seed: str = "s3://acme/forms",
          rights: DataRights = DataRights.CUSTOMER_OWNED, **over) -> SpecIR:
    base = dict(
        task_primitive=TaskPrimitive.EXTRACT, owner_id=owner, seed_data_ref=seed,
        data_rights=rights, io_schema={"patient_id": "str"}, seed_data_count=200,
    )
    base.update(over)
    return SpecIR(**base)


def _cartridge(adapter: str = "a1", base: str = "Qwen/Qwen3-1.7B") -> Cartridge:
    return Cartridge(
        base_ref=base, adapter_ref=adapter, adapter_bytes=30 * MB,
        model_card={"built": True}, eval_certificate={"passed": True},
        licence_chain={"permitted": True},
    )


# =========================================================================== #
# §7 — two hashes, and why they must differ
# =========================================================================== #
def test_the_identity_hash_separates_requests_the_cache_hash_unifies():
    """Two customers, identical requirements, different budgets and owners.
    Same artefact; different requests."""
    a = _spec()
    b = _spec(owner="beta", budget_ceiling_usd=999.0, notes="different wording")
    assert h_cache(a) == h_cache(b)
    assert h_ident(a) != h_ident(b)


def test_the_budget_is_excluded_on_purpose():
    """§7: two ceilings, one artefact."""
    assert "budget_ceiling_usd" in CACHE_EXCLUDED
    assert h_cache(_spec()) == h_cache(_spec(budget_ceiling_usd=5.0))


def test_requirements_that_change_the_artefact_change_the_key():
    for field, value in (
        ("task_primitive", TaskPrimitive.CLASSIFY),
        ("offline_required", True),
        ("quality_gate", 0.99),
        ("io_schema", {"totally": "different"}),
    ):
        assert h_cache(_spec()) != h_cache(_spec(**{field: value})), field


def test_a_probe_of_a_different_handset_does_not_force_a_rebuild():
    """Two units of the same phone predict the same plan. Keeping the serial
    number in the key would rebuild for every handset."""
    profile = {"ram_free_mb": 2600, "bw_eff_gbps": 9.24, "device_id": "unit-a",
               "measured_at": "2026-01-01", "probe_version": "1"}
    other = {**profile, "device_id": "unit-b", "measured_at": "2026-08-30"}
    assert h_cache(_spec(device_profile=profile)) == \
        h_cache(_spec(device_profile=other))


def test_different_measured_hardware_does_change_the_key():
    slow = {"ram_free_mb": 2600, "bw_eff_gbps": 4.0}
    fast = {"ram_free_mb": 2600, "bw_eff_gbps": 18.5}
    assert h_cache(_spec(device_profile=slow)) != h_cache(_spec(device_profile=fast))


# =========================================================================== #
# §8 — the correction. Two independent barriers.
# =========================================================================== #
def test_the_seed_reference_is_in_the_cache_key():
    """The invariant. Excluding it looks like a free hit-rate win and would
    serve one customer a model trained on another's confidential documents."""
    assert "seed_data_ref" in CACHE_REQUIRED
    assert h_cache(_spec(seed="s3://acme/forms")) != \
        h_cache(_spec(seed="s3://beta/forms"))


def test_identical_requirements_on_different_data_never_collide():
    """The first barrier, stated as the scenario it prevents."""
    acme = _spec(owner="acme", seed="s3://acme/patient-forms")
    beta = _spec(owner="beta", seed="s3://beta/patient-forms")
    assert h_cache(acme) != h_cache(beta)


def test_the_owner_check_refuses_a_cross_owner_hit_on_private_data():
    """The second barrier, redundant with the first and deliberately so."""
    assert lookup_allowed(requester="beta", owner="acme", hit_is_public=False) is False
    assert lookup_allowed(requester="acme", owner="acme", hit_is_public=False) is True
    assert lookup_allowed(requester="beta", owner="acme", hit_is_public=True) is True


def test_only_public_domain_data_is_shareable():
    """A licence to *use* data is not a licence to serve someone else a model
    trained on it."""
    assert data_is_public(_spec(rights=DataRights.PUBLIC_DOMAIN)) is True
    for rights in (DataRights.CUSTOMER_OWNED, DataRights.LICENSED_FOR_TRAINING,
                   DataRights.THIRD_PARTY_NO_TRAINING, DataRights.UNKNOWN):
        assert data_is_public(_spec(rights=rights)) is False, rights


def test_h_cache_refuses_to_build_a_key_missing_a_required_field():
    """A guard, not a comment. A schema change that drops one of these would
    otherwise silently stop distinguishing builds it must distinguish."""
    import modelrig.cachekey as ck

    original = ck.CACHE_EXCLUDED
    try:
        ck.CACHE_EXCLUDED = frozenset({*original, "seed_data_ref"})
        with pytest.raises(ValueError, match="missing required fields"):
            h_cache(_spec())
    finally:
        ck.CACHE_EXCLUDED = original


def test_the_registry_serves_within_an_owner(tmp_path):
    reg = CartridgeRegistry(tmp_path)
    spec = _spec()
    reg.admit(_cartridge(), spec=spec)
    assert reg.lookup(spec) is not None


def test_the_registry_never_serves_across_owners_on_private_data(tmp_path):
    """Both barriers exercised end to end. Even naming the *same* seed ref —
    which should be impossible — the owner check refuses."""
    reg = CartridgeRegistry(tmp_path)
    reg.admit(_cartridge(), spec=_spec(owner="acme", seed="s3://acme/forms"))

    assert reg.lookup(_spec(owner="beta", seed="s3://beta/forms")) is None   # key
    assert reg.lookup(_spec(owner="beta", seed="s3://acme/forms")) is None   # owner
    assert reg.stats().cross_owner_refusals == 1


def test_a_public_corpus_may_be_shared_across_owners(tmp_path):
    reg = CartridgeRegistry(tmp_path)
    shared = dict(seed="hf://squad", rights=DataRights.PUBLIC_DOMAIN)
    reg.admit(_cartridge(), spec=_spec(owner="acme", **shared))
    assert reg.lookup(_spec(owner="beta", **shared)) is not None


def test_an_entry_admitted_without_a_spec_is_never_served_from_cache(tmp_path):
    """The safe default: provenance nobody recorded cannot be shown shareable."""
    reg = CartridgeRegistry(tmp_path)
    reg.admit(_cartridge())
    assert reg.lookup(_spec()) is None


# =========================================================================== #
# §13 — the recall mechanism
# =========================================================================== #
def test_a_recalled_cartridge_is_certified_but_not_servable():
    """Certification records that it passed its gate. Status records that a
    defect was found afterwards. Both are true at once."""
    cart = _cartridge().recall("tokenizer defect in the base")
    assert cart.certified is True
    assert cart.servable is False
    assert cart.status is Status.RECALLED


def test_a_recall_must_state_its_reason():
    with pytest.raises(ValueError, match="must state the defect"):
        _cartridge().recall("   ")


def test_a_deprecated_cartridge_is_still_servable():
    """It is not the newest. That is not the same as being unsafe."""
    cart = _cartridge()
    cart.status = Status.DEPRECATED
    assert cart.servable is True
    assert Status.RECALLED.servable is False


def test_the_status_survives_a_round_trip(tmp_path):
    cart = _cartridge().recall("licence revoked upstream")
    restored = Cartridge.load(cart.save(tmp_path / "c.json"))
    assert restored.status is Status.RECALLED
    assert restored.recall_reason == "licence revoked upstream"


def test_a_fleet_recall_is_one_query_and_one_call(tmp_path):
    """This is what lineage is for. 'Which customers are affected?' must be a
    query, not an investigation."""
    reg = CartridgeRegistry(tmp_path)
    for i in range(3):
        reg.admit(_cartridge(adapter=f"a{i}"), spec=_spec(seed=f"s3://acme/{i}"))
    reg.admit(_cartridge(adapter="other", base="Qwen/Qwen3-0.6B"),
              spec=_spec(seed="s3://acme/other"))

    assert reg.blast_radius("Qwen/Qwen3-1.7B")["affected"] == 3
    affected = reg.recall_base("Qwen/Qwen3-1.7B", "tokenizer defect")
    assert len(affected) == 3
    assert len(reg.servable()) == 1          # the 0.6B cartridge is untouched


def test_a_recalled_cartridge_is_never_served_from_the_cache(tmp_path):
    """Without this the recall is cosmetic: the runtime keeps loading it."""
    reg = CartridgeRegistry(tmp_path)
    spec = _spec()
    cid = reg.admit(_cartridge(), spec=spec)
    assert reg.lookup(spec) is not None
    reg.recall(cid, "safety regression found in production")
    assert reg.lookup(spec) is None
    assert reg.by_spec_hash(spec.hash) is None


def test_blast_radius_reports_the_share_of_the_fleet(tmp_path):
    reg = CartridgeRegistry(tmp_path)
    reg.admit(_cartridge(adapter="a"), spec=_spec(seed="s3://a"))
    reg.admit(_cartridge(adapter="b", base="Qwen/Qwen3-0.6B"), spec=_spec(seed="s3://b"))
    assert reg.blast_radius("Qwen/Qwen3-1.7B")["share"] == pytest.approx(0.5)


# =========================================================================== #
# §2-§3 — deduplication
# =========================================================================== #
def test_the_dedup_table_reproduces():
    """Five of §2's six rows. The k=10 cell disagrees with the spec's own
    formula and is treated as a slip — see dedup_ratio's docstring."""
    for k, expected in ((1, 0.97), (2, 1.90), (100, 27.18), (500, 34.74)):
        assert dedup_ratio(k) == pytest.approx(expected, abs=0.01)


def test_the_ceiling_is_just_the_size_ratio():
    """No amount of scale exceeds it."""
    assert dedup_ceiling() == pytest.approx(1120 / 30, abs=0.01)
    assert dedup_ratio(10_000) < dedup_ceiling()
    assert dedup_ratio(1_000_000) == pytest.approx(dedup_ceiling(), rel=0.01)


def test_cas_is_worse_for_the_first_cartridge_on_a_base():
    """Not a bug. It stores base AND adapter where naive stores one merged
    model, so a demo with more bases than customers measures CAS as worse."""
    assert break_even_k() == pytest.approx(1.03, abs=0.01)
    assert dedup_ratio(1.0) < 1.0
    assert dedup_ratio(1.1) > 1.0


def test_dedup_is_governed_by_k_not_by_customer_count():
    """The design consequence: adding customers helps, adding bases hurts."""
    assert dedup_ratio(500 / 5) == dedup_ratio(1000 / 10)     # same k, double n
    assert dedup_ratio(500 / 5) > dedup_ratio(500 / 50)       # same n, more bases


def test_the_dedup_report_says_whether_it_is_paying():
    thin = DedupReport(cartridges=3, distinct_bases=6, base_mb=1120, adapter_mb=30)
    fat = DedupReport(cartridges=500, distinct_bases=5, base_mb=1120, adapter_mb=30)
    assert thin.paying is False
    assert fat.paying is True
    assert fat.headroom < thin.headroom


def test_an_adapter_as_large_as_its_base_never_dedups():
    with pytest.raises(ValueError, match="never dedups"):
        break_even_k(base_mb=100, adapter_mb=100)


# =========================================================================== #
# §5 — the cache hierarchy and margin
# =========================================================================== #
def test_the_margin_arithmetic_reproduces():
    """§5's worked figures: c_bar = $22.4 at 88.7%, against 79.9% uncached."""
    mix = CacheMix()
    assert expected_cost(mix, 40.0) == pytest.approx(22.4, abs=0.05)
    assert gross_margin(mix, 40.0, 199.0) == pytest.approx(0.887, abs=0.001)
    assert gross_margin(CacheMix.cold_start(), 40.0, 199.0) == \
        pytest.approx(0.799, abs=0.001)


def test_caching_is_worth_about_nine_points_of_margin():
    from modelrig.economics import margin_report

    assert margin_report(CacheMix())["points_from_caching"] == pytest.approx(8.8, abs=0.3)


def test_the_compose_tier_is_inert_at_cold_start():
    """A margin projection that assumes it from month one is wrong: the tier
    contributes nothing until the library holds tens of adapters."""
    cold = CacheMix.cold_start()
    assert cold.compose == 0.0
    assert cold.hit_rate == 0.0
    assert expected_cost(cold, 40.0) == pytest.approx(40.0)


def test_the_tiers_must_be_a_distribution():
    with pytest.raises(ValueError, match="sum to 1"):
        CacheMix(exact=0.5, compose=0.5, warm_start=0.5, miss=0.5)


# =========================================================================== #
# §12 — diversity is insurance, not fewer defects
# =========================================================================== #
def test_expected_affected_is_independent_of_base_count():
    """The intuitive argument for diversity is wrong on expectation: more bases
    means more defects, each affecting proportionally fewer cartridges."""
    assert expected_affected(500, 1, 0.05) == expected_affected(500, 20, 0.05)


def test_diversity_pays_only_when_loss_is_superlinear():
    concentrated = expected_loss(500, 1, 0.05, gamma=1.5)
    diverse = expected_loss(500, 10, 0.05, gamma=1.5)
    assert diverse < concentrated

    # At gamma = 1 the expression collapses and diversity buys nothing.
    assert expected_loss(500, 1, 0.05, gamma=1.0) == \
        pytest.approx(expected_loss(500, 10, 0.05, gamma=1.0))


def test_sublinear_loss_is_refused_rather_than_computed():
    with pytest.raises(ValueError, match="concentration preferable"):
        expected_loss(500, 4, 0.05, gamma=0.8)


def test_the_tradeoff_moves_in_both_directions_at_once():
    """Loss falls with more bases; dedup falls with them too. The optimum is
    finite, and the exchange rate is a business judgement."""
    rows = diversity_tradeoff(500, [1, 4, 16])
    assert rows[0]["expected_loss"] > rows[-1]["expected_loss"]
    assert rows[0]["dedup_ratio"] > rows[-1]["dedup_ratio"]
    assert rows[0]["stored_gb"] < rows[-1]["stored_gb"]


# =========================================================================== #
# §14-§15 — the append-only corpus
# =========================================================================== #
def test_the_corpus_may_only_grow():
    g0 = Generation(0, frozenset({"a", "b"}))
    g1 = append(g0, ["c"])
    assert check_monotone([g0, g1]) == []
    assert g1.size == 3


def test_a_shrinking_corpus_is_caught_and_named():
    """'The corpus shrank' is not actionable; the ids that went missing are."""
    problems = check_monotone([
        Generation(0, frozenset({"a", "b", "c"})),
        Generation(1, frozenset({"a"})),
    ])
    assert problems
    assert "dropped 2 record(s)" in problems[0]
    assert "I-03" in problems[0]


def test_growth_is_linear_and_permanent():
    """§15's figures: 200 requests/day at a 6% correction rate is 12 a day,
    about 17 MB a year."""
    g = growth(daily_volume=200, correction_rate=0.06, days=365)
    assert g["corrections_per_day"] == pytest.approx(12.0)
    assert g["mb_per_year"] == pytest.approx(17.5, abs=0.6)
    assert growth(200, 0.06, 730)["bytes"] > g["bytes"]      # never decreases


def test_permitted_mitigations_lose_no_records():
    for name in PERMITTED_MITIGATIONS:
        result = apply_mitigation(name, 100 * MB)
        assert result["records_lost"] == 0
        assert result["bytes_after"] <= 100 * MB


def test_forbidden_mitigations_are_refused_with_the_reason():
    """Each of these looks like a reasonable storage optimisation and each
    lets the corpus shrink, which voids the collapse guarantee."""
    for name in FORBIDDEN_MITIGATIONS:
        with pytest.raises(CorpusError, match="forbidden under I-03"):
            apply_mitigation(name, 100 * MB)


def test_the_real_to_synthetic_floor_is_reported():
    assert effective_size(real=200, synthetic=8_000, rho_min=0.10)["holds"] is False
    ok = effective_size(real=1_000, synthetic=8_000, rho_min=0.10)
    assert ok["holds"] is True
    assert ok["n_eff"] > 1_000


# =========================================================================== #
# §16 — synthetic data is a recipe
# =========================================================================== #
def _recipe(**over) -> SyntheticRecipe:
    base = dict(real_corpus_hash="abc123", recipe_name="backtranslate+evolve",
                teacher_hash="def456", rng_seed=42, n_generated=60_000,
                statistics={"mean_len": 128.0, "entropy": 4.2})
    base.update(over)
    return SyntheticRecipe(**base)


def test_the_recipe_replaces_the_corpus_almost_entirely():
    saving = _recipe().saving_against(200 * MB)
    assert saving["recipe_bytes"] < 1_000
    assert saving["saving"] > 0.999


def test_the_recipe_reference_is_stable_and_content_addressed():
    assert _recipe().ref == _recipe().ref
    assert _recipe().ref != _recipe(rng_seed=43).ref
    assert _recipe().ref != _recipe(teacher_hash="other").ref


def test_regeneration_is_accepted_on_equivalence_not_bit_equality():
    """Floating-point reduction order differs between GPU models, so demanding
    bit equality would make the recipe useless on any machine but the one that
    ran it."""
    recipe = _recipe()
    ok, drift = recipe.equivalent({"mean_len": 129.0, "entropy": 4.21})
    assert ok is True and drift == []

    ok, drift = recipe.equivalent({"mean_len": 160.0, "entropy": 4.2})
    assert ok is False
    assert "outside 2%" in drift[0]


def test_a_missing_statistic_is_drift_not_a_pass():
    ok, drift = _recipe().equivalent({"mean_len": 128.0})
    assert ok is False
    assert "missing from the regenerated set" in drift[0]


def test_the_storage_plan_keeps_real_data_and_drops_synthetic():
    plan = storage_plan(real_bytes=17 * MB, synthetic_bytes=200 * MB, recipe=_recipe())
    assert plan["stored_bytes"] < plan["naive_bytes"]
    assert plan["saving"] > 0.9
    assert "append-only" in plan["why"]
