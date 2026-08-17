"""Tests for FORGE (B-04) and the PLANNER (B-05)."""
from __future__ import annotations

import pytest

from modelrig.feasibility import YamlDeviceProfiler
from modelrig.forge import Forge
from modelrig.ir import DataRights, SpecIR
from modelrig.planner import Planner
from modelrig.primitives import TaskPrimitive

PROFILER = YamlDeviceProfiler("configs/devices.yaml")

LAB_REQUEST = (
    "Read lab requisition forms into our system on an Android tablet. "
    "Must work when the internet dies. We have 200 real forms. "
    "Flag anything uncertain, do not guess. Hindi and English. 95 percent."
)


# --- FORGE --------------------------------------------------------------- #
def test_forge_extracts_slots_from_plain_language():
    state = Forge().parse(LAB_REQUEST)
    assert state.slots["task_primitive"].value is TaskPrimitive.EXTRACT
    assert state.slots["device_target"].value == "android_tablet_4gb"
    assert state.slots["seed_data_count"].value == 200
    assert state.slots["quality_gate"].value == pytest.approx(0.95)
    assert set(state.slots["languages"].value) == {"hi-IN", "en-IN"}


def test_forge_flags_the_offline_ambiguity():
    """'when the internet dies' is ambiguous — always offline, or just resilient?"""
    state = Forge().parse(LAB_REQUEST)
    slot = state.slots["offline_required"]
    assert slot.ambiguous is True
    assert slot.filled is False          # never silently defaulted
    assert "offline_required" in [s.name for s in state.unfilled()]


def test_forge_asks_only_unfilled_slots_ranked_by_gain():
    state = Forge().parse(LAB_REQUEST)
    questions = state.questions(limit=4)
    assert 0 < len(questions) <= 4       # four questions, not forty
    gains = [s.information_gain for s in state.unfilled()]
    assert gains == sorted(gains, reverse=True)


def test_forge_emits_a_hash_addressed_spec():
    forge = Forge()
    state = forge.answer(
        forge.parse(LAB_REQUEST),
        offline_required=True,
        data_rights=DataRights.CUSTOMER_OWNED,
    )
    spec = forge.to_spec(state)
    assert spec.task_primitive is TaskPrimitive.EXTRACT
    assert spec.offline_required is True
    assert len(spec.hash) == 32


def test_forge_refuses_to_guess_the_primitive():
    forge = Forge()
    with pytest.raises(ValueError, match="task_primitive"):
        forge.to_spec(forge.parse("please do the needful with our stuff"))


def test_forge_identical_descriptions_hash_identically():
    forge = Forge()
    a, _ = forge.interview(LAB_REQUEST, offline_required=True)
    b, _ = forge.interview(LAB_REQUEST, offline_required=True)
    assert a.hash == b.hash          # identical hash -> served from cache


# --- PLANNER ------------------------------------------------------------- #
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


def test_planner_admits_a_plan_deterministically():
    result = Planner(profiler=PROFILER).plan(_spec())
    assert result.admitted is True
    assert result.path == "deterministic"     # no LLM involved on the common path
    assert result.plan.base_ref
    assert "largest base that fits" in result.plan.provenance["rule"]


def test_spare_ram_buys_precision_only_once_nothing_larger_fits():
    """A-01's k-bit law spends memory on SIZE first, precision second.

    The law says 4-bit is the optimal accuracy-per-bit point: given spare memory,
    a bigger model beats a wider one. Model sizes are discrete, so once the next
    base up will not fit at 4-bit the residual genuinely cannot buy size — and
    only then should it buy precision.
    """
    from modelrig.planner import default_catalog, select_base
    from modelrig.planner.predicates import memory

    catalog = default_catalog()
    spec = _spec()                                   # 4 GB tablet
    chosen = select_base(spec, catalog)
    device = catalog.device(spec.device_target)

    # Nothing larger fits at 4-bit ...
    chain = catalog.bases(TaskPrimitive.EXTRACT)
    larger = [m for m in chain if m.params > chosen.params]
    for model in larger:
        m = memory(model, catalog, quantiser="q4_k_m", context=2048, device=device)
        assert m.total > device.free_ram, f"{model.ref} fits at 4-bit but was not chosen"

    # ... so widening the chosen base is the only remaining use for the residual.
    wide = memory(chosen, catalog, quantiser="int8", context=2048, device=device)
    assert wide.total <= device.free_ram


def test_planner_caps_the_base_at_the_device_tier():
    """A 4 GB tablet must not be handed an 8B base."""
    result = Planner(profiler=PROFILER).plan(_spec(device_target="android_tablet_4gb"))
    assert result.admitted
    from modelrig.catalogue import DEFAULT_CATALOGUE

    base = DEFAULT_CATALOGUE.base(result.plan.base_ref)
    assert base.params_b <= 2.0


def test_planner_reuses_precedent_after_a_pass():
    planner = Planner(profiler=PROFILER)
    spec = _spec()
    first = planner.plan(spec)
    planner.record_outcome(spec, first.plan, passed=True)
    second = planner.plan(spec)
    assert second.path == "precedent"
    assert second.plan.base_ref == first.plan.base_ref


def test_planner_mutates_an_infeasible_llm_proposal():
    """The LLM proposes; the validator disposes."""
    def bad_proposer(spec, catalogue):
        from modelrig.ir import BuildPlanIR

        return BuildPlanIR(spec_hash=spec.hash, base_ref="Qwen/Qwen3-8B",
                           bit_width="fp16", target="gguf", budget_usd=10.0)

    planner = Planner(profiler=PROFILER, proposer=bad_proposer)
    result = planner.plan(_spec(device_target="android_lowend"))
    assert result.path == "proposed"
    assert result.mutations, "an infeasible proposal must be mutated, not accepted"


def test_distil_mode_is_derived_from_tokenizer_identity_not_chosen():
    """P_tok is definitional, not a preference (§3.3).

    Qwen3 spans 0.6B to 32B on one tokenizer, so a 32B teacher CAN distil into a
    small Qwen student at logit level — and when it can, it should, because logit
    KD transfers the full probability mass. The planner derives that; it never
    accepts a mode as input.
    """
    from modelrig.planner import default_catalog
    from modelrig.planner.predicates import derive_distil_mode

    result = Planner(profiler=PROFILER).plan(_spec(seed_data_count=200))
    catalog = default_catalog()
    base = catalog.model(result.plan.base_ref)
    teacher = catalog.model(result.plan.teacher_ref) if result.plan.teacher_ref else None

    assert result.plan.distil_mode == derive_distil_mode(base, teacher)
    if teacher is not None:
        expected = "logit_kd" if base.tokenizer == teacher.tokenizer else "sequence_kd"
        assert result.plan.distil_mode == expected


def test_planner_records_outcomes_for_meta_learning():
    planner = Planner(profiler=PROFILER)
    spec = _spec()
    result = planner.plan(spec)
    planner.record_outcome(spec, result.plan, passed=True)
    assert len(planner.outcomes) == 1
    # Cold start: the learner is inert and gamma is near zero, so the prior
    # carries the prediction entirely (§5).
    assert planner.outcomes.active is False
    assert planner.outcomes.gamma < 0.05
    assert planner.outcomes.predict(0.7, (1.0,) * 6) == 0.7
