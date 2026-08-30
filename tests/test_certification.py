"""Per-device certification (Part 3 §10-§11).

The claim under test: a certificate names one device and one artefact kind, and
absence is reported as absence. The throughput number is the part customers ask
about; the on-device eval subset is the part that decides whether a certificate
is issued at all.
"""
from __future__ import annotations

import pytest

from modelrig.cartridge import Cartridge
from modelrig.certification import (
    MIN_EVAL_SUBSET,
    ArtefactKind,
    CertificationError,
    CertificationLedger,
    DeviceCertificate,
    OnDeviceRun,
    VerificationSource,
    certify,
    unverified_devices,
    verify_on_device,
)
from modelrig.probe import DeviceProfile, ProfileSource, Tier
from modelrig.quantformat import target_for, target_formats_for

DEVICE = "sm-a536b-probe-8f3a"


def _profile(**over) -> DeviceProfile:
    base = dict(
        device_id=DEVICE, ram_total_mb=6000, ram_free_mb=4400, bw_eff_gbps=9.24,
        overhead_ms_per_token=2.16, thermal_derate_180s=0.68, probe_lo_mb=400,
        probe_hi_mb=1200, storage_free_mb=20_000, source=ProfileSource.PROBE,
    )
    base.update(over)
    return DeviceProfile(**base)


def _run(n: int = 24, drift: int = 0, **over) -> OnDeviceRun:
    """A clean on-device run; ``drift`` outputs differ from the workstation."""
    reference = tuple(f"answer-{i}" for i in range(n))
    outputs = tuple(
        (f"DRIFTED-{i}" if i < drift else f"answer-{i}") for i in range(n)
    )
    base = dict(
        device_id=DEVICE, artefact_kind=ArtefactKind.MERGED,
        tokens_per_s_burst=7.9, tokens_per_s_sustained=5.4, peak_ram_mb=1487,
        outputs=outputs, reference_outputs=reference, scores=(0.931,) * n,
        reference_score=0.94, power_draw_w=3.5, ran_at="2026-08-30T00:00:00+00:00",
    )
    base.update(over)
    return OnDeviceRun(**base)


# =========================================================================== #
# §11 — the record
# =========================================================================== #
def test_a_clean_run_reproduces_the_manifest_entry():
    outcome = verify_on_device(_run(), _profile(), soc="sm6375")
    assert outcome.certified is True
    cert = outcome.certificate
    assert cert.device_id == DEVICE
    assert cert.source is VerificationSource.CUSTOMER_DEVICE
    assert cert.tokens_per_s == pytest.approx(7.9)
    assert cert.tokens_per_s_sustained == pytest.approx(5.4)
    assert cert.peak_ram_mb == 1487
    assert cert.eval_subset_score == pytest.approx(0.931)
    assert cert.output_parity == pytest.approx(1.0)


def test_energy_is_power_times_time_not_the_specs_arithmetic():
    """§11's illustrative record pairs 5.4 tok/s with 41.2 J per 1k tokens.

    Those cannot both hold: 1000 tokens at 5.4 tok/s takes 185 s, so 41.2 J
    implies a 0.22 W package draw — about 16x below the 3.5 W Part 4 §13.6
    establishes. E = P*t is computed from the measured draw instead.
    """
    cert = verify_on_device(_run(), _profile()).certificate
    assert cert.energy_per_1k_tokens_j == pytest.approx(3.5 * 1000 / 5.4, abs=0.5)
    assert cert.energy_per_1k_tokens_j > 100      # not the spec's 41.2


def test_the_promise_uses_the_sustained_rate():
    cert = verify_on_device(_run(), _profile()).certificate
    assert cert.tokens_per_s_sustained < cert.tokens_per_s


def test_a_run_with_no_power_measurement_reports_no_energy():
    """Never a default: an unmeasured draw is absent, not assumed."""
    cert = verify_on_device(_run(power_draw_w=None), _profile()).certificate
    assert cert.energy_per_1k_tokens_j is None


# =========================================================================== #
# §11 — the eval subset is what closes the loop
# =========================================================================== #
def test_throughput_alone_does_not_certify_anything():
    """A cartridge that generates fast and answers differently is the worse
    outcome. Tokens per second is not evidence about outputs."""
    outcome = verify_on_device(
        _run(outputs=(), reference_outputs=(), scores=(), reference_score=None),
        _profile(),
    )
    assert outcome.certified is False
    assert any("parity" in r for r in outcome.refusals)
    assert any("below the floor" in r for r in outcome.refusals)


def test_the_subset_floor_is_enforced():
    below = verify_on_device(_run(n=MIN_EVAL_SUBSET - 1), _profile())
    at = verify_on_device(_run(n=MIN_EVAL_SUBSET), _profile())
    assert below.certified is False
    assert at.certified is True


def test_diverging_outputs_refuse_the_certificate():
    """The check §11 exists for: a different runtime on a different SoC can
    produce different answers from identical weights."""
    outcome = verify_on_device(_run(n=24, drift=6), _profile())
    assert outcome.certified is False
    assert any("not reproducing the artefact that was evaluated" in r
               for r in outcome.refusals)


def test_a_score_regression_in_transit_refuses():
    outcome = verify_on_device(_run(scores=(0.62,) * 24), _profile())
    assert outcome.certified is False
    assert any("degraded in transit, not in training" in r for r in outcome.refusals)


def test_exceeding_production_ram_refuses_even_when_outputs_match():
    outcome = verify_on_device(_run(peak_ram_mb=4200), _profile())
    assert outcome.certified is False
    assert any("available in production" in r for r in outcome.refusals)


def test_an_unmeasured_profile_cannot_certify_anything():
    outcome = verify_on_device(_run(), _profile(source=ProfileSource.INTERPOLATED))
    assert outcome.certified is False
    assert any("cannot be attributed to hardware anyone measured" in r
               for r in outcome.refusals)


def test_a_certificate_must_name_one_device():
    outcome = verify_on_device(_run(), _profile(device_id="some-other-unit"))
    assert outcome.certified is False
    assert any("must name one device" in r for r in outcome.refusals)


def test_a_severe_thermal_derate_is_recorded_rather_than_refused():
    """It is a measurement, and the promise uses the sustained figure anyway."""
    cert = verify_on_device(
        _run(tokens_per_s_burst=20.0, tokens_per_s_sustained=6.0), _profile()
    ).certificate
    assert cert is not None
    assert any("loses over half its throughput" in n for n in cert.notes)


def test_a_benchmark_that_speeds_up_under_load_is_rejected_outright():
    """Not a device that defies physics — a broken run. Fitting it would
    launder the error into a promise."""
    with pytest.raises(CertificationError, match="corrupted the run"):
        _run(tokens_per_s_burst=5.0, tokens_per_s_sustained=9.0)


def test_unpaired_outputs_cannot_be_compared():
    with pytest.raises(CertificationError, match="parity needs pairs"):
        _run(outputs=("a", "b"), reference_outputs=("a",))


# =========================================================================== #
# §11 — certification is per device
# =========================================================================== #
def test_a_device_never_seen_is_reported_as_unverified():
    """A ledger that answers 'yes' for a device it never saw is worse than no
    ledger at all."""
    ledger = CertificationLedger()
    certify(ledger, _run(), _profile())
    assert ledger.certified_for(DEVICE) is True
    assert ledger.certified_for("pixel-7a") is False
    assert "UNVERIFIED on pixel-7a" in ledger.status_for("pixel-7a")
    assert "never run on this device" in ledger.status_for("pixel-7a")


def test_the_ledger_lists_only_what_it_measured():
    ledger = CertificationLedger()
    certify(ledger, _run(), _profile())
    assert ledger.devices == [DEVICE]
    assert unverified_devices(ledger, [DEVICE, "pixel-7a", "moto-g"]) == \
        ["pixel-7a", "moto-g"]


def test_a_failed_verification_leaves_no_trace_in_the_ledger():
    ledger = CertificationLedger()
    outcome = certify(ledger, _run(drift=10), _profile())
    assert outcome.certified is False
    assert ledger.entries == []
    assert ledger.certified_for(DEVICE) is False


def test_re_verifying_supersedes_rather_than_accumulates():
    ledger = CertificationLedger()
    certify(ledger, _run(), _profile())
    certify(ledger, _run(tokens_per_s_sustained=4.1), _profile())
    assert len(ledger.entries) == 1
    assert ledger.for_device(DEVICE).tokens_per_s_sustained == pytest.approx(4.1)


def test_a_fleet_promise_is_made_against_the_slowest_certified_device():
    ledger = CertificationLedger()
    certify(ledger, _run(), _profile())
    certify(ledger, _run(device_id="moto-g", tokens_per_s_burst=4.0,
                         tokens_per_s_sustained=2.2),
            _profile(device_id="moto-g"))
    assert ledger.slowest().device_id == "moto-g"
    assert ledger.build_card()["worst_case_sustained_tok_s"] == pytest.approx(2.2)


# =========================================================================== #
# §13 — certification does not transfer between artefact kinds
# =========================================================================== #
def test_a_merged_certificate_does_not_cover_the_separate_artefact():
    """They are different artefacts with different numerics. Certify whichever
    one actually ships."""
    ledger = CertificationLedger()
    certify(ledger, _run(artefact_kind=ArtefactKind.MERGED), _profile())
    assert ledger.certified_for(DEVICE, ArtefactKind.MERGED) is True
    assert ledger.certified_for(DEVICE, ArtefactKind.SEPARATE) is False
    status = ledger.status_for(DEVICE, ArtefactKind.SEPARATE)
    assert "different numerics" in status
    assert "merged artefact was certified there" in status


def test_the_separate_artefact_certificate_says_what_it_does_not_cover():
    cert = verify_on_device(
        _run(artefact_kind=ArtefactKind.SEPARATE), _profile()
    ).certificate
    assert any("is not covered" in n for n in cert.notes)


# =========================================================================== #
# The ladder, and what a certificate licenses
# =========================================================================== #
def test_only_a_measured_source_may_promise():
    assert VerificationSource.CUSTOMER_DEVICE.may_promise is True
    assert VerificationSource.DEVICE_LAB.may_promise is True
    assert VerificationSource.EMULATED.may_promise is False
    assert VerificationSource.INTERPOLATED.may_promise is False
    assert VerificationSource.EMULATED.tier is Tier.EMULATED


def test_a_throughput_only_certificate_may_not_promise():
    """Loaded directly rather than issued: the ledger can hold a legacy row, and
    it must still refuse to speak for it."""
    ledger = CertificationLedger.from_list([{
        "device_id": DEVICE, "source": "customer_device", "artefact_kind": "merged",
        "tokens_per_s": 7.9, "tokens_per_s_sustained": 5.4, "peak_ram_mb": 1487,
        "verified_at": "2026-08-30T00:00:00+00:00",
    }])
    assert ledger.certified_for(DEVICE) is False
    assert ledger.may_promise_on(DEVICE) is False
    assert "THROUGHPUT ONLY" in ledger.status_for(DEVICE)


def test_the_ledger_round_trips(tmp_path):
    ledger = CertificationLedger()
    certify(ledger, _run(), _profile(), soc="sm6375")
    restored = CertificationLedger.load(ledger.save(tmp_path / "perf.json"))
    assert restored.to_list() == ledger.to_list()
    assert restored.certified_for(DEVICE) is True


def test_the_build_card_states_the_caveat():
    ledger = CertificationLedger()
    certify(ledger, _run(), _profile())
    card = ledger.build_card()
    assert card["certified_devices"] == [DEVICE]
    assert "has not been verified" in card["caveat"]


# =========================================================================== #
# The cartridge manifest
# =========================================================================== #
def _cartridge(**over) -> Cartridge:
    base = dict(
        base_ref="Qwen/Qwen3-1.7B", adapter_ref="deadbeef",
        model_card={"built": True}, eval_certificate={"passed": True},
        licence_chain={"permitted": True},
    )
    base.update(over)
    return Cartridge(**base)


def test_a_build_certificate_is_not_a_device_certificate():
    """Two questions with two answers, and the manifest keeps them apart."""
    cart = _cartridge()
    assert cart.certified is True                 # evaluated and licensed
    assert cart.certified_for(DEVICE) is False    # never run on this silicon
    assert "UNVERIFIED" in cart.device_status(DEVICE)


def test_the_manifest_carries_the_measured_performance_array():
    cart = _cartridge()
    certify(cart.measured_performance, _run(), _profile())
    assert cart.certified_for(DEVICE) is True
    assert "CERTIFIED" in cart.device_status(DEVICE)
    assert cart.to_dict()["measured_performance"][0]["device_id"] == DEVICE


def test_an_uncertified_build_is_never_certified_for_a_device():
    """No amount of on-device evidence substitutes for an eval and a licence."""
    cart = _cartridge(licence_chain={"permitted": False})
    certify(cart.measured_performance, _run(), _profile())
    assert cart.certified_for(DEVICE) is False
    assert "UNCERTIFIED" in cart.device_status(DEVICE)


def test_the_cartridge_round_trips_with_its_ledger(tmp_path):
    cart = _cartridge()
    certify(cart.measured_performance, _run(), _profile())
    restored = Cartridge.load(cart.save(tmp_path / "cartridge.json"))
    assert restored.certified_for(DEVICE) is True
    assert restored.id == cart.id      # device evidence is not part of identity


def test_certifying_a_device_does_not_change_the_cartridge_id():
    """Where it ran is evidence about the artefact, not part of what it is."""
    cart = _cartridge()
    before = cart.id
    certify(cart.measured_performance, _run(), _profile())
    assert cart.id == before


# =========================================================================== #
# §10 — the accelerator decides the container
# =========================================================================== #
def test_the_accelerator_selects_the_runtime_container():
    assert target_for("cpu") == "gguf"
    assert target_for("npu") == "executorch"
    assert target_for("ane") == "coreml"


def test_an_npu_is_never_offered_a_container_it_cannot_load():
    assert "gguf" not in target_formats_for("npu")


def test_offline_removes_the_server_only_containers():
    assert "vllm" in target_formats_for("gpu")
    assert "vllm" not in target_formats_for("gpu", offline=True)


def test_an_unmapped_accelerator_falls_back_conservatively():
    """Not a licence to invent a container: fall back to what everything loads."""
    assert target_formats_for("tpu-of-the-week") == target_formats_for("cpu")


def test_a_probed_accelerator_narrows_the_planner_grid():
    """§10: selecting the container from the device's NAME is the guess the
    probe exists to replace."""
    from modelrig.ir import DataRights, SpecIR
    from modelrig.ir import ProfileSource as SpecProfileSource
    from modelrig.planner import default_catalog
    from modelrig.planner.core import _grid
    from modelrig.primitives import TaskPrimitive

    catalog = default_catalog()
    base = catalog.model("Qwen/Qwen3-1.7B")

    def targets(**over):
        spec = SpecIR(
            task_primitive=TaskPrimitive.EXTRACT, device_target="android_tablet_4gb",
            seed_data_count=200, data_rights=DataRights.CUSTOMER_OWNED, **over,
        )
        return {t for *_rest, t in _grid(spec, base, catalog)}

    guessed = targets()
    probed = targets(
        device_profile=_profile(accelerator="npu").to_dict(),
        profile_source=SpecProfileSource.PROBE,
    )
    assert "gguf" in guessed                 # the device-name prior allows it
    assert probed == {"executorch"}          # the measurement does not


def test_an_assumed_profile_does_not_override_the_prior():
    """The coordinate stops being inferred only once someone measures it."""
    from modelrig.planner.core import _probed_accelerator
    from modelrig.ir import ProfileSource as SpecProfileSource, SpecIR
    from modelrig.primitives import TaskPrimitive

    spec = SpecIR(
        task_primitive=TaskPrimitive.EXTRACT,
        device_profile=_profile(accelerator="npu").to_dict(),
        profile_source=SpecProfileSource.ASSUMED,
    )
    assert _probed_accelerator(spec) is None


def test_certificates_are_ordered_deterministically():
    ledger = CertificationLedger()
    certify(ledger, _run(device_id="zz-last"), _profile(device_id="zz-last"))
    certify(ledger, _run(device_id="aa-first"), _profile(device_id="aa-first"))
    assert [e["device_id"] for e in ledger.to_list()] == ["aa-first", "zz-last"]


def test_a_certificate_survives_a_dict_round_trip():
    cert = verify_on_device(_run(), _profile(), soc="sm6375").certificate
    assert DeviceCertificate.from_dict(cert.to_dict()) == cert
