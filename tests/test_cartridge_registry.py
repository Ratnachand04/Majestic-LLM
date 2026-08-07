"""Tests for the cartridge and the content-addressed registry (B-08)."""
from __future__ import annotations

from pathlib import Path

import pytest

from modelrig.cartridge import (
    Cartridge,
    IOContract,
    RuntimePolicy,
    ToolBinding,
    build_model_card,
)
from modelrig.ir import BuildPlanIR, SpecIR
from modelrig.licence import Licence
from modelrig.primitives import TaskPrimitive
from modelrig.registry import CartridgeRegistry


def _cart(**over) -> Cartridge:
    base = dict(
        base_ref="Qwen/Qwen3-1.7B",
        adapter_ref="adapter-hash",
        adapter_bytes=30_000_000,
        model_card={"task_primitive": "extract"},
        eval_certificate={"score": 0.94, "passed": True},
        licence_chain={"permitted": True, "resolved_licence": "apache-2.0"},
        spec_hash="spec1",
        plan_hash="plan1",
    )
    base.update(over)
    return Cartridge(**base)


# --- the five slots ------------------------------------------------------- #
def test_cartridge_has_five_slots_and_content_addressed_identity():
    c = _cart()
    assert c.base_ref and c.adapter_ref                 # slots 1-2
    assert isinstance(c.io_contract, IOContract)        # slot 4
    assert isinstance(c.runtime_policy, RuntimePolicy)  # slot 5
    assert c.id == _cart().id                           # same inputs, same id
    assert _cart(spec_hash="other").id != c.id


def test_certified_requires_card_certificate_and_licence():
    assert _cart().certified is True
    assert _cart(model_card={}).certified is False
    assert _cart(licence_chain={"permitted": False}).certified is False


def test_requires_network_follows_tool_bindings():
    offline = _cart()
    online = _cart(tool_bindings=[ToolBinding("doctor-notify", requires_network=True)])
    assert offline.requires_network is False
    assert online.requires_network is True


def test_cartridge_file_roundtrip(tmp_path: Path):
    c = _cart(tool_bindings=[ToolBinding("db", scopes=("read",), privileged=True)])
    c.save(tmp_path / "c.json")
    loaded = Cartridge.load(tmp_path / "c.json")
    assert loaded.id == c.id
    assert loaded.tool_bindings[0].privileged is True


def test_model_card_is_generated_from_telemetry():
    spec = SpecIR(task_primitive=TaskPrimitive.EXTRACT, device_target="android_midrange")
    plan = BuildPlanIR(spec_hash=spec.hash, base_ref="Qwen/Qwen3-1.7B")
    card = build_model_card(spec, plan, {"n_test": 50})
    assert card["base_model"] == "Qwen/Qwen3-1.7B"
    assert card["known_limitations"]          # honest bounds are always attached


# --- the registry --------------------------------------------------------- #
def test_registry_refuses_uncertified_cartridges(tmp_path: Path):
    reg = CartridgeRegistry(tmp_path)
    with pytest.raises(ValueError, match="not certified"):
        reg.admit(_cart(model_card={}))


def test_registry_admits_and_reloads(tmp_path: Path):
    reg = CartridgeRegistry(tmp_path)
    cid = reg.admit(_cart())
    assert cid in reg.list()
    assert reg.get(cid).base_ref == "Qwen/Qwen3-1.7B"


def test_spec_hash_cache_hit(tmp_path: Path):
    reg = CartridgeRegistry(tmp_path)
    reg.admit(_cart())
    assert reg.by_spec_hash("spec1") is not None      # identical spec -> free
    assert reg.by_spec_hash("unseen") is None
    assert reg.stats().cache_hit_rate == 0.5


def test_base_stored_once_gives_the_dedup_win(tmp_path: Path):
    """500 customers on one base: ~16 GB stored, not 550 GB."""
    reg = CartridgeRegistry(tmp_path)
    for i in range(50):
        reg.admit(_cart(spec_hash=f"spec{i}", adapter_ref=f"a{i}"))
    stats = reg.stats()
    assert stats.cartridges == 50
    assert stats.distinct_bases == 1                 # base stored ONCE
    assert stats.dedup_ratio > 10


def test_lineage_names_every_cartridge_on_a_base(tmp_path: Path):
    """When a base is defective, the fleet-recall query must be exact."""
    reg = CartridgeRegistry(tmp_path)
    a = reg.admit(_cart(spec_hash="s1", adapter_ref="a1"))
    b = reg.admit(_cart(spec_hash="s2", adapter_ref="a2"))
    reg.admit(_cart(spec_hash="s3", adapter_ref="a3",
                    base_ref="Qwen/Qwen3-0.6B"))
    derived = reg.derived_from("Qwen/Qwen3-1.7B")
    assert set(derived) == {a, b}
    assert reg.lineage(a)["licence_chain"]["permitted"] is True


def test_registry_persists_across_instances(tmp_path: Path):
    cid = CartridgeRegistry(tmp_path).admit(_cart())
    assert cid in CartridgeRegistry(tmp_path).list()


def test_licence_enum_available():
    assert Licence.APACHE_2_0.value == "apache-2.0"
