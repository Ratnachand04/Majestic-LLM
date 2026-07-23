"""Tests for the YAML/env configuration loader."""
from __future__ import annotations

from pathlib import Path

from majestic.config import Settings, load_settings


def test_defaults():
    s = Settings()
    assert s.core.model == "mock"
    assert s.routing.confidence_threshold == 0.7
    assert s.core_model == "mock"  # backward-compat accessor


def test_loads_default_yaml():
    s = load_settings("configs/default.yaml")
    assert s.core.model == "mock"
    assert s.retrieval.top_k == 5
    assert s.verification.enabled is True


def test_env_override(monkeypatch):
    monkeypatch.setenv("MAJESTIC_ROUTING_POLICY", "confidence")
    monkeypatch.setenv("MAJESTIC_SEED", "42")
    s = load_settings("configs/default.yaml")
    assert s.routing.policy == "confidence"
    assert s.seed == 42


def test_missing_file_uses_defaults(tmp_path: Path):
    s = load_settings(tmp_path / "does_not_exist.yaml")
    assert s.core.model == "mock"
