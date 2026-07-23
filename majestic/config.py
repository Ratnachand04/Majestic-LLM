"""Configuration loading for Majestic LLM.

Settings are read from ``configs/*.yaml`` and may be overridden by environment
variables (prefixed ``MAJESTIC_``) and a local ``.env`` file. Nothing here
requires a network or GPU; the loader is pure-Python + PyYAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Environment variables that override YAML values. Mapping: env var -> dotted path.
_ENV_OVERRIDES: dict[str, str] = {
    "MAJESTIC_CORE_MODEL": "core.model",
    "MAJESTIC_ROUTING_POLICY": "routing.policy",
    "MAJESTIC_VECTOR_STORE_URL": "retrieval.vector_store",
    "MAJESTIC_MODEL_REGISTRY_PATH": "registry_path",
    "MAJESTIC_SEED": "seed",
}


@dataclass
class CoreSettings:
    """Reasoning-core settings (mirrors ``core:`` in default.yaml)."""

    model: str = "mock"
    mixture_of_experts: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.7


@dataclass
class RoutingSettings:
    """Router policy settings (mirrors ``routing:``)."""

    policy: str = "cheapest_sufficient"
    confidence_threshold: float = 0.7


@dataclass
class RetrievalSettings:
    """Retrieval / RAG settings (mirrors ``retrieval:``)."""

    vector_store: str = "memory"
    top_k: int = 5
    embedding_model: str = "hash"


@dataclass
class VerificationSettings:
    """Verifier settings (mirrors ``verification:``)."""

    enabled: bool = True


@dataclass
class Settings:
    """Top-level runtime settings, populated from ``configs/default.yaml``."""

    core: CoreSettings = field(default_factory=CoreSettings)
    routing: RoutingSettings = field(default_factory=RoutingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    verification: VerificationSettings = field(default_factory=VerificationSettings)
    seed: int = 0
    registry_path: str = "./registry"

    # --- convenience flat accessors (kept for backward compatibility) ---
    @property
    def core_model(self) -> str:
        return self.core.model

    @property
    def routing_policy(self) -> str:
        return self.routing.policy

    @property
    def confidence_threshold(self) -> float:
        return self.routing.confidence_threshold


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines from a .env file into ``os.environ``.

    Existing environment variables win; missing file is silently ignored. This
    is a minimal parser (no interpolation) to avoid an extra dependency.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _coerce(current: Any, value: str) -> Any:
    """Coerce a string env value to the type of the existing field value."""
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _set_dotted(obj: Any, dotted: str, value: str) -> None:
    """Set ``obj.a.b = value`` given a dotted path, coercing to the field type."""
    parts = dotted.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        return
    setattr(target, leaf, _coerce(getattr(target, leaf), value))


def _fill_dataclass(instance: Any, data: dict[str, Any]) -> None:
    """Recursively populate a dataclass instance from a nested dict."""
    if not isinstance(data, dict):
        return
    valid = {f.name: f for f in fields(instance)}
    for key, value in data.items():
        if key not in valid:
            continue  # ignore unknown / placeholder keys
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _fill_dataclass(current, value)
        elif value is not None and value != "TODO":
            setattr(instance, key, value)


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    """Load :class:`Settings` from a YAML file, applying env overrides.

    Precedence (low -> high): dataclass defaults, YAML file, ``.env`` file,
    process environment variables.
    """
    settings = Settings()
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        _fill_dataclass(settings, raw)

    _load_dotenv()
    for env_key, dotted in _ENV_OVERRIDES.items():
        if os.environ.get(env_key):
            _set_dotted(settings, dotted, os.environ[env_key])

    return settings
