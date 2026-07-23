"""Shared pytest fixtures. Everything here runs offline (no network / GPU)."""
from __future__ import annotations

import pytest

from majestic.config import Settings
from majestic.factory import build_orchestrator


@pytest.fixture
def settings() -> Settings:
    """Default settings with the offline mock core."""
    return Settings()


@pytest.fixture
def orchestrator(settings):
    """A fully wired offline orchestrator (mock core + echo expert)."""
    return build_orchestrator(settings)
