"""A trivial expert used for the offline pipeline and smoke tests."""
from __future__ import annotations

from typing import Any

from majestic.experts.base import Expert


class EchoExpert(Expert):
    """Returns its ``text`` argument unchanged. The simplest possible expert."""

    name = "echo"
    capabilities = ("echo",)

    def run(self, **kwargs: Any) -> Any:
        return kwargs.get("text", "")

    def estimate_cost(self, **kwargs: Any) -> float:
        return 0.0
