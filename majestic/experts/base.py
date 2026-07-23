"""Base class for experts and tools in the pool."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Expert(ABC):
    """An expert model or tool the router can dispatch to."""
    name: str = "expert"
    capabilities: tuple[str, ...] = ()

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the expert on the given arguments."""
        raise NotImplementedError

    def estimate_cost(self, **kwargs: Any) -> float:
        """Rough cost/latency estimate used by the router. Override as needed."""
        return 1.0
