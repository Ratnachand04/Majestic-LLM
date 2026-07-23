"""The expert & tool pool.

The router dispatches to a heterogeneous pool of systems. The *real*, runnable
experts live in dedicated modules and are re-exported here for convenience:

- :class:`~majestic.experts.echo.EchoExpert` — trivial passthrough.
- :class:`~majestic.experts.tools.WebTool` — search / fetch.
- :class:`~majestic.experts.tools.CodeExecTool` — sandboxed code execution.
- :class:`~majestic.experts.specialist.SpecialistExpert` — small local classifier.

``DiffusionExpert`` and ``WorldModelExpert`` remain optional research stubs: the
interfaces are wired so the router can dispatch to them, but they are out of
scope for this build (see the project plan) and raise ``NotImplementedError``.
"""
from __future__ import annotations

from typing import Any

from majestic.experts.base import Expert
from majestic.experts.echo import EchoExpert
from majestic.experts.specialist import SpecialistExpert
from majestic.experts.tools import CodeExecTool, WebTool

__all__ = [
    "EchoExpert",
    "WebTool",
    "CodeExecTool",
    "SpecialistExpert",
    "DiffusionExpert",
    "WorldModelExpert",
]


class DiffusionExpert(Expert):
    """Image / video / audio generation. Optional; out of scope for this build."""

    name = "diffusion"
    capabilities = ("image_gen", "video_gen", "audio_gen")

    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("DiffusionExpert is an optional stub (see plan).")


class WorldModelExpert(Expert):
    """JEPA / Dreamer-style latent planning. Optional; out of scope for this build."""

    name = "world_model"
    capabilities = ("plan", "simulate")

    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("WorldModelExpert is an optional stub (see plan).")
