"""Framework-agnostic route handlers.

``generate`` runs a request through the compound system's Orchestrator. The
FastAPI wiring lives in :mod:`api.app`; this module holds the logic so it can be
tested without a running server.
"""
from __future__ import annotations

from typing import Optional

from api.schemas import GenerateRequest, GenerateResponse
from majestic.orchestrator import Orchestrator
from majestic.types import Modality, Request

# A lazily-built default orchestrator, shared across calls.
_DEFAULT: Optional[Orchestrator] = None


def _get_orchestrator() -> Orchestrator:
    global _DEFAULT
    if _DEFAULT is None:
        from majestic.factory import build_orchestrator

        _DEFAULT = build_orchestrator()
    return _DEFAULT


def generate(
    request: GenerateRequest, orchestrator: Optional[Orchestrator] = None
) -> GenerateResponse:
    """Handle a generation request via the Orchestrator."""
    orch = orchestrator or _get_orchestrator()
    try:
        modality = Modality(request.modality)
    except ValueError:
        modality = Modality.TEXT
    response = orch.handle(Request(content=request.prompt, modality=modality))
    return GenerateResponse(
        output=response.content,
        trace=response.trace,
        verified=response.verified,
    )
