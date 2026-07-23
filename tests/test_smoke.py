"""Phase 0 smoke tests: the compound system runs end to end, offline."""
from __future__ import annotations

import majestic
from majestic.types import Modality, Request, Response


def test_package_imports():
    assert majestic.__version__


def test_orchestrator_end_to_end(orchestrator):
    """A request flows encode -> plan -> route -> execute -> verify -> respond."""
    resp = orchestrator.handle(Request(content="hello world"))
    assert isinstance(resp, Response)
    assert resp.content == "hello world"  # echo expert round-trips the content
    assert resp.verified is True
    # trace records each pipeline stage.
    assert any(t.startswith("encode") for t in resp.trace)
    assert any(t.startswith("plan") for t in resp.trace)
    assert "run:echo" in resp.trace


def test_response_empty_content_fails_verification(orchestrator):
    resp = orchestrator.handle(Request(content=""))
    assert resp.verified is False


def test_modalities_available():
    assert Modality.TEXT.value == "text"
    assert Request(content="x").modality is Modality.TEXT
