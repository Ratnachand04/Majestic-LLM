"""Tests for the API: the framework-agnostic handler and the FastAPI app."""
from __future__ import annotations

import pytest

from api.routes import generate
from api.schemas import GenerateRequest


def test_generate_handler_offline():
    resp = generate(GenerateRequest(prompt="hello world"))
    assert resp.output == "hello world"
    assert resp.verified is True
    assert any(t.startswith("plan") for t in resp.trace)


def test_generate_handler_bad_modality_defaults_to_text():
    resp = generate(GenerateRequest(prompt="hi", modality="nonsense"))
    assert resp.output == "hi"


def test_fastapi_app():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.app import create_app

    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}

    r = client.post("/generate", json={"prompt": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["output"] == "hello world"
    assert body["verified"] is True
