"""FastAPI application exposing the compound system.

Run locally::

    uvicorn api.app:app --reload
    # then: curl -s localhost:8000/generate -d '{"prompt":"hello"}' -H 'content-type: application/json'

FastAPI/uvicorn are optional (``pip install -e '.[api]'``); this module imports
them lazily so the rest of the package stays importable without them.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from api.routes import generate as _generate
from api.schemas import GenerateRequest


class GenerateBody(BaseModel):
    prompt: str
    modality: str = "text"
    options: dict[str, Any] = {}


class GenerateReply(BaseModel):
    output: Any
    trace: list[str] = []
    verified: bool = False


def create_app():
    """Build the FastAPI app. Imported lazily so tests can skip without fastapi."""
    from fastapi import FastAPI

    app = FastAPI(title="Majestic LLM", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateReply)
    def generate_route(body: GenerateBody) -> GenerateReply:
        result = _generate(
            GenerateRequest(prompt=body.prompt, modality=body.modality, options=body.options)
        )
        return GenerateReply(
            output=result.output, trace=result.trace, verified=result.verified
        )

    return app


# Module-level app for ``uvicorn api.app:app``.
app = create_app()
