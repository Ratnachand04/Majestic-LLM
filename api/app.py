"""FastAPI application exposing the compound system and the model studio.

Run locally::

    python run.py
    # or: uvicorn api.app:app --reload
    # then open http://localhost:8000/

FastAPI/uvicorn are optional (``pip install -e '.[api]'``); this module imports
them lazily so the rest of the package stays importable without them.

Two surfaces live here and they do different jobs:

``/generate``      the compound runtime — routes a request through the
                   Orchestrator and returns an answer with its trace.
``/api/*``         the model studio — describe a task, build a certified
                   specialist from labelled examples, serve it, and freeze it
                   into a standalone binary.

The frontend (``frontend/index.html``, vanilla HTML/CSS/JS, no build step) is
served at ``/`` so one process serves both the API and the page that calls it —
no separate dev server, no CORS configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from api.routes import generate as _generate
from api.schemas import GenerateRequest

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class GenerateBody(BaseModel):
    prompt: str
    modality: str = "text"
    options: dict[str, Any] = {}


class GenerateReply(BaseModel):
    output: Any
    trace: list[str] = []
    verified: bool = False


class Example(BaseModel):
    text: str
    label: str


class BuildBody(BaseModel):
    description: str
    examples: list[Example]
    quality_gate: float = 0.80
    offline: bool = True


class PredictBody(BaseModel):
    cartridge_id: str
    texts: list[str]


class PackageBody(BaseModel):
    cartridge_id: str


def create_app():
    """Build the FastAPI app. Imported lazily so tests can skip without fastapi."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    from api import studio

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

    # -- the model studio ------------------------------------------------- #
    @app.get("/api/sample-dataset")
    def sample_dataset() -> dict[str, Any]:
        rows = studio.sample_dataset()
        return {"examples": rows, "count": len(rows)}

    @app.post("/api/build")
    def build_route(body: BuildBody) -> dict[str, Any]:
        """Build a specialist. A refusal is a result, not an error status."""
        outcome = studio.build(
            body.description,
            [(e.text, e.label) for e in body.examples],
            quality_gate=body.quality_gate,
            offline=body.offline,
        )
        return outcome.as_dict()

    @app.get("/api/models")
    def models_route() -> dict[str, Any]:
        return {"models": studio.list_models()}

    @app.post("/api/predict")
    def predict_route(body: PredictBody) -> dict[str, Any]:
        return studio.predict(body.cartridge_id, body.texts)

    @app.post("/api/package")
    def package_route(body: PackageBody) -> dict[str, Any]:
        """Freeze a cartridge into a standalone .exe. Slow — tens of seconds."""
        return studio.package(body.cartridge_id)

    @app.get("/api/download/{cartridge_id}")
    def download_route(cartridge_id: str):
        path = studio.exe_path(cartridge_id)
        if path is None:
            raise HTTPException(
                status_code=404,
                detail=f"cartridge {cartridge_id[:12]} has not been packaged yet",
            )
        return FileResponse(
            path, media_type="application/octet-stream", filename=path.name
        )

    # -- the frontend: one static page, no build step, same-origin as the API - #
    index_path = FRONTEND_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(index_path)

    return app


# Module-level app for ``uvicorn api.app:app``.
app = create_app()
