"""End-to-end demo: run a request through the compound system (mocks by default).

    python demo/run_demo.py "your prompt here"

With the default config this is fully offline: mock core + echo expert +
hashing encoder + basic verifier. Point ``core.model`` at a small HF instruct
model (or set MAJESTIC_CORE_MODEL) to use a real reasoning core.
"""
from __future__ import annotations

import pathlib
import sys

# Allow running as a plain script (``python demo/run_demo.py``) by putting the
# repo root on the path before importing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from majestic.factory import build_orchestrator  # noqa: E402
from majestic.types import Request  # noqa: E402


KNOWLEDGE = [
    "Majestic LLM is a compound AI system: a reasoning core plans and routes "
    "across experts and tools, grounded by retrieval, with a verifier before "
    "every response.",
    "ModelRig is the specialist factory that distills small, device-optimized "
    "models from the Majestic core and gates every build on a held-out eval.",
]


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    prompt = " ".join(argv) if argv else "What is Majestic LLM?"

    # Seed a tiny knowledge base so retrieval grounding is visible in the demo.
    orch = build_orchestrator(knowledge=KNOWLEDGE)
    response = orch.handle(Request(content=prompt))

    print(f"prompt   : {prompt}")
    print(f"response : {response.content}")
    print(f"trace    : {' -> '.join(response.trace)}")
    print(f"verified : {response.verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
