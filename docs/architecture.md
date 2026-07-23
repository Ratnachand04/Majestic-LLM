# Architecture

This implements the design from the project plans:

- **Storyline & Approval Brief** — product framing.
- **ModelRig — Consolidated Plan & Architecture** — the specialist factory.
- **Universal Model — Plan & Architecture** — the compound "Majestic" system
  and the beyond-LLM palette (JEPA, diffusion, state-space models, world models).

## Request lifecycle (the compound system)
```
encode -> retrieve -> plan -> route -> execute -> synthesize -> verify -> respond
```
Implemented in `majestic/orchestrator.py`, assembled by `majestic/factory.py`.

## Mapping to code
| Concept | Code | Status |
|---|---|---|
| Reasoning core (planner) | `majestic/core/` (`mock`, `hf_core`) | mock offline; real HF lazy |
| Perception / shared bus | `majestic/perception/encoders.py`, `representation/bus.py` | hashing encoder; ST optional; bus interface |
| Capability router | `majestic/router/` (`rule_router`) | rule + confidence + cost + escalation |
| Expert & tool pool | `majestic/experts/` (`echo`, `tools`, `specialist`, `pool`) | web, code-exec, classifier real; diffusion/world-model stubs |
| Retrieval & memory | `majestic/retrieval/` (`store`, `memory`, `rag`) | in-memory cosine; FAISS optional |
| Verifier | `majestic/verification/` (`verifier`, `checks`) | schema, code, math, factuality |
| Orchestrator (lifecycle) | `majestic/orchestrator.py` | complete |
| BuildSpec + planes | `modelrig/{buildspec,planes,classifier,compiler}.py` | data→train→eval→compress→export |
| Registry | `modelrig/registry.py` | filesystem-backed |
| Device feasibility | `modelrig/feasibility.py` | heuristic predictor + verdict |
| Teacher → factory loop | `modelrig/teacher_loop.py` | distill + telemetry |
| Heavy training | `modelrig/training_hf.py` | real PEFT LoRA, lazy + integration-gated |
| Entry points | `cli/main.py`, `api/app.py` | CLI + FastAPI |

## Deliberate stubs (out of scope for this build)
Wired as interfaces, not implemented — see the plan:
- **Frontier pre-training** of a foundation model from scratch.
- **JEPA / world-model** training (`representation/bus.py`,
  `experts/pool.py:WorldModelExpert`).
- **Image/video/audio generation** (`experts/pool.py:DiffusionExpert`).
- **GGUF/CoreML/TFLite/ExecuTorch exporters** and a **learned** perf predictor
  (`modelrig/feasibility.py`) — heuristic versions ship; production exporters/ML
  predictor are TODOs.

## Offline-first strategy
The default path uses only a light dependency core (pyyaml, pydantic, numpy).
Every heavy capability (transformers, peft, sentence-transformers, faiss, onnx)
is imported lazily with a working fallback, so `pytest` is green with no network
or GPU and the system stays runnable on modest hardware.
