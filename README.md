# Majestic LLM

A compound ("universal") AI system: a multimodal reasoning core that plans and
routes across a pool of specialist models and tools, grounded by retrieval and
memory, with a verifier before every response. It also acts as the **teacher**
for the ModelRig factory, which distills small, device-optimized specialists.

> **Status: implemented and runnable.** The compound system (core, router,
> experts/tools, retrieval, verifier, orchestrator) and the ModelRig factory
> (BuildSpec → compile → data/train/eval/compress/export), the device-aware
> feasibility engine, and the teacher→factory loop all run **offline on CPU**
> with tiny/mock defaults. Heavy paths (real HF models, PEFT LoRA,
> sentence-transformers, FAISS, GGUF/ONNX export) are wired behind **lazy
> imports** and activate only when their optional dependencies are installed.
> Frontier pre-training, JEPA/world-models and image/video generation remain
> deliberate stubs (see `docs/architecture.md`).

## Quickstart
```bash
uv venv --python 3.12 .venv && uv pip install -e '.[api,dev]'   # or: pip install -e '.[api,dev]'
python demo/run_demo.py "What is Majestic LLM?"   # end-to-end, offline
pytest                                            # tests are green offline
python -m cli.main info                           # CLI
uvicorn api.app:app                               # local API -> POST /generate
```
See [docs/USAGE.md](docs/USAGE.md) for the full walkthrough (RAG, building a
specialist, feasibility checks, the API).

## Layout
```
majestic-llm/
├── README.md · pyproject.toml · requirements.txt   (light core deps; heavy deps optional/lazy)
├── .gitignore · .env.example · LICENSE · Makefile
├── configs/        default · models · routing · devices   (YAML settings)
├── docs/           architecture.md · USAGE.md · CONTRIBUTING.md
├── majestic/                       ← the compound "Majestic" system
│   ├── types.py · config.py · orchestrator.py · factory.py   (request lifecycle + assembly)
│   ├── core/          reasoning_core · mock · hf_core         (planning/reasoning LLM + mock)
│   ├── perception/    encoders.py                (hashing + optional sentence-transformers)
│   ├── router/        router · simple · rule_router          (cheapest-sufficient + escalation)
│   ├── experts/       echo · tools · specialist · pool        (code, web, classifier, stubs)
│   ├── retrieval/     store · memory · rag        (vector store, memory, RAG)
│   ├── verification/  verifier · checks           (schema, code, math, factuality)
│   └── representation/ bus.py                     (JEPA plug point)
├── modelrig/                       ← the specialist factory (teacher→factory loop)
│   ├── buildspec · compiler · planes · classifier · datasets  (data→train→eval→compress→export)
│   ├── feasibility.py                            (device-aware perf predictor + guarantees)
│   ├── registry · factory · teacher_loop · training_hf
├── api/            schemas · routes · app   (framework-agnostic handler + FastAPI)
├── cli/            main.py            (info · validate-spec · build · feasibility)
├── demo/           run_demo.py        (offline end-to-end demo)
└── tests/          conftest + 12 test modules (green offline; heavy paths gated)
```

## Architecture (one line)
encode -> plan/route -> run experts/tools (grounded by retrieval) -> verify ->
respond; hard cases + traffic feed the router and the ModelRig factory
(the teacher -> factory loop).

## Design principles (enforced)
- **Orchestrate, don't monolith** — capability = core + router + experts + tools + retrieval.
- **Runs on modest hardware** — tiny/mock defaults; model size is fully configurable.
- **Tests never need a GPU or network** — heavy/integration tests are marked and skipped by default.
- **Eval-driven factory** — no build passes unless it clears a held-out quality gate.
- **Verify before responding** — the verifier gates every response.
- **Config-driven & reproducible** — `configs/*.yaml` + `.env`, seeded splits.

## Notes
- Lint: this project targets `ruff`. On hosts where an OS application-control
  policy blocks unsigned binaries (e.g., ruff's bundled executable), use
  `python -m pyflakes` as a correctness fallback.
- Optional dependency groups: `.[api]`, `.[ml]`, `.[retrieval]`, `.[compress]`, `.[dev]`.
