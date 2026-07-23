# Contributing

## Build order (how the system was assembled)
The code was built in phases; this is also the recommended reading/extension order.

| Phase | Area | Key modules |
|------:|------|-------------|
| 0 | Config, mocks, orchestrator, demo | `majestic/config.py`, `core/mock.py`, `orchestrator.py`, `factory.py` |
| 1 | Reasoning core (real + mock) | `core/hf_core.py`, `core/reasoning_core.py` |
| 2 | Router + tools/experts | `router/rule_router.py`, `experts/tools.py`, `experts/specialist.py` |
| 3 | Retrieval + memory + RAG | `retrieval/store.py`, `retrieval/memory.py`, `retrieval/rag.py` |
| 4 | Verifier | `verification/verifier.py`, `verification/checks.py` |
| 5 | ModelRig factory | `modelrig/{buildspec,compiler,planes,classifier,factory,registry}.py` |
| 6 | Feasibility engine | `modelrig/feasibility.py` |
| 7 | Teacher loop + CLI + API | `modelrig/teacher_loop.py`, `cli/main.py`, `api/app.py` |
| 8 | Hardening + docs | this file, `docs/USAGE.md` |

## Principles
- **Offline-first.** Nothing in the default path may require a network or GPU.
  Import heavy dependencies **lazily**, inside the function that needs them, and
  provide a working fallback (mock core, hashing encoder, heuristic classifier).
- **Tests are green offline.** Gate anything needing extras/network/GPU behind
  `@pytest.mark.integration` (skipped by default via `pyproject.toml`).
- **The verifier gates every response**; **the eval plane gates every build.**
- **Config-driven.** No hardcoded paths, models, or secrets — read `configs/*.yaml`
  and `.env`. Seed anything stochastic (data splits, generation).
- **Honest bounds.** If something can't be implemented, leave a clear
  `NotImplementedError` + `TODO`; never fake outputs or weaken a test to pass.

## Extending the system
- **Add an expert/tool:** subclass `majestic.experts.base.Expert`, set `name` +
  `capabilities`, implement `run(**kwargs)` and (optionally) `estimate_cost`.
  Register it in `majestic/factory.py`. The `RuleRouter` will route to it by
  capability keywords.
- **Add a verification check:** subclass `majestic.verification.checks.Check`,
  implement `applies` + `run` → `CheckResult`, and add it to
  `PipelineVerifier._default_checks`.
- **Add a build plane / method:** implement a `modelrig.planes.Plane`, wire it in
  `DefaultCompiler` and `modelrig.factory._PLANES`. New training methods extend
  `TrainingMethod` and branch in `TrainingPlane`.
- **Add a device:** append to `configs/devices.yaml`
  (`ram_gb`, `accelerator`, `compute_gflops`, `battery_wh`).

## Dev workflow
```bash
uv pip install -e '.[api,dev]'
pytest                 # offline suite
ruff check .           # lint (see note below)
ruff format .
```
**Lint note:** some locked-down hosts block unsigned binaries via OS application
control, which prevents ruff's bundled executable from running. There,
`python -m pyflakes majestic modelrig api cli demo tests` is a correctness
fallback until you can run ruff in a normal environment/CI.

## Coding standards
Full type hints, docstrings on public classes/functions, explicit errors (no
silent failures), stdlib `logging` via `majestic.logging_utils.get_logger`.
