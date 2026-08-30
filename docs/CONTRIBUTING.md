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
| 9 | **Compiler architecture** | see below |

### Phase 9 — the compiler architecture
The system became a *model compiler* rather than a build script. New modules, in
dependency order:

| Layer | Modules |
|---|---|
| Foundations | `modelrig/primitives.py` (the eight), `licence.py` (GAP-08), `catalogue.py` |
| The contract | `modelrig/ir.py` — Spec / Build Plan / Artefact IR, hash-addressed (GAP-01) |
| Verification | `modelrig/gates.py` — Gates 1/2/3, checked before any GPU |
| Front end | `modelrig/forge/` — slot-filling interview, scored by the planner |
| Passes | `modelrig/planner/` — precedent warm start, validator, mutation |
| Build | `modelrig/data_factory.py`, `proving_ground.py` |
| Output | `modelrig/cartridge.py`, `registry.py` (content-addressed, lineage) |
| Runtime | `majestic/fabric/` (typed DAG + static analyser), `majestic/router/deferral.py`, `majestic/flywheel.py` |
| End to end | `modelrig/pipeline.py` |

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

### Phase 10 — full architecture conformance
Closed the remaining gaps between the diagrams and the code, and made the
architecture's rules executable:

| Area | Modules |
|---|---|
| Parallel candidates | `modelrig/candidates.py` (score both, pick one) |
| Quantisation | `modelrig/quantisation.py` (customer-distribution calibration, flip rate, recalibration, SpQR escalation) |
| Constrained decoding | `modelrig/grammar.py` (GBNF for object/enum/tool-call) |
| Serving | `majestic/serving.py` (adapter pool, batching, device budget) |
| Fabric runtime | `majestic/fabric/executor.py` (the DAG runs, not just analysed) |
| Context assembly | `majestic/retrieval/context.py` (chunk caps, boundary placement) |
| Constrained tools | `majestic/agent/react.py` |
| Self-check | `modelrig/conformance.py` + `cli validate` |

## Rules specific to the compiler
- **Never let a gate become a warning.** Gates 1/2/3 and the Data Factory's QA
  gates BLOCK. If something should merely warn, put it in `warnings`, not
  `reasons`.
- **Refuse rather than degrade.** Below the seed floor, off-catalogue, or
  licence-blocked: raise or return a refusal with a concrete reason. Never ship a
  model that will quietly get worse.
- **Label unmeasured numbers.** Anything predicted rather than measured must say
  so (`PerfEstimate.measured`, `DeviceProfile.measured`). See
  [research-gaps.md](research-gaps.md).
- **The repairer needs external evidence.** Never add a code path where a
  component "fixes itself" without a `FailureReport`.
- **An LLM proposes; a validator disposes.** No LLM output may reach a GPU
  without passing `gate2_plan_feasibility`.
- **Encode a rule where it is enforced, then test it.** A rule that lives only in
  a docstring drifts. If a diagram states a rule, add it to
  `modelrig/conformance.py` so `cli validate` fails when the code stops obeying
  it, and cite the diagram in the finding.
- **Never plan with a theoretical constant when a deployed one exists.** The
  int4 size constant is the observed on-disk figure, not the pure-4-bit one;
  the difference is ~18% of every device budget.

## Extending the system
- **Add a task primitive:** extend `TaskPrimitive` and `_SPECS` in
  `modelrig/primitives.py` with a seed floor, default metric and minimum base
  size. The set is *closed* on purpose — adding one is a taxonomy decision, so
  re-run `coverage_report()` against real traffic (GAP-06).
- **Add a base model:** append to `BASES` in `modelrig/catalogue.py` with its KV
  geometry (`n_layers`, `n_kv_heads`, `head_dim`) and licence. Keep the catalogue
  NARROW: each extra base fragments multi-adapter serving economics (A-04).
- **Add a verification axis:** add a `_method` to `ProvingGround`, append it in
  `evaluate()`, and map its failure to mutations in `_suggest`.
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
