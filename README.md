# Majestic LLM

**A model compiler, not a model.** You do not build one model flexible enough to
build any small model — you build a compiler. There is a specification IR, a
planner that runs verified passes over it, and a backend matrix that targets
devices. Flexibility is a property of the architecture, not of any one component.

Majestic turns a plain-language description into a **certified, device-sized
specialist**: it interviews the customer, type-checks the request against
physics, law and its own catalogue, refuses what is impossible *before spending
anything*, then builds, proves and packages what is possible. What ships is not a
model file — it is a **scorecard** proving a small specialist does one job as
well as a large generalist, plus the installable artefact for the device the
customer owns.

> **Status: implemented and runnable, offline on CPU.** The seven subsystems, the
> three-level IR with its three gates, the device-aware feasibility engine
> (including KV-cache accounting), the typed-DAG static analyser, the learned
> deferral router and the flywheel all run with tiny/mock defaults and no
> downloads. Heavy paths (real HF models, PEFT LoRA, sentence-transformers,
> FAISS, GGUF/ONNX export) sit behind lazy imports. Frontier pre-training,
> JEPA/world-models and image generation remain deliberate stubs.
> See [docs/research-gaps.md](docs/research-gaps.md) for what is genuinely
> unsolved versus merely unimplemented.

## Quickstart
```bash
uv venv --python 3.12 .venv && uv pip install -e '.[api,dev]'
python demo/run_compiler_demo.py     # the ten-act worked example, end to end
python -m cli.main validate          # model compatibility + architecture conformance
python -m cli.main budget            # the B-09 device RAM budget, computed
pytest                               # green offline: no network, no GPU
```
Full walkthrough: [docs/USAGE.md](docs/USAGE.md). Conformance evidence:
[docs/conformance.md](docs/conformance.md).

## The compiler
```
FORGE -> Spec IR -> [GATE 1] -> PLANNER -> Build Plan IR -> [GATE 2]
      -> DATA FACTORY -> TRAINER -> PROVING GROUND -> [GATE 3]
      -> CARTRIDGE -> REGISTRY -> FABRIC
```

| # | Subsystem | What it does |
|---|---|---|
| 1 | **FORGE** | slot-filling interview; asks four questions, not forty; refuses to guess an ambiguous slot |
| 2 | **PLANNER** | deterministic search over a typed catalogue; an LLM proposes only when no precedent matches, and a validator always disposes |
| 3 | **DATA FACTORY** | seed-anchored amplification behind blocking QA gates; refuses below the real-seed floor |
| 4 | **TRAINER** | LoRA/QLoRA by default; zero custom training loops |
| 5 | **PROVING GROUND** | seven axes, all must pass; re-run in full after quantisation |
| 6 | **REGISTRY** | content-addressed cartridges; base stored once, adapters are deltas |
| 7 | **FABRIC** | typed DAG, statically verified for offline closure, RAM and cost |

**Every gate is checked before any GPU is spent.** A refusal in eight seconds is
a better product than a failed build in ninety minutes:

```
admitted : False
stage    : gate1  (no planning, no data, no GPU)
  - seed data 5 is below the floor of 200 real examples for primitive 'generate'
  - quality bar unachievable: 'generate' needs at least 1.7B but
    android_lowend caps the base at 1.396B at 4-bit
```

## What is actually enforced
- **The largest base that fits at 4-bit** — never a smaller base at higher
  precision. Under a fixed RAM budget that is the k-bit scaling law's verdict.
- **Memory is weights + KV cache + runtime**, never weights alone. Free RAM must
  be ≥ 2× the model file, which is why a 4 GB phone tops out near 1.7B at 4-bit.
- **MoE is excluded from constrained devices.** Sparse activation is not sparse
  residency: 30B must be held to activate 3B.
- **Candidates are trained in parallel and one is chosen.** The largest base that
  fits RAM may still break the latency budget, and the budget is a contract.
- **Answer-flip rate**, not just accuracy. A quantised model can hold identical
  aggregate accuracy while changing a large share of individual answers. The
  quantiser is calibrated on the customer's own distribution and confidence is
  recalibrated afterwards.
- **A compiled grammar, not a hopeful prompt.** Schema violation is impossible
  rather than unlikely.
- **The licence chain is solved, not assumed.** Closed-API teacher outputs are
  refused outright; base, teacher, data rights and jurisdiction compose into an
  artefact licence or a refusal.
- **Offline means offline.** Escalation is architecturally disabled, and a graph
  that secretly needs the network is caught statically.
- **Untrusted content cannot reach a privileged tool.** Every retrieved page is
  input, not data.
- **The repairer acts only on external evidence.** Unaided self-reflection
  degrades performance, so it is forbidden by construction.
- **Knowledge lives in the index; behaviour lives in the adapter.** Reindexing is
  seconds and free; changing behaviour is a rebuild.

## Layout
```
majestic-llm/
├── README.md · pyproject.toml · requirements.txt   (light core deps; heavy deps optional/lazy)
├── configs/        default · models · routing · devices
├── docs/           architecture · USAGE · CONTRIBUTING · research-gaps
├── modelrig/                       ← the compiler
│   ├── ir.py                       Spec / Build Plan / Artefact IR (hash-addressed)
│   ├── gates.py                    Gate 1/2/3 — checked before any GPU
│   ├── primitives.py · catalogue.py · licence.py   (the closed set, parts, legal)
│   ├── forge.py · planner.py       front end + optimisation passes
│   ├── candidates.py               parallel builds → score both → pick one
│   ├── data_factory.py · proving_ground.py   amplification + the seven axes
│   ├── quantisation.py · grammar.py   calibration, flip rate, constrained decoding
│   ├── cartridge.py · registry.py  content-addressed artefacts + lineage
│   ├── feasibility.py              KV-cache-aware device predictor
│   ├── conformance.py              model compatibility + architecture self-check
│   ├── pipeline.py                 end to end, with the repair loop
│   └── buildspec · compiler · planes · classifier · datasets · training_hf
├── majestic/                       ← the compound runtime
│   ├── orchestrator.py · factory.py   request lifecycle + assembly
│   ├── fabric/     graph · analyser · executor   (typed DAG that runs)
│   ├── serving.py                  adapter pool, batching, device RAM budget
│   ├── agent/react.py              ReAct with a constrained tool-call schema
│   ├── router/     rule_router · deferral   (capability routing, learned deferral)
│   ├── flywheel.py                 corrections → KTO signal → rebuild-if-wins
│   ├── core · perception · experts · retrieval · verification · representation
├── api/ · cli/ · demo/ · tests/
```

## Honest bounds
Two load-bearing promises are **unvalidated** and the code says so everywhere it
touches them: on-device swap latency and thermals (GAP-10, every device profile
ships `measured: false`), and multi-generation flywheel quality (GAP-05). Only
the right-hand column of C-01 is defensible; everything else is available to a
competent competitor within months.

## Notes
- Lint targets `ruff`. Where an OS application-control policy blocks unsigned
  binaries, `python -m pyflakes majestic modelrig api cli demo tests` is a
  correctness fallback.
- Optional dependency groups: `.[api]`, `.[ml]`, `.[retrieval]`, `.[compress]`, `.[dev]`.
