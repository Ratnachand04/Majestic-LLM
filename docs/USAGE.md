# Usage

Everything below runs **offline on CPU** with the default tiny/mock components.
Optional heavy features activate only when you install their extras.

## 1. Environment
```bash
# with uv (preferred)
uv venv --python 3.12 .venv
uv pip install -e '.[api,dev]'

# or with pip
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e '.[api,dev]'
```
Optional extras: `.[ml]` (torch/transformers/peft/trl), `.[retrieval]`
(sentence-transformers/faiss), `.[compress]` (bitsandbytes/gptq/onnx).

Configuration is read from `configs/default.yaml`, then overridden by a local
`.env` and by `MAJESTIC_*` environment variables (copy `.env.example` → `.env`).

## 2. Run the compound system (demo)
```bash
python demo/run_demo.py "What is Majestic LLM?"
```
Output shows the answer, the pipeline `trace`
(`encode -> retrieve -> plan -> route -> run -> ...`) and `verified`.

Programmatically:
```python
from majestic.factory import build_orchestrator
from majestic.types import Request

orch = build_orchestrator(knowledge=["Paris is the capital of France."])
resp = orch.handle(Request(content="What is the capital of France?"))
print(resp.content, resp.trace, resp.verified)
```

### Use a real reasoning core
Set `core.model` in `configs/default.yaml` (or `MAJESTIC_CORE_MODEL`) to a small
open instruct model id and install `.[ml]`. If loading fails the system falls
back to the mock core, so it always stays runnable.

## 3. The API
```bash
uvicorn api.app:app --reload
curl -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt":"hello world"}'
```
`GET /health` returns `{"status":"ok"}`.

## 4. Compile a specification into a certified cartridge

This is the main flow: plain language in, a **scorecard plus an installable
artefact** out. Every stage is gated before it spends anything.

```bash
# See the ten-act worked example end to end, offline:
python demo/run_compiler_demo.py

# 1. Interview: plain language -> a typed, hash-addressed Spec IR
python -m cli.main forge "Classify support tickets by sentiment on an android \
phone. Must work offline. We have 120 real tickets." \
    --offline --seed-count 120 --out spec.json

# 2. Compile: Gate 1 -> planner (Gate 2) -> data -> train -> proving (Gate 3)
python -m cli.main compile --spec spec.json --data tickets.jsonl

# 3. What primitives are supported at all?
python -m cli.main primitives
```

`forge` prints the slots it filled by itself and the few questions still worth
asking, ranked by information gain. It refuses to guess an ambiguous slot —
"works offline" meaning *always* versus *survives a dropout* is asked, not
assumed.

`compile` prints each gate, the plan chosen under the device cap, the seven-axis
scorecard, the answer-flip rate against FP16, and the cartridge id. When a build
is impossible it **refuses with reasons and spends nothing**:

```
admitted  : False
stage     : gate1  (no planning, no data, no GPU)
  - seed data 5 is below the floor of 200 real examples for primitive 'generate'
  - quality bar unachievable: 'generate' needs at least 1.7B but
    android_lowend caps the base at 1.396B at 4-bit
```

Programmatically:
```python
from modelrig.forge import Forge
from modelrig.ir import DataRights
from modelrig.pipeline import MajesticCompiler

forge = Forge()
state = forge.parse("Classify tickets by sentiment on an android phone")
state = forge.answer(state, data_rights=DataRights.CUSTOMER_OWNED,
                     seed_data_count=120, offline_required=True)
spec = forge.to_spec(state)

result = MajesticCompiler().compile(spec, corpus)
print(result.admitted, result.refusal)
for axis in result.scorecard.axes:
    print(axis.name, axis.score, axis.passed)
```

## 4b. Verify a Fabric graph before it runs

No agent framework can statically prove a graph runs without a network. This one
can, because Fabric is a typed DAG:

```bash
python -m cli.main verify-graph graph.json --offline
```
```json
{"name": "flow",
 "nodes": [{"name": "extract", "kind": "cartridge", "ram_mb": 30},
           {"name": "notify", "kind": "tool", "requires_network": true}],
 "edges": [["extract", "notify"]]}
```
It reports broken offline closure, RAM and cost bounds, and any path where
untrusted retrieved content can reach a privileged tool — then suggests
splitting into an offline core and an online tail.

## 5. Build a device specialist (ModelRig)
Distill a capability into a tiny, quantized, device-checked model:
```bash
python -m cli.main build --capability sentiment --device android_midrange
```
This curates data → fits a classifier → **evaluates on a held-out split (the
gate)** → quantizes (int8/packed-int4) → exports an artifact, and registers it.

From a spec file:
```bash
python -m cli.main validate-spec my_spec.json
python -m cli.main build --spec my_spec.json
```
`my_spec.json`:
```json
{"task": "sentiment", "base_model": "centroid", "method": "centroid",
 "quantization": "int8", "target_score": 0.7, "dataset": "builtin:sentiment"}
```
Methods: `centroid`/`none_rag` run offline; `lora`/`qlora`/`distill` need `.[ml]`.

Programmatically:
```python
from modelrig.factory import Factory
from modelrig.buildspec import BuildSpec, TrainingMethod

result = Factory(base_path="./registry").build(
    BuildSpec(task="sentiment", base_model="centroid",
              method=TrainingMethod.CENTROID, quantization="int8")
)
print(result.success, result.eval_report, result.artifact_path)
```

## 5b. Validate compatibility and conformance

```bash
python -m cli.main validate --warnings   # catalogue vs published configs + rules
python -m cli.main budget --ram-gb 4 --base-params-b 1.7 --adapters 20
```

`validate` runs two independent checks: whether the catalogue's KV geometry
matches published model configs (a wrong `n_kv_heads` corrupts every device
verdict), and whether the code obeys the architecture's own rules. Findings cite
the diagram they came from. Evidence and the full rule table:
[conformance.md](conformance.md).

`budget` computes the on-device RAM table line by line — base, KV cache,
embedder, grammar state, adapters, OS headroom — and exits non-zero when the
deployment does not fit.

## 6. Feasibility check
Will a build fit a device *before* you run it?
```bash
python -m cli.main feasibility --spec my_spec.json --device android_midrange
```
Returns predicted RAM (broken into **weights + KV cache + runtime**), latency,
battery, the device budget, headroom, and — on failure — concrete reasons.

The KV cache is the term nobody budgets for: it grows linearly with context
length and is the usual cause of on-device OOM. A model that fits at 2k context
can be infeasible at 128k, and the engine says so.

> **Unmeasured.** Every device profile ships `measured: false`. On-device latency
> and battery are heuristics until a physical device lab fills them in (GAP-10).
> The code labels every such number rather than implying it was measured.

## 7. Testing
```bash
pytest                      # offline suite (green, no network/GPU)
pytest -m integration       # heavy paths (needs the optional extras)
```
