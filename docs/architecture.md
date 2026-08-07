# Architecture

**Majestic is a model compiler, not a model.** There is no universal model that
emits small models. There is a specification IR, a planner that runs verified
passes over it, and a backend matrix that targets devices. Flexibility is a
property of the architecture, not of any one component — which is exactly why
the "one flexible model" framing has no answer and this one does.

## The compiler mapping

| LLVM / TVM | Majestic | What the role does |
|---|---|---|
| C, Rust, Swift source | plain-language description | human intent enters |
| parser and lowering | **FORGE** ([`modelrig/forge.py`](../modelrig/forge.py)) | intent becomes a typed artefact |
| intermediate representation | **Spec IR** ([`modelrig/ir.py`](../modelrig/ir.py)) | the single contract everything binds to |
| optimisation passes | **PLANNER** ([`modelrig/planner.py`](../modelrig/planner.py)) | choices under hard constraints |
| back ends (x86, ARM, WASM) | targets: GGUF, ExecuTorch, CoreML, ONNX, vLLM | one plan, many deployment shapes |
| object file | **cartridge** ([`modelrig/cartridge.py`](../modelrig/cartridge.py)) | the shippable, versioned artefact |
| test suite | **PROVING GROUND** ([`modelrig/proving_ground.py`](../modelrig/proving_ground.py)) | proof the artefact does what was asked |

## The seven subsystems

| # | Subsystem | Code | Role |
|---|---|---|---|
| 1 | FORGE | `modelrig/forge.py` | slot-filling interview → Spec IR |
| 2 | PLANNER | `modelrig/planner.py` | constrained search + validator → Build Plan IR |
| 3 | DATA FACTORY | `modelrig/data_factory.py` | seed-anchored amplification with collapse guardrails |
| 4 | TRAINER | `modelrig/planes.py`, `modelrig/training_hf.py` | LoRA/QLoRA (heavy) or the offline classifier path |
| 5 | PROVING GROUND | `modelrig/proving_ground.py` | seven axes, gate-enforced, repair loop |
| 6 | REGISTRY | `modelrig/registry.py` | content-addressed cartridges, dedup, lineage |
| 7 | FABRIC | `majestic/fabric/` | typed DAG runtime, statically verified |

End to end: [`modelrig/pipeline.py`](../modelrig/pipeline.py).

**LLMs occupy exactly five roles**; everything else is deterministic software.
1 front end (FORGE), 2 teacher (data factory), 3 judge and 4 repairer (proving
ground), 5 planner tie-breaker — and the planner's proposals are always disposed
of by a validator.

## The three-level IR and its gates

```
LEVEL 1  SPEC IR        what the customer wants
   |     GATE 1  primitive in the set of eight? seed floor met? data rights?
   |             quality bar achievable at this device tier?
LEVEL 2  BUILD PLAN IR  how the system will make it
   |     GATE 2  fits device RAM WITH the KV cache? latency? tokenizer identity
   |             for logit KD? offline closure? licence chain? budget?
LEVEL 3  ARTEFACT IR    what actually exists on disk
         GATE 3  eval passed on REAL held-out data? re-evaluated after
                 quantisation with answer-flip inside bound? regression,
                 safety and privacy clean? certification attached?
OUTPUT   CARTRIDGE — admitted to the registry
```

Every gate is checked **before** any GPU is spent. An infeasible plan caught at
Gate 2 costs nothing; caught after training it costs sixty dollars. **A refusal
in eight seconds is a better product than a failed build in ninety minutes.**

## The compound runtime (majestic/)

The inference-side system that a cartridge runs inside:

| Concept | Code | Status |
|---|---|---|
| Reasoning core | `majestic/core/` (`mock`, `hf_core`) | mock offline; real HF lazy |
| Perception / bus | `majestic/perception/encoders.py`, `representation/bus.py` | hashing encoder; ST optional |
| Capability router | `majestic/router/rule_router.py` | rule + confidence + cost + escalation |
| **Learned deferral** | `majestic/router/deferral.py` | quality-gap routing, offline hard-lock |
| Expert & tool pool | `majestic/experts/` | web, code-exec, classifier real; diffusion/world-model stubs |
| Retrieval & memory | `majestic/retrieval/` | in-memory cosine; FAISS optional |
| Verifier | `majestic/verification/` | schema, code, math, factuality |
| **Fabric** | `majestic/fabric/` | typed DAG + static analyser |
| **Flywheel** | `majestic/flywheel.py` | corrections → KTO signal → rebuild-if-wins |

## Load-bearing constraints from the prior art

- **A-01** Deployable size = weights **+ KV cache + runtime**, never weights
  alone. Free RAM must be ≥ 2× the model file size, which is why a 4 GB phone
  tops out near 1.7B at 4-bit. 4-bit is the optimal accuracy-per-bit point, so
  the correct move under a fixed RAM budget is the largest base that fits at
  4-bit — never a smaller base at higher precision.
- **A-02** Outputs of a closed commercial API cannot train a competing model.
  Logit KD needs an identical tokenizer; sequence-level KD is the default path.
- **A-03** LoRA underperforms full fine-tuning on genuinely new domains but
  forgets far less — so the planner escalates tiers deliberately, and a
  regression suite runs either way.
- **A-05** Quantisation is where a good fine-tune quietly dies. Aggregate
  accuracy parity is an illusion; **answer-flip rate** is reported.
- **A-07** Sparse activation ≠ sparse residency: MoE bases are excluded from
  memory-constrained targets.
- **A-08** Confidence-based deferral is provably weak for specialists under
  distribution shift — the deferral rule is learned, not a softmax threshold.
- **A-09** Web scraping is a tool the model *calls*, never a capability it
  *contains*; retrieved content is untrusted input.
- **B-06** Below a real-seed floor the factory refuses rather than shipping a
  model that will quietly degrade.
- **B-07** The repairer acts only on external evidence; unaided self-reflection
  degrades performance.

## Knowledge vs behaviour

**Knowledge lives in the index. Behaviour lives in the adapter.** Adding four
hundred products is a reindex: seconds, no GPU, no cost. Changing *how* the model
answers is a rebuild: hours, tens of dollars. Conflating the two turns every
catalogue update into a retraining bill.

## Deliberate stubs

Wired as interfaces, not implemented: frontier pre-training; JEPA/world-model
training; image/video generation; production GGUF/CoreML/TFLite exporters; the
learned perf predictor. See [research-gaps.md](research-gaps.md) for what is
genuinely unsolved versus merely unimplemented.

## Offline-first strategy

The default path uses only a light dependency core (pyyaml, pydantic, numpy).
Every heavy capability is imported lazily with a working fallback, so `pytest` is
green with no network or GPU and the system runs on modest hardware.
