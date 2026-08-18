# Research gap register

Where no paper exists. These are original engineering, and therefore both the
schedule risk and the defensible contribution. Each gap names the code that
addresses it today and what is still missing.

| Gap | Open problem | Status in this repo | What is still missing |
|---|---|---|---|
| **GAP-01** | A typed IR for task specifications | **Addressed** — [`modelrig/ir.py`](../modelrig/ir.py) implements the three-level hash-addressed stack; [`modelrig/gates.py`](../modelrig/gates.py) implements the checkable predicates | Schema coverage will widen as primitives are validated |
| **GAP-02** | Predicting gate-pass *before* spending money | **Scaffolded** — `OutcomePredictor` in [`modelrig/feasibility.py`](../modelrig/feasibility.py) records (spec, plan, outcome) triples and serves a frequency baseline | The learned predictor over spec embeddings. Cold-start builds still burn money |
| **GAP-03** | Calibrated confidence from sub-2B models | **Measured, not solved** — ECE is computed as an axis in [`modelrig/proving_ground.py`](../modelrig/proving_ground.py) and warned on at Gate 3 | Calibration heads trained at 0.6-4B; a distilled single-pass uncertainty estimator |
| **GAP-04** | Static verification of offline closure | **Addressed** — [`majestic/fabric/analyser.py`](../majestic/fabric/analyser.py) proves offline closure, RAM and cost bounds, and injection reachability over a typed DAG | Richer type checking on edge payloads |
| **GAP-05** | Multi-generation flywheel quality dynamics | **Instrumented, not characterised** — [`majestic/flywheel.py`](../majestic/flywheel.py) refuses regressions and `GenerationLog.degrading()` is an early warning | Longitudinal experiments across ~20 generations to derive the real:synthetic floor. `MAX_SYNTHETIC_RATIO` is a guess |
| **GAP-06** | Compressing business requests into a closed primitive set | **Provisional** — the eight primitives are defined in [`modelrig/primitives.py`](../modelrig/primitives.py) with `coverage_report()` to measure them | Validation against real incoming specs, and the published coverage rate |
| **GAP-07** | Cross-tokenizer logit distillation | **Avoided safely** — the planner defaults to sequence-level KD and Gate 2 refuses cross-family `logit_kd` outright; `cli validate` warns that no `llama`/`smollm` teacher exists | Either a validated cross-tokenizer method, or accept the single-vendor teacher dependency the warning exposes |
| **GAP-08** | Automated licence-chain resolution | **Addressed** — [`modelrig/licence.py`](../modelrig/licence.py) composes base, teacher, data rights and jurisdiction, and refuses illegal combinations before training | Real legal review of the rule table; more jurisdictions |
| **GAP-09** | An eval a non-technical customer will trust | **Partially** — the scorecard carries samples and honest failure cases | Customer trust and acceptance measured as a first-class product metric |
| **GAP-11** | Which questions are worth asking a customer who can leave | **Addressed in principle** — [`modelrig/forge/infogain.py`](../modelrig/forge/infogain.py) measures information gain by running the **Planner** over each candidate answer, so a slot whose answers all yield the same plan scores exactly zero; [`core.py`](../modelrig/forge/core.py) applies `IG·stake·Λ > γ` ([detail](forge.md)) | **γ is a hypothesis.** No published elicitation study measures per-question attrition, because their subjects are paid to finish. 0.12 reproduces the observed shape; it is not a measurement. `DELTA_SHARE` is likewise a stake weight |
| **GAP-10** | Adapter-swap latency and thermals on real mobile hardware | **Route out found** — the [device probe](device-verification.md) makes the customer's own hardware the measurement instrument: two benchmarks calibrate `t = S/BW + c` exactly, `P_lat` may then promise, and `devices.yaml` is demoted to priors | Actual probe deployments. **Adapter-swap latency specifically is still unmeasured**, and the SIMD→format mapping is still folklore |

## How to read this

Two gaps are load-bearing product promises that remain **unvalidated**:

- **GAP-10** — the claim that twenty specialists live in ~1.5 GB on a mid-range
  phone with ~100 ms swaps is an assumption in this design. MELTing Point
  (2403.12844) shows the binding mobile constraints are energy and thermal
  throttling, not raw speed. The code never claims otherwise: `PerfEstimate.measured`
  is `False` everywhere and Gate 1/Gate 2 emit a warning saying so.
- **GAP-05** — the flywheel is the only permanent moat, and it could silently
  degrade models over years before anyone notices.

Everything the code asserts about these is a prediction, and it says so.
