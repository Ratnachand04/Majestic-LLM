# Conformance and compatibility

Three questions, answered separately because they fail for different reasons.
Run all three with `python -m cli.main validate --warnings`.

## 1. Model compatibility

Do the catalogue entries match the published model configurations? A wrong
`n_kv_heads` silently corrupts every KV-cache estimate, which corrupts every
device verdict, which is the number the product sells.

| Model | Layers | KV heads | Head dim | Params | Tokenizer | Verified |
|---|---:|---:|---:|---:|---|:--:|
| Qwen3-0.6B | 28 | 8 | 128 | 0.6B | qwen | ✅ |
| Qwen3-1.7B | 28 | 8 | 128 | 1.7B | qwen | ✅ |
| Qwen3-4B | 36 | 8 | 128 | 4.0B | qwen | ✅ |
| Qwen3-8B | 36 | 8 | 128 | 8.2B | qwen | ✅ |
| Qwen3-14B (teacher) | 40 | 8 | 128 | 14.8B | qwen | ✅ |
| Qwen3-32B (teacher) | 64 | 8 | 128 | 32.8B | qwen | ✅ |
| Qwen3-30B-A3B (MoE) | 48 | 4 | 128 | 30.5B / 3.3B active | qwen | ✅ |
| SmolLM2-360M | 32 | 5 | 64 | 0.36B | smollm | ✅ |
| Llama-3.2-1B | 16 | 8 | 64 | 1.24B | llama | ✅ |

**Result: 76 checks, 0 errors, 2 warnings.**

The two warnings are real and worth keeping visible: there is no teacher in the
`llama` or `smollm` tokenizer families, so bases from those families can only use
sequence-level KD, never logit KD (A-02, GAP-07). That is a catalogue-shape
consequence, not a defect — and exactly the single-vendor dependency GAP-07 warns
about, since every teacher is currently Qwen.

## 2. A correction the validation forced

A-01 states two int4 figures that do not reconcile:

- "4-bit at **~0.55 GB per billion** parameters"
- a size ladder whose entries all imply **~0.65 GB per billion**
  (0.6B→0.4 GB, 1.7B→1.1 GB, 3–4B→2.2 GB, 7–8B→4.5 GB)

Both are correct about different things. 0.55 is the pure-4-bit weight cost;
the ladder reports real on-disk size, where mixed-precision schemes (Q4_K_M)
keep embeddings and some tensors wider than 4 bits.

**Planning uses the ladder figure.** Budgeting with the theoretical number
under-counts every footprint by ~18%, which on a 4 GB device is the difference
between fitting and an OOM in the field. Both constants are named in
[`modelrig/catalogue.py`](../modelrig/catalogue.py) and a conformance check fails
if the constant stops reproducing the ladder.

With that correction the computed device budget reproduces B-09 line for line:

| Line | B-09 | Computed |
|---|---:|---:|
| Base model, 1.7B at 4-bit | 1.10 | 1.10 |
| KV cache at planned context | 0.22 | 0.23 |
| Embedder for the local index | 0.09 | 0.09 |
| Grammar state and runtime | 0.06 | 0.06 |
| Twenty adapters, resident | 0.60 | 0.59 |
| Application and OS headroom | 1.40 | 1.40 |
| **Total committed** | **3.47** | **3.47** |

## 3. Architecture conformance

Rules that live only in documentation drift. Each is executable, and each
failure names the diagram it came from.

| Rule | Source | Enforced by |
|---|---|---|
| Largest base that fits at 4-bit, never smaller-at-higher-precision | A-01 | `catalogue.bases_for(largest_first=True)`, `planner._eligible_bases` |
| Deployable size = weights + KV cache + runtime | A-01 | `feasibility.HeuristicPerfPredictor` |
| Free RAM ≥ 2× model file size | A-01 | `feasibility.evaluate` |
| int4 constant reproduces the size ladder | A-01 | `conformance` check |
| Logit KD requires an identical tokenizer | A-02 | Gate 2 |
| Closed-API teacher outputs are refused | A-02 | `licence.resolve_licence_chain` |
| Reverse-KL for extraction/classification, forward-KL for drafting | A-02 | `planner._kl_objective` |
| GKD only inside the repair loop | A-02 | `pipeline._repair_plans` |
| Full fine-tune only above a novel-vocabulary threshold | A-03 | `planner._needs_full_finetune` |
| Different bases cannot share a GPU | A-04 | `serving.AdapterPool.fragmentation_report` |
| Calibration from the customer's distribution | A-05 | `quantisation.build_calibration_set` |
| Answer-flip rate reported, re-eval mandatory | A-05 | `proving_ground`, Gate 3 |
| Confidence recalibrated after quantisation | A-05 | `quantisation.TemperatureScaler` |
| Escalate to SpQR above the outlier threshold | A-05 | `quantisation.next_escalation` |
| Cap chunks; strongest evidence at the boundaries | A-06 | `retrieval.context.assemble` |
| MoE excluded from memory-constrained targets | A-07 | `planner._eligible_bases` |
| Learned deferral, not a softmax threshold | A-08 | `router.deferral` |
| Offline hard-locks escalation | A-08, B-10 | `DeferralRouter`, `FabricRuntime` |
| Constrained tool-call schema, capped step depth | A-09 | `agent.react`, `grammar` |
| Untrusted content cannot reach a privileged tool | A-09, B-10 | analyser + runtime + ReAct loop |
| Every gate checked before any GPU | B-03 | `gates`, `pipeline` |
| Seed floor or refuse | B-06 | `data_factory` |
| Seven axes, all must pass | B-07 | `proving_ground` |
| Repairer acts only on external evidence | B-07 | `Repairer.repair` |
| Base stored once; lineage is a query | B-08 | `registry.CartridgeRegistry` |
| Parallel candidates, score both, pick one | B-01, C-02 | `candidates` |
| Nothing probed or derived is ever asked | P4-10 | `forge.slots.validate_table` |
| The elicited set stays a minority of the schema | P4-09 | `conformance` check |
| γ leaves 4 questions completing and 40 not | P4-04 | `conformance` check |
| Raising κ raises **both** θ\* and Λ — more refusals *and* more questions | P4-04 | `conformance` check |
| Only a measured source may promise | P3-07 | `conformance` check |
| An unseen device is reported as unverified | P3-11 | `CertificationLedger` |
| Every loadable container is one the planner can build | P3-10 | `conformance` check |
| Every accelerator keeps an offline-capable container | P3-10 | `conformance` check |
| The plan space stays enumerable | P2-02 | `conformance` check |
| HARD and SOFT partition the predicates | P2-16 | `conformance` check |
| Predicates sound by construction stay marked hard | P2-16 | `conformance` check |
| Predicates are ordered by c/(1−ρ) | P2-04 | `conformance` check |

The κ row is the one worth watching. `θ*` and `Λ` are computed from the
same `(V, κ)` pair by different formulas, and there is no structural reason they
must move together — it is a property of the tier priors, so it is asserted
rather than assumed. If they ever diverge, the regulated tier silently becomes
the *permissive* one, which is the worst possible failure mode and would produce
no error anywhere else.

## 4. What validation changed in the code

Three defects were found by writing these checks:

1. **The planner picked the smallest base that fit, not the largest.** A direct
   inversion of the k-bit scaling law — it would have shipped systematically
   weaker models than each device could carry.
2. **The int4 constant under-counted deployed weights by ~18%** (above).
3. **`TaskPrimitive` mixes in `str`,** so `isinstance(x, str)` matched enum
   members and `str(member)` produced `"TaskPrimitive.CLASSIFY"`. Any caller
   passing an enum where a string was also accepted crashed.

## 5. Still unvalidated

Conformance is not the same as truth. The checks prove the code obeys the
architecture; they cannot prove the architecture's *physical* claims. Both
GAP-05 and GAP-10 remain open, and every number they touch is labelled
`measured=False`. See [research-gaps.md](research-gaps.md).
