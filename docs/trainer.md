# The Trainer — complete mathematics, zero novelty

*Part 9 of the architecture. Implemented in [`modelrig/trainer.py`](../modelrig/trainer.py)
and [`modelrig/preflight.py`](../modelrig/preflight.py).*

```
train : Plan × Dataset → AdapterRef
```

Every decision — base, rank, method, learning rate, epochs, replay fraction,
preference stage — **arrives pre-made in the Build Plan**. The Trainer selects
nothing.

This is deliberate and it is why the system is verifiable. A choice made inside a
GPU job is unauditable: it happens after money is spent, with no predicate having
validated it. Pushing every decision upstream is what lets a build be refused in
eight seconds instead of failing in ninety minutes.

> **A subsystem with no opinions has no bugs of judgement.** It can still have
> bugs of execution, which is what the pre-flight is for.

## Zero novelty is the correct outcome

Low-rank adaptation is Hu et al. 2021. 4-bit training is Dettmers et al. 2023.
LoRA+ is Hayou et al. 2024. The preference methods are Rafailov, Hong and
Ethayarajh. The implementation is peft and TRL.

Writing a custom training loop would add risk, subtract reliability and
contribute no defensible claim. **Saying so plainly is the cheapest credibility
available** — one sentence makes reviewers believe the Grade-A claims elsewhere.

What this subsystem *does* contribute is the arithmetic the Planner needs, none
of which was derived anywhere in the series despite being quoted throughout.

---

## Where "~30 MB" comes from

Quoted for eight documents without derivation. Here it is:

```
P_train(r) = r · L · Σ over targets (d_in + d_out)
```

For the 1.7B geometry (d=2048, L=28, d_ffn=5504, GQA d_kv=1024) the per-layer
fan is **36,992**, so `P_train(r) ≈ 1.04M · r`:

| r | params | fp16 |
|---|---|---|
| 8 | 8.3M | 17 MB |
| **16** | **16.6M** | **33 MB** |
| 32 | 33.1M | 66 MB |
| 64 | 66.3M | 133 MB |

**The series-wide ~30 MB adapter is r=16 on a 1.7B base.**

---

## Memory is the binding constraint

Full fine-tuning costs 16 bytes per parameter — bf16 weights, bf16 grads, fp32
master, Adam *m*, Adam *v*. For 1.7B that is **27.2 GB before activations**, and
it does not fit a 24 GB card. That is the entire reason QLoRA is here.

```
M_QLoRA ≈ P·β_eff/8  +  16·P_train(r)  +  M_act
        ≈ 0.97 GB    +  0.27 GB        +  activations
```

**The optimiser term collapses because Adam states scale with *trainable*
parameters, which are ~1% of the total.** At batch 4 the total is 2.9 GB against
27.7 for full fine-tuning — a 10× reduction, and the difference between "needs an
A100" and "runs on a laptop GPU".

Note β_eff = 4.5, not 4.0 — NF4 with block scales, the same correction as the
deployment side.

**Gradient checkpointing** stores O(√L) activations and recomputes the rest:
~5× less memory for ~30% more compute. Majestic's builds are compute-cheap and
memory-constrained, so it is on by default.

---

## The accumulation trap

```
b_eff = b_micro × grad_accum × n_devices
```

Gradient accumulation buys effective batch at no memory cost — but it is **not
exactly** a larger batch, and the difference bites on variable-length inputs.

Averaging per microbatch and then across accumulation steps weights **short
sequences more heavily** than token-level averaging would:

| | loss | tokens | |
|---|---|---|---|
| microbatch A | 1.0 | 900 | |
| microbatch B | 3.0 | 100 | |
| **naive mean** | | | **2.00** — B counts equally |
| **token-normalised** | | | **1.20** — B counts for a tenth |

For variable-length documents — exactly Majestic's case — this is a silent,
systematic bias toward whatever the short inputs happen to teach. The two agree
only when lengths are equal, **which is why the bug is invisible on fixed-length
data**.

---

## The cost correction

LoRA skips backward-to-weights for the frozen base — those gradients are never
materialised — so it is `4PN`, not `6PN`.

| build | tokens | wall clock | cost |
|---|---|---|---|
| Apex extraction (1.7B, 2.1k examples, 3 epochs) | 3.2 M | ~220 s | **$0.12** |
| synthesis-heavy (4B, 60k examples, 3 epochs) | 90 M | ~4.2 h | **$8.33** |

> The series quoted **"$5–40 for training"** throughout. At the low end that is
> wrong by a factor of roughly **fifty**.

Compounding with Part 8's teacher-cost correction:

| | data | train | eval | quantise | **total** |
|---|---|---|---|---|---|
| **augmentation-dominated `extract`** | ~$0 | $0.12 | $0.20 | $0.15 | **$0.47** |
| synthesis-dominated `transform` | $10 | $8.33 | $3.00 | $0.15 | **$21.48** |

**An extraction build with labelled seeds costs under a dollar, not $20–120.**
Three things follow: the research compute budget is overstated by roughly 8×, per-build
pricing becomes nearly pure margin so the constraint is willingness-to-pay rather
than cost recovery, and hyperparameter sweeps become rational.

---

## Hedge and sweep are different things

Part 2 §15 proved parallel candidates are strictly worse in expected cost, ratio
`2/(2−p) > 1`. **That result governs *redundancy* — the same plan run twice
hoping one passes.** It does not govern a hyperparameter sweep, where the
candidates differ systematically and the Proving Ground performs *model
selection*.

```
redundancy: a second draw from one distribution
search:     the maximum over several          E[max qᵢ] > E[q]  strictly
```

`allow_parallel_candidates` was governing both, which meant one flag making two
opposite decisions. They are now separate: **hedge** stays opt-in (§15 applies),
**sweep** defaults on (four ranks at ~$0.12 a run is under a dollar, so guessing
the rank is a false economy). The legacy flag reads as `hedge` — the
conservative direction, since treating it as a sweep would silently enable
something nobody asked for.

---

## Why LoRA forgets less

`ΔW = (α/r)BA` has rank at most *r*, so the model can only move within an
*r*-dimensional subspace of each weight matrix's *dk*-dimensional space. A rank-16
update to a 2048×2048 matrix moves in 16 of 2048 directions — **most of the base
model's structure is structurally unreachable.**

Not regularisation in the penalty sense; a hard constraint on the reachable set.
And it explains the trade-off directly: the same constraint that prevents
forgetting prevents learning genuinely new domains. **One mechanism, both
effects.**

## Why KTO is the flywheel's method

Field feedback is a thumbs-up or thumbs-down on **one** output. It is not a pair.
Building DPO pairs from it requires generating a counterfactual or discarding
unmatched signal — both lossy.

**KTO consumes exactly what the flywheel produces, with no transformation.** That
is the whole argument, and it is why the invariants name KTO specifically rather
than "a preference method".

Watch class balance: field feedback skews negative — people report failures, not
successes — so `λ_y` is set from the observed ratio rather than left at defaults
tuned on balanced data.

ORPO needs no reference model (a reference doubles resident memory) and folds two
stages into one, which makes it the natural default for cost-sensitive builds —
subject to A/B validation rather than assumption.

---

## The pre-flight

The cheapest code in the subsystem and the highest-value: it costs nothing to run
and prevents the failures that are otherwise found only *after* a build completes
and evaluates cleanly.

**Every failure below reports success.** The loss falls, the checkpoint saves, the
scorecard fills in, and the model is quietly wrong. A silent failure that survives
evaluation is more expensive than a crash, because it ships.

| failure | symptom | check |
|---|---|---|
| **wrong chat template** | trains fine, subtly broken | template hash must match the base's |
| tokenizer mismatch | garbled learning | hash in both stages, compare |
| microbatch loss normalisation | short sequences overweighted | normalise by tokens |
| padding in the loss | learns to emit padding | assert the mask excludes pad |
| NaN divergence | garbage checkpoint | fail fast rather than save |

The first is the most common silent failure in fine-tuning generally, and it is a
hard assertion rather than a convention.

**Contamination is checked at two levels.** Example hashes catch the obvious case;
source-document provenance catches the subtle one, because an augmented variant of
a held-out document leaks it even though the bytes differ.

The pre-flight runs **inside `train`**, never in a caller — a structural property
rather than a convention, since the caller that forgets will be the one written in
a hurry six months from now.

---

## Novelty

**None.** Every component is published, every implementation borrowed, and this is
correct for a subsystem whose job is to have no opinions. Its contribution to the
system is different in kind: it makes every decision auditable by making none of
them.

## Still open

- Reproducibility is **eval-equivalent**, not bitwise — floating-point reduction
  order differs across GPU models. This shares a definition with Part 5 §16's
  synthetic regeneration and the two should not drift apart.
- The recipe table `(primitive, method, kd_direction) → config` is not yet
  extracted from the Planner into a standalone module.
- `ACTIVATION_CONSTANT = 19` is derived from the residual stream plus QKV plus
  MLP intermediate for this geometry; it is not measured.
