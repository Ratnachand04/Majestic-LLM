# The Proving Ground — the subsystem that can say no

*Part 7 of the architecture. Implemented in [`modelrig/stats.py`](../modelrig/stats.py)
and [`modelrig/proving_ground.py`](../modelrig/proving_ground.py).*

Every other part of Majestic produces candidates. This is the only one that can
reject them, and what Majestic sells is not a model file — it is the certificate
this subsystem issues.

> **If the statistics are wrong, the product is a lie with good typography.**

This page records two corrections to every earlier document in the series.

---

## Correction 1 — the gate is a hypothesis test, not a comparison

Every prior document wrote the gate as `q̂ ≥ q_gate`. That is wrong: `q̂` is an
estimate with variance and the comparison ignores it.

```
H₀: q ≤ q_gate        H₁: q > q_gate
```

The burden of proof sits on the model, which is the right direction — a
certificate asserts something, and an assertion needs evidence.

### What that costs, concretely

Apex: n = 50, observed 0.94, gate 0.93. Every earlier document treated this as a
pass. The Wilson interval:

```
0.940 [0.838, 0.979] at n=50
```

**Fourteen points.** "0.937 versus 0.93" is not a comparison — it is noise.

Exact one-sided 95% lower bounds at n = 50:

| result | q̂ | LCB | certifies 0.93? |
|---|---|---|---|
| 50/50 | 1.000 | **0.942** | ✓ barely |
| 49/50 | 0.980 | 0.906 | ✗ |
| 47/50 | 0.940 | 0.852 | ✗ |

**At n = 50 a single error makes a 0.93 claim uncertifiable.** The Apex build
this series has treated as a success cannot honestly claim its stated gate.

### How much data a gate needs

| detect | q₁ | required n |
|---|---|---|
| 3 points | 0.96 | **380** |
| 5 points | 0.98 | 116 |
| 8 points below | 0.85 | 82 |
| 10 points below | 0.83 | 55 |

Fifty examples resolves about **ten points**. It cannot resolve the three-point
differences gate thresholds are written at, and the schema's `holdout_count ≥ 20`
resolves *nothing at all* — `resolvable_difference(20, 0.93)` returns infinity.

> §4's table quotes 150 for the q₁ = 0.98 row. The one-sided formula the section
> header specifies gives **116**; that row alone matches a two-sided z. The other
> three rows reproduce exactly, so the formula is taken as authoritative.

### The lower bound governs status, not admission

A build whose *point estimate* clears the gate still ships — as **PROVISIONAL**,
with the interval stated. Refusing it outright would reject every build with a
realistic held-out set and would not make the customer better informed. What the
LCB governs is whether the certificate may say **CERTIFIED**, which is the claim
that has to be earned.

```
94% (between 84% and 98% — based on 50 of your examples) — PROVISIONAL:
the estimate clears your target but this much data cannot yet prove it.
It will tighten with use
```

Three honest routes, and one dishonest one:

| route | |
|---|---|
| collect more held-out data | ✓ best |
| **certify what n supports** — offer 0.85 rather than 0.93 | ✓ honest |
| Bayesian narrowing with a capped prior | ✓ with care |
| provisional certification that strengthens with field data | ✓ |
| report the point estimate and hide the interval | ✗ what the design did |

### Provisional certification

Corrections are ground truth, so `n_eff(t) = n₀ + rVt` and the interval narrows
with deployment. At 200 requests/day and a 6% correction rate that is 12 labelled
examples a day — **about 27 days** to reach n = 380. This does not manufacture
confidence; it defers the strong claim and says when it will arrive.

### Bayesian narrowing has a trap

The prior encodes "models like this usually work", so a genuinely broken build
gets credit for its ancestors' successes. Two guards, both enforced:

1. **Cap prior strength at n.** The prior may never outvote the customer's own data.
2. **Report both numbers.** A customer entitled to know their model works is
   entitled to know which number came from their data.

---

## Correction 2 — seven blocking gates would reject four good models in five

```
P(ship | model is good) = ∏(1 − βᵢ)
```

| per-axis power | P(ship \| good) |
|---|---|
| 0.80 | **21%** |
| 0.90 | 48% |
| 0.95 | 70% |

At 80% power per axis, **four out of five good models are rejected**. That is not
conservatism — it is a broken gate, and it would show up directly as a
catastrophic first-attempt pass rate.

**Running an axis and blocking on it are different decisions.** Conflating them
caused the problem.

| axis | runs | blocks | why |
|---|---|---|---|
| task metric | ✓ | **✓** | it is the job |
| safety | ✓ | **✓** | unbounded loss |
| privacy | ✓ | **✓** | unbounded loss |
| contamination | ✓ | **✓** | deterministic — costs no power |
| judge | ✓ | advisory | noisy estimator |
| behavioural | ✓ | advisory | |
| regression | ✓ | advisory | some forgetting is correct for a specialist |
| calibration | ✓ | advisory | fixable post-hoc by re-fitting temperature |

`0.9³ = 73%` instead of 21%.

**The invariant that survives: no axis may be skipped.** Every one runs, every
result appears on the scorecard, and advisory failures are shown prominently.
What changed is only which failures halt the pipeline. This supersedes
`INVARIANTS.md`: **seven mandatory axes, four blocking conditions.**

---

## Why churn predicts field failure

```
φ = φ₊ + φ₋        Δ = φ₋ − φ₊
```

| | φ₊ | φ₋ | Δ | φ |
|---|---|---|---|---|
| Model A | 0.01 | 0.03 | 0.02 | **0.04** |
| Model B | 0.15 | 0.17 | 0.02 | **0.32** |

Identical accuracy delta; **B has rewritten a third of its answers.** Δ measures
only the net; φ measures the churn.

The mechanism: if quantisation perturbs the decision function by ε, then

```
φ ≈ P(|margin| < ε)
```

φ is an estimate of **how much probability mass sits near the decision
boundary** — a direct measure of fragility. On the held-out set those
perturbations happen to cancel; on new data there is no reason they should. Δ
contains no information about fragility at all.

That gives C4 a *mechanism* rather than a correlation, and a second, cheaper
experiment: if φ measures boundary mass, high-φ models should show **higher
bootstrap variance** in accuracy across resamples. `bootstrap_variance()` makes
that testable with no additional models and no field data.

---

## Judge attenuation

```
n_usable ≈ n(1 − f)ρ²
```

The ρ² is classical attenuation — measurement error in the instrument reduces
effective information quadratically. At n = 50, f = 0.15, ρ = 0.80:
**n_usable ≈ 27**. Half the data gone to instrument noise, which is the
quantitative reason the Judge is advisory rather than blocking.

## Paired comparison, honestly read

Student and teacher are evaluated on the *same* items, so McNemar applies and
only the discordant pairs carry information. At n = 50 those might number five to
eight, where the test has almost no power:

> **The honest headline is "we could not detect a difference", not "the student
> matches the teacher".** Those are different claims and only the first is
> supported.

`McNemarResult.headline()` returns the weaker claim when the discordant count is
too small to resolve one.

---

## Novelty, honestly

| claim | grade |
|---|---|
| Multi-axis evaluation | none — HELM |
| Judge de-biasing | none — published |
| **Flip rate with a boundary-mass mechanism** | **strong** — C4, now testable |
| **Provisional certification narrowing with field data** | moderate — falls out of the architecture |
| **Blocking/advisory partition from power analysis** | moderate — a design result |
| Repairer as contextual bandit | none — standard, correctly applied |

Grade A, carried by the flip-rate mechanism.

## Still open

- `holdout_count ≥ 20` remains in the Spec IR schema. §4 shows it resolves
  nothing; raising it is a schema change with customer-facing consequences and
  is not made here.
- The Repairer's `θ_m | F` table is hand-written. Repair attempts are unusually
  informative training data — each is a controlled comparison on the same spec
  with one coordinate changed — so it should be learned once outcomes accumulate.
