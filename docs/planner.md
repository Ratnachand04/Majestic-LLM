# The Planner

The middle end of the compiler. It receives a validated specification and emits
either an executable build plan or a **refusal**, before any GPU is allocated.

> *It decides whether the thing the customer asked for is physically, legally and
> statistically possible, and if so, how to build it most cheaply.*

```bash
python -m cli.main plan --spec spec.json --explain
python -m cli.main plan --spec spec.json --tier regulated
```

## Three properties

**Deterministic.** The common path involves no language model. Same spec, same
catalogue, same history gives the same plan — which is what makes the system
auditable, cacheable and reproducible, not a stylistic preference.

**Refusal is a first-class output.** `Refusal` sits alongside `BuildPlanIR` in
the return type. Every AutoML system and every fine-tuning platform always
returns *something*. Majestic returns nothing when nothing will work.

**Constraints span domains that do not normally meet.** RAM arithmetic is
physics, licence composition is law, seed sufficiency is statistics — and a plan
must satisfy all three simultaneously.

## Why enumeration, not search

The plan space is about **1.4 × 10⁶** points. NAS operates over 10¹⁸ or more,
which forces RL or evolutionary search at thousands of GPU-hours per result. At
10⁶ with aggressive early pruning, exhaustive enumeration terminates in
milliseconds.

But the stronger argument is structural. `P_ram` is **antitone** in model size:
if a 4B fits, every smaller base fits. So the feasible set is a *down-set in a
total order* — a chain — and has a unique maximal element:

$$b^\star = \max\{\, b \in \mathcal{B} : P_{\text{ram}}(s, b, w{=}4) \,\}$$

`select_base()` is that maximum: six comparisons, no enumeration. Monotone
structure, not compute budget, is why rejecting search is correct.
`verify_antitone()` asserts the property over the live catalogue, because if an
exotic architecture ever breaks it, `select_base` silently stops being a maximum
and starts being a guess.

## The seven predicates

| Predicate | Domain | Soundness | Module |
|---|---|---|---|
| `P_tok` | definitional | **hard** | derives the distil mode from tokenizer identity |
| `P_ram` | physics | **hard** | exact KV formula, effective bits, GQA |
| `P_lic` | law | **hard** | join-semilattice over restriction atoms |
| `P_off` | structural | **hard** | offline closure at plan level |
| `P_seed` | statistics | *soft* | derived coverage floor |
| `P_lat` | physics + epistemic | *soft* | refuses on unmeasured hardware |
| `P_cost` | financial | *soft* | integer micro-USD |

### Ordering

Predicates form a conjunction, so order cannot affect correctness — only cost.
Expected cost under ordering σ is $\sum_i c_{\sigma(i)} \prod_{j<i} \rho_{\sigma(j)}$,
minimised by sorting ascending on $c_i / (1-\rho_i)$. This is the classical
sequential-testing result, the same rule a query optimiser uses.

The derived order is `P_tok → P_ram → P_seed → P_lat → P_lic → P_cost → P_off`,
and it costs **3.3 predicate evaluations per candidate instead of 7**.

> The spec's §4 table lists a different order (`P_tok, P_seed, P_lic, P_ram,
> P_off, P_lat, P_cost`). That table is inconsistent with the formula stated
> immediately above it — it places `P_lic` (key 20) before `P_ram` (key 5), and
> `P_off` (key 40) before `P_lat` (key 10). The code follows the **formula**,
> which is what §4 actually prescribes.

### `P_ram` — the exact form

$$M = M_w + M_{kv} + M_{emb} + M_{rt}, \qquad
M_w = \frac{N_b \cdot w_{\text{eff}}}{8}(1+\epsilon), \qquad
M_{kv} = 2 L H_{kv} d_h C B\, p_{kv}$$

Two corrections from §14, and they run in **opposite directions** — so a naive
implementation can look roughly right on aggregate while being wrong on both
terms. They are tested separately.

- **Effective bits, not nominal.** A "4-bit" k-quant is 4.5–5.5 bits per
  parameter once scales, zero-points and block metadata are counted. Using 4.0
  under-counts by 15–25% and produces plans that pass `P_ram` then OOM.
- **GQA dimension.** The KV term uses $d_{kv} = n_{kv} \times d_{\text{head}}$,
  several times smaller than $d_{\text{model}}$. Substituting the model dimension
  overestimates by the group factor (4×–8×) and causes spurious refusals.

The "free RAM ≥ 2× model file" rule is a crude approximation usable only when the
architecture is unknown — which is never, because the catalogue holds it.

### `P_lat` — the refusal that matters most

Prefill is **compute-bound** ($t \approx 2N_b C_{in}/F_{\text{eff}}$); decode is
**memory-bandwidth-bound** ($t \approx M_w / BW_{\text{eff}}$). Conflating them is
the most common source of wrong latency estimates.

*Why decode is bandwidth-bound:* decoding one token performs ~$2N_b$ FLOPs while
moving ~$M_w$ bytes — an arithmetic intensity of $16/w$, about **4 FLOPs per byte
at 4-bit**. Processors balance at 50–200. Decode sits an order of magnitude below
the ridge, so no extra compute helps; only smaller weights or faster memory do.
Hence **quantisation buys latency roughly linearly**.

**If `latency_source` is `unmeasured`, this predicate fails** regardless of the
arithmetic. Effective throughput runs at 30–60% of peak and varies with thermal
state, kernel quality and quantisation format. A planner that interpolates them
is fabricating a promise. Set `io_schema.accept_unmeasured_latency` to accept an
estimate explicitly.

### `P_seed` — a derived floor, not an asserted one

A behavioural mode of mass $\mu$ is absent from $n$ i.i.d. seeds with probability
$(1-\mu)^n \approx e^{-\mu n}$. Amplification cannot recover it: the synthetic
corpus **inherits the support of the seed set**, and no volume of generation
extends that support. Requiring miss probability ≤ η gives

$$n_{\min} \ge \frac{-\ln \eta}{\mu_{\min}}$$

| Primitive | μ_min | Derived floor |
|---|---:|---:|
| `classify` | 0.050 | 60 |
| `extract` | 0.030 | 100 |
| `summarise` / `rewrite` | 0.025 | 120 |
| `answer` | 0.030 | 100 |
| `route` | 0.020 | 150 |
| `generate` | 0.015 | 200 |
| `toolcall` | 0.012 | 250 |

The assumption is explicit and falsifiable: it presumes **i.i.d. seeds**. Real
corpora arrive in correlated batches, so effective sample size is below the raw
count and **these floors are optimistic**. `(η, μ_min)` are marked
`# HYPOTHESIS — R4` and must not be tuned until the formula agrees with a
previous guess.

### `P_lic` — a join-semilattice

Licences are modelled as their **set of restriction atoms**; composition is set
union. That buys two properties as theorems rather than tests:

- **Associativity** — the solver needs no canonical component ordering.
- **Monotonicity** — adding a component can only tighten the result, so a plan
  that fails the licence check **cannot be rescued by adding anything**.

Licences are only a *partial* order (Gemma terms and CC-BY-SA are incomparable),
so a "most restrictive wins" chain would be ill-defined. The powerset lattice
handles incomparability for free.

## The decision: when to refuse a *feasible* plan

Hard predicates are not probabilistic — if a model does not fit, it does not fit.
The interesting case is soft: the plan is feasible but predicted quality is low.

$$\mathbb{E}[\text{build}] = p(V+\kappa) - (C+\kappa)
\quad\Longrightarrow\quad
\theta^\star = \frac{C+\kappa}{V+\kappa}$$

With C=\$40, V=\$300, κ=\$200: **θ\* = 0.48**.

- **κ dominates as it grows.** In regulated domains θ\* sits near 1.
- **Even a free build can be refused.** As C→0, θ\* → κ/(V+κ), still bounded away
  from zero. The damage of shipping a bad model is not the compute you burned —
  which is why "just build it and see" is wrong even at zero marginal cost.
- **It all rests on calibration.** If p̂ is badly calibrated near θ\*, the optimal
  policy collapses to "build everything" and the contribution evaporates. That is
  claim C2, and it is an empirical test of this entire mechanism.

Note that κ alone does not order the tiers: θ\* rises when V falls too, so a tier
that values the model very little becomes *conservative*, however failure-tolerant
it is. The tier priors account for that.

## Refusal, and why `minimal_cover` is the product

Enumeration produces one witness per eliminated candidate — thousands of entries
saying the same few things. `minimal_cover` solves the **set cover**: the universe
is the eliminated candidates, each predicate is the set it killed. Greedy is a
`ln n`-approximation and at seven predicates is very often exactly optimal. Ties
break toward **hard** predicates, because a certain failure is more actionable
than an inferential one.

```
REFUSED — no feasible plan

  P_ram   [hard] Qwen/Qwen3-4B at awq_int4 needs 4165 MB (weights 2313 + KV 302
                 at 2048 ctx + runtime 150 + app 1400) but android_tablet_4gb has
                 4000 MB free — over by 165 MB
  P_lat   [soft] Llama-3.2-1B on android_tablet_4gb: estimated 10.0 s (7.3 s
                 prefill over 500 tokens + 2.7 s decode over 60) against a 2.0 s
                 budget — prefill dominates

  Remedies, in order of expected quality:
  1. drop to the next smaller base
  2. reduce input length below 500 tokens — prefill is 73% of the total

  NOTE: latency_source = unmeasured for android_tablet_4gb.

  144 of 144 candidate plans eliminated before any GPU was allocated.
```

## Parallel candidates are opt-in (§15)

For k identical candidates with pass probability p:

$$\frac{\mathbb{E}[\text{cost}]_k}{\mathbb{E}[\text{cost}]_1} = \frac{kp}{1-(1-p)^k} > 1
\qquad \forall\, p \in (0,1)$$

At p = 0.5 two candidates cost **33% more** in expectation than building one and
retrying. What they buy is wall-clock time and variance reduction — a
time-for-money trade the customer must choose. The Planner never enables them on
its own initiative; set `io_schema.allow_parallel_candidates`.

When they *are* requested, alternates are chosen to **maximise failure-mode
diversity** (different base family), because correlated failures make the ratio
worse than the independence analysis suggests.

## Hard vs soft, and how to evaluate it (§16)

`P_ram`, `P_tok`, `P_lic` and `P_off` are sound **by construction**. Their
precision is exactly 1 and no experiment can inform it. Aggregate refusal
precision is therefore **inflatable**: padding an evaluation corpus with
memory-infeasible specs drives it toward 1 while measuring nothing.

Report `p_soft` as the primary result, stratified by which predicate fired.
`HARD` and `SOFT` carry the partition in code so this cannot be blurred by
accident.

> The novelty is that the hard predicates are checked at all — nobody does that.
> The research question is whether the soft ones can be calibrated well enough to
> be worth checking. **Two different claims; make both, separately.**

## Module map

| Module | Contents |
|---|---|
| `catalog.py` | typed loaders; malformed catalogue is a startup failure |
| `licence_lattice.py` | the join-semilattice and its atoms |
| `costmodel.py` | integer micro-USD; every constant carries `# SOURCE:` |
| `predicates.py` | the seven predicates, ordering, hard/soft partition |
| `refusal.py` | witness set, `minimal_cover`, rendered refusals |
| `objective.py` | `U(π,s)`, `θ*`, the quality prior |
| `metalearn.py` | empirical-Bayes shrinkage, precedent index |
| `core.py` | `select_base`, enumeration, `plan()` |
| `compat.py` | the stateful adapter the build pipeline uses |

## Open questions

1. **Is p̂ calibratable at all?** The whole refusal apparatus rests on it, and the
   answer might be no.
2. **Are V and κ estimable per customer?** Tiering is a guess and is unvalidated.
3. **Does precedent reuse lock in early mistakes?** `PrecedentIndex` exposes an
   `exploration_rate` floor; the explore/exploit problem is not solved here.
4. **Do the derived seed floors survive non-i.i.d. data?** They are optimistic.
