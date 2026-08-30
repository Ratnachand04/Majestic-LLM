"""Registry economics: dedup, cache tiers, and blast radius (Part 5 §2-§6, §12).

The Registry is graded C for novelty and that grade is right — content-addressed
storage is textbook. But it is the subsystem whose arithmetic determines **gross
margin**, and every equation here maps to a line in a P&L. That is why it gets
the same rigour as the Planner despite never being publishable.

Three results worth knowing before reading the code:

* **Dedup is governed by ``k = n/b``, not by ``n``.** Adding customers helps;
  adding bases hurts proportionally.
* **The dedup ceiling is just the base-to-adapter size ratio.** No amount of
  scale exceeds it, and at 1120 MB / 30 MB that is 37x.
* **Base diversity does not reduce expected loss.** It buys protection against
  *correlated catastrophic* failure, which is a different thing and is only
  worth paying for because loss is superlinear in the number simultaneously
  affected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

MB = 1_000_000

#: Typical artefact sizes, for the worked figures.
BASE_MB = 1120
ADAPTER_MB = 30

#: §5's tier costs, as a share of a full build.
COMPOSE_SHARE = 0.05      # forward passes only, no gradients
WARM_START_SHARE = 0.60   # initialise from the nearest adapter


# =========================================================================== #
# §2-§3 — deduplication
# =========================================================================== #
def dedup_ratio(k: float, base_mb: float = BASE_MB, adapter_mb: float = ADAPTER_MB) -> float:
    """``D(k) = k*S_B / (S_B + k*S_A)`` — naive storage over content-addressed.

    ``k`` is cartridges per base. Reparameterising by ``k`` rather than ``n`` is
    the whole insight: ``D`` does not depend on the customer count except
    through this ratio.

    .. note::

       §2's table quotes ``D(10) = 8.6``. The formula above, at the same sizes
       the table uses, gives **7.89** — and the table's other five rows (0.97,
       1.9, 27.3, 34.7, 37.3) all reproduce exactly. No size pair yields 8.6 at
       ``k = 10`` while also giving a 37.3 ceiling, so that cell is a slip
       rather than a different parameterisation. The formula is authoritative.
    """
    if k < 0 or base_mb <= 0 or adapter_mb <= 0:
        raise ValueError("k must be non-negative and sizes positive")
    return k * base_mb / (base_mb + k * adapter_mb)


def dedup_ceiling(base_mb: float = BASE_MB, adapter_mb: float = ADAPTER_MB) -> float:
    """``lim_{k->inf} D(k) = S_B / S_A``.

    The asymptote is *just* the size ratio. Scale cannot beat it, so the only
    way to raise the ceiling is a smaller adapter or a larger base — and the
    latter costs device memory, which is the trade the Planner is already making.
    """
    return base_mb / adapter_mb


def break_even_k(base_mb: float = BASE_MB, adapter_mb: float = ADAPTER_MB) -> float:
    """``k* = S_B / (S_B - S_A)`` — where content-addressing starts paying.

    Just above 1, and the consequence is worth stating because it looks like a
    bug: **CAS is marginally worse for the first cartridge on a base**, since it
    stores base *and* adapter where naive stores one merged model. A demo with
    three customers across six bases will measure CAS as worse than naive, and
    that measurement is correct.
    """
    if adapter_mb >= base_mb:
        raise ValueError("an adapter at least as large as its base never dedups")
    return base_mb / (base_mb - adapter_mb)


@dataclass(frozen=True)
class DedupReport:
    cartridges: int
    distinct_bases: int
    base_mb: float
    adapter_mb: float

    @property
    def k(self) -> float:
        return self.cartridges / self.distinct_bases if self.distinct_bases else 0.0

    @property
    def ratio(self) -> float:
        return dedup_ratio(self.k, self.base_mb, self.adapter_mb) if self.k else 0.0

    @property
    def ceiling(self) -> float:
        return dedup_ceiling(self.base_mb, self.adapter_mb)

    @property
    def paying(self) -> bool:
        return self.k > break_even_k(self.base_mb, self.adapter_mb)

    @property
    def headroom(self) -> float:
        """How much of the achievable dedup is still unrealised."""
        return round(1.0 - self.ratio / self.ceiling, 4) if self.ceiling else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cartridges": self.cartridges,
            "distinct_bases": self.distinct_bases,
            "cartridges_per_base": round(self.k, 2),
            "dedup_ratio": round(self.ratio, 2),
            "ceiling": round(self.ceiling, 1),
            "break_even_k": round(break_even_k(self.base_mb, self.adapter_mb), 2),
            "paying": self.paying,
            "headroom": self.headroom,
            "note": (
                "dedup is governed by cartridges-per-base, not customer count: "
                "adding customers helps, adding bases hurts proportionally"
            ),
        }


# =========================================================================== #
# §5-§6 — the cache hierarchy
# =========================================================================== #
@dataclass(frozen=True)
class CacheMix:
    """The four tiers an incoming spec can land in, as probabilities."""

    exact: float = 0.15
    compose: float = 0.20
    warm_start: float = 0.25
    miss: float = 0.40

    def __post_init__(self) -> None:
        total = self.exact + self.compose + self.warm_start + self.miss
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"tier probabilities must sum to 1, got {total:.4f}")
        if min(self.exact, self.compose, self.warm_start, self.miss) < 0:
            raise ValueError("tier probabilities must be non-negative")

    @property
    def hit_rate(self) -> float:
        return 1.0 - self.miss

    @classmethod
    def cold_start(cls) -> CacheMix:
        """An empty registry. **The compose tier contributes nothing.**

        ``p_compose = 0`` until the library holds tens of adapters per
        primitive, so a margin projection that assumes it from month one is
        wrong. It arrives about when the meta-learner does, and for the same
        reason: both need history.
        """
        return cls(exact=0.0, compose=0.0, warm_start=0.0, miss=1.0)


def expected_cost(
    mix: CacheMix, full_cost: float = 40.0, *,
    compose_share: float = COMPOSE_SHARE, warm_share: float = WARM_START_SHARE,
) -> float:
    """``c_bar`` — the mean cost of serving one build across the tiers.

    An exact hit costs nothing, which is why it does not appear in the sum.
    """
    return (
        mix.compose * compose_share * full_cost
        + mix.warm_start * warm_share * full_cost
        + mix.miss * full_cost
    )


def gross_margin(mix: CacheMix, full_cost: float = 40.0, price: float = 199.0,
                 **kw: Any) -> float:
    """``1 - c_bar / P``."""
    if price <= 0:
        raise ValueError("price must be positive")
    return 1.0 - expected_cost(mix, full_cost, **kw) / price


def margin_report(mix: CacheMix, full_cost: float = 40.0,
                  price: float = 199.0) -> dict[str, Any]:
    """What caching is worth, against the no-cache baseline.

    The gap between the two margins is the entire economic argument for the
    Registry, which is why cache-hit rate belongs on the same dashboard as
    revenue rather than in an engineering metric somewhere.
    """
    cached = gross_margin(mix, full_cost, price)
    uncached = gross_margin(CacheMix.cold_start(), full_cost, price)
    return {
        "expected_cost": round(expected_cost(mix, full_cost), 2),
        "price": price,
        "gross_margin": round(cached, 4),
        "margin_without_cache": round(uncached, 4),
        "points_from_caching": round((cached - uncached) * 100, 1),
        "hit_rate": round(mix.hit_rate, 4),
    }


# =========================================================================== #
# §11-§12 — blast radius, and why diversity is insurance
# =========================================================================== #
def expected_affected(n: int, b: int, failure_rate: float) -> float:
    """Cartridges hit per year, in expectation. **Independent of ``b``.**

    ``b * lambda * (n/b) = lambda * n``. More bases means more defects, each
    affecting proportionally fewer cartridges, and the product does not move.
    The intuitive argument for diversity — smaller blast radius — is therefore
    wrong on expectation, and saying otherwise in a design review is a mistake
    somebody will catch.
    """
    if b <= 0:
        raise ValueError("base count must be positive")
    return failure_rate * n


def expected_loss(n: int, b: int, failure_rate: float, gamma: float = 1.5) -> float:
    """``E[loss] = lambda * n^gamma * b^(1-gamma)`` under ``L(m) = m^gamma``.

    Decreasing in ``b`` only when ``gamma > 1``. That superlinearity is the
    entire justification for base diversity, and it is real: the reputational
    damage from "every customer's model is broken at once" genuinely exceeds
    ``n`` times the damage from one being broken.

    At ``gamma = 1`` the expression collapses to ``lambda * n`` and diversity
    buys exactly nothing.
    """
    if gamma < 1.0:
        raise ValueError("sublinear loss would make concentration preferable")
    if b <= 0:
        raise ValueError("base count must be positive")
    return failure_rate * (n ** gamma) * (b ** (1.0 - gamma))


def diversity_tradeoff(
    n: int, base_options: Sequence[int], failure_rate: float = 0.05,
    gamma: float = 1.5, base_mb: float = BASE_MB, adapter_mb: float = ADAPTER_MB,
) -> list[dict[str, Any]]:
    """The trade laid out: loss falls with ``b``, dedup falls with it too.

    Both move, in opposite directions, so the optimum is finite. This returns
    the rows rather than picking a winner, because the exchange rate between a
    storage megabyte and a unit of reputational loss is a business judgement and
    not something this module should invent.
    """
    rows = []
    for b in base_options:
        k = n / b
        rows.append({
            "bases": b,
            "cartridges_per_base": round(k, 2),
            "expected_affected": round(expected_affected(n, b, failure_rate), 2),
            "expected_loss": round(expected_loss(n, b, failure_rate, gamma), 2),
            "dedup_ratio": round(dedup_ratio(k, base_mb, adapter_mb), 2),
            "stored_gb": round((b * base_mb + n * adapter_mb) / 1000, 2),
        })
    return rows


__all__ = [
    "ADAPTER_MB", "BASE_MB", "COMPOSE_SHARE", "WARM_START_SHARE",
    "CacheMix", "DedupReport",
    "break_even_k", "dedup_ceiling", "dedup_ratio", "diversity_tradeoff",
    "expected_affected", "expected_cost", "expected_loss", "gross_margin",
    "margin_report",
]
