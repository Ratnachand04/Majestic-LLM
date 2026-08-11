"""Tests for the runtime topology (A-04, B-09)."""
from __future__ import annotations

from majestic.serving import (
    AdapterHandle,
    AdapterPool,
    ServerTopology,
    device_budget,
    plan_device_deployment,
    swap_latency_ms,
)


def _adapters(n: int, base: str = "Qwen/Qwen3-1.7B", tier: str = "standard"):
    return [AdapterHandle(f"c{i}", base, tier=tier) for i in range(n)]


# --- the adapter pool ------------------------------------------------------ #
def test_pool_pages_adapters_and_reports_hits():
    pool = AdapterPool(capacity_mb=1000)
    a = AdapterHandle("c1", "base")
    assert pool.acquire(a) is False     # miss: paged in
    assert pool.acquire(a) is True      # hit: already resident
    assert pool.hit_rate == 0.5


def test_pool_evicts_when_capacity_is_exhausted():
    pool = AdapterPool(capacity_mb=100)   # room for ~3 adapters at 30 MB
    for adapter in _adapters(6):
        pool.acquire(adapter)
    assert pool.evictions > 0
    assert pool.resident_mb <= 100


def test_eviction_is_tier_aware():
    """A free-tier adapter is paged out before a premium one."""
    pool = AdapterPool(capacity_mb=70)
    pool.acquire(AdapterHandle("premium", "base", tier="premium"))
    pool.acquire(AdapterHandle("free", "base", tier="free"))
    pool.acquire(AdapterHandle("new", "base", tier="standard"))
    assert "premium" in pool.resident_ids
    assert "free" not in pool.resident_ids


def test_fragmentation_report_makes_the_narrow_catalogue_argument_concrete():
    """Customers on different bases cannot share a GPU (A-04)."""
    pool = AdapterPool(capacity_mb=1000)
    for a in _adapters(3, base="Qwen/Qwen3-1.7B"):
        pool.acquire(a)
    assert pool.fragmentation_report()["shareable"] is True

    pool.acquire(AdapterHandle("other", "meta-llama/Llama-3.2-1B-Instruct"))
    report = pool.fragmentation_report()
    assert report["shareable"] is False
    assert report["distinct_bases"] == 2
    assert "cannot share a GPU" in report["note"]


# --- server tier ----------------------------------------------------------- #
def test_many_tenants_share_one_resident_base():
    topo = ServerTopology(base_ref="Qwen/Qwen3-1.7B")
    assert topo.max_tenants() > 100
    assert topo.unit_economics()["measured"] is False


def test_batch_groups_same_base_and_rejects_others():
    topo = ServerTopology(base_ref="Qwen/Qwen3-1.7B")
    requests = _adapters(3) + [AdapterHandle("x", "meta-llama/Llama-3.2-1B-Instruct")]
    out = topo.batch(requests)
    assert out["batched"] == 3
    assert out["rejected_wrong_base"] == 1


# --- device tier: the B-09 budget table ----------------------------------- #
def test_device_budget_reproduces_the_b09_table():
    budget = device_budget(total_gb=4.0, base_params_b=1.7, context_length=2048,
                           n_adapters=20)
    labels = {line.label for line in budget.lines}
    assert any("Base model" in label for label in labels)
    assert any("KV cache" in label for label in labels)
    assert any("adapters" in label for label in labels)
    # 1.7B at 4-bit is ~1.1 GB, and the whole table must fit a 4 GB tablet.
    weights = next(line for line in budget.lines if "Base model" in line.label)
    assert 0.9 <= weights.gb <= 1.2
    assert budget.fits is True
    assert budget.committed_gb <= 4.0


def test_twenty_adapters_are_cheap_because_they_share_the_base():
    one = device_budget(n_adapters=1).committed_gb
    twenty = device_budget(n_adapters=20).committed_gb
    # Nineteen extra specialists cost only their deltas, not nineteen bases.
    assert twenty - one < 0.7


def test_longer_context_can_break_the_budget():
    """KV cache grows linearly with context — the term nobody budgets for."""
    ok, reason = plan_device_deployment(1.7, total_gb=4.0, context_length=2048)
    assert ok.fits is True and reason is None
    over, reason = plan_device_deployment(1.7, total_gb=4.0, context_length=65536)
    assert over.fits is False
    assert "over budget" in reason


def test_bigger_base_breaks_a_small_device():
    over, reason = plan_device_deployment(8.0, total_gb=4.0)
    assert over.fits is False
    assert "Base model" in reason


def test_swap_latency_is_flagged_unmeasured():
    """GAP-10: this is the central unvalidated product promise."""
    out = swap_latency_ms(20)
    assert out["measured"] is False
    assert "assumption" in out["caveat"]


def test_budget_is_flagged_unmeasured():
    assert device_budget().measured is False
