"""FABRIC — the typed DAG runtime (B-10).

Nodes are cartridges, tools and control flow. The graph is STATICALLY VERIFIED
for offline closure, RAM and cost before it ever runs.
"""
from majestic.fabric.analyser import (
    AnalysisResult,
    OfflineRewrite,
    analyse,
    critical_path,
    prove_offline,
    rewrite_offline,
    routing_investment,
    system_accuracy,
)
from majestic.fabric.capacity import (
    CapacityReport,
    Channel,
    analyse_capacity,
    bits_for_domain,
    bits_for_schema,
    classify,
    path_capacity,
    repeated_capacity,
)
from majestic.fabric.executor import (
    ExecutionResult,
    FabricRuntime,
    NodeTrace,
    OfflineViolation,
    TaintViolation,
    split_offline_core,
)
from majestic.fabric.graph import UNBOUNDED_BITS, FabricGraph, Node, NodeKind, TaintRole
from majestic.fabric.schedule import (
    SWAP_MS,
    PagingPlan,
    belady,
    compare,
    expected_swaps,
    grouped_order,
    lru,
    reference_string,
    schedule,
)

__all__ = [
    # §11-§12 — quantitative information flow, the strong result
    "CapacityReport", "Channel", "UNBOUNDED_BITS", "analyse_capacity",
    "bits_for_domain", "bits_for_schema", "classify", "path_capacity",
    "repeated_capacity",
    # §6 — offline closure by rewriting, not configuration
    "OfflineRewrite", "prove_offline", "rewrite_offline",
    # §14-§15 — latency and adapter paging
    "SWAP_MS", "PagingPlan", "belady", "compare", "critical_path",
    "expected_swaps", "grouped_order", "lru", "reference_string", "schedule",
    # §17 — where to invest
    "routing_investment", "system_accuracy",
    "TaintRole",
    "AnalysisResult",
    "ExecutionResult",
    "FabricGraph",
    "FabricRuntime",
    "Node",
    "NodeKind",
    "NodeTrace",
    "OfflineViolation",
    "TaintViolation",
    "analyse",
    "split_offline_core",
]
