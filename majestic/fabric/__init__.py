"""FABRIC — the typed DAG runtime (B-10).

Nodes are cartridges, tools and control flow. The graph is STATICALLY VERIFIED
for offline closure, RAM and cost before it ever runs.
"""
from majestic.fabric.analyser import AnalysisResult, analyse
from majestic.fabric.executor import (
    ExecutionResult,
    FabricRuntime,
    NodeTrace,
    OfflineViolation,
    TaintViolation,
    split_offline_core,
)
from majestic.fabric.graph import FabricGraph, Node, NodeKind

__all__ = [
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
