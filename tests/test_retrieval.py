"""Tests for vector store, memory, and the RAG pipeline (all offline)."""
from __future__ import annotations

from majestic.factory import build_orchestrator
from majestic.perception.encoders import HashingTextEncoder
from majestic.retrieval.memory import InMemoryMemory
from majestic.retrieval.rag import VectorRAG
from majestic.retrieval.store import InMemoryVectorStore
from majestic.types import Request

_KB = [
    "The capital of France is Paris.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Majestic LLM is a compound AI system with a reasoning core and a router.",
]


def _rag() -> VectorRAG:
    rag = VectorRAG(HashingTextEncoder(dim=512), InMemoryVectorStore(), top_k=2)
    rag.add_documents(_KB)
    return rag


def test_vector_store_ranks_by_similarity():
    store = InMemoryVectorStore()
    enc = HashingTextEncoder(dim=256)
    store.add("a", enc.encode("cats and dogs are pets"), "pets")
    store.add("b", enc.encode("stock market finance economy"), "finance")
    hits = store.search_scored(enc.encode("my pet cat"), top_k=2)
    assert hits[0][1] == "pets"
    assert hits[0][0] >= hits[1][0]


def test_vector_store_empty_returns_nothing():
    assert InMemoryVectorStore().search([0.0] * 8) == []


def test_rag_retrieves_relevant_document():
    rag = _rag()
    ctx = rag.retrieve(Request(content="What is the capital of France?"))
    assert any("Paris" in c for c in ctx)


def test_rag_filters_by_min_score():
    rag = _rag()
    rag.min_score = 0.99  # nothing should clear this
    assert rag.retrieve(Request(content="capital of France")) == []


def test_memory_write_and_recall():
    mem = InMemoryMemory()
    mem.write("user asked about Paris")
    mem.write("user asked about photosynthesis")
    assert mem.recall("paris") == ["user asked about Paris"]
    assert len(mem.all()) == 2


def test_orchestrator_uses_grounding_in_answer():
    orch = build_orchestrator(knowledge=_KB)
    resp = orch.handle(Request(content="Tell me about Majestic LLM reasoning core"))
    assert "grounded on" in resp.content
    assert any("run:echo" in t or t.startswith("retrieve") for t in resp.trace)
    assert any(t.startswith("retrieve") for t in resp.trace)
