"""Assembly point: build a ready-to-use :class:`Orchestrator` from settings.

The demo, CLI and API all go through here so there is a single place that
decides which concrete components to wire together. Selection is driven by
``Settings`` (which itself comes from ``configs/default.yaml`` + env).
"""
from __future__ import annotations

from majestic.config import Settings, load_settings
from majestic.core.mock import MockReasoningCore
from majestic.core.reasoning_core import ReasoningCore
from majestic.experts.echo import EchoExpert
from majestic.experts.registry import ExpertRegistry
from majestic.experts.specialist import SpecialistExpert
from majestic.experts.tools import CodeExecTool, WebTool
from majestic.logging_utils import get_logger
from majestic.orchestrator import Orchestrator
from majestic.perception.encoders import Encoder, HashingTextEncoder
from majestic.retrieval.memory import InMemoryMemory
from majestic.retrieval.rag import VectorRAG
from majestic.retrieval.store import InMemoryVectorStore, VectorStore
from majestic.router.rule_router import RuleRouter
from majestic.verification.verifier import PipelineVerifier

logger = get_logger(__name__)


def build_core(
    settings: Settings, available_targets: tuple[str, ...] = ("echo",)
) -> ReasoningCore:
    """Select a reasoning core based on settings.

    ``core.model == "mock"`` (default) returns the offline mock core. Any other
    value is treated as an open HF instruct model id and loaded lazily; if the
    heavy dependencies are missing the loader falls back to the mock core so the
    system stays runnable.
    """
    model = settings.core.model
    if model == "mock":
        return MockReasoningCore(default_target="echo")

    # Real model path — imported lazily so the base install needs no ML deps.
    try:
        from majestic.core.hf_core import HFReasoningCore

        return HFReasoningCore(
            model_id=model,
            max_new_tokens=settings.core.max_new_tokens,
            temperature=settings.core.temperature,
            available_targets=available_targets,
        )
    except Exception as exc:  # noqa: BLE001 - fall back rather than crash
        logger.warning("Falling back to MockReasoningCore (%s): %s", model, exc)
        return MockReasoningCore(default_target="echo")


def build_encoder(settings: Settings) -> Encoder:
    """Select the text encoder. ``hash`` is offline; anything else is a model id."""
    name = settings.retrieval.embedding_model
    if name == "hash":
        return HashingTextEncoder()
    try:
        from majestic.perception.encoders import SentenceTransformerEncoder

        return SentenceTransformerEncoder(name)
    except Exception as exc:  # noqa: BLE001 - stay runnable offline
        logger.warning("Falling back to HashingTextEncoder (%s): %s", name, exc)
        return HashingTextEncoder()


def build_vector_store(settings: Settings, encoder: Encoder) -> VectorStore:
    """Select the vector store. ``memory`` (default) needs no external services."""
    kind = settings.retrieval.vector_store
    if kind == "faiss":
        try:
            from majestic.retrieval.store import FaissVectorStore

            dim = getattr(encoder, "dim", None)
            if dim is None:
                raise ValueError("faiss store needs a fixed-dim encoder")
            return FaissVectorStore(dim)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to InMemoryVectorStore: %s", exc)
    return InMemoryVectorStore()


def build_orchestrator(
    settings: Settings | None = None, knowledge: list[str] | None = None
) -> Orchestrator:
    """Construct the default compound system.

    ``knowledge`` optionally seeds the RAG store with grounding documents.
    """
    settings = settings or load_settings()

    experts = ExpertRegistry()
    experts.register(EchoExpert())
    experts.register(SpecialistExpert())
    experts.register(CodeExecTool())
    experts.register(WebTool())

    core = build_core(settings, available_targets=tuple(e.name for e in experts.all()))

    router = RuleRouter(
        experts,
        confidence_threshold=settings.routing.confidence_threshold,
        escalate_target="echo",
    )

    encoder = build_encoder(settings)
    store = build_vector_store(settings, encoder)
    rag = VectorRAG(encoder, store, top_k=settings.retrieval.top_k)
    if knowledge:
        rag.add_documents(knowledge)

    verifier = PipelineVerifier() if settings.verification.enabled else None
    memory = InMemoryMemory()

    return Orchestrator(
        core=core,
        router=router,
        experts=experts,
        rag=rag,
        verifier=verifier,
        encoder=encoder,
        memory=memory,
    )
