"""The orchestrator: the end-to-end request lifecycle for Majestic LLM.

    encode -> retrieve -> plan -> route -> execute -> verify -> respond

The orchestrator wires the components together and records a trace of the path
taken. Encoder, RAG, memory and verifier are optional; when omitted the flow
degrades gracefully (no encoding, no grounding, no verification gate).
"""
from __future__ import annotations

from typing import Any, Optional

from majestic.core.reasoning_core import ReasoningCore
from majestic.experts.registry import ExpertRegistry
from majestic.logging_utils import get_logger
from majestic.perception.encoders import Encoder
from majestic.retrieval.rag import RAGPipeline
from majestic.retrieval.store import Memory
from majestic.router.router import Router
from majestic.types import Request, Response
from majestic.verification.verifier import Verifier

logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        core: ReasoningCore,
        router: Router,
        experts: ExpertRegistry,
        rag: Optional[RAGPipeline] = None,
        verifier: Optional[Verifier] = None,
        encoder: Optional[Encoder] = None,
        memory: Optional[Memory] = None,
    ) -> None:
        self.core = core
        self.router = router
        self.experts = experts
        self.rag = rag
        self.verifier = verifier
        self.encoder = encoder
        self.memory = memory

    def handle(self, request: Request) -> Response:
        """Run one request through the compound system."""
        trace: list[str] = []

        # 1. encode (perception -> shared bus). Optional; failures are non-fatal.
        embedding: Optional[list[float]] = None
        if self.encoder is not None:
            try:
                embedding = self.encoder.encode(request.content)
                trace.append(f"encode:{request.modality.value}")
            except NotImplementedError:
                logger.debug("encoder not implemented for %s", request.modality)

        # 2. retrieve grounding context.
        grounding: list[str] = []
        if self.rag is not None:
            grounding = self.rag.retrieve(request)
            if grounding:
                trace.append(f"retrieve:{len(grounding)}")

        # 3. plan (decompose into steps).
        plan = self.core.plan(request)
        trace.append(f"plan:{len(plan.steps)}")

        # 4. route + execute each step.
        results: list[Any] = []
        for step in plan.steps:
            decision = self.router.route(step)
            target = decision.target
            if decision.escalate:
                trace.append(f"escalate:{step.target}")
            try:
                expert = self.experts.get(target)
            except KeyError:
                logger.warning("no expert registered for target %r; skipping", target)
                trace.append(f"missing:{target}")
                continue
            try:
                result = expert.run(**step.args)
            except NotImplementedError:
                logger.warning("expert %r is a stub; skipping step", target)
                trace.append(f"stub:{target}")
                continue
            results.append(result)
            trace.append(f"run:{target}")

        # 5. synthesize final answer.
        answer = self.core.synthesize(request, results, grounding=grounding)

        # 6. verify before returning.
        response = Response(
            content=answer,
            trace=trace,
            metadata={
                "embedding_dim": len(embedding) if embedding else 0,
                "grounding": grounding,
            },
        )
        if self.verifier is not None:
            response.verified = self.verifier.verify(response)

        # 7. persist to memory (optional).
        if self.memory is not None:
            self.memory.write({"request": request.content, "response": answer})

        return response
