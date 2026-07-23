"""A minimal verifier for Phase 0.

Passes when the response carries non-empty content. Phase 4 replaces this with a
:class:`~majestic.verification.checks` pipeline (schema, code-exec, factuality,
math) that can actually catch wrong answers.
"""
from __future__ import annotations

from majestic.types import Response
from majestic.verification.verifier import Verifier


class BasicVerifier(Verifier):
    """Trivial non-emptiness check."""

    def verify(self, response: Response) -> bool:
        content = response.content
        if content is None:
            return False
        if isinstance(content, str) and not content.strip():
            return False
        return True
