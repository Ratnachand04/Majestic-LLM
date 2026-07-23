"""Verifier: checks a response before it is returned.

Checks by type: facts -> retrieval, code -> execution, math -> tools,
schema -> validation. Reliability is a designed property, not a by-product.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from majestic.logging_utils import get_logger
from majestic.types import Response

logger = get_logger(__name__)


class Verifier(ABC):
    @abstractmethod
    def verify(self, response: Response) -> bool:
        """Return True iff the response passes all applicable checks."""
        raise NotImplementedError


class PipelineVerifier(Verifier):
    """Runs a list of pluggable checks; passes only if all applicable ones pass.

    Per-check outcomes are recorded on ``response.metadata['verification']`` so a
    caller can see *why* a response was (un)verified.
    """

    def __init__(self, checks: list | None = None) -> None:
        self.checks = checks if checks is not None else self._default_checks()

    @staticmethod
    def _default_checks() -> list:
        from majestic.verification.checks import (
            CodeExecutionCheck,
            FactualityCheck,
            MathCheck,
            NonEmptyCheck,
            SchemaCheck,
        )

        return [
            NonEmptyCheck(),
            SchemaCheck(),
            MathCheck(),
            CodeExecutionCheck(),
            FactualityCheck(),
        ]

    def verify(self, response: Response) -> bool:
        results = []
        passed = True
        for check in self.checks:
            try:
                if not check.applies(response):
                    continue
                result = check.run(response)
            except Exception as exc:  # noqa: BLE001 - a broken check must not crash us
                logger.warning("check %r raised: %s", getattr(check, "name", check), exc)
                continue
            results.append(result)
            if not result.passed:
                passed = False
                logger.info("verification failed [%s]: %s", result.name, result.reason)
        response.metadata["verification"] = [
            {"name": r.name, "passed": r.passed, "reason": r.reason} for r in results
        ]
        return passed
