"""Pluggable verification checks.

Each check declares whether it *applies* to a response and, if so, returns a
:class:`CheckResult`. The :class:`~majestic.verification.verifier.PipelineVerifier`
runs every applicable check and passes only if all of them pass.

Checks by type (from the design): schema -> validation, code -> execution,
math -> tools, facts -> retrieval. Reliability is a designed property.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from majestic.types import Response

# Common capitalized tokens to ignore when checking factual claims.
_STOPWORD_CAPS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "you", "i", "what", "who", "when", "where", "why", "how",
    "tell", "explain", "describe", "is", "are", "and", "or", "but", "in", "on",
    "of", "to", "about", "please", "summarize", "here",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str = ""


class Check(ABC):
    name: str = "check"

    @abstractmethod
    def applies(self, response: Response) -> bool:
        """Whether this check is relevant to the given response."""

    @abstractmethod
    def run(self, response: Response) -> CheckResult:
        """Evaluate the response; only called when :meth:`applies` is True."""


# --------------------------------------------------------------------------- #
class NonEmptyCheck(Check):
    """The response must carry non-empty content. Always applies."""

    name = "non_empty"

    def applies(self, response: Response) -> bool:
        return True

    def run(self, response: Response) -> CheckResult:
        content = response.content
        empty = content is None or (isinstance(content, str) and not content.strip())
        return CheckResult(self.name, not empty, "" if not empty else "empty content")


# --------------------------------------------------------------------------- #
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_arith(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression, or return None if not evaluable."""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(n.op)](_eval(n.operand))
        raise ValueError("disallowed expression")

    try:
        return _eval(node)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


class MathCheck(Check):
    """Catch wrong arithmetic like ``2 + 2 = 5`` by evaluating each equation."""

    name = "math"
    _EQ_RE = re.compile(r"([0-9][0-9+\-*/().\s]*?)\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)")

    def _equations(self, text: str) -> list[tuple[str, float]]:
        found = []
        for lhs, rhs in self._EQ_RE.findall(text):
            if any(op in lhs for op in "+-*/") and _safe_arith(lhs) is not None:
                found.append((lhs.strip(), float(rhs)))
        return found

    def applies(self, response: Response) -> bool:
        return isinstance(response.content, str) and bool(
            self._equations(response.content)
        )

    def run(self, response: Response) -> CheckResult:
        for lhs, rhs in self._equations(str(response.content)):
            value = _safe_arith(lhs)
            if value is not None and abs(value - rhs) > 1e-6:
                return CheckResult(
                    self.name, False, f"{lhs} = {value:g}, not {rhs:g}"
                )
        return CheckResult(self.name, True)


# --------------------------------------------------------------------------- #
class SchemaCheck(Check):
    """Validate that content is JSON with the required keys.

    Applies when ``response.metadata['schema']`` is a list of required keys.
    """

    name = "schema"

    def applies(self, response: Response) -> bool:
        return bool(response.metadata.get("schema"))

    def run(self, response: Response) -> CheckResult:
        required = response.metadata["schema"]
        content = response.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return CheckResult(self.name, False, "content is not valid JSON")
        if not isinstance(content, dict):
            return CheckResult(self.name, False, "content is not a JSON object")
        missing = [k for k in required if k not in content]
        if missing:
            return CheckResult(self.name, False, f"missing keys: {missing}")
        return CheckResult(self.name, True)


# --------------------------------------------------------------------------- #
class CodeExecutionCheck(Check):
    """Re-run code and confirm it produces the expected stdout.

    Applies when ``metadata['verify_code']`` is ``{"code": ..., "expected": ...}``.
    """

    name = "code_exec"

    def __init__(self, tool: Any | None = None) -> None:
        self._tool = tool

    def _get_tool(self) -> Any:
        if self._tool is not None:
            return self._tool
        from majestic.experts.tools import CodeExecTool

        return CodeExecTool()

    def applies(self, response: Response) -> bool:
        spec = response.metadata.get("verify_code")
        return isinstance(spec, dict) and "code" in spec

    def run(self, response: Response) -> CheckResult:
        spec = response.metadata["verify_code"]
        result = self._get_tool().run(code=spec["code"])
        if result.get("timed_out"):
            return CheckResult(self.name, False, "execution timed out")
        if result.get("returncode") not in (0, None):
            return CheckResult(self.name, False, result.get("stderr", "error"))
        expected = spec.get("expected")
        if expected is not None and result.get("stdout") != str(expected):
            return CheckResult(
                self.name, False, f"stdout {result.get('stdout')!r} != {expected!r}"
            )
        return CheckResult(self.name, True)


# --------------------------------------------------------------------------- #
class FactualityCheck(Check):
    """Heuristic: factual claims in the answer must be supported by grounding.

    Applies when ``metadata['grounding']`` is present. Extracts numbers and
    proper-noun-like tokens from the answer and fails if too large a fraction of
    them are absent from the grounding text. Heuristic, not an entailment model —
    it reliably catches swapped entities/numbers (Paris -> London, 4 -> 5).
    """

    name = "factuality"

    def __init__(self, tolerance: float = 0.34) -> None:
        self.tolerance = tolerance

    def applies(self, response: Response) -> bool:
        return bool(response.metadata.get("grounding"))

    def _claims(self, text: str) -> list[str]:
        proper = [
            w
            for w in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text)
            if w.lower() not in _STOPWORD_CAPS
        ]
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        return proper + numbers

    def run(self, response: Response) -> CheckResult:
        grounding = " ".join(response.metadata.get("grounding", [])).lower()
        claims = self._claims(str(response.content))
        if not claims:
            return CheckResult(self.name, True, "no checkable claims")
        unsupported = [c for c in claims if c.lower() not in grounding]
        ratio = len(unsupported) / len(claims)
        if ratio > self.tolerance:
            return CheckResult(
                self.name, False, f"unsupported claims: {sorted(set(unsupported))}"
            )
        return CheckResult(self.name, True)
