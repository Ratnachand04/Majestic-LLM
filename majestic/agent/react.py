"""The ReAct loop, constrained for small models (A-09).

    reason -> act -> observe -> repeat until done

Gorilla (2305.15334) is the clearest proof of the Majestic thesis inside the tool
domain: a fine-tuned 7B model beat far larger general models at correct API
invocation. Specialisation wins where the task is narrow and well defined.

But there is a hard capability ceiling. **Sub-2B models degrade badly across
multi-step ReAct loops**, so Majestic makes two changes to the textbook design:

**Cartridges emit a CONSTRAINED tool-call schema, not free-form reasoning.**
Every action is a grammar-validated ``{"tool": ..., "args": {...}}`` object. A
model that cannot express a malformed call cannot make one.

**Step depth is capped by the device tier.** The loop refuses to exceed the depth
the planner admitted, rather than wandering until a timeout.

Security: retrieved and scraped content is untrusted INPUT (2302.12173). Tools
declare ``produces_untrusted`` and ``privileged``, and the loop refuses to pass
tainted observations into a privileged tool — the same rule the Fabric analyser
proves statically, enforced here at runtime for dynamically chosen calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class StepLimitExceeded(RuntimeError):
    """The loop hit the step depth the device tier can sustain."""


@dataclass
class ToolSpec:
    """A registered tool and its trust properties."""

    name: str
    handler: Callable[..., Any]
    description: str = ""
    args: dict[str, str] = field(default_factory=dict)
    requires_network: bool = False
    produces_untrusted: bool = False
    privileged: bool = False

    def to_manifest(self) -> dict[str, Any]:
        """The entry that lands in cartridge slot 3 (tool bindings)."""
        return {
            "name": self.name,
            "scopes": tuple(self.args),
            "requires_network": self.requires_network,
            "trusts_untrusted_input": self.produces_untrusted,
            "privileged": self.privileged,
        }


@dataclass
class ToolCall:
    """One constrained call emitted by a cartridge."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """One reason/act/observe turn."""

    thought: str = ""
    call: Optional[ToolCall] = None
    observation: Any = None
    tainted: bool = False
    error: str = ""


class ToolRegistry:
    """Where side effects live."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> ToolSpec:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unregistered tool {name!r}; registered: {list(self._tools)}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def manifest(self) -> list[dict[str, Any]]:
        return [t.to_manifest() for t in self._tools.values()]

    def offline_safe(self) -> bool:
        return not any(t.requires_network for t in self._tools.values())


def parse_tool_call(raw: str, registry: ToolRegistry | None = None) -> ToolCall:
    """Parse a constrained tool call. Raises ``ValueError`` on anything else.

    Free-form prose is rejected outright rather than salvaged: a cartridge that
    was trained to emit a grammar-constrained object and did not is a defect to
    surface, not to paper over.
    """
    match = _JSON_RE.search(raw or "")
    if not match:
        raise ValueError("no tool-call object found; free-form output is not accepted")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool call is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict) or "tool" not in data:
        raise ValueError("tool call must be an object with a 'tool' key")

    name = str(data["tool"])
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("'args' must be an object")
    if registry is not None and not registry.has(name):
        raise ValueError(f"unregistered tool {name!r}")
    return ToolCall(tool=name, args=args)


@dataclass
class ReActResult:
    answer: Any = None
    steps: list[Step] = field(default_factory=list)
    completed: bool = False
    refusal: str = ""

    @property
    def depth(self) -> int:
        return len(self.steps)


class ReActLoop:
    """Reason, act, observe — bounded by what the device tier can sustain."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: Callable[[Any, list[Step]], str],
        max_steps: int = 3,
        offline: bool = False,
    ) -> None:
        self.registry = registry
        self.policy = policy      # returns a constrained tool call, or a final answer
        self.max_steps = max_steps
        self.offline = offline

    def run(self, request: Any) -> ReActResult:
        result = ReActResult()

        for _ in range(self.max_steps):
            raw = self.policy(request, result.steps)

            # A terminal answer ends the loop.
            if isinstance(raw, str) and raw.startswith("ANSWER:"):
                result.answer = raw[len("ANSWER:"):].strip()
                result.completed = True
                return result

            step = Step(thought=str(raw)[:200])
            try:
                call = parse_tool_call(str(raw), self.registry)
            except ValueError as exc:
                step.error = str(exc)
                result.steps.append(step)
                result.refusal = f"malformed tool call: {exc}"
                logger.warning("react: %s", result.refusal)
                return result

            step.call = call
            tool = self.registry.get(call.tool)

            if self.offline and tool.requires_network:
                step.error = "offline: tool requires the network"
                result.steps.append(step)
                result.refusal = (
                    f"tool {tool.name!r} requires the network in an offline deployment"
                )
                logger.warning("react: %s", result.refusal)
                return result

            incoming_taint = any(s.tainted for s in result.steps)
            if tool.privileged and incoming_taint:
                step.error = "refused: untrusted content would reach a privileged tool"
                result.steps.append(step)
                result.refusal = (
                    f"untrusted observation would reach privileged tool {tool.name!r} "
                    "(indirect prompt injection, 2302.12173)"
                )
                logger.warning("react: %s", result.refusal)
                return result

            try:
                step.observation = tool.handler(**call.args)
            except Exception as exc:  # noqa: BLE001 - report, do not crash the loop
                step.error = str(exc)
                result.steps.append(step)
                result.refusal = f"tool {tool.name!r} raised: {exc}"
                return result

            step.tainted = incoming_taint or tool.produces_untrusted
            result.steps.append(step)

        result.refusal = (
            f"step depth {self.max_steps} exhausted without an answer; sub-2B models "
            "degrade badly across longer loops, so the cap is deliberate (A-09)"
        )
        logger.info("react: %s", result.refusal)
        return result
