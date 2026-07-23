"""A reasoning core backed by a small open Hugging Face instruct model.

Heavy dependencies (``torch``/``transformers``) are imported lazily inside the
constructor so that importing this module never requires them. If loading fails
for any reason (deps missing, no network, out of memory) the caller
(:func:`majestic.factory.build_core`) falls back to the mock core.

The core never performs the task itself: :meth:`plan` asks the model to
decompose the request into steps, and :meth:`synthesize` asks it to write the
final answer from the step results and any retrieval grounding.
"""
from __future__ import annotations

import json
import re
from typing import Any

from majestic.core.reasoning_core import ReasoningCore
from majestic.logging_utils import get_logger
from majestic.types import Plan, Request, Step

logger = get_logger(__name__)

_PLAN_SYSTEM = (
    "You are a planning module. Decompose the user request into a short ordered "
    "list of steps. Reply with ONLY a JSON array of objects, each with keys "
    '"description" and "target". Valid targets: {targets}. Keep it minimal.'
)

_SYNTH_SYSTEM = (
    "You are a synthesis module. Using the step results and any grounding "
    "context, write a clear, correct final answer to the user's request. "
    "Do not mention the steps or that you are a module."
)


class HFReasoningCore(ReasoningCore):
    """Wraps a small ``transformers`` causal-LM instruct model.

    Parameters
    ----------
    model_id:
        A *small* open instruct model id (keep it tiny for modest hardware).
    max_new_tokens, temperature:
        Generation controls.
    available_targets:
        Targets the planner may route to. Defaults to a generic set; the factory
        can pass the live expert names.
    device:
        ``"cpu"`` (default), ``"cuda"``, or ``"auto"``.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        available_targets: tuple[str, ...] = ("echo", "web", "code_exec", "specialist"),
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.available_targets = available_targets
        self.device = device

        # Lazy, guarded import: keeps the module importable without torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading HF reasoning core %s on %s", model_id, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self._model.eval()

    # ------------------------------------------------------------------ #
    def _generate(self, system: str, user: str) -> str:
        """Run one chat turn and return the assistant's text."""
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 - model without a chat template
            prompt = f"{system}\n\nUser: {user}\nAssistant:"

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text.strip()

    # ------------------------------------------------------------------ #
    def plan(self, request: Request) -> Plan:
        """Ask the model for a JSON step list; fall back to a single echo step."""
        system = _PLAN_SYSTEM.format(targets=", ".join(self.available_targets))
        raw = self._generate(system, str(request.content))
        steps = _parse_steps(raw, self.available_targets, str(request.content))
        return Plan(steps=steps)

    def synthesize(
        self, request: Request, results: list[Any], grounding: list[str] | None = None
    ) -> str:
        """Write the final answer from step results + grounding."""
        blocks = []
        if grounding:
            blocks.append("Grounding:\n" + "\n".join(f"- {g}" for g in grounding))
        if results:
            blocks.append(
                "Step results:\n" + "\n".join(f"- {r}" for r in results)
            )
        user = f"Request: {request.content}\n\n" + "\n\n".join(blocks)
        return self._generate(_SYNTH_SYSTEM, user)


# ---------------------------------------------------------------------- #
def _parse_steps(
    raw: str, targets: tuple[str, ...], content: str
) -> list[Step]:
    """Best-effort parse of a JSON step array from model output.

    Robust to extra prose around the JSON. Falls back to a single echo step so
    the pipeline always has something to run.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            steps = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                target = str(item.get("target", "echo"))
                if target not in targets:
                    target = "echo"
                steps.append(
                    Step(
                        description=str(item.get("description", "")),
                        target=target,
                        args={"text": content},
                    )
                )
            if steps:
                return steps
        except json.JSONDecodeError:
            logger.debug("planner returned non-JSON; using fallback step")
    return [Step(description="handle request", target="echo", args={"text": content})]
