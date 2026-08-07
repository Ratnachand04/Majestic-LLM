"""FORGE — from plain language to a typed specification (B-04).

The non-technical front door. It parses a free-text description into a partial
Spec IR, scores confidence per slot, detects ambiguity, and then asks ONLY the
slots left empty — ranked by information gain. Four questions, not forty.

The interview terminates because it fills a FIXED SCHEMA rather than chatting:
every slot carries a type and a validator, and a specification that cannot be
type-checked cannot be compiled.

Silent defaulting on an ambiguous slot is the top cause of a wrong build, so
ambiguity is surfaced as a question instead of being guessed. "Works offline" —
always, or only when the network drops? — is the canonical example.

The parser here is deterministic and offline (keyword and pattern rules). An LLM
front end (LLM role 1) can replace :meth:`Forge.parse` without changing anything
downstream, because the output contract is the Spec IR.

GAP-09 (open): eval research optimises correlation with expert raters, not
whether a pharmacy owner BELIEVES the number. Customer trust and acceptance are
first-class product metrics and are not yet measured here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from majestic.logging_utils import get_logger
from modelrig.ir import AbstentionPolicy, DataRights, SpecIR
from modelrig.primitives import TaskPrimitive, spec_for

logger = get_logger(__name__)

# Keyword rules mapping description text onto the closed primitive set.
_PRIMITIVE_RULES: list[tuple[TaskPrimitive, tuple[str, ...]]] = [
    (TaskPrimitive.EXTRACT,
     ("extract", "pull out", "read fields", "parse", "capture fields", "digitise",
      "digitize", "read . forms", "invoice", "receipt", "form")),
    (TaskPrimitive.CLASSIFY,
     ("classify", "categorise", "categorize", "label", "tag", "sentiment",
      "sort into", "bucket")),
    (TaskPrimitive.SUMMARISE, ("summarise", "summarize", "summary", "condense", "tl;dr")),
    (TaskPrimitive.REWRITE,
     ("translate", "rewrite", "rephrase", "reformat", "restyle", "convert to")),
    (TaskPrimitive.GENERATE, ("draft", "write a", "compose", "generate a reply", "author")),
    (TaskPrimitive.ANSWER,
     ("answer", "question", "q and a", "q&a", "ask about", "look up in", "faq")),
    (TaskPrimitive.ROUTE, ("triage", "route", "dispatch", "assign to", "prioritise", "urgency")),
    (TaskPrimitive.TOOLCALL, ("call the api", "tool call", "invoke", "trigger the")),
]

_LANG_RULES = {
    "hindi": "hi-IN", "hi-in": "hi-IN", "english": "en-IN", "tamil": "ta-IN",
    "telugu": "te-IN", "marathi": "mr-IN", "bengali": "bn-IN", "spanish": "es",
    "french": "fr", "german": "de", "arabic": "ar",
}

_DEVICE_RULES = {
    "tablet": "android_tablet_4gb", "android": "android_midrange",
    "phone": "android_midrange", "mobile": "android_midrange",
    "flagship": "android_flagship", "raspberry": "raspberry_pi4",
    "pi": "raspberry_pi4", "laptop": "laptop_cpu", "server": "laptop_gpu",
    "gpu": "laptop_gpu", "desktop": "laptop_cpu",
}

_OFFLINE_HINTS = ("offline", "no internet", "without internet", "internet dies",
                  "no network", "air-gapped", "airgapped", "poor connectivity")
_AMBIGUOUS_OFFLINE = ("when the internet dies", "if the network drops",
                      "poor connectivity", "when offline")

_ABSTAIN_HINTS = {
    AbstentionPolicy.FLAG: ("flag", "review", "human check", "do not guess", "don't guess"),
    AbstentionPolicy.ABSTAIN: ("refuse", "abstain", "say it doesn't know", "skip"),
    AbstentionPolicy.ESCALATE: ("escalate", "ask the cloud", "fall back to the cloud"),
}


@dataclass
class Slot:
    """One slot of the Spec IR schema, with the confidence FORGE has in it."""

    name: str
    value: Any = None
    confidence: float = 0.0
    question: str = ""
    information_gain: float = 0.0
    ambiguous: bool = False

    @property
    def filled(self) -> bool:
        return self.value is not None and self.confidence >= 0.6 and not self.ambiguous


@dataclass
class InterviewState:
    """Partial spec plus the questions still worth asking."""

    slots: dict[str, Slot] = field(default_factory=dict)
    raw_description: str = ""

    def unfilled(self) -> list[Slot]:
        """Slots still to ask about, highest information gain first."""
        return sorted(
            (s for s in self.slots.values() if not s.filled),
            key=lambda s: -s.information_gain,
        )

    def questions(self, limit: int = 4) -> list[str]:
        """The next few questions. Four, not forty."""
        return [s.question for s in self.unfilled()[:limit] if s.question]


# Information gain per slot: how much the answer narrows the build space.
_GAIN = {
    "task_primitive": 1.0, "device_target": 0.9, "offline_required": 0.85,
    "seed_data_ref": 0.8, "io_schema": 0.7, "quality_gate": 0.6,
    "abstention_policy": 0.55, "data_rights": 0.5, "latency_budget_ms": 0.45,
    "languages": 0.4, "policy_rules": 0.3, "budget_ceiling_usd": 0.2,
}

_QUESTIONS = {
    "task_primitive": "What should the model DO with the input — extract fields, "
                      "classify it, answer questions about it, or draft a reply?",
    "device_target": "Which device will run this? (Scan the QR probe and we will "
                     "detect RAM and chip automatically.)",
    "offline_required": "Must it work with no internet ALWAYS, or only survive "
                        "the network dropping occasionally?",
    "seed_data_ref": "Can you upload real examples? We need them to build and, "
                     "separately, to prove the result.",
    "io_schema": "What fields or labels must come out, exactly?",
    "quality_gate": "What accuracy would make this worth deploying?",
    "abstention_policy": "When the model is unsure, should it flag for review or "
                         "answer anyway?",
    "data_rights": "Is the example data yours to train on?",
    "latency_budget_ms": "How fast must a single result come back?",
    "languages": "Which languages must it handle?",
    "policy_rules": "Anything it must never say or do?",
    "budget_ceiling_usd": "What is your budget ceiling for this build?",
}


class Forge:
    """Slot-filling specification builder."""

    def __init__(self, default_device: str = "laptop_cpu") -> None:
        self.default_device = default_device

    # -- step 1: parse into a partial Spec IR ---------------------------- #
    def parse(self, description: str, **known: Any) -> InterviewState:
        """Extract every slot the description already determines."""
        text = description.lower()
        state = InterviewState(raw_description=description)

        def put(name: str, value: Any, confidence: float, ambiguous: bool = False) -> None:
            state.slots[name] = Slot(
                name=name, value=value, confidence=confidence,
                question=_QUESTIONS.get(name, ""),
                information_gain=_GAIN.get(name, 0.1), ambiguous=ambiguous,
            )

        for slot in _GAIN:
            put(slot, None, 0.0)

        # task primitive
        for primitive, keywords in _PRIMITIVE_RULES:
            if any(re.search(k, text) for k in keywords):
                put("task_primitive", primitive, 0.85)
                break

        # device
        for keyword, device in _DEVICE_RULES.items():
            if keyword in text:
                put("device_target", device, 0.7)
                break

        # offline — the canonical ambiguity
        if any(h in text for h in _OFFLINE_HINTS):
            ambiguous = any(h in text for h in _AMBIGUOUS_OFFLINE)
            put("offline_required", True, 0.5 if ambiguous else 0.9, ambiguous=ambiguous)

        # languages
        langs = [code for word, code in _LANG_RULES.items() if word in text]
        if langs:
            put("languages", sorted(set(langs)), 0.8)

        # latency
        m = re.search(r"(\d+(?:\.\d+)?)\s*(ms|millisecond|second|sec|s)\b", text)
        if m:
            value = float(m.group(1))
            ms = value if m.group(2).startswith("m") else value * 1000
            put("latency_budget_ms", int(ms), 0.8)

        # quality bar
        m = re.search(r"(\d{2,3})\s*(?:%|percent)", text)
        if m:
            put("quality_gate", min(float(m.group(1)) / 100.0, 1.0), 0.8)

        # seed data volume
        m = re.search(r"(\d+)\s+(?:real\s+)?(?:examples|forms|samples|documents|tickets|rows)",
                      text)
        if m:
            put("seed_data_count", int(m.group(1)), 0.9)

        # abstention
        for policy, hints in _ABSTAIN_HINTS.items():
            if any(h in text for h in hints):
                put("abstention_policy", policy, 0.8)
                break

        # caller-supplied answers always win
        for name, value in known.items():
            put(name, value, 1.0)

        return state

    # -- step 2 & 3: answer slots, then emit ----------------------------- #
    def answer(self, state: InterviewState, **answers: Any) -> InterviewState:
        """Refill slots from customer answers."""
        for name, value in answers.items():
            slot = state.slots.get(name) or Slot(name=name)
            slot.value = value
            slot.confidence = 1.0
            slot.ambiguous = False
            slot.question = _QUESTIONS.get(name, "")
            slot.information_gain = _GAIN.get(name, 0.1)
            state.slots[name] = slot
        return state

    def to_spec(self, state: InterviewState, **overrides: Any) -> SpecIR:
        """Emit a hash-addressed Spec IR from the filled slots.

        Raises ``ValueError`` when a slot with no safe default is still unfilled —
        FORGE refuses to guess rather than silently defaulting.
        """
        values = {n: s.value for n, s in state.slots.items() if s.filled}
        values.update(overrides)

        primitive = values.get("task_primitive")
        if primitive is None:
            raise ValueError(
                "task_primitive is unfilled: "
                + _QUESTIONS["task_primitive"]
            )
        if isinstance(primitive, str):
            primitive = TaskPrimitive(primitive.lower())

        prim = spec_for(primitive)
        spec = SpecIR(
            task_primitive=primitive,
            io_schema=values.get("io_schema") or {},
            languages=values.get("languages") or ["en"],
            device_target=values.get("device_target") or self.default_device,
            offline_required=bool(values.get("offline_required", False)),
            latency_budget_ms=values.get("latency_budget_ms"),
            quality_gate=float(values.get("quality_gate", 0.9)),
            seed_data_ref=values.get("seed_data_ref"),
            seed_data_count=int(values.get("seed_data_count", 0)),
            abstention_policy=values.get("abstention_policy", AbstentionPolicy.FLAG),
            policy_rules=values.get("policy_rules") or [],
            data_rights=values.get("data_rights", DataRights.UNKNOWN),
            budget_ceiling_usd=float(values.get("budget_ceiling_usd", 40.0)),
            jurisdiction=values.get("jurisdiction", "IN"),
            max_step_depth=int(values.get("max_step_depth", 1)),
            notes=state.raw_description[:500],
        )
        if not spec.io_schema:
            spec.io_schema = {"metric": prim.default_metric}
        logger.info(
            "forge: emitted spec %s for primitive %s", spec.hash[:8], primitive.value
        )
        return spec

    def interview(self, description: str, **answers: Any) -> tuple[SpecIR, list[str]]:
        """Convenience: parse, apply answers, and report what is still unknown."""
        state = self.parse(description)
        if answers:
            state = self.answer(state, **answers)
        remaining = state.questions()
        return self.to_spec(state), remaining
