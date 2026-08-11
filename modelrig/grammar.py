"""Compiled grammars — slot 4 of the cartridge (B-08).

    "Prompt template plus a compiled grammar that makes schema violation
     impossible."

The distinction matters. A prompt that *asks* for JSON produces invalid JSON some
of the time; a grammar that constrains decoding cannot. For a small specialist
this is the difference between a parseable output contract and a downstream
integration that fails unpredictably — and it is why A-09 says Majestic trains
cartridges to emit a CONSTRAINED tool-call schema rather than free-form
reasoning, since sub-2B models degrade badly across free-form multi-step loops.

This module compiles a Spec IR's ``io_schema`` into a GBNF grammar (the format
llama.cpp consumes) and provides an offline validator that enforces the same
contract without a model, so the pipeline can prove conformance in tests.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from modelrig.primitives import TaskPrimitive

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# GBNF primitives shared by every generated grammar.
_PREAMBLE = r'''
ws      ::= [ \t\n]*
string  ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""
number  ::= "-"? [0-9]+ ("." [0-9]+)?
boolean ::= "true" | "false"
null    ::= "null"
'''.strip()


@dataclass
class Grammar:
    """A compiled output contract."""

    name: str
    gbnf: str
    required_keys: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()
    kind: str = "object"          # object | enum | text
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, output: str) -> tuple[bool, str]:
        """Check an output against the contract. Returns ``(ok, reason)``.

        This is the same contract the grammar enforces during decoding, applied
        after the fact so the Proving Ground can verify conformance offline.
        """
        text = output.strip()
        if self.kind == "enum":
            if text in self.enum_values:
                return True, ""
            return False, f"{text!r} is not one of {list(self.enum_values)}"

        if self.kind == "object":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                return False, f"not valid JSON: {exc.msg}"
            if not isinstance(parsed, dict):
                return False, "output is not a JSON object"
            missing = [k for k in self.required_keys if k not in parsed]
            if missing:
                return False, f"missing required keys: {missing}"
            return True, ""

        return (True, "") if text else (False, "empty output")


def _sanitise(name: str) -> str:
    """GBNF rule names must be identifiers."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return cleaned if _IDENT_RE.match(cleaned) else f"f_{cleaned}"


def compile_object_grammar(
    fields: dict[str, str], name: str = "output"
) -> Grammar:
    """Compile ``{field: type}`` into a GBNF grammar for a JSON object.

    Every declared field is REQUIRED and emitted in a fixed order. Fixing the
    order is deliberate: it removes a whole class of decoding ambiguity and makes
    the output byte-comparable across runs.
    """
    if not fields:
        raise ValueError("an object grammar needs at least one field")

    type_rule = {
        "string": "string", "str": "string", "text": "string",
        "number": "number", "int": "number", "integer": "number", "float": "number",
        "boolean": "boolean", "bool": "boolean",
    }

    parts: list[str] = []
    for i, (key, type_name) in enumerate(fields.items()):
        rule = type_rule.get(str(type_name).lower(), "string")
        sep = '"," ws ' if i else ""
        parts.append(f'{sep}"\\"{key}\\"" ws ":" ws {rule} ws')

    root = 'root ::= "{" ws ' + " ".join(parts) + ' "}"'
    gbnf = f"{root}\n\n{_PREAMBLE}"
    return Grammar(
        name=name,
        gbnf=gbnf,
        required_keys=tuple(fields),
        kind="object",
        metadata={"fields": dict(fields)},
    )


def compile_enum_grammar(labels: list[str], name: str = "label") -> Grammar:
    """Compile a closed label set. The model cannot emit anything else."""
    if not labels:
        raise ValueError("an enum grammar needs at least one label")
    alternatives = " | ".join(f'"{label}"' for label in labels)
    return Grammar(
        name=name,
        gbnf=f"root ::= {alternatives}",
        enum_values=tuple(labels),
        kind="enum",
        metadata={"labels": list(labels)},
    )


def compile_toolcall_grammar(tools: list[dict[str, Any]], name: str = "toolcall") -> Grammar:
    """Compile a constrained tool-call schema (A-09).

    Small models degrade badly across free-form multi-step loops, so a cartridge
    emits a *constrained call*, never free-form reasoning. The grammar admits
    exactly the registered tool names and an argument object.
    """
    if not tools:
        raise ValueError("a tool-call grammar needs at least one tool")
    names = " | ".join(f'"\\"{t["name"]}\\""' for t in tools)
    root = (
        'root ::= "{" ws "\\"tool\\"" ws ":" ws toolname ws "," ws '
        '"\\"args\\"" ws ":" ws object ws "}"'
    )
    gbnf = "\n".join([
        root,
        f"toolname ::= {names}",
        'object ::= "{" ws ( pair ( ws "," ws pair )* )? ws "}"',
        'pair ::= string ws ":" ws value',
        'value ::= string | number | boolean | null',
        "",
        _PREAMBLE,
    ])
    return Grammar(
        name=name,
        gbnf=gbnf,
        required_keys=("tool", "args"),
        kind="object",
        metadata={"tools": [t["name"] for t in tools]},
    )


def compile_for_spec(
    primitive: TaskPrimitive | str,
    io_schema: dict[str, Any] | None = None,
) -> Grammar | None:
    """Compile the grammar a Spec IR implies, or ``None`` for free-text output.

    ``io_schema`` may carry ``fields`` (for extraction), ``labels`` (for
    classification and routing) or ``tools`` (for tool calls).
    """
    io_schema = io_schema or {}
    # TaskPrimitive mixes in str, so an isinstance(str) check matches members too
    # and str(member) yields "TaskPrimitive.CLASSIFY". Test for the enum first.
    prim = primitive if isinstance(primitive, TaskPrimitive) else TaskPrimitive(
        str(primitive).lower()
    )

    if prim is TaskPrimitive.TOOLCALL:
        tools = io_schema.get("tools")
        return compile_toolcall_grammar(tools) if tools else None

    if prim in (TaskPrimitive.CLASSIFY, TaskPrimitive.ROUTE):
        labels = io_schema.get("labels")
        return compile_enum_grammar(list(labels), name=prim.value) if labels else None

    if prim is TaskPrimitive.EXTRACT:
        fields = io_schema.get("fields")
        return compile_object_grammar(dict(fields), name="extract") if fields else None

    # summarise / rewrite / generate / answer are free text by contract.
    return None
