"""Review item 4 — how robust are the artifact integrity check and Fabric's
typed-workflow validation, and what are they *not*?

Three parts, and the third is the one the review asks for by name.

**A. Malformed artifacts.** A real build directory is copied and then damaged
one way at a time. Each mutation declares whether ``verify_manifest`` should
reject it. Silent acceptance of a damaged artifact is the failure that matters:
a model that loads and answers is far worse than one that refuses to load.

**B. Unsafe data flow.** Graphs with planted violations, going beyond the
straight chains already reported: branches that rejoin, a confirm node on one
arm only, capacity-constrained mediators, cycles, orphaned sinks, several
untrusted sources. Each declares its expected verdict.

**C. The boundary.** Neither mechanism is a sandbox, and the difference is
demonstrated rather than asserted. The manifest establishes integrity
*relative to itself*, so an attacker who can rewrite the artifact can also
rewrite the manifest; the check catches damage and accident, not an adversary
with write access. Fabric reasons over *declared* node effects, so a handler
that lies about what it does is outside what any static analysis of the
declaration can see. Both are stated as measured demonstrations below.

    python -m evals.integrity_robustness
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from majestic.fabric.analyser import analyse  # noqa: E402
from majestic.fabric.graph import FabricGraph, Node, NodeKind  # noqa: E402
from modelrig.integrity import (  # noqa: E402
    digest_file,
    verify_manifest,
    write_manifest,
)

OUT = Path("outputs/evals/integrity_robustness.json")
REJECT, ACCEPT = "reject", "accept"


# =========================================================================== #
# A. Malformed artifacts
# =========================================================================== #
@dataclass
class Mutation:
    name: str
    why: str
    expect: str                       # REJECT | ACCEPT
    apply: Callable[[Path], None]


def _any_payload(root: Path) -> Path:
    """A content file the manifest covers (never the manifest itself)."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "checksums.json" and p.stat().st_size > 64:
            return p
    raise RuntimeError("no payload file in the fixture")


def _flip_a_byte(root: Path) -> None:
    p = _any_payload(root)
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    p.write_bytes(bytes(raw))


def _truncate(root: Path) -> None:
    p = _any_payload(root)
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])


def _empty_a_file(root: Path) -> None:
    _any_payload(root).write_bytes(b"")


def _delete_a_file(root: Path) -> None:
    _any_payload(root).unlink()


def _add_a_file(root: Path) -> None:
    (root / "unexpected.bin").write_bytes(b"not in the manifest")


def _rename_a_file(root: Path) -> None:
    p = _any_payload(root)
    p.rename(p.with_suffix(p.suffix + ".moved"))


def _drop_manifest_entry(root: Path) -> None:
    mf = root / "checksums.json"
    files = json.loads(mf.read_text(encoding="utf-8"))
    files.pop(next(iter(files)))
    mf.write_text(json.dumps(files, indent=2), encoding="utf-8")


def _empty_manifest(root: Path) -> None:
    (root / "checksums.json").write_text("{}", encoding="utf-8")


def _remove_manifest(root: Path) -> None:
    (root / "checksums.json").unlink()


def _corrupt_manifest_json(root: Path) -> None:
    (root / "checksums.json").write_text("{not json", encoding="utf-8")


def _manifest_traversal_entry(root: Path) -> None:
    mf = root / "checksums.json"
    files = json.loads(mf.read_text(encoding="utf-8"))
    files["../../escaped.bin"] = "0" * 64
    mf.write_text(json.dumps(files, indent=2), encoding="utf-8")


def _absolute_path_entry(root: Path) -> None:
    mf = root / "checksums.json"
    files = json.loads(mf.read_text(encoding="utf-8"))
    files["/etc/passwd"] = "0" * 64
    mf.write_text(json.dumps(files, indent=2), encoding="utf-8")


def _zero_hash(root: Path) -> None:
    mf = root / "checksums.json"
    files = json.loads(mf.read_text(encoding="utf-8"))
    files[next(iter(files))] = "0" * 64
    mf.write_text(json.dumps(files, indent=2), encoding="utf-8")


def _consistent_forgery(root: Path) -> None:
    """Tamper with the payload AND re-stamp the manifest to match.

    This is the adversary the mechanism does not address, and the case exists
    to show that plainly rather than to leave the reader guessing.
    """
    p = _any_payload(root)
    p.write_bytes(b"replaced payload, wholly different model")
    mf = root / "checksums.json"
    files = json.loads(mf.read_text(encoding="utf-8"))
    files[p.relative_to(root).as_posix()] = digest_file(p)
    mf.write_text(json.dumps(files, indent=2), encoding="utf-8")


def _untouched(root: Path) -> None:
    return None


MUTATIONS = [
    Mutation("untouched", "the control: an intact artifact must verify",
             ACCEPT, _untouched),
    Mutation("single-byte-flip", "one bit of a weight file changed",
             REJECT, _flip_a_byte),
    Mutation("truncated-file", "a half-written download",
             REJECT, _truncate),
    Mutation("zero-length-file", "a file emptied in place",
             REJECT, _empty_a_file),
    Mutation("deleted-file", "a file the manifest lists is gone",
             REJECT, _delete_a_file),
    Mutation("extra-file", "a file nobody declared appears beside the model",
             REJECT, _add_a_file),
    Mutation("renamed-file", "same bytes, different path",
             REJECT, _rename_a_file),
    Mutation("manifest-entry-dropped", "the manifest no longer covers a file",
             REJECT, _drop_manifest_entry),
    Mutation("manifest-emptied", "an empty manifest proves nothing",
             REJECT, _empty_manifest),
    Mutation("manifest-missing", "no manifest at all",
             REJECT, _remove_manifest),
    Mutation("manifest-not-json", "a corrupted manifest",
             REJECT, _corrupt_manifest_json),
    Mutation("manifest-path-traversal", "an entry escaping the artifact root",
             REJECT, _manifest_traversal_entry),
    Mutation("manifest-absolute-path", "an entry naming an absolute path",
             REJECT, _absolute_path_entry),
    Mutation("manifest-zero-hash", "an entry whose digest cannot match",
             REJECT, _zero_hash),
    Mutation("consistent-forgery", "payload AND manifest rewritten together",
             ACCEPT, _consistent_forgery),
]


def _fixture(dest: Path) -> Path:
    """A small artifact directory with a real manifest over it."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps({"arch": "demo"}), encoding="utf-8")
    (dest / "model.bin").write_bytes(bytes(range(256)) * 32)
    (dest / "tokenizer").mkdir(exist_ok=True)
    (dest / "tokenizer" / "vocab.txt").write_text("a\nb\nc\n" * 64, encoding="utf-8")
    write_manifest(dest)
    return dest


def run_integrity() -> list[dict[str, Any]]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        pristine = _fixture(Path(tmp) / "pristine")
        for m in MUTATIONS:
            work = Path(tmp) / f"case-{m.name}"
            shutil.copytree(pristine, work)
            m.apply(work)
            try:
                verify_manifest(work)
                observed, detail = ACCEPT, ""
            except Exception as exc:                        # noqa: BLE001
                observed, detail = REJECT, f"{type(exc).__name__}: {exc}"
            rows.append({
                "case": m.name, "why": m.why, "expected": m.expect,
                "observed": observed, "agrees": observed == m.expect,
                "detail": detail[:140],
            })
    return rows


# =========================================================================== #
# B. Unsafe data flow
# =========================================================================== #
@dataclass
class GraphCase:
    name: str
    why: str
    expect_safe: bool
    build: Callable[[], FabricGraph]
    offline: bool = True


def _g(name: str) -> FabricGraph:
    return FabricGraph(name=name)


def _chain_with_confirm(confirm: bool) -> FabricGraph:
    g = _g(f"confirm-{confirm}")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    g.add(Node("model", NodeKind.CARTRIDGE))
    g.connect("input", "model")
    if confirm:
        g.add(Node("human", NodeKind.CONFIRM))
        g.connect("model", "human")
        g.add(Node("sink", NodeKind.TOOL, privileged=True))
        g.connect("human", "sink")
    else:
        g.add(Node("sink", NodeKind.TOOL, privileged=True))
        g.connect("model", "sink")
    return g


def _branch_one_arm_confirmed() -> FabricGraph:
    """Two arms to one privileged sink; only one passes a human.

    The unsafe arm is the whole point: an analysis that stops at the first
    clean path would call this safe.
    """
    g = _g("branch-one-arm-confirmed")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    g.add(Node("left", NodeKind.CARTRIDGE))
    g.add(Node("right", NodeKind.CARTRIDGE))
    g.add(Node("human", NodeKind.CONFIRM))
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect("input", "left")
    g.connect("input", "right")
    g.connect("left", "human")
    g.connect("human", "sink")
    g.connect("right", "sink")          # unconfirmed arm
    return g


def _branch_both_arms_confirmed() -> FabricGraph:
    g = _g("branch-both-arms-confirmed")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    for arm in ("left", "right"):
        g.add(Node(arm, NodeKind.CARTRIDGE))
        g.add(Node(f"human-{arm}", NodeKind.CONFIRM))
        g.connect("input", arm)
        g.connect(arm, f"human-{arm}")
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect("human-left", "sink")
    g.connect("human-right", "sink")
    return g


def _two_untrusted_sources() -> FabricGraph:
    g = _g("two-untrusted-sources")
    g.add(Node("web", NodeKind.INPUT, produces_untrusted=True))
    g.add(Node("user", NodeKind.INPUT, produces_untrusted=True))
    g.add(Node("merge", NodeKind.CARTRIDGE))
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect("web", "merge")
    g.connect("user", "merge")
    g.connect("merge", "sink")
    return g


def _trusted_only() -> FabricGraph:
    g = _g("trusted-only")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=False))
    g.add(Node("model", NodeKind.CARTRIDGE))
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect("input", "model")
    g.connect("model", "sink")
    return g


def _untrusted_to_unprivileged() -> FabricGraph:
    g = _g("untrusted-to-unprivileged")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    g.add(Node("model", NodeKind.CARTRIDGE))
    g.add(Node("out", NodeKind.OUTPUT))
    g.connect("input", "model")
    g.connect("model", "out")
    return g


def _network_in_offline_graph() -> FabricGraph:
    g = _g("network-in-offline-graph")
    g.add(Node("input", NodeKind.INPUT))
    g.add(Node("fetch", NodeKind.TOOL, requires_network=True))
    g.add(Node("out", NodeKind.OUTPUT))
    g.connect("input", "fetch")
    g.connect("fetch", "out")
    return g


def _cycle() -> FabricGraph:
    g = _g("cycle")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    g.add(Node("a", NodeKind.CARTRIDGE))
    g.add(Node("b", NodeKind.CARTRIDGE))
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect("input", "a")
    g.connect("a", "b")
    g.connect("b", "a")                 # the cycle
    g.connect("b", "sink")
    return g


def _long_laundering_chain() -> FabricGraph:
    """Sixteen cartridges between the attacker and the sink.

    Distance is not sanitisation; a propagating chain of any length still
    delivers the taint.
    """
    g = _g("long-laundering-chain")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    prev = "input"
    for i in range(16):
        g.add(Node(f"c{i}", NodeKind.CARTRIDGE))
        g.connect(prev, f"c{i}")
        prev = f"c{i}"
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect(prev, "sink")
    return g


def _diamond_shortcut() -> FabricGraph:
    """A confirmed long way round and an unconfirmed short way."""
    g = _g("diamond-shortcut")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    g.add(Node("slow", NodeKind.CARTRIDGE))
    g.add(Node("human", NodeKind.CONFIRM))
    g.add(Node("sink", NodeKind.TOOL, privileged=True))
    g.connect("input", "slow")
    g.connect("slow", "human")
    g.connect("human", "sink")
    g.connect("input", "sink")          # the shortcut
    return g


GRAPHS = [
    GraphCase("chain-unconfirmed", "untrusted input reaches a privileged sink",
              False, lambda: _chain_with_confirm(False)),
    GraphCase("chain-confirmed", "a human clears the taint",
              True, lambda: _chain_with_confirm(True)),
    GraphCase("branch-one-arm-confirmed",
              "one unconfirmed arm is enough to make the graph unsafe",
              False, _branch_one_arm_confirmed),
    GraphCase("branch-both-arms-confirmed", "every arm passes a human",
              True, _branch_both_arms_confirmed),
    GraphCase("two-untrusted-sources", "two sources, one privileged sink",
              False, _two_untrusted_sources),
    GraphCase("trusted-only", "no untrusted source at all",
              True, _trusted_only),
    GraphCase("untrusted-to-unprivileged", "untrusted input, nothing privileged",
              True, _untrusted_to_unprivileged),
    GraphCase("network-in-offline-graph", "a networked tool under offline mode",
              False, _network_in_offline_graph),
    GraphCase("cycle", "a cycle breaks the acyclicity the analysis needs",
              False, _cycle),
    GraphCase("long-laundering-chain", "sixteen hops do not sanitise",
              False, _long_laundering_chain),
    GraphCase("diamond-shortcut", "a confirmed path plus an unconfirmed shortcut",
              False, _diamond_shortcut),
]


def run_graphs() -> list[dict[str, Any]]:
    rows = []
    for case in GRAPHS:
        graph = case.build()
        try:
            result = analyse(graph, offline_required=case.offline)
            safe = not result.violations
            detail = "; ".join(result.violations)[:160]
            acyclic = result.acyclic
        except Exception as exc:                            # noqa: BLE001
            safe, detail, acyclic = False, f"{type(exc).__name__}: {exc}"[:160], False
        rows.append({
            "case": case.name, "why": case.why,
            "expected_safe": case.expect_safe, "observed_safe": safe,
            "agrees": safe == case.expect_safe,
            "acyclic": acyclic, "detail": detail,
        })
    return rows


# =========================================================================== #
# C. The boundary — what neither mechanism does
# =========================================================================== #
def run_boundary() -> list[dict[str, Any]]:
    """Measured demonstrations that these are validation, not a sandbox."""
    out: list[dict[str, Any]] = []

    # 1. integrity is relative to the manifest, so a consistent rewrite passes
    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture(Path(tmp) / "a")
        _consistent_forgery(root)
        try:
            verify_manifest(root)
            passed = True
        except Exception:                                   # noqa: BLE001
            passed = False
    out.append({
        "claim": "the manifest does not authenticate",
        "demonstrated": passed,
        "meaning": ("an attacker who can write the artifact can write the manifest "
                    "with it, and the check then passes. It establishes integrity "
                    "RELATIVE to a manifest — it detects damage and accident, not "
                    "an adversary with write access. Authentication would need a "
                    "signature over the manifest with a key the attacker lacks."),
    })

    # 2. Fabric reasons over DECLARED effects, not actual handler behaviour
    g = _g("lying-handler")
    g.add(Node("input", NodeKind.INPUT, produces_untrusted=True))
    # declares itself harmless; nothing here executes it to find out
    g.add(Node("exfiltrate", NodeKind.TOOL, requires_network=False,
               privileged=False, taint_role="clear"))
    g.add(Node("out", NodeKind.OUTPUT))
    g.connect("input", "exfiltrate")
    g.connect("exfiltrate", "out")
    result = analyse(g, offline_required=True)
    out.append({
        "claim": "Fabric validates declarations, not implementations",
        "demonstrated": not result.violations,
        "meaning": ("a node that declares no network use and clears taint is "
                    "accepted on that declaration. Whether its handler honours it "
                    "is outside what analysing the graph can see. The guarantee is "
                    "conditional on declarations being truthful, which is an "
                    "assumption about the node registry, not a property proven here."),
    })

    # 3. the analysis is over the graph as written, not over runtime dispatch
    out.append({
        "claim": "coverage is of these templates and declared effects only",
        "demonstrated": True,
        "meaning": ("the corpus below is planted: it establishes that the checks "
                    "fire on constructed violations. It does not quantify "
                    "behaviour on real user graphs, prove handler isolation, or "
                    "establish superiority over runtime testing."),
    })
    return out


# =========================================================================== #
def main() -> int:
    integrity = run_integrity()
    graphs = run_graphs()
    boundary = run_boundary()

    i_ok = sum(r["agrees"] for r in integrity)
    g_ok = sum(r["agrees"] for r in graphs)
    report = {
        "integrity": {"n": len(integrity), "agree": i_ok,
                      "rate": round(i_ok / len(integrity), 4), "rows": integrity},
        "dataflow": {"n": len(graphs), "agree": g_ok,
                     "rate": round(g_ok / len(graphs), 4), "rows": graphs},
        "boundary": boundary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("A. MALFORMED ARTIFACTS")
    print(f"   {i_ok}/{len(integrity)} behaved as specified")
    for r in integrity:
        mark = "ok " if r["agrees"] else "XX "
        print(f"   {mark}{r['case']:<26}{r['expected']:<7}-> {r['observed']:<7}"
              f"{r['detail'][:52]}")
    print("\nB. UNSAFE DATA FLOW")
    print(f"   {g_ok}/{len(graphs)} behaved as specified")
    for r in graphs:
        mark = "ok " if r["agrees"] else "XX "
        exp = "safe" if r["expected_safe"] else "unsafe"
        obs = "safe" if r["observed_safe"] else "unsafe"
        print(f"   {mark}{r['case']:<28}{exp:<7}-> {obs:<7}{r['detail'][:48]}")
    print("\nC. WHAT THESE ARE NOT")
    for b in boundary:
        print(f"   - {b['claim']}")
        print(f"     {b['meaning'][:150]}")
    print(f"\nwritten to {OUT}")
    return 0 if (i_ok == len(integrity) and g_ok == len(graphs)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
