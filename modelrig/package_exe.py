"""Ship a built cartridge as a standalone Windows executable.

The deployment story this subsystem exists for: a customer asks for a
specialist, the compiler builds and certifies one, and what they receive is a
single file that runs on a machine with **no Python, no pip, no model download**.

Why an ``.exe`` rather than a service. The whole architecture is built around
models that live on the customer's own device — offline closure is proven
statically, the weights are quantised to fit device RAM, and the privacy
argument is that data never leaves. Shipping a hosted endpoint would contradict
all three. A self-contained binary is the packaging that matches the design.

What goes inside is small, because the artefact is small: a quantised centroid
model is tens of kilobytes of weights. The bulk of the binary is the Python
runtime and numpy, not the model — which is why one runner binary per cartridge
is affordable, and why swapping cartridges is cheaper still (see
:func:`bundle_spec`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from majestic.logging_utils import get_logger

logger = get_logger(__name__)

#: The runner source, written beside the weights and compiled by PyInstaller.
#: Kept as a template rather than a checked-in module so the bundled entry point
#: carries the cartridge id it was built for and cannot be pointed at another.
#:
#: Note what it does NOT do: it does not reimplement inference. ``classifier``
#: is bundled verbatim and imported, so the binary runs the same code path the
#: Proving Ground certified. A second copy of the dequantisation logic here
#: would be free to drift from the one that was measured, and a shipped model
#: that predicts differently from the certified one is the worst defect this
#: system could have.
_RUNNER_TEMPLATE = '''\
"""Standalone runner for Majestic cartridge {cartridge_id}.

Built by modelrig.package_exe. Runs offline with no Python installation.
"""
import sys
from pathlib import Path

import classifier

CARTRIDGE_ID = {cartridge_id!r}
TASK = {task!r}
LABELS = {labels!r}


def _bundled_dir():
    """Where the weights live, whether frozen or run from source."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))


def load_model():
    return classifier.load_model(_bundled_dir())


def predict(model, texts):
    return classifier.predict(model, list(texts))


def main():
    model = load_model()
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        print(f"Majestic cartridge {{CARTRIDGE_ID[:12]}} - {{TASK}}")
        print(f"labels: {{', '.join(LABELS)}}")
        print()
        print("usage:")
        print("  majestic-model.exe \\"some text\\"     classify one input")
        print("  majestic-model.exe                    interactive mode")
        print("  echo text | majestic-model.exe -      read stdin, one per line")
        return 0

    if args == ["-"]:
        lines = [ln.strip() for ln in sys.stdin if ln.strip()]
        for line, label in zip(lines, predict(model, lines)):
            print(f"{{label}}\\t{{line}}")
        return 0

    if args:
        for label in predict(model, [" ".join(args)]):
            print(label)
        return 0

    print(f"Majestic cartridge {{CARTRIDGE_ID[:12]}} - {{TASK}}")
    print(f"labels: {{', '.join(LABELS)}}   (blank line or Ctrl-C to quit)")
    while True:
        try:
            text = input("\\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            return 0
        print(predict(model, [text])[0])


if __name__ == "__main__":
    raise SystemExit(main())
'''


@dataclass
class PackageResult:
    """What the packager produced, or why it could not."""

    cartridge_id: str
    exe_path: Path | None = None
    size_bytes: int = 0
    ok: bool = False
    error: str = ""
    log: list[str] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / 1_000_000, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cartridge_id": self.cartridge_id,
            "exe_path": str(self.exe_path) if self.exe_path else None,
            "exe_name": self.exe_path.name if self.exe_path else None,
            "size_mb": self.size_mb,
            "ok": self.ok,
            "error": self.error,
        }


def bundle_spec(cartridge_id: str, registry_path: str | Path = "./registry") -> dict[str, Any]:
    """What would go into the binary, without building it.

    Useful on its own: it reports the weight footprint separately from the
    runtime footprint, which is the number that decides whether shipping one
    binary per cartridge is sane or whether they should share a runner.
    """
    weights_dir = Path(registry_path) / "weights" / cartridge_id
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"no weights for cartridge {cartridge_id!r}")
    files = sorted(weights_dir.iterdir())
    return {
        "cartridge_id": cartridge_id,
        "weight_files": [f.name for f in files],
        "weight_bytes": sum(f.stat().st_size for f in files),
        "note": (
            "the weights are kilobytes; the binary is mostly the Python runtime "
            "and numpy, so a shared runner with swappable cartridges is cheaper "
            "than one binary each once there are many"
        ),
    }


def package(
    cartridge_id: str,
    registry_path: str | Path = "./registry",
    dist_dir: str | Path = "./dist",
    *,
    task: str = "classify",
    labels: list[str] | None = None,
    clean: bool = True,
    timeout: int = 600,
) -> PackageResult:
    """Compile a cartridge into a standalone ``.exe``.

    Requires PyInstaller. The build is slow (tens of seconds) because it is
    freezing an interpreter, not because the model is large — so it belongs
    behind an explicit user action rather than on the build path.
    """
    result = PackageResult(cartridge_id=cartridge_id)

    weights_dir = Path(registry_path) / "weights" / cartridge_id
    if not weights_dir.is_dir():
        result.error = (
            f"no weights for cartridge {cartridge_id!r} at {weights_dir}. Only a "
            "built-and-admitted cartridge can be packaged"
        )
        return result

    import importlib.util

    if importlib.util.find_spec("PyInstaller") is None:
        result.error = (
            "PyInstaller is not installed. Install it with "
            "`uv pip install pyinstaller` (or `pip install pyinstaller`)"
        )
        return result

    meta = json.loads((weights_dir / "model.json").read_text(encoding="utf-8"))
    labels = labels or list(meta.get("labels", []))

    # Absolute throughout: the subprocess runs with ``cwd`` set to the work
    # directory, so any relative path here would resolve against the wrong root.
    work = (Path(dist_dir) / "_build" / cartridge_id).resolve()
    if clean and work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    runner = work / "majestic_runner.py"
    runner.write_text(
        _RUNNER_TEMPLATE.format(cartridge_id=cartridge_id, task=task, labels=labels),
        encoding="utf-8",
    )
    for f in weights_dir.iterdir():
        shutil.copy2(f, work / f.name)

    # The certified inference path, bundled verbatim rather than reimplemented.
    # classifier.py depends only on the standard library and numpy, so it drops
    # in as a flat module with no package machinery.
    shutil.copy2(Path(__file__).parent / "classifier.py", work / "classifier.py")

    out_dir = (Path(dist_dir) / cartridge_id[:12]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ``--add-data`` separator is os-specific: ';' on Windows, ':' elsewhere.
    sep = ";" if sys.platform == "win32" else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--noconfirm", "--clean",
        "--name", "majestic-model",
        "--distpath", str(out_dir),
        "--workpath", str(work / "build"),
        "--specpath", str(work),
        "--add-data", f"{work / 'model.json'}{sep}.",
        "--add-data", f"{work / 'weights.npz'}{sep}.",
        "--paths", str(work),          # so `import classifier` resolves
        str(runner),
    ]

    logger.info("packaging cartridge %s -> %s", cartridge_id[:12], out_dir)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(work),
        )
    except subprocess.TimeoutExpired:
        result.error = f"PyInstaller timed out after {timeout}s"
        return result

    result.log = (proc.stdout or "").splitlines()[-25:]
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        result.error = "PyInstaller failed:\n" + "\n".join(tail)
        return result

    exe = out_dir / ("majestic-model.exe" if sys.platform == "win32" else "majestic-model")
    if not exe.exists():
        result.error = f"PyInstaller reported success but {exe} is missing"
        return result

    result.exe_path = exe
    result.size_bytes = exe.stat().st_size
    result.ok = True
    logger.info("packaged %s (%.1f MB)", exe.name, result.size_mb)
    return result


def usage_note(result: PackageResult) -> str:
    """How the customer runs what they were just given."""
    if not result.ok or result.exe_path is None:
        return result.error
    name = result.exe_path.name
    return textwrap.dedent(f"""\
        {name}  ({result.size_mb} MB, no Python required)

          {name} "text to classify"     one input
          {name}                        interactive
          type input.txt | {name} -     a file, one line each
    """)


__all__ = ["PackageResult", "bundle_spec", "package", "usage_note"]
