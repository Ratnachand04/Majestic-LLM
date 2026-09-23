"""Portable SHA-256 manifests. Integrity relative to a manifest, not authentication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(directory: Path) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            files[path.relative_to(directory).as_posix()] = digest_file(path)
    (directory / "checksums.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    return files


def verify_manifest(directory: Path) -> None:
    base = directory.resolve()
    files = json.loads((base / "checksums.json").read_text(encoding="utf-8"))
    if not files:
        raise ValueError("empty artifact integrity manifest")
    actual = {p.relative_to(base).as_posix() for p in base.rglob("*")
              if p.is_file() and p.name != "checksums.json"}
    if actual != set(files):
        raise ValueError("artifact files differ from integrity manifest")
    for relative, expected in files.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"missing or invalid artifact file: {relative}")
        if digest_file(path) != expected:
            raise ValueError(f"artifact integrity failed: {relative}")
