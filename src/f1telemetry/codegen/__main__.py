"""``uv run f1-codegen`` - write, or verify, every generated artifact.

``--check`` regenerates in memory and compares against the committed files, exiting
non-zero on any difference. That is the whole of P0-T6: CI cannot pass with stale
generated code, and generated code cannot change without the contract changing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from f1telemetry.codegen.emit_parquet import emit_parquet_schema
from f1telemetry.codegen.emit_python import emit_python_channels
from f1telemetry.codegen.emit_typescript import emit_typescript_channels
from f1telemetry.contracts.channels import ChannelContract, load_channel_contract, repo_root

__all__ = ["Artifact", "build_artifacts", "check", "main", "main_check", "write"]


@dataclass(frozen=True, slots=True)
class Artifact:
    path: Path
    content: str


def _artifacts(contract: ChannelContract, root: Path) -> tuple[Artifact, ...]:
    return (
        Artifact(
            root / "src" / "f1telemetry" / "generated" / "channels.py",
            emit_python_channels(contract),
        ),
        Artifact(
            root / "src" / "f1telemetry" / "generated" / "parquet_schema.py",
            emit_parquet_schema(contract),
        ),
        Artifact(
            root / "web" / "src" / "generated" / "channels.ts",
            emit_typescript_channels(contract),
        ),
    )


def build_artifacts(root: Path | None = None) -> tuple[Artifact, ...]:
    """Render every artifact from ``channels.yaml`` without touching the filesystem."""
    base = repo_root() if root is None else root
    return _artifacts(load_channel_contract(base / "channels.yaml"), base)


def write(root: Path | None = None) -> list[Path]:
    """Write every artifact, creating parent directories. Returns the paths written."""
    written: list[Path] = []
    for artifact in build_artifacts(root):
        artifact.path.parent.mkdir(parents=True, exist_ok=True)
        previous = artifact.path.read_text(encoding="utf-8") if artifact.path.is_file() else None
        if previous != artifact.content:
            artifact.path.write_text(artifact.content, encoding="utf-8", newline="\n")
        written.append(artifact.path)
    return written


def check(root: Path | None = None) -> list[str]:
    """Return a list of stale or missing artifacts. Empty means the tree is current."""
    stale: list[str] = []
    for artifact in build_artifacts(root):
        if not artifact.path.is_file():
            stale.append(f"missing: {artifact.path}")
            continue
        current = artifact.path.read_text(encoding="utf-8")
        if current != artifact.content:
            stale.append(f"stale: {artifact.path}")
    return stale


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1-codegen",
        description="Generate the channel registry, Parquet schema and TypeScript types.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed artifacts are current instead of writing them",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only report changes and problems",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = repo_root()
    if args.check:
        return _report(check(root), root, checking=True, quiet=args.quiet)
    changed = [
        artifact
        for artifact in build_artifacts(root)
        if not artifact.path.is_file()
        or artifact.path.read_text(encoding="utf-8") != artifact.content
    ]
    write(root)
    stale = check(root)
    if stale:
        for line in stale:
            print(line, file=sys.stderr)
        return 1
    if args.quiet:
        return 0
    print(f"codegen: {len(changed)} file(s) written, {len(changed)} file(s) changed")
    for artifact in changed:
        print(f"  {_relative(artifact.path, root)}")
    return 0


def main_check(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``f1-check-contract``: the CI gate, identical to ``--check``."""
    forward = ["--check"]
    if argv is not None and "--quiet" in argv:
        forward.append("--quiet")
    return main(forward)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _report(stale: list[str], root: Path, *, checking: bool, quiet: bool) -> int:
    if stale:
        verb = "not up to date" if checking else "failed"
        print(f"codegen: generated code is {verb}", file=sys.stderr)
        for line in stale:
            print(f"  {line}", file=sys.stderr)
        print("run: uv run f1-codegen", file=sys.stderr)
        return 1
    if not quiet:
        print("codegen: generated code is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
