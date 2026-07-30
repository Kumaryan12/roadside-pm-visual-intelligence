"""Canonical command surface for existing MUMMA-281 stages.

This runner delegates to historical scripts in a subprocess. It intentionally
preserves their current working directory, defaults, output paths, and Python
environment while providing one discoverable entry point.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .stages import PROJECT_ROOT, STAGES, validate_stage_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=sorted(STAGES))
    parser.add_argument("--list", action="store_true", help="List registered stages.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delegated command without running the historical script.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_stage_registry()

    if args.list:
        for name, stage in STAGES.items():
            print(f"{name:28} {stage.role}")
            print(f"{'':28} validation: {stage.validation_note}")
        return 0

    if args.stage is None:
        raise SystemExit("Choose a stage or pass --list.")

    stage = STAGES[args.stage]
    command = [sys.executable, str(stage.script)]
    print("Delegating to historical implementation:")
    print(" ".join(command))
    print(f"Validation note: {stage.validation_note}")
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

