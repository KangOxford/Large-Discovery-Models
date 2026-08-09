#!/usr/bin/env python3
"""Create a non-overwriting skeleton for a manifest-registered task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.task_scaffold import TaskScaffoldError, scaffold_task


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tasks/<task_id> and config/<task_id> skeletons."
    )
    parser.add_argument("task_id", help="Lowercase Python identifier, for example protein_design.")
    parser.add_argument("--description", required=True, help="One-line task description.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        created = scaffold_task(
            args.task_id,
            description=args.description,
            repository_root=REPO_ROOT,
        )
    except TaskScaffoldError as exc:
        raise SystemExit(str(exc)) from exc
    for path in created:
        print(path.relative_to(REPO_ROOT))
    print(f"Created {len(created)} files. Replace placeholders, then run scripts/validate_tasks.py --task {args.task_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
