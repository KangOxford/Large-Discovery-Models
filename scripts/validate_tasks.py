#!/usr/bin/env python3
"""Validate registered task manifests, layouts, and dependency hooks."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.registration.registry import (
    TASK_DEFINITIONS,
    TASK_DISCOVERY_ERROR,
    TaskRegistrationError,
    validate_task_layout,
)
from ldm_tts.registration.experiment import ExperimentContractError, load_experiment_contract


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tasks registered under tasks/.")
    parser.add_argument("--task", default="", help="Validate only this task ID.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Fail unless every selected task has a qualified experiment contract.",
    )
    return parser.parse_args(argv)


def validate_registered_tasks(
    task_id: str = "",
    *,
    require_qualified: bool = False,
) -> list[dict[str, str]]:
    if TASK_DISCOVERY_ERROR is not None:
        return [{
            "task": task_id or "registry",
            "level": "error",
            "message": str(TASK_DISCOVERY_ERROR),
            "path": str(REPO_ROOT / "tasks"),
        }]
    if task_id:
        if task_id not in TASK_DEFINITIONS:
            raise TaskRegistrationError(
                f"Unknown task {task_id!r}; expected one of {sorted(TASK_DEFINITIONS)}"
            )
        definitions = [TASK_DEFINITIONS[task_id]]
    else:
        definitions = list(TASK_DEFINITIONS.values())

    rows: list[dict[str, str]] = []
    for definition in definitions:
        task_row_start = len(rows)
        issues = validate_task_layout(definition, repository_root=REPO_ROOT)
        if definition.dependency_checker:
            try:
                module_name, function_name = definition.dependency_checker.split(":", 1)
                checker = getattr(importlib.import_module(module_name), function_name)
                if not callable(checker):
                    raise TypeError("resolved object is not callable")
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                rows.append({
                    "task": definition.task_id,
                    "level": "error",
                    "message": f"Cannot load dependency checker: {exc}",
                    "path": definition.dependency_checker,
                })
        rows.extend({
            "task": definition.task_id,
            "level": issue.level,
            "message": issue.message,
            "path": str(issue.path),
        } for issue in issues)

        if definition.experiment_contract_path is None:
            rows.append({
                "task": definition.task_id,
                "level": "error" if require_qualified else "info",
                "message": (
                    "A qualified experiment contract is required."
                    if require_qualified
                    else "No experiment contract is registered; campaign qualification is unknown."
                ),
                "path": str(REPO_ROOT / definition.relative_root / "experiment.json"),
            })
        else:
            contract_path = REPO_ROOT / definition.experiment_contract_path
            try:
                contract = load_experiment_contract(contract_path)
            except ExperimentContractError:
                # validate_task_layout already emits the actionable parse error.
                pass
            else:
                is_qualified = contract.qualification == "qualified"
                rows.append({
                    "task": definition.task_id,
                    "level": "ok" if is_qualified else ("error" if require_qualified else "info"),
                    "message": (
                        "Experiment contract is qualified."
                        if is_qualified
                        else "Experiment contract is draft; campaign qualification is incomplete."
                    ),
                    "path": str(contract_path),
                })

        task_rows = rows[task_row_start:]
        if not any(row["level"] == "error" for row in task_rows):
            rows.append({
                "task": definition.task_id,
                "level": "ok",
                "message": "Task registration and layout are valid.",
                "path": str(REPO_ROOT / definition.relative_root),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = validate_registered_tasks(
            args.task,
            require_qualified=args.require_qualified,
        )
    except TaskRegistrationError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(f"[{row['level'].upper()}] {row['task']}: {row['message']} ({row['path']})")
    return 1 if any(row["level"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
