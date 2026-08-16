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
from ldm_tts.registration.qualification import (
    QUALIFICATION_EVIDENCE_NAME,
    QUALIFICATION_STAGES,
    QualificationEvidenceError,
    load_qualification_evidence,
    qualification_stage_index,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tasks registered under tasks/.")
    parser.add_argument("--task", default="", help="Validate only this task ID.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Fail unless every selected task has a qualified experiment contract.",
    )
    parser.add_argument(
        "--require-stage",
        choices=QUALIFICATION_STAGES,
        default="",
        help="Fail unless machine-readable evidence reaches this qualification stage.",
    )
    return parser.parse_args(argv)


def validate_registered_tasks(
    task_id: str = "",
    *,
    require_qualified: bool = False,
    require_stage: str = "",
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
        contract = None
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

        evidence_path = (
            REPO_ROOT
            / definition.relative_root
            / "resources"
            / QUALIFICATION_EVIDENCE_NAME
        )
        if not evidence_path.is_file():
            rows.append({
                "task": definition.task_id,
                "level": "error" if require_stage else "warning",
                "message": (
                    f"Qualification evidence is required at stage {require_stage!r}."
                    if require_stage
                    else "No machine-readable qualification evidence is registered."
                ),
                "path": str(evidence_path),
            })
        else:
            try:
                evidence = load_qualification_evidence(
                    evidence_path,
                    repository_root=REPO_ROOT,
                    expected_task_id=definition.task_id,
                )
            except QualificationEvidenceError as exc:
                rows.append({
                    "task": definition.task_id,
                    "level": "error",
                    "message": str(exc),
                    "path": str(evidence_path),
                })
            else:
                evidence_errors: list[str] = []
                if (
                    evidence.stage_index
                    >= qualification_stage_index("contract_verified")
                ):
                    if contract is None:
                        evidence_errors.append(
                            "contract-verified evidence requires experiment.json"
                        )
                    else:
                        source_commit = str(contract.benchmark.get("source_commit", ""))
                        if evidence.benchmark_commit != source_commit:
                            evidence_errors.append(
                                "qualification benchmark_commit does not match experiment.json"
                            )
                if evidence.stage == "campaign_qualified":
                    if contract is not None and contract.qualification != "qualified":
                        evidence_errors.append(
                            "campaign-qualified evidence requires a qualified experiment contract"
                        )
                    if (
                        contract is not None
                        and evidence.contract_profile not in contract.profiles
                    ):
                        evidence_errors.append(
                            f"contract profile {evidence.contract_profile!r} is not defined"
                        )
                for message in evidence_errors:
                    rows.append({
                        "task": definition.task_id,
                        "level": "error",
                        "message": message,
                        "path": str(evidence_path),
                    })
                meets_required_stage = not require_stage or evidence.meets(require_stage)
                rows.append({
                    "task": definition.task_id,
                    "level": "ok" if meets_required_stage else "error",
                    "message": (
                        f"Qualification evidence reaches stage {evidence.stage!r}."
                        if meets_required_stage
                        else f"Qualification evidence is at {evidence.stage!r}; "
                        f"required stage is {require_stage!r}."
                    ),
                    "path": str(evidence_path),
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
            require_stage=args.require_stage,
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
