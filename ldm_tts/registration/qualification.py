"""Machine-readable qualification evidence for registered LDM tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


QUALIFICATION_EVIDENCE_NAME = "qualification_evidence.json"
QUALIFICATION_EVIDENCE_SCHEMA_VERSION = 1
QUALIFICATION_STAGES = (
    "scaffolded",
    "registered",
    "mock_verified",
    "contract_verified",
    "seed_evaluated",
    "tiny_campaign_verified",
    "campaign_qualified",
)
QUALIFICATION_GATE_STATUSES = frozenset({"passed", "pending", "failed"})


class QualificationEvidenceError(ValueError):
    """Raised when qualification evidence is malformed or inconsistent."""


@dataclass(frozen=True)
class QualificationGate:
    """Evidence and status for one ordered qualification stage."""

    name: str
    status: str
    evidence: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class QualificationEvidence:
    """Validated task qualification state and its supporting artifacts."""

    task_id: str
    stage: str
    benchmark_commit: str
    contract_profile: str
    gates: dict[str, QualificationGate]
    path: Path
    raw: dict[str, Any]

    @property
    def stage_index(self) -> int:
        return qualification_stage_index(self.stage)

    def meets(self, required_stage: str) -> bool:
        return self.stage_index >= qualification_stage_index(required_stage)


def qualification_stage_index(stage: str) -> int:
    """Return the ordered index for a qualification stage."""

    try:
        return QUALIFICATION_STAGES.index(stage)
    except ValueError as exc:
        raise QualificationEvidenceError(
            f"Unknown qualification stage {stage!r}; expected one of "
            f"{list(QUALIFICATION_STAGES)}"
        ) from exc


def load_qualification_evidence(
    path: Path,
    *,
    repository_root: Path | None = None,
    expected_task_id: str = "",
) -> QualificationEvidence:
    """Load and strictly validate one qualification evidence document."""

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise QualificationEvidenceError(
            f"Qualification evidence does not exist: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise QualificationEvidenceError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise QualificationEvidenceError(
            f"Qualification evidence must be an object: {path}"
        )
    _reject_unknown(
        payload,
        {
            "schema_version",
            "task_id",
            "stage",
            "benchmark_commit",
            "contract_profile",
            "gates",
        },
        "qualification evidence",
        path,
    )
    if payload.get("schema_version") != QUALIFICATION_EVIDENCE_SCHEMA_VERSION:
        raise QualificationEvidenceError(
            f"Unsupported schema_version {payload.get('schema_version')!r} in {path}; "
            f"expected {QUALIFICATION_EVIDENCE_SCHEMA_VERSION}."
        )

    task_id = _nonempty_string(payload.get("task_id"), "task_id", path)
    if expected_task_id and task_id != expected_task_id:
        raise QualificationEvidenceError(
            f"Qualification evidence task_id {task_id!r} does not match registered "
            f"task {expected_task_id!r} in {path}"
        )
    stage = _nonempty_string(payload.get("stage"), "stage", path)
    current_index = qualification_stage_index(stage)
    benchmark_commit = _nonempty_string(
        payload.get("benchmark_commit"), "benchmark_commit", path
    )
    contract_profile = payload.get("contract_profile", "")
    if not isinstance(contract_profile, str):
        raise QualificationEvidenceError(
            f"contract_profile must be a string in {path}"
        )
    contract_profile = contract_profile.strip()
    if current_index >= qualification_stage_index("campaign_qualified"):
        if not contract_profile:
            raise QualificationEvidenceError(
                f"campaign_qualified evidence must name contract_profile in {path}"
            )
    if current_index >= qualification_stage_index("contract_verified"):
        if benchmark_commit.lower() == "unqualified":
            raise QualificationEvidenceError(
                f"contract_verified evidence must pin benchmark_commit in {path}"
            )

    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, dict):
        raise QualificationEvidenceError(f"gates must be an object in {path}")
    missing = [name for name in QUALIFICATION_STAGES if name not in raw_gates]
    unknown = sorted(str(name) for name in raw_gates if name not in QUALIFICATION_STAGES)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise QualificationEvidenceError(
            f"Qualification gates must match the ordered stages in {path}: "
            + "; ".join(details)
        )

    root = None if repository_root is None else Path(repository_root).resolve()
    gates: dict[str, QualificationGate] = {}
    for index, name in enumerate(QUALIFICATION_STAGES):
        gate = _validate_gate(name, raw_gates[name], path, root)
        should_have_passed = index <= current_index
        if should_have_passed and gate.status != "passed":
            raise QualificationEvidenceError(
                f"Gate {name!r} must be passed at stage {stage!r} in {path}"
            )
        if not should_have_passed and gate.status == "passed":
            raise QualificationEvidenceError(
                f"Gate {name!r} is passed beyond declared stage {stage!r} in {path}"
            )
        gates[name] = gate

    return QualificationEvidence(
        task_id=task_id,
        stage=stage,
        benchmark_commit=benchmark_commit,
        contract_profile=contract_profile,
        gates=gates,
        path=path,
        raw=payload,
    )


def _validate_gate(
    name: str,
    raw: Any,
    path: Path,
    repository_root: Path | None,
) -> QualificationGate:
    if not isinstance(raw, dict):
        raise QualificationEvidenceError(f"gates.{name} must be an object in {path}")
    _reject_unknown(raw, {"status", "evidence", "description"}, f"gates.{name}", path)
    status = _nonempty_string(raw.get("status"), f"gates.{name}.status", path)
    if status not in QUALIFICATION_GATE_STATUSES:
        raise QualificationEvidenceError(
            f"gates.{name}.status must be one of "
            f"{sorted(QUALIFICATION_GATE_STATUSES)} in {path}"
        )
    raw_evidence = raw.get("evidence", [])
    if not isinstance(raw_evidence, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_evidence
    ):
        raise QualificationEvidenceError(
            f"gates.{name}.evidence must be a string array in {path}"
        )
    evidence = tuple(item.strip() for item in raw_evidence)
    if status == "passed" and not evidence:
        raise QualificationEvidenceError(
            f"Passed gate {name!r} must cite at least one evidence path in {path}"
        )
    if len(set(evidence)) != len(evidence):
        raise QualificationEvidenceError(
            f"gates.{name}.evidence must not contain duplicates in {path}"
        )
    for item in evidence:
        relative = PurePosixPath(item)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise QualificationEvidenceError(
                f"Evidence path {item!r} must be repository-relative in {path}"
            )
        if repository_root is not None and not (repository_root / relative).exists():
            raise QualificationEvidenceError(
                f"Evidence path does not exist for gate {name!r}: {item}"
            )
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise QualificationEvidenceError(
            f"gates.{name}.description must be a string in {path}"
        )
    return QualificationGate(name, status, evidence, description.strip())


def _nonempty_string(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualificationEvidenceError(f"{field} must be a non-empty string in {path}")
    return value.strip()


def _reject_unknown(
    payload: dict[str, Any],
    allowed: set[str],
    field: str,
    path: Path,
) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise QualificationEvidenceError(
            f"Unknown {field} field(s) in {path}: {', '.join(unknown)}"
        )


__all__ = [
    "QUALIFICATION_EVIDENCE_NAME",
    "QUALIFICATION_EVIDENCE_SCHEMA_VERSION",
    "QUALIFICATION_GATE_STATUSES",
    "QUALIFICATION_STAGES",
    "QualificationEvidence",
    "QualificationEvidenceError",
    "QualificationGate",
    "load_qualification_evidence",
    "qualification_stage_index",
]
