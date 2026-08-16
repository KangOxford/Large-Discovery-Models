"""Versioned scientific and budget contracts for registered LDM tasks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_CONTRACT_NAME = "experiment.json"
EXPERIMENT_CONTRACT_SCHEMA_VERSION = 1
ACTIVE_CONTRACT_PATH_ENV = "LDM_EXPERIMENT_CONTRACT_PATH"
ACTIVE_CONTRACT_PROFILE_ENV = "LDM_EXPERIMENT_CONTRACT_PROFILE"
METRIC_DIRECTIONS = frozenset({"maximize", "minimize"})
METRIC_ROLES = ("reported", "optimized", "diagnostic")
PROPOSAL_PROVIDER_KINDS = frozenset(
    {
        "unspecified",
        "deterministic",
        "model_endpoint",
        "external_service",
        "dataset",
        "simulator",
        "hybrid",
    }
)


class ExperimentContractError(ValueError):
    """Raised when an experiment contract is invalid or violated."""


@dataclass(frozen=True)
class ExperimentProfile:
    """One named, runner-enforced campaign profile."""

    name: str
    description: str
    budget: dict[str, int | float]
    locked_args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "budget": dict(self.budget),
            "locked_args": dict(self.locked_args),
        }


@dataclass(frozen=True)
class ExperimentContract:
    """Validated task-level benchmark, metric, evaluation, and budget contract."""

    task_id: str
    qualification: str
    benchmark: dict[str, Any]
    proposal_provider: dict[str, Any]
    metrics: dict[str, tuple[dict[str, Any], ...]]
    evaluation: dict[str, Any]
    budget: dict[str, Any]
    profiles: dict[str, ExperimentProfile]
    path: Path
    raw: dict[str, Any]

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def profile(self, name: str) -> ExperimentProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ExperimentContractError(
                f"Unknown experiment contract profile {name!r} in {self.path}; "
                f"expected one of {sorted(self.profiles)}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw))


def load_experiment_contract(path: Path) -> ExperimentContract:
    """Load and strictly validate one JSON experiment contract."""

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentContractError(f"Experiment contract does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExperimentContractError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExperimentContractError(f"Experiment contract must be an object: {path}")

    _reject_unknown(
        payload,
        {
            "schema_version",
            "task_id",
            "qualification",
            "benchmark",
            "proposal_provider",
            "metrics",
            "evaluation",
            "budget",
            "profiles",
        },
        "experiment contract",
        path,
    )
    if payload.get("schema_version") != EXPERIMENT_CONTRACT_SCHEMA_VERSION:
        raise ExperimentContractError(
            f"Unsupported schema_version {payload.get('schema_version')!r} in {path}; "
            f"expected {EXPERIMENT_CONTRACT_SCHEMA_VERSION}."
        )
    task_id = _nonempty_string(payload.get("task_id"), "task_id", path)
    qualification = _nonempty_string(
        payload.get("qualification", "draft"), "qualification", path
    )
    if qualification not in {"draft", "qualified"}:
        raise ExperimentContractError(
            f"qualification must be 'draft' or 'qualified' in {path}"
        )

    benchmark = _object(payload.get("benchmark"), "benchmark", path)
    _reject_unknown(
        benchmark,
        {"source_url", "source_commit", "task_path"},
        "benchmark",
        path,
    )
    _nonempty_string(benchmark.get("source_url"), "benchmark.source_url", path)
    _nonempty_string(benchmark.get("source_commit"), "benchmark.source_commit", path)
    if "task_path" in benchmark:
        _nonempty_string(benchmark["task_path"], "benchmark.task_path", path)

    proposal_provider = _validate_proposal_provider(
        payload.get(
            "proposal_provider",
            {
                "kind": "unspecified",
                "requires_endpoint_preflight": False,
                "supports_collection": False,
            },
        ),
        path,
    )

    metrics_payload = _object(payload.get("metrics"), "metrics", path)
    _reject_unknown(metrics_payload, set(METRIC_ROLES), "metrics", path)
    metrics: dict[str, tuple[dict[str, Any], ...]] = {}
    for role in METRIC_ROLES:
        values = metrics_payload.get(role, [])
        if not isinstance(values, list):
            raise ExperimentContractError(f"metrics.{role} must be an array in {path}")
        metrics[role] = tuple(
            _validate_metric(item, role, path) for item in values
        )
    if not metrics["reported"]:
        raise ExperimentContractError(f"metrics.reported must not be empty in {path}")
    if not metrics["optimized"]:
        raise ExperimentContractError(f"metrics.optimized must not be empty in {path}")

    evaluation = _object(payload.get("evaluation"), "evaluation", path)
    _reject_unknown(
        evaluation,
        {"datasets", "settings", "per_candidate_limits"},
        "evaluation",
        path,
    )
    datasets = evaluation.get("datasets")
    if not isinstance(datasets, list) or not datasets or not all(
        isinstance(item, str) and item.strip() for item in datasets
    ):
        raise ExperimentContractError(
            f"evaluation.datasets must be a non-empty string array in {path}"
        )
    _object(evaluation.get("settings", {}), "evaluation.settings", path)
    per_candidate_limits = _object(
        evaluation.get("per_candidate_limits", {}),
        "evaluation.per_candidate_limits",
        path,
    )

    budget = _object(payload.get("budget"), "budget", path)
    for key, value in budget.items():
        _validate_nonnegative_number(value, f"budget.{key}", path)

    profiles_payload = _object(payload.get("profiles", {}), "profiles", path)
    profiles: dict[str, ExperimentProfile] = {}
    for name, raw_profile in profiles_payload.items():
        if not isinstance(name, str) or not name.strip():
            raise ExperimentContractError(f"profile names must be non-empty strings in {path}")
        profile = _object(raw_profile, f"profiles.{name}", path)
        _reject_unknown(
            profile,
            {"description", "budget", "locked_args"},
            f"profiles.{name}",
            path,
        )
        profile_budget = _object(
            profile.get("budget", {}), f"profiles.{name}.budget", path
        )
        for key, value in profile_budget.items():
            _validate_nonnegative_number(value, f"profiles.{name}.budget.{key}", path)
        locked_args = _object(
            profile.get("locked_args", {}), f"profiles.{name}.locked_args", path
        )
        profiles[name] = ExperimentProfile(
            name=name,
            description=str(profile.get("description", "")).strip(),
            budget=dict(profile_budget),
            locked_args={_normalize_arg_name(key): value for key, value in locked_args.items()},
        )

    if qualification == "qualified":
        if benchmark["source_commit"].strip().lower() == "unqualified":
            raise ExperimentContractError(
                f"qualified contracts must pin benchmark.source_commit in {path}"
            )
        if not per_candidate_limits:
            raise ExperimentContractError(
                f"qualified contracts must define evaluation.per_candidate_limits in {path}"
            )
        if not profiles:
            raise ExperimentContractError(
                f"qualified contracts must define at least one campaign profile in {path}"
            )

    return ExperimentContract(
        task_id=task_id,
        qualification=qualification,
        benchmark=dict(benchmark),
        proposal_provider=proposal_provider,
        metrics=metrics,
        evaluation=dict(evaluation),
        budget=dict(budget),
        profiles=profiles,
        path=path,
        raw=payload,
    )


def validate_profile_args(
    contract: ExperimentContract,
    profile_name: str,
    args: Mapping[str, Any] | None,
) -> ExperimentProfile:
    """Reject config arguments that violate a named campaign profile."""

    profile = contract.profile(profile_name)
    normalized = {_normalize_arg_name(str(key)): value for key, value in (args or {}).items()}
    mismatches: list[str] = []
    for key, expected in profile.locked_args.items():
        if key not in normalized:
            mismatches.append(f"--{key} is required and must equal {expected!r}")
            continue
        actual = normalized[key]
        if not _json_values_equal(actual, expected):
            mismatches.append(f"--{key}={actual!r}, expected {expected!r}")
    if mismatches:
        detail = "; ".join(mismatches)
        raise ExperimentContractError(
            f"Config violates experiment contract profile {profile_name!r}: {detail}"
        )
    return profile


def load_active_experiment_contract() -> tuple[ExperimentContract | None, str]:
    """Load the runner-selected contract and profile from environment variables."""

    raw_path = os.environ.get(ACTIVE_CONTRACT_PATH_ENV, "").strip()
    profile = os.environ.get(ACTIVE_CONTRACT_PROFILE_ENV, "").strip()
    if not raw_path:
        return None, profile
    return load_experiment_contract(Path(raw_path)), profile


def snapshot_experiment_contract(
    contract: ExperimentContract,
    run_dir: Path,
    *,
    profile: str = "",
) -> Path:
    """Atomically snapshot the exact contract used by one campaign."""

    run_dir = Path(run_dir)
    destination = run_dir / "experiment_contract.json"
    payload = contract.to_dict()
    payload["snapshot"] = {
        "source_path": str(contract.path),
        "sha256": contract.digest,
        "profile": profile,
    }
    _atomic_json_write(destination, payload)
    return destination


def _validate_metric(
    raw: Any,
    role: str,
    path: Path,
) -> dict[str, Any]:
    metric = _object(raw, f"metrics.{role}[]", path)
    _reject_unknown(
        metric,
        {"name", "direction", "description", "modes"},
        "metric",
        path,
    )
    name = _nonempty_string(metric.get("name"), f"metrics.{role}.name", path)
    direction = _nonempty_string(
        metric.get("direction"), f"metrics.{role}.{name}.direction", path
    )
    if direction not in METRIC_DIRECTIONS:
        raise ExperimentContractError(
            f"Metric {name!r} has invalid direction {direction!r} in {path}"
        )
    if "modes" in metric:
        modes = metric["modes"]
        if (
            not isinstance(modes, list)
            or not modes
            or not all(isinstance(mode, str) and mode.strip() for mode in modes)
        ):
            raise ExperimentContractError(
                f"metrics.{role}.{name}.modes must be a non-empty string array in {path}"
            )
        normalized_modes = [mode.strip() for mode in modes]
        if len(set(normalized_modes)) != len(normalized_modes):
            raise ExperimentContractError(
                f"metrics.{role}.{name}.modes must not contain duplicates in {path}"
            )
        metric = {**metric, "modes": normalized_modes}
    return dict(metric)


def _validate_proposal_provider(raw: Any, path: Path) -> dict[str, Any]:
    provider = _object(raw, "proposal_provider", path)
    _reject_unknown(
        provider,
        {"kind", "requires_endpoint_preflight", "supports_collection"},
        "proposal_provider",
        path,
    )
    kind = _nonempty_string(provider.get("kind"), "proposal_provider.kind", path)
    if kind not in PROPOSAL_PROVIDER_KINDS:
        raise ExperimentContractError(
            f"proposal_provider.kind must be one of {sorted(PROPOSAL_PROVIDER_KINDS)} "
            f"in {path}"
        )
    requires_endpoint_preflight = _boolean(
        provider.get("requires_endpoint_preflight"),
        "proposal_provider.requires_endpoint_preflight",
        path,
    )
    supports_collection = _boolean(
        provider.get("supports_collection"),
        "proposal_provider.supports_collection",
        path,
    )
    if kind == "model_endpoint" and not requires_endpoint_preflight:
        raise ExperimentContractError(
            f"model_endpoint proposal providers must require endpoint preflight in {path}"
        )
    if kind == "deterministic" and requires_endpoint_preflight:
        raise ExperimentContractError(
            f"deterministic proposal providers cannot require endpoint preflight in {path}"
        )
    return {
        "kind": kind,
        "requires_endpoint_preflight": requires_endpoint_preflight,
        "supports_collection": supports_collection,
    }


def _object(value: Any, field: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object in {path}")
    return value


def _nonempty_string(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentContractError(f"{field} must be a non-empty string in {path}")
    return value.strip()


def _boolean(value: Any, field: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise ExperimentContractError(f"{field} must be a boolean in {path}")
    return value


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    field: str,
    path: Path,
) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise ExperimentContractError(
            f"Unknown {field} field(s) in {path}: {', '.join(unknown)}"
        )


def _validate_nonnegative_number(value: Any, field: str, path: Path) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ExperimentContractError(f"{field} must be a non-negative number in {path}")


def _normalize_arg_name(value: str) -> str:
    return value.strip().lstrip("-").replace("_", "-")


def _json_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, tuple):
        left = list(left)
    if isinstance(right, tuple):
        right = list(right)
    return left == right


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


__all__ = [
    "ACTIVE_CONTRACT_PATH_ENV",
    "ACTIVE_CONTRACT_PROFILE_ENV",
    "EXPERIMENT_CONTRACT_NAME",
    "EXPERIMENT_CONTRACT_SCHEMA_VERSION",
    "ExperimentContract",
    "ExperimentContractError",
    "ExperimentProfile",
    "load_active_experiment_contract",
    "load_experiment_contract",
    "snapshot_experiment_contract",
    "validate_profile_args",
]
