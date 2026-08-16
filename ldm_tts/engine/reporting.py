"""Reusable exports for completed or in-progress engine campaigns."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


OBJECTIVE_DIRECTIONS = frozenset({"maximize", "minimize"})


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object with an artifact-specific error."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"campaign artifact does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"campaign artifact must contain an object: {path}")
    return payload


def load_successful_observations(checkpoint_path: Path) -> list[dict[str, Any]]:
    """Load successful observations from an engine checkpoint in stored order."""

    checkpoint = read_json_object(checkpoint_path)
    state = checkpoint.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint state must be an object: {checkpoint_path}")
    observations = state.get("observations")
    if not isinstance(observations, list):
        raise ValueError(f"checkpoint observations must be an array: {checkpoint_path}")
    successful: list[dict[str, Any]] = []
    for index, observation in enumerate(observations, start=1):
        if not isinstance(observation, dict):
            raise ValueError(f"checkpoint observation {index} must be an object")
        evaluation = observation.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError(f"checkpoint observation {index} has no evaluation object")
        if evaluation.get("status") == "succeeded":
            successful.append(observation)
    return successful


def build_trajectory_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    objective_name: str,
    direction: str,
) -> list[dict[str, Any]]:
    """Build ordered objective and incumbent rows from successful observations."""

    objective_name = str(objective_name).strip()
    if not objective_name:
        raise ValueError("objective_name must not be empty")
    if direction not in OBJECTIVE_DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(OBJECTIVE_DIRECTIONS)}")
    rows: list[dict[str, Any]] = []
    incumbent: float | None = None
    for index, observation in enumerate(observations, start=1):
        candidate = observation.get("candidate")
        evaluation = observation.get("evaluation")
        if not isinstance(candidate, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError(f"observation {index} is missing candidate or evaluation")
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"observation {index} has no metrics object")
        value = _finite_number(metrics.get(objective_name), f"observation {index} objective")
        if incumbent is None:
            incumbent = value
        elif direction == "maximize":
            incumbent = max(incumbent, value)
        else:
            incumbent = min(incumbent, value)
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"observation {index} has no candidate_id")
        round_idx = observation.get("round_idx", index - 1)
        if not isinstance(round_idx, int) or isinstance(round_idx, bool):
            raise ValueError(f"observation {index} has invalid round_idx")
        rows.append(
            {
                "evaluation": index,
                "round": round_idx,
                "candidate_id": candidate_id,
                objective_name: value,
                f"best_{objective_name}": incumbent,
            }
        )
    return rows


def normalize_budget_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fill omitted zero counters and normalize integral JSON numbers."""

    raw_limits = payload.get("limits", {})
    raw_counters = payload.get("counters", {})
    if not isinstance(raw_limits, Mapping) or not isinstance(raw_counters, Mapping):
        raise ValueError("budget limits and counters must be objects")
    names = sorted(set(raw_limits) | set(raw_counters))
    limits = {
        str(name): _json_number(value, f"budget limit {name!r}")
        for name, value in raw_limits.items()
    }
    counters = {
        str(name): _json_number(raw_counters.get(name, 0), f"budget counter {name!r}")
        for name in names
    }
    remaining = {
        str(name): (
            None
            if name not in raw_limits
            else _json_number(
                max(0, float(raw_limits[name]) - float(raw_counters.get(name, 0))),
                f"budget remaining {name!r}",
            )
        )
        for name in names
    }
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("budget metadata must be an object")
    return {
        "limits": limits,
        "counters": counters,
        "remaining": remaining,
        "metadata": dict(metadata),
    }


def write_trajectory_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write trajectory rows with a stable header."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = tuple(rows[0]) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_campaign_result(
    campaign_dir: Path,
    *,
    objective_name: str,
    direction: str,
) -> dict[str, Any]:
    """Build a concise, task-neutral result from standard engine artifacts."""

    campaign_dir = Path(campaign_dir)
    observations = load_successful_observations(campaign_dir / "checkpoint.json")
    rows = build_trajectory_rows(
        observations,
        objective_name=objective_name,
        direction=direction,
    )
    budget = normalize_budget_snapshot(read_json_object(campaign_dir / "budget.json"))
    status = read_json_object(campaign_dir / "status.json")
    campaign = read_json_object(campaign_dir / "campaign.json")
    best_index = None
    if rows:
        selector = max if direction == "maximize" else min
        best_index = selector(
            range(len(rows)),
            key=lambda index: rows[index][objective_name],
        )
    best_candidate = None
    if best_index is not None:
        observation = observations[best_index]
        best_candidate = {
            "candidate_id": rows[best_index]["candidate_id"],
            objective_name: rows[best_index][objective_name],
            "metrics": dict(observation["evaluation"]["metrics"]),
        }
    return {
        "task": campaign.get("task", ""),
        "run_id": campaign.get("run_id", campaign_dir.name),
        "status": status.get("status", "unknown"),
        "finished": status.get("status") == "completed",
        "objective": {"name": objective_name, "direction": direction},
        "evaluation_count": len(rows),
        "evaluations": [
            {
                "iteration": row["evaluation"],
                "candidate_id": row["candidate_id"],
                objective_name: row[objective_name],
            }
            for row in rows
        ],
        "best_candidate": best_candidate,
        "budget": budget,
    }


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _json_number(value: Any, field: str) -> int | float:
    result = _finite_number(value, field)
    if result.is_integer():
        return int(result)
    return result


__all__ = [
    "OBJECTIVE_DIRECTIONS",
    "build_campaign_result",
    "build_trajectory_rows",
    "load_successful_observations",
    "normalize_budget_snapshot",
    "read_json_object",
    "write_trajectory_csv",
]
