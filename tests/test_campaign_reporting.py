from __future__ import annotations

import csv
import json
from pathlib import Path

from ldm_tts.engine.reporting import (
    build_campaign_result,
    build_trajectory_rows,
    load_successful_observations,
    normalize_budget_snapshot,
    write_trajectory_csv,
)


def _write_campaign(path: Path) -> None:
    path.mkdir()
    observations = [
        {
            "round_idx": 0,
            "candidate": {"candidate_id": "first"},
            "evaluation": {
                "status": "succeeded",
                "metrics": {"score": 0.5, "cost": 3.0},
            },
        },
        {
            "round_idx": 1,
            "candidate": {"candidate_id": "failed"},
            "evaluation": {"status": "failed", "metrics": {}},
        },
        {
            "round_idx": 2,
            "candidate": {"candidate_id": "best"},
            "evaluation": {
                "status": "succeeded",
                "metrics": {"score": 0.75, "cost": 1.5},
            },
        },
    ]
    artifacts = {
        "checkpoint.json": {"state": {"observations": observations}},
        "budget.json": {
            "limits": {"evaluations": 2.0, "llm_requests": 0, "gpu_hours": 1.5},
            "counters": {"evaluations": 2.0, "gpu_hours": 0.5},
            "metadata": {"task": "fixture"},
        },
        "status.json": {"status": "completed"},
        "campaign.json": {"task": "fixture", "run_id": "run-1"},
    }
    for name, payload in artifacts.items():
        (path / name).write_text(json.dumps(payload), encoding="utf-8")


def test_campaign_reporting_builds_incumbents_and_normalizes_budgets(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaign"
    _write_campaign(campaign_dir)

    observations = load_successful_observations(campaign_dir / "checkpoint.json")
    maximize = build_trajectory_rows(
        observations,
        objective_name="score",
        direction="maximize",
    )
    minimize = build_trajectory_rows(
        observations,
        objective_name="cost",
        direction="minimize",
    )

    assert [row["best_score"] for row in maximize] == [0.5, 0.75]
    assert [row["best_cost"] for row in minimize] == [3.0, 1.5]
    report = build_campaign_result(
        campaign_dir,
        objective_name="score",
        direction="maximize",
    )
    assert report["finished"] is True
    assert report["evaluation_count"] == 2
    assert report["best_candidate"]["candidate_id"] == "best"
    assert report["budget"]["limits"]["evaluations"] == 2
    assert report["budget"]["limits"]["gpu_hours"] == 1.5
    assert report["budget"]["counters"]["llm_requests"] == 0


def test_trajectory_csv_and_standalone_budget_normalization(tmp_path: Path) -> None:
    budget = normalize_budget_snapshot(
        {
            "limits": {"whole": 60.0, "fractional": 2.5, "untouched": 1},
            "counters": {"whole": 60.0, "fractional": 0.25},
        }
    )
    assert budget["counters"] == {"fractional": 0.25, "untouched": 0, "whole": 60}
    assert budget["remaining"] == {"fractional": 2.25, "untouched": 1, "whole": 0}

    path = write_trajectory_csv(
        tmp_path / "trajectory.csv",
        [{"evaluation": 1, "candidate_id": "one", "score": 0.5}],
    )
    with path.open(encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"evaluation": "1", "candidate_id": "one", "score": "0.5"}
        ]
