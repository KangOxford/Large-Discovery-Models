"""Tests for TrajectoryRecorder + RoundRecord.

Covers:

* Round context manager records per-round events.
* Final JSON structure has all required top-level fields.
* Fatal error path writes both ``*.json`` and ``*.error.json``.
* ``resolve_trajectory_path`` applies dir-or-file disambiguation.
* Successful run + re-parse JSON to recover the round timeline.
"""

import json
import os
from pathlib import Path

import pytest

from strbo_v1.llm_advisor.trajectory import (
    RoundRecord,
    TrajectoryRecorder,
    resolve_trajectory_path,
)


def test_resolve_trajectory_path_directory() -> None:
    p = resolve_trajectory_path("output/bo_llm", method="bo", seed=7)
    assert str(p) == "output/bo_llm/bo_seed=7_trajectory.json"


def test_resolve_trajectory_path_file() -> None:
    p = resolve_trajectory_path("output/bo_llm/foo.json", method="bo", seed=7)
    assert str(p) == "output/bo_llm/foo.json"


def test_resolve_trajectory_path_uppercase_suffix() -> None:
    p = resolve_trajectory_path("output/bo_llm/foo.JSON", method="bo", seed=7)
    assert str(p) == "output/bo_llm/foo.JSON"


def test_recorder_writes_final_json(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(path=tmp_path / "traj.json", method="m", seed=0)
    rec.begin_run(config={"k": "v"}, llm_model="mock-llm")
    with rec.round_context(0) as r:
        r.pre_state_snapshot = {"pool": ["A", "B"]}
        r.bo_suggestions = [{"smiles": "C"}]
        r.llm_interactions = {"phase_a": {}, "phase_b": {}}
        r.scores = {"C": [-1.0]}
    rec.set_status("completed")
    rec.set_final_history([("A", -2.0), ("B", -3.0), ("C", -1.0)])
    rec.write_final()

    assert (tmp_path / "traj.json").exists()
    payload = json.loads((tmp_path / "traj.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["run_metadata"]["method"] == "m"
    assert payload["run_metadata"]["llm_model"] == "mock-llm"
    assert payload["history"] == [
        {"index": 0, "smiles": "A", "score": -2.0},
        {"index": 1, "smiles": "B", "score": -3.0},
        {"index": 2, "smiles": "C", "score": -1.0},
    ]
    assert len(payload["rounds"]) == 1
    rd = payload["rounds"][0]
    assert rd["round_idx"] == 0
    assert rd["pre_state_snapshot"] == {"pool": ["A", "B"]}
    assert rd["bo_suggestions"] == [{"smiles": "C"}]
    assert rd["scores"] == {"C": [-1.0]}


def test_recorder_fatal_error_writes_sidecar(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(path=tmp_path / "traj.json", method="m", seed=0)
    rec.begin_run(config={}, llm_model="m")
    with rec.round_context(0):
        pass
    try:
        raise ValueError("intentional test error")
    except ValueError as exc:
        rec.record_fatal_error(round_idx=rec.current_round, exc=exc)
        rec.set_final_history([])
        rec.dump_emergency_json()
        rec.write_final()

    main = json.loads((tmp_path / "traj.json").read_text(encoding="utf-8"))
    side = json.loads((tmp_path / "traj.json.error.json").read_text(encoding="utf-8"))
    assert main["status"] == "fatal_error"
    assert side["status"] == "fatal_error"
    assert main["fatal_error"]["exc_type"] == "ValueError"
    assert main["fatal_error"]["message"] == "intentional test error"
    assert "Traceback" in main["fatal_error"]["traceback"]


def test_recorder_write_final_is_idempotent(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(path=tmp_path / "traj.json", method="m", seed=0)
    rec.begin_run(config={}, llm_model="m")
    rec.write_final()
    rec.write_final()              # second call: no-op
    assert (tmp_path / "traj.json").exists()


def test_recorder_multi_round_context(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(path=tmp_path / "traj.json", method="m", seed=0)
    rec.begin_run(config={}, llm_model="m")
    for i in range(3):
        with rec.round_context(i) as r:
            r.scores = {"X": [float(i)]}
    rec.set_status("completed")
    rec.set_final_history([])
    rec.write_final()
    payload = json.loads((tmp_path / "traj.json").read_text(encoding="utf-8"))
    assert [r["round_idx"] for r in payload["rounds"]] == [0, 1, 2]
    assert [r["scores"] for r in payload["rounds"]] == [
        {"X": [0.0]}, {"X": [1.0]}, {"X": [2.0]},
    ]


def test_round_context_manager_raises_current_round(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(path=tmp_path / "traj.json", method="m", seed=0)
    rec.begin_run(config={}, llm_model="m")
    assert rec.current_round == -1
    with rec.round_context(5):
        assert rec.current_round == 5
    assert rec.current_round == -1
