"""tests/bo/ldm/orchestrator/test_decision_log.py"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from bo.ldm.orchestrator.decision_log import DecisionLog


def test_init_creates_file(tmp_path: Path):
    log_path = tmp_path / "decisions.json"
    log = DecisionLog(log_path)
    assert log_path.exists()
    data = json.loads(log_path.read_text())
    assert "decisions" in data
    assert data["decisions"] == []


def test_update_config_snapshot(tmp_path: Path):
    log_path = tmp_path / "decisions.json"
    log = DecisionLog(log_path)
    log.update_config_snapshot({"llm_temperature": 0.25, "bias_weight": 0.1})
    data = json.loads(log_path.read_text())
    assert data["config_snapshot"]["llm_temperature"] == 0.25


def test_append_single_entry(tmp_path: Path):
    log_path = tmp_path / "decisions.json"
    log = DecisionLog(log_path)
    log.append({"antigen_id": "1ADQ_A", "iteration": 1, "outcome": "applied"})
    data = json.loads(log_path.read_text())
    assert len(data["decisions"]) == 1
    entry = data["decisions"][0]
    assert entry["antigen_id"] == "1ADQ_A"
    assert "timestamp" in entry


def test_append_multiple_entries(tmp_path: Path):
    log_path = tmp_path / "decisions.json"
    log = DecisionLog(log_path)
    for i in range(5):
        log.append({"iteration": i, "outcome": "applied"})
    data = json.loads(log_path.read_text())
    assert len(data["decisions"]) == 5


def test_concurrent_append_safe(tmp_path: Path):
    log_path = tmp_path / "decisions.json"
    log = DecisionLog(log_path)
    threads = []
    errors = []

    def worker(idx):
        try:
            for _ in range(10):
                log.append({"worker": idx, "outcome": "applied"})
        except Exception as e:
            errors.append(e)

    for i in range(4):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    assert not errors
    data = json.loads(log_path.read_text())
    assert len(data["decisions"]) == 40


def test_atomic_write_no_tmp_leftover(tmp_path: Path):
    log_path = tmp_path / "decisions.json"
    log = DecisionLog(log_path)
    log.append({"x": 1})
    log.append({"x": 2})
    # No .tmp files left in dir
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []