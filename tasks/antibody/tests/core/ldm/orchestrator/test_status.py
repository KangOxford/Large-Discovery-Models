"""tests/core/ldm/orchestrator/test_status.py"""
from __future__ import annotations

from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus


def test_default_status():
    s = OrchestratorStatus(
        iteration=1, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=0
    )
    assert s.iteration == 1
    assert s.antigen_id == "1ADQ_A"
    assert s.antigen_seed == 42
    assert s.iter_seed == 0
    assert s.current_search_dsl is None
    assert s.current_bias_dsl is None
    assert s.full_history == []
    assert s.best_value == 0.0
    assert s.best_sequence == []
    assert s.n_evals == 0
    assert s.n_iters_without_improvement == 0


def test_status_with_history():
    history = [("ARYYGSYWYFD", -73.6, 0), ("SSWRWTVSKDK", -67.2, 1)]
    s = OrchestratorStatus(
        iteration=2, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=1,
        full_history=history, best_value=-73.6, best_sequence=[0, 14, 19, 19, 5, 15, 19, 18, 19, 4, 2],
    )
    assert len(s.full_history) == 2
    assert s.best_value == -73.6
    assert s.best_sequence[0] == 0