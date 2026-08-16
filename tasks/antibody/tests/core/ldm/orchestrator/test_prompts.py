"""tests/core/ldm/orchestrator/test_prompts.py"""
from __future__ import annotations

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.orchestrator.prompts import build_system_prompt, build_user_prompt, render_history_csv
from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus


def test_render_history_csv_basic():
    history = [
        ("ARYYGSYWYFD", -73.6, 0),
        ("SSWRWTVSKDK", -67.2, 1),
    ]
    csv = render_history_csv(history)
    lines = csv.splitlines()
    assert lines[0] == "seq,score,iter"
    assert "ARYYGSYWYFD" in lines[1]
    assert "-73.6000" in lines[1]
    assert "0" in lines[1]


def test_render_history_csv_with_int_seq():
    history = [([0, 14, 19, 19, 5, 15, 19, 18, 19, 4, 2], -73.6, 0)]
    csv = render_history_csv(history)
    assert "ARYYGSYWYFD" in csv.splitlines()[1]


def test_render_history_csv_max_rows():
    history = [(f"SEQ{i:08d}", -i, i) for i in range(20)]
    csv = render_history_csv(history, max_rows=6)
    lines = csv.splitlines()
    assert len(lines) == 7  # header + 3 head + 3 tail


def test_build_system_prompt_loads_file():
    cfg = DSLConfig()
    text = build_system_prompt(cfg)
    assert "ROLE" in text
    assert "OUTPUT FORMAT" in text
    assert "ATOMS" in text
    assert "STRATEGY" in text
    assert "INITIALIZATION PHASE" in text
    assert "rationale" in text.lower()


def test_build_system_prompt_injects_budget():
    cfg = DSLConfig(acq_search_budget=777)
    text = build_system_prompt(cfg)
    assert "777" in text


def test_build_system_prompt_injects_num_review():
    cfg = DSLConfig(num_llm_review=42)
    text = build_system_prompt(cfg)
    assert "42" in text


def test_build_user_prompt_includes_status():
    cfg = DSLConfig()
    status = OrchestratorStatus(
        iteration=5, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=4,
        best_value=-95.5, best_sequence=[0, 14, 19, 19, 5, 15, 19, 18, 19, 4, 2],
        n_evals=50, n_iters_without_improvement=3,
        full_history=[("ARYYGSYWYFD", -73.6, 0)],
    )
    text = build_user_prompt(status, cfg)
    assert "1ADQ_A" in text
    assert "-95.5000" in text
    assert "iter 5" in text
    assert "AntBO default" in text or "no DSL" in text  # current TR is None


def test_build_user_prompt_includes_feedback():
    cfg = DSLConfig()
    status = OrchestratorStatus(iteration=1, antigen_id="X", antigen_seed=0, iter_seed=0)
    text = build_user_prompt(status, cfg, last_rejection_reason="TR too large")
    assert "TR too large" in text
    text2 = build_user_prompt(status, cfg, last_rejection_reason=None)
    assert "first attempt" in text2


class TestBudgetFormulaInPrompt:
    """Verify the prompt contains the explicit budget formula for each atom type."""

    def test_local_search_budget_formula(self):
        cfg = DSLConfig()
        text = build_system_prompt(cfg)
        assert "LocalSearch(center, fixed, radius, restart=R, steps=S)" in text
        assert "budget = R * (S + 1)" in text

    def test_neighbor_sampling_budget_formula(self):
        cfg = DSLConfig()
        text = build_system_prompt(cfg)
        assert "NeighborSampling(center, fixed, radius, mut_pr, budget=N)" in text
        assert "budget = N" in text

    def test_latin_hypercube_budget_formula(self):
        cfg = DSLConfig()
        text = build_system_prompt(cfg)
        assert "LatinHyperCubeSampling(num=N)" in text

    def test_or_budget_formula(self):
        cfg = DSLConfig()
        text = build_system_prompt(cfg)
        assert "Or(atom1, atom2, ...)" in text
        assert "sum(atom.budget for atom in children)" in text

    def test_common_mistake_warning(self):
        cfg = DSLConfig()
        text = build_system_prompt(cfg)
        assert "Common mistake" in text
        assert "restart=3, steps=100" in text
        assert "budget=303" in text

    def test_injected_budget_appears_in_examples(self):
        cfg = DSLConfig(acq_search_budget=400)
        text = build_system_prompt(cfg)
        # 400 should appear in formula and examples
        assert "400" in text
        # acq_div2_2 = 400//2 - 1 = 199
        assert "steps=199" in text

    def test_injected_budget_for_default(self):
        cfg = DSLConfig(acq_search_budget=600)
        text = build_system_prompt(cfg)
        assert "600" in text
        # acq_div2_2 = 600//2 - 1 = 299
        assert "steps=299" in text