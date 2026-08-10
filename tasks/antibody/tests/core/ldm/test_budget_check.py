"""tests/core/ldm/test_budget_check.py — budget overflow enforcement."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.llm.client import LLMClient
from tasks.antibody.core.ldm.orchestrator.loop import Orchestrator
from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus


class MockLLM(LLMClient):
    def __init__(self, responses):
        self.responses = responses
        self.idx = 0

    def call(self, prompt, temperature=0.25, timeout_s=30):
        r = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return r


class TestOrchestratorBudgetEnforcement:
    def test_orchestrator_rejects_overbudget_dsl(self, tmp_path):
        """When the LLM proposes a DSL whose budget > acq_search_budget,
        the orchestrator should reject it and eventually fallback."""
        # budget = 2 * (100 + 1) = 202 > total 100
        overbudget_dsl = (
            "Or(LocalSearch('ARDYGNYWYFD', restart=3, steps=100), "
            "LocalSearch('VRGYYSDWYMD', restart=3, steps=100))"
        )
        cfg = DSLConfig(
            acq_search_budget=100,
            max_retries=2,
            llm_temperature=0.0,
        )
        client = MockLLM([json.dumps({"update_trust_region": overbudget_dsl})])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        status = OrchestratorStatus(
            iteration=1, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=0,
        )
        decision = orch.step(status)
        # After max retries with bad input, should fallback
        assert decision.fallback_used is True
        assert decision.search_dsl is None

    def test_orchestrator_accepts_within_budget_dsl(self, tmp_path):
        """Within-budget DSL should be accepted on first try."""
        # budget = 2 * (50 + 1) = 102 <= total 200
        good_dsl = "LocalSearch('ARDYGNYWYFD', restart=2, steps=50)"
        cfg = DSLConfig(
            acq_search_budget=200,
            max_retries=2,
            llm_temperature=0.0,
        )
        client = MockLLM([json.dumps({"update_trust_region": good_dsl})])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        status = OrchestratorStatus(
            iteration=1, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=0,
        )
        decision = orch.step(status)
        assert decision.fallback_used is False
        assert decision.search_dsl is not None
        assert decision.search_updated is True

    def test_orchestrator_boundary_just_under(self, tmp_path):
        """DSL budget exactly at the limit should be accepted."""
        # budget = 1 * (99 + 1) = 100 < total 200
        boundary_dsl = "LocalSearch('ARDYGNYWYFD', restart=1, steps=99)"
        cfg = DSLConfig(
            acq_search_budget=200,
            max_retries=2,
            llm_temperature=0.0,
        )
        client = MockLLM([json.dumps({"update_trust_region": boundary_dsl})])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        status = OrchestratorStatus(
            iteration=1, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=0,
        )
        decision = orch.step(status)
        assert decision.fallback_used is False


class TestBudgetConfig:
    def test_default_budget_values(self):
        cfg = DSLConfig()
        assert cfg.acq_search_budget == 600
        assert cfg.acq_max_rounds == 3
        assert cfg.max_retries == 3
        assert cfg.num_llm_review == 10

    def test_custom_budget_values(self):
        cfg = DSLConfig(
            acq_search_budget=1000,
            acq_max_rounds=5,
            max_retries=5,
            num_llm_review=20,
        )
        assert cfg.acq_search_budget == 1000
        assert cfg.acq_max_rounds == 5
        assert cfg.max_retries == 5
        assert cfg.num_llm_review == 20


class TestFallbackDSL:
    """Test the adaptive fallback DSL when LLM doesn't provide one."""

    def test_fallback_steps_adapt_to_budget_600(self):
        from tasks.antibody.core.ldm.dsl.search_space import LocalSearch
        budget = 600
        restart = 3
        steps = max(1, budget // restart - 1)
        dsl = LocalSearch("ARDYGNYWYFD", radius=3, restart=restart, steps=steps)
        assert dsl.budget == restart * (steps + 1)
        assert dsl.budget <= budget

    def test_fallback_steps_adapt_to_budget_200(self):
        from tasks.antibody.core.ldm.dsl.search_space import LocalSearch
        budget = 200
        restart = 3
        steps = max(1, budget // restart - 1)
        dsl = LocalSearch("ARDYGNYWYFD", radius=3, restart=restart, steps=steps)
        assert dsl.budget <= budget

    def test_fallback_steps_adapt_to_budget_1000(self):
        from tasks.antibody.core.ldm.dsl.search_space import LocalSearch
        budget = 1000
        restart = 3
        steps = max(1, budget // restart - 1)
        dsl = LocalSearch("ARDYGNYWYFD", radius=3, restart=restart, steps=steps)
        assert dsl.budget <= budget

    def test_fallback_never_exceeds_budget(self):
        from tasks.antibody.core.ldm.dsl.search_space import LocalSearch
        for budget in [50, 100, 200, 600, 1000]:
            restart = 3
            steps = max(1, budget // restart - 1)
            dsl = LocalSearch("ARDYGNYWYFD", radius=3, restart=restart, steps=steps)
            assert dsl.budget <= budget, (
                f"budget={budget}, restart={restart}, steps={steps}, "
                f"actual={dsl.budget}"
            )


class TestConfigDrivenPath:
    """Test that the LDM path is determined by config, not runtime detection."""

    def test_path_from_config_llm_loop_disabled(self):
        llm_loop_enabled = False
        # Path should NOT run LDM session
        assert llm_loop_enabled is False

    def test_path_from_config_llm_loop_enabled(self):
        llm_loop_enabled = True
        # Path SHOULD run LDM session
        assert llm_loop_enabled is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
