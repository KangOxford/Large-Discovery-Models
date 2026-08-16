"""tests/core/ldm/orchestrator/test_loop.py"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.llm.client import LLMClient
from tasks.antibody.core.ldm.orchestrator.loop import Orchestrator, OrchestratorDecision
from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus


class MockLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self._idx = 0

    def call(self, prompt, temperature=0.25, timeout_s=30):
        self.calls.append({"prompt": prompt, "temperature": temperature, "timeout_s": timeout_s})
        if self._idx >= len(self.responses):
            return self.responses[-1]
        r = self.responses[self._idx]
        self._idx += 1
        return r


def make_status(**overrides) -> OrchestratorStatus:
    defaults = {
        "iteration": 1, "antigen_id": "1ADQ_A", "antigen_seed": 42, "iter_seed": 0,
        "best_value": -73.6, "best_sequence": [0] * 11,
        "n_evals": 1, "n_iters_without_improvement": 0,
        "full_history": [("ARYYGSYWYFD", -73.6, 0)],
    }
    defaults.update(overrides)
    return OrchestratorStatus(**defaults)


class TestOrchestratorDisabled:
    def test_llm_loop_disabled_still_allows_step(self, tmp_path: Path):
        """llm_loop_enabled=False no longer blocks step() — phase control
        is handled at injection level. step() should still call the LLM."""
        cfg = DSLConfig(llm_loop_enabled=False, max_retries=1)
        client = MockLLMClient(['{"rationale": "test"}'])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert isinstance(d, OrchestratorDecision)
        assert d.source == "llm"
        assert d.applied is True
        assert len(client.calls) == 1


class TestOrchestratorCache:
    def test_cache_hit_skips_llm(self, tmp_path: Path):
        cfg = DSLConfig()
        client = MockLLMClient(['{"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=1, restart=1, steps=10)"}'])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        status = make_status()
        d1 = orch.step(status)
        assert len(client.calls) == 1
        d2 = orch.step(status)
        assert len(client.calls) == 1
        assert d2.source == "cache"

    def test_cache_hit_on_noop(self, tmp_path: Path):
        cfg = DSLConfig()
        client = MockLLMClient(["{}"])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        status = make_status()
        d1 = orch.step(status)
        d2 = orch.step(status)
        assert d1.source == "llm"
        assert d2.source == "cache"
        assert len(client.calls) == 1


class TestOrchestratorLLMCall:
    def test_noop_decision_when_llm_returns_empty(self, tmp_path: Path):
        cfg = DSLConfig()
        client = MockLLMClient(["{}"])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.search_dsl is None
        assert d.bias_dsl is None
        assert d.applied is True
        assert d.source == "llm"

    def test_trust_region_update(self, tmp_path: Path):
        cfg = DSLConfig()
        client = MockLLMClient([
            '{"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=1, restart=1, steps=10)"}'
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.search_dsl is not None
        assert d.bias_dsl is None
        assert d.source == "llm"
        assert d.applied is True

    def test_bias_update(self, tmp_path: Path):
        cfg = DSLConfig()
        client = MockLLMClient(['{"update_bias": "MaxCysteine(1)"}'])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.bias_dsl is not None
        assert d.search_dsl is None
        sample_seq = [0] * 11
        assert d.bias_dsl(sample_seq) == 0.0

    def test_both_updates(self, tmp_path: Path):
        cfg = DSLConfig()
        client = MockLLMClient([
            '{"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=1, restart=1, steps=10)",'
            ' "update_bias": "MaxCysteine(1)"}'
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.search_dsl is not None
        assert d.bias_dsl is not None

    def test_rationale_passed_through(self, tmp_path: Path):
        cfg = DSLConfig()
        client = MockLLMClient([
            '{"rationale": "narrowing TR", '
            '"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=1, restart=1, steps=10)"}'
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.rationale == "narrowing TR"


class TestOrchestratorRetryAndFallback:
    def test_retry_on_invalid_json(self, tmp_path: Path):
        cfg = DSLConfig(max_retries=3)
        client = MockLLMClient([
            "not json",
            '{"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=1, restart=1, steps=10)"}',
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.applied is True
        assert d.source == "llm"
        assert len(client.calls) == 2

    def test_fallback_after_max_retries(self, tmp_path: Path):
        cfg = DSLConfig(max_retries=2)
        client = MockLLMClient([
            '{"update_trust_region": "NotAnAtom(\'x\')"}',
            '{"update_trust_region": "NotAnAtom(\'y\')"}',
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.fallback_used is True
        assert d.source == "fallback"
        assert len(client.calls) == 2

    def test_partial_update_triggers_retry_when_tr_rejected(self, tmp_path: Path):
        """If update_trust_region is rejected but update_bias is valid,
        the orchestrator should retry (not return success) so the LLM
        gets feedback and can fix the TR.
        """
        cfg = DSLConfig(max_retries=3)
        # First response: TR bad, bias good — should trigger retry
        # Second response: TR good, bias good — should succeed
        client = MockLLMClient([
            '{"update_trust_region": "NotAnAtom(\'x\')", "update_bias": "MaxCysteine(1)"}',
            '{"update_trust_region": "LocalSearch(\'ARDYGNYWYFD\', restart=1, steps=10)", "update_bias": "MaxCysteine(1)"}',
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.search_dsl is not None
        assert d.bias_dsl is not None
        assert d.applied is True
        assert len(client.calls) == 2  # retried once


class TestOrchestratorAnyRejected:
    """Test the any_rejected retry logic."""

    def test_tr_rejected_triggers_retry(self, tmp_path: Path):
        """When only TR is attempted and rejected, retry."""
        cfg = DSLConfig(max_retries=3)
        client = MockLLMClient([
            '{"update_trust_region": "NotAnAtom(\'x\')"}',
            '{"update_trust_region": "LocalSearch(\'ARDYGNYWYFD\', restart=1, steps=10)"}',
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.search_dsl is not None
        assert d.applied is True
        assert len(client.calls) == 2

    def test_bias_rejected_triggers_retry(self, tmp_path: Path):
        """When only bias is attempted and rejected, retry."""
        cfg = DSLConfig(max_retries=3)
        client = MockLLMClient([
            '{"update_bias": "NotABias()"}',
            '{"update_bias": "MaxCysteine(1)"}',
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.bias_dsl is not None
        assert d.applied is True
        assert len(client.calls) == 2

    def test_all_rejected_triggers_retry(self, tmp_path: Path):
        """When both fields are rejected, retry."""
        cfg = DSLConfig(max_retries=3)
        client = MockLLMClient([
            '{"update_trust_region": "BadAtom()", "update_bias": "BadBias()"}',
            '{"update_trust_region": "LocalSearch(\'ARDYGNYWYFD\', restart=1, steps=10)", "update_bias": "MaxCysteine(1)"}',
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.search_dsl is not None
        assert d.applied is True
        assert len(client.calls) == 2

    def test_budget_overflow_triggers_retry(self, tmp_path: Path):
        """When LLM proposes atoms exceeding acq_search_budget, retry."""
        # restart=3, steps=200 → budget = 3*201 = 603 > 100
        cfg = DSLConfig(acq_search_budget=100, max_retries=3)
        client = MockLLMClient([
            '{"update_trust_region": "LocalSearch(\'ARDYGNYWYFD\', restart=3, steps=200)"}',
            '{"update_trust_region": "LocalSearch(\'ARDYGNYWYFD\', restart=1, steps=10)"}',
        ])
        orch = Orchestrator(cfg, client, decision_log_path=tmp_path / "log.json")
        d = orch.step(make_status())
        assert d.search_dsl is not None
        assert d.applied is True
        assert len(client.calls) == 2


class TestOrchestratorDecisionLog:
    def test_log_records_decisions(self, tmp_path: Path):
        log_path = tmp_path / "log.json"
        cfg = DSLConfig()
        client = MockLLMClient(['{"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=0, restart=1, steps=1)"}'])
        orch = Orchestrator(cfg, client, decision_log_path=log_path)
        orch.step(make_status(iteration=1))
        orch.step(make_status(iteration=2))
        import json
        data = json.loads(log_path.read_text())
        assert "decisions" in data
        assert len(data["decisions"]) == 2

    def test_log_records_rationale(self, tmp_path: Path):
        log_path = tmp_path / "log.json"
        cfg = DSLConfig()
        client = MockLLMClient([
            '{"rationale": "test reason", "update_bias": "MaxCysteine(1)"}'
        ])
        orch = Orchestrator(cfg, client, decision_log_path=log_path)
        orch.step(make_status())
        data = json.loads(log_path.read_text())
        assert data["decisions"][0]["rationale"] == "test reason"
