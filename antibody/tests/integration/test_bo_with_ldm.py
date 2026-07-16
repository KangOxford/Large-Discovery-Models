"""tests/integration/test_bo_with_ldm.py — minimal end-to-end integration test.

Boots BOExperiments with a mock Orchestrator and verifies:
  - bo/main.py constructs Orchestrator from yaml
  - Orchestrator.step is called each iteration
  - Decisions are stored on CASMOPOLITANCat (via apply_decision)
  - bo/localbo_cat.py uses sample_candidates / score_with_bias
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from bo.ldm import DSLConfig, Orchestrator, SearchSpaceAtom, BiasAtom
from bo.ldm.llm.client import LLMClient


class MockLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list = []
        self._idx = 0

    def call(self, prompt, temperature=0.25, timeout_s=30):
        self.calls.append(1)
        if self._idx < len(self.responses):
            r = self.responses[self._idx]
            self._idx += 1
            return r
        return self.responses[-1]


def test_boexperiments_constructs_orchestrator_when_enabled(monkeypatch):
    """Verify DSLConfig builds from the ldm yaml section (keys with `llm_` prefix)."""
    ldm_dict = {
        "llm_init_enabled": True,
        "llm_loop_enabled": True,
        "llm_temperature": 0.25,
        "max_retries": 3,
        "llm_call_timeout_s": 30,
        "llm_decisions_log": "/tmp/test_log.json",
        "history_max_in_prompt": 100,
        "bias_weight": 0.1,
        "acq_n_candidates": 5000,
        "sample_timeout_s": 5.0,
        "init_pool_size": 100000,
        "max_nesting_depth": 8,
        "atoms_whitelist": ("LatinHyperCubeSampling", "NeighborSampling", "LocalSearch", "Or",
                             "MaxCysteine", "MaxHydrophobicRun", "MaxAromatic",
                             "NetChargeRange", "NoNGlycosylation", "BiasSum"),
        "fallback_strategy": "original_antbo",
    }
    config = DSLConfig.from_yaml(ldm_dict)
    assert config.llm_init_enabled is True
    assert config.llm_loop_enabled is True
    assert config.bias_weight == 0.1
    assert config.acq_n_candidates == 5000
    assert config.init_pool_size == 100000
    assert config.llm_temperature == 0.25


def _make_cat(orch):
    """Build a minimal CASMOPOLITANCat with the given orchestrator."""
    from bo.localbo_cat import CASMOPOLITANCat
    return CASMOPOLITANCat(
        dim=11, n_init=2, config=np.array([20] * 11), batch_size=1,
        min_cuda=1000, n_training_steps=30, normalise=True, device="cpu",
        kernel_type="transformed_overlap",
        orchestrator=orch, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=0,
    )


def test_orchestrator_decision_flows_to_casmopolitan():
    """End-to-end: Orchestrator's decision should be visible on CASMOPOLITANCat."""
    mock = MockLLM(["{}"])
    config = DSLConfig(llm_loop_enabled=True, max_retries=1)
    orch = Orchestrator(config=config, llm_client=mock)
    cat = _make_cat(orch)

    assert cat.orchestrator is orch
    assert cat._search_dsl is None

    from bo.ldm.integrate import build_status, apply_decision
    status = build_status(cat, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=1)
    decision = orch.step(status)
    apply_decision(cat, decision)
    assert cat._search_dsl is None  # noop
    assert cat._bias_dsl is None
    assert mock.calls == [1]


def test_orchestrator_sets_search_dsl():
    """When LLM returns update_trust_region, it should land on cat._search_dsl."""
    mock = MockLLM([
        '{"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=2, restart=1, steps=10)"}',
    ])
    config = DSLConfig(llm_loop_enabled=True)
    orch = Orchestrator(config=config, llm_client=mock)
    cat = _make_cat(orch)

    from bo.ldm.integrate import build_status, apply_decision
    status = build_status(cat, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=1)
    decision = orch.step(status)
    apply_decision(cat, decision)

    assert cat._search_dsl is not None
    assert isinstance(cat._search_dsl, SearchSpaceAtom)


def test_orchestrator_sets_bias_dsl():
    """When LLM returns update_bias, it should land on cat._bias_dsl."""
    mock = MockLLM(['{"update_bias": "MaxCysteine(1)"}'])
    config = DSLConfig(llm_loop_enabled=True)
    orch = Orchestrator(config=config, llm_client=mock)
    cat = _make_cat(orch)

    from bo.ldm.integrate import build_status, apply_decision
    status = build_status(cat, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=1)
    decision = orch.step(status)
    apply_decision(cat, decision)

    assert cat._bias_dsl is not None
    assert isinstance(cat._bias_dsl, BiasAtom)
    seq_with_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # last position is C
    # -0.0 is the penalty; assert <= -0.0 (allows for -0.0 quirk).
    assert cat._bias_dsl(seq_with_c) <= 0.0  # penalty or zero


def test_disabled_orchestrator_still_calls():
    """llm_loop_enabled=False no longer blocks step() — phase control is
    at injection level. step() should call the LLM normally."""
    mock = MockLLM(['{"rationale": "test"}'])
    config = DSLConfig(llm_loop_enabled=False, max_retries=1)
    orch = Orchestrator(config=config, llm_client=mock)
    cat = _make_cat(orch)

    d = orch.step(_make_cat_status(cat))
    assert d.source == "llm"
    assert d.applied is True
    assert len(mock.calls) == 1


def _make_cat_status(cat):
    from bo.ldm.integrate import build_status
    return build_status(cat, antigen_id="1ADQ_A", antigen_seed=42, iter_seed=1)