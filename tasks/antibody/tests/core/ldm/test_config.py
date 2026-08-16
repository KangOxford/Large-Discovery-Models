"""tests/core/ldm/test_config.py"""
from __future__ import annotations

import pytest

from tasks.antibody.core.ldm.config import DSLConfig


class TestDSLConfig:
    def test_default_construction(self):
        c = DSLConfig()
        assert c.llm_temperature == 0.25
        assert c.bias_weight == 0.05
        assert c.acq_n_candidates == 5000
        assert c.init_pool_size == 100000
        assert c.sample_timeout_s == 5.0
        assert c.batch_size == 1
        assert c.atoms_whitelist[0] == "LatinHyperCubeSampling"

    def test_from_yaml_minimal(self):
        c = DSLConfig.from_yaml({})
        assert c.llm_temperature == 0.25

    def test_from_yaml_with_overrides(self):
        c = DSLConfig.from_yaml({"llm_temperature": 0.7, "acq_n_candidates": 7777})
        assert c.llm_temperature == 0.7
        assert c.acq_n_candidates == 7777
        assert c.init_pool_size == 100000
        assert c.sample_timeout_s == 5.0

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown LDM config keys"):
            DSLConfig.from_yaml({"bogus_key": 1})

    def test_frozen(self):
        c = DSLConfig()
        with pytest.raises(Exception):
            c.llm_temperature = 0.5  # type: ignore[misc]
