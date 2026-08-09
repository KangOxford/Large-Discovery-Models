"""tests/bo/ldm/dsl/test_or_atom.py"""
from __future__ import annotations

import numpy as np

from bo.ldm.dsl.alphabet import aa_to_idx, hamming
from bo.ldm.dsl.search_space import LocalSearch, NeighborSampling, Or


class TestOr:
    def test_contains_any(self):
        a = LocalSearch("ARYYGSYWYFD", radius=2)
        b = LocalSearch("VRGYYSDWYMD", radius=2)
        atom = Or(a, b)
        ca = [aa_to_idx(c) for c in "ARYYGSYWYFD"]
        cb = [aa_to_idx(c) for c in "VRGYYSDWYMD"]
        assert ca in atom
        assert cb in atom

    def test_budget_sum(self):
        a = LocalSearch("ARYYGSYWYFD", restart=2, steps=100)
        b = NeighborSampling("VRGYYSDWYMD", budget=200)
        atom = Or(a, b)
        assert atom.budget == 2 * (100 + 1) + 200

    def test_size_ub_sum(self):
        a = LocalSearch("ARYYGSYWYFD", radius=0, restart=1, steps=1)
        b = LocalSearch("VRGYYSDWYMD", radius=0, restart=1, steps=1)
        atom = Or(a, b)
        assert atom.size_ub == a.size_ub + b.size_ub

    def test_requires_min_2(self):
        import pytest
        with pytest.raises(ValueError):
            Or(LocalSearch("ARYYGSYWYFD"))
