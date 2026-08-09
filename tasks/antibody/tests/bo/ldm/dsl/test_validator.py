"""tests/bo/ldm/dsl/test_validator.py"""
from __future__ import annotations

from bo.ldm.dsl.search_space import LocalSearch, NeighborSampling, Or
from bo.ldm.dsl.validator import validate_bias_atom, validate_search_atom


class TestValidateSearch:
    def test_ok_local_search(self):
        atom = LocalSearch("ARYYGSYWYFD", radius=2, restart=1, steps=10)
        errors = validate_search_atom(atom, sample_timeout_s=2.0)
        assert errors == []

    def test_ok_neighbor_sampling(self):
        atom = NeighborSampling("ARYYGSYWYFD", mut_pr=0.5, budget=100)
        errors = validate_search_atom(atom, sample_timeout_s=2.0)
        assert errors == []

    def test_ok_or(self):
        atom = Or(
            LocalSearch("ARYYGSYWYFD", radius=2, restart=1, steps=10),
            LocalSearch("VRGYYSDWYMD", radius=2, restart=1, steps=10),
        )
        errors = validate_search_atom(atom, sample_timeout_s=2.0)
        assert errors == []

    def test_too_deep(self):
        atom = LocalSearch("ARYYGSYWYFD", restart=1, steps=1)
        for _ in range(9):
            atom = Or(atom, LocalSearch("VRGYYSDWYMD", restart=1, steps=1))
        errors = validate_search_atom(atom, max_depth=8, sample_timeout_s=2.0)
        assert any("depth" in e for e in errors)


class TestValidateBias:
    def test_ok(self):
        from bo.ldm.dsl.bias import MaxCysteine
        assert validate_bias_atom(MaxCysteine(1)) == []

    def test_too_deep(self):
        from bo.ldm.dsl.bias import BiasSum, MaxCysteine
        atom = MaxCysteine(1)
        for _ in range(5):
            atom = BiasSum(atom, MaxCysteine(2))
        errors = validate_bias_atom(atom, max_depth=3)
        assert any("depth" in e for e in errors)
