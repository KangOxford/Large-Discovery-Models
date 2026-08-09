"""tests/bo/ldm/dsl/test_sandbox.py"""
from __future__ import annotations

import pytest

from bo.ldm.dsl.bias import BiasSum, MaxCysteine, NoNGlycosylation
from bo.ldm.dsl.exceptions import DSLSyntaxError
from bo.ldm.dsl.sandbox import safe_exec_dsl
from bo.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
    SearchSpaceAtom,
)

WHITELIST_SEARCH = ("LatinHyperCubeSampling", "NeighborSampling", "LocalSearch", "Or")
WHITELIST_BIAS = ("MaxCysteine", "MaxHydrophobicRun", "MaxAromatic",
                  "NetChargeRange", "NoNGlycosylation", "BiasSum")


class TestSafeExecSearch:
    def test_local_search(self):
        atom = safe_exec_dsl(
            "LocalSearch('ARYYGSYWYFD', radius=3, restart=2, steps=100)",
            whitelist=WHITELIST_SEARCH,
        )
        assert isinstance(atom, LocalSearch)
        assert atom.restart == 2

    def test_neighbor_sampling(self):
        atom = safe_exec_dsl(
            "NeighborSampling('ARYYGSYWYFD', mut_pr=0.5, budget=500)",
            whitelist=WHITELIST_SEARCH,
        )
        assert isinstance(atom, NeighborSampling)

    def test_lhs(self):
        atom = safe_exec_dsl("LatinHyperCubeSampling(num=1000)", whitelist=WHITELIST_SEARCH)
        assert isinstance(atom, LatinHyperCubeSampling)

    def test_or(self):
        src = "Or(LocalSearch('ARYYGSYWYFD', restart=2, steps=100), LocalSearch('VRGYYSDWYMD', restart=1, steps=80))"
        atom = safe_exec_dsl(src, whitelist=WHITELIST_SEARCH)
        assert isinstance(atom, Or)

    def test_pipe_operator(self):
        src = "LocalSearch('ARYYGSYWYFD') | LocalSearch('VRGYYSDWYMD')"
        atom = safe_exec_dsl(src, whitelist=WHITELIST_SEARCH)
        assert isinstance(atom, Or)

    def test_blocks_import(self):
        with pytest.raises(DSLSyntaxError):
            safe_exec_dsl("import os; LocalSearch('ARYYGSYWYFD')", whitelist=WHITELIST_SEARCH)

    def test_blocks_unknown(self):
        with pytest.raises((DSLSyntaxError, NameError)):
            safe_exec_dsl("NotAnAtom('x')", whitelist=WHITELIST_SEARCH)

    def test_syntax_error(self):
        with pytest.raises(DSLSyntaxError):
            safe_exec_dsl("LocalSearch('ARYYGSYWYFD',", whitelist=WHITELIST_SEARCH)

    def test_expect_kind(self):
        atom = safe_exec_dsl("LocalSearch('ARYYGSYWYFD')", whitelist=WHITELIST_SEARCH, expect_kind=LocalSearch)
        assert isinstance(atom, LocalSearch)


class TestSafeExecBias:
    def test_simple(self):
        atom = safe_exec_dsl("MaxCysteine(1)", whitelist=WHITELIST_BIAS)
        assert isinstance(atom, MaxCysteine)

    def test_sum(self):
        atom = safe_exec_dsl("MaxCysteine(1) + NoNGlycosylation()", whitelist=WHITELIST_BIAS)
        assert isinstance(atom, BiasSum)
