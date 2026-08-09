"""tests/bo/ldm/dsl/test_atoms.py — tests for LatinHyperCubeSampling, NeighborSampling, LocalSearch, Or."""
from __future__ import annotations

import numpy as np
import pytest

from bo.ldm.dsl.alphabet import SEQ_LEN, aa_to_idx, hamming
from bo.ldm.dsl.exceptions import SamplingTimeout
from bo.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
    SearchSpaceAtom,
)


CENTER = "ARDYGNYWYFD"
CENTER_IDX = [aa_to_idx(c) for c in CENTER]


class TestLatinHyperCubeSampling:
    def test_basic(self):
        a = LatinHyperCubeSampling(num=100)
        assert a.num == 100
        assert a.budget == 100

    def test_invalid_num(self):
        with pytest.raises(ValueError):
            LatinHyperCubeSampling(num=0)

    def test_contains_always_true(self):
        a = LatinHyperCubeSampling(num=10)
        assert [0] * 11 in a
        assert [19] * 11 in a

    def test_sample_returns_unique(self):
        a = LatinHyperCubeSampling(num=100)
        rng = np.random.default_rng(42)
        samples = a.sample(50, rng=rng, timeout_s=10.0)
        assert len(samples) == 50
        keys = {tuple(s) for s in samples}
        assert len(keys) == 50

    def test_repr(self):
        assert repr(LatinHyperCubeSampling(num=500)) == "LatinHyperCubeSampling(num=500)"


class TestNeighborSampling:
    def test_basic(self):
        a = NeighborSampling(CENTER, mut_pr=0.5, budget=500)
        assert a.budget == 500
        assert a.center == CENTER

    def test_invalid_mut_pr(self):
        with pytest.raises(ValueError):
            NeighborSampling(CENTER, mut_pr=0.0, budget=10)

    def test_contains(self):
        a = NeighborSampling(CENTER, radius=3)
        assert CENTER_IDX in a
        seq = list(CENTER_IDX); seq[0] = (seq[0] + 1) % 20
        assert seq in a
        seq = list(CENTER_IDX)
        for i in range(4):
            seq[i] = (seq[i] + 1) % 20
        assert seq not in a

    def test_radius_none_contains_all(self):
        a = NeighborSampling(CENTER, radius=None, budget=10)
        assert [0] * 11 in a
        assert [19] * 11 in a

    def test_fixed_positions(self):
        a = NeighborSampling(CENTER, fixed="***....****", radius=None, budget=10)
        seq = list(CENTER_IDX)
        seq[5] = (seq[5] + 1) % 20  # position 5 is fixed
        assert seq not in a

    def test_sample_produces_valid_seqs(self):
        a = NeighborSampling(CENTER, mut_pr=0.8, radius=4, budget=100)
        rng = np.random.default_rng(42)
        samples = a.sample(50, rng=rng, timeout_s=10.0)
        assert len(samples) == 50
        for s in samples:
            assert s in a

    def test_repr(self):
        a = NeighborSampling(CENTER, mut_pr=0.5, budget=1000)
        r = repr(a)
        assert "NeighborSampling" in r
        assert CENTER in r


class TestLocalSearch:
    def test_basic(self):
        a = LocalSearch(CENTER, radius=3, restart=2, steps=100)
        assert a.restart == 2
        assert a.steps == 100
        assert a.budget == 2 * (100 + 1)  # restart * (steps + 1) for center evals

    def test_invalid_restart(self):
        with pytest.raises(ValueError):
            LocalSearch(CENTER, restart=0, steps=100)

    def test_invalid_steps(self):
        with pytest.raises(ValueError):
            LocalSearch(CENTER, restart=1, steps=0)

    def test_contains(self):
        a = LocalSearch(CENTER, radius=2)
        assert CENTER_IDX in a
        seq = list(CENTER_IDX)
        for i in range(3):
            seq[i] = (seq[i] + 1) % 20
        assert seq not in a

    def test_sample_one_raises(self):
        a = LocalSearch(CENTER, restart=1, steps=10)
        rng = np.random.default_rng(42)
        with pytest.raises(NotImplementedError):
            a.sample_one(rng)

    def test_repr(self):
        a = LocalSearch(CENTER, radius=3, restart=2, steps=100)
        r = repr(a)
        assert "LocalSearch" in r
        assert "restart=2" in r
        assert "steps=100" in r


class TestOr:
    def test_basic(self):
        a = LocalSearch(CENTER, restart=2, steps=100)
        b = LocalSearch("VRGYYSDWYMD", restart=1, steps=80)
        atom = Or(a, b)
        assert atom.budget == 2 * (100 + 1) + 1 * (80 + 1)

    def test_contains_any(self):
        a = LocalSearch(CENTER, radius=2)
        b = LocalSearch("VRGYYSDWYMD", radius=2)
        atom = Or(a, b)
        assert CENTER_IDX in atom
        assert [aa_to_idx(c) for c in "VRGYYSDWYMD"] in atom

    def test_size_ub_sum(self):
        a = LocalSearch(CENTER, radius=0, restart=1, steps=1)
        b = LocalSearch("VRGYYSDWYMD", radius=0, restart=1, steps=1)
        atom = Or(a, b)
        assert atom.size_ub == a.size_ub + b.size_ub

    def test_requires_min_2(self):
        with pytest.raises(ValueError):
            Or(LocalSearch(CENTER, restart=1, steps=1))

    def test_mixed_atoms(self):
        a = LocalSearch(CENTER, restart=2, steps=100)
        b = NeighborSampling("VRGYYSDWYMD", budget=200)
        c = LatinHyperCubeSampling(num=100)
        atom = Or(a, b, c)
        assert atom.budget == 2 * (100 + 1) + 200 + 100
