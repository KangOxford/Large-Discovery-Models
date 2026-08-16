"""tests/core/ldm/dsl/test_bias_atoms.py"""
from __future__ import annotations

import pytest

from tasks.antibody.core.ldm.dsl.alphabet import aa_to_idx
from tasks.antibody.core.ldm.dsl.bias import (
    BiasSum,
    MaxAromatic,
    MaxCysteine,
    MaxHydrophobicRun,
    NetChargeRange,
    NoNGlycosylation,
)


def _seq(s: str) -> list[int]:
    return [aa_to_idx(c) for c in s]


class TestMaxCysteine:
    def test_zero_cys_returns_zero(self):
        atom = MaxCysteine(1)
        assert atom(_seq("AAA")) == 0.0

    def test_one_cys_at_threshold_returns_zero(self):
        atom = MaxCysteine(1)
        assert atom(_seq("CA")) == 0.0

    def test_two_cys_above_threshold_penalized(self):
        atom = MaxCysteine(1)
        assert atom(_seq("CC")) == -1.0

    def test_five_cys(self):
        atom = MaxCysteine(1)
        assert atom(_seq("CCCCC")) == -4.0


class TestMaxHydrophobicRun:
    def test_no_hydrophobic_run(self):
        atom = MaxHydrophobicRun(4)
        assert atom(_seq("RRRRR")) == 0.0  # R is polar

    def test_exactly_threshold(self):
        atom = MaxHydrophobicRun(4)
        # 4 consecutive A (hydrophobic): AAAA in HYDROPHOBIC set
        assert atom(_seq("AAAA")) == 0.0

    def test_above_threshold(self):
        atom = MaxHydrophobicRun(4)
        # 5 consecutive A
        assert atom(_seq("AAAAA")) == pytest.approx(-0.5)


class TestMaxAromatic:
    def test_zero_aromatic(self):
        atom = MaxAromatic(4)
        assert atom(_seq("RRRRR")) == 0.0

    def test_within_threshold(self):
        atom = MaxAromatic(4)
        # 2 F's: bonus min(2,2)*0.15 = 0.3
        assert atom(_seq("FF")) == pytest.approx(0.3)

    def test_above_threshold_penalized(self):
        atom = MaxAromatic(4)
        # 5 F's: bonus 0.3, penalty 0.25*(5-4)=0.25 -> 0.05
        assert atom(_seq("FFFFF")) == pytest.approx(0.05)


class TestNetChargeRange:
    def test_zero_charge_in_range(self):
        atom = NetChargeRange(-1.0, 1.0)
        assert atom(_seq("AAAAAAAAAA")) == 0.0

    def test_above_max(self):
        atom = NetChargeRange(-1.0, 1.0)
        # R = +1: 10 R's = charge 10
        assert atom(_seq("RRRRRRRRRR")) == pytest.approx(-0.5 * 9)  # 9 over max

    def test_below_min(self):
        atom = NetChargeRange(-1.0, 1.0)
        # D = -1: 10 D's = charge -10
        assert atom(_seq("DDDDDDDDDD")) == pytest.approx(-0.5 * 9)


class TestNoNGlycosylation:
    def test_no_motif(self):
        atom = NoNGlycosylation()
        assert atom(_seq("AAAAAAAAAA")) == 0.0

    def test_has_motif_NXS(self):
        atom = NoNGlycosylation()
        # N-A-S = N(11), A(0), S(15)
        seq = [11, 0, 15] + [0] * 8
        assert atom(seq) == -1.0

    def test_has_motif_NXT(self):
        atom = NoNGlycosylation()
        seq = [11, 0, 16] + [0] * 8  # T = 16
        assert atom(seq) == -1.0

    def test_motif_with_P_at_pos_2_allowed(self):
        atom = NoNGlycosylation()
        # N-P-S is NOT a glycomotif (regex requires [^P] at pos 2)
        seq = [11, 12, 15] + [0] * 8
        assert atom(seq) == 0.0


class TestBiasSum:
    def test_sum_of_atoms(self):
        a = MaxCysteine(1)
        b = NoNGlycosylation()
        s = a + b
        seq = _seq("CC")  # 2 Cys, no motif
        assert s(seq) == pytest.approx(a(seq) + b(seq))

    def test_chained_addition(self):
        a = MaxCysteine(1) + NoNGlycosylation() + MaxHydrophobicRun(4)
        seq = _seq("AAAAAAAAAA")
        expected = MaxCysteine(1)(seq) + NoNGlycosylation()(seq) + MaxHydrophobicRun(4)(seq)
        assert a(seq) == pytest.approx(expected)

    def test_invalid_atom_raises(self):
        with pytest.raises(TypeError):
            BiasSum("not a bias atom")

    def test_atoms_property(self):
        a = MaxCysteine(1)
        b = MaxAromatic(4)
        s = a + b
        assert s.atoms == [a, b]