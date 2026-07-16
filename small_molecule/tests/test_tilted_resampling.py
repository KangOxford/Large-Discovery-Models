import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from strbo_v1.ldm_tilted_case2.candidate_record import CandidateRecord
from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config
from strbo_v1.ldm_tilted_case2.resampling import (
    effective_sample_size,
    gumbel_top_k,
    probability_entropy,
    robust_z,
    selected_rank_by_ehvi,
    tilted_probabilities,
    weighted_sample_candidates,
)
from strbo_v1.rng import RNG


def test_robust_z_all_equal_returns_zero():
    z = robust_z(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(z, np.zeros(3))


def test_robust_z_clips_outlier():
    z = robust_z(np.array([0.0, 0.0, 100.0]), clip=1.0)
    assert float(z.max()) <= 1.0
    assert float(z.min()) >= -1.0


def test_eta_zero_respects_q0():
    cfg = TiltedLDMCase2Config(method="m1_direct_llm_sir", eta_ehvi_tilt=0.0)
    prob = tilted_probabilities(np.array([0.2, 0.8]), np.array([100.0, 0.0]), cfg)
    np.testing.assert_allclose(prob, np.array([0.2, 0.8]))


def test_alpha_zero_uses_ehvi_only():
    cfg = TiltedLDMCase2Config(method="m1_direct_llm_sir", alpha_base_measure=0.0)
    prob = tilted_probabilities(np.array([0.9, 0.1]), np.array([0.0, 5.0]), cfg)
    assert prob[1] > prob[0]


def test_alpha_zero_equal_ehvi_uniform():
    cfg = TiltedLDMCase2Config(method="m1_direct_llm_sir", alpha_base_measure=0.0)
    prob = tilted_probabilities(np.array([0.9, 0.1]), np.array([1.0, 1.0]), cfg)
    np.testing.assert_allclose(prob, np.array([0.5, 0.5]))


def test_eta_increase_shifts_mass_to_high_ehvi():
    q0 = np.array([0.5, 0.5])
    ehvi = np.array([0.0, 4.0])
    low = tilted_probabilities(q0, ehvi, TiltedLDMCase2Config("m1_direct_llm_sir", eta_ehvi_tilt=1.0))
    high = tilted_probabilities(q0, ehvi, TiltedLDMCase2Config("m1_direct_llm_sir", eta_ehvi_tilt=5.0))
    assert high[1] > low[1]


def test_probabilities_sum_to_one():
    prob = tilted_probabilities(
        np.array([2.0, 3.0, 5.0]),
        np.array([1.0, 2.0, 3.0]),
        TiltedLDMCase2Config("m1_direct_llm_sir"),
    )
    assert np.all(prob >= 0.0)
    assert np.isclose(float(prob.sum()), 1.0)


def test_gumbel_top_k_unique():
    idx = gumbel_top_k(np.array([0.2, 0.3, 0.5]), 3, RNG(seed=1))
    assert len(idx) == len(set(idx))
    assert sorted(idx) == [0, 1, 2]


def test_gumbel_top_k_reproducible_with_seed():
    prob = np.array([0.1, 0.2, 0.3, 0.4])
    assert gumbel_top_k(prob, 2, RNG(seed=2)) == gumbel_top_k(prob, 2, RNG(seed=2))


def test_weighted_sample_marks_selected():
    candidates = [
        CandidateRecord("A", "A", "m", ["s"], {"s": 1}),
        CandidateRecord("B", "B", "m", ["s"], {"s": 1}),
    ]
    selected = weighted_sample_candidates(candidates, np.array([0.0, 1.0]), 1, RNG(seed=3))
    assert selected == [candidates[1]]
    assert candidates[1].selected is True
    assert candidates[1].resampling_probability == 1.0


def test_effective_sample_size_bounds():
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    collapsed = np.array([1.0, 0.0, 0.0, 0.0])
    assert effective_sample_size(uniform) == 4.0
    assert effective_sample_size(collapsed) == 1.0
    assert probability_entropy(uniform) > probability_entropy(collapsed)


def test_selected_rank_by_ehvi():
    candidates = [
        CandidateRecord("A", "A", "m", ["s"], {"s": 1}, ehvi=0.1),
        CandidateRecord("B", "B", "m", ["s"], {"s": 1}, ehvi=0.5, selected=True),
        CandidateRecord("C", "C", "m", ["s"], {"s": 1}, ehvi=0.3),
    ]
    assert selected_rank_by_ehvi(candidates) == [1]
