import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.core.gp import GPConfig
from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.ehvi_all import compute_ehvi_for_candidates
from tasks.small_molecule.core.rng import RNG


def cfg():
    return TiltedLDMCase2Config(
        method="m1_direct_llm_sir",
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
        ehvi_n_samples=16,
    )


def candidates():
    return [
        CandidateRecord("CCO", "CCO", "m", ["s"], {"s": 1}),
        CandidateRecord("CCN", "CCN", "m", ["s"], {"s": 1}),
    ]


def history():
    return [
        ("CCO", (-1.0, 6.0)),
        ("CCN", (-0.8, 5.5)),
        ("CCC", (-0.5, 5.0)),
    ]


def test_ehvi_result_shape():
    result = compute_ehvi_for_candidates(history(), candidates(), cfg(), RNG(seed=0))
    assert result.ehvi.shape == (2,)
    assert np.all(result.ehvi >= 0.0)
    assert result.fallback_reason is None


def test_ehvi_uses_two_objectives_no_collapse():
    recs = candidates()
    result = compute_ehvi_for_candidates(history(), recs, cfg(), RNG(seed=1))
    assert len(result.mu_per_obj) == 2
    assert len(result.sigma_per_obj) == 2
    for rec in recs:
        assert rec.mu is not None and len(rec.mu) == 2
        assert rec.sigma is not None and len(rec.sigma) == 2
        assert rec.ehvi is not None


def test_ehvi_insufficient_history_fallback():
    recs = candidates()
    result = compute_ehvi_for_candidates([("CCO", (-1.0, 6.0))], recs, cfg(), RNG(seed=2))
    np.testing.assert_array_equal(result.ehvi, np.zeros(2))
    assert result.fallback_reason == "insufficient_history"
    assert all(rec.ehvi == 0.0 for rec in recs)


def test_ehvi_gp_failure_fallback(monkeypatch):
    def fail_fit(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr("tasks.small_molecule.core.ldm_tilted_case2.ehvi_all.GPSurrogate.fit", fail_fit)
    recs = candidates()
    result = compute_ehvi_for_candidates(history(), recs, cfg(), RNG(seed=3))
    np.testing.assert_array_equal(result.ehvi, np.zeros(2))
    assert result.fallback_reason == "gp_failed"
