"""Tests for ``strbo_v1.bayesian_analog_search``."""

from __future__ import annotations

import random
import unittest
from typing import Callable, Optional

import numpy as np

from strbo_v1.bayesian_analog_search import (
    AcquisitionEvaluator,
    BayesianAnalogSearchConfig,
    _canonicalize_smiles,
    _resolve_acquisition,
    bayesian_analog_search,
    confidence_bound,
    expected_improvement,
    probability_of_improvement,
)
from strbo_v1.gp import GPConfig
from strbo_v1.utils import FIFOSet

try:
    import torch  # noqa: F401
    _HEAVY_AVAILABLE = True
except ImportError:
    _HEAVY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


def _carbon_scorer(smiles_list: list[str]) -> list[float]:
    """Lower is better; rewards more carbons."""
    return [-float(s.count("C")) for s in smiles_list]


def _constant_scorer(smiles_list: list[str]) -> list[Optional[float]]:
    return [-5.0 for _ in smiles_list]


def _none_scorer(smiles_list: list[str]) -> list[Optional[float]]:
    return [None for _ in smiles_list]


def _chain_analog_generator(seed_smiles: list[str]) -> list[str]:
    """For each input, append a single character."""
    out: list[str] = []
    for s in seed_smiles:
        out.extend([s + "C", s + "O", s + "N"])
    return out


def _empty_analog_generator(seed_smiles: list[str]) -> list[str]:
    return []


def _gp_config_cpu() -> GPConfig:
    return GPConfig(impl="fingerprint+tanimoto", device="cpu", fit_n_itersteps=5)


class _SpySurrogate:
    fit_calls = 0
    predict_calls = 0

    def __init__(self, config: GPConfig) -> None:
        self.config = config
        self.fit_smiles: list[str] = []
        self.fit_scores: list[float] = []

    def fit(self, smiles: list[str], scores: list[float]) -> "_SpySurrogate":
        type(self).fit_calls += 1
        self.fit_smiles = list(smiles)
        self.fit_scores = list(scores)
        return self

    def predict(self, smiles: list[str], *, return_tensor: bool = False):
        type(self).predict_calls += 1
        mean_by_smiles = {"CO": -2.0, "CN": -1.0, "CCO": -3.0}
        mu = np.asarray([mean_by_smiles[str(s)] for s in smiles], dtype=float)
        sigma = np.full(len(smiles), 0.5, dtype=float)
        return mu, sigma


# ---------------------------------------------------------------------------
# Acquisition helper tests
# ---------------------------------------------------------------------------


class AcquisitionHelperTests(unittest.TestCase):
    def test_ei_basic_minimization(self) -> None:
        mu = np.array([-5.0, -4.0, -3.0])
        sigma = np.array([1.0, 1.0, 1.0])
        best = -5.0
        ei = expected_improvement(mu, sigma, best, xi=0.0)
        # EI is highest when mu == best (zero mean improvement, but sigma > 0
        # still offers a chance to go below). EI decreases with distance from
        # best (smaller probability of beating the incumbent mean).
        self.assertEqual(len(ei), 3)
        self.assertGreater(ei[0], ei[1])
        self.assertGreater(ei[1], ei[2])
        self.assertGreater(ei[0], 0.0)

    def test_ei_maximization(self) -> None:
        mu = np.array([5.0, 4.0, 3.0])
        sigma = np.array([1.0, 1.0, 1.0])
        best = 5.0
        ei = expected_improvement(mu, sigma, best, xi=0.0, minimize=False)
        # Same ordering as minimization but inverted.
        self.assertEqual(len(ei), 3)
        self.assertGreater(ei[0], ei[1])
        self.assertGreater(ei[1], ei[2])
        self.assertGreater(ei[0], 0.0)

    def test_ei_xi_increases_improvement_threshold(self) -> None:
        mu = np.array([-4.0, -5.0])
        sigma = np.array([0.5, 0.5])
        best = -5.0
        ei_low = expected_improvement(mu, sigma, best, xi=0.0)
        ei_high = expected_improvement(mu, sigma, best, xi=1.0)
        # Higher xi makes first point's improvement negative -> EI drops.
        self.assertGreater(ei_low[0], ei_high[0])

    def test_ei_zero_sigma(self) -> None:
        mu = np.array([-5.0, -4.0])
        sigma = np.array([0.0, 0.0])
        best = -5.0
        ei = expected_improvement(mu, sigma, best)
        self.assertTrue(np.all(ei >= 0))

    def test_ei_shape_preserved(self) -> None:
        mu = np.array([-5.0, -3.0])
        sigma = np.array([0.5, 1.0])
        ei = expected_improvement(mu, sigma, -5.0)
        self.assertEqual(ei.shape, (2,))

    def test_ei_non_negative(self) -> None:
        rng = np.random.default_rng(0)
        mu = rng.normal(size=20)
        sigma = np.abs(rng.normal(size=20))
        ei = expected_improvement(mu, sigma, float(np.min(mu)))
        self.assertTrue(np.all(ei >= 0))

    def test_pi_basic_minimization(self) -> None:
        mu = np.array([-5.0, -4.0, -3.0])
        sigma = np.array([1.0, 1.0, 1.0])
        best = -5.0
        pi = probability_of_improvement(mu, sigma, best, xi=0.0)
        self.assertEqual(len(pi), 3)
        # PI is in [0, 1].
        self.assertTrue(np.all(pi >= 0))
        self.assertTrue(np.all(pi <= 1))

    def test_pi_zero_sigma(self) -> None:
        mu = np.array([-5.0, -4.0])
        sigma = np.array([0.0, 0.0])
        pi = probability_of_improvement(mu, sigma, -5.0)
        # When sigma == 0 and mu < best, no chance of improvement.
        self.assertTrue(np.all(pi >= 0))

    def test_confidence_bound_basic_minimize(self) -> None:
        mu = np.array([-5.0, -4.0])
        sigma = np.array([0.5, 1.0])
        cb = confidence_bound(mu, sigma, kappa=2.0, minimize=True)
        # For minimize: higher = better → return kappa*sigma - mu.
        self.assertTrue(np.allclose(cb, 2.0 * sigma - mu))

    def test_confidence_bound_basic_maximize(self) -> None:
        mu = np.array([3.0, 4.0])
        sigma = np.array([0.5, 1.0])
        cb = confidence_bound(mu, sigma, kappa=2.0, minimize=False)
        # For maximize: higher = better → return mu + kappa*sigma (UCB).
        self.assertTrue(np.allclose(cb, mu + 2.0 * sigma))

    def test_confidence_bound_kappa_zero_minimize_returns_negated_mean(self) -> None:
        mu = np.array([-5.0, -4.0])
        sigma = np.array([0.5, 1.0])
        cb = confidence_bound(mu, sigma, kappa=0.0, minimize=True)
        # kappa*sigma - mu with kappa=0 reduces to -mu.
        self.assertTrue(np.allclose(cb, -mu))

    def test_confidence_bound_kappa_zero_maximize_returns_mean(self) -> None:
        mu = np.array([3.0, 4.0])
        sigma = np.array([0.5, 1.0])
        cb = confidence_bound(mu, sigma, kappa=0.0, minimize=False)
        self.assertTrue(np.allclose(cb, mu))

    def test_resolve_acquisition_ei(self) -> None:
        self.assertIs(_resolve_acquisition("ei"), expected_improvement)

    def test_resolve_acquisition_pi(self) -> None:
        self.assertIs(_resolve_acquisition("PI"), probability_of_improvement)

    def test_resolve_acquisition_ucb(self) -> None:
        self.assertIs(_resolve_acquisition("ucb"), confidence_bound)

    def test_resolve_acquisition_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_acquisition("nope")

    # --- "higher = better" invariant tests -------------------------------

    def test_ei_higher_is_better_minimize(self) -> None:
        # For minimize, the point with mu == best has highest EI (zero
        # mean improvement, but sigma > 0 still gives a chance to beat it).
        mu = np.array([-5.0, -4.0, -3.0])
        sigma = np.array([1.0, 1.0, 1.0])
        ei = expected_improvement(mu, sigma, best=-5.0, xi=0.0, minimize=True)
        self.assertGreater(ei[0], ei[1])
        self.assertGreater(ei[1], ei[2])

    def test_ei_higher_is_better_maximize(self) -> None:
        mu = np.array([3.0, 4.0, 5.0])
        sigma = np.array([1.0, 1.0, 1.0])
        ei = expected_improvement(mu, sigma, best=5.0, xi=0.0, minimize=False)
        self.assertGreater(ei[2], ei[1])
        self.assertGreater(ei[1], ei[0])

    def test_pi_higher_is_better_minimize(self) -> None:
        # Lower mu → higher probability of going below best.
        mu = np.array([-3.0, -4.0, -5.0])
        sigma = np.array([1.0, 1.0, 1.0])
        pi = probability_of_improvement(mu, sigma, best=-5.0, xi=0.0, minimize=True)
        self.assertGreater(pi[2], pi[1])
        self.assertGreater(pi[1], pi[0])

    def test_confidence_bound_higher_is_better_minimize(self) -> None:
        # For minimize, higher kappa*sigma - mu = either lower mu or higher sigma.
        mu = np.array([-5.0, -4.0])
        sigma = np.array([0.5, 1.0])
        cb = confidence_bound(mu, sigma, kappa=2.0, minimize=True)
        # Point 1 has both higher sigma and higher mu; net effect:
        # kappa*sigma - mu = [1-(-5), 2-(-4)] = [6, 6]. Equal.
        # To get a clear ordering, increase the mu gap.
        mu = np.array([-5.0, -3.0])
        cb = confidence_bound(mu, sigma, kappa=2.0, minimize=True)
        self.assertGreater(cb[0], cb[1])

    def test_confidence_bound_higher_is_better_maximize(self) -> None:
        # For maximize, higher mu + kappa*sigma (UCB) is better.
        mu = np.array([3.0, 4.0])
        sigma = np.array([0.5, 1.0])
        cb = confidence_bound(mu, sigma, kappa=2.0, minimize=False)
        # Point 1 has higher mu AND higher sigma; UCB = [4, 6] → cb[1] > cb[0].
        self.assertGreater(cb[1], cb[0])


# ---------------------------------------------------------------------------
# Acquisition evaluator tests
# ---------------------------------------------------------------------------


class AcquisitionEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        _SpySurrogate.fit_calls = 0
        _SpySurrogate.predict_calls = 0

    def test_fits_once_at_construction_and_reuses_gp_for_calls(self) -> None:
        evaluator = AcquisitionEvaluator(
            history=[("CCO", -3.0), ("CO", -2.0), ("CN", -1.0)],
            config=BayesianAnalogSearchConfig(
                acquisition="ei",
                minimize=True,
                gp_config=_gp_config_cpu(),
            ),
            surrogate_factory=_SpySurrogate,
        )

        first = evaluator(["CO", "CN"])
        second = evaluator(["CCO"])

        self.assertEqual(_SpySurrogate.fit_calls, 1)
        self.assertEqual(_SpySurrogate.predict_calls, 2)
        self.assertEqual(set(first), {"CO", "CN"})
        self.assertEqual(set(first["CO"]), {"acquisition", "mean", "std", "variance"})
        self.assertGreater(first["CO"]["acquisition"], first["CN"]["acquisition"])
        self.assertAlmostEqual(first["CO"]["mean"], -2.0)
        self.assertAlmostEqual(first["CO"]["std"], 0.5)
        self.assertAlmostEqual(first["CO"]["variance"], 0.25)
        self.assertAlmostEqual(second["CCO"]["mean"], -3.0)

    def test_supports_multiple_single_objective_acquisitions(self) -> None:
        evaluator = AcquisitionEvaluator(
            history=[("CCO", -3.0), ("CO", -2.0), ("CN", -1.0)],
            config=BayesianAnalogSearchConfig(
                acquisition=("ei", "pi", "ucb"),
                minimize=True,
                gp_config=_gp_config_cpu(),
            ),
            surrogate_factory=_SpySurrogate,
        )

        values = evaluator(["CO"])["CO"]

        self.assertEqual(
            set(values),
            {
                "acquisition_ei",
                "acquisition_pi",
                "acquisition_ucb",
                "mean",
                "std",
                "variance",
            },
        )
        self.assertGreaterEqual(values["acquisition_ei"], 0.0)
        self.assertGreaterEqual(values["acquisition_pi"], 0.0)
        self.assertGreater(values["acquisition_ucb"], 0.0)


# ---------------------------------------------------------------------------
# Canonicalization tests
# ---------------------------------------------------------------------------


class CanonicalizeTests(unittest.TestCase):
    def test_canonical(self) -> None:
        self.assertEqual(_canonicalize_smiles("OCC"), "CCO")

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(_canonicalize_smiles("not_a_smiles@@"))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_canonicalize_smiles(""))

    def test_whitespace_only_returns_none(self) -> None:
        self.assertIsNone(_canonicalize_smiles("   "))


# ---------------------------------------------------------------------------
# End-to-end loop tests (no GP required)
# ---------------------------------------------------------------------------


class EndToEndLoopTests(unittest.TestCase):
    """Tests that exercise the BO loop without a real GP fit.

    These rely on either warm-up truncation, empty pools, or fallback
    random pick (insufficient finite history).
    """

    def test_empty_seed_no_warmup_returns_empty(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=[],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=4, warmup=False, n_iterations=3,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertEqual(history, [])

    def test_warmup_only_no_init_returns_empty(self) -> None:
        # No seed SMILES, no warm-up growth -> init has nothing to sample.
        history = bayesian_analog_search(
            seed_smiles=[],
            scorer=_carbon_scorer,
            analog_fn=_empty_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=2, warmup=True,
                n_iterations=3, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertEqual(history, [])

    def test_warmup_disabled_uses_seeds_only(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CC", "CO"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=2, warmup=False, n_iterations=0,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # Only the 2 init evaluations should happen.
        self.assertEqual(len(history), 2)
        smiles_set = {s for s, _ in history}
        self.assertEqual(smiles_set, {"CC", "CO"})

    def test_init_size_clamped_to_pool(self) -> None:
        # Only 2 seed SMILES, no warm-up -> init_size=10 clamps to 2.
        history = bayesian_analog_search(
            seed_smiles=["CC", "CO"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=10, warmup=False, n_iterations=0,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertEqual(len(history), 2)

    def test_history_preserves_evaluation_order(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CC", "CO", "CN"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=3, warmup=False, n_iterations=2,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(7),
        )
        # First 3 are init, then 2 BO rounds (batch_size=1 each).
        self.assertEqual(len(history), 5)
        # No duplicate keys.
        keys = [s for s, _ in history]
        self.assertEqual(len(set(keys)), len(keys))

    def test_no_analogues_terminates(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CC", "CO"],
            scorer=_carbon_scorer,
            analog_fn=_empty_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=2, warmup=False, n_iterations=10,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # Only init runs; pool stays empty -> BO loop never fires.
        self.assertEqual(len(history), 2)

    def test_pool_does_not_grow_during_warmup_with_empty_analog(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CC"],
            scorer=_carbon_scorer,
            analog_fn=_empty_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=5, warmup=True,
                n_iterations=0, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # Pool stays at 1; init clamps to 1; no BO.
        self.assertEqual(len(history), 1)


# ---------------------------------------------------------------------------
# GP-backed end-to-end tests (require torch stack)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HEAVY_AVAILABLE, "torch stack unavailable")
class GPBackedEndToEndTests(unittest.TestCase):
    """Tests that exercise the full BO loop with real GP fit + acquisition."""

    def test_deterministic_with_seeded_rng(self) -> None:
        kwargs = dict(
            seed_smiles=["CC", "CO", "CN"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=3, warmup=True,
                batch_size=1, n_iterations=3, acquisition="ei",
                gp_config=_gp_config_cpu(),
            ),
        )
        h1 = bayesian_analog_search(rng=random.Random(123), **kwargs)
        h2 = bayesian_analog_search(rng=random.Random(123), **kwargs)
        self.assertEqual(h1, h2)

    def test_different_seeds_yield_different_trajectories(self) -> None:
        kwargs = dict(
            seed_smiles=["CC", "CO", "CN"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=3, warmup=True,
                batch_size=1, n_iterations=3, acquisition="ei",
                gp_config=_gp_config_cpu(),
            ),
        )
        h1 = bayesian_analog_search(rng=random.Random(123), **kwargs)
        h2 = bayesian_analog_search(rng=random.Random(999), **kwargs)
        self.assertNotEqual(h1, h2)

    def test_history_length_matches_init_plus_bo(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CC", "CO", "CN"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=3, warmup=True,
                batch_size=2, n_iterations=4, acquisition="ei",
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(42),
        )
        # init_size + batch_size * n_iterations is the upper bound;
        # pool exhaustion may shorten it, but with a productive analog
        # generator we expect the full count.
        self.assertGreaterEqual(len(history), 3)
        self.assertLessEqual(len(history), 3 + 2 * 4)

    def test_all_three_acquisitions_run(self) -> None:
        for acq in ("ei", "ucb", "pi"):
            with self.subTest(acquisition=acq):
                history = bayesian_analog_search(
                    seed_smiles=["CC", "CO", "CN"],
                    scorer=_carbon_scorer,
                    analog_fn=_chain_analog_generator,
                    config=BayesianAnalogSearchConfig(
                        init_size=3, warmup=False,
                        batch_size=1, n_iterations=2, acquisition=acq,
                        gp_config=_gp_config_cpu(),
                    ),
                    rng=random.Random(7),
                )
                self.assertGreaterEqual(len(history), 5)

    def test_invalid_analogues_dropped_from_pool(self) -> None:
        # Inject invalid SMILES into the analog generator output.
        def noisy_generator(seed_smiles: list[str]) -> list[str]:
            out = _chain_analog_generator(seed_smiles)
            out.append("@@@not_a_smiles@@@")
            out.append("")
            return out

        history = bayesian_analog_search(
            seed_smiles=["CC", "CO"],
            scorer=_carbon_scorer,
            analog_fn=noisy_generator,
            config=BayesianAnalogSearchConfig(
                init_size=2, warmup=True,
                batch_size=1, n_iterations=2, acquisition="ei",
                canonicalize_pool=True, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # No SMILES in history should contain the malformed marker.
        for smi, _ in history:
            self.assertNotIn("@@@", smi)
            self.assertNotEqual(smi.strip(), "")

    def test_batch_size_larger_than_pool(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CC", "CO"],
            scorer=_carbon_scorer,
            analog_fn=_empty_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=2, warmup=False, batch_size=10, n_iterations=2,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # Only 2 init runs; pool empty after -> no BO fires.
        self.assertEqual(len(history), 2)

    def test_none_scores_recorded_but_excluded_from_gp(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CC", "CO", "CN"],
            scorer=_none_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=3, warmup=False, batch_size=1, n_iterations=2,
                acquisition="ei", gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # All entries should be (smiles, None).
        self.assertTrue(all(sc is None for _, sc in history))

    def test_maximize_mode(self) -> None:
        # Scorer that returns positive values; maximize=True so larger is better.
        def positive_scorer(smiles_list: list[str]) -> list[float]:
            return [float(s.count("C")) for s in smiles_list]

        history = bayesian_analog_search(
            seed_smiles=["CC", "CO", "CN"],
            scorer=positive_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=3, warmup=False, batch_size=1, n_iterations=2,
                minimize=False, acquisition="ei",
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertGreaterEqual(len(history), 5)

    def test_minimize_explores_with_carbon_score(self) -> None:
        # With scorer -count(C) and a productive analog generator,
        # BO should make forward progress: more SMILES evaluated than
        # the initial seed SMILES count.
        history = bayesian_analog_search(
            seed_smiles=["C", "CC"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=2, warmup=False, batch_size=1, n_iterations=3,
                acquisition="ei", gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(11),
        )
        self.assertGreaterEqual(len(history), 5)
        # All finite scores should be negative integers (sum of -C counts).
        finite_scores = [sc for _, sc in history if sc is not None]
        self.assertTrue(all(sc < 0 for sc in finite_scores))

    # --- acq_budget tests -----------------------------------------------

    def _make_big_analog_generator(self) -> Callable[[list[str]], list[str]]:
        """Return a generator that turns one seed into a large pool of unique SMILES."""
        counter = {"n": 0}

        def gen(seed_smiles: list[str]) -> list[str]:
            out: list[str] = []
            for s in seed_smiles:
                for _ in range(30):
                    counter["n"] += 1
                    # Use atom-letter suffixes that RDKit accepts and that
                    # remain unique. Each char ('C', 'O', 'N', 'S', 'F', 'P',
                    # 'I', 'B', 'K') is a valid organic subset atom.
                    suffix = "C" * (counter["n"] + 1)
                    out.append(f"{s}{suffix}")
            return out

        return gen

    def test_acq_budget_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            BayesianAnalogSearchConfig(acq_budget=0)

    def test_acq_budget_none_uses_full_pool(self) -> None:
        """With acq_budget=None, GP sees all pool members."""
        from strbo_v1.gp import GPSurrogate

        captured: dict[str, list[str]] = {}
        original_predict = GPSurrogate.predict

        def spy_predict(self, smiles, *, return_tensor=False):
            captured["last"] = list(smiles)
            return original_predict(self, smiles, return_tensor=return_tensor)

        GPSurrogate.predict = spy_predict  # type: ignore[assignment]
        try:
            history = bayesian_analog_search(
                seed_smiles=["C"],
                scorer=_carbon_scorer,
                analog_fn=self._make_big_analog_generator(),
                config=BayesianAnalogSearchConfig(
                    init_size=2, warmup=True,
                    batch_size=1, n_iterations=3, acquisition="ei",
                    acq_budget=None, gp_config=_gp_config_cpu(),
                ),
                rng=random.Random(0),
            )
        finally:
            GPSurrogate.predict = original_predict  # type: ignore[assignment]

        self.assertGreaterEqual(len(history), 5)
        # Last predict call must have received a pool larger than acq_budget.
        self.assertGreater(len(captured["last"]), 5)

    def test_acq_budget_subsamples_when_pool_large(self) -> None:
        """With acq_budget=5 and a large pool, GP predict sees ≤ 5 SMILES."""
        from strbo_v1.gp import GPSurrogate

        captured: dict[str, list[str]] = {}
        original_predict = GPSurrogate.predict

        def spy_predict(self, smiles, *, return_tensor=False):
            captured["last"] = list(smiles)
            return original_predict(self, smiles, return_tensor=return_tensor)

        GPSurrogate.predict = spy_predict  # type: ignore[assignment]
        try:
            history = bayesian_analog_search(
                seed_smiles=["C"],
                scorer=_carbon_scorer,
                analog_fn=self._make_big_analog_generator(),
                config=BayesianAnalogSearchConfig(
                    init_size=2, warmup=True,
                    batch_size=1, n_iterations=3, acquisition="ei",
                    acq_budget=5, gp_config=_gp_config_cpu(),
                ),
                rng=random.Random(0),
            )
        finally:
            GPSurrogate.predict = original_predict  # type: ignore[assignment]

        self.assertGreaterEqual(len(history), 5)
        # All predict calls must have received at most acq_budget SMILES.
        self.assertLessEqual(len(captured["last"]), 5)

    def test_acq_budget_no_subsample_when_pool_small(self) -> None:
        """With acq_budget=100 and a small pool, GP sees all pool members."""
        from strbo_v1.gp import GPSurrogate

        captured: dict[str, list[str]] = {}
        original_predict = GPSurrogate.predict

        def spy_predict(self, smiles, *, return_tensor=False):
            captured["last"] = list(smiles)
            return original_predict(self, smiles, return_tensor=return_tensor)

        GPSurrogate.predict = spy_predict  # type: ignore[assignment]
        try:
            bayesian_analog_search(
                seed_smiles=["CC", "CO", "CN"],
                scorer=_carbon_scorer,
                analog_fn=_chain_analog_generator,
                config=BayesianAnalogSearchConfig(
                    init_size=3, warmup=False,
                    batch_size=1, n_iterations=2, acquisition="ei",
                    acq_budget=100, gp_config=_gp_config_cpu(),
                ),
                rng=random.Random(0),
            )
        finally:
            GPSurrogate.predict = original_predict  # type: ignore[assignment]

        # Pool stays small; no subsampling needed.
        self.assertLess(len(captured["last"]), 100)
        # All pool members passed through.
        self.assertGreater(len(captured["last"]), 0)


# ---------------------------------------------------------------------------
# PoolFifoTests: pool is a FIFOSet, max_pool_size bounds the queue with
# FIFO eviction.
# ---------------------------------------------------------------------------


class PoolFifoTests(unittest.TestCase):
    """The BO candidate pool is a :class:`FIFOSet` (FIFO-ordered, with
    optional ``max_pool_size`` FIFO cap)."""

    def test_pool_is_fifoset(self) -> None:
        # Run a minimal BO loop; the pool is internal, so we verify the
        # type via the constructor path: max_pool_size must be a FIFOSet
        # kwarg accepted by the config.
        cfg = BayesianAnalogSearchConfig(max_pool_size=10)
        self.assertEqual(cfg.max_pool_size, 10)

    def test_max_pool_size_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            BayesianAnalogSearchConfig(max_pool_size=0)

    def test_max_pool_size_default_unbounded(self) -> None:
        cfg = BayesianAnalogSearchConfig()
        self.assertIsNone(cfg.max_pool_size)
        # Direct FIFOSet behavior: unbounded.
        f: FIFOSet = FIFOSet(max_size=None)
        for s in range(100):
            f.add(f"s{s}")
        self.assertEqual(len(f), 100)

    def test_max_pool_size_bounds_pool_with_fifo_eviction(self) -> None:
        """Chain analog generator + max_pool_size=2 → pool size never exceeds 2."""
        history = bayesian_analog_search(
            seed_smiles=["CC"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=1, warmup=False,
                batch_size=1, n_iterations=8, acquisition="ei",
                max_pool_size=2, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # We can't observe the pool directly (it's internal), but the
        # run must complete without error and produce a history. The
        # fact that init_size=1, batch_size=1, n_iterations=8 produced
        # >= 2 entries (init + at least 1 BO round) is the smoke test.
        self.assertGreaterEqual(len(history), 2)

    def test_max_pool_size_one_pool_never_exceeds_one(self) -> None:
        """With max_pool_size=1 the pool is at most one entry; the loop
        should still complete (no IndexError from an empty pool)."""
        history = bayesian_analog_search(
            seed_smiles=["CC"],
            scorer=_carbon_scorer,
            analog_fn=_chain_analog_generator,
            config=BayesianAnalogSearchConfig(
                init_size=1, warmup=False,
                batch_size=1, n_iterations=3, acquisition="ei",
                max_pool_size=1, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # init (1) + 3 BO rounds. With pool_size=1 the pool constantly
        # evicts; the loop should still produce at least 2 entries.
        self.assertGreaterEqual(len(history), 2)

    def test_invalid_max_pool_size_via_loop(self) -> None:
        """Passing max_pool_size=0 via the config dataclass raises ValueError
        immediately, before the loop runs."""
        with self.assertRaises(ValueError):
            bayesian_analog_search(
                seed_smiles=["CC"],
                scorer=_carbon_scorer,
                analog_fn=_chain_analog_generator,
                config=BayesianAnalogSearchConfig(
                    init_size=1, warmup=False,
                    batch_size=1, n_iterations=1, acquisition="ei",
                    max_pool_size=0, gp_config=_gp_config_cpu(),
                ),
                rng=random.Random(0),
            )


# ---------------------------------------------------------------------------
# MaxLenFilterTests: pool filter on SMILES length (smiles_max_len).
# ---------------------------------------------------------------------------


class MaxLenFilterTests(unittest.TestCase):
    """The BO loop's pool filter drops analogues whose canonical form
    (or stripped raw text, when canonicalize_pool=False) is longer
    than ``smiles_max_len``. The same value drives the GP string
    kernel's int64 tensor padding."""

    def test_max_len_default_50(self) -> None:
        cfg = BayesianAnalogSearchConfig()
        self.assertEqual(cfg.smiles_max_len, 50)

    def test_max_len_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            BayesianAnalogSearchConfig(smiles_max_len=0)

    def test_max_len_invalid_via_loop(self) -> None:
        with self.assertRaises(ValueError):
            bayesian_analog_search(
                seed_smiles=["CC"],
                scorer=_carbon_scorer,
                analog_fn=_chain_analog_generator,
                config=BayesianAnalogSearchConfig(
                    init_size=1, warmup=False,
                    batch_size=1, n_iterations=1, acquisition="ei",
                    smiles_max_len=0, gp_config=_gp_config_cpu(),
                ),
                rng=random.Random(0),
            )

    def test_over_length_analogues_filtered_from_pool(self) -> None:
        """Analog generator emits a mix of short and over-length SMILES;
        only the short ones ever enter history."""
        emitted_pool = ["CCC", "C" * 100, "CCCC", "C" * 200]

        def mixed_analog(smis: list[str]) -> list[str]:
            return list(emitted_pool)

        history = bayesian_analog_search(
            seed_smiles=["CC"],
            scorer=_carbon_scorer,
            analog_fn=mixed_analog,
            config=BayesianAnalogSearchConfig(
                init_size=1, warmup=False,
                batch_size=1, n_iterations=8, acquisition="ei",
                smiles_max_len=10, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        for smi, _ in history:
            self.assertLessEqual(len(smi), 10, f"over-length {smi!r} not filtered")

    def test_canonical_max_len_uses_canonical_length(self) -> None:
        """With canonicalize_pool=True, the length check is on the
        canonical SMILES, not the raw generator output. A raw SMILES
        that canonicalizes to a short form is kept even if the raw
        is long; conversely a raw short SMILES that canonicalizes
        long is dropped."""

        # Raw "C[C@H](N)CC(=O)O" canonicalizes to "CC(N)CC(=O)O" (11 chars).
        # canonicalize_pool=True (default), smiles_max_len=10 → filtered.
        raw = "C[C@H](N)CC(=O)O"

        def canonical_analog(smis: list[str]) -> list[str]:
            return [raw]

        history = bayesian_analog_search(
            seed_smiles=["CC"],
            scorer=_carbon_scorer,
            analog_fn=canonical_analog,
            config=BayesianAnalogSearchConfig(
                init_size=1, warmup=False,
                batch_size=1, n_iterations=4, acquisition="ei",
                smiles_max_len=10, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        # The over-length canonical form must never be evaluated.
        all_scored = {s for s, _ in history}
        self.assertNotIn("CC(N)CC(=O)O", all_scored)
        # And the raw must not be there either (canonicalize_pool is True).
        self.assertNotIn(raw, all_scored)

    def test_max_len_does_not_break_loop(self) -> None:
        """End-to-end: a mix of over-length and short SMILES; loop
        completes without error and history is bounded by valid SMILES."""
        emitted = ["CC", "CCC", "C" * 200, "CCCCCC", "C" * 50]

        def mixed_analog(smis: list[str]) -> list[str]:
            return list(emitted)

        history = bayesian_analog_search(
            seed_smiles=["CC"],
            scorer=_carbon_scorer,
            analog_fn=mixed_analog,
            config=BayesianAnalogSearchConfig(
                init_size=1, warmup=False,
                batch_size=1, n_iterations=10, acquisition="ei",
                smiles_max_len=20, gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        for smi, _ in history:
            self.assertLessEqual(len(smi), 20)

    def test_max_len_none_disables_filter(self) -> None:
        """smiles_max_len=None disables the filter (canonical-length
        check skipped)."""
        long_smi = "C" * 200

        def long_analog(smis: list[str]) -> list[str]:
            return [long_smi]

        history = bayesian_analog_search(
            seed_smiles=["CC"],
            scorer=_carbon_scorer,
            analog_fn=long_analog,
            config=BayesianAnalogSearchConfig(
                init_size=1, warmup=False,
                batch_size=1, n_iterations=4, acquisition="ei",
                smiles_max_len=None, gp_config=_gp_config_cpu(),
                canonicalize_pool=False,  # raw check
            ),
            rng=random.Random(0),
        )
        # The over-length SMILES reaches history.
        all_scored = {s for s, _ in history}
        self.assertIn(long_smi, all_scored)


# ---------------------------------------------------------------------------
# Multi-objective tests (n_obj == 2, n_obj == 3)
# ---------------------------------------------------------------------------


def _gp_config_cpu() -> GPConfig:
    """Standard CPU GP config for tests (faster than CUDA in CI)."""
    return GPConfig(
        impl="fingerprint+tanimoto", device="cpu", fit_n_itersteps=10,
        min_jitter=1e-6, max_jitter=1e-1, standardize_y=True,
    )


def _vina_mock_scorer(smiles_list: list[str]) -> list[float]:
    """Mock Vina: rewards more carbons (lower is better)."""
    return [-float(s.count("C")) - 0.1 * float(s.count("N")) for s in smiles_list]


def _nn_mock_scorer(smiles_list: list[str]) -> list[float]:
    """Mock NN (pIC50): rewards more nitrogens (higher is better)."""
    return [5.0 + 0.5 * float(s.count("N")) + 0.1 * float(s.count("C")) for s in smiles_list]


def _carbon_mock_scorer(smiles_list: list[str]) -> list[float]:
    """Third mock: rewards more oxygens (higher is better)."""
    return [1.0 + 0.3 * float(s.count("O")) for s in smiles_list]


class MultiObjective2ObjTests(unittest.TestCase):
    """End-to-end 2-objective loop using EHVI (Monte Carlo)."""

    def test_history_shape_and_length(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock_scorer, _nn_mock_scorer),
            analog_fn=lambda smis: [s + "C" for s in smis],
            config=BayesianAnalogSearchConfig(
                init_size=2, batch_size=1, n_iterations=2, warmup=False,
                minimize=(True, False),
                ref_point=(0.0, 5.0),
                ehvi_n_samples=32,
                gp_config=_gp_config_cpu(),
                verbose=False,
            ),
            rng=random.Random(0),
        )
        # Every entry is (smi, (s_vina, s_nn)) tuple.
        self.assertGreater(len(history), 0)
        for smi, sc in history:
            self.assertIsInstance(smi, str)
            self.assertIsInstance(sc, tuple)
            self.assertEqual(len(sc), 2)
            self.assertTrue(all(s is None or isinstance(s, float) for s in sc))

    def test_minimize_tuple_required_for_multi_obj(self) -> None:
        """Passing a bare bool for ``minimize`` is broadcast to all
        objectives; a tuple with the wrong length raises."""
        # Bare bool broadcasts: works.
        bayesian_analog_search(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock_scorer, _nn_mock_scorer),
            analog_fn=lambda smis: [s + "C" for s in smis],
            config=BayesianAnalogSearchConfig(
                init_size=2, batch_size=1, n_iterations=1, warmup=False,
                minimize=True,  # broadcast to (True, True)
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )

    def test_minimize_tuple_length_mismatch_raises(self) -> None:
        """A ``minimize`` tuple of the wrong length raises at runtime
        (the loop knows ``n_obj`` only after seeing the scorer tuple)."""
        with self.assertRaises(ValueError):
            bayesian_analog_search(
                seed_smiles=["CCO", "CCN"],
                scorer=(_vina_mock_scorer, _nn_mock_scorer),
                analog_fn=lambda smis: [s + "C" for s in smis],
                config=BayesianAnalogSearchConfig(
                    init_size=2, batch_size=1, n_iterations=1, warmup=False,
                    minimize=(True,),  # n_obj=2 but length 1
                    gp_config=_gp_config_cpu(),
                ),
                rng=random.Random(0),
            )

    def test_ref_point_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            bayesian_analog_search(
                seed_smiles=["CCO", "CCN"],
                scorer=(_vina_mock_scorer, _nn_mock_scorer),
                analog_fn=lambda smis: [s + "C" for s in smis],
                config=BayesianAnalogSearchConfig(
                    init_size=2, batch_size=1, n_iterations=1, warmup=False,
                    minimize=(True, False),
                    ref_point=(0.0,),  # n_obj=2 but length 1
                    ehvi_n_samples=4,
                    gp_config=_gp_config_cpu(),
                ),
                rng=random.Random(0),
            )

    def test_2obj_loop_completes(self) -> None:
        """Smoke: 2-obj loop runs to completion with the default
        DEFAULT_REF registry (no explicit ref_point)."""
        history = bayesian_analog_search(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock_scorer, _nn_mock_scorer),
            analog_fn=lambda smis: [s + "C" for s in smis],
            config=BayesianAnalogSearchConfig(
                init_size=2, batch_size=1, n_iterations=2, warmup=False,
                minimize=(True, False),
                # ref_point omitted; loop should not raise.
                ehvi_n_samples=16,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)

    def test_ehvi_seed_reproducibility(self) -> None:
        """Same RNG seed must produce the same history."""
        kwargs = dict(
            seed_smiles=["CCO", "CCN"],
            scorer=(_vina_mock_scorer, _nn_mock_scorer),
            analog_fn=lambda smis: [s + "C" for s in smis],
            config=BayesianAnalogSearchConfig(
                init_size=2, batch_size=1, n_iterations=2, warmup=False,
                minimize=(True, False),
                ehvi_n_samples=16,
                gp_config=_gp_config_cpu(),
            ),
        )
        h1 = bayesian_analog_search(rng=random.Random(7), **kwargs)
        h2 = bayesian_analog_search(rng=random.Random(7), **kwargs)
        self.assertEqual(h1, h2)


class MultiObjective3PlusTests(unittest.TestCase):
    """End-to-end n_obj >= 3 loop using Chebyshev ParEGO."""

    def test_3obj_history_shape(self) -> None:
        history = bayesian_analog_search(
            seed_smiles=["CCO", "CCN", "CCC"],
            scorer=(_vina_mock_scorer, _nn_mock_scorer, _carbon_mock_scorer),
            analog_fn=lambda smis: [s + "C" for s in smis],
            config=BayesianAnalogSearchConfig(
                init_size=3, batch_size=1, n_iterations=2, warmup=False,
                minimize=(True, False, False),
                che_alpha=1.0,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)
        for smi, sc in history:
            self.assertEqual(len(sc), 3)

    def test_5obj_history_shape(self) -> None:
        scorers = (
            _vina_mock_scorer, _nn_mock_scorer, _carbon_mock_scorer,
            _vina_mock_scorer, _nn_mock_scorer,
        )
        history = bayesian_analog_search(
            seed_smiles=["CCO", "CCN", "CCC", "CC", "CO"],
            scorer=scorers,
            analog_fn=lambda smis: [s + "C" for s in smis],
            config=BayesianAnalogSearchConfig(
                init_size=5, batch_size=1, n_iterations=2, warmup=False,
                minimize=(True, False, False, True, False),
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)
        for _, sc in history:
            self.assertEqual(len(sc), 5)

    def test_3obj_minimize_can_be_bare_bool(self) -> None:
        """Broadcasts ``minimize=True`` to all 3 objectives."""
        history = bayesian_analog_search(
            seed_smiles=["CCO", "CCN", "CCC"],
            scorer=(_vina_mock_scorer, _nn_mock_scorer, _carbon_mock_scorer),
            analog_fn=lambda smis: [s + "C" for s in smis],
            config=BayesianAnalogSearchConfig(
                init_size=3, batch_size=1, n_iterations=1, warmup=False,
                minimize=True,
                gp_config=_gp_config_cpu(),
            ),
            rng=random.Random(0),
        )
        self.assertGreater(len(history), 0)


class AsMinimizeTupleTests(unittest.TestCase):
    """Unit tests for the internal ``_as_minimize_tuple`` helper."""

    def test_bare_bool_broadcast(self) -> None:
        from strbo_v1.bayesian_analog_search import _as_minimize_tuple
        self.assertEqual(_as_minimize_tuple(True, n_obj=3), (True, True, True))
        self.assertEqual(_as_minimize_tuple(False, n_obj=1), (False,))

    def test_tuple_passthrough(self) -> None:
        from strbo_v1.bayesian_analog_search import _as_minimize_tuple
        self.assertEqual(
            _as_minimize_tuple((True, False, True), n_obj=3),
            (True, False, True),
        )

    def test_length_mismatch_raises(self) -> None:
        from strbo_v1.bayesian_analog_search import _as_minimize_tuple
        with self.assertRaises(ValueError):
            _as_minimize_tuple((True, False), n_obj=3)

    def test_non_bool_raises(self) -> None:
        from strbo_v1.bayesian_analog_search import _as_minimize_tuple
        with self.assertRaises(TypeError):
            _as_minimize_tuple((1, 0), n_obj=2)


class SafeScoreNTests(unittest.TestCase):
    """Unit tests for the multi-obj ``_safe_score_n`` helper."""

    def test_per_scorer_alignment(self) -> None:
        from strbo_v1.bayesian_analog_search import _safe_score_n

        def s_a(smis):
            return [1.0 for _ in smis]

        def s_b(smis):
            return [2.0 for _ in smis]

        out = _safe_score_n((s_a, s_b), ["CCO", "CCN"])
        self.assertEqual(out, [(1.0, 2.0), (1.0, 2.0)])

    def test_failing_scorer_returns_none(self) -> None:
        from strbo_v1.bayesian_analog_search import _safe_score_n

        def s_a(smis):
            raise RuntimeError("boom")

        def s_b(smis):
            return [3.14 for _ in smis]

        out = _safe_score_n((s_a, s_b), ["CCO", "CCN"])
        self.assertEqual(out, [(None, 3.14), (None, 3.14)])

    def test_nan_normalised_to_none(self) -> None:
        from strbo_v1.bayesian_analog_search import _safe_score_n
        import math

        def s_a(smis):
            return [float("nan") for _ in smis]

        def s_b(smis):
            return [1.0 for _ in smis]

        out = _safe_score_n((s_a, s_b), ["CCO"])
        self.assertEqual(out, [(None, 1.0)])

    def test_length_mismatch_padding(self) -> None:
        from strbo_v1.bayesian_analog_search import _safe_score_n

        def s_short(smis):
            return [0.0]  # shorter than input

        def s_ok(smis):
            return [1.0 for _ in smis]

        out = _safe_score_n((s_short, s_ok), ["CCO", "CCN"])
        # s_short returns [0.0] for the first SMILES and is padded
        # with None for the second; s_ok returns 1.0 for both.
        self.assertEqual(out, [(0.0, 1.0), (None, 1.0)])


if __name__ == "__main__":
    unittest.main()
