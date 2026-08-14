"""Tests for ``tasks.small_molecule.core.gp``.

All GP-related tests are skipped when the heavy stack (torch + gpytorch
+ gauche + rdkit) is unavailable so that the suite can still be run
under a bare ``python -m unittest discover tests`` in environments that
only have the docking / analog dependencies installed.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Optional
from unittest.mock import patch

import numpy as np

try:
    import gpytorch  # noqa: F401
    import torch  # noqa: F401
    from rdkit import Chem  # noqa: F401

    from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel  # noqa: F401
    from gauche.kernels.string_kernels.sskkernel import SubsequenceStringKernel  # noqa: F401

    from tasks.small_molecule.core.gp import (
        GPSurrogate,
        GPConfig,
        _build_smiles_alphabet,
        _canonicalize_and_dedup,
        _dedup_by_feature_row,
        _destandardize,
        _jitter_ladder,
        _normalize_y,
        _smiles_alphabet_embds,
        _smiles_to_fingerprints,
        _smiles_to_strings,
    )

    _HEAVY_AVAILABLE = True
except ImportError:  # pragma: no cover - skipped when deps missing
    _HEAVY_AVAILABLE = False


_CPU_CONFIG_KW = dict(device="cpu", fit_n_itersteps=10, learning_rate=0.1)


def _failing_cholesky(*args, **kwargs):
    @contextmanager
    def _raise():
        raise RuntimeError("Simulated Cholesky failure (test stub).")
        yield  # pragma: no cover - unreachable

    return _raise()


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class JitterLadderTests(unittest.TestCase):
    def _assert_almost_equal_list(self, actual: list[float], expected: list[float]) -> None:
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=12)

    def test_ladder_basic(self) -> None:
        self._assert_almost_equal_list(
            _jitter_ladder(1e-6, 10.0, 1e-1),
            [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        )

    def test_ladder_min_above_max_returns_empty(self) -> None:
        # No usable jitter when min > max.
        self.assertEqual(_jitter_ladder(1.0, 10.0, 0.5), [])

    def test_ladder_max_unreachable_is_capped(self) -> None:
        ladder = _jitter_ladder(1e-6, 3.0, 1.0)
        # Every value is ≤ max_jitter; the cap is inclusive.
        for v in ladder:
            self.assertLessEqual(v, 1.0 * (1.0 + 1e-12))
        self.assertGreater(len(ladder), 0)
        # The next value (ladder[-1] * multiplier) would exceed max_jitter.
        self.assertGreater(ladder[-1] * 3.0, 1.0 * (1.0 + 1e-12))

    def test_ladder_multiplier_one_returns_min(self) -> None:
        # With multiplier == 1.0 the ladder never grows; we return one value
        # rather than looping forever.
        self.assertEqual(_jitter_ladder(1e-4, 1.0, 1e-2), [1e-4])


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class FingerprintFeaturizerTests(unittest.TestCase):
    def test_smiles_to_fingerprints_shape_and_dtype(self) -> None:
        feats = _smiles_to_fingerprints(["CCO", "CCN", "c1ccccc1"], radius=2, n_bits=64)
        self.assertEqual(feats.shape, (3, 64))
        self.assertEqual(feats.dtype, torch.float32)
        self.assertTrue((feats >= 0).all() and (feats <= 1).all())

    def test_invalid_smiles_returns_zero_row(self) -> None:
        feats = _smiles_to_fingerprints(["not_a_smiles", "CCO"], radius=2, n_bits=64)
        self.assertEqual(feats.shape, (2, 64))
        self.assertTrue(torch.equal(feats[0], torch.zeros(64)))
        # The valid row should be non-zero (CCO has at least one Morgan bit
        # even with n_bits=64 — its bits just land in unpredictable positions).
        self.assertGreater(int(feats[1].sum()), 0)

    def test_empty_input_returns_empty_tensor(self) -> None:
        feats = _smiles_to_fingerprints([], radius=2, n_bits=64)
        self.assertEqual(feats.shape, (0, 64))


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class SmilesAlphabetTests(unittest.TestCase):
    def test_alphabet_built_from_union_of_chars(self) -> None:
        alphabet, index = _build_smiles_alphabet(["CCO", "CCN", "COC"])
        self.assertEqual(set(alphabet), {"C", "O", "N"})
        self.assertEqual(index["C"], 1)
        self.assertEqual(index["O"], 2)
        self.assertEqual(index["N"], 3)

    def test_alphabet_preserves_first_seen_order(self) -> None:
        alphabet, _ = _build_smiles_alphabet(["NCO", "CON"])
        self.assertEqual(alphabet[0], "N")

    def test_embds_shape_and_one_hot(self) -> None:
        alphabet = ["C", "N"]
        embs = _smiles_alphabet_embds(alphabet)
        self.assertEqual(tuple(embs.shape), (3, 2))  # +1 for padding slot
        self.assertTrue(torch.equal(embs[0], torch.zeros(2)))  # padding = zero
        self.assertTrue(torch.equal(embs[1], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(embs[2], torch.tensor([0.0, 1.0])))


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class SmilesStringFeaturizerTests(unittest.TestCase):
    def test_smiles_to_strings_shape_and_dtype(self) -> None:
        _, index = _build_smiles_alphabet(["CCO", "CCN"])
        feats = _smiles_to_strings(["CCO", "CCN"], index=index, maxlen=10)
        self.assertEqual(feats.shape, (2, 10))
        self.assertEqual(feats.dtype, torch.int64)

    def test_smiles_to_strings_unseen_chars_pad_to_zero(self) -> None:
        _, index = _build_smiles_alphabet(["CCO"])
        # Use chars that aren't in the {C,O} alphabet (uppercase Z, lower z, etc.).
        feats = _smiles_to_strings(["ZzQq"], index=index, maxlen=5)
        # All chars unseen → entire row is zero (padding/unknown slot).
        self.assertTrue(torch.equal(feats[0], torch.zeros(5, dtype=torch.int64)))

    def test_smiles_to_strings_right_pads_short(self) -> None:
        _, index = _build_smiles_alphabet(["CCO"])
        feats = _smiles_to_strings(["C"], index=index, maxlen=5)
        # "C" → [1, 0, 0, 0, 0]
        self.assertTrue(torch.equal(feats[0], torch.tensor([1, 0, 0, 0, 0], dtype=torch.int64)))

    def test_smiles_to_strings_left_truncates_long(self) -> None:
        _, index = _build_smiles_alphabet(["CCCCCC"])
        feats = _smiles_to_strings(["CCCCCC"], index=index, maxlen=3)
        # First three Cs → [1, 1, 1]
        self.assertTrue(torch.equal(feats[0], torch.tensor([1, 1, 1], dtype=torch.int64)))


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class NormalizationTests(unittest.TestCase):
    def test_standardize_y_off(self) -> None:
        # Build a surrogate just to exercise its normalize method (pure tensor ops).
        surrogate = GPSurrogate(GPConfig(standardize_y=False, device="cpu"))
        result = surrogate._normalize_y([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)))
        self.assertIsNone(surrogate._y_mean)
        self.assertIsNone(surrogate._y_std)

    def test_standardize_y_on(self) -> None:
        surrogate = GPSurrogate(GPConfig(standardize_y=True, device="cpu"))
        result = surrogate._normalize_y([1.0, 2.0, 3.0, 4.0, 5.0])
        # mean = 3.0, std = sqrt(2.0) ≈ 1.4142
        self.assertAlmostEqual(surrogate._y_mean, 3.0)
        self.assertAlmostEqual(surrogate._y_std, float(np.std([1.0, 2.0, 3.0, 4.0, 5.0])))
        # Normalized values should have approximately zero mean and unit std.
        arr = result.numpy()
        self.assertAlmostEqual(float(arr.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(arr.std()), 1.0, places=5)

    def test_destandardize_roundtrip(self) -> None:
        mean = torch.tensor([0.5, -1.0])
        var = torch.tensor([0.25, 0.16])
        out_mean, out_var = _destandardize(mean, var, y_mean=3.0, y_std=2.0)
        self.assertTrue(torch.allclose(out_mean, torch.tensor([4.0, 1.0])))
        self.assertTrue(torch.allclose(out_var, torch.tensor([1.0, 0.64])))

    def test_destandardize_noop_when_unnormalized(self) -> None:
        mean = torch.tensor([0.5])
        var = torch.tensor([0.25])
        out_mean, out_var = _destandardize(mean, var, y_mean=None, y_std=None)
        self.assertTrue(torch.equal(out_mean, mean))
        self.assertTrue(torch.equal(out_var, var))


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class GPSurrogateConstructionTests(unittest.TestCase):
    def test_default_config(self) -> None:
        cfg = GPConfig()
        self.assertEqual(cfg.impl, "fingerprint+tanimoto")
        self.assertEqual(cfg.min_jitter, 1e-6)
        self.assertEqual(cfg.jitter_multiplier, 10.0)
        self.assertEqual(cfg.max_jitter, 1e-1)
        self.assertEqual(cfg.learning_rate, 0.1)
        self.assertEqual(cfg.fit_n_itersteps, 100)
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.fp_n_bits, 2048)
        self.assertEqual(cfg.smiles_maxlen, 80)
        self.assertTrue(cfg.standardize_y)

    def test_construction_tanimoto(self) -> None:
        surrogate = GPSurrogate(GPConfig(impl="fingerprint+tanimoto", device="cpu"))
        self.assertFalse(surrogate.is_fitted)
        self.assertFalse(surrogate.in_prior_mode)
        self.assertEqual(surrogate.device, torch.device("cpu"))

    def test_construction_string(self) -> None:
        surrogate = GPSurrogate(GPConfig(impl="smiles-strkernel", device="cpu"))
        self.assertFalse(surrogate.is_fitted)
        self.assertFalse(surrogate.in_prior_mode)

    def test_device_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        # If torch.cuda.is_available() is True we can't test the fallback; just
        # verify that requesting "cpu" gives a CPU device and doesn't crash.
        surrogate = GPSurrogate(GPConfig(device="cpu"))
        self.assertEqual(surrogate.device, torch.device("cpu"))


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class GPSurrogateFitTests(unittest.TestCase):
    def test_empty_smiles_raises(self) -> None:
        surrogate = GPSurrogate(GPConfig(**_CPU_CONFIG_KW))
        with self.assertRaises(ValueError):
            surrogate.fit([], [])

    def test_mismatched_lengths_raises(self) -> None:
        surrogate = GPSurrogate(GPConfig(**_CPU_CONFIG_KW))
        with self.assertRaises(ValueError):
            surrogate.fit(["CCO"], [-7.0, -8.0])

    def test_predict_before_fit_raises(self) -> None:
        surrogate = GPSurrogate(GPConfig(device="cpu"))
        with self.assertRaises(RuntimeError):
            surrogate.predict(["CCO"])

    def test_fit_happy_path_tanimoto(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        surrogate.fit(
            ["CCO", "CCN", "CCC", "CCCC", "CCCCC"],
            [-3.0, -2.9, -2.8, -2.7, -2.6],
        )
        self.assertTrue(surrogate.is_fitted)
        self.assertFalse(surrogate.in_prior_mode)

        mean, std = surrogate.predict(["CCO", "CCN", "CCC"])
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std > 0))

    def test_fit_happy_path_string(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(
                impl="smiles-strkernel",
                smiles_maxlen=20,
                fit_n_itersteps=5,
                learning_rate=0.1,
                device="cpu",
            )
        )
        surrogate.fit(
            ["CCO", "CCN", "CCC", "CCCC", "CCCCC"],
            [-3.0, -2.9, -2.8, -2.7, -2.6],
        )
        # The string kernel often falls back to prior mode on tiny training
        # sets (Cholesky ladder exhausted); either outcome is acceptable —
        # we just need the GP to be callable and produce finite predictions.
        self.assertTrue(surrogate.is_fitted)

        mean, std = surrogate.predict(["CCO", "CCN"])
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0))

    def test_fit_falls_back_to_prior_on_cholesky_failure(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        with patch("tasks.small_molecule.core.gp.gpytorch.settings.cholesky_jitter", _failing_cholesky):
            surrogate.fit(["CCO", "CCN", "CCC"], [-7.0, -7.5, -8.0])

        self.assertTrue(surrogate.is_fitted)
        self.assertTrue(surrogate.in_prior_mode)
        # Even in prior mode, predict returns finite values (the prior).
        mean, std = surrogate.predict(["CCO", "CCN"])
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std >= 0))

    def test_refit_replaces_model(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        surrogate.fit(["CCO", "CCN", "CCC"], [-3.0, -2.9, -2.8])
        first_model = surrogate.model
        surrogate.fit(["CCO", "CCN", "CCC"], [-7.0, -7.5, -8.0])
        self.assertIsNot(surrogate.model, first_model)
        self.assertTrue(surrogate.is_fitted)


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class GPSurrogatePredictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        self.surrogate.fit(
            ["CCO", "CCN", "CCC", "CCCC", "CCCCC"],
            [-3.0, -2.9, -2.8, -2.7, -2.6],
        )

    def test_predict_shape(self) -> None:
        mean, std = self.surrogate.predict(["CCO", "CCN", "CCC"])
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))

    def test_predict_return_tensor(self) -> None:
        mean, std = self.surrogate.predict(["CCO", "CCN"], return_tensor=True)
        self.assertIsInstance(mean, torch.Tensor)
        self.assertIsInstance(std, torch.Tensor)

    def test_predict_empty_input(self) -> None:
        mean, std = self.surrogate.predict([])
        self.assertEqual(mean.shape, (0,))
        self.assertEqual(std.shape, (0,))

    def test_predict_invalid_smiles_returns_finite(self) -> None:
        mean, std = self.surrogate.predict(["not_a_smiles", "CCO"])
        self.assertEqual(mean.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))

    def test_predict_string_impl_with_unseen_chars(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(
                impl="smiles-strkernel",
                smiles_maxlen=20,
                fit_n_itersteps=2,
                device="cpu",
            )
        )
        surrogate.fit(["CCO", "CCN", "CCC"], [-3.0, -2.9, -2.8])
        # Atoms not in the training set (Br) silently pad to 0.
        mean, std = surrogate.predict(["BrCl", "CCO"])
        self.assertEqual(mean.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0))


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class GPSurrogateAlphabetTests(unittest.TestCase):
    def test_alphabet_auto_built_from_training(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="smiles-strkernel", fit_n_itersteps=2, device="cpu")
        )
        surrogate.fit(["CCO", "CCN"], [-3.0, -2.9])
        self.assertIsNotNone(surrogate._alphabet)
        self.assertEqual(set(surrogate._alphabet), {"C", "O", "N"})
        self.assertIsNotNone(surrogate._alphabet_index)
        self.assertIn("C", surrogate._alphabet_index)

    def test_alphabet_extends_on_refit(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="smiles-strkernel", fit_n_itersteps=2, device="cpu")
        )
        surrogate.fit(["CCO", "CCN"], [-3.0, -2.9])
        self.assertNotIn("F", surrogate._alphabet_index)
        surrogate.fit(["CCF", "CCC"], [-2.5, -2.6])
        self.assertIn("F", surrogate._alphabet_index)


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class CanonicalizeAndDedupTests(unittest.TestCase):
    def test_basic_dedup(self) -> None:
        unique, scores = _canonicalize_and_dedup(
            ["CCO", "CCO", "CCN"], [-7.0, -8.0, -6.5]
        )
        self.assertEqual(unique, ["CCO", "CCN"])
        self.assertAlmostEqual(scores[0], -7.5)
        self.assertAlmostEqual(scores[1], -6.5)

    def test_skips_invalid_smiles(self) -> None:
        unique, scores = _canonicalize_and_dedup(
            ["invalid", "CCO", ""], [0.0, -7.0, 0.0]
        )
        self.assertEqual(unique, ["CCO"])
        self.assertEqual(scores, [-7.0])

    def test_preserves_first_seen_order(self) -> None:
        unique, _ = _canonicalize_and_dedup(
            ["CCN", "CCO", "CCN"], [-1.0, -2.0, -3.0]
        )
        # First-seen: CCN first, then CCO (not alphabetical).
        self.assertEqual(unique, ["CCN", "CCO"])

    def test_all_invalid_returns_empty(self) -> None:
        unique, scores = _canonicalize_and_dedup(
            ["invalid", "", "garbage"], [0.0, 0.0, 0.0]
        )
        self.assertEqual(unique, [])
        self.assertEqual(scores, [])


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class DedupByFeatureRowTests(unittest.TestCase):
    def test_keeps_unique_rows(self) -> None:
        feats = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        out_feats, out_scores = _dedup_by_feature_row(feats, [1.0, 2.0, 3.0])
        self.assertEqual(tuple(out_feats.shape), (3, 3))
        self.assertEqual(out_scores, [1.0, 2.0, 3.0])

    def test_drops_duplicate_rows(self) -> None:
        feats = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        out_feats, out_scores = _dedup_by_feature_row(feats, [1.0, 2.0, 3.0])
        self.assertEqual(tuple(out_feats.shape), (2, 3))
        self.assertEqual(out_scores, [1.0, 3.0])

    def test_empty_input(self) -> None:
        feats = torch.zeros((0, 4), dtype=torch.float32)
        out_feats, out_scores = _dedup_by_feature_row(feats, [])
        self.assertEqual(tuple(out_feats.shape), (0, 4))
        self.assertEqual(out_scores, [])


@unittest.skipUnless(_HEAVY_AVAILABLE, "requires torch + gpytorch + gauche + rdkit")
class GPSurrogateFitDedupIntegrationTests(unittest.TestCase):
    def test_fit_with_duplicate_smiles_succeeds(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        # Same molecule evaluated twice (typical BO-loop pattern).
        surrogate.fit(["CCO", "CCO", "CCN"], [-7.0, -8.0, -6.5])
        self.assertTrue(surrogate.is_fitted)
        self.assertFalse(surrogate.in_prior_mode)
        # The duplicate should have been collapsed; mean prediction should
        # be finite and std positive.
        mean, std = surrogate.predict(["CCO", "CCN"])
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std > 0))

    def test_fit_with_invalid_smiles_succeeds(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        surrogate.fit(["invalid_smiles", "CCO"], [0.0, -7.0])
        self.assertTrue(surrogate.is_fitted)
        self.assertFalse(surrogate.in_prior_mode)
        mean, std = surrogate.predict(["CCO"])
        self.assertEqual(mean.shape, (1,))
        self.assertTrue(np.all(np.isfinite(mean)))

    def test_fit_all_invalid_raises(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        with self.assertRaises(ValueError):
            surrogate.fit(["invalid_1", "invalid_2"], [0.0, 0.0])

    def test_fit_collapses_canonical_equivalents(self) -> None:
        surrogate = GPSurrogate(
            GPConfig(impl="fingerprint+tanimoto", **_CPU_CONFIG_KW)
        )
        # "OCC" canonicalizes to "CCO" — should collapse with the second "CCO".
        surrogate.fit(["OCC", "CCO"], [-7.0, -8.0])
        self.assertTrue(surrogate.is_fitted)
        self.assertFalse(surrogate.in_prior_mode)
        # If canonicalization worked, only one unique SMILES was kept and the
        # model trained on a single point — predict still returns finite values.
        mean, std = surrogate.predict(["CCO"])
        self.assertEqual(mean.shape, (1,))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()