"""Tests for ``strbo_v1.objective_nn`` (``NNScorer``).

Loads the committed G12D artifact for end-to-end coverage of the wrapper.
For the ``on_error`` policies we inject a broken ``predict`` via a
subclass override; no model corruption.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from strbo_v1 import NNScorer, NNScorerConfig  # noqa: E402
from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH  # noqa: E402
from strbo_v1.objective_nn import NNScorer as _NNScorer  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_MODEL = REPO_ROOT / DEFAULT_NN_MODEL_PATH
COMMITTED_METADATA = REPO_ROOT / "activity_modeling" / "best_g12d_model_metadata.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AlwaysFailingScorer(_NNScorer):
    """Subclass that swaps the loaded model for a dummy whose ``predict``
    raises. Used to exercise the ``on_error`` policies."""

    def __init__(self, config: NNScorerConfig) -> None:
        # Skip the base class's joblib.load; inject a failing model directly.
        self.config = config
        self.metadata: dict[str, Any] = {}
        self._model = _AlwaysFailingModel()


class _AlwaysFailingModel:
    def predict(self, X: Any) -> Any:  # pragma: no cover - exercised in tests
        raise RuntimeError("intentional test failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class LoadCommittedModelTests(unittest.TestCase):
    """The committed artifact must load and expose ``.predict``."""

    def test_loads_committed_model(self) -> None:
        scorer = NNScorer(NNScorerConfig(model_path=str(COMMITTED_MODEL)))
        self.assertTrue(hasattr(scorer, "_model"))
        self.assertTrue(callable(getattr(scorer._model, "predict", None)))

    def test_metadata_loaded_when_sidecar_present(self) -> None:
        scorer = NNScorer(NNScorerConfig(model_path=str(COMMITTED_MODEL)))
        self.assertIsInstance(scorer.metadata, dict)
        self.assertNotEqual(scorer.metadata, {})
        self.assertIn("G12D", scorer.metadata.get("task", ""))
        self.assertEqual(
            scorer.metadata.get("best_model"), "ensemble_nn_ridge_rf"
        )
        # best_metric_row is nested in the committed metadata.
        metric_row = scorer.metadata.get("best_metric_row") or {}
        self.assertIn("rmse", metric_row)

    def test_explicit_metadata_path_overrides_default(self) -> None:
        scorer = NNScorer(
            NNScorerConfig(
                model_path=str(COMMITTED_MODEL),
                metadata_path=str(COMMITTED_METADATA),
            )
        )
        self.assertIn("G12D", scorer.metadata.get("task", ""))

    def test_missing_model_raises_runtime_error(self) -> None:
        bogus = REPO_ROOT / "activity_modeling" / "does_not_exist.joblib"
        with self.assertRaises(RuntimeError) as cm:
            NNScorer(NNScorerConfig(model_path=str(bogus)))
        self.assertIn(str(bogus), str(cm.exception))

    def test_missing_metadata_path_does_not_fail(self) -> None:
        """Metadata is informational; a missing or unparseable sidecar
        must NOT fail construction."""
        with tempfile.TemporaryDirectory() as tmp:
            # Copy the committed joblib to a path with no sibling metadata.
            import joblib
            import shutil
            tmpdir = Path(tmp)
            staged = tmpdir / "staged_model.joblib"
            shutil.copyfile(COMMITTED_MODEL, staged)
            scorer = NNScorer(NNScorerConfig(model_path=str(staged)))
        self.assertEqual(scorer.metadata, {})


class CallContractTests(unittest.TestCase):
    """``scorer(smiles_list)`` returns a list[float] matching the contract
    in ``strbo_v1/scorer.py``: i-th output = i-th SMILES, ``nan`` on
    invalid input, finite on success."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = NNScorer(NNScorerConfig(model_path=str(COMMITTED_MODEL)))

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(self.scorer([]), [])

    def test_canonicalization_is_idempotent(self) -> None:
        # The model was trained on canonical isomeric SMILES; non-canonical
        # forms must be canonicalized to the same representation.
        s1 = self.scorer(["C(C)O"])[0]
        s2 = self.scorer(["CCO"])[0]
        self.assertAlmostEqual(s1, s2, places=12)

    def test_invalid_smiles_yields_nan(self) -> None:
        result = self.scorer(["@@@"])
        self.assertEqual(len(result), 1)
        self.assertTrue(math.isnan(result[0]))

    def test_empty_string_yields_nan(self) -> None:
        self.assertTrue(math.isnan(self.scorer([""])[0]))

    def test_mixed_valid_invalid_alignment(self) -> None:
        smiles = ["CCO", "@@@", "INVALID", "CCN"]
        out = self.scorer(smiles)
        self.assertEqual(len(out), len(smiles))
        # Invalid positions (1, 2) must be nan.
        self.assertTrue(math.isfinite(out[0]))
        self.assertTrue(math.isnan(out[1]))
        self.assertTrue(math.isnan(out[2]))
        self.assertTrue(math.isfinite(out[3]))

    def test_length_preserved(self) -> None:
        for n in (1, 5, 50):
            smis = ["CCO"] * n
            out = self.scorer(smis)
            self.assertEqual(len(out), n)

    def test_returns_native_python_floats(self) -> None:
        out = self.scorer(["CCO", "CCN", "c1ccccc1"])
        for v in out:
            # Must be a real ``float``, not ``numpy.float64`` (which the
            # BO loop handles, but the contract documents ``list[float]``).
            self.assertIs(type(v), float)

    def test_score_in_training_range(self) -> None:
        # The committed model is trained on public KRAS G12D IC50 with
        # p_activity in a pIC50-like range. A simple SMILES like ethanol
        # should produce a sane, in-range prediction.
        score = self.scorer(["CCO"])[0]
        self.assertTrue(math.isfinite(score))
        self.assertGreater(score, 0.0)
        self.assertLess(score, 12.0)


class OnErrorPolicyTests(unittest.TestCase):
    """``NNScorerConfig.on_error`` controls what happens when
    ``model.predict`` raises."""

    def test_default_is_all_nan(self) -> None:
        cfg = NNScorerConfig(model_path=str(COMMITTED_MODEL))
        self.assertEqual(cfg.on_error, "all_nan")

    def test_on_error_all_nan_returns_all_nan(self) -> None:
        cfg = NNScorerConfig(model_path=str(COMMITTED_MODEL), on_error="all_nan")
        scorer = _AlwaysFailingScorer(cfg)
        out = scorer(["CCO", "CCN", "c1ccccc1"])
        self.assertEqual(len(out), 3)
        for v in out:
            self.assertTrue(math.isnan(v))

    def test_on_error_raise_propagates(self) -> None:
        cfg = NNScorerConfig(model_path=str(COMMITTED_MODEL), on_error="raise")
        scorer = _AlwaysFailingScorer(cfg)
        with self.assertRaises(RuntimeError) as cm:
            scorer(["CCO"])
        self.assertIn("intentional test failure", str(cm.exception))


class PackageExportsTests(unittest.TestCase):
    """The public surface re-exported from ``strbo_v1.__init__`` is
    consistent."""

    def test_nnscorer_exported(self) -> None:
        from strbo_v1 import NNScorer as Imported  # noqa: F401
        self.assertIs(Imported, NNScorer)

    def test_nnscorer_config_exported(self) -> None:
        from strbo_v1 import NNScorerConfig as Imported  # noqa: F401
        self.assertIs(Imported, NNScorerConfig)

    def test_scorer_typealias_exported(self) -> None:
        from strbo_v1 import Scorer  # noqa: F401
        # Scorer is a TypeAlias; the binding is the alias object itself.
        self.assertTrue(callable(Scorer) or hasattr(Scorer, "__call__"))


if __name__ == "__main__":
    unittest.main()
