"""NN-backed single-point Scorer for the LDM-TTS small-molecule loop.

This module wraps a caller-supplied trained regression model. The documented
reference artifact is ``best_g12d_model.joblib`` for public KRAS G12D
direct-assay IC50 records. It is not distributed in Git. The scorer is callable and
mirrors :class:`VinaScorer`'s contract so the same LDM-TTS loop can swap
between mock, NN, and Vina objectives::

    scorer = NNScorer(NNScorerConfig(model_path="resources/models/best_g12d_model.joblib"))
    scores = scorer(smiles_list)    # list[float], i-th output is i-th SMILES

Output is predicted pIC50 (continuous, training range 3-11.5; higher =
more potent). Invalid SMILES yield ``float("nan")``; per-batch inference
failures fall back to all-``nan`` (configurable). The BO loop's
``_safe_score`` helper converts non-finite floats to ``None`` and excludes
them from the GP fit.

Implementation notes
--------------------

* The artifact is a joblib pickle whose custom transformers
  (``MorganFingerprintTransformer``, ``TanimotoKNNRegressor``,
  ``AverageRegressor``, ``RDKitDescriptorTransformer``) are tagged
  ``__module__ = "train_g12c_qsar"`` for pickle compatibility. We import the
  legacy training module (no side effects at
  import time) and register ``sys.modules["train_g12c_qsar"]`` to
  resolve both the canonical (``tasks.small_molecule.core.activity_modeling.train_g12c_qsar``) and
  aliased (``train_g12c_qsar``) module paths during unpickling.
* The trained pipelines expect canonical SMILES with
  ``isomericSmiles=True`` (matching ``predict_g12c_activity.py``); we
  mirror that preprocessing here.

Public surface (re-exported from :mod:`tasks.small_molecule.core.__init__`):

* :class:`NNScorerConfig`
* :class:`NNScorer`
* :data:`Scorer` (re-exported from :mod:`tasks.small_molecule.core.scorer`)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import joblib
import numpy as np
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles

from tasks.small_molecule.core.model_artifact import (
    ArtifactIntegrityError,
    find_metadata_path,
    load_metadata_file,
    verify_declared_sha256,
)
from tasks.small_molecule.core.scorer import Scorer

__all__ = [
    "NNScorerConfig",
    "NNScorer",
    "Scorer",
]


# ---------------------------------------------------------------------------
# Pickle-module shim
# ---------------------------------------------------------------------------
#
# The reference G12D model is a sklearn / joblib pickle whose
# custom classes (``MorganFingerprintTransformer``, ``TanimotoKNNRegressor``,
# ``AverageRegressor``, ``RDKitDescriptorTransformer``, ``smiles_identity``,
# ``to_dense_matrix``) are tagged ``__module__ = "train_g12c_qsar"`` by
# ``register_pickle_module_alias()`` in ``core/activity_modeling/train_g12c_qsar.py``.
# When the same script is run as ``__main__``, ``sys.modules["train_g12c_qsar"]``
# is set to ``__main__``; when imported normally, Python sets
# ``sys.modules["tasks.small_molecule.core.activity_modeling.train_g12c_qsar"]``. Registering the imported
# module under the alias key covers both cases.

try:
    import tasks.small_molecule.core.activity_modeling.train_g12c_qsar as _train_module  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "NNScorer requires the task's core/activity_modeling package. "
        f"Could not import tasks.small_molecule.core.activity_modeling.train_g12c_qsar: {exc}"
    ) from exc

sys.modules.setdefault("train_g12c_qsar", _train_module)


# ---------------------------------------------------------------------------
# SMILES canonicalization used by the activity-model training pipeline.
# ---------------------------------------------------------------------------

_RDKIT_LOGGER = RDLogger.logger()
# Suppress the per-SMILES parse-error chatter. The model handles invalid
# SMILES via canonicalization returning ""; we do not want a noisy stderr.
RDLogger.DisableLog("rdApp.*")


def _canonicalize_smiles(smiles: str) -> str:
    """Return the canonical isomeric SMILES, or ``""`` on failure.

    Matches the training pipeline so inference uses the same representation.
    """
    text = str(smiles or "").strip()
    if not text:
        return ""
    try:
        mol = MolFromSmiles(text)
    except Exception:
        return ""
    if mol is None:
        return ""
    return MolToSmiles(mol, isomericSmiles=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NNScorerConfig:
    """Configuration for :class:`NNScorer`.

    Attributes
    ----------
    model_path
        Filesystem path to a joblib artifact with a ``.predict`` method.
    metadata_path
        Optional sidecar JSON metadata (e.g. ``best_g12d_model_metadata.json``).
        Defaults to ``<model_path>.metadata.json`` when omitted.
    on_error
        Inference-failure policy: ``"all_nan"`` returns a list of
        ``float("nan")`` of the same length as the input (the BO loop
        converts non-finite floats to ``None`` internally, so a broken
        batch degrades gracefully); ``"raise"`` re-raises the underlying
        exception for loud failure.
    """

    model_path: str
    metadata_path: str = ""
    on_error: Literal["all_nan", "raise"] = "all_nan"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class NNScorer:
    """Callable NN scoring interface backed by a joblib regression model.

    Mirrors :class:`tasks.small_molecule.core.objective_vina.VinaScorer`:

    * Constructed from a :class:`NNScorerConfig` (the model path /
      metadata path / error policy are fixed at ``__init__`` and cannot
      be overridden per call).
    * ``scorer(smiles_list)`` returns a ``list[float]`` of the same
      length as the input. The i-th element is the model's score for
      the i-th SMILES, or ``float("nan")`` on any per-row failure
      (invalid SMILES, empty input, ...). Per-batch inference failures
      follow ``config.on_error``.

    Preprocessing matches the task's activity-model training pipeline:
    SMILES are canonicalized via RDKit (``isomericSmiles=True``) before
    being passed to ``model.predict``.
    """

    def __init__(self, config: NNScorerConfig) -> None:
        self.config = config
        self.metadata: dict[str, Any] = self._load_metadata(config)
        self._model = self._load_model(config.model_path, self.metadata)
        self.last_results: list[dict[str, Any]] = []

    # -- public API ----------------------------------------------------------

    def __call__(self, smiles_list: Sequence[str]) -> list[float]:
        smiles_list = list(smiles_list)
        n = len(smiles_list)
        if n == 0:
            self.last_results = []
            return []

        out: list[float] = [float("nan")] * n
        self.last_results = [
            {
                "input_smiles": smi,
                "canonical_smiles": "",
                "status": "invalid_smiles",
                "score": None,
            }
            for smi in smiles_list
        ]
        canonical_smis: list[str] = []
        valid_indices: list[int] = []

        for i, smi in enumerate(smiles_list):
            canon = _canonicalize_smiles(smi)
            if canon:
                canonical_smis.append(canon)
                valid_indices.append(i)
                self.last_results[i] = {
                    "input_smiles": smi,
                    "canonical_smiles": canon,
                    "status": "pending",
                    "score": None,
                }

        if not canonical_smis:
            return out

        try:
            preds = self._predict(canonical_smis)
        except Exception as exc:
            for idx in valid_indices:
                self.last_results[idx]["status"] = "inference_failed"
                self.last_results[idx]["message"] = str(exc)
            if self.config.on_error == "all_nan":
                return [float("nan")] * n
            raise RuntimeError(
                f"NNScorer inference failed for {n} SMILES "
                f"(model_path={self.config.model_path!r}): {exc}"
            ) from exc

        for j, idx in enumerate(valid_indices):
            out[idx] = self._to_native_float(preds[j])
            self.last_results[idx]["score"] = out[idx] if math.isfinite(out[idx]) else None
            self.last_results[idx]["status"] = "ok" if math.isfinite(out[idx]) else "nonfinite_prediction"
        return out

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _load_model(model_path: str, metadata: dict[str, Any]) -> Any:
        path = Path(str(model_path or "")).expanduser()
        if not str(path):
            raise RuntimeError("NNScorerConfig.model_path is empty.")
        if not path.is_file():
            raise RuntimeError(
                f"NNScorer model not found: {path} "
                f"(set NNScorerConfig.model_path to an existing joblib file)."
            )
        if metadata.get("_load_error"):
            raise RuntimeError(
                "NNScorer could not read model integrity metadata before "
                f"deserialization: {metadata['_load_error']}"
            )
        try:
            verify_declared_sha256(path, metadata)
        except (ArtifactIntegrityError, OSError) as exc:
            raise RuntimeError(str(exc)) from exc
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"NNScorer failed to load model {path}: {exc}"
            ) from exc
        if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
            raise TypeError(
                f"NNScorer loaded object ({type(model).__name__}) from {path} "
                "has no callable .predict method; expected a fitted sklearn "
                "Pipeline / RegressorMixin."
            )
        return model

    @staticmethod
    def _load_metadata(config: NNScorerConfig) -> dict[str, Any]:
        meta_path_str = config.metadata_path
        if not meta_path_str:
            # Try the metadata sidecar conventions used by activity models:
            #   resources/models/best_g12d_model.joblib
            #   resources/models/best_g12d_model_metadata.json
            # The training script appends ``_metadata.json`` to the model
            # stem, so we look for ``<stem>_metadata.json`` first, then
            # ``<stem>.metadata.json`` as a fallback.
            candidate = find_metadata_path(Path(config.model_path))
            if candidate is not None:
                meta_path_str = str(candidate)
        if not meta_path_str:
            return {}
        try:
            data = load_metadata_file(Path(meta_path_str))
        except (OSError, ValueError) as exc:
            # Metadata without a usable integrity declaration remains
            # informational for custom caller-trusted artifacts.
            return {"_load_error": str(exc), "_path": meta_path_str}
        return data

    def _predict(self, canonical_smis: list[str]) -> Any:
        # The reference ensemble emits a sklearn "feature names" warning
        # when called with a plain list (the LightGBM sub-pipeline was
        # fit on a pandas Series). We pass through numpy and silence the
        # warning at the source to keep the BO loop's stderr clean.
        with np.errstate(all="ignore"), _suppress_feature_name_warning():
            return self._model.predict(np.asarray(canonical_smis, dtype=object))

    @staticmethod
    def _to_native_float(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if not math.isfinite(result):
            return float("nan")
        return result


# ---------------------------------------------------------------------------
# Context manager: silence the LGBMRegressor "X does not have valid feature
# names" UserWarning emitted when the trained pipeline is called with a
# numpy array (the BO loop sees plain Python lists / numpy 1-D arrays).
# ---------------------------------------------------------------------------


class _suppress_feature_name_warning:  # noqa: N801 (intentional: matches sklearn idiom)
    def __enter__(self) -> "_suppress_feature_name_warning":
        import warnings

        self._filters = list(warnings.filters)
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        import warnings

        warnings.filters = self._filters
