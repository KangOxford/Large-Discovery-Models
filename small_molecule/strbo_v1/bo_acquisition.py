"""Reusable BO acquisition evaluator over a fitted GP surrogate."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

from strbo_v1.acquisition import (
    confidence_bound,
    expected_improvement,
    probability_of_improvement,
)
from strbo_v1.gp import GPConfig, GPSurrogate


def normalize_acquisition_names(
    acquisition: Union[str, Sequence[str]],
) -> Tuple[str, ...]:
    """Return normalized acquisition names, preserving first-seen order."""
    if isinstance(acquisition, str):
        raw_names = (acquisition,)
    else:
        raw_names = tuple(acquisition)
    names: list[str] = []
    for raw in raw_names:
        name = str(raw).lower().strip()
        _resolve_acquisition(name)
        if name == "lcb":
            name = "ucb"
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("at least one acquisition name is required")
    return tuple(names)


def single_acquisition_name(acquisition: Union[str, Sequence[str]]) -> str:
    names = normalize_acquisition_names(acquisition)
    if len(names) != 1:
        raise ValueError(
            "bayesian_analog_search/select_candidates require one acquisition "
            f"name; got {names!r}. Use AcquisitionEvaluator for multi-acquisition queries."
        )
    return names[0]


def _resolve_acquisition(name: str) -> Callable[..., np.ndarray]:
    name = name.lower().strip()
    if name == "ei":
        return expected_improvement
    if name == "pi":
        return probability_of_improvement
    if name in ("ucb", "lcb"):
        return confidence_bound
    raise ValueError(
        f"Unknown acquisition {name!r}; expected one of 'ei', 'pi', 'ucb'."
    )


def _infer_n_obj_from_history(
    history: Sequence[Tuple[str, Union[float, Tuple[Optional[float], ...]]]],
) -> int:
    for _, score in history:
        if score is None:
            continue
        if isinstance(score, (int, float)):
            return 1
        seq = tuple(score)
        if not seq:
            raise ValueError("history entry has empty score tuple; cannot infer n_obj.")
        return len(seq)
    return 1


def _as_minimize_tuple(minimize: Union[bool, Sequence[bool]], n_obj: int) -> Tuple[bool, ...]:
    if isinstance(minimize, bool):
        return (minimize,) * n_obj
    seq = tuple(minimize)
    if len(seq) != n_obj:
        raise ValueError(
            f"minimize length ({len(seq)}) does not match n_obj ({n_obj})"
        )
    return seq


def _score_at_objective(
    raw_score: Union[float, Tuple[Optional[float], ...], None],
    n_obj: int,
    objective_index: int,
) -> Optional[float]:
    if raw_score is None:
        return None
    if isinstance(raw_score, (int, float)):
        if n_obj != 1:
            raise ValueError("bare float score is only valid for n_obj == 1")
        score = float(raw_score)
    else:
        seq = tuple(raw_score)
        if len(seq) != n_obj:
            raise ValueError(
                f"history score tuple length {len(seq)} does not match n_obj={n_obj}"
            )
        value = seq[objective_index]
        if value is None:
            return None
        score = float(value)
    return score if np.isfinite(score) else None


def _finite_history_for_objective(
    history: Sequence[Tuple[str, Union[float, Tuple[Optional[float], ...]]]],
    objective_index: int,
) -> Tuple[list[str], np.ndarray, int]:
    n_obj = _infer_n_obj_from_history(history)
    if objective_index < 0 or objective_index >= n_obj:
        raise ValueError(
            f"objective_index {objective_index} is out of range for n_obj={n_obj}"
        )
    smiles: list[str] = []
    scores: list[float] = []
    for smi, raw_score in history:
        score = _score_at_objective(raw_score, n_obj, objective_index)
        if score is None:
            continue
        smiles.append(str(smi))
        scores.append(score)
    return smiles, np.asarray(scores, dtype=float), n_obj


class AcquisitionEvaluator:
    """Fit a GP once, then query acquisition values many times.

    The evaluator is single-objective. For multi-objective histories,
    pass ``objective_index`` to choose which score column to model.
    """

    def __init__(
        self,
        history: Sequence[Tuple[str, Union[float, Tuple[Optional[float], ...]]]],
        config: Any,
        *,
        acquisitions: Optional[Union[str, Sequence[str]]] = None,
        objective_index: int = 0,
        surrogate_factory: Callable[[GPConfig], Any] = GPSurrogate,
    ) -> None:
        if config is None:
            raise ValueError("AcquisitionEvaluator requires a config")
        self.config = config
        self.acquisition_names = normalize_acquisition_names(
            acquisitions if acquisitions is not None else config.acquisition
        )
        smiles, scores, n_obj = _finite_history_for_objective(history, objective_index)
        if len(smiles) < 2:
            raise ValueError(
                "AcquisitionEvaluator requires at least 2 finite history scores"
            )
        minimize_t = _as_minimize_tuple(config.minimize, n_obj)
        self.minimize = minimize_t[objective_index]
        self.best = float(np.min(scores)) if self.minimize else float(np.max(scores))
        self.surrogate = surrogate_factory(config.gp_config)
        self.surrogate.fit(smiles, scores.tolist())

    def __call__(self, query_smiles: Sequence[str]) -> dict[str, dict[str, float]]:
        points = [str(s) for s in query_smiles if s is not None and str(s).strip()]
        if not points:
            return {}
        mu, sigma = self.surrogate.predict(points, return_tensor=False)
        mu_arr = np.asarray(mu, dtype=float).ravel()
        sigma_arr = np.asarray(sigma, dtype=float).ravel()
        if mu_arr.shape != sigma_arr.shape or mu_arr.shape != (len(points),):
            raise ValueError(
                f"surrogate prediction shape mismatch for {len(points)} points: "
                f"mu={mu_arr.shape}, sigma={sigma_arr.shape}"
            )
        acq_by_name = self._evaluate_acquisitions(mu_arr, sigma_arr)
        return self._format_results(points, mu_arr, sigma_arr, acq_by_name)

    def _evaluate_acquisitions(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        for name in self.acquisition_names:
            fn = _resolve_acquisition(name)
            if name in ("ei", "pi"):
                raw = fn(mu, sigma, self.best, xi=self.config.xi, minimize=self.minimize)
            else:
                raw = fn(mu, sigma, kappa=self.config.kappa, minimize=self.minimize)
            values[name] = np.asarray(raw, dtype=float).ravel()
        return values

    def _format_results(
        self,
        points: list[str],
        mu: np.ndarray,
        sigma: np.ndarray,
        acq_by_name: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        single = len(self.acquisition_names) == 1
        for i, smi in enumerate(points):
            row = {
                "mean": float(mu[i]),
                "std": float(sigma[i]),
                "variance": float(sigma[i] ** 2),
            }
            for name, values in acq_by_name.items():
                key = "acquisition" if single else f"acquisition_{name}"
                row[key] = float(values[i])
            out[smi] = row
        return out


__all__ = [
    "AcquisitionEvaluator",
    "normalize_acquisition_names",
    "single_acquisition_name",
]
