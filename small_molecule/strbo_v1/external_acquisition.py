"""Structured external acquisition interface."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

from strbo_v1.acquisition import (
    chebyshev_scalarize,
    expected_hypervolume_improvement,
    sample_simplex_weights,
)
from strbo_v1.bo_acquisition import AcquisitionEvaluator
from strbo_v1.external_common import item, normalized_request, smiles_from
from strbo_v1.gp import GPConfig, GPSurrogate
from strbo_v1.rng import RNG, as_rng


HistoryValue = Union[float, Tuple[Optional[float], ...], None]


def evaluate_acquisition(
    *,
    history: Optional[Sequence[dict[str, Any]]] = None,
    query_smiles: Optional[Sequence[str]] = None,
    request: Optional[dict[str, Any]] = None,
    config: Optional[Any] = None,
    gp_device: Optional[str] = None,
    surrogate_factory: Callable[[GPConfig], Any] = GPSurrogate,
    rng: Optional[Any] = None,
) -> dict[str, Any]:
    """Evaluate GP posterior details and acquisition values for query SMILES."""
    req = normalized_request(request)
    history_raw = list(history if history is not None else req.get("history", []))
    query = smiles_from(query_smiles, req, key="query_smiles")
    n_obj = _infer_n_obj(history_raw)
    cfg = config or build_acquisition_config(req, n_obj=n_obj, gp_device=gp_device)
    objective_index = req.get("objective_index")
    if objective_index is not None or n_obj == 1:
        items = _single_objective_items(
            history_raw, query, cfg, int(objective_index or 0), surrogate_factory
        )
    else:
        items = _multi_objective_items(
            history_raw, query, cfg, n_obj, surrogate_factory, rng
        )
    return {"ok": True, "items": items, "errors": []}


def build_acquisition_config(
    request: Optional[dict[str, Any]] = None,
    *,
    n_obj: int,
    gp_device: Optional[str] = None,
) -> Any:
    from strbo_v1.bayesian_analog_search import BayesianAnalogSearchConfig

    req = normalized_request(request)
    gp_cfg = _build_gp_config(req, gp_device=gp_device)
    return BayesianAnalogSearchConfig(
        acquisition=req.get("acquisition", "ei"),
        minimize=_minimize_tuple(req.get("minimize", True), n_obj),
        ref_point=_optional_float_tuple(req.get("ref_point")),
        ehvi_n_samples=int(req.get("ehvi_n_samples", 128)),
        che_alpha=float(req.get("che_alpha", 1.0)),
        xi=float(req.get("xi", 0.01)),
        kappa=float(req.get("kappa", 2.0)),
        gp_config=gp_cfg,
    )


def _build_gp_config(req: dict[str, Any], *, gp_device: Optional[str]) -> GPConfig:
    method = str(req.get("method", "bo-tanimoto"))
    impl = "smiles-strkernel" if "strkernel" in method else "fingerprint+tanimoto"
    return GPConfig(
        impl=impl,
        device=gp_device or "cuda",
        fit_n_itersteps=int(req.get("gp_fit_itersteps", 100)),
        learning_rate=float(req.get("gp_learning_rate", 0.05)),
        fp_radius=int(req.get("gp_fp_radius", 2)),
        fp_n_bits=int(req.get("gp_fp_n_bits", 2048)),
    )


def _single_objective_items(
    history_raw: Sequence[dict[str, Any]],
    query: list[str],
    config: Any,
    objective_index: int,
    surrogate_factory: Callable[[GPConfig], Any],
) -> list[dict[str, Any]]:
    evaluator = AcquisitionEvaluator(
        _history_pairs(history_raw, _infer_n_obj(history_raw)),
        config,
        objective_index=objective_index,
        surrogate_factory=surrogate_factory,
    )
    details = evaluator(query)
    return [_acq_item(smi, details[smi]) for smi in query]


def _multi_objective_items(
    history_raw: Sequence[dict[str, Any]],
    query: list[str],
    config: Any,
    n_obj: int,
    surrogate_factory: Callable[[GPConfig], Any],
    rng: Optional[Any],
) -> list[dict[str, Any]]:
    pairs = _history_pairs(history_raw, n_obj)
    smiles_train, score_cols = _finite_score_columns(pairs, n_obj)
    gps = _fit_surrogates(smiles_train, score_cols, config.gp_config, surrogate_factory)
    mu_cols, sigma_cols = _predict_surrogates(gps, query)
    key, acq = _multi_acquisition_values(mu_cols, sigma_cols, score_cols, pairs, config, rng)
    return [
        _multi_acq_item(query[i], mu_cols, sigma_cols, acq, key, i)
        for i in range(len(query))
    ]


def _multi_acquisition_values(
    mu_cols: list[np.ndarray],
    sigma_cols: list[np.ndarray],
    score_cols: list[np.ndarray],
    pairs: Sequence[Tuple[str, HistoryValue]],
    config: Any,
    rng: Optional[Any],
) -> tuple[str, np.ndarray]:
    if len(mu_cols) == 2:
        return "acquisition_ehvi", _ehvi_values(mu_cols, sigma_cols, pairs, config, rng)
    return "acquisition_chebyshev", _chebyshev_values(mu_cols, score_cols, config, rng)


def _ehvi_values(
    mu_cols: list[np.ndarray],
    sigma_cols: list[np.ndarray],
    pairs: Sequence[Tuple[str, HistoryValue]],
    config: Any,
    rng: Optional[Any],
) -> np.ndarray:
    ref = config.ref_point if config.ref_point is not None else (0.0, 0.0)
    pareto = [tuple(score) for _, score in pairs if isinstance(score, tuple)]
    return expected_hypervolume_improvement(
        mu_per_obj=mu_cols,
        sigma_per_obj=sigma_cols,
        pareto_points=pareto,
        ref=ref,
        minimize=tuple(config.minimize),
        n_samples=config.ehvi_n_samples,
        rng=as_rng(rng) if rng is not None else RNG(seed=0),
    )


def _chebyshev_values(
    mu_cols: list[np.ndarray],
    score_cols: list[np.ndarray],
    config: Any,
    rng: Optional[Any],
) -> np.ndarray:
    n_obj = len(mu_cols)
    rng_obj = as_rng(rng) if rng is not None else RNG(seed=0)
    weights = sample_simplex_weights(rng_obj, n_obj, config.che_alpha)
    ideal = _ideal_point(score_cols, tuple(config.minimize))
    return np.asarray([
        chebyshev_scalarize(
            [mu_cols[j][i] for j in range(n_obj)], weights, ideal, config.minimize
        )
        for i in range(len(mu_cols[0]))
    ], dtype=float)


def _fit_surrogates(
    smiles_train: list[str],
    score_cols: list[np.ndarray],
    gp_config: GPConfig,
    surrogate_factory: Callable[[GPConfig], Any],
) -> list[Any]:
    gps = []
    for scores in score_cols:
        gp = surrogate_factory(gp_config)
        gp.fit(smiles_train, scores.tolist())
        gps.append(gp)
    return gps


def _predict_surrogates(gps: Sequence[Any], query: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mu_cols, sigma_cols = [], []
    for gp in gps:
        mu, sigma = gp.predict(query, return_tensor=False)
        mu_cols.append(np.asarray(mu, dtype=float).ravel())
        sigma_cols.append(np.asarray(sigma, dtype=float).ravel())
    return mu_cols, sigma_cols


def _multi_acq_item(
    smi: str,
    mu_cols: list[np.ndarray],
    sigma_cols: list[np.ndarray],
    acq: np.ndarray,
    key: str,
    index: int,
) -> dict[str, Any]:
    objectives = [
        {
            "index": j,
            "mean": float(mu_cols[j][index]),
            "std": float(sigma_cols[j][index]),
            "variance": float(sigma_cols[j][index] ** 2),
        }
        for j in range(len(mu_cols))
    ]
    details = {"objectives": objectives, key: float(acq[index])}
    return item(smi, ok=True, value=float(acq[index]), error=None, details=details)


def _acq_item(smi: str, details: dict[str, float]) -> dict[str, Any]:
    acq_keys = [key for key in details if key.startswith("acquisition")]
    value = details[acq_keys[0]] if acq_keys else None
    return item(smi, ok=True, value=value, error=None, details=details)


def _finite_score_columns(
    pairs: Sequence[Tuple[str, HistoryValue]], n_obj: int
) -> tuple[list[str], list[np.ndarray]]:
    smiles = []
    columns: list[list[float]] = [[] for _ in range(n_obj)]
    for smi, score in pairs:
        if not isinstance(score, tuple) or any(v is None for v in score):
            continue
        smiles.append(smi)
        for i, value in enumerate(score):
            columns[i].append(float(value))
    if len(smiles) < 2:
        raise ValueError("evaluate_acquisition requires at least 2 finite history scores")
    return smiles, [np.asarray(col, dtype=float) for col in columns]


def _history_pairs(history_raw: Sequence[dict[str, Any]], n_obj: int) -> list[Tuple[str, HistoryValue]]:
    pairs = []
    for i, entry in enumerate(history_raw):
        if not isinstance(entry, dict) or "smiles" not in entry:
            raise ValueError(f"history[{i}] must be an object with 'smiles'")
        if "scores" in entry:
            pairs.append((str(entry["smiles"]), _scores_tuple(entry, i, n_obj)))
        else:
            score = entry.get("score")
            pairs.append((str(entry["smiles"]), None if score is None else float(score)))
    return pairs


def _scores_tuple(entry: dict[str, Any], index: int, n_obj: int) -> tuple[Optional[float], ...]:
    values = tuple(None if v is None else float(v) for v in entry["scores"])
    if len(values) != n_obj:
        raise ValueError(f"history[{index}] scores length does not match n_obj={n_obj}")
    return values


def _infer_n_obj(history_raw: Sequence[dict[str, Any]]) -> int:
    for entry in history_raw:
        if isinstance(entry, dict) and "scores" in entry:
            scores = entry["scores"]
            if not isinstance(scores, list) or not scores:
                raise ValueError("history scores must be a non-empty list")
            return len(scores)
        if isinstance(entry, dict) and "score" in entry:
            return 1
    return 1


def _ideal_point(score_cols: list[np.ndarray], minimize: tuple[bool, ...]) -> list[float]:
    return [
        float(np.min(col)) if minimize[i] else float(np.max(col))
        for i, col in enumerate(score_cols)
    ]


def _minimize_tuple(raw: Any, n_obj: int) -> Union[bool, tuple[bool, ...]]:
    if isinstance(raw, bool):
        return raw if n_obj == 1 else (raw,) * n_obj
    values = tuple(bool(v) for v in raw)
    if len(values) != n_obj:
        raise ValueError(f"minimize length ({len(values)}) != n_obj ({n_obj})")
    return values


def _optional_float_tuple(raw: Any) -> Optional[tuple[float, ...]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    return tuple(float(x) for x in raw)
__all__ = ["build_acquisition_config", "evaluate_acquisition"]
