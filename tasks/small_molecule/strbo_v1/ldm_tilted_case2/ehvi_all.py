"""Finite-candidate posterior acquisition adapter for case2 tilted selection.

The filename and ``EhviResult.ehvi`` field are retained for trajectory and
import compatibility. Scoring itself is dispatched through the shared
``ldm_tts.acquisition`` module and can be EHVI or posterior mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from ldm_tts.acquisition import make_acquisition, pareto_front
from strbo_v1.gp import GPConfig, GPSurrogate
from strbo_v1.ldm_tilted_case2.candidate_record import CandidateRecord
from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config
from strbo_v1.rng import RNG, as_rng


@dataclass
class EhviResult:
    ehvi: np.ndarray
    mu_per_obj: list[np.ndarray]
    sigma_per_obj: list[np.ndarray]
    fallback_reason: str | None = None


def fit_two_objective_gps(
    history: Sequence[tuple[str, Sequence[float | None]]],
    cfg: TiltedLDMCase2Config,
    rng: RNG | None,
) -> list[GPSurrogate]:
    """Fit one GP per objective using finite two-objective history rows."""
    finite_rows = _finite_history(history)
    if len(finite_rows) < 2:
        raise ValueError("insufficient_history")
    smiles = [row[0] for row in finite_rows]
    gps: list[GPSurrogate] = []
    rng_obj = as_rng(rng)
    for obj_idx in range(2):
        rng_obj.torch()
        gp = GPSurrogate(_copy_gp_config(cfg.gp_config, rng_obj.seed + obj_idx))
        scores = [row[1][obj_idx] for row in finite_rows]
        gp.fit(smiles, scores, verbose=cfg.verbose)
        gps.append(gp)
    return gps


def predict_candidates(
    gps: Sequence[GPSurrogate],
    candidates: Sequence[CandidateRecord],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    smiles = [candidate.canonical_smiles or candidate.raw_smiles for candidate in candidates]
    mu_per_obj: list[np.ndarray] = []
    sigma_per_obj: list[np.ndarray] = []
    for gp in gps:
        mu, sigma = gp.predict(smiles)
        mu_per_obj.append(np.asarray(mu, dtype=float).ravel())
        sigma_per_obj.append(np.asarray(sigma, dtype=float).ravel())
    _write_predictions(candidates, mu_per_obj, sigma_per_obj)
    return mu_per_obj, sigma_per_obj


def compute_ehvi_for_candidates(
    history: Sequence[tuple[str, Sequence[float | None]]],
    candidates: Sequence[CandidateRecord],
    cfg: TiltedLDMCase2Config,
    rng: RNG | None = None,
) -> EhviResult:
    if not candidates:
        return EhviResult(np.zeros((0,), dtype=float), [], [])
    finite_rows = _finite_history(history)
    if len(finite_rows) < 2:
        return _fallback(candidates, "insufficient_history")
    try:
        gps = fit_two_objective_gps(finite_rows, cfg, rng)
        mu_per_obj, sigma_per_obj = predict_candidates(gps, candidates)
        ehvi = _compute_ehvi(finite_rows, mu_per_obj, sigma_per_obj, cfg, rng)
    except Exception:
        return _fallback(candidates, "gp_failed")
    _write_ehvi(candidates, ehvi)
    return EhviResult(ehvi, mu_per_obj, sigma_per_obj)


def _finite_history(
    history: Sequence[tuple[str, Sequence[float | None]]],
) -> list[tuple[str, tuple[float, float]]]:
    rows: list[tuple[str, tuple[float, float]]] = []
    for smiles, scores in history:
        if len(scores) != 2:
            continue
        first, second = scores[0], scores[1]
        if first is None or second is None:
            continue
        if isfinite(float(first)) and isfinite(float(second)):
            rows.append((smiles, (float(first), float(second))))
    return rows


def _copy_gp_config(config: GPConfig, seed: int) -> GPConfig:
    data = dict(config.__dict__)
    data["seed"] = int(seed)
    return GPConfig(**data)


def _compute_ehvi(
    finite_rows: Sequence[tuple[str, tuple[float, float]]],
    mu_per_obj: Sequence[np.ndarray],
    sigma_per_obj: Sequence[np.ndarray],
    cfg: TiltedLDMCase2Config,
    rng: RNG | None,
) -> np.ndarray:
    points = [scores for _smiles, scores in finite_rows]
    pareto = pareto_front(points, cfg.minimize)
    acquisition = make_acquisition(
        cfg.acquisition,
        minimize=cfg.minimize,
        weights=cfg.acquisition_weights,
        n_samples=cfg.ehvi_n_samples,
    )
    scores = acquisition.score(
        mu_per_obj,
        sigma_per_obj,
        pareto_points=pareto,
        ref_point=cfg.ref_point,
        rng=rng,
    )
    return np.asarray(scores, dtype=float).ravel()


def _fallback(candidates: Sequence[CandidateRecord], reason: str) -> EhviResult:
    zeros = np.zeros((len(candidates),), dtype=float)
    _write_ehvi(candidates, zeros)
    return EhviResult(zeros, [], [], fallback_reason=reason)


def _write_predictions(
    candidates: Sequence[CandidateRecord],
    mu_per_obj: Sequence[np.ndarray],
    sigma_per_obj: Sequence[np.ndarray],
) -> None:
    for idx, candidate in enumerate(candidates):
        candidate.mu = [float(mu[idx]) for mu in mu_per_obj]
        candidate.sigma = [float(sigma[idx]) for sigma in sigma_per_obj]


def _write_ehvi(candidates: Sequence[CandidateRecord], ehvi: np.ndarray) -> None:
    for candidate, value in zip(candidates, ehvi):
        candidate.ehvi = float(value)
