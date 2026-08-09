"""EHVI-tilted probabilities and weighted sampling."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.rng import RNG, as_rng


MAD_SCALE = 1.4826


def robust_z(values: np.ndarray, *, clip: float = 5.0, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = MAD_SCALE * mad
    if scale <= eps:
        scale = float(np.std(arr))
    if scale <= eps:
        return np.zeros_like(arr)
    return np.clip((arr - median) / (scale + eps), -clip, clip)


def tilted_logits(q0: np.ndarray, ehvi: np.ndarray, cfg: TiltedLDMCase2Config) -> np.ndarray:
    q = _normalize_probability(q0)
    acq_z = robust_z(ehvi, clip=cfg.z_clip, eps=cfg.eps)
    if q.shape != acq_z.shape:
        raise ValueError(f"q0 and ehvi shape mismatch: {q.shape} vs {acq_z.shape}")
    return cfg.alpha_base_measure * np.log(q + cfg.eps) + cfg.eta_ehvi_tilt * acq_z


def tilted_probabilities(
    q0: np.ndarray,
    ehvi: np.ndarray,
    cfg: TiltedLDMCase2Config,
) -> np.ndarray:
    logits = tilted_logits(q0, ehvi, cfg)
    if logits.size == 0:
        return logits
    shifted = logits - float(np.max(logits))
    exp_values = np.exp(shifted)
    return exp_values / float(exp_values.sum())


def gumbel_top_k(prob: np.ndarray, k: int, rng: RNG | None) -> list[int]:
    p = _normalize_probability(prob)
    if p.size == 0 or k <= 0:
        return []
    rng_obj = as_rng(rng)
    k = min(int(k), len(p))
    scores = np.log(p + 1e-300) + rng_obj.numpy.gumbel(size=len(p))
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def weighted_sample_candidates(
    candidates: Sequence[CandidateRecord],
    prob: np.ndarray,
    k: int,
    rng: RNG | None,
) -> list[CandidateRecord]:
    p = _normalize_probability(prob)
    if len(candidates) != len(p):
        raise ValueError("candidate and probability length mismatch")
    for candidate, value in zip(candidates, p):
        candidate.resampling_probability = float(value)
    indices = gumbel_top_k(p, k, rng)
    selected = [candidates[i] for i in indices]
    for candidate in selected:
        candidate.selected = True
    return selected


def effective_sample_size(prob: np.ndarray) -> float:
    p = _normalize_probability(prob)
    if p.size == 0:
        return 0.0
    return float(1.0 / np.sum(p * p))


def probability_entropy(prob: np.ndarray, *, eps: float = 1e-12) -> float:
    p = _normalize_probability(prob)
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + eps)))


def selected_rank_by_ehvi(candidates: Sequence[CandidateRecord]) -> list[int]:
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: float("-inf") if item[1].ehvi is None else float(item[1].ehvi),
        reverse=True,
    )
    ranks = {idx: rank + 1 for rank, (idx, _candidate) in enumerate(ordered)}
    return [ranks[i] for i, candidate in enumerate(candidates) if candidate.selected]


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / len(arr), dtype=float)
    return arr / total
