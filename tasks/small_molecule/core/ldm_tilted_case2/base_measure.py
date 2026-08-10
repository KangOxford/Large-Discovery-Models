"""Empirical base-measure construction for M1."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord


def apply_m1_base_measure(
    candidates: Sequence[CandidateRecord],
    smoothing: float = 0.0,
) -> np.ndarray:
    """M1 q0 from direct LLM occurrence counts."""
    counts = np.asarray([_total_occurrences(candidate) for candidate in candidates], dtype=float)
    if smoothing < 0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}")
    if smoothing > 0 and counts.size:
        counts = counts + float(smoothing)
    q0 = _normalize_positive(counts)
    _write_q0(candidates, q0)
    return q0


def q0_entropy(q0: np.ndarray, *, eps: float = 1e-12) -> float:
    prob = _normalize_positive(np.asarray(q0, dtype=float))
    if prob.size == 0:
        return 0.0
    return float(-np.sum(prob * np.log(prob + eps)))


def q0_effective_support(q0: np.ndarray, *, eps: float = 1e-12) -> float:
    return float(math.exp(q0_entropy(q0, eps=eps)))


def _total_occurrences(candidate: CandidateRecord) -> int:
    return sum(int(count) for count in candidate.occurrence_by_source.values())


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return values
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / len(values), dtype=float)
    return values / total


def _write_q0(candidates: Sequence[CandidateRecord], q0: np.ndarray) -> None:
    for candidate, value in zip(candidates, q0):
        candidate.q0_base_mass = float(value)
