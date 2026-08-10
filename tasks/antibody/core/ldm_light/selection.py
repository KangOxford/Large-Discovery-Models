"""Acquisition reduction rules shared by antibody LDM variants."""
from __future__ import annotations

from typing import Sequence

import numpy as np


REDUCTION_RULES = ("max", "softmax")


def acquisition_probabilities(
    scores: Sequence[float],
    *,
    reduction: str,
    eta: float,
) -> np.ndarray:
    """Return a stable categorical distribution over acquisition scores."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one acquisition score is required")

    reduction = str(reduction).lower()
    if reduction not in REDUCTION_RULES:
        raise ValueError(f"reduction must be one of {REDUCTION_RULES}")
    if np.isnan(float(eta)) or float(eta) < 0:
        raise ValueError("eta must be non-negative or positive infinity")

    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.size, 1.0 / values.size, dtype=float)

    safe_values = np.where(finite, values, -np.inf)
    if reduction == "max" or np.isposinf(float(eta)):
        probabilities = np.zeros(values.size, dtype=float)
        probabilities[int(np.argmax(safe_values))] = 1.0
        return probabilities

    if float(eta) == 0.0:
        return np.full(values.size, 1.0 / values.size, dtype=float)

    floor = float(values[finite].min()) - 1.0
    stable_values = np.where(finite, values, floor)
    logits = float(eta) * (stable_values - stable_values.max())
    weights = np.exp(logits)
    return weights / weights.sum()


def select_by_acquisition(
    scores: Sequence[float],
    *,
    batch_size: int,
    reduction: str,
    eta: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[float]]:
    """Select unique candidate indices by Max or acquisition Softmax."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    probabilities = acquisition_probabilities(values, reduction=reduction, eta=eta)
    k = min(int(batch_size), values.size)

    if str(reduction).lower() == "max" or np.isposinf(float(eta)):
        finite_values = np.where(np.isfinite(values), values, -np.inf)
        selected = np.argsort(-finite_values, kind="stable")[:k]
    else:
        selected = rng.choice(values.size, size=k, replace=False, p=probabilities)
    return [int(index) for index in selected], [float(value) for value in probabilities]
