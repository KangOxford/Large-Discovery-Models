"""Acquisition functions for sequential minimization."""

from __future__ import annotations

import math


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def expected_improvement_minimize(mean: float, stddev: float, best_value: float, *, xi: float = 0.0) -> float:
    improvement = best_value - mean - xi
    if stddev <= 1e-12:
        return max(improvement, 0.0)
    z = improvement / stddev
    return improvement * normal_cdf(z) + stddev * normal_pdf(z)


def lower_confidence_bound_gain(mean: float, stddev: float, best_value: float, *, beta: float = 1.96) -> float:
    lcb = mean - beta * stddev
    return best_value - lcb


def acquisition_score(
    mean: float,
    stddev: float,
    best_value: float,
    *,
    kind: str = "ei",
    xi: float = 0.0,
    beta: float = 1.96,
) -> float:
    """Return a larger-is-better score for a minimization problem."""

    if kind == "ei":
        return expected_improvement_minimize(mean, stddev, best_value, xi=xi)
    if kind == "lcb":
        return lower_confidence_bound_gain(mean, stddev, best_value, beta=beta)
    if kind == "ei_lcb":
        ei = expected_improvement_minimize(mean, stddev, best_value, xi=xi)
        lcb_gain = max(lower_confidence_bound_gain(mean, stddev, best_value, beta=beta), 0.0)
        return ei + 0.05 * lcb_gain
    raise ValueError(f"Unknown acquisition kind: {kind}")
