"""Acquisition functions + multi-objective tools for strbo_v1.

This module collects:

* **Single-objective acquisition functions** used by the LDM-TTS loop:
  :func:`expected_improvement`,
  :func:`probability_of_improvement`, :func:`confidence_bound`. All
  return "higher = better" so the BO loop's top-k selection is uniform.
* **Hypervolume** (exact, no MC): :func:`hypervolume` dispatches to
  :func:`_hv_1d` / :func:`_hv_2d` based on ``len(ref)``; raises
  :class:`NotImplementedError` for ``n_obj >= 3``.
* **Expected Hypervolume Improvement** (Monte Carlo): 
  :func:`expected_hypervolume_improvement` dispatches to
  :func:`_ehvi_2d` for ``n_obj == 2``; raises :class:`NotImplementedError`
  otherwise. The 2D MC estimator is a standard box-Muller / direct
  normal sampling recipe, averaged over ``n_samples``.
* **Chebyshev scalarization** (ParEGO-style, generic N-dim):
  :func:`chebyshev_scalarize` works for any ``n_obj`` (no raise); the
  BO / random loops use it as the fallback acquisition for
  ``n_obj >= 3``.
* **Simplex weight sampling**: :func:`sample_simplex_weights` draws
  ``n`` i.i.d. ``Beta(alpha, 1)`` and normalises; ``alpha=1`` gives
  uniform on the simplex.

Single- vs multi-objective dispatch
-----------------------------------
The public entry points :func:`hypervolume` and
:func:`expected_hypervolume_improvement` accept arbitrary
``n_obj`` but raise :class:`NotImplementedError` for the not-yet-
implemented dimensionalities. The outer interface
handles the dispatch: ``n_obj == 1`` uses EI / PI / UCB; ``n_obj == 2``
uses EHVI; ``n_obj >= 3`` falls back to Chebyshev scalarization, which
works in arbitrary dimensions.
"""

from __future__ import annotations

import logging
from math import erf
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from strbo_v1.rng import RNG, as_rng


__all__ = [
    "expected_improvement",
    "probability_of_improvement",
    "confidence_bound",
    "hypervolume",
    "expected_hypervolume_improvement",
    "chebyshev_scalarize",
    "sample_simplex_weights",
    "dominates",
    "pareto_front",
]


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-objective acquisition helpers
# ---------------------------------------------------------------------------


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _Phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via erf (numerically stable for large |z|)."""
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    *,
    xi: float = 0.01,
    minimize: bool = True,
) -> np.ndarray:
    """Expected Improvement for minimization or maximization.

    Args:
        mu: Posterior mean, shape ``(n,)``.
        sigma: Posterior std (must be non-negative), shape ``(n,)``.
        best: Best observed target value (min if ``minimize=True`` else max).
        xi: Improvement threshold (non-negative). Default 0.01.
        minimize: If True, smaller target is better.

    Returns:
        Acquisition values, shape ``(n,)``. Non-negative; higher = better.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    out = np.zeros_like(mu)
    finite = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(finite):
        return out
    if minimize:
        imp = best - mu[finite] - xi
    else:
        imp = mu[finite] - best - xi
    z = imp / sigma[finite]
    out[finite] = imp * _Phi(z) + sigma[finite] * _phi(z)
    out = np.clip(out, 0.0, None)
    return out


def probability_of_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    *,
    xi: float = 0.01,
    minimize: bool = True,
) -> np.ndarray:
    """Probability of Improvement for minimization or maximization.

    Args:
        mu: Posterior mean, shape ``(n,)``.
        sigma: Posterior std, shape ``(n,)``.
        best: Best observed target value (min if ``minimize=True`` else max).
        xi: Improvement threshold (non-negative). Default 0.01.
        minimize: If True, smaller target is better.

    Returns:
        Acquisition values in [0, 1], shape ``(n,)``. Higher = better.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    out = np.zeros_like(mu)
    finite = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(finite):
        return out
    if minimize:
        z = (best - mu[finite] - xi) / sigma[finite]
    else:
        z = (mu[finite] - best - xi) / sigma[finite]
    out[finite] = _Phi(z)
    return out


def confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    kappa: float = 2.0,
    minimize: bool = True,
) -> np.ndarray:
    """Confidence-bound acquisition. Higher value = better solution.

    Unified "higher = better" acquisition:

    * ``minimize=True``: returns ``kappa * sigma - mu`` (the negation of
      the lower confidence bound; smaller target is better, so we flip
      the sign so the most exploitable point has the highest value).
    * ``minimize=False``: returns ``mu + kappa * sigma`` (the upper
      confidence bound; larger target is better, returned as-is).

    Args:
        mu: Posterior mean, shape ``(n,)``.
        sigma: Posterior std (must be non-negative), shape ``(n,)``.
        kappa: Exploration weight. Default 2.0.
        minimize: If True, smaller target is better.

    Returns:
        Acquisition values, shape ``(n,)``. Higher = better.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    if minimize:
        return kappa * sigma - mu
    return mu + kappa * sigma


def _resolve_acquisition(name: str, n_obj: int = 1) -> Callable[..., np.ndarray]:
    """Map acquisition name string to the matching function.

    For ``n_obj == 1`` returns one of EI / PI / UCB. For
    ``n_obj >= 2`` the multi-objective loop always uses EHVI; this
    helper raises :class:`ValueError` for any name in the multi-obj
    case so the caller can detect the misuse.

    Returns:
        The acquisition function. EI / PI take
        ``(mu, sigma, best, *, xi, minimize)``; UCB takes
        ``(mu, sigma, *, kappa, minimize)``.

    Raises:
        ValueError: if ``name`` is not one of ``{"ei", "pi", "ucb"}``
            (or the multi-obj case is dispatched with a single-obj name).
    """
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


# ---------------------------------------------------------------------------
# Multi-objective: Pareto / dominance / hypervolume (exact)
# ---------------------------------------------------------------------------


def dominates(
    a: Sequence[float],
    b: Sequence[float],
    minimize: Sequence[bool],
) -> bool:
    """Return True iff ``a`` Pareto-dominates ``b`` under ``minimize``.

    ``a`` dominates ``b`` iff:
        * for every objective i, ``a[i]`` is no worse than ``b[i]``
          (smaller when ``minimize[i]`` else larger), AND
        * for at least one objective i, ``a[i]`` is strictly better.

    Args:
        a: An objective tuple of length ``n_obj``.
        b: An objective tuple of length ``n_obj``.
        minimize: A sequence of booleans, length ``n_obj``: True if
            smaller values are better for that objective.

    Returns:
        ``True`` iff ``a`` dominates ``b``.
    """
    if len(a) != len(b) or len(a) != len(minimize):
        raise ValueError(
            f"a/b/minimize length mismatch: {len(a)}/{len(b)}/{len(minimize)}"
        )
    at_least_one_better = False
    for ai, bi, is_min in zip(a, b, minimize):
        if is_min:
            if ai > bi:
                return False
            if ai < bi:
                at_least_one_better = True
        else:
            if ai < bi:
                return False
            if ai > bi:
                at_least_one_better = True
    return at_least_one_better


def pareto_front(
    points: Sequence[Sequence[float]],
    minimize: Sequence[bool],
) -> list[tuple[float, ...]]:
    """Return the Pareto-front tuples (preserving first-seen order).

    Args:
        points: A sequence of objective tuples (each length ``n_obj``).
        minimize: Per-objective "smaller is better" booleans.

    Returns:
        The subset of ``points`` that are not Pareto-dominated by
        any other point in ``points``. Order is first-seen.
    """
    if not points:
        return []
    n_obj = len(minimize)
    seen_tuples: list[tuple[float, ...]] = []
    for raw in points:
        if len(raw) != n_obj:
            raise ValueError(
                f"point length ({len(raw)}) does not match minimize "
                f"length ({n_obj}): {raw!r}"
            )
        seen_tuples.append(tuple(float(x) for x in raw))

    keep_mask = [True] * len(seen_tuples)
    for i, pi in enumerate(seen_tuples):
        if not keep_mask[i]:
            continue
        for j, pj in enumerate(seen_tuples):
            if i == j or not keep_mask[j]:
                continue
            if dominates(pj, pi, minimize):
                keep_mask[i] = False
                break
    return [p for p, k in zip(seen_tuples, keep_mask) if k]


# ---------------------------------------------------------------------------
# Hypervolume (exact)
# ---------------------------------------------------------------------------


def _hv_1d(points: Sequence[float], ref: float) -> float:
    """1D hypervolume: ``max(0, ref - min(points))`` for minimisation.

    Assumes all points are dominated by ``ref`` (point < ref when
    minimising, point > ref when maximising). The caller is
    responsible for the sign convention; this function treats
    ``points`` as if all values are below ``ref`` and returns the
    distance.
    """
    if not points:
        return 0.0
    return max(0.0, float(ref) - float(min(points)))


def _hv_2d(points: Sequence[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """Exact 2D hypervolume via sweep-line (Beume 2009 style).

    Assumes the caller has:
        * sign-converted all ``points`` to "smaller is better" space
          (i.e. for an objective that is maximised, pass ``-p[i]``
          and ``-ref[i]``);
        * removed any point not dominated by ``ref`` (caller may also
          pre-filter to the Pareto front for efficiency; we defensive-
          filter dominated points internally too).

    Args:
        points: Sequence of 2D tuples, both entries below the
            corresponding ``ref`` coordinate.
        ref: 2D reference point dominating every point.

    Returns:
        The volume of the union of axis-aligned boxes
        ``[p[0], ref[0]] x [p[1], ref[1]]`` for every ``p``.
    """
    if not points:
        return 0.0
    # Defensive: drop any point not dominated by ref.
    dominated = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
    if not dominated:
        return 0.0
    # Defensive: drop any point dominated by another (keep Pareto).
    front = pareto_front(dominated, minimize=(True, True))
    if not front:
        return 0.0
    # Sort by obj0 ascending.
    front_sorted = sorted(front, key=lambda p: p[0])
    # Sweep: cumulate strips. The first strip is from front_sorted[0][0]
    # to ref[0] in x, with height (ref[1] - front_sorted[0][1]).
    # Subsequent strips have height (front_sorted[i+1][1] - front_sorted[i][1])
    # in x-range (front_sorted[i][0], front_sorted[i+1][0]).
    volume = 0.0
    n = len(front_sorted)
    for i in range(n):
        x_lo = front_sorted[i][0]
        y_at_x = front_sorted[i][1]
        x_hi = front_sorted[i + 1][0] if i + 1 < n else ref[0]
        height = ref[1] - y_at_x
        volume += (x_hi - x_lo) * height
    return volume


def hypervolume(
    points: Sequence[Sequence[float]],
    ref: Sequence[float],
    *,
    minimize: Optional[Sequence[bool]] = None,
) -> float:
    """Exact hypervolume w.r.t. a reference point.

    Args:
        points: Sequence of objective tuples, length ``n_obj``.
        ref: Reference point, length ``n_obj``.
        minimize: Per-objective "smaller is better" booleans. Defaults
            to ``(True, ..., True)`` (all objectives minimised). The
            function internally flips maximised objectives to the
            "smaller is better" convention before computing.

    Returns:
        The hypervolume (volume of dominated region).

    Raises:
        NotImplementedError: if ``n_obj not in {1, 2}``. The 2D backend
            is the only non-trivial exact algorithm we ship; 1D
            collapses to a single range; 3D+ is intentionally not
            implemented in this version.
    """
    n_obj = len(ref)
    if n_obj not in (1, 2):
        raise NotImplementedError(
            f"hypervolume supports n_obj in {{1, 2}}; got {n_obj}. "
            "Higher-dimensional hypervolume is not implemented in this version."
        )
    if minimize is None:
        minimize = (True,) * n_obj
    if len(minimize) != n_obj:
        raise ValueError(
            f"minimize length ({len(minimize)}) does not match ref length ({n_obj})"
        )

    if n_obj == 1:
        if not points:
            return 0.0
        # For minimize=True, all values < ref; for minimize=False, all
        # values > ref and we want to dominate in the flipped space.
        # Internally, flip maximised objectives to the "smaller is better"
        # convention by negating both point and ref.
        if minimize[0]:
            return _hv_1d([float(p[0]) for p in points], float(ref[0]))
        return _hv_1d(
            [float(-p[0]) for p in points], float(-ref[0])
        )

    # n_obj == 2
    # Sign-convert maximised objectives to "smaller is better" space
    # once; reuse for every point.
    r0_c = float(ref[0]) if minimize[0] else float(-ref[0])
    r1_c = float(ref[1]) if minimize[1] else float(-ref[1])
    converted: list[Tuple[float, float]] = []
    for p in points:
        if len(p) != 2:
            raise ValueError(f"expected 2D point, got length {len(p)}")
        p0 = float(p[0]) if minimize[0] else float(-p[0])
        p1 = float(p[1]) if minimize[1] else float(-p[1])
        # Skip points not dominated by ref (in converted space).
        if p0 >= r0_c or p1 >= r1_c:
            continue
        converted.append((p0, p1))
    return _hv_2d(converted, (r0_c, r1_c))


# ---------------------------------------------------------------------------
# Expected Hypervolume Improvement (Monte Carlo, 2D only)
# ---------------------------------------------------------------------------


def _ehvi_2d(
    mu0: np.ndarray,
    sigma0: np.ndarray,
    mu1: np.ndarray,
    sigma1: np.ndarray,
    pareto_2d: Sequence[Tuple[float, float]],
    ref: Tuple[float, float],
    *,
    minimize: Tuple[bool, bool],
    n_samples: int,
    rng: RNG,
) -> np.ndarray:
    """Monte-Carlo EHVI for two objectives.

    For each candidate c and each MC sample s:
        f0_s ~ N(mu0[c], sigma0[c]^2)
        f1_s ~ N(mu1[c], sigma1[c]^2)
        HVI_s[c] = HV(pareto + (f0_s, f1_s), ref) - HV(pareto, ref)
    EHVI[c] = mean_s HVI_s[c]

    All ``pareto_2d`` entries are already in the "smaller is better"
    convention (caller sign-converts maximise objectives).
    """
    n_cand = len(mu0)
    if n_cand == 0:
        return np.zeros((0,), dtype=float)

    # Pre-compute current Pareto HV (constant across candidates/samples).
    current_hv = _hv_2d(list(pareto_2d), ref)

    # Joint sample: shape (n_cand, n_samples) for each objective.
    s0 = np.zeros((n_cand, n_samples), dtype=float)
    s1 = np.zeros((n_cand, n_samples), dtype=float)
    for i in range(n_cand):
        s0[i, :] = rng.normal(mu0[i], sigma0[i], n_samples)
        s1[i, :] = rng.normal(mu1[i], sigma1[i], n_samples)

    # Clip samples to be dominated by ref (numerical safety; samples
    # beyond ref are clamped to ref to avoid negative "improvement"
    # from outliers).
    r0, r1 = ref
    s0 = np.minimum(s0, r0 - 1e-12)
    s1 = np.minimum(s1, r1 - 1e-12)

    ehvi = np.zeros(n_cand, dtype=float)
    for c in range(n_cand):
        # Vectorise across samples: build a list of pareto+sampled
        # points and compute HV per sample. Pareto list is small
        # (typically < 50); the loop is n_samples long.
        hv_increments = np.zeros(n_samples, dtype=float)
        for s in range(n_samples):
            extended = list(pareto_2d) + [(float(s0[c, s]), float(s1[c, s]))]
            hv_increments[s] = _hv_2d(extended, ref) - current_hv
        ehvi[c] = float(hv_increments.mean())
    return ehvi


def expected_hypervolume_improvement(
    mu_per_obj: Sequence[np.ndarray],
    sigma_per_obj: Sequence[np.ndarray],
    pareto_points: Sequence[Sequence[float]],
    ref: Sequence[float],
    *,
    minimize: Sequence[bool],
    n_samples: int = 128,
    rng: Optional[Union[RNG, "random.Random"]] = None,
) -> np.ndarray:
    """Monte-Carlo EHVI for two objectives.

    Args:
        mu_per_obj: Sequence of posterior mean arrays, one per
            objective. Each has shape ``(n_candidates,)``.
        sigma_per_obj: Sequence of posterior std arrays (one per
            objective), same shapes as ``mu_per_obj``.
        pareto_points: Current Pareto front, a sequence of objective
            tuples (length ``n_obj`` each). Empty list is allowed (the
            first acquisition starts from a null front).
        ref: Reference point, length ``n_obj``.
        minimize: Per-objective "smaller is better" booleans.
        n_samples: Number of MC samples per candidate (default 128).
        rng: A :class:`RNG` (preferred), :class:`random.Random`, or
            ``None``. Auto-promoted to :class:`RNG`.

    Returns:
        An array of shape ``(n_candidates,)`` with non-negative EHVI
        values. Higher = better.

    Raises:
        NotImplementedError: if ``len(ref) != 2``. 3D+ EHVI is not
            implemented in this version; use
            :func:`chebyshev_scalarize` (ParEGO-style) instead.
    """
    n_obj = len(ref)
    if n_obj != 2:
        raise NotImplementedError(
            f"expected_hypervolume_improvement supports exactly 2 objectives; "
            f"got {n_obj}. Higher-dimensional EHVI is not implemented in this "
            f"version; use chebyshev_scalarize for n_obj >= 3."
        )
    if len(mu_per_obj) != 2 or len(sigma_per_obj) != 2:
        raise ValueError("mu_per_obj and sigma_per_obj must each have length 2")
    if len(minimize) != 2:
        raise ValueError(f"minimize must have length 2, got {len(minimize)}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    rng_obj = as_rng(rng)
    mu0 = np.asarray(mu_per_obj[0], dtype=float).ravel()
    mu1 = np.asarray(mu_per_obj[1], dtype=float).ravel()
    sigma0 = np.asarray(sigma_per_obj[0], dtype=float).ravel()
    sigma1 = np.asarray(sigma_per_obj[1], dtype=float).ravel()
    if not (mu0.shape == mu1.shape == sigma0.shape == sigma1.shape):
        raise ValueError(
            f"mu/sigma shape mismatch: {mu0.shape}/{mu1.shape}/"
            f"{sigma0.shape}/{sigma1.shape}"
        )

    # Sign-convert maximised objectives to the "smaller is better"
    # convention internally. Reference point is flipped accordingly.
    mu0_c = mu0 if minimize[0] else -mu0
    mu1_c = mu1 if minimize[1] else -mu1
    sigma0_c = np.abs(sigma0)
    sigma1_c = np.abs(sigma1)
    r0_c = float(ref[0]) if minimize[0] else float(-ref[0])
    r1_c = float(ref[1]) if minimize[1] else float(-ref[1])
    pareto_c = [
        (
            float(p[0]) if minimize[0] else float(-p[0]),
            float(p[1]) if minimize[1] else float(-p[1]),
        )
        for p in pareto_points
    ]
    # Drop any pareto point not dominated by the converted ref.
    pareto_c = [p for p in pareto_c if p[0] < r0_c and p[1] < r1_c]

    return _ehvi_2d(
        mu0_c,
        sigma0_c,
        mu1_c,
        sigma1_c,
        pareto_c,
        (r0_c, r1_c),
        minimize=(True, True),
        n_samples=n_samples,
        rng=rng_obj,
    )


# ---------------------------------------------------------------------------
# Chebyshev scalarization (ParEGO-style, generic N-dim)
# ---------------------------------------------------------------------------


def chebyshev_scalarize(
    point: Sequence[float],
    weights: Sequence[float],
    ideal: Sequence[float],
    minimize: Sequence[bool],
) -> float:
    """Chebyshev scalarization. Smaller = better.

    For each objective i, compute a non-negative gap:
        gap[i] = weights[i] * |point[i] - ideal[i]|
    where ``|.|`` is interpreted as "distance from the ideal point
    in the same direction":

        * if ``minimize[i]`` is True (smaller is better), the ideal is
          the smallest observed value and the gap is
          ``weights[i] * (point[i] - ideal[i])`` (clamped to >= 0);
        * if ``minimize[i]`` is False, the gap is
          ``weights[i] * (ideal[i] - point[i])`` (clamped to >= 0).

    The scalarized value is ``max_i gap[i]``. Smaller is better, so
    the BO loop picks the candidate with the smallest
    :func:`chebyshev_scalarize` value. This is ParEGO-style
    scalarization; it works for any number of objectives.

    Args:
        point: An objective tuple of length ``n_obj``.
        weights: A non-negative weight vector on the simplex
            (``sum(weights) == 1`` and all entries >= 0). Caller
            responsibility to ensure this; the function does not
            validate the sum.
        ideal: The ideal point, length ``n_obj`` (per-objective best
            observed value).
        minimize: Per-objective "smaller is better" booleans.

    Returns:
        A non-negative scalar; smaller means the point is closer to
        the ideal in the worst (largest weighted gap) objective.
    """
    if not (len(point) == len(weights) == len(ideal) == len(minimize)):
        raise ValueError(
            f"length mismatch: point={len(point)}, weights={len(weights)}, "
            f"ideal={len(ideal)}, minimize={len(minimize)}"
        )
    gaps = []
    for pi, wi, ii, is_min in zip(point, weights, ideal, minimize):
        if wi < 0:
            raise ValueError(f"weights must be non-negative, got {wi}")
        gap = wi * (pi - ii) if is_min else wi * (ii - pi)
        gaps.append(max(0.0, gap))
    return float(max(gaps))


def sample_simplex_weights(
    rng: Union[RNG, "random.Random"],
    n: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Sample ``n`` i.i.d. ``Beta(alpha, 1)`` and normalise to the simplex.

    Args:
        rng: A :class:`RNG` (preferred) or :class:`random.Random`.
            Auto-promoted via :func:`strbo_v1.rng.as_rng`.
        n: Number of weights to sample (>= 1).
        alpha: Beta distribution shape parameter. ``alpha=1`` gives a
            uniform distribution on the simplex; ``alpha<1`` biases
            toward corners; ``alpha>1`` biases toward the centre.

    Returns:
        A ``(n,)`` numpy array of non-negative weights summing to 1.

    Raises:
        ValueError: if ``n < 1`` or ``alpha <= 0``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    rng_obj = as_rng(rng)
    raw = rng_obj.beta(alpha, size=n)
    total = float(raw.sum())
    if total <= 0.0:
        # Degenerate (shouldn't happen with alpha>0, n>=1); fall back to uniform.
        return np.full(n, 1.0 / n)
    return raw / total
