"""Iterative random analog search over SMILES.

Each round:
  1. **Refill pool** while ``len(pool) < pool_min_size``.
     Default ``pool_min_size=1`` keeps the pool buffered with
     one pending candidate; the analog generator is invoked whenever
     the pool would otherwise deplete. Pass ``None`` to disable
     refill entirely (no analog generation, only the seed SMILES will
     be evaluated). Pick strategy: ``"random"`` (uniform) or ``"best"``
     (Chebyshev scalarization with a random simplex reference
     direction; respects ``minimize`` per objective).
  2. **Score a batch** of ``batch_size`` SMILES drawn from the pool
     (FIFO if ``pool_max_size`` is set, else uniform random).
     Scoring does **not** expand; expansion only happens in step 1
     (lazy expansion).
  3. Repeat until ``n_iterations`` evaluations are recorded.

The pool is a :class:`strbo_v1.utils.FIFOSet` (FIFO-ordered, with
O(1) membership). When ``pool_max_size`` is set, the underlying
deque has ``maxlen=pool_max_size`` and appending a new analogue
auto-evicts the oldest entry. Batch picking is FIFO (``popleft``) when
bounded, and uniform-random over the queue when unbounded. This bounds
memory and ensures older candidates eventually time out.

The refill expansion is per-target (one SMILES at a time) — unlike the
BO loop, the random loop doesn't have a natural batch boundary
because refill is lazy and reactive. The ``analog_fn`` is invoked
with a single-element list and returns a flat list of analogues; the
``expanded`` set ensures the same seed is never passed to
``analog_fn`` twice within a run.

Analogue SMILES whose ``len(str)`` exceeds ``smiles_max_len`` (default
50) are dropped at the pool-ingestion step in both the seed-seeding
loop and the analog-generator output. The filter uses the raw stripped
text length (no canonicalization), so non-canonical runs see the
literal length cap.

Single- vs multi-objective
--------------------------
The public :func:`random_analog_search` accepts a
:class:`strbo_v1.scorer.Scorers` value (single scorer or tuple
thereof). Internally the history is stored as
``OrderedDict[str, tuple[Optional[float], ...]]`` of length ``n_obj``.
For the public return type, ``n_obj == 1`` is unpacked to the legacy
``list[tuple[str, Optional[float]]]`` shape; ``n_obj >= 2`` is returned
as ``list[tuple[str, tuple[Optional[float], ...]]]``.

The ``expansion="best"`` strategy uses
:func:`strbo_v1.acquisition.chebyshev_scalarize` with a sampled
simplex weight vector. For ``n_obj == 1`` this reduces to a
``minimize``-aware scalarization equivalent to the legacy "best
score in history" pick. For ``n_obj >= 2`` it is ParEGO-style
multi-objective selection. The ``expansion="random"`` strategy is
unchanged.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from strbo_v1.acquisition import chebyshev_scalarize, sample_simplex_weights
from strbo_v1.rng import RNG, as_rng
from strbo_v1.scorer import Scorer, Scorers, as_scorer_tuple
from strbo_v1.utils import FIFOSet


def _as_minimize_tuple(
    minimize: Union[bool, Sequence[bool]], n_obj: int
) -> Tuple[bool, ...]:
    """Normalise ``minimize`` to a tuple of booleans of length ``n_obj``."""
    if isinstance(minimize, bool):
        return (minimize,) * n_obj
    seq = list(minimize)
    if len(seq) != n_obj:
        raise ValueError(
            f"minimize length ({len(seq)}) does not match n_obj ({n_obj})"
        )
    for i, v in enumerate(seq):
        if not isinstance(v, bool):
            raise TypeError(
                f"minimize[{i}] must be bool, got {type(v).__name__}"
            )
    return tuple(seq)


def _safe_score_single(
    scorer: Scorer,
    smiles_list: list,
) -> list:
    """Normalise one scorer's output to ``[Optional[float]]`` aligned to input."""
    try:
        raw = scorer(list(smiles_list))
    except Exception:
        return [None] * len(smiles_list)
    if raw is None:
        return [None] * len(smiles_list)
    out: list = []
    for val in list(raw):
        if val is None:
            out.append(None)
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(f if np.isfinite(f) else None)
    while len(out) < len(smiles_list):
        out.append(None)
    return out[: len(smiles_list)]


def _safe_score_n(
    scorers: Tuple[Scorer, ...],
    smiles_list: list,
) -> list:
    """Score a batch through N scorers and stack into per-SMILES tuples."""
    n_obj = len(scorers)
    per_obj = [_safe_score_single(sc, smiles_list) for sc in scorers]
    return list(zip(*per_obj))


def _pick_expansion_target_che(
    candidates: list,
    history: "OrderedDict[str, Tuple[Optional[float], ...]]",
    pool: FIFOSet,
    expanded: set,
    minimize_t: Tuple[bool, ...],
    rng_obj: RNG,
) -> Optional[str]:
    """Pick one SMILES to expand next (input to ``analog_fn``).

    Strategy: Chebyshev scalarization with a random simplex reference
    direction. The ``candidates`` list is the union of history
    and pool minus ``expanded``; among the candidates that have at
    least one finite score vector, we compute the per-objective
    ideal point, sample a simplex weight vector, and pick the
    candidate that minimises the Chebyshev scalarized value.
    For ``n_obj == 1`` this collapses to the legacy
    "best score in history" pick (after sign inversion for
    maximisation). Falls back to uniform random when no candidate
    has any finite score.
    """
    n_obj = len(minimize_t)
    finite = [
        (s, history[s])
        for s in candidates
        if history.get(s) is not None
        and len(history[s]) == n_obj
        and all(sc is not None for sc in history[s])
    ]
    if not finite:
        return rng_obj.python.choice(sorted(candidates))
    # Per-objective ideal point.
    ideal = np.asarray(
        [
            min(p[1][i] for p in finite) if minimize_t[i] else max(p[1][i] for p in finite)
            for i in range(n_obj)
        ],
        dtype=float,
    )
    weights = sample_simplex_weights(rng_obj, n=n_obj, alpha=1.0)
    scored = [
        (
            s,
            chebyshev_scalarize(
                point=np.asarray(history[s], dtype=float),
                weights=weights,
                ideal=ideal,
                minimize=minimize_t,
            ),
        )
        for s, _ in finite
    ]
    # Tie-break by SMILES string for determinism.
    scored.sort(key=lambda x: (x[1], x[0]))
    return scored[0][0]


def _pick_expansion_target(
    strategy: str,
    history: "OrderedDict[str, Optional[float]]",
    pool,
    expanded: set,
    *,
    minimize: bool = True,
    rng=None,
) -> Optional[str]:
    """Backwards-compatible single-objective ``_pick_expansion_target``.

    The legacy single-obj API used bare-float history entries; the
    multi-obj loop uses :func:`_pick_expansion_target_che` with
    tuple-valued history. This wrapper is kept for existing tests
    and for any third-party code that imported the helper directly.
    New code should call :func:`random_analog_search` directly or use
    :func:`_pick_expansion_target_che`.
    """
    if rng is None:
        rng_obj = RNG(seed=None)
    else:
        rng_obj = as_rng(rng)
    candidates = (set(history.keys()) | set(pool)) - expanded
    if not candidates:
        return None
    if strategy == "best":
        # Wrap the legacy single-float history in a length-1 tuple for
        # the Chebyshev helper, then unwrap.
        wrapped: "OrderedDict[str, Tuple[Optional[float], ...]]" = OrderedDict(
            (s, (h,)) for s, h in history.items()
        )
        # Only consider history members (the legacy API picked from
        # history ∪ pool, but for "best" we scored only history; pool
        # SMILES with no score are still candidates via random).
        chosen = _pick_expansion_target_che(
            sorted(candidates), wrapped, pool, expanded,
            (minimize,), rng_obj,
        )
        if chosen in history:
            return chosen
        # None of the candidates have a finite score → random pick.
        return rng_obj.python.choice(sorted(candidates))
    return rng_obj.python.choice(sorted(candidates))


def _add_unique_to_pool(
    pool: FIFOSet,
    raw_smis: Iterable[str],
    history: "OrderedDict[str, Tuple[Optional[float], ...]]",
    expanded: set,
    *,
    smiles_max_len: int,
) -> None:
    """Append valid, novel SMILES from ``raw_smis`` to ``pool``."""
    for raw in raw_smis:
        text = str(raw or "").strip()
        if not text:
            continue
        if smiles_max_len is not None and len(text) > smiles_max_len:
            continue
        if text in expanded or text in history or text in pool:
            continue
        pool.add(text)


def _return_history(
    history: "OrderedDict[str, Tuple[Optional[float], ...]]", n_obj: int
) -> list:
    """Unpack the internal tuple-of-floats history to the public shape."""
    if n_obj == 1:
        return [(smi, scores[0]) for smi, scores in history.items()]
    return [(smi, scores) for smi, scores in history.items()]


def random_analog_search(
    seed_smiles: Iterable[str],
    scorer: Scorers,
    analog_fn: "Callable[[Sequence[str]], Sequence[str]]",
    *,
    n_iterations: int = 10,
    batch_size: int = 1,
    pool_min_size: Optional[int] = 1,
    pool_max_size: Optional[int] = None,
    smiles_max_len: int = 50,
    expansion: str = "random",
    minimize: Union[bool, Tuple[bool, ...]] = True,
    rng: Optional[Union[RNG, random.Random]] = None,
    verbose: bool = False,
) -> list:
    """Iterative random analog search.

    Args:
        seed_smiles: Initial SMILES to seed the pool. SMILES longer than
            ``smiles_max_len`` are silently dropped at seed time.
        scorer: A :class:`Scorers` value — single :class:`Scorer` or
            a tuple of scorers for the multi-objective case.
        analog_fn: A batched analog generator. Takes a sequence of
            seed SMILES and returns a flat sequence of analogue
            SMILES — a simple ``Iterable[str] -> Iterable[str]``
            contract. The refill step invokes it with a
            single-element list.
        n_iterations: Maximum number of evaluations to perform.
        batch_size: SMILES scored per round (parallel batch).
        pool_min_size: When set (default ``1``), refill via expansion
            while ``len(pool) < pool_min_size``. Pass ``None`` to
            disable refill entirely (only the seed SMILES will ever be
            evaluated).
        pool_max_size: When set, the pool is a bounded FIFO
            ``deque(maxlen=pool_max_size)``. Older candidates are
            auto-evicted and picked first.
        smiles_max_len: SMILES length cap. Analogue SMILES (and seed
            SMILES) whose ``len(stripped_text)`` exceeds this cap are
            silently dropped. ``None`` disables the filter. Default
            ``50``.
        expansion: ``"random"`` (uniform) or ``"best"`` (Chebyshev
            scalarization with a random simplex reference direction,
            respecting ``minimize`` per objective).
        minimize: ``True`` (default) treats smaller scores as better
            (Vina convention). For multi-objective, pass a tuple of
            booleans matching the scorer tuple. A bare ``bool`` is
            broadcast to all objectives.
        rng: Optional ``random.Random`` (auto-promoted to :class:`RNG`)
            or :class:`RNG`. The :class:`RNG` class unifies Python and
            NumPy streams for reproducibility across the loop's pool
            sampling and the Chebyshev simplex-weight draws.
        verbose: If True, print a one-line progress message per scoring
            batch and per refill expansion.

    Returns:
        A list of ``(smiles, score)`` tuples in the order each SMILES
        was evaluated. * Single-objective: ``score`` is
        ``Optional[float]``. * Multi-objective: ``score`` is a tuple
        of ``Optional[float]`` of length ``n_obj``. The list may be
        shorter than ``n_iterations`` if the pool is exhausted early.

    Example:
        .. code-block:: python

            from strbo_v1 import VinaScorer, VinaScorerConfig
            from strbo_v1.analog import generate_analogs, ReasynConfig
            from strbo_v1.random_search import random_analog_search

            scorer = VinaScorer(VinaScorerConfig(vina_bin="../bin/vina", cache_dir=...))
            reasyn_config = ReasynConfig(...)
            def analog_fn(smis):
                df = generate_analogs(smis, reasyn_config)
                return df["smiles"].tolist() if df is not None and len(df) > 0 else []

            history = random_analog_search(
                seed_smiles=["CCO", "CCN"],
                scorer=scorer,  # single
                analog_fn=analog_fn,
                n_iterations=5, batch_size=1,
                pool_min_size=1, pool_max_size=20,
                smiles_max_len=50, expansion="random",
                rng=random.Random(42),
            )
    """
    if rng is None:
        rng = RNG(seed=None)
    if expansion not in ("random", "best"):
        raise ValueError(f"expansion must be 'random' or 'best', got {expansion!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if n_iterations < 0:
        raise ValueError(f"n_iterations must be >= 0, got {n_iterations}")
    if pool_min_size is not None and pool_min_size < 1:
        raise ValueError(f"pool_min_size must be >= 1, got {pool_min_size}")
    if pool_max_size is not None and pool_max_size < 1:
        raise ValueError(f"pool_max_size must be >= 1, got {pool_max_size}")
    if (
        pool_min_size is not None
        and pool_max_size is not None
        and pool_min_size > pool_max_size
    ):
        raise ValueError(
            f"pool_min_size ({pool_min_size}) must be <= "
            f"pool_max_size ({pool_max_size})"
        )
    if smiles_max_len is not None and smiles_max_len < 1:
        raise ValueError(f"smiles_max_len must be >= 1, got {smiles_max_len}")

    scorers_tuple = as_scorer_tuple(scorer)
    n_obj = len(scorers_tuple)
    if n_obj == 0:
        raise ValueError("scorer tuple is empty; need at least 1 scorer")
    minimize_t = _as_minimize_tuple(minimize, n_obj)
    rng_obj = as_rng(rng)

    pool: FIFOSet = FIFOSet(max_size=pool_max_size)

    for smiles in seed_smiles:
        text = str(smiles or "").strip()
        if not text:
            continue
        if smiles_max_len is not None and len(text) > smiles_max_len:
            continue
        pool.add(text)

    expanded: set = set()
    history: "OrderedDict[str, Tuple[Optional[float], ...]]" = OrderedDict()

    while len(history) < n_iterations:
        # ----- 1. Refill pool until min_size (if set) -----
        if pool_min_size is not None:
            while len(pool) < pool_min_size and len(history) < n_iterations:
                candidates = (set(history.keys()) | set(pool)) - expanded
                if not candidates:
                    break
                if expansion == "best":
                    target = _pick_expansion_target_che(
                        sorted(candidates), history, pool, expanded,
                        minimize_t, rng_obj,
                    )
                else:
                    target = rng_obj.python.choice(sorted(candidates))
                if target is None:
                    break
                try:
                    analogues = list(analog_fn([target]))
                except Exception:
                    analogues = []
                expanded.add(target)
                if verbose:
                    print(
                        f"    [iter {len(history)+1}/{n_iterations}] refill: expanded {target!r} "
                        f"(pool={len(pool)}, got {len(analogues)} analogs)",
                        flush=True,
                    )
                _add_unique_to_pool(
                    pool, analogues, history, expanded,
                    smiles_max_len=smiles_max_len,
                )

        if not pool:
            break

        # ----- 2. Pick batch from pool -----
        k = min(batch_size, len(pool), n_iterations - len(history))
        if k <= 0:
            break

        if pool_max_size is not None:
            batch = [pool.popleft() for _ in range(k)]
            pick_mode = "FIFO"
        else:
            batch = rng_obj.python.sample(list(pool), k=k)
            for smi in batch:
                pool.discard(smi)
            pick_mode = "random"

        # ----- 3. Score (no expansion here) -----
        if verbose:
            print(
                f"    [iter {len(history)+1}/{n_iterations}] scoring batch of {k} "
                f"({pick_mode}, pool={len(pool)}, n_obj={n_obj})...",
                flush=True,
            )
        normalized = _safe_score_n(scorers_tuple, batch)

        for smi, sc in zip(batch, normalized):
            history[smi] = sc
            if verbose:
                print(
                    f"    [iter {len(history)}/{n_iterations}]   score for {smi!r}: {sc}",
                    flush=True,
                )

    return _return_history(history, n_obj)


# ---------------------------------------------------------------------------
# Public advisor step
# ---------------------------------------------------------------------------


def select_next_batch(
    pool: Sequence[str],
    *,
    batch_size: int,
    rng: Optional[Union[RNG, random.Random]] = None,
) -> List[str]:
    """Pick a uniform-random batch from ``pool`` (the random-search advisor step).

    This is the per-round decision step for the random-search family
    (``random`` and ``random-best``). The "best" strategy in
    :func:`random_analog_search` only affects the *expansion* (refill)
    target — which pool member to expand via analog generation — not
    the *evaluation* pick. Both methods use this function for the
    per-round scoring decision.

    Args:
        pool: Candidate SMILES to choose from. The function does not
            modify the list. May be empty (returns ``[]``).
        batch_size: How many to pick.
        rng: Optional :class:`RNG` or :class:`random.Random`.
            ``None`` (default) builds a fresh non-deterministic
            :class:`RNG`.

    Returns:
        ``min(batch_size, len(pool))`` SMILES, sampled without
        replacement via the provided ``rng`` (uniform random).
        Order in the output matches the RNG draw order.

    Raises:
        ValueError: If ``batch_size < 1``.
    """
    if batch_size is None or batch_size < 1:
        raise ValueError(
            f"select_next_batch requires batch_size >= 1, got {batch_size!r}"
        )
    pool_list = [str(s) for s in pool if s is not None and str(s).strip()]
    if not pool_list:
        return []
    k = min(batch_size, len(pool_list))
    rng_obj = as_rng(rng) if rng is not None else RNG(seed=None)
    return rng_obj.python.sample(pool_list, k=k)


__all__ = [
    "random_analog_search",
    "select_next_batch",
    "Scorer",
    "_pick_expansion_target",
    "_pick_expansion_target_che",
]
