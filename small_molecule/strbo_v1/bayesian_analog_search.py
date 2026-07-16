"""Bayesian analog search over SMILES.

Phases (all analog generation is **one batched call per phase**):
  1. **Warm-up** (optional): a single batched call to ``analog_fn``
     with the SMILES needed to grow the pool to ``init_size``.
     Stops early if the pool is already at or above ``init_size``.
  2. **Initialization**: sample ``init_size`` SMILES from the pool
     (without replacement), batch-score them, record into history,
     then make a single batched call to ``analog_fn(init_chosen)``
     and add the resulting analogues back to the pool.
  3. **BO loop**: fit ``GPSurrogate``s on the history (one per
     objective in the multi-objective case), score every pool member
     with the configured acquisition function, take the top-
     ``batch_size`` as candidates, batch-score them, record, then
     make a single batched call to ``analog_fn(candidates)`` and add
     the resulting analogues back to the pool.

The candidate pool is a :class:`strbo_v1.utils.FIFOSet`: a
FIFO-ordered queue with O(1) membership check. When
``max_pool_size`` is set on the config, the pool is bounded and the
oldest entry is auto-evicted on append. Iteration is FIFO (insertion
order). The pick order at each BO round is still GP-acquisition-driven
(top-k by acquisition value).

The ``analog_fn`` parameter takes a sequence of seed SMILES and
returns a flat sequence of analogue SMILES — a simple
``Iterable[str] -> Iterable[str]`` contract. The loop never sees a
DataFrame; any DataFrame handling (e.g., for ReaSyn's
``generate_analogs`` output) lives at the runner boundary in
``run_search.py``. Per-input alignment is intentionally not exposed:
the pool's ``FIFOSet`` + the ``seen`` set handle dedup at the
SMILES level, so an analogue appears in the pool at most once
regardless of how many seeds produced it.

Analogue SMILES whose canonical form (or stripped raw text, when
``canonicalize_pool=False``) is longer than ``smiles_max_len`` (default
50) are dropped at the pool-ingestion step. The same value is used by
the GP string kernel to truncate int64-encoded inputs, so the pool
never accumulates candidates the GP would have to truncate at fit
time.

Single- vs multi-objective dispatch
-----------------------------------
The public :func:`bayesian_analog_search` accepts a
:class:`strbo_v1.scorer.Scorers` (either a single :data:`Scorer` or a
tuple thereof). Internally the scorer tuple is normalised via
:func:`strbo_v1.scorer.as_scorer_tuple` and the loop dispatches by
``n_obj = len(scorers)``:

* ``n_obj == 1``: classical single-objective path with EI / PI / UCB
  on the GP posterior mean / std. The history is internally stored
  as ``(smiles, (score,))`` and unpacked to ``(smiles, score)`` for
  the public return type to preserve the legacy single-obj API.
* ``n_obj == 2``: Expected Hypervolume Improvement (EHVI) via
  Monte-Carlo sampling from the per-objective GP posteriors. One
  GP is fit per objective (shared ``gp_config``); ``mu`` and
  ``sigma`` are 2-tuples of ``np.ndarray`` shaped
  ``(n_candidates,)``.
* ``n_obj >= 3``: ParEGO-style Chebyshev scalarization. One GP per
  objective (shared ``gp_config``). For each BO round, a single
  simplex weight vector is sampled via
  :func:`strbo_v1.acquisition.sample_simplex_weights`; each pool
  member's predicted mean is scalarized via
  :func:`strbo_v1.acquisition.chebyshev_scalarize` and the smallest
  scalarized value wins. This works in arbitrary dimensions; 2D EHVI
  remains the recommended path for ``n_obj == 2`` but Chebyshev is
  also available there.

For ``n_obj >= 1`` the loop never raises on dimensionality grounds
(the outer interface stays general); algorithm-specific
:func:`hypervolume` / :func:`expected_hypervolume_improvement` raise
:class:`NotImplementedError` for n_obj out of their supported range
(1D + 2D, and 2D respectively). The single-objective path uses
EI / PI / UCB and never calls these helpers.
"""

from __future__ import annotations

import logging
import random
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from strbo_v1.acquisition import (
    chebyshev_scalarize,
    confidence_bound,
    expected_hypervolume_improvement,
    expected_improvement,
    probability_of_improvement,
    sample_simplex_weights,
)
from strbo_v1.bo_acquisition import AcquisitionEvaluator, single_acquisition_name
from strbo_v1.gp import GPConfig, GPSurrogate
from strbo_v1.rng import RNG, as_rng
from strbo_v1.scorer import Scorer, Scorers, as_scorer_tuple
from strbo_v1.utils import FIFOSet


LOGGER = logging.getLogger(__name__)


# Public type alias for the public history return. Internally the loop
# always works with ``(smiles, (s1, s2, ...))`` tuples; for the single-
# objective public API we unpack the one-element tuple to a bare float
# to preserve the legacy ``list[tuple[str, Optional[float]]]`` shape.
HistoryTuple = Tuple[str, Tuple[Optional[float], ...]]
HistoryTuplePublic = Union[
    Tuple[str, Optional[float]], Tuple[str, Tuple[Optional[float], ...]]
]


# ---------------------------------------------------------------------------
# Acquisition helpers (single-objective dispatch; multi-obj lives in
# strbo_v1.acquisition).
# ---------------------------------------------------------------------------


def _resolve_acquisition(name: str) -> Callable[..., np.ndarray]:
    """Map acquisition name string to the matching function (single-obj only).

    For multi-objective loops the candidate selection is handled by
    :func:`_select_candidates_2obj` (EHVI) or
    :func:`_select_candidates_3plus` (Chebyshev ParEGO); this helper
    is for the ``n_obj == 1`` path only.

    Returns:
        The acquisition function. EI / PI take
        ``(mu, sigma, best, *, xi, minimize)``; UCB takes
        ``(mu, sigma, *, kappa, minimize)``.

    Raises:
        ValueError: if ``name`` is not one of ``{"ei", "pi", "ucb"}``.
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
# SMILES canonicalization
# ---------------------------------------------------------------------------


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Validate and canonicalize a SMILES string.

    Returns:
        Canonical SMILES, or ``None`` if RDKit cannot parse the input.
    """
    try:
        from rdkit import Chem  # type: ignore
    except ImportError:
        return str(smiles or "").strip() or None
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _add_analogues_to_pool(
    analogues: Iterable[str],
    pool: FIFOSet,
    seen: set[str],
    *,
    canonicalize_pool: bool,
    smiles_max_len: int,
) -> None:
    """Add ``analogues`` to ``pool`` after canonicalization and dedup.

    Invalid SMILES are dropped with a debug-level warning. SMILES already
    in ``seen`` (evaluated) or ``pool`` (pending) are silently skipped.
    SMILES whose canonical form (or stripped raw text, if
    ``canonicalize_pool=False``) is longer than ``smiles_max_len`` are
    silently dropped; this is the same cap the GP string kernel uses
    to truncate int64-encoded inputs and prevents the pool from
    accumulating candidates the GP would have to truncate at fit time.
    """
    for raw in analogues:
        if not canonicalize_pool:
            text = str(raw or "").strip()
            if not text:
                continue
            if smiles_max_len is not None and len(text) > smiles_max_len:
                continue
            if text in seen or text in pool:
                continue
            pool.add(text)
            seen.add(text)
            continue
        canon = _canonicalize_smiles(raw)
        if canon is None:
            LOGGER.debug("dropping invalid analogue: %r", raw)
            continue
        if smiles_max_len is not None and len(canon) > smiles_max_len:
            LOGGER.debug(
                "dropping over-length canonical analogue (%d > %d): %r",
                len(canon), smiles_max_len, canon,
            )
            continue
        if canon in seen or canon in pool:
            continue
        pool.add(canon)
        seen.add(canon)


def _as_minimize_tuple(
    minimize: Union[bool, Sequence[bool]], n_obj: int
) -> Tuple[bool, ...]:
    """Normalise ``minimize`` to a tuple of booleans of length ``n_obj``.

    A bare ``bool`` is broadcast to a length-``n_obj`` tuple of that
    value. A sequence is converted to a tuple after a length check.
    """
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BayesianAnalogSearchConfig:
    """Configuration for one :func:`bayesian_analog_search` call.

    Defaults are conservative: 10 init evaluations, 1 BO candidate per
    round, 10 BO rounds -> ~20 evaluations total.

    Attributes:
        init_size: Number of SMILES randomly sampled from the pool for
            the initialization batch.
        batch_size: Number of top-acquisition candidates per BO round.
        n_iterations: Maximum number of BO rounds (not evaluations).
        warmup: If True, expand the pool via a single batched call to
            ``analog_fn`` until it has at least ``init_size`` members
            (or the pool stops growing) before the initialization phase.
            No per-iteration safety cap — the warm-up is one call.
        acquisition: One of ``"ei"``, ``"ucb"``, ``"pi"`` or a sequence
            of those names for :class:`AcquisitionEvaluator`. The BO loop
            consumes one acquisition name; multi-objective loops use EHVI
            (n_obj=2) or Chebyshev ParEGO (n_obj>=3) regardless of this field.
        xi: Improvement threshold for EI / PI. Default 0.01.
        kappa: Exploration weight for UCB. Default 2.0.
        minimize: ``True`` (default) treats smaller scores as better
            (Vina convention). For multi-objective, pass a tuple of
            booleans matching the scorer tuple (e.g.
            ``(True, False)`` for ``vina+nn``). A bare ``bool`` is
            broadcast to all objectives.
        canonicalize_pool: If True, validate analogues via RDKit and use
            canonical SMILES for dedup. Invalid SMILES are dropped.
        acq_budget: Optional[int = None
            Maximum number of pool SMILES fed to the GP + acquisition
            step. When ``len(pool) > acq_budget``, a uniform random
            subsample of size ``acq_budget`` is taken via the seeded
            ``rng``; the top-``batch_size`` candidates come from this
            subsample. ``None`` (default) keeps the full pool.
        max_pool_size: Optional[int = None
            FIFO cap on the candidate pool. When set, the pool is a
            bounded :class:`strbo_v1.utils.FIFOSet` and the oldest
            entry is auto-evicted on append. ``None`` (default) is
            unbounded.
        smiles_max_len: int = 50
            SMILES length cap. Analogue SMILES whose canonical form
            (or stripped raw text, when ``canonicalize_pool=False``)
            is longer than ``smiles_max_len`` are dropped at the
            pool-ingestion step. The same value is used by the GP
            string kernel to truncate int64-encoded inputs. ``None``
            disables the filter. Default ``50``.
        gp_config: Configuration passed to ``GPSurrogate`` for each BO
            round (a fresh instance is created per round and per
            objective in the multi-objective case).
        ref_point: Optional[tuple of floats, length n_obj. Used only
            in the multi-objective path (n_obj=2) for EHVI / HV. For
            n_obj != 2 this field is ignored. ``None`` means "use
            the per-backend DEFAULT_REF registry"; pass an explicit
            tuple to override. CLI users typically leave this
            ``None`` and pass ``--ref-point`` instead.
        ehvi_n_samples: Number of Monte-Carlo samples for EHVI in the
            2-objective path. Default 128. Ignored for n_obj != 2.
        che_alpha: Concentration parameter for the Beta distribution
            used to sample simplex weights in the Chebyshev-ParEGO
            path (n_obj >= 3). ``alpha=1`` gives a uniform simplex
            distribution. Default 1.0. Ignored for n_obj < 3.
        verbose: If True, print one-line progress messages to stdout.
    """

    init_size: int = 10
    batch_size: int = 1
    n_iterations: int = 10
    warmup: bool = True
    acquisition: Union[str, Tuple[str, ...]] = "ei"
    xi: float = 0.01
    kappa: float = 2.0
    minimize: Union[bool, Tuple[bool, ...]] = True
    canonicalize_pool: bool = True
    acq_budget: Optional[int] = None
    max_pool_size: Optional[int] = None
    smiles_max_len: int = 50
    gp_config: GPConfig = field(default_factory=GPConfig)
    ref_point: Optional[Tuple[float, ...]] = None
    ehvi_n_samples: int = 128
    che_alpha: float = 1.0
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.acq_budget is not None and self.acq_budget < 1:
            raise ValueError(f"acq_budget must be >= 1, got {self.acq_budget}")
        if self.max_pool_size is not None and self.max_pool_size < 1:
            raise ValueError(f"max_pool_size must be >= 1, got {self.max_pool_size}")
        if self.smiles_max_len is not None and self.smiles_max_len < 1:
            raise ValueError(f"smiles_max_len must be >= 1, got {self.smiles_max_len}")
        if self.ehvi_n_samples < 1:
            raise ValueError(
                f"ehvi_n_samples must be >= 1, got {self.ehvi_n_samples}"
            )
        if self.che_alpha <= 0:
            raise ValueError(f"che_alpha must be > 0, got {self.che_alpha}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _return_history(
    history: "OrderedDict[str, Tuple[Optional[float], ...]]", n_obj: int
) -> List:
    """Unpack the internal tuple-of-floats history to the public shape.

    For ``n_obj == 1`` the per-entry score tuple has length 1; we unpack
    to a bare float to preserve the legacy single-obj API. For
    ``n_obj >= 2`` the tuple is returned as-is.
    """
    if n_obj == 1:
        return [(smi, scores[0]) for smi, scores in history.items()]
    return [(smi, scores) for smi, scores in history.items()]


def bayesian_analog_search(
    seed_smiles: Iterable[str],
    scorer: Scorers,
    analog_fn: Callable[[Sequence[str]], Sequence[str]],
    *,
    config: Optional[BayesianAnalogSearchConfig] = None,
    rng: Optional[Union[RNG, random.Random]] = None,
) -> list:
    """Iterative Bayesian analog search (single- or multi-objective).

    Args:
        seed_smiles: Initial SMILES to seed the pool.
        scorer: A :class:`Scorers` value — either a single
            :class:`Scorer` (any callable mapping
            ``Sequence[str] -> Sequence[float]``) or a tuple of
            scorers for the multi-objective case. The i-th output
            of each scorer must be the score of the i-th input
            SMILES; failed evaluations should be signalled with
            ``float("nan")`` (the loop treats non-finite floats as
            ``None`` and excludes them from the GP fit).
        analog_fn: A batched analog generator. Takes a sequence of
            seed SMILES and returns a flat sequence of analogue
            SMILES — a simple ``Iterable[str] -> Iterable[str]``
            contract. The loop never sees a DataFrame; any
            DataFrame handling (e.g., for ReaSyn's
            ``generate_analogs`` output) lives at the runner
            boundary. Each phase (warm-up, init, BO round) invokes
            this callable **once** with the full target list.
        config: Loop configuration. Defaults to a fresh
            ``BayesianAnalogSearchConfig``.
        rng: An :class:`RNG` (preferred), :class:`random.Random`,
            or ``None``. Auto-promoted to :class:`RNG` for
            multi-objective MC sampling.

    Returns:
        A list of ``(smiles, score)`` tuples in evaluation order.
        * Single-objective: ``score`` is ``Optional[float]``
          (legacy shape).
        * Multi-objective: ``score`` is a tuple of
          ``Optional[float]`` of length ``n_obj`` (one per scorer).
        The list may be shorter than expected if the pool exhausts
        early or analog generation produces no new SMILES.

    Example:
        .. code-block:: python

            from strbo_v1 import (
                VinaScorer, VinaScorerConfig, BayesianAnalogSearchConfig,
                bayesian_analog_search,
            )
            from strbo_v1.analog import generate_analogs, ReasynConfig
            from strbo_v1.gp import GPConfig

            # Single-objective (legacy shape preserved)
            scorer = VinaScorer(VinaScorerConfig(vina_bin="../bin/vina", cache_dir=...))
            reasyn_config = ReasynConfig(...)
            def analog_fn(smis):
                df = generate_analogs(smis, reasyn_config)
                return df["smiles"].tolist() if df is not None and len(df) > 0 else []

            history = bayesian_analog_search(
                seed_smiles=["CCO", "CCN"],
                scorer=scorer,
                analog_fn=analog_fn,
                config=BayesianAnalogSearchConfig(
                    init_size=10, batch_size=1, n_iterations=10,
                    acquisition="ei", smiles_max_len=50,
                    gp_config=GPConfig(impl="fingerprint+tanimoto", device="cuda"),
                ),
                rng=random.Random(42),
            )

            # Multi-objective (vina+nn) using EHVI
            from strbo_v1 import NNScorer, NNScorerConfig
            nn_scorer = NNScorer(NNScorerConfig(model_path="activity_modeling/best_g12d_model.joblib"))
            history = bayesian_analog_search(
                seed_smiles=["CCO", "CCN"],
                scorer=(scorer, nn_scorer),     # 2-obj tuple
                analog_fn=analog_fn,
                config=BayesianAnalogSearchConfig(
                    init_size=10, batch_size=1, n_iterations=10,
                    minimize=(True, False),      # vina min, nn max
                    ref_point=(0.0, 5.0),       # overrides DEFAULT_REF
                    ehvi_n_samples=128,
                ),
                rng=random.Random(42),
            )
    """
    if config is None:
        config = BayesianAnalogSearchConfig()
    if rng is None:
        rng = RNG(seed=None)
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")
    if config.init_size < 1:
        raise ValueError(f"init_size must be >= 1, got {config.init_size}")
    if config.n_iterations < 0:
        raise ValueError(f"n_iterations must be >= 0, got {config.n_iterations}")

    scorers_tuple = as_scorer_tuple(scorer)
    n_obj = len(scorers_tuple)
    if n_obj == 0:
        raise ValueError("scorer tuple is empty; need at least 1 scorer")
    minimize_t = _as_minimize_tuple(config.minimize, n_obj)
    rng_obj = as_rng(rng)

    acquisition_name = single_acquisition_name(config.acquisition)
    acq_fn = _resolve_acquisition(acquisition_name)
    uses_best = acquisition_name in ("ei", "pi")

    # ------------------------------------------------------------------ init
    pool: FIFOSet = FIFOSet(max_size=config.max_pool_size)
    seen: set[str] = set()
    # Internal history is always tuple-of-floats, regardless of n_obj.
    history: "OrderedDict[str, Tuple[Optional[float], ...]]" = OrderedDict()

    _add_analogues_to_pool(
        seed_smiles, pool, seen,
        canonicalize_pool=config.canonicalize_pool,
        smiles_max_len=config.smiles_max_len,
    )
    LOGGER.info(
        "bayesian_analog_search: n_obj=%d, seeded pool with %d SMILES",
        n_obj, len(pool),
    )

    # --------------------------------------------------------------- warm-up
    if config.warmup and len(pool) < config.init_size and len(pool) > 0:
        n_needed = config.init_size - len(pool)
        targets = rng_obj.python.sample(list(pool), k=min(n_needed, len(pool)))
        if config.verbose:
            print(
                f"    [warm-up] single batch: expanding {len(targets)} SMILES "
                f"to grow pool from {len(pool)} toward {config.init_size}",
                flush=True,
            )
        warm_analogs = _safe_call_analog(analog_fn, targets)
        before = len(pool)
        _add_analogues_to_pool(
            warm_analogs, pool, seen,
            canonicalize_pool=config.canonicalize_pool,
            smiles_max_len=config.smiles_max_len,
        )
        LOGGER.info(
            "bayesian_analog_search: warm-up pool %d -> %d (target %d)",
            before, len(pool), config.init_size,
        )

    # ----------------------------------------------------- initialization
    k_eff = min(config.init_size, len(pool))
    if k_eff == 0:
        LOGGER.warning(
            "bayesian_analog_search: pool empty after warm-up; returning empty history"
        )
        return _return_history(history, n_obj)

    init_chosen = rng_obj.python.sample(list(pool), k=k_eff)
    for smi in init_chosen:
        pool.discard(smi)

    if config.verbose:
        print(f"    [init] scoring {len(init_chosen)} SMILES (n_obj={n_obj})...", flush=True)
    init_scores = _safe_score_n(scorers_tuple, init_chosen)
    for smi, score_tuple in zip(init_chosen, init_scores):
        history[smi] = score_tuple
        if config.verbose:
            print(f"    [init] scored {smi!r} = {score_tuple}", flush=True)

    if config.verbose:
        print(
            f"    [init] expanding {len(init_chosen)} SMILES in one batch call...",
            flush=True,
        )
    init_analogs = _safe_call_analog(analog_fn, init_chosen)
    _add_analogues_to_pool(
        init_analogs, pool, seen,
        canonicalize_pool=config.canonicalize_pool,
        smiles_max_len=config.smiles_max_len,
    )

    LOGGER.info(
        "bayesian_analog_search: initialization scored %d SMILES; pool size now %d",
        k_eff, len(pool),
    )

    # ---------------------------------------------------------- BO loop
    for it in range(config.n_iterations):
        if not pool:
            LOGGER.info("bayesian_analog_search: pool empty; stopping at iteration %d", it)
            break

        if config.verbose:
            print(
                f"    [BO round {it+1}/{config.n_iterations}] selecting from pool "
                f"(size {len(pool)}, n_obj={n_obj})...",
                flush=True,
            )
        candidates, _acq_values = _select_candidates(
            pool=pool,
            history=history,
            config=config,
            n_obj=n_obj,
            minimize_t=minimize_t,
            acq_fn=acq_fn,
            uses_best=uses_best,
            rng_obj=rng_obj,
        )
        if not candidates:
            LOGGER.info(
                "bayesian_analog_search: no candidates selected at iteration %d; stopping",
                it,
            )
            break

        if config.verbose:
            print(
                f"    [BO round {it+1}/{config.n_iterations}] scoring {len(candidates)} SMILES...",
                flush=True,
            )
        cand_scores = _safe_score_n(scorers_tuple, candidates)
        for smi, score_tuple in zip(candidates, cand_scores):
            history[smi] = score_tuple
            pool.discard(smi)
            if config.verbose:
                print(
                    f"    [BO round {it+1}/{config.n_iterations}] scored {smi!r} = {score_tuple}",
                    flush=True,
                )

        if config.verbose:
            print(
                f"    [BO round {it+1}/{config.n_iterations}] expanding "
                f"{len(candidates)} SMILES in one batch call...",
                flush=True,
            )
        cand_analogs = _safe_call_analog(analog_fn, candidates)
        _add_analogues_to_pool(
            cand_analogs, pool, seen,
            canonicalize_pool=config.canonicalize_pool,
            smiles_max_len=config.smiles_max_len,
        )

        LOGGER.info(
            "bayesian_analog_search: BO round %d evaluated %d SMILES; pool size now %d",
            it + 1, len(candidates), len(pool),
        )

    return _return_history(history, n_obj)


# ---------------------------------------------------------------------------
# Public advisor step
# ---------------------------------------------------------------------------


def select_candidates(
    pool: Sequence[str],
    history: Sequence[Tuple[str, Union[float, Tuple[Optional[float], ...]]]],
    config: BayesianAnalogSearchConfig,
    *,
    rng: Optional[Union[RNG, random.Random]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Pick the top-``config.batch_size`` candidates from ``pool`` given ``history``.

    Pure advisor step (one BO round). The caller manages the surrounding
    loop, the analog generator, and the black-box scorer; this function
    answers "given what we have already evaluated, which SMILES from
    this pool should we evaluate next?" with the same algorithm dispatch
    as the in-loop :func:`bayesian_analog_search` (single-obj EI/PI/UCB,
    2-obj EHVI, ``n_obj>=3`` Chebyshev ParEGO).

    Args:
        pool: Candidate SMILES to choose from. Already validated and
            length-filtered by the caller; this function does not
            modify the list. May be empty (returns ``([], []``)).
        history: ``(smiles, score)`` tuples in evaluation order. For
            ``n_obj==1`` the score is a float (``Optional[float]``);
            for ``n_obj>=2`` it is a tuple of ``Optional[float]`` of
            length ``n_obj``. ``None`` scores are filtered out
            before the GP fit (consistent with the in-loop behavior).
        config: Loop configuration. Only selection-relevant fields
            are consumed: ``batch_size``, ``minimize``, ``ref_point``,
            ``ehvi_n_samples``, ``che_alpha``, ``acq_budget``,
            ``gp_config``, ``acquisition``, ``xi``, ``kappa``.
        rng: Optional :class:`RNG` or :class:`random.Random`.
            ``None`` (default) builds a fresh non-deterministic
            :class:`RNG`.

    Returns:
        ``(picks, acq_values)`` where ``picks`` is a list of
        ``min(config.batch_size, len(pool))`` SMILES (top-k by
        acquisition) and ``acq_values`` is the corresponding 1D
        ``np.ndarray`` of acquisition values (length ``len(picks)``).
        For the random-uniform fallback path (history has < 2
        finite scores, or GP fit fails) ``acq_values`` is all zeros.
        Higher = better for n_obj in {1, 2}; *smaller = better* for
        n_obj >= 3 (Chebyshev scalarization).

    Raises:
        ValueError: If history entries are inconsistent (mixed
            ``score``/``scores`` fields) or a multi-objective entry
            has the wrong tuple length.
    """
    if config is None:
        raise ValueError("select_candidates requires a non-None config.")
    if config.batch_size is None or config.batch_size < 1:
        raise ValueError(
            f"select_candidates requires config.batch_size >= 1, "
            f"got {config.batch_size!r}"
        )

    n_obj = _infer_n_obj_from_history(history)
    minimize_t = _as_minimize_tuple(config.minimize, n_obj)
    rng_obj = as_rng(rng) if rng is not None else RNG(seed=None)

    pool_list = [str(s) for s in pool if s is not None and str(s).strip()]
    if not pool_list:
        return [], np.zeros((0,), dtype=float)
    k = min(config.batch_size, len(pool_list))

    internal_history: "OrderedDict[str, Tuple[Optional[float], ...]]" = OrderedDict()
    for smi, sc in history:
        if sc is None:
            internal_history[str(smi)] = (None,) * n_obj
            continue
        if isinstance(sc, (int, float)):
            if n_obj != 1:
                raise ValueError(
                    f"history entry {smi!r} has a bare float score but n_obj={n_obj}; "
                    f"expected a tuple of {n_obj} floats."
                )
            internal_history[str(smi)] = (float(sc) if np.isfinite(float(sc)) else None,)
        else:
            seq = tuple(sc)
            if len(seq) != n_obj:
                raise ValueError(
                    f"history entry {smi!r} has score tuple length "
                    f"{len(seq)}; expected {n_obj}."
                )
            internal_history[str(smi)] = tuple(
                None if v is None else (float(v) if np.isfinite(float(v)) else None)
                for v in seq
            )

    pool_set: FIFOSet = FIFOSet(max_size=config.max_pool_size)
    seen: set = set()
    for s in pool_list:
        if s in seen:
            continue
        if s in internal_history:
            continue
        pool_set.add(s)
        seen.add(s)

    acquisition_name = single_acquisition_name(config.acquisition)
    acq_fn = _resolve_acquisition(acquisition_name)
    uses_best = acquisition_name in ("ei", "pi")

    picks, acq_values = _select_candidates(
        pool=pool_set,
        history=internal_history,
        config=config,
        n_obj=n_obj,
        minimize_t=minimize_t,
        acq_fn=acq_fn,
        uses_best=uses_best,
        rng_obj=rng_obj,
    )
    return picks, acq_values


def _infer_n_obj_from_history(
    history: Sequence[Tuple[str, Union[float, Tuple[Optional[float], ...]]]],
) -> int:
    """Infer the number of objectives from the first history entry.

    A bare-float score implies ``n_obj == 1``; a tuple score's
    length determines ``n_obj`` for ``n_obj >= 2``. An empty
    history defaults to ``n_obj == 1``.

    Raises:
        ValueError: If a multi-objective entry has zero length.
    """
    if not history:
        return 1
    for _, sc in history:
        if sc is None:
            continue
        if isinstance(sc, (int, float)):
            return 1
        seq = tuple(sc)
        if len(seq) < 1:
            raise ValueError(
                "history entry has empty score tuple; cannot infer n_obj."
            )
        return len(seq)
    return 1


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _safe_call_analog(
    analog_fn: Callable[[Sequence[str]], Sequence[str]],
    smiles_list: list,
) -> list:
    """Call ``analog_fn`` once with exception handling.

    Returns the flat list of analog SMILES. On exception, returns an
    empty list (the pool stays unchanged for this phase). ``None`` and
    non-iterable returns are normalized to ``[]``.
    """
    if not smiles_list:
        return []
    try:
        result = analog_fn(list(smiles_list))
    except Exception as exc:
        LOGGER.warning("analog_fn raised: %s; recording no analogs", exc)
        return []
    if result is None:
        return []
    return list(result)


def _safe_score_single(
    scorer: Callable[[Sequence[str]], Sequence[float]],
    smiles_list: list,
) -> list:
    """Normalise one scorer's output to ``[Optional[float]]`` aligned to input."""
    try:
        raw = scorer(list(smiles_list))
    except Exception as exc:
        LOGGER.warning("scorer raised: %s; recording None for all", exc)
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
    """Score a batch through N scorers and stack into per-SMILES tuples.

    Returns a list of length ``len(smiles_list)`` where each entry is a
    tuple of length ``n_obj`` with ``Optional[float]``. The i-th tuple
    element is the i-th scorer's score for the i-th SMILES. Any
    scorer's exception or non-finite value becomes ``None`` for that
    SMILES (the GP fit then drops it).
    """
    n_obj = len(scorers)
    per_obj = [_safe_score_single(sc, smiles_list) for sc in scorers]
    # Transpose: list of per-SMILES tuples.
    return list(zip(*per_obj))


def _maybe_subsample_pool(
    pool: FIFOSet,
    acq_budget: Optional[int],
    rng_obj: RNG,
) -> list:
    """Return the full pool or a uniform-random subsample of size ``acq_budget``."""
    pool_list_full = list(pool)
    if acq_budget is not None and len(pool_list_full) > acq_budget:
        subsampled = rng_obj.python.sample(pool_list_full, k=acq_budget)
        return sorted(subsampled)
    return pool_list_full


def _collect_finite_history_single(
    history: "OrderedDict[str, Tuple[Optional[float], ...]]",
) -> list:
    """Return ``[(smi, float), ...]`` for entries with all-finite scores (n_obj=1)."""
    out = []
    for smi, scores in history.items():
        if all(s is not None for s in scores):
            out.append((smi, float(scores[0])))
    return out


def _collect_finite_history_n(
    history: "OrderedDict[str, Tuple[Optional[float], ...]]",
    n_obj: int,
) -> list:
    """Return ``[(smi, (s0, s1, ...)), ...]`` for entries with all-finite scores."""
    out = []
    for smi, scores in history.items():
        if len(scores) == n_obj and all(s is not None for s in scores):
            out.append((smi, tuple(float(s) for s in scores)))
    return out


def _fit_gps(
    smiles_arr: list,
    scores_per_obj: list,
    config: BayesianAnalogSearchConfig,
    rng_obj: RNG,
) -> list:
    """Fit one GPSurrogate per objective, sharing the ``gp_config``.

    Returns a list of length ``n_obj`` of fitted :class:`GPSurrogate`
    instances. The torch seed is re-set before each fit for
    reproducibility.
    """
    gps = []
    for sc in scores_per_obj:
        rng_obj.torch()
        gp = GPSurrogate(config.gp_config)
        gp.fit(smiles_arr, sc.tolist())
        gps.append(gp)
    return gps


def _select_candidates(
    *,
    pool: FIFOSet,
    history: "OrderedDict[str, Tuple[Optional[float], ...]]",
    config: BayesianAnalogSearchConfig,
    n_obj: int,
    minimize_t: Tuple[bool, ...],
    acq_fn: Callable,
    uses_best: bool,
    rng_obj: RNG,
) -> Tuple[List[str], np.ndarray]:
    """Dispatch to the right candidate-selection strategy for ``n_obj``.

    ``n_obj == 1``: classical single-obj acquisition (EI/PI/UCB) on
    the GP posterior mean / std. ``n_obj == 2``: EHVI via
    :func:`expected_hypervolume_improvement` (Monte Carlo). ``n_obj >= 3``:
    Chebyshev ParEGO scalarization.

    Returns:
        ``(picks, acq_values)`` where ``picks`` is the list of
        ``k`` chosen SMILES and ``acq_values`` is the corresponding
        1D array of acquisition values, length ``k`` (empty for
        the random-uniform fallback path). Higher = better,
        consistent with the in-loop semantics.
    """
    pool_list = _maybe_subsample_pool(pool, config.acq_budget, rng_obj)
    if not pool_list:
        return [], np.zeros((0,), dtype=float)
    k = min(config.batch_size, len(pool_list))

    finite_pairs = _collect_finite_history_n(history, n_obj)
    if len(finite_pairs) < 2:
        LOGGER.debug(
            "history has %d finite score(s); falling back to uniform-random pick",
            len(finite_pairs),
        )
        picks = rng_obj.python.sample(pool_list, k=k)
        return picks, np.zeros((k,), dtype=float)

    if n_obj == 1:
        return _select_candidates_single(
            pool_list=pool_list, history_pairs=finite_pairs,
            config=config, minimize_t=minimize_t, acq_fn=acq_fn,
            uses_best=uses_best, k=k, rng_obj=rng_obj,
        )
    if n_obj == 2:
        return _select_candidates_2obj(
            pool_list=pool_list, history_pairs=finite_pairs,
            config=config, minimize_t=minimize_t, k=k, rng_obj=rng_obj,
        )
    return _select_candidates_3plus(
        pool_list=pool_list, history_pairs=finite_pairs,
        config=config, minimize_t=minimize_t, k=k, rng_obj=rng_obj,
    )


def _select_candidates_single(
    *,
    pool_list: list,
    history_pairs: list,
    config: BayesianAnalogSearchConfig,
    minimize_t: Tuple[bool, ...],
    acq_fn: Callable,
    uses_best: bool,
    k: int,
    rng_obj: RNG,
) -> Tuple[list, np.ndarray]:
    """Single-obj EI/PI/UCB candidate selection on the GP posterior.

    Returns:
        ``(picks, acq_values)`` where ``picks`` is a list of
        ``k`` SMILES and ``acq_values`` is a 1D ``np.ndarray`` of
        the corresponding acquisition values (length ``k``).
    """
    smiles_arr = [s for s, _ in history_pairs]
    scores_arr = np.asarray([sc[0] for _, sc in history_pairs], dtype=float)
    try:
        rng_obj.torch()
        gp = GPSurrogate(config.gp_config)
        gp.fit(smiles_arr, scores_arr.tolist())
        mu, sigma = gp.predict(pool_list, return_tensor=False)
    except Exception as exc:
        warnings.warn(
            f"GPSurrogate fit/predict failed ({exc!r}); falling back to uniform-random pick",
            RuntimeWarning, stacklevel=2,
        )
        picks = rng_obj.python.sample(pool_list, k=k)
        return picks, np.zeros((k,), dtype=float)

    minimize = minimize_t[0]
    if uses_best:
        best = float(np.min(scores_arr)) if minimize else float(np.max(scores_arr))
        acq_values = acq_fn(mu, sigma, best, xi=config.xi, minimize=minimize)
    else:
        acq_values = acq_fn(mu, sigma, kappa=config.kappa, minimize=minimize)

    order = np.argsort(acq_values, kind="stable")[::-1]
    picks = [pool_list[int(i)] for i in order[:k]]
    return picks, np.asarray(acq_values, dtype=float)[order[:k]]


def _select_candidates_2obj(
    *,
    pool_list: list,
    history_pairs: list,
    config: BayesianAnalogSearchConfig,
    minimize_t: Tuple[bool, ...],
    k: int,
    rng_obj: RNG,
) -> Tuple[list, np.ndarray]:
    """2-objective EHVI candidate selection (Monte Carlo).

    Returns:
        ``(picks, ehvi_values)`` where ``picks`` is a list of ``k``
        SMILES and ``ehvi_values`` is a 1D ``np.ndarray`` of the
        corresponding EHVI values (length ``k``).
    """
    smiles_arr = [s for s, _ in history_pairs]
    scores_per_obj = [
        np.asarray([sc[i] for _, sc in history_pairs], dtype=float)
        for i in range(2)
    ]
    try:
        gps = _fit_gps(smiles_arr, scores_per_obj, config, rng_obj)
        mu_per_obj, sigma_per_obj = [], []
        for gp in gps:
            mu, sigma = gp.predict(pool_list, return_tensor=False)
            mu_per_obj.append(np.asarray(mu, dtype=float).ravel())
            sigma_per_obj.append(np.asarray(sigma, dtype=float).ravel())
    except Exception as exc:
        warnings.warn(
            f"GPSurrogate (2-obj EHVI) fit/predict failed ({exc!r}); "
            f"falling back to uniform-random pick",
            RuntimeWarning, stacklevel=2,
        )
        picks = rng_obj.python.sample(pool_list, k=k)
        return picks, np.zeros((k,), dtype=float)

    pareto = [sc for _, sc in history_pairs]  # current Pareto == full history (greedy)
    ref = config.ref_point if config.ref_point is not None else (0.0, 0.0)
    if len(ref) != 2:
        raise ValueError(
            f"ref_point must be length 2 for n_obj=2; got {len(ref)}"
        )
    try:
        ehvi = expected_hypervolume_improvement(
            mu_per_obj=mu_per_obj,
            sigma_per_obj=sigma_per_obj,
            pareto_points=pareto,
            ref=list(ref),
            minimize=tuple(minimize_t),
            n_samples=config.ehvi_n_samples,
            rng=rng_obj,
        )
    except Exception as exc:
        warnings.warn(
            f"EHVI computation failed ({exc!r}); falling back to uniform-random pick",
            RuntimeWarning, stacklevel=2,
        )
        picks = rng_obj.python.sample(pool_list, k=k)
        return picks, np.zeros((k,), dtype=float)

    ehvi_arr = np.asarray(ehvi, dtype=float)
    order = np.argsort(ehvi_arr, kind="stable")[::-1]
    picks = [pool_list[int(i)] for i in order[:k]]
    return picks, ehvi_arr[order[:k]]


def _select_candidates_3plus(
    *,
    pool_list: list,
    history_pairs: list,
    config: BayesianAnalogSearchConfig,
    minimize_t: Tuple[bool, ...],
    k: int,
    rng_obj: RNG,
) -> Tuple[list, np.ndarray]:
    """Chebyshev ParEGO candidate selection for ``n_obj >= 3``.

    For each BO round, one simplex weight vector is sampled via
    :func:`sample_simplex_weights` (Beta + normalise). Each pool
    member's predicted mean (one row of stacked ``mu``) is then
    scalarized via :func:`chebyshev_scalarize` against the per-
    objective ideal point, and the candidate with the smallest
    scalarized value wins. Exploration is on the simplex direction
    (each round a new ``lambda`` is sampled), not on the sigma axis.

    Returns:
        ``(picks, scalarize_values)`` where ``picks`` is a list of
        ``k`` SMILES and ``scalarize_values`` is a 1D ``np.ndarray``
        of the corresponding Chebyshev scalarized values (length
        ``k``; *smaller = better*, the opposite of the n_obj=1
        and n_obj=2 directions).
    """
    n_obj = len(minimize_t)
    smiles_arr = [s for s, _ in history_pairs]
    scores_per_obj = [
        np.asarray([sc[i] for _, sc in history_pairs], dtype=float)
        for i in range(n_obj)
    ]
    try:
        gps = _fit_gps(smiles_arr, scores_per_obj, config, rng_obj)
        mu_per_obj = []
        for gp in gps:
            mu, _ = gp.predict(pool_list, return_tensor=False)
            mu_per_obj.append(np.asarray(mu, dtype=float).ravel())
    except Exception as exc:
        warnings.warn(
            f"GPSurrogate (Chebyshev n>={n_obj}) fit/predict failed "
            f"({exc!r}); falling back to uniform-random pick",
            RuntimeWarning, stacklevel=2,
        )
        picks = rng_obj.python.sample(pool_list, k=k)
        return picks, np.zeros((k,), dtype=float)

    # Per-objective ideal point from history (min for minimize, max for maximize).
    ideal = np.asarray(
        [
            float(np.min(s)) if minimize_t[i] else float(np.max(s))
            for i, s in enumerate(scores_per_obj)
        ],
        dtype=float,
    )
    # Sample one simplex weight vector (size n_obj).
    weights = sample_simplex_weights(rng_obj, n=n_obj, alpha=config.che_alpha)
    n_cand = len(pool_list)
    scalarize_vals = np.empty(n_cand, dtype=float)
    for c in range(n_cand):
        point = np.asarray(
            [mu_per_obj[i][c] for i in range(n_obj)], dtype=float
        )
        scalarize_vals[c] = chebyshev_scalarize(
            point=point, weights=weights, ideal=ideal, minimize=minimize_t,
        )
    order = np.argsort(scalarize_vals, kind="stable")
    picks = [pool_list[int(i)] for i in order[:k]]
    return picks, scalarize_vals[order[:k]]


__all__ = [
    "AcquisitionEvaluator",
    "bayesian_analog_search",
    "BayesianAnalogSearchConfig",
    "select_candidates",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    print(
        "strbo_v1.bayesian_analog_search exposes BayesianAnalogSearchConfig + "
        "bayesian_analog_search. Import this module rather than executing it directly."
    )
    sys.exit(0)
