"""Bayesian analog search with LLM advisor (``bo-*-ldm`` methods).

This is the public entry point for the two methods
``bo-tanimoto-ldm`` and ``bo-strkernel-ldm``. It is a thin wrapper
around :func:`strbo_v1.llm_advisor.orchestrator.run_bo_with_llm` that
mirrors the shape of :func:`strbo_v1.bayesian_analog_search.bayesian_analog_search`:

* Same call signature: ``(seed_smiles, scorer, analog_fn, *, config, rng)``.
* Same return contract: ``list[(smiles, score_or_scores)]`` where
  ``score_or_scores`` is a float for ``n_obj == 1`` and a list of
  floats for ``n_obj >= 2``. ``None`` for failed evals.
* Same input contract for the analog generator: a flat
  ``Sequence[str] -> Sequence[str]`` (ReaSyn's ``generate_analogs``
  output is unwrapped to a list of SMILES at the call boundary).

The two ``bo-*-ldm`` methods differ only in their underlying GP
implementation (``fingerprint+tanimoto`` vs ``smiles-strkernel``),
which is set in :class:`BayesianLDMSearchConfig.bo_config`. The
LLM advisor, BO loop structure, and history contract are identical.

Multi-objective note
--------------------
The LDM orchestrator natively handles multi-objective: it stores
``list[float]`` (length n_obj) score tuples in the history and hands
them straight to :func:`strbo_v1.bayesian_analog_search.select_candidates`,
which dispatches EHVI (n_obj=2) / Chebyshev ParEGO (n_obj>=3) /
EI/PI/UCB (n_obj=1). There is **no single-obj collapse** in the
LDM: the multi-obj machinery in ``bayesian_analog_search.py`` is
reused verbatim.

Auxiliary return: the function also returns the LLM trajectory (a
dict mirroring the ``TrajectoryRecorder`` final-JSON shape) so
``run_search.py`` can merge it under a top-level ``"llm_trajectory"``
key in the run's main JSON.

Public surface:

* :class:`BayesianLDMSearchConfig`
* :func:`bayesian_ldm_search`
"""

from __future__ import annotations

import logging
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union,
)

from strbo_v1.bayesian_analog_search import BayesianAnalogSearchConfig
from strbo_v1.llm_advisor.client import OpenAIChatClient
from strbo_v1.llm_advisor.config import LLMClientConfig, load_env
from strbo_v1.llm_advisor.orchestrator import (
    OrchestratorConfig,
    run_bo_with_llm,
)
from strbo_v1.llm_advisor.reasyn_pool import ReasynConfigPool
from strbo_v1.llm_advisor.state import AnalogueRecord
from strbo_v1.rng import RNG
from strbo_v1.scorer import Scorers

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BayesianLDMSearchConfig:
    """Configuration for :func:`bayesian_ldm_search`.

    Attributes:
        init_size: Number of initial SMILES scored before the BO loop
            starts.
        batch_size: BO candidates per round.
        n_iterations: Number of BO rounds.
        smiles_max_len: SMILES length cap (same semantics as
            :class:`BayesianAnalogSearchConfig`).
        bo_config: The underlying :class:`BayesianAnalogSearchConfig`
            (GP impl, acquisition, multi-objective settings, etc.).
        llm_config: :class:`LLMClientConfig` (api_key, base_url, model).
        pool_max_size: Pool FIFO cap. None = unbounded.
        method: One of ``"bo-tanimoto-ldm"`` /
            ``"bo-strkernel-ldm"``. Echoed back into the JSON.
        seed: RNG seed.
        minimize: ``(True,)`` for single-objective minimize; for
            multi-objective, a tuple aligned with the scorers.
        objective_legend: Per-objective metadata for the LLM
            prompt (e.g. ``[{"name": "vina", "minimize": True}]``).
        trajectory_dir: Optional path. If set, the per-round
            trajectory JSON is written there in addition to being
            returned in-memory.
        verbose: If True, print one-line progress messages to stdout
            (mirrors the ``bayesian_analog_search.verbose=True``
            convention). The LDM does not collapse multi-obj
            scores for verbose output; it shows
            ``= [v0, v1, ...]`` for n_obj>=2 and ``= -7.2`` for
            n_obj=1.
        guidance: Free-form text appended to all three LLM system
            prompts (Stage A1 actions, A2 review-analogs, B
            review-suggestions). Use to steer the LLM's behaviour
            without changing code. ``""`` disables guidance.
    """

    init_size: int = 5
    batch_size: int = 1
    n_iterations: int = 5
    smiles_max_len: int = 50
    bo_config: Optional[BayesianAnalogSearchConfig] = None
    llm_config: Optional[LLMClientConfig] = None
    pool_max_size: Optional[int] = None
    pool_min_size: int = 1
    method: str = "bo-tanimoto-ldm"
    seed: int = 0
    minimize: Tuple[bool, ...] = (True,)
    objective_legend: List[Dict[str, Any]] = field(default_factory=list)
    trajectory_dir: Optional[str] = None
    verbose: bool = False
    guidance: str = ""

    def __post_init__(self) -> None:
        if self.bo_config is None:
            raise ValueError("bo_config is required")
        if self.llm_config is None:
            raise ValueError("llm_config is required")
        if not self.minimize:
            raise ValueError("minimize must be non-empty")
        if self.init_size < 1:
            raise ValueError(f"init_size must be >= 1, got {self.init_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.n_iterations < 0:
            raise ValueError(
                f"n_iterations must be >= 0, got {self.n_iterations}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_rng(
    rng: Optional[Union[RNG, random.Random]],
) -> Union[RNG, random.Random]:
    if rng is None:
        return RNG(seed=None)
    return rng


def _to_analogue_records(
    raw: Sequence[str],
    seed: str,
    generator_hint: str = "default",
) -> List[AnalogueRecord]:
    """Wrap a flat list of analogue SMILES into :class:`AnalogueRecord`."""
    return [
        AnalogueRecord(
            analogue_smiles=s,
            seed_smiles=seed,
        )
        for s in raw
        if s
    ]


def _build_analog_wrapper(
    analog_fn: Callable[[Sequence[str]], Sequence[str]],
) -> Callable[[Sequence[str]], Sequence[AnalogueRecord]]:
    """Wrap a flat analog_fn into one that returns AnalogueRecord lists.

    The orchestrator's analog-fn contract is
    ``Sequence[str] -> Sequence[AnalogueRecord]``; the rest of the
    project uses ``Sequence[str] -> Sequence[str]``. This wrapper
    bridges the two.
    """
    def wrapped(seeds: Sequence[str]) -> Sequence[AnalogueRecord]:
        out: List[AnalogueRecord] = []
        try:
            analogues = list(analog_fn(list(seeds)))
        except Exception as exc:                                # pragma: no cover
            LOGGER.warning("analog_fn raised: %s; skipping", exc)
            return out
        for s in seeds:
            out.extend(_to_analogue_records(analogues, seed=s))
        return out

    return wrapped


def _build_ldm_reasyn_pool(bo_cfg: BayesianAnalogSearchConfig) -> ReasynConfigPool:
    """Build a :class:`ReasynConfigPool` from the LDM config.

    For now we use the default pool. Callers that need custom ReaSyn
    presets can pass a pre-built pool through other channels
    (out of scope for the public entry point).
    """
    return ReasynConfigPool.from_env()


def _build_orchestrator_config(
    config: BayesianLDMSearchConfig, n_obj: int,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        init_size=config.init_size,
        batch_size=config.batch_size,
        n_iterations=config.n_iterations,
        smiles_max_len=config.smiles_max_len,
        bo_config=config.bo_config,
        method=config.method,
        seed=config.seed,
        objective_legend=list(config.objective_legend),
        minimize=config.minimize,
        pool_max_size=config.pool_max_size,
        pool_min_size=config.pool_min_size,
        n_obj=n_obj,
        verbose=bool(config.verbose),
        guidance=config.guidance,
    )


def _scorer_to_callable(
    scorer: Scorers,
) -> Tuple[Callable[[Sequence[str]], Sequence[Any]], int]:
    """Adapt a :class:`Scorers` (single or tuple) to the LDM
    orchestrator's contract.

    Returns ``(callable, n_obj)`` where:

    * ``n_obj == 1``: the callable returns a list of floats (one per
      SMILES). The orchestrator stores each as a bare float in
      the history (``history[smi] = 0.5``).
    * ``n_obj >= 2``: the callable returns a list of
      ``list[float]`` of length ``n_obj`` (one list per SMILES,
      ordered the same as the scorers). The orchestrator stores
      each as ``list[float]`` in the history and hands the
      multi-obj history straight to
      :func:`strbo_v1.bayesian_analog_search.select_candidates`,
      which dispatches EHVI (n_obj=2), Chebyshev ParEGO
      (n_obj>=3), or EI/PI/UCB (n_obj=1) by ``n_obj`` inferred
      from the history.

    No single-obj collapse occurs here — the multi-obj scores
    are forwarded as-is.
    """
    n_obj = len(scorer) if isinstance(scorer, tuple) else 1
    if n_obj == 1:
        s = scorer if not isinstance(scorer, tuple) else scorer[0]

        def call_single(smis: Sequence[str]) -> Sequence[float]:
            return list(s(list(smis)))
        return call_single, n_obj

    def call_multi(smis: Sequence[str]) -> Sequence[List[float]]:
        # Each inner scorer returns a list of per-SMILES scores. We
        # stack them into a list of ``list[float]`` of length n_obj.
        results: List[List[Any]] = [
            list(s(list(smis))) for s in scorer  # type: ignore[union-attr]
        ]
        out: List[List[float]] = []
        for i in range(len(smis)):
            row: List[float] = []
            for r in results:
                v = r[i]
                if v is None or (isinstance(v, float) and v != v):
                    # Scorer failed for this SMILES — propagate NaN;
                    # the orchestrator's _coerce_score_value will
                    # convert it to None and drop it from the GP fit.
                    row.append(float("nan"))
                else:
                    try:
                        row.append(float(v))
                    except (TypeError, ValueError):
                        row.append(float("nan"))
            out.append(row)
        return out

    return call_multi, n_obj


def _history_to_canonical(
    history: "OrderedDict[str, Any]",
    n_obj: int,
) -> List[Tuple[str, Any]]:
    """Convert an LDM-orchestrator history into the
    ``bayesian_analog_search`` return shape.

    The orchestrator's history values are already in the public
    shape (``float`` for n_obj==1, ``list[float]`` for n_obj>=2,
    ``None`` for failed evals). This function just drops
    NaN-coerced failed-eval floats back to ``None`` for the
    public API consistency.
    """
    out: List[Tuple[str, Any]] = []
    for smi, sc in history.items():
        if n_obj == 1:
            if sc is None:
                score: Any = None
            elif isinstance(sc, (list, tuple)):
                # The orchestrator stores multi-obj as a list. For
                # the public single-obj return, take the first
                # element (or None if the list contains None).
                score = float(sc[0]) if sc and sc[0] is not None else None
            else:
                f = float(sc)
                score = f if (f == f) else None  # NaN → None
        else:
            if sc is None or not isinstance(sc, (list, tuple)) or len(sc) != n_obj:
                score = tuple(None for _ in range(n_obj))
            else:
                score = tuple(
                    None if (v is None or (isinstance(v, float) and v != v))
                    else float(v)
                    for v in sc
                )
        out.append((smi, score))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def bayesian_ldm_search(
    seed_smiles: Iterable[str],
    scorer: Scorers,
    analog_fn: Callable[[Sequence[str]], Sequence[str]],
    *,
    config: BayesianLDMSearchConfig,
    rng: Optional[Union[RNG, random.Random]] = None,
    llm: Optional[Any] = None,
) -> Tuple[List[Tuple[str, Any]], Optional[Dict[str, Any]]]:
    """Run one BO trajectory with LLM advisor.

    Args:
        seed_smiles: Initial SMILES to seed the pool.
        scorer: A :class:`Scorers` value (single or tuple of
            per-objective scorers).
        analog_fn: A flat ``Sequence[str] -> Sequence[str]``
            analog generator. The wrapper unwraps ReaSyn's
            DataFrame-like output.
        config: :class:`BayesianLDMSearchConfig`.
        rng: An :class:`RNG` (preferred), :class:`random.Random`,
            or ``None``.
        llm: Optional pre-built LLM client (e.g. a
            :class:`MockLLMClient` for tests). If ``None`` (the
            default for production), an :class:`OpenAIChatClient`
            is constructed from ``config.llm_config``.

    Returns:
        ``(history, trajectory_dict)`` where:

        * ``history`` is a list of ``(smiles, score)`` tuples in
          evaluation order. For ``n_obj == 1`` ``score`` is a float
          (or ``None`` for failed evals). For ``n_obj >= 2`` it's a
          list of floats (or ``None``). This shape matches
          :func:`bayesian_analog_search`.
        * ``trajectory_dict`` is the LLM advisor's per-round
          trajectory (see ``strbo_v1.llm_advisor.trajectory``),
          or ``None`` if ``config.trajectory_dir`` was not set and
          the orchestrator did not record one. ``run_search.py``
          merges this dict into the main JSON under a top-level
          ``"llm_trajectory"`` key.

    Raises:
        ValueError: On invalid config (missing ``bo_config`` /
            ``llm_config``, non-positive batch / init sizes, etc.).
        Exception: Any non-recoverable error from the underlying
            scorer or GP fit propagates after the orchestrator
            writes its emergency sidecar (if recording).
    """
    # ------------------------------------------------------------------
    # Build runtime objects
    # ------------------------------------------------------------------
    load_env()                                                # dotenv → os.environ
    rng = _ensure_rng(rng)

    # Native multi-obj: pass the scorer tuple straight to the
    # orchestrator. ``_scorer_to_callable`` returns the orchestrator-
    # compatible callable AND the n_obj count. The orchestrator then
    # hands the multi-obj history to ``select_candidates`` which
    # dispatches by n_obj (EHVI / Chebyshev / EI/PI/UCB).
    internal_scorer, n_obj = _scorer_to_callable(scorer)
    internal_analog_fn = _build_analog_wrapper(analog_fn)
    internal_pool = _build_ldm_reasyn_pool(config.bo_config)  # type: ignore[arg-type]
    internal_orch_cfg = _build_orchestrator_config(config, n_obj=n_obj)
    llm_client = llm if llm is not None else OpenAIChatClient(config.llm_config)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    raw_history: List[Tuple[str, Any]] = run_bo_with_llm(
        seed_smiles=list(seed_smiles),
        scorer=internal_scorer,
        llm=llm_client,
        analog_fn=internal_analog_fn,
        reasyn_pool=internal_pool,
        config=internal_orch_cfg,
        trajectory_path=config.trajectory_dir,
    )

    # ------------------------------------------------------------------
    # Translate history shape
    # ------------------------------------------------------------------
    ordered: "OrderedDict[str, Any]" = OrderedDict(raw_history)
    history = _history_to_canonical(ordered, n_obj=n_obj)

    # ------------------------------------------------------------------
    # Read trajectory back (if recorded)
    # ------------------------------------------------------------------
    trajectory: Optional[Dict[str, Any]] = None
    if config.trajectory_dir is not None:
        from strbo_v1.llm_advisor.trajectory import resolve_trajectory_path
        import json as _json
        path = resolve_trajectory_path(
            config.trajectory_dir, method=config.method, seed=config.seed,
        )
        if path.exists():
            try:
                trajectory = _json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:                            # pragma: no cover
                LOGGER.warning("could not read trajectory %s: %s", path, exc)

    return history, trajectory


__all__ = [
    "BayesianLDMSearchConfig",
    "bayesian_ldm_search",
]
