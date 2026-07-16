"""BO orchestrator with three-stage LLM intervention.

:func:`run_bo_with_llm` is the public entry point. It mirrors the
state machine from the design doc:

    per BO round:
        Stage A1 — actions:
            LLM decides how to change the pool (propose / reject /
            analog / noop).  If an ``analog`` block produces non-empty
            results, call ``analog_fn`` and proceed to Stage A2.
        Stage A2 — review analogs (conditional):
            LLM reviews the generated analogues and decides keep /
            reject for each.  Kept analogues are added to the pool
            (with dedup).
        Pool-size loop:
            If ``len(pool) < pool_min_size``, repeat Stage A1 (+A2)
            with explicit feedback until the pool is large enough or
            ``max_pool_size_iters`` is reached.
        BO step: GP.fit + acquisition -> top-k
        Stage B — review suggestions:
            LLM reviews BO picks and may override or skip.
        Score final_candidates, remove from pool, update history.

The orchestrator never raises on LLM failures (LLMAdvisor handles
retries and fallback). It does propagate the underlying scorer's
exceptions, which the Exception Catcher in :func:`run_bo_with_llm`
catches and turns into a final ``status="fatal_error"`` trajectory
JSON.

Multi-objective note
--------------------
``n_obj`` is the number of objectives, derived from ``len(scorer)``
(``1`` for a single :class:`Scorer`; ``>= 2`` for a tuple). The
internal history is :data:`ScoreValue` (``float`` for n_obj=1,
``list[float]`` for n_obj>=2). The LDM orchestrator hands the same
multi-obj history to ``select_candidates`` from
:mod:`strbo_v1.bayesian_analog_search`, which dispatches
``n_obj==1`` EI/PI/UCB, ``n_obj==2`` EHVI, ``n_obj>=3`` Chebyshev
ParEGO — i.e. the LDM reuses the production multi-obj machinery
verbatim, with no single-obj collapse.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from strbo_v1.llm_advisor.advisor import LLMAdvisor, LLMAttemptRecord
from strbo_v1.llm_advisor.blocks import (
    LLMBlock,
    ReviewBOBlock,
    ReviewAnalogsBlock,
)
from strbo_v1.llm_advisor.parser import SemanticError, format_error_for_prompt
from strbo_v1.llm_advisor.reasyn_pool import ReasynConfigPool, pick_reasyn_config
from strbo_v1.llm_advisor.round_state import (
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
)
from strbo_v1.llm_advisor.state import (
    AnalogueRecord,
    GPSummary,
    PickRecord,
    ScoreValue,
)
from strbo_v1.llm_advisor.trajectory import (
    RoundRecord,
    TrajectoryRecorder,
    resolve_trajectory_path,
    serialize_analogues,
    serialize_attempts,
    serialize_blocks,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: score hygiene
# ---------------------------------------------------------------------------


def _is_finite_float(x: Any) -> bool:
    """True iff ``x`` is a non-None, finite float."""
    if x is None:
        return False
    if not isinstance(x, (int, float)):
        return False
    f = float(x)
    return f == f and abs(f) != float("inf")


def _coerce_score_value(v: Any) -> Optional[ScoreValue]:
    """Sanitize a single raw scorer output into :data:`ScoreValue`.

    * None / non-numeric → ``None``
    * NaN / inf → ``None``
    * finite int/float → ``float`` (n_obj=1) or stays as-is
    * list / tuple of finite numbers → ``list[float]`` (n_obj>=2)
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        cleaned: List[Optional[float]] = []
        for x in v:
            if x is None:
                cleaned.append(None)
                continue
            try:
                f = float(x)
            except (TypeError, ValueError):
                cleaned.append(None)
                continue
            cleaned.append(f if (f == f and abs(f) != float("inf")) else None)
        return list(cleaned)  # type: ignore[return-value]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if (f == f and abs(f) != float("inf")) else None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers: best anchor
# ---------------------------------------------------------------------------


def compute_best(
    history: Dict[str, ScoreValue],
    minimize: Sequence[bool],
    n_obj: int,
) -> Union[str, List[str]]:
    """Compute the "best" anchor for the LLM.

    * ``n_obj == 1``: single best SMILES (``str``); empty string if no
      finite history.
    * ``n_obj >= 2``: Pareto front — list of non-dominated SMILES; empty
      list if no history. The list may be long; callers / prompt
      renderers should truncate for display.
    """
    # Collect finite items with scores normalized to list[float] (n_obj=1 → len 1).
    items: List[Tuple[str, Tuple[float, ...]]] = []
    for s, sc in history.items():
        if n_obj == 1:
            if not _is_finite_float(sc):
                continue
            items.append((s, (float(sc),)))
        else:
            if not isinstance(sc, (list, tuple)) or len(sc) != n_obj:
                continue
            if not all(_is_finite_float(x) for x in sc):
                continue
            items.append((s, tuple(float(x) for x in sc)))
    if not items:
        return "" if n_obj == 1 else []

    if n_obj == 1:
        is_min = bool(minimize[0])
        items.sort(key=lambda x: x[1][0], reverse=not is_min)
        return items[0][0]

    # Pareto front
    front: List[str] = []
    for i, (s_i, sc_i) in enumerate(items):
        dominated = False
        for j, (s_j, sc_j) in enumerate(items):
            if i == j:
                continue
            q_no_worse = True
            q_strictly_better = False
            for pk, qk, is_min in zip(sc_i, sc_j, minimize):
                if is_min:
                    if qk > pk:
                        q_no_worse = False
                        break
                    if qk < pk:
                        q_strictly_better = True
                else:
                    if qk < pk:
                        q_no_worse = False
                        break
                    if qk > pk:
                        q_strictly_better = True
            if q_no_worse and q_strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(s_i)
    return front


def _any_obj_improved(
    new_history: Dict[str, ScoreValue],
    last_best_per_obj: List[Optional[float]],
    minimize: Sequence[bool],
    n_obj: int,
) -> bool:
    """Return True iff at least one objective has a new best score.

    If ``last_best_per_obj`` has any None entry (e.g. first round), we
    treat the round as "improved" if any objective now has a finite
    score.
    """
    if n_obj == 1:
        is_min = bool(minimize[0])
        cur_best: Optional[float] = None
        for sc in new_history.values():
            if _is_finite_float(sc):
                f = float(sc)
                if cur_best is None:
                    cur_best = f
                elif is_min:
                    cur_best = min(cur_best, f)
                else:
                    cur_best = max(cur_best, f)
        if cur_best is None:
            return False
        prev = last_best_per_obj[0] if last_best_per_obj else None
        if prev is None:
            return True
        if is_min:
            return cur_best < prev
        return cur_best > prev

    # n_obj >= 2
    cur_best_per_obj: List[Optional[float]] = [None] * n_obj
    for sc in new_history.values():
        if not isinstance(sc, (list, tuple)) or len(sc) != n_obj:
            continue
        for i in range(n_obj):
            v = sc[i]
            if not _is_finite_float(v):
                continue
            f = float(v)
            cur = cur_best_per_obj[i]
            if cur is None:
                cur_best_per_obj[i] = f
            elif minimize[i]:
                cur_best_per_obj[i] = min(cur, f)
            else:
                cur_best_per_obj[i] = max(cur, f)
    for i, (cur, prev) in enumerate(zip(cur_best_per_obj, last_best_per_obj)):
        if cur is None:
            continue
        if prev is None:
            return True
        if minimize[i]:
            if cur < prev:
                return True
        else:
            if cur > prev:
                return True
    return False


def _cur_best_per_obj(
    new_history: Dict[str, ScoreValue],
    minimize: Sequence[bool],
    n_obj: int,
) -> List[Optional[float]]:
    """Compute current per-objective best (used to update ``last_best_per_obj``)."""
    if n_obj == 1:
        is_min = bool(minimize[0])
        cur: Optional[float] = None
        for sc in new_history.values():
            if not _is_finite_float(sc):
                continue
            f = float(sc)
            if cur is None:
                cur = f
            elif is_min:
                cur = min(cur, f)
            else:
                cur = max(cur, f)
        return [cur]
    out: List[Optional[float]] = [None] * n_obj
    for sc in new_history.values():
        if not isinstance(sc, (list, tuple)) or len(sc) != n_obj:
            continue
        for i in range(n_obj):
            v = sc[i]
            if not _is_finite_float(v):
                continue
            f = float(v)
            cur_i = out[i]
            if cur_i is None:
                out[i] = f
            elif minimize[i]:
                out[i] = min(cur_i, f)
            else:
                out[i] = max(cur_i, f)
    return out


# ---------------------------------------------------------------------------
# BO step (call into existing strbo_v1; native multi-obj)
# ---------------------------------------------------------------------------


def _history_to_select_candidates_form(
    history: Dict[str, ScoreValue], n_obj: int,
) -> List[Tuple[str, Tuple[Optional[float], ...]]]:
    """Convert the LDM's ``float | list[float]`` history into the
    ``(smiles, (s1, s2, ...))`` tuple form expected by
    :func:`strbo_v1.bayesian_analog_search.select_candidates`.

    Skips entries with any NaN/None (the GP fit drops them too).
    """
    out: List[Tuple[str, Tuple[Optional[float], ...]]] = []
    for s, sc in history.items():
        if n_obj == 1:
            if not _is_finite_float(sc):
                continue
            out.append((s, (float(sc),)))
        else:
            if not isinstance(sc, (list, tuple)) or len(sc) != n_obj:
                continue
            if not all(_is_finite_float(x) for x in sc):
                continue
            out.append((s, tuple(float(x) for x in sc)))
    return out


def _run_bo_step(
    pool: Sequence[str],
    history: Dict[str, ScoreValue],
    *,
    bo_config: Any,
    rng: Any,
    top_k: int,
    n_obj: int,
) -> Tuple[List[PickRecord], GPSummary]:
    """Run GP + acquisition on ``pool``/``history`` and return PickRecords.

    The orchestrator passes the multi-obj history to
    :func:`strbo_v1.bayesian_analog_search.select_candidates`, which
    dispatches EHVI (n_obj=2), Chebyshev ParEGO (n_obj>=3), or EI/PI/UCB
    (n_obj=1) by ``n_obj`` inferred from the history.

    When the history is empty (first round, or all evals failed),
    ``select_candidates`` defaults to ``n_obj=1`` — which conflicts
    with our multi-obj ``bo_config.minimize``. In that case we draw
    random picks directly and bypass the length check.
    """
    from strbo_v1.bayesian_analog_search import select_candidates

    hist_for_sc = _history_to_select_candidates_form(history, n_obj)
    n_train = sum(1 for v in history.values() if _coerce_finite_history_v(v, n_obj))

    if not hist_for_sc:
        # Empty history → draw random picks (matches the uniform-random
        # fallback in ``select_candidates`` when history has < 2 finite
        # scores). We bypass the minimize length check by not calling
        # select_candidates at all.
        rng_attr = getattr(rng, "python", rng)
        try:
            sample_k = min(top_k, len(pool))
            picks = rng_attr.sample(list(pool), k=sample_k)
        except Exception:
            picks = list(pool)[: min(top_k, len(pool))]
        acq_values = [0.0] * len(picks)
    else:
        picks, acq_values = select_candidates(
            pool=list(pool),
            history=hist_for_sc,
            config=bo_config,
            rng=rng,
        )
        picks = picks[:top_k]
        acq_values = acq_values[: len(picks)]

    summary = GPSummary(n_train=n_train)

    if not picks:
        return [], summary

    mu_sigma_map = _query_gp_per_pick(picks, history, bo_config, rng, n_obj)
    records: List[PickRecord] = []
    for smi, acq in zip(picks, acq_values):
        mu, sigma = mu_sigma_map.get(smi, _zero_mu_sigma(n_obj))
        records.append(PickRecord(
            smiles=smi, acq_value=float(acq), mu=mu, sigma=sigma,
        ))
    return records, summary


def _coerce_finite_history_v(v: ScoreValue, n_obj: int) -> bool:
    """True iff ``v`` is a finite score usable for GP fit."""
    if n_obj == 1:
        return _is_finite_float(v)
    return (
        isinstance(v, (list, tuple))
        and len(v) == n_obj
        and all(_is_finite_float(x) for x in v)
    )


def _zero_mu_sigma(n_obj: int) -> Tuple[ScoreValue, ScoreValue]:
    """Default mu / sigma for the empty-pick / GP-failure fallback."""
    if n_obj == 1:
        return 0.0, 0.0
    return [0.0] * n_obj, [0.0] * n_obj


def _query_gp_per_pick(
    picks: List[str],
    history: Dict[str, ScoreValue],
    bo_config: Any,
    rng: Any,
    n_obj: int,
) -> Dict[str, Tuple[ScoreValue, ScoreValue]]:
    """Fit a quick GP and predict mu/sigma for each pick.

    n_obj == 1: one GP, scalar mu/sigma.
    n_obj >= 2: one GP per objective, list[float] mu/sigma.
    """
    from strbo_v1.gp import GPSurrogate

    if n_obj == 1:
        finite = [
            (s, float(history[s])) for s in history
            if _is_finite_float(history[s])
        ]
        if len(finite) < 2:
            return {p: (0.0, 0.0) for p in picks}
        try:
            gp = GPSurrogate(bo_config.gp_config)
            rng_attr = getattr(rng, "torch", None)
            if rng_attr is not None:
                rng_attr()
            gp.fit([s for s, _ in finite], [v for _, v in finite])
            mu, sigma = gp.predict(picks, return_tensor=False)
            return {
                p: (float(mu[i]), float(sigma[i]))
                for i, p in enumerate(picks)
            }
        except Exception as exc:
            LOGGER.debug("GP per-pick query failed: %s", exc)
            return {p: (0.0, 0.0) for p in picks}

    # n_obj >= 2
    smiles_for_fit: List[str] = []
    finite_per_obj: List[List[float]] = [[] for _ in range(n_obj)]
    for s, sc in history.items():
        if not isinstance(sc, (list, tuple)) or len(sc) != n_obj:
            continue
        if not all(_is_finite_float(x) for x in sc):
            continue
        smiles_for_fit.append(s)
        for i in range(n_obj):
            finite_per_obj[i].append(float(sc[i]))
    if len(smiles_for_fit) < 2:
        return {p: _zero_mu_sigma(n_obj) for p in picks}
    try:
        mu_per_obj: List[List[float]] = [[] for _ in range(n_obj)]
        sigma_per_obj: List[List[float]] = [[] for _ in range(n_obj)]
        for i in range(n_obj):
            gp = GPSurrogate(bo_config.gp_config)
            rng_attr = getattr(rng, "torch", None)
            if rng_attr is not None:
                rng_attr()
            gp.fit(smiles_for_fit, finite_per_obj[i])
            mu_i, sigma_i = gp.predict(picks, return_tensor=False)
            mu_per_obj[i] = list(np_as_ravel_float(mu_i))
            sigma_per_obj[i] = list(np_as_ravel_float(sigma_i))
        return {
            p: (
                [mu_per_obj[i][j] for i in range(n_obj)],
                [sigma_per_obj[i][j] for i in range(n_obj)],
            )
            for j, p in enumerate(picks)
        }
    except Exception as exc:
        LOGGER.debug("GP per-pick query failed: %s", exc)
        return {p: _zero_mu_sigma(n_obj) for p in picks}


def np_as_ravel_float(arr: Any) -> List[float]:
    """Robust ``np.asarray(..., dtype=float).ravel().tolist()``."""
    try:
        import numpy as np
        return list(np.asarray(arr, dtype=float).ravel())
    except Exception:
        try:
            return [float(x) for x in arr]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Configuration for :func:`run_bo_with_llm`.

    Mirrors the strbo_v1 BO loop's essential knobs; the underlying
    BayesianAnalogSearchConfig is wrapped to add LLM-specific fields.
    """

    # BO knobs (forwarded to strbo_v1)
    init_size: int = 5
    batch_size: int = 1
    n_iterations: int = 5
    max_completion_rounds: Optional[int] = None
    smiles_max_len: int = 50
    bo_config: Any = None                              # BayesianAnalogSearchConfig

    # LLM knobs
    max_pool_size_iters: int = 5                       # pool-size loop cap

    # Identification
    method: str = "llm-bo-tanimoto"
    seed: int = 0
    objective_legend: List[Dict[str, Any]] = field(default_factory=list)
    minimize: Sequence[bool] = (True,)
    pdf_context: str = ""
    pool_max_size: Optional[int] = None
    # Minimum pool size enforced in Stage A1.  When ``len(pool) <
    # pool_min_size``, a bare ``noop`` block is rejected and the LLM
    # is re-prompted to emit ``propose`` or ``analog`` to refill.
    # Default 1 = no enforcement.
    pool_min_size: int = 1
    # Number of objectives, derived from ``len(scorer)`` at runtime.
    # 1 for a single :class:`Scorer`; >= 2 for a tuple. The orchestrator
    # never infers n_obj from history — it always uses this explicit
    # value so the multi-obj / single-obj dispatch is unambiguous.
    n_obj: int = 1
    # When True, print one-line progress messages to stdout (mirrors
    # the ``bayesian_analog_search.verbose=True`` convention).
    verbose: bool = False
    # Free-form guidance text appended to all three LLM system prompts
    # (Stage A1 actions, A2 review-analogs, B review-suggestions).
    # Use to steer the LLM's behaviour without changing code
    # (e.g. remind it that the pool is the BO acquisition space, or
    # describe the vina / nn objectives). Empty string disables.
    guidance: str = ""

    def __post_init__(self) -> None:
        if not self.minimize:
            raise ValueError("minimize must be non-empty")
        if self.bo_config is None:
            raise ValueError("bo_config is required")
        if self.n_obj < 1:
            raise ValueError(f"n_obj must be >= 1, got {self.n_obj}")
        if len(self.minimize) != self.n_obj:
            raise ValueError(
                f"minimize length ({len(self.minimize)}) does not match "
                f"n_obj ({self.n_obj})"
            )
        if self.pool_min_size > 1 and self.pool_min_size < self.batch_size:
            raise ValueError(
                f"pool_min_size ({self.pool_min_size}) must be >= "
                f"batch_size ({self.batch_size})"
            )
        if (
            self.max_completion_rounds is not None
            and self.max_completion_rounds < self.n_iterations
        ):
            raise ValueError(
                f"max_completion_rounds ({self.max_completion_rounds}) must be "
                f">= n_iterations ({self.n_iterations})"
            )


def _score_via_scorer(
    scorer: Callable[[Sequence[str]], Sequence[Any]],
    smis: Sequence[str],
) -> Dict[str, Optional[ScoreValue]]:
    """Run ``scorer`` over ``smis`` and return ``{smi: score}``.

    ``score`` is ``float`` for n_obj=1 and ``list[float]`` (length
    n_obj) for n_obj>=2. ``None`` means the scorer failed for that
    SMILES (it will be dropped from the GP fit).
    """
    if not smis:
        return {}
    try:
        raw = list(scorer(list(smis)))
    except Exception as exc:
        LOGGER.warning(
            "scorer raised: %s: %s; all-None for batch",
            type(exc).__name__, exc,
        )
        return {s: None for s in smis}
    out: Dict[str, Optional[ScoreValue]] = {}
    for s, v in zip(smis, raw):
        out[s] = _coerce_score_value(v)
    return out


def _target_history_size(config: OrchestratorConfig) -> int:
    return config.init_size + config.n_iterations * config.batch_size


def _max_completion_rounds(config: OrchestratorConfig) -> int:
    if config.max_completion_rounds is not None:
        return config.max_completion_rounds
    return max(config.n_iterations, config.n_iterations * 3)


def run_bo_with_llm(
    *,
    seed_smiles: Sequence[str],
    scorer: Callable[[Sequence[str]], Sequence[Any]],
    llm: Any,                                            # LLMClient
    analog_fn: Optional[Callable[[Sequence[str]], Sequence[AnalogueRecord]]] = None,
    reasyn_pool: Optional[ReasynConfigPool] = None,
    config: OrchestratorConfig,
    trajectory_path: Optional[Any] = None,
) -> List[Tuple[str, Any]]:
    """Run a single BO trajectory with three-stage LLM intervention.

    Args:
        seed_smiles: initial pool SMILES.
        scorer: callable mapping a sequence of SMILES to a sequence
            of ``float`` (n_obj=1) or ``list[float]`` (n_obj>=2); or
            ``None`` for failed evals.
        llm: an :class:`LLMClient` (production or mock).
        analog_fn: optional callable mapping seed SMILES to a flat
            sequence of :class:`AnalogueRecord`.  If None, the LLM
            cannot produce analogues (its ``analog`` blocks are
            rejected at validation time).
        reasyn_pool: optional :class:`ReasynConfigPool` for
            ``pick_reasyn_config``.  Required if ``analog_fn`` is set.
        config: :class:`OrchestratorConfig` (``n_obj`` must be set
            explicitly; the runner derives it from ``len(scorer)``).
        trajectory_path: if not None, write a trajectory JSON to
            this path.

    Returns:
        ``(smiles, score)`` tuples in evaluation order. ``score`` is
        ``float`` for n_obj==1 and ``list[float]`` (length n_obj) for
        n_obj>=2. ``None`` for failed evals.
    """
    if analog_fn is not None and reasyn_pool is None:
        raise ValueError("reasyn_pool is required when analog_fn is set")

    # ------------------------------------------------------------------
    # Recorder setup
    # ------------------------------------------------------------------
    path: Optional[Any] = None
    recorder: Optional[TrajectoryRecorder] = None
    if trajectory_path is not None:
        path = resolve_trajectory_path(
            trajectory_path, method=config.method, seed=config.seed,
        )
        recorder = TrajectoryRecorder(path=path, method=config.method, seed=config.seed)
        recorder.begin_run(
            config=_config_to_dict(config),
            llm_model=getattr(llm, "model_name", "?"),
        )

    advisor = LLMAdvisor(
        llm=llm, max_retries=3, use_rdkit=True,
        guidance=config.guidance,
    )

    # ------------------------------------------------------------------
    # Init pool + history
    # ------------------------------------------------------------------
    pool: List[str] = list(dict.fromkeys(seed_smiles))  # dedup, preserve order
    history: "OrderedDict[str, ScoreValue]" = OrderedDict()
    stagnation_counter = 0
    last_best_per_obj: List[Optional[float]] = [None] * config.n_obj

    try:
        # ---- Initialization ----
        pool_min = getattr(config, "pool_min_size", 1) or 1
        if pool_min > 1:
            # LDM path: seed SMILES stay in the pool untouched.
            # Stage A1's pool-size loop expands pool to pool_min_size.
            pass
        else:
            # Standard BO: score init_size SMILES, move to history.
            init_chosen = list(pool)[: config.init_size]
            for smi in init_chosen:
                pool.remove(smi)
            init_scores = _score_via_scorer(scorer, init_chosen)
            for s in init_chosen:
                history[s] = init_scores[s]

        # ---- BO rounds ----
        target_history_size = _target_history_size(config)
        max_rounds = _max_completion_rounds(config)
        round_idx = 0
        while round_idx < max_rounds and len(history) < target_history_size:
            ctx = (
                recorder.round_context(round_idx) if recorder is not None
                else _NullRoundContext(round_idx)
            )
            with ctx as rr:
                _run_one_round(
                    round_idx=round_idx,
                    config=config,
                    pool=pool,
                    history=history,
                    stagnation_counter=stagnation_counter,
                    last_best_per_obj=list(last_best_per_obj),
                    scorer=scorer,
                    llm=llm,
                    advisor=advisor,
                    analog_fn=analog_fn,
                    reasyn_pool=reasyn_pool,
                    recorder=rr,
                )
            stagnation_counter, last_best_per_obj = _stagnation_and_best(
                history=history,
                minimize=config.minimize,
                n_obj=config.n_obj,
                stagnation_counter=stagnation_counter,
                last_best_per_obj=last_best_per_obj,
            )
            round_idx += 1

        # ---- Done ----
        if recorder is not None:
            recorder.set_status("completed")
            recorder.set_final_history(list(history.items()))
        return list(history.items())
    except Exception as exc:
        if recorder is not None:
            recorder.record_fatal_error(
                round_idx=recorder.current_round,
                exc=exc,
            )
            recorder.set_final_history(list(history.items()))
            recorder.dump_emergency_json()
        raise
    finally:
        if recorder is not None:
            recorder.write_final()


# ---------------------------------------------------------------------------
# Stagnation / best update
# ---------------------------------------------------------------------------


def _stagnation_and_best(
    *,
    history: "OrderedDict[str, ScoreValue]",
    minimize: Sequence[bool],
    n_obj: int,
    stagnation_counter: int,
    last_best_per_obj: List[Optional[float]],
) -> Tuple[int, List[Optional[float]]]:
    """Update stagnation counter and per-obj best.

    Reset on any-obj improvement. Returns (new_counter, new_last_best).
    """
    cur = _cur_best_per_obj(dict(history), minimize, n_obj)
    if _any_obj_improved(dict(history), last_best_per_obj, minimize, n_obj):
        return 0, cur
    return stagnation_counter + 1, last_best_per_obj


# ---------------------------------------------------------------------------
# Null context (for runs without a trajectory file)
# ---------------------------------------------------------------------------


class _NullRoundContext:
    """No-op context manager that mimics ``round_context``."""

    def __init__(self, round_idx: int) -> None:
        self.round_idx = round_idx
        self.round_record = RoundRecord(round_idx=round_idx, phase="bo", timestamp="")

    def __enter__(self) -> RoundRecord:
        return self.round_record

    def __exit__(self, *args) -> None:
        return None


# ---------------------------------------------------------------------------
# The single round — three-stage flow
# ---------------------------------------------------------------------------


def _run_one_round(
    *,
    round_idx: int,
    config: OrchestratorConfig,
    pool: List[str],
    history: "OrderedDict[str, ScoreValue]",
    stagnation_counter: int,
    last_best_per_obj: List[Optional[float]],
    scorer: Callable,
    llm: Any,
    advisor: LLMAdvisor,
    analog_fn: Optional[Callable],
    reasyn_pool: Optional[ReasynConfigPool],
    recorder: Any,                                       # RoundRecord or _NullRoundContext
) -> None:
    """Run one BO round (Stage A1+A2 + BO + Stage B + score)."""

    pool_min = getattr(config, "pool_min_size", 1) or 1
    all_attempts_A1: List[LLMAttemptRecord] = []
    all_attempts_A2: List[LLMAttemptRecord] = []
    final_blocks_A1: List[LLMBlock] = []
    final_blocks_A2: List[LLMBlock] = []
    fb_A1 = False
    fb_A2 = False
    n_obj = config.n_obj

    if config.verbose:
        print(
            f"    [round {round_idx+1}/{config.n_iterations}] Stage A1: deciding actions "
            f"(pool size {len(pool)}, n_obj={n_obj})...",
            flush=True,
        )

    # ---- Build Stage A1 snapshot ----
    snap_dict = _snapshot_actions(
        pool=pool,
        history=history,
        config=config,
        round_idx=round_idx,
        stagnation_counter=stagnation_counter,
    )
    action_state = _action_state_from_snapshot(snap_dict)
    recorder.pre_state_snapshot = snap_dict

    # ---- Pool-size loop: Stage A1 (+ optional A2) ----
    for _iter in range(config.max_pool_size_iters):
        # Stage A1: LLM decides actions
        blocks_A1, attempts_A1, fb_A1 = advisor.decide_actions(action_state)
        all_attempts_A1.extend(attempts_A1)
        final_blocks_A1 = blocks_A1

        # Apply actions: propose → pool, reject → pool.remove,
        # analog → call analog_fn → returns new analogues
        new_analogs = _apply_actions(
            blocks=blocks_A1,
            pool=pool,
            analog_fn=analog_fn,
            reasyn_pool=reasyn_pool,
        )

        # Stage A2: review analogues (synchronous, only if non-empty)
        if new_analogs:
            if config.verbose:
                print(
                    f"    [round {round_idx+1}/{config.n_iterations}] Stage A2: "
                    f"{len(new_analogs)} new analogues; LLM reviewing...",
                    flush=True,
                )
            review_state = _build_review_analogs_state(
                action_state=action_state,
                new_analogs=new_analogs,
                pool=pool,
                history=history,
                round_idx=round_idx,
                config=config,
            )
            blocks_A2, attempts_A2, fb_A2 = advisor.decide_review_analogs(review_state)
            all_attempts_A2.extend(attempts_A2)
            final_blocks_A2 = blocks_A2
            _apply_review_analogs(blocks=blocks_A2, pool=pool, new_analogs=new_analogs)

        # Check pool size — exit loop if sufficient
        if len(pool) >= pool_min or fb_A1:
            break

        if config.verbose:
            print(
                f"    [Stage A1 pool-size loop iter {_iter+1}/{config.max_pool_size_iters}] "
                f"pool has {len(pool)} < min {pool_min}; retrying with feedback",
                flush=True,
            )
        # Inject pool-size error for next iteration
        LOGGER.warning(
            "Pool-size loop iter %d: pool has %d SMILES < min %d; "
            "calling Stage A1 again with feedback.",
            _iter + 1, len(pool), pool_min,
        )
        pool_err = SemanticError(
            f"pool has {len(pool)} SMILES (< min {pool_min}); "
            f"you MUST emit `propose` with new SMILES or `analog` "
            f"to expand existing members. `noop` is rejected."
        )
        action_state = dataclasses.replace(
            action_state,
            pool=tuple(pool),
            previous_errors=tuple(
                list(action_state.previous_errors)
                + [format_error_for_prompt(pool_err)]
            ),
            attempt=1,
        )

    if config.verbose:
        types = sorted({b.type for b in final_blocks_A1})
        print(
            f"    [round {round_idx+1}/{config.n_iterations}] Stage A1 applied: "
            f"pool={len(pool)}, blocks={types}",
            flush=True,
        )
    recorder.llm_interactions["stage_a1"] = {
        "executed": True,
        "attempts": serialize_attempts(all_attempts_A1),
        "fallback_used": fb_A1,
        "final_blocks": serialize_blocks(final_blocks_A1),
        "pool_size_loop_final_pool_size": len(pool),
    }
    recorder.llm_interactions["stage_a2"] = {
        "executed": bool(all_attempts_A2),
        "attempts": serialize_attempts(all_attempts_A2),
        "fallback_used": fb_A2,
        "final_blocks": serialize_blocks(final_blocks_A2),
    }
    recorder.pool_after_phase_a = list(pool)

    # ---- BO step ----
    if config.verbose:
        print(
            f"    [round {round_idx+1}/{config.n_iterations}] BO step: selecting from "
            f"pool (size {len(pool)}, n_obj={n_obj})...",
            flush=True,
        )
    from strbo_v1.rng import RNG
    rng = RNG(seed=config.seed + round_idx)

    pick_records, summary = _run_bo_step(
        pool=pool,
        history=history,
        bo_config=config.bo_config,
        rng=rng,
        top_k=config.batch_size,
        n_obj=n_obj,
    )
    recorder.bo_suggestions = [p.to_dict() for p in pick_records]
    if config.verbose:
        print(
            f"    [round {round_idx+1}/{config.n_iterations}] BO step: selected "
            f"{len(pick_records)}: {[p.smiles for p in pick_records]}",
            flush=True,
        )

    # ---- Stage B: review suggestions ----
    if config.verbose:
        print(
            f"    [round {round_idx+1}/{config.n_iterations}] Stage B: reviewing "
            f"{len(pick_records)} suggestion(s)...",
            flush=True,
        )
    post_snap = _snapshot_suggestions(
        pool=pool,
        history=history,
        bo_picks=pick_records,
        acq_function=str(getattr(config.bo_config, "acquisition", "ei")),
        config=config,
        round_idx=round_idx,
        stagnation_counter=stagnation_counter,
    )
    post_state = _suggestion_state_from_snapshot(post_snap)

    blocks_B, attempts_B, fb_B = advisor.decide_review_suggestions(post_state)
    review_bo_block = _first_of_type(blocks_B, "review_bo")
    final_candidates, overrides = _apply_review_suggestions(review_bo_block, pick_records)
    recorder.llm_interactions["stage_b"] = {
        "executed": True,
        "attempts": serialize_attempts(attempts_B),
        "fallback_used": fb_B,
        "final_blocks": serialize_blocks(blocks_B),
        "review_bo_block": review_bo_block.to_dict() if review_bo_block else None,
        "final_candidates": final_candidates,
        "overrides": overrides,
    }
    if config.verbose:
        print(
            f"    [round {round_idx+1}/{config.n_iterations}] Stage B: "
            f"final_candidates={final_candidates}",
            flush=True,
        )

    # ---- Score ----
    if final_candidates:
        scores = _score_via_scorer(scorer, final_candidates)
        for s in final_candidates:
            history[s] = scores[s]
            if s in pool:
                pool.remove(s)
        # Trajectory scores: per-SMILES list (length n_obj).
        recorder.scores = {
            s: _score_to_traj(scores[s], n_obj) for s in final_candidates
        }
    else:
        recorder.scores = {}

    if config.verbose and final_candidates:
        print(
            f"    [round {round_idx+1}/{config.n_iterations}] scoring "
            f"{len(final_candidates)} SMILES (n_obj={n_obj})...",
            flush=True,
        )
        for s in final_candidates:
            sc = scores[s]
            sc_str = _format_score_for_verbose(sc, n_obj)
            print(
                f"    [round {round_idx+1}/{config.n_iterations}]   scored "
                f"{s!r} = {sc_str}",
                flush=True,
            )

    recorder.pool_after = list(pool)


def _score_to_traj(sc: Optional[ScoreValue], n_obj: int) -> List[Optional[float]]:
    """Render a single :data:`ScoreValue` to the trajectory's per-SMILES list."""
    if sc is None:
        return [None] * max(1, n_obj)
    if n_obj == 1:
        return [float(sc) if not isinstance(sc, (list, tuple)) else float(sc[0])]
    if isinstance(sc, (list, tuple)):
        return [float(v) if v is not None else None for v in sc]
    return [None] * n_obj


def _format_score_for_verbose(sc: Optional[ScoreValue], n_obj: int) -> str:
    """Format a :data:`ScoreValue` for stdout (matches
    ``bayesian_analog_search.verbose=True`` style)."""
    if sc is None:
        return "None"
    if n_obj == 1:
        return f"{float(sc):.3f}" if not isinstance(sc, (list, tuple)) else f"{float(sc[0]):.3f}"
    if isinstance(sc, (list, tuple)):
        return "[" + ", ".join(
            "None" if v is None else f"{float(v):.3f}" for v in sc
        ) + "]"
    return "?"


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _serialize_history_for_state(
    history: "OrderedDict[str, ScoreValue]",
) -> List[Tuple[str, Any]]:
    """Convert history to a list of ``(smi, score_for_state)`` for
    embedding in the round state. The score is kept as the raw
    :data:`ScoreValue` (``float`` or ``list[float]``).
    """
    return [(s, sc) for s, sc in history.items()]


def _snapshot_actions(
    *,
    pool: List[str],
    history: "OrderedDict[str, ScoreValue]",
    config: OrchestratorConfig,
    round_idx: int,
    stagnation_counter: int,
) -> Dict[str, Any]:
    return {
        "round_idx": round_idx,
        "n_total_rounds": config.n_iterations,
        "pdf_context": config.pdf_context,
        "objective_legend": list(config.objective_legend),
        "pool": list(pool),
        "pool_size_cap": config.pool_max_size,
        "history": _serialize_history_for_state(history),
        "best": compute_best(dict(history), config.minimize, config.n_obj),
        "stagnation_counter": stagnation_counter,
        "previous_errors": [],
        "attempt": 1,
        "pool_min_size": getattr(config, "pool_min_size", 1) or 1,
        "guidance": config.guidance,
    }


def _snapshot_suggestions(
    *,
    pool: List[str],
    history: "OrderedDict[str, ScoreValue]",
    bo_picks: List[PickRecord],
    acq_function: str,
    config: OrchestratorConfig,
    round_idx: int,
    stagnation_counter: int,
) -> Dict[str, Any]:
    return {
        "round_idx": round_idx,
        "n_total_rounds": config.n_iterations,
        "pdf_context": config.pdf_context,
        "objective_legend": list(config.objective_legend),
        "pool": list(pool),
        "pool_size_cap": config.pool_max_size,
        "history": _serialize_history_for_state(history),
        "best": compute_best(dict(history), config.minimize, config.n_obj),
        "stagnation_counter": stagnation_counter,
        "previous_errors": [],
        "attempt": 1,
        "bo_suggestions": [p.to_dict() for p in bo_picks],
        "acq_function": acq_function,
        "guidance": config.guidance,
    }


def _action_state_from_snapshot(snap: Dict[str, Any]) -> PreActionState:
    return PreActionState(
        round_idx=snap["round_idx"],
        n_total_rounds=snap["n_total_rounds"],
        pdf_context=snap["pdf_context"],
        objective_legend=snap["objective_legend"],
        pool=tuple(snap["pool"]),
        pool_size_cap=snap["pool_size_cap"],
        history=tuple(snap["history"]),
        best=snap["best"],
        stagnation_counter=snap["stagnation_counter"],
        previous_errors=tuple(snap["previous_errors"]),
        attempt=snap["attempt"],
        pool_min_size=int(snap.get("pool_min_size", 1) or 1),
        guidance=snap.get("guidance", ""),
    )


def _build_review_analogs_state(
    *,
    action_state: PreActionState,
    new_analogs: List[AnalogueRecord],
    pool: List[str],
    history: "OrderedDict[str, ScoreValue]",
    round_idx: int,
    config: OrchestratorConfig,
) -> PreReviewAnalogsState:
    return PreReviewAnalogsState(
        round_idx=round_idx,
        n_total_rounds=config.n_iterations,
        pdf_context=config.pdf_context,
        objective_legend=list(config.objective_legend),
        pool=tuple(pool),
        pool_size_cap=config.pool_max_size,
        history=tuple(history.items()),
        best=action_state.best,
        stagnation_counter=action_state.stagnation_counter,
        previous_errors=(),
        attempt=1,
        new_analogs=tuple(new_analogs),
        guidance=config.guidance,
    )


def _suggestion_state_from_snapshot(snap: Dict[str, Any]) -> PostSuggestionState:
    return PostSuggestionState(
        round_idx=snap["round_idx"],
        n_total_rounds=snap["n_total_rounds"],
        pdf_context=snap["pdf_context"],
        objective_legend=snap["objective_legend"],
        pool=tuple(snap["pool"]),
        pool_size_cap=snap["pool_size_cap"],
        history=tuple(snap["history"]),
        best=snap["best"],
        stagnation_counter=snap["stagnation_counter"],
        previous_errors=tuple(snap["previous_errors"]),
        attempt=snap["attempt"],
        bo_suggestions=tuple(
            PickRecord(
                smiles=p["smiles"],
                acq_value=p["acq_value"],
                mu=p["mu"],
                sigma=p["sigma"],
            )
            for p in snap["bo_suggestions"]
        ),
        acq_function=snap["acq_function"],
        guidance=snap.get("guidance", ""),
    )


# ---------------------------------------------------------------------------
# Block application helpers
# ---------------------------------------------------------------------------


def _first_of_type(blocks: Sequence[LLMBlock], type_name: str):
    """Return the first block of a given type, or None."""
    for b in blocks:
        if b.type == type_name:
            return b
    return None


def _history_keys(history) -> set:
    if isinstance(history, dict):
        return set(history.keys())
    if isinstance(history, OrderedDict):
        return set(history.keys())
    return set()


def _apply_actions(
    *,
    blocks: Sequence[LLMBlock],
    pool: List[str],
    analog_fn: Optional[Callable],
    reasyn_pool: Optional[ReasynConfigPool],
) -> List[AnalogueRecord]:
    """Apply Stage A1 action blocks to the pool in-place.

    Returns the list of new :class:`AnalogueRecord` produced by the
    ``analog`` block (empty if none).  The caller decides whether to
    pass them to Stage A2 for review.
    """
    new_analogs: List[AnalogueRecord] = []

    # propose
    prop = _first_of_type(blocks, "propose")
    if prop is not None:
        for s in prop.smiles:
            if s and s not in pool and s not in _history_keys(pool):
                pool.append(s)

    # reject
    rej = _first_of_type(blocks, "reject")
    if rej is not None:
        for t in rej.targets:
            if t in pool:
                pool.remove(t)

    # analog: call analog_fn on seeds → returns new analogues
    ana = _first_of_type(blocks, "analog")
    if ana is not None and analog_fn is not None and reasyn_pool is not None:
        cfg = pick_reasyn_config(
            reasyn_pool, ana.generator_hint, ana.reasyn_config_override,
        )
        for seed in ana.seeds:
            try:
                analogues = list(analog_fn([seed]))
            except Exception as exc:
                LOGGER.warning("analog_fn raised for seed %s: %s", seed, exc)
                analogues = []
            for a in analogues:
                if a.analogue_smiles and a.analogue_smiles not in pool \
                        and a.analogue_smiles not in _history_keys(pool):
                    new_analogs.append(a)

    return new_analogs


def _apply_review_analogs(
    *,
    blocks: Sequence[LLMBlock],
    pool: List[str],
    new_analogs: List[AnalogueRecord],
) -> None:
    """Apply Stage A2 review_analogs block: keep adds to pool, reject drops."""
    ra = _first_of_type(blocks, "review_analogs")
    if ra is None:
        # No review block: drop all (fail-closed).
        return
    for rec in new_analogs:
        ver = ra.decisions.get(rec.analogue_smiles, "reject")
        if ver == "keep":
            if rec.analogue_smiles not in pool:
                pool.append(rec.analogue_smiles)
        # 'reject' or 'rescore_with_different_params' drops the analogue.


def _apply_review_suggestions(
    review_bo_block: Optional[ReviewBOBlock],
    bo_picks: List[PickRecord],
) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """Translate a review_bo block into ``(final_candidates, overrides)``."""
    if not bo_picks:
        return [], {}
    if review_bo_block is None:
        return [p.smiles for p in bo_picks], {}

    final: List[str] = []
    overrides: Dict[str, Optional[str]] = {}
    for p in bo_picks:
        ver = review_bo_block.decisions.get(p.smiles, "ok")
        if ver == "skip":
            overrides[p.smiles] = None
            continue
        if ver.startswith("override:"):
            new_smi = ver[len("override:"):].strip()
            final.append(new_smi)
            overrides[p.smiles] = new_smi
        else:
            final.append(p.smiles)
            overrides[p.smiles] = p.smiles
    return final, overrides


# ---------------------------------------------------------------------------
# Config echo helper
# ---------------------------------------------------------------------------


def _config_to_dict(config: OrchestratorConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "init_size": config.init_size,
        "batch_size": config.batch_size,
        "n_iterations": config.n_iterations,
        "max_completion_rounds": config.max_completion_rounds,
        "smiles_max_len": config.smiles_max_len,
        "max_pool_size_iters": config.max_pool_size_iters,
        "method": config.method,
        "seed": config.seed,
        "objective_legend": list(config.objective_legend),
        "minimize": list(config.minimize),
        "pool_max_size": config.pool_max_size,
        "pool_min_size": config.pool_min_size,
        "n_obj": config.n_obj,
        "guidance": config.guidance,
    }
    if config.bo_config is not None and hasattr(config.bo_config, "__dict__"):
        try:
            import dataclasses as _dc
            if _dc.is_dataclass(config.bo_config):
                out["bo_config"] = _dc.asdict(config.bo_config)
            else:
                out["bo_config"] = dict(config.bo_config.__dict__)
        except Exception:
            out["bo_config"] = repr(config.bo_config)
    # Strip the non-JSON-serializable LLMClientConfig (it lives in
    # ``BayesianLDMSearchConfig`` and is captured separately by
    # ``run_search.py`` under the ``llm`` key).
    if "llm_config" in out:
        llm_cfg = out.pop("llm_config")
        if llm_cfg is not None:
            try:
                import dataclasses as _dc
                if _dc.is_dataclass(llm_cfg):
                    out["llm_config"] = _dc.asdict(llm_cfg)
                else:
                    out["llm_config"] = dict(getattr(llm_cfg, "__dict__", {}))
            except Exception:
                # Last resort: stringify the relevant fields.
                out["llm_config"] = {
                    "model": getattr(llm_cfg, "model", "?"),
                }
    return out


__all__ = [
    "OrchestratorConfig",
    "run_bo_with_llm",
    "compute_best",
    "_cur_best_per_obj",
    "_any_obj_improved",
]
