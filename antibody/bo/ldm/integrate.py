"""bo/ldm/integrate.py — bridge between Orchestrator and CASMOPOLITANCat.

This module is the ONLY allowed bridge between ``bo/`` and ``bo.ldm/``.

Public surface:
    build_status(cat, antigen_id, antigen_seed, iter_seed, antigen_context) -> OrchestratorStatus
    apply_decision(cat, decision) -> None  # set cat._search_dsl, cat._bias_dsl
    sample_candidates(cat, n, x_center, timeout_s) -> np.ndarray  # DSL or Hamming
    score_with_bias(cat, ei_scores, candidates, bias_weight) -> np.ndarray
"""
from __future__ import annotations

import numpy as np

from bo.ldm.dsl.bias import BiasAtom
from bo.ldm.dsl.search_space import SearchSpaceAtom
from bo.ldm.orchestrator.loop import OrchestratorDecision
from bo.ldm.orchestrator.status import OrchestratorStatus


def build_status(
    cat,
    antigen_id: str,
    antigen_seed: int,
    iter_seed: int,
    antigen_context: dict | None = None,
) -> OrchestratorStatus:
    """Construct an OrchestratorStatus from CASMOPOLITANCat state."""
    full_history = []
    if hasattr(cat, "fx") and cat.fx is not None and len(cat.fx) > 0:
        for i in range(len(cat.fx)):
            seq = cat.x[i].tolist() if hasattr(cat, "x") else []
            full_history.append((seq, float(cat.fx[i, 0]), i))
    if hasattr(cat, "fx") and len(cat.fx) > 0:
        best_idx = int(cat.fx.argmin())
        best_value = float(cat.fx[best_idx, 0])
        best_sequence = cat.x[best_idx].tolist()
        n_init = getattr(cat, "n_init", 0)
        if best_idx < n_init:
            n_iters_without_improvement = max(0, len(cat.fx) - n_init)
        else:
            n_iters_without_improvement = len(cat.fx) - 1 - best_idx
    else:
        best_value = 0.0
        best_sequence = []
        n_iters_without_improvement = 0
    n_evals = len(cat.fx) if hasattr(cat, "fx") and cat.fx is not None else 0

    return OrchestratorStatus(
        iteration=int(getattr(cat, "_iteration_count", 0)),
        antigen_id=antigen_id,
        antigen_seed=antigen_seed,
        iter_seed=iter_seed,
        current_search_dsl=getattr(cat, "_search_dsl", None),
        current_bias_dsl=getattr(cat, "_bias_dsl", None),
        full_history=full_history,
        best_value=best_value,
        best_sequence=best_sequence,
        n_evals=n_evals,
        n_iters_without_improvement=n_iters_without_improvement,
        antigen_context=antigen_context or {},
    )


def apply_decision(cat, decision: OrchestratorDecision) -> None:
    """Store decision.search_dsl / decision.bias_dsl on the cat instance."""
    cat._search_dsl: SearchSpaceAtom | None = decision.search_dsl
    cat._bias_dsl: BiasAtom | None = decision.bias_dsl


def sample_candidates(
    cat,
    n: int,
    x_center: np.ndarray | None = None,
    timeout_s: float = 5.0,
) -> np.ndarray:
    """Sample ``n`` candidate sequences.

    If ``cat._search_dsl`` is None: use AntBO's original Hamming-ball
    sampling around ``x_center`` (original AntBO path, unpolluted).
    If ``cat._search_dsl`` is set: use structural DSL sampling.

    Returns ndarray of shape ``(k, dim)`` with ``k <= n``.
    """
    search_dsl: SearchSpaceAtom | None = getattr(cat, "_search_dsl", None)
    if search_dsl is None:
        from bo.localbo_utils import random_sample_within_discrete_tr_ordinal
        candidates = np.array([
            random_sample_within_discrete_tr_ordinal(
                x_center, cat.length_discrete, cat.config
            )
            for _ in range(n)
        ])
        return candidates

    rng = np.random.default_rng()
    samples = search_dsl.sample(n=n, rng=rng, timeout_s=timeout_s)
    if not samples:
        return np.empty((0, cat.dim), dtype=int)
    return np.array(samples, dtype=int)


def score_with_bias(
    cat,
    ei_scores: np.ndarray,
    candidates: np.ndarray,
    bias_weight: float = 0.1,
) -> np.ndarray:
    """Add bias contribution to EI scores if bias DSL is set."""
    bias_dsl: BiasAtom | None = getattr(cat, "_bias_dsl", None)
    if bias_dsl is None:
        return ei_scores
    bias_scores = np.array([
        bias_dsl([int(x) for x in seq]) for seq in candidates
    ])
    return ei_scores + bias_weight * bias_scores
