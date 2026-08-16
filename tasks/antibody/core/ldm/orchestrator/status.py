"""core/ldm/orchestrator/status.py — OrchestratorStatus dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrchestratorStatus:
    """Status snapshot passed to Orchestrator.step() each BO iteration.

    Fields:
        iteration: 1-indexed BO iteration number.
        antigen_id: e.g. ``"1ADQ_A"``.
        antigen_seed: integer seed (for caching).
        iter_seed: per-iteration seed (for caching).
        current_search_dsl: opaque ``SearchSpaceAtom | None`` (None = AntBO default).
        current_bias_dsl: opaque ``BiasAtom | None`` (None = zero bias).
        current_search_size_estimate: float; if None, computed lazily.
        full_history: list of ``(seq, score, iter)`` triples, sorted by iter asc.
        best_value: best score observed so far.
        best_sequence: best sequence (as list[int] of length 11).
        n_evals: number of evaluations performed.
        n_iters_without_improvement: stall counter.
        last_dsl_rejection_reason: human-readable error from last failed update.
        antigen_context: dict from ``tasks.antibody.core.ldm.antigen_context.collect_absolut_antigen_context``.
    """

    iteration: int
    antigen_id: str
    antigen_seed: int
    iter_seed: int
    current_search_dsl: object = None  # SearchSpaceAtom | None
    current_bias_dsl: object = None    # BiasAtom | None
    current_search_size_estimate: float | None = None
    full_history: list = field(default_factory=list)
    best_value: float = 0.0
    best_sequence: list = field(default_factory=list)
    n_evals: int = 0
    n_iters_without_improvement: int = 0
    n_evals_in_current_tr: int = 0
    last_dsl_rejection_reason: str | None = None
    antigen_context: dict = field(default_factory=dict)