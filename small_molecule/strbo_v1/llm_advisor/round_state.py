"""Three stage-specific round-state dataclasses.

A round of LLM-driven BO has three LLM invocations; each one receives
a different snapshot:

* :class:`PreActionState` — handed to Stage A1 (actions).  The LLM
  sees the current pool and decides ``propose`` / ``reject`` /
  ``analog`` / ``noop``.

* :class:`PreReviewAnalogsState` — handed to Stage A2 (review analogs).
  Shown only when an ``analog`` action produced non-empty results.
  The LLM decides ``keep`` / ``reject`` for each generated analogue.

* :class:`PostSuggestionState` — handed to Stage B (review suggestions).
  The LLM sees the post-mutation pool and the BO suggestions computed
  from that pool.  Its only decision is :class:`ReviewBOBlock`.

All inherit :class:`_BaseRoundState` (private) for the common
read-only fields.

The ``previous_errors`` and ``attempt`` fields are populated by
:func:`strbo_v1.llm_advisor.advisor.LLMAdvisor._decide_with_retry`
on each retry, so the LLM gets concrete feedback on what to fix.

Design note (per user's review)
--------------------------------
The state carries only what the LLM needs to make a decision:
the current pool, the history of evaluated points, the current
"best" anchor (single best SMILES for n_obj=1, Pareto front for
n_obj>=2), and the stagnation counter. The rich GP metadata, the
Tanimoto diversity score, and the per-objective best score were
dropped — the user deemed them noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from strbo_v1.llm_advisor.state import AnalogueRecord, PickRecord, ScoreValue


# ---------------------------------------------------------------------------
# Private base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BaseRoundState:
    """Common read-only fields shared by all stage round states.

    The history values are :data:`ScoreValue` (``float`` for n_obj=1,
    ``list[float]`` of length n_obj for n_obj>=2). They are stored as
    the orchestrator produced them (the orchestrator does not wrap
    a single-obj score in a 1-tuple — it keeps it as a bare float so
    the prompt can render ``-7.2`` for single-obj and ``[-7.2, 5.4]``
    for multi-obj).
    """

    # Round metadata
    round_idx: int
    n_total_rounds: int
    pdf_context: str = ""
    objective_legend: List[Dict[str, Any]] = field(default_factory=list)

    # Pool snapshot (read-only — frozen dataclass)
    pool: Tuple[str, ...] = field(default_factory=tuple)
    pool_size_cap: Optional[int] = None

    # History snapshot as (smiles, score) tuples where score is
    # float for n_obj==1, list[float] for n_obj>=2.
    history: Tuple[Tuple[str, ScoreValue], ...] = field(default_factory=tuple)

    # "Best" anchor for the LLM:
    #   n_obj==1:  str (single best SMILES, or "" if no history)
    #   n_obj>=2:  list[str] (Pareto front, or [] if no history)
    best: Union[str, List[str]] = ""

    # Stagnation counter (any objective improvement → reset).
    stagnation_counter: int = 0

    # Retry context (advisor mutates per attempt)
    previous_errors: Tuple[str, ...] = field(default_factory=tuple)
    attempt: int = 1

    # Free-form guidance text appended to all three LLM system prompts
    # (Stage A1 actions, A2 review-analogs, B review-suggestions).
    # Sourced from ``OrchestratorConfig.guidance``. Recorded in the
    # trajectory's ``pre_state_snapshot`` for audit. Empty by default.
    guidance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "n_total_rounds": self.n_total_rounds,
            "pdf_context": self.pdf_context,
            "objective_legend": list(self.objective_legend),
            "pool": list(self.pool),
            "pool_size_cap": self.pool_size_cap,
            "history": [
                {
                    "smiles": s,
                    "score": (list(sc) if isinstance(sc, (list, tuple)) else sc),
                }
                for s, sc in self.history
            ],
            "best": (list(self.best) if isinstance(self.best, (list, tuple)) else self.best),
            "stagnation_counter": self.stagnation_counter,
            "previous_errors": list(self.previous_errors),
            "attempt": self.attempt,
            "guidance": self.guidance,
        }


# ---------------------------------------------------------------------------
# Stage A1 — actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreActionState(_BaseRoundState):
    """Stage A1 snapshot: current pool + pool-size requirement.

    The LLM emits pool-management blocks (``propose`` / ``reject`` /
    ``analog`` / ``noop``).  ``bo_suggestions`` is always empty at
    this stage — BO has not run yet.
    """

    # Minimum pool size enforced in Stage A1.  When ``len(pool) <
    # pool_min_size``, a bare ``noop`` block is rejected and the LLM
    # is re-prompted to emit ``propose`` or ``analog`` to refill.
    # Default 1 = no enforcement.
    pool_min_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["phase"] = "A_actions"
        d["bo_suggestions"] = []
        d["pool_min_size"] = self.pool_min_size
        return d


# ---------------------------------------------------------------------------
# Stage A2 — review analogs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreReviewAnalogsState(_BaseRoundState):
    """Stage A2 snapshot: newly generated analogues to review.

    Shown only when an ``analog`` action produced non-empty results.
    The LLM decides ``keep`` / ``reject`` for each analogue.
    """

    new_analogs: Tuple[AnalogueRecord, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["phase"] = "A_review_analogs"
        d["new_analogs"] = [a.to_dict() for a in self.new_analogs]
        d["bo_suggestions"] = []
        return d


# ---------------------------------------------------------------------------
# Stage B — review suggestions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostSuggestionState(_BaseRoundState):
    """Stage B snapshot: post-mutation pool + BO suggestions.

    The LLM emits exactly one ``review_bo`` block.
    """

    bo_suggestions: Tuple[PickRecord, ...] = field(default_factory=tuple)
    acq_function: str = "ei"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["phase"] = "B_suggestions"
        d["bo_suggestions"] = [p.to_dict() for p in self.bo_suggestions]
        d["acq_function"] = self.acq_function
        return d


__all__ = [
    "PreActionState",
    "PreReviewAnalogsState",
    "PostSuggestionState",
]
