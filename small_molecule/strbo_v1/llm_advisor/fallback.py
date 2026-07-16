"""Fallback decisions when LLM retries are exhausted.

Three entry points (one per stage):

* :func:`fallback_actions` — safe no-op when Stage A1 retries are
  exhausted.  Returns a :class:`NoopBlock`.

* :func:`fallback_review_analogs` — keep all generated analogues
  (maximizes pool growth when LLM is unreliable).  Returns a
  :class:`ReviewAnalogsBlock` with all-``keep`` verdicts.

* :func:`fallback_review_suggestions` — accept every BO suggestion
  as-is.  Returns a :class:`ReviewBOBlock` with all-``ok`` verdicts.

All functions never raise; they always return a list of blocks so the
orchestrator can apply them and keep the BO loop running.
"""

from __future__ import annotations

from typing import List

from strbo_v1.llm_advisor.blocks import (
    NoopBlock,
    ReviewAnalogsBlock,
    ReviewBOBlock,
)
from strbo_v1.llm_advisor.round_state import (
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
)
from strbo_v1.llm_advisor.state import AnalogueRecord


# ---------------------------------------------------------------------------
# Stage A1 — actions
# ---------------------------------------------------------------------------


def fallback_actions(state: PreActionState) -> List:
    """Stage A1 fallback: noop.

    The rationale is hard-coded so trajectory records are consistent
    across runs.
    """
    return [
        NoopBlock(
            rationale="FALLBACK: no pool changes after LLM retries exhausted."
        )
    ]


# ---------------------------------------------------------------------------
# Stage A2 — review analogs
# ---------------------------------------------------------------------------


def fallback_review_analogs(state: PreReviewAnalogsState) -> List:
    """Stage A2 fallback: keep all generated analogues.

    Maximizes pool growth when the LLM is unreliable.
    """
    if not state.new_analogs:
        return []
    return [
        ReviewAnalogsBlock(
            rationale=(
                f"FALLBACK: keeping all {len(state.new_analogs)} "
                "generated analogues after LLM retries exhausted."
            ),
            decisions={a.analogue_smiles: "keep" for a in state.new_analogs},
        )
    ]


# ---------------------------------------------------------------------------
# Stage B — review suggestions
# ---------------------------------------------------------------------------


def fallback_review_suggestions(state: PostSuggestionState) -> List:
    """Stage B fallback: accept every BO suggestion unchanged.

    The LLM is effectively bypassed: BO's top-k is scored as-is.
    """
    if not state.bo_suggestions:
        return []
    return [
        ReviewBOBlock(
            rationale="FALLBACK: accepting all BO suggestions unchanged.",
            decisions={p.smiles: "ok" for p in state.bo_suggestions},
        )
    ]
