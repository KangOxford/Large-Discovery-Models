"""Tests for fallback decision tables (three stages).

Covers the design doc's fallback policy:

* Stage A1 (actions): LLM retries exhausted -> noop.
* Stage A2 (review analogs): LLM retries exhausted -> keep all analogues.
* Stage B (review suggestions): LLM retries exhausted -> all ok.
* Stage B: no BO suggestions -> empty list.
"""

import pytest

from strbo_v1.llm_advisor import (
    AnalogueRecord,
    NoopBlock,
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
    ReviewAnalogsBlock,
    ReviewBOBlock,
    fallback_actions,
    fallback_review_analogs,
    fallback_review_suggestions,
)
from strbo_v1.llm_advisor.state import PickRecord


def _pre_action():
    return PreActionState(
        round_idx=0, n_total_rounds=3,
        pool=("CCO",),
        history=(("CCO", -1.0),),
        pool_min_size=1,
    )


def _pre_review_analogs(pending=()):
    return PreReviewAnalogsState(
        round_idx=0, n_total_rounds=3,
        pool=("CCO",),
        history=(("CCO", -1.0),),
        new_analogs=pending,
    )


def _post(bo=()):
    return PostSuggestionState(
        round_idx=0, n_total_rounds=3,
        pool=("CCO",),
        history=(("CCO", -1.0),),
        bo_suggestions=bo,
        acq_function="ei",
    )


# ---------------------------------------------------------------------------
# Stage A1 — actions
# ---------------------------------------------------------------------------


def test_fallback_actions_returns_noop() -> None:
    blocks = fallback_actions(_pre_action())
    assert len(blocks) == 1
    assert isinstance(blocks[0], NoopBlock)
    assert "FALLBACK" in blocks[0].rationale


# ---------------------------------------------------------------------------
# Stage A2 — review analogs
# ---------------------------------------------------------------------------


def test_fallback_review_analogs_empty_returns_empty() -> None:
    blocks = fallback_review_analogs(_pre_review_analogs(pending=()))
    assert blocks == []


def test_fallback_review_analogs_keeps_all() -> None:
    pending = (
        AnalogueRecord(seed_smiles="CCO", analogue_smiles="A1", reasyn_score=0.9),
        AnalogueRecord(seed_smiles="CCN", analogue_smiles="A2", reasyn_score=0.8),
    )
    blocks = fallback_review_analogs(_pre_review_analogs(pending=pending))
    assert len(blocks) == 1
    assert isinstance(blocks[0], ReviewAnalogsBlock)
    assert blocks[0].decisions == {"A1": "keep", "A2": "keep"}
    assert "FALLBACK" in blocks[0].rationale


# ---------------------------------------------------------------------------
# Stage B — review suggestions
# ---------------------------------------------------------------------------


def test_fallback_review_suggestions_with_suggestions_accepts_all() -> None:
    bo = (
        PickRecord(smiles="CCC", acq_value=0.5, mu=-0.5, sigma=0.3),
        PickRecord(smiles="CCO", acq_value=0.4, mu=-0.6, sigma=0.3),
    )
    blocks = fallback_review_suggestions(_post(bo=bo))
    assert len(blocks) == 1
    assert isinstance(blocks[0], ReviewBOBlock)
    assert blocks[0].decisions == {"CCC": "ok", "CCO": "ok"}
    assert "FALLBACK" in blocks[0].rationale


def test_fallback_review_suggestions_with_no_suggestions_returns_empty() -> None:
    blocks = fallback_review_suggestions(_post(bo=()))
    assert blocks == []


def test_fallback_review_suggestions_single_suggestion() -> None:
    bo = (PickRecord(smiles="CCC", acq_value=0.5, mu=-0.5, sigma=0.3),)
    blocks = fallback_review_suggestions(_post(bo=bo))
    assert len(blocks) == 1
    assert blocks[0].decisions == {"CCC": "ok"}
