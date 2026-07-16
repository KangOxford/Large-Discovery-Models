"""Tests for prompt rendering (three stages).

Validates that:

* Stage A1, A2, B system prompts differ and include the right framing.
* User prompts for each stage render with the right header and include
  the right fields.
* Truncation caps (pool, history, PDF) are honored.
* Multi-obj (history values are list[float]; PickRecord.mu/sigma are
  list[float]) is rendered correctly.
* Best anchor is shown for both single-obj (str) and multi-obj
  (list[str] Pareto front).
"""

import pytest

from strbo_v1.llm_advisor.prompt import (
    HISTORY_CAP,
    PDF_CAP_CHARS,
    POOL_CAP,
    SYSTEM_ACTIONS,
    SYSTEM_REVIEW_ANALOGS,
    SYSTEM_REVIEW_SUGGESTIONS,
    _format_best,
    _format_bo_suggestions,
    _format_guidance_suffix,
    _format_history,
    _format_mu_sigma,
    _format_pool,
    format_system_actions,
    format_system_review_analogs,
    format_system_review_suggestions,
    render_user_actions,
    render_user_review_analogs,
    render_user_suggestions,
)
from strbo_v1.llm_advisor.round_state import (
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
)
from strbo_v1.llm_advisor.state import (
    AnalogueRecord,
    PickRecord,
)


def _pre(**over):
    base = dict(
        round_idx=0, n_total_rounds=5,
        pdf_context="short pdf",
        objective_legend=[{"name": "vina", "minimize": True, "ref": 0.0}],
        pool=("CCO", "CCN"),
        pool_size_cap=100,
        history=(("CCO", -7.2),),  # n_obj=1: float
        best="CCO",                  # n_obj=1 best
        stagnation_counter=0,
        previous_errors=(), attempt=1,
        pool_min_size=1,
    )
    base.update(over)
    return PreActionState(**base)


def _pre_review(**over):
    base = dict(
        round_idx=0, n_total_rounds=5,
        pdf_context="",
        objective_legend=[],
        pool=("CCO",),
        pool_size_cap=100,
        history=(("CCO", -7.2),),
        best="CCO",
        stagnation_counter=0,
        previous_errors=(), attempt=1,
        new_analogs=(),
    )
    base.update(over)
    return PreReviewAnalogsState(**base)


def _post(**over):
    base = dict(
        round_idx=0, n_total_rounds=5,
        pdf_context="short pdf",
        objective_legend=[{"name": "vina", "minimize": True, "ref": 0.0}],
        pool=("CCO", "CCN"),
        pool_size_cap=100,
        history=(("CCO", -7.2),),
        best="CCO",
        stagnation_counter=0,
        previous_errors=(), attempt=1,
        bo_suggestions=(
            PickRecord(
                smiles="CCN", acq_value=0.5, mu=-6.9, sigma=0.3,
            ),
        ),
        acq_function="ei",
    )
    base.update(over)
    return PostSuggestionState(**base)


def _post_multi(**over):
    """PostSuggestionState with n_obj=2 history (list[float])."""
    base = dict(
        round_idx=0, n_total_rounds=5,
        pdf_context="short pdf",
        objective_legend=[
            {"name": "vina", "minimize": True, "ref": 0.0},
            {"name": "nn", "minimize": False, "ref": 5.0},
        ],
        pool=("CCO", "CCN"),
        pool_size_cap=100,
        history=(("CCO", [-7.2, 5.4]), ("CCN", [-6.9, 6.0])),  # multi-obj
        best=["CCO", "CCN"],                                          # Pareto front
        stagnation_counter=0,
        previous_errors=(), attempt=1,
        bo_suggestions=(
            PickRecord(
                smiles="CCN", acq_value=0.5,
                mu=[-6.9, 6.0], sigma=[0.3, 0.4],
            ),
        ),
        acq_function="ei",
    )
    base.update(over)
    return PostSuggestionState(**base)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


def test_stage_a1_system_prompt_says_stage_a1() -> None:
    s = SYSTEM_ACTIONS.format(round_idx=0)
    assert "STAGE A1" in s
    assert "BO will run AFTER your decisions" in s


def test_stage_a2_system_prompt_says_stage_a2() -> None:
    s = SYSTEM_REVIEW_ANALOGS.format(round_idx=0)
    assert "STAGE A2" in s
    assert "review_analogs" in s


def test_stage_b_system_prompt_says_stage_b() -> None:
    s = SYSTEM_REVIEW_SUGGESTIONS.format(round_idx=0)
    assert "STAGE B" in s
    assert "review_bo" in s


def test_system_prompts_differ() -> None:
    a1 = SYSTEM_ACTIONS.format(round_idx=0)
    a2 = SYSTEM_REVIEW_ANALOGS.format(round_idx=0)
    b = SYSTEM_REVIEW_SUGGESTIONS.format(round_idx=0)
    assert a1 != a2
    assert a1 != b
    assert a2 != b


# ---------------------------------------------------------------------------
# User prompt rendering
# ---------------------------------------------------------------------------


def test_stage_a1_user_prompt_has_no_bo_suggestions() -> None:
    p = render_user_actions(_pre())
    assert "BO suggestions" not in p
    assert "Stage A1" in p


def test_stage_a2_user_prompt_lists_analogs() -> None:
    pending = (AnalogueRecord(seed_smiles="CCO", analogue_smiles="A1", reasyn_score=0.9, num_steps=2),)
    p = render_user_review_analogs(_pre_review(new_analogs=pending))
    assert "Stage A2" in p
    assert "A1" in p
    assert "1 items" in p


def test_stage_a2_user_prompt_no_analogs() -> None:
    p = render_user_review_analogs(_pre_review(new_analogs=()))
    assert "Stage A2" in p
    assert "0 items" in p


def test_stage_b_user_prompt_has_bo_suggestions() -> None:
    p = render_user_suggestions(_post())
    assert "Stage B" in p
    assert "BO suggestions" in p
    assert "CCN" in p


def test_stage_b_user_prompt_allowed_blocks_hint() -> None:
    p = render_user_suggestions(_post())
    assert "review_bo" in p


def test_stage_a1_user_prompt_allowed_blocks_hint() -> None:
    p = render_user_actions(_pre())
    assert "propose" in p
    assert "reject" in p
    assert "analog" in p
    assert "noop" in p


def test_previous_errors_appear_when_present() -> None:
    p = render_user_actions(_pre(previous_errors=("ParseError: bad JSON",)))
    assert "Previous errors" in p
    assert "ParseError: bad JSON" in p


def test_no_previous_errors_block_when_empty() -> None:
    p = render_user_actions(_pre(previous_errors=()))
    assert "Previous errors" not in p


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_pool_truncated_to_cap() -> None:
    big_pool = tuple(f"C{'C' * i}" for i in range(POOL_CAP + 20))
    p = render_user_actions(_pre(pool=big_pool))
    assert f"(+{len(big_pool) - POOL_CAP} more, truncated)" in p
    assert f"C{'C' * 0}" in p
    assert f"C{'C' * (POOL_CAP - 1)}" in p
    assert f"C{'C' * (POOL_CAP + 5)}" not in p


def test_history_truncated_to_cap() -> None:
    hist = tuple(
        (f"C{'C' * i}", -1.0 * i) for i in range(HISTORY_CAP + 10)
    )
    p = render_user_actions(_pre(history=hist))
    assert f"showing last {HISTORY_CAP} of {len(hist)}" in p


def test_pdf_truncated() -> None:
    long_pdf = "x" * (PDF_CAP_CHARS + 500)
    p = render_user_actions(_pre(pdf_context=long_pdf))
    assert "truncated" in p
    assert f"original length={len(long_pdf)}" in p
    assert "x" * (PDF_CAP_CHARS + 100) not in p


def test_short_pdf_not_truncated() -> None:
    short = "x" * 100
    p = render_user_actions(_pre(pdf_context=short))
    assert "truncated" not in p
    assert short in p


# ---------------------------------------------------------------------------
# Format header (shared between Stage A1, A2, B)
# ---------------------------------------------------------------------------


def test_format_header_present_in_stage_a1() -> None:
    from strbo_v1.llm_advisor.prompt import _FORMAT_HEADER
    assert "RESPONSE FORMAT" in SYSTEM_ACTIONS
    assert _FORMAT_HEADER in SYSTEM_ACTIONS


def test_format_header_present_in_stage_a2() -> None:
    from strbo_v1.llm_advisor.prompt import _FORMAT_HEADER
    assert "RESPONSE FORMAT" in SYSTEM_REVIEW_ANALOGS
    assert _FORMAT_HEADER in SYSTEM_REVIEW_ANALOGS


def test_format_header_present_in_stage_b() -> None:
    from strbo_v1.llm_advisor.prompt import _FORMAT_HEADER
    assert "RESPONSE FORMAT" in SYSTEM_REVIEW_SUGGESTIONS
    assert _FORMAT_HEADER in SYSTEM_REVIEW_SUGGESTIONS


def test_format_header_uses_imperative_language() -> None:
    from strbo_v1.llm_advisor.prompt import _FORMAT_HEADER
    assert "MUST" in _FORMAT_HEADER
    assert "Do NOT" in _FORMAT_HEADER
    assert "BEFORE YOU REPLY" in _FORMAT_HEADER


def test_stage_b_prompt_mentions_empty_decisions() -> None:
    assert "top-0" in SYSTEM_REVIEW_SUGGESTIONS
    assert "empty" in SYSTEM_REVIEW_SUGGESTIONS.lower() or '"decisions": {}' in SYSTEM_REVIEW_SUGGESTIONS


# ---------------------------------------------------------------------------
# pool_size_requirement hint (Stage A1 only)
# ---------------------------------------------------------------------------


def test_pool_size_requirement_appears_when_below_min() -> None:
    state = _pre(
        pool=("CCO",),
        pool_min_size=3,
    )
    p = render_user_actions(state)
    assert "Pool size requirement" in p
    assert "MUST emit" in p
    assert "1 SMILES" in p
    assert "minimum is 3" in p


def test_pool_size_requirement_absent_when_above_min() -> None:
    state = _pre(
        pool=("CCO", "CCN", "CCC"),
        pool_min_size=3,
    )
    p = render_user_actions(state)
    assert "Pool size requirement" not in p


def test_pool_size_requirement_absent_when_min_is_one() -> None:
    state = _pre(
        pool=("CCO",),
        pool_min_size=1,
    )
    p = render_user_actions(state)
    assert "Pool size requirement" not in p


# ---------------------------------------------------------------------------
# Best anchor rendering (multi-obj Pareto vs single-obj)
# ---------------------------------------------------------------------------


def test_format_best_single_objective_str() -> None:
    """n_obj=1: a single SMILES (or "(none yet — pool is fresh)")."""
    assert _format_best("CCO", n_obj=1) == "  CCO"
    assert _format_best("", n_obj=1) == "  (none yet — pool is fresh)"


def test_format_best_multi_objective_pareto() -> None:
    """n_obj>=2: list of Pareto SMILES (capped at PARETO_CAP)."""
    front = ["CCO", "CCN", "CCC"]
    out = _format_best(front, n_obj=2)
    assert "CCO" in out and "CCN" in out and "CCC" in out
    # Truncation
    big = [f"S{i}" for i in range(20)]
    out2 = _format_best(big, n_obj=2)
    assert "more in the Pareto front" in out2


def test_stage_a1_user_prompt_shows_best() -> None:
    p = render_user_actions(_pre(best="CCO"))
    assert "Best so far" in p
    assert "CCO" in p


def test_stage_a1_user_prompt_no_best_when_empty() -> None:
    p = render_user_actions(_pre(best="", history=()))
    assert "Best so far" in p
    assert "(none yet" in p


def test_stage_b_user_prompt_shows_pareto_front() -> None:
    p = render_user_suggestions(_post_multi())
    assert "Best so far" in p
    assert "CCO" in p
    assert "CCN" in p
    # For truncated Pareto fronts, the label "Pareto front" appears.
    big_front = [f"S{i}" for i in range(20)]
    p_big = render_user_suggestions(_post_multi(best=big_front))
    assert "Pareto front" in p_big


# ---------------------------------------------------------------------------
# Multi-obj: PickRecord mu/sigma rendering
# ---------------------------------------------------------------------------


def test_format_mu_sigma_scalar() -> None:
    assert _format_mu_sigma(0.5) == "0.500"
    assert _format_mu_sigma(-1.234) == "-1.234"


def test_format_mu_sigma_list() -> None:
    out = _format_mu_sigma([-7.2, 5.4])
    assert out == "[-7.200, 5.400]"


def test_format_bo_suggestions_scalar_mu() -> None:
    p = PickRecord(smiles="CCO", acq_value=0.5, mu=-1.0, sigma=0.3)
    out = _format_bo_suggestions([p])
    assert "mu=-1.000" in out
    assert "sigma=0.300" in out


def test_format_bo_suggestions_list_mu() -> None:
    p = PickRecord(
        smiles="CCO", acq_value=0.5,
        mu=[-7.2, 5.4], sigma=[0.3, 0.4],
    )
    out = _format_bo_suggestions([p])
    assert "mu=[-7.200, 5.400]" in out
    assert "sigma=[0.300, 0.400]" in out


# ---------------------------------------------------------------------------
# History rendering: float vs list[float]
# ---------------------------------------------------------------------------


def test_format_history_scalar_scores() -> None:
    hist = [("CCO", -7.2), ("CCN", -6.9)]
    out = _format_history(hist)
    assert "CCO  ->  [-7.200]" in out
    assert "CCN  ->  [-6.900]" in out


def test_format_history_list_scores() -> None:
    """Multi-obj: each row is ``[v0, v1, v2, ...]``."""
    hist = [("CCO", [-7.2, 5.4]), ("CCN", [-6.9, 6.0])]
    out = _format_history(hist)
    assert "CCO  ->  [-7.200, 5.400]" in out
    assert "CCN  ->  [-6.900, 6.000]" in out


# ---------------------------------------------------------------------------
# GP summary removed (per user's design review)
# ---------------------------------------------------------------------------


def test_stage_a1_user_prompt_does_not_show_gp_summary() -> None:
    """The GP summary block was removed from the LLM-facing prompt."""
    p = render_user_actions(_pre())
    assert "GP summary" not in p
    assert "train_score_mean" not in p


def test_stage_b_user_prompt_does_not_show_gp_summary() -> None:
    p = render_user_suggestions(_post())
    assert "GP summary" not in p
    assert "train_score_mean" not in p


def test_stage_a1_user_prompt_does_not_show_diversity() -> None:
    """Tanimoto diversity was removed from the LLM-facing prompt."""
    p = render_user_actions(_pre())
    assert "Tanimoto" not in p
    assert "diversity" not in p.lower() or "diversity_metric" not in p


# ---------------------------------------------------------------------------
# Stagnation: phrasing is "any objective's improvement"
# ---------------------------------------------------------------------------


def test_stagnation_phrase_mentions_any_objective() -> None:
    p = render_user_actions(_pre())
    assert "any objective" in p


# ---------------------------------------------------------------------------
# External guidance (LLM_GUIDANCE / --llm-guide)
# ---------------------------------------------------------------------------


def test_format_guidance_suffix_empty_inputs() -> None:
    """Empty / whitespace-only guidance returns an empty string (no-op)."""
    assert _format_guidance_suffix("") == ""
    assert _format_guidance_suffix("   ") == ""
    assert _format_guidance_suffix("\n\n") == ""


def test_format_guidance_suffix_appends_block() -> None:
    out = _format_guidance_suffix("prefer analog")
    assert "## EXTERNAL GUIDANCE" in out
    assert "prefer analog" in out
    # The block is appended at the end of the system prompt.
    assert out.endswith("prefer analog")


def test_format_guidance_suffix_strips_outer_whitespace() -> None:
    out = _format_guidance_suffix("  \n  text  \n  ")
    assert out.strip().endswith("text")
    assert "## EXTERNAL GUIDANCE" in out


def test_format_system_actions_baseline_no_guidance() -> None:
    """When guidance is empty, the formatted system prompt equals
    the legacy ``SYSTEM_ACTIONS.format(round_idx=...)`` byte-for-byte
    (no regression)."""
    s = format_system_actions(round_idx=3, guidance="")
    assert s == SYSTEM_ACTIONS.format(round_idx=3)
    assert "## EXTERNAL GUIDANCE" not in s


def test_format_system_actions_appends_guidance() -> None:
    s = format_system_actions(round_idx=3, guidance="Use analog heavily")
    assert s.startswith(SYSTEM_ACTIONS.format(round_idx=3))
    assert "## EXTERNAL GUIDANCE" in s
    assert "Use analog heavily" in s


def test_format_system_review_analogs_baseline_no_guidance() -> None:
    s = format_system_review_analogs(round_idx=2, guidance="")
    assert s == SYSTEM_REVIEW_ANALOGS.format(round_idx=2)
    assert "## EXTERNAL GUIDANCE" not in s


def test_format_system_review_analogs_appends_guidance() -> None:
    s = format_system_review_analogs(round_idx=2, guidance="Be strict")
    assert s.startswith(SYSTEM_REVIEW_ANALOGS.format(round_idx=2))
    assert "## EXTERNAL GUIDANCE" in s
    assert "Be strict" in s


def test_format_system_review_suggestions_baseline_no_guidance() -> None:
    s = format_system_review_suggestions(round_idx=4, guidance="")
    assert s == SYSTEM_REVIEW_SUGGESTIONS.format(round_idx=4)
    assert "## EXTERNAL GUIDANCE" not in s


def test_format_system_review_suggestions_appends_guidance() -> None:
    s = format_system_review_suggestions(round_idx=4, guidance="Don't override")
    assert s.startswith(SYSTEM_REVIEW_SUGGESTIONS.format(round_idx=4))
    assert "## EXTERNAL GUIDANCE" in s
    assert "Don't override" in s


def test_format_system_actions_preserves_round_idx_format() -> None:
    """``{round_idx}`` placeholder is filled; guidance is appended after."""
    s = format_system_actions(round_idx=7, guidance="X")
    assert "STAGE A1 (actions) of round 7" in s
    assert "## EXTERNAL GUIDANCE" in s
    assert s.endswith("X")


def test_multiline_guidance_renders_verbatim() -> None:
    g = "Line 1\nLine 2\nLine 3"
    s = format_system_actions(round_idx=0, guidance=g)
    assert "Line 1" in s
    assert "Line 2" in s
    assert "Line 3" in s
