"""Tests for LLMAdvisor retry + fallback behavior.

Uses :class:`MockLLMClient` to script the LLM responses and exercise:

* Happy path: one attempt succeeds, ``fallback_used=False``.
* Transient failure: first attempt invalid, second succeeds —
  ``previous_errors`` recorded, ``fallback_used=False``.
* Total failure: every attempt invalid, ``fallback_used=True`` and
  the returned blocks are the stage fallback.
* Stage isolation: Stage A1 failure does not leak into Stage B's
  ``previous_errors`` (and vice versa).
"""

import sys
import types

import pytest

from strbo_v1.llm_advisor import (
    NoopBlock,
    PostSuggestionState,
    PreActionState,
    PreReviewAnalogsState,
    ProposeBlock,
    ReviewAnalogsBlock,
    ReviewBOBlock,
    AnalogueRecord,
    SemanticError,
)
from strbo_v1.llm_advisor.advisor import LLMAdvisor
from strbo_v1.llm_advisor.client import MockLLMClient, OpenAIChatClient, _serialize_blocks
from strbo_v1.llm_advisor.config import LLMClientConfig
from strbo_v1.llm_advisor.state import PickRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_openai_client_disables_sdk_retries(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    OpenAIChatClient(
        LLMClientConfig(api_key="key", base_url="https://example.test", model="model"),
        timeout=12.0,
    )

    assert captured["timeout"] == 12.0
    assert captured["max_retries"] == 0


def _make_pre_action_state() -> PreActionState:
    return PreActionState(
        round_idx=0, n_total_rounds=3,
        pool=("CCO", "CCN"),
        history=(("CCO", -7.2),),  # n_obj=1: bare float
        pool_min_size=1,
        best="CCO",
    )


def _make_pre_review_analogs_state(
    pending: tuple = (),
) -> PreReviewAnalogsState:
    return PreReviewAnalogsState(
        round_idx=0, n_total_rounds=3,
        pool=("CCO",),
        history=(("CCO", -7.2),),
        new_analogs=pending,
        best="CCO",
    )


def _make_post_state() -> PostSuggestionState:
    return PostSuggestionState(
        round_idx=0, n_total_rounds=3,
        pool=("CCO", "CCC"),
        history=(("CCO", -7.2),),
        bo_suggestions=(PickRecord(smiles="CCC", acq_value=0.5, mu=-6.8, sigma=0.3),),
        acq_function="ei",
        best="CCO",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_stage_a1_happy_path() -> None:
    client = MockLLMClient(scripted_blocks=[
        [NoopBlock(rationale="do nothing")],
    ])
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_actions(_make_pre_action_state())
    assert fb is False
    assert len(attempts) == 1
    assert [b.type for b in blocks] == ["noop"]


def test_attempt_records_raw_input_and_output() -> None:
    client = MockLLMClient(scripted_blocks=[
        [NoopBlock(rationale="do nothing")],
    ])
    advisor = LLMAdvisor(llm=client, max_retries=1, use_rdkit=False)
    _blocks, attempts, _fb = advisor.decide_actions(_make_pre_action_state())

    record = attempts[0].to_dict()
    assert record["stage"] == "A_actions"
    assert "STAGE A1" in record["system_prompt"]
    assert "### Pool" in record["user_prompt"]
    assert record["raw_response"]
    assert record["raw_output"] == record["raw_response"]
    assert record["json_mode"] is True


def test_stage_b_happy_path() -> None:
    client = MockLLMClient(scripted_blocks=[
        [ReviewBOBlock(rationale="all good", decisions={"CCC": "ok"})],
    ])
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_review_suggestions(_make_post_state())
    assert fb is False
    assert [b.type for b in blocks] == ["review_bo"]


# ---------------------------------------------------------------------------
# Retry recovery
# ---------------------------------------------------------------------------


def test_stage_a1_recovers_on_second_attempt() -> None:
    """First attempt invalid; second succeeds."""
    client = MockLLMClient(scripted_responses=[
        "not json",                                      # attempt 1: ParseError
        _serialize_blocks([NoopBlock(rationale="r")]),  # attempt 2: ok
    ])
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_actions(_make_pre_action_state())
    assert fb is False
    assert len(attempts) == 2
    assert attempts[0].validation_errors and "ParseError" in attempts[0].validation_errors[0]
    assert attempts[1].validation_errors == []
    assert [b.type for b in blocks] == ["noop"]


def test_stage_b_recovers_on_second_attempt() -> None:
    client = MockLLMClient(scripted_responses=[
        # First response is a Stage A1 type -> SemanticError (stage mismatch)
        _serialize_blocks([NoopBlock(rationale="wrong stage")]),
        _serialize_blocks([ReviewBOBlock(rationale="ok", decisions={"CCC": "ok"})]),
    ])
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_review_suggestions(_make_post_state())
    assert fb is False
    assert len(attempts) == 2
    assert attempts[0].validation_errors
    assert "B_suggestions" in attempts[0].validation_errors[0]
    assert attempts[1].validation_errors == []


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_stage_a1_falls_back_after_retries_exhausted() -> None:
    client = MockLLMClient(scripted_responses=[
        "garbage1", "garbage2", "garbage3",
    ])
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_actions(_make_pre_action_state())
    assert fb is True
    assert len(attempts) == 3
    # Fallback for Stage A1 is just a noop.
    assert [b.type for b in blocks] == ["noop"]


def test_stage_a2_fallback_keeps_all_pending_analogues() -> None:
    """When new_analogs is non-empty, the Stage A2 fallback keeps all."""
    pending = (AnalogueRecord(
        seed_smiles="CCO", analogue_smiles="A1", reasyn_score=0.9, num_steps=2,
    ),)
    client = MockLLMClient(scripted_responses=["garbage1", "garbage2"])
    advisor = LLMAdvisor(llm=client, max_retries=2, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_review_analogs(
        _make_pre_review_analogs_state(pending=pending)
    )
    assert fb is True
    # Expect one review_analogs (all keep).
    assert len(blocks) == 1
    assert isinstance(blocks[0], ReviewAnalogsBlock)
    assert blocks[0].decisions == {"A1": "keep"}


def test_stage_b_falls_back_to_all_ok() -> None:
    client = MockLLMClient(scripted_responses=["bad"] * 3)
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_review_suggestions(_make_post_state())
    assert fb is True
    assert len(attempts) == 3
    # Fallback is one review_bo with all 'ok'.
    rb = next(b for b in blocks if b.type == "review_bo")
    assert rb.decisions == {"CCC": "ok"}


def test_stage_b_fallback_with_no_bo_suggestions_is_empty() -> None:
    """When BO produced no suggestions, Stage B fallback is an empty list."""
    state = _make_post_state()
    state = state.__class__(
        **{
            **{k: v for k, v in state.__dict__.items() if k != "bo_suggestions"},
            "bo_suggestions": (),
        }
    )
    client = MockLLMClient(scripted_responses=["bad"])
    advisor = LLMAdvisor(llm=client, max_retries=1, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_review_suggestions(state)
    assert fb is True
    assert blocks == []


# ---------------------------------------------------------------------------
# Transport errors
# ---------------------------------------------------------------------------


class _FailingClient(MockLLMClient):
    """Mock that raises on chat."""

    def chat(self, system, user, *, json_mode=True):
        raise ConnectionError("network down")


def test_transport_errors_become_retryable_validation_error() -> None:
    client = _FailingClient(scripted_blocks=[])  # never reached
    advisor = LLMAdvisor(llm=client, max_retries=2, use_rdkit=False)
    blocks, attempts, fb = advisor.decide_actions(_make_pre_action_state())
    assert fb is True
    assert len(attempts) == 2
    assert all("transport" in a.validation_errors[0] for a in attempts)


# ---------------------------------------------------------------------------
# Cross-stage isolation
# ---------------------------------------------------------------------------


def test_stage_a1_and_stage_b_dont_share_previous_errors() -> None:
    """A failing Stage A1 must not feed its errors into Stage B's prompt."""
    a_client = MockLLMClient(scripted_responses=["bad1", "bad2", "bad3"])
    a_advisor = LLMAdvisor(llm=a_client, max_retries=3, use_rdkit=False)
    a_blocks, a_attempts, a_fb = a_advisor.decide_actions(_make_pre_action_state())
    assert a_fb is True

    # Now a fresh Stage B call: previous_errors should be empty.
    captured = []

    class CapturingClient(MockLLMClient):
        def chat(self, system, user, *, json_mode=True):
            captured.append((system[:30], user))
            return _serialize_blocks([ReviewBOBlock(rationale="ok", decisions={"CCC": "ok"})])

    client = CapturingClient(scripted_blocks=[])
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=False)
    b_blocks, b_attempts, b_fb = advisor.decide_review_suggestions(_make_post_state())
    assert b_fb is False
    assert "Previous errors" not in captured[0][1]


# ---------------------------------------------------------------------------
# max_retries validation
# ---------------------------------------------------------------------------


def test_advisor_rejects_max_retries_zero() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        LLMAdvisor(llm=MockLLMClient(scripted_blocks=[]), max_retries=0)


# ---------------------------------------------------------------------------
# pool_min_size: Noop retry with refill message
# ---------------------------------------------------------------------------


def test_noop_retries_with_pool_refill_message() -> None:
    """When the LLM emits Noop and the pool is below min, the advisor
    retries and the second-attempt user prompt contains the refill
    instruction.
    """
    # 1st attempt: Noop (will be rejected by validate_semantics).
    # 2nd attempt: Propose (refills the pool).
    client = MockLLMClient(scripted_blocks=[
        [NoopBlock(rationale="do nothing")],
        [ProposeBlock(rationale="refill", smiles=["c1ccccc1N"])],
    ])
    advisor = LLMAdvisor(llm=client, max_retries=3, use_rdkit=True)
    state = PreActionState(
        round_idx=0, n_total_rounds=2, pdf_context="",
        objective_legend=[], pool=("CCO",), pool_size_cap=10,
        history=(("CCO", -1.0),),
        pool_min_size=3,
        best="CCO",
        stagnation_counter=0,
        previous_errors=(), attempt=1,
    )
    blocks, attempts, fallback = advisor.decide_actions(state)
    # Two attempts (one rejection, one success).
    assert len(attempts) == 2
    # No fallback used.
    assert fallback is False
    # Final blocks: the successful propose.
    assert len(blocks) == 1
    assert isinstance(blocks[0], ProposeBlock)
    # The first attempt recorded the validation error.
    assert "pool has 1 SMILES" in attempts[0].validation_errors[0]
    assert "MUST emit" in attempts[0].validation_errors[0]


# ---------------------------------------------------------------------------
# External guidance (LLM_GUIDANCE / --llm-guide)
# ---------------------------------------------------------------------------


class _CapturingClient(MockLLMClient):
    """A MockLLMClient that records the (system, user) tuple for every
    call. Use to assert what the advisor actually sent to the LLM."""

    def __init__(self, return_block, **kwargs):
        super().__init__(**kwargs)
        self._return_block = return_block
        self.calls: list = []

    def chat(self, system, user, *, json_mode=True):
        self.calls.append((system, user))
        return _serialize_blocks([self._return_block])


def test_advisor_decide_actions_passes_guidance_to_system_prompt() -> None:
    """When ``LLMAdvisor.guidance`` is non-empty, the system prompt
    sent to the LLM contains the GUIDANCE block."""
    client = _CapturingClient(
        return_block=ProposeBlock(rationale="r", smiles=["c1ccccc1N"]),
    )
    advisor = LLMAdvisor(
        llm=client, max_retries=1, use_rdkit=False,
        guidance="Use analog heavily",
    )
    state = _make_pre_action_state()
    blocks, attempts, fallback = advisor.decide_actions(state)
    assert fallback is False
    assert len(client.calls) == 1
    sys_prompt = client.calls[0][0]
    assert "## EXTERNAL GUIDANCE" in sys_prompt
    assert "Use analog heavily" in sys_prompt


def test_advisor_decide_review_analogs_passes_guidance_to_system_prompt() -> None:
    client = _CapturingClient(
        return_block=ReviewAnalogsBlock(
            rationale="r", decisions={"A1": "keep"},
        ),
    )
    advisor = LLMAdvisor(
        llm=client, max_retries=1, use_rdkit=False,
        guidance="Be strict on analogues",
    )
    state = _make_pre_review_analogs_state(
        pending=(AnalogueRecord(
            seed_smiles="CCO", analogue_smiles="A1",
            reasyn_score=0.9, num_steps=2,
        ),),
    )
    blocks, attempts, fallback = advisor.decide_review_analogs(state)
    assert fallback is False
    assert len(client.calls) == 1
    sys_prompt = client.calls[0][0]
    assert "## EXTERNAL GUIDANCE" in sys_prompt
    assert "Be strict on analogues" in sys_prompt


def test_advisor_decide_review_suggestions_passes_guidance_to_system_prompt() -> None:
    client = _CapturingClient(
        return_block=ReviewBOBlock(
            rationale="r", decisions={"CCC": "ok"},
        ),
    )
    advisor = LLMAdvisor(
        llm=client, max_retries=1, use_rdkit=False,
        guidance="Don't override BO",
    )
    blocks, attempts, fallback = advisor.decide_review_suggestions(_make_post_state())
    assert fallback is False
    assert len(client.calls) == 1
    sys_prompt = client.calls[0][0]
    assert "## EXTERNAL GUIDANCE" in sys_prompt
    assert "Don't override BO" in sys_prompt


def test_advisor_no_guidance_keeps_baseline_system_prompt() -> None:
    """When ``guidance=""`` (the default), the system prompt is
    byte-identical to the legacy non-guidance rendering."""
    client = _CapturingClient(
        return_block=ProposeBlock(rationale="r", smiles=["c1ccccc1N"]),
    )
    advisor = LLMAdvisor(llm=client, max_retries=1, use_rdkit=False)
    state = _make_pre_action_state()
    blocks, attempts, fallback = advisor.decide_actions(state)
    assert fallback is False
    sys_prompt = client.calls[0][0]
    assert "## EXTERNAL GUIDANCE" not in sys_prompt
    # The system prompt still starts with the format header and stage A1
    # text — i.e. the no-guidance path is identical to before.
    assert "STAGE A1" in sys_prompt


def test_advisor_whitespace_only_guidance_is_treated_as_empty() -> None:
    client = _CapturingClient(
        return_block=ProposeBlock(rationale="r", smiles=["c1ccccc1N"]),
    )
    advisor = LLMAdvisor(
        llm=client, max_retries=1, use_rdkit=False,
        guidance="   \n  \t  ",
    )
    state = _make_pre_action_state()
    blocks, attempts, fallback = advisor.decide_actions(state)
    assert fallback is False
    sys_prompt = client.calls[0][0]
    assert "## EXTERNAL GUIDANCE" not in sys_prompt


def test_advisor_default_guidance_is_empty() -> None:
    """The LLMAdvisor default for ``guidance`` is the empty string."""
    assert LLMAdvisor.__dataclass_fields__["guidance"].default == ""


def test_advisor_guidance_applied_to_every_attempt() -> None:
    """When retries happen, the guidance text is re-injected into
    every attempt's system prompt (it's at the advisor level, not
    attempt level)."""
    # 1st attempt: invalid (garbage). 2nd attempt: valid.
    client = MockLLMClient(scripted_responses=[
        "not json",
        _serialize_blocks(
            [ProposeBlock(rationale="r", smiles=["c1ccccc1N"])]
        ),
    ])
    captured: list = []

    class _RecClient(MockLLMClient):
        def chat(self, system, user, *, json_mode=True):
            captured.append(system)
            return super().chat(system=system, user=user, json_mode=json_mode)

    rec = _RecClient(
        scripted_responses=[
            "not json",
            _serialize_blocks(
                [ProposeBlock(rationale="r", smiles=["c1ccccc1N"])]
            ),
        ],
    )
    advisor = LLMAdvisor(llm=rec, max_retries=3, use_rdkit=False,
                         guidance="Use analog heavily")
    state = _make_pre_action_state()
    blocks, attempts, fallback = advisor.decide_actions(state)
    assert fallback is False
    assert len(captured) == 2
    for s in captured:
        assert "## EXTERNAL GUIDANCE" in s
        assert "Use analog heavily" in s
