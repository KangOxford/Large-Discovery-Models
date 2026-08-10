from __future__ import annotations

import pytest

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.dsl.search_space import LocalSearch
from tasks.antibody.core.ldm.llm.client import LLMClient
from tasks.antibody.core.ldm.llm.function import LLMFunction, ReviewFunction


class ScriptedClient(LLMClient):
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.prompts: list[str] = []

    def call(self, prompt, temperature, timeout_s):
        self.prompts.append(prompt)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class IntegerFunction(LLMFunction):
    def construct_prompt(self, value):
        suffix = f" Previous error: {self.last_error}" if self.last_error else ""
        return f"Return {value}.{suffix}"

    def parse_response(self, raw, value):
        return int(raw)

    def fallback(self, value):
        return -value


def test_base_methods_define_required_subclass_contract():
    function = LLMFunction(ScriptedClient([]))
    with pytest.raises(NotImplementedError):
        function.construct_prompt()
    with pytest.raises(NotImplementedError):
        function.parse_response("")
    with pytest.raises(NotImplementedError, match="has no fallback"):
        function.fallback()


def test_retry_recovers_from_transport_and_parse_errors():
    client = ScriptedClient([RuntimeError("offline"), "not-an-int", "12"])
    function = IntegerFunction(client, max_retries=3, temperature=0.4, timeout_s=7)

    assert function(12) == 12
    assert function.fallback_used is False
    assert function.previous_attempts == [
        (None, "offline"),
        ("not-an-int", "invalid literal for int() with base 10: 'not-an-int'"),
    ]
    assert "offline" in client.prompts[1]
    assert "invalid literal" in client.prompts[2]


def test_retry_exhaustion_uses_fallback_and_resets_between_calls():
    client = ScriptedClient(["bad", "still bad", "8"])
    function = IntegerFunction(client, max_retries=2)

    assert function(4) == -4
    assert function.fallback_used is True
    assert function.last_error is not None

    assert function(8) == 8
    assert function.fallback_used is False
    assert function.previous_attempts == []


@pytest.fixture
def review_config():
    return DSLConfig(max_retries=2, llm_temperature=0.3, llm_call_timeout_s=11)


def test_review_take_accepts_fence_scalar_id_and_rationale(review_config):
    function = ReviewFunction(ScriptedClient([]), review_config)

    result = function.parse_response(
        '```json\n{"action":"take","id":1,"rationale":"best score"}\n```',
        num_review=3,
        remaining_slots=1,
    )

    assert result == ("take", "best score", [1])


@pytest.mark.parametrize(
    ("raw", "kwargs", "message"),
    [
        ('{"action":"take"}', {}, "take requires ids"),
        ('{"action":"take","ids":[3]}', {"num_review": 3}, "out of range"),
        ('{"action":"take","ids":[0,1]}', {"remaining_slots": 1}, "only 1 slot"),
        ('{"action":"search"}', {}, "search requires update_trust_region"),
        ('{"action":"wait"}', {}, "Unknown action"),
    ],
)
def test_review_rejects_invalid_decisions(review_config, raw, kwargs, message):
    function = ReviewFunction(ScriptedClient([]), review_config)
    with pytest.raises(ValueError, match=message):
        function.parse_response(raw, **kwargs)


def test_review_search_returns_validated_atom(review_config):
    function = ReviewFunction(ScriptedClient([]), review_config, acq_name="ucb")
    raw = (
        '{"action":"search","rationale":"explore",'
        '"update_trust_region":"LocalSearch(\\"ARDYGNYWYFD\\", radius=1, restart=1, steps=2)"}'
    )

    action, rationale, atom = function.parse_response(raw, remaining_budget=3)

    assert action == "search"
    assert rationale == "explore"
    assert isinstance(atom, LocalSearch)
    assert atom.budget == 3


def test_review_search_enforces_remaining_budget(review_config):
    function = ReviewFunction(ScriptedClient([]), review_config)
    raw = (
        '{"action":"search",'
        '"update_trust_region":"LocalSearch(\\"ARDYGNYWYFD\\", restart=2, steps=3)"}'
    )

    with pytest.raises(ValueError, match="exceeds remaining 7"):
        function.parse_response(raw, remaining_budget=7)


def test_review_call_retries_with_error_feedback(review_config):
    client = ScriptedClient([
        '{"action":"take","ids":[9]}',
        '{"action":"take","ids":[1]}',
    ])
    function = ReviewFunction(client, review_config)

    result = function(review_text="Candidates", num_review=2, remaining_slots=1)

    assert result == ("take", None, [1])
    assert "ERROR" not in client.prompts[0]
    assert "out of range" in client.prompts[1]
    assert function.fallback_used is False


def test_review_fallback_takes_top_candidate(review_config):
    client = ScriptedClient(["not json", "still not json"])
    function = ReviewFunction(client, review_config)

    assert function(review_text="Candidates") == ("take", None, [0])
    assert function.fallback_used is True
