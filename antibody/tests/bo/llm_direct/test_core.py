from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import numpy as np

from bo.ldm.llm.client import LLMClient
from bo.llm_direct import (
    CountingLLMClient,
    parse_generated_sequences,
    propose_generated_many,
    select_scored_candidates,
)


VALID_SEQS = [
    "ADGHTKQNPRA",
    "QEGHSTKNPRA",
    "TQGHDKNPRAA",
    "ADKQGHTNPRA",
    "GHKQADTNPRS",
]


def make_args(**kwargs):
    defaults = dict(
        max_retries=1,
        temperature=0.0,
        timeout_s=10,
        history_top_k=5,
        fallback_random=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_parse_generated_sequences_accepts_json_list_only():
    raw = '["ADGHTKQNPRA", "QEGHSTKNPRA", "BAD"]'
    parsed = parse_generated_sequences(raw, seq_len=11, observed=set(), max_sequences=5)

    assert parsed == ["ADGHTKQNPRA", "QEGHSTKNPRA"]


def test_parse_generated_sequences_filters_observed_and_invalid():
    raw = '["ADGHTKQNPRA", "QEGHSTKNPRA", "FFFFFFFFFFF"]'
    parsed = parse_generated_sequences(raw, seq_len=11, observed={"ADGHTKQNPRA"}, max_sequences=5)

    assert parsed == ["QEGHSTKNPRA"]


def test_propose_generated_many_uses_one_call_many_for_m_samples():
    class ManyClient(LLMClient):
        def __init__(self):
            self.calls = 0

        def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
            raise AssertionError("propose_generated_many should use call_many")

        def call_many(self, prompt: str, temperature: float, timeout_s: int, n: int) -> list[str]:
            self.calls += 1
            self.last_prompt = prompt
            return [f'["{seq}"]' for seq in VALID_SEQS[:n]]

    base = ManyClient()
    llm = CountingLLMClient(base)
    candidates, decision = propose_generated_many(
        llm=llm,
        rng=None,
        antigen="SMOKE",
        seq_len=11,
        n=5,
        observed=set(),
        rows=[],
        antigen_context=None,
        args=make_args(),
        mode="LDM_gen_softmax",
    )

    assert base.calls == 1
    assert llm.total_calls == 1
    assert llm.total_completions == 5
    assert [item["sequence"] for item in candidates] == VALID_SEQS
    assert decision["n_requested"] == 5
    assert "candidate_pool" not in base.last_prompt


def test_counting_client_call_many_reuses_openai_request_kwargs():
    class OpenAIStyleInner:
        model = "test-model"

        def __init__(self):
            self._client = MagicMock()
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content="a")),
                MagicMock(message=MagicMock(content="b")),
            ]
            self._client.chat.completions.create.return_value = response

        def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
            raise AssertionError("call_many should use the OpenAI-compatible batch path")

        def make_chat_completion_kwargs(
            self,
            prompt: str,
            temperature: float,
            timeout_s: int,
            **overrides,
        ) -> dict:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "timeout": timeout_s,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            kwargs.update(overrides)
            return kwargs

    inner = OpenAIStyleInner()
    llm = CountingLLMClient(inner)

    assert llm.call_many("prompt", temperature=0.2, timeout_s=5, n=2) == ["a", "b"]
    inner._client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "prompt"}],
        temperature=0.2,
        timeout=5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        n=2,
    )


def test_select_scored_candidates_argmax():
    scored = [
        {"sequence": VALID_SEQS[0], "acquisition_score": 0.1},
        {"sequence": VALID_SEQS[1], "acquisition_score": 3.0},
        {"sequence": VALID_SEQS[2], "acquisition_score": 1.0},
    ]
    ids, probs = select_scored_candidates(
        scored,
        batch_size=1,
        selection="argmax",
        eta=1.0,
        rng=np.random.default_rng(0),
    )

    assert ids == [1]
    assert probs == [0.0, 1.0, 0.0]


def test_select_scored_candidates_softmax_eta():
    scored = [
        {"sequence": VALID_SEQS[0], "acquisition_score": 0.0},
        {"sequence": VALID_SEQS[1], "acquisition_score": 1.0},
    ]
    _, probs = select_scored_candidates(
        scored,
        batch_size=1,
        selection="softmax",
        eta=1.0,
        rng=np.random.default_rng(0),
    )
    expected = np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum()

    assert np.allclose(probs, expected)
    assert probs[1] > probs[0]
