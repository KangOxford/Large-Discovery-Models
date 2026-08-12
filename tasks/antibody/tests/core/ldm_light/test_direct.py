from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from tasks.antibody.core.ldm_light.direct import (
    parse_direct_sequences,
    propose_direct_batch,
)


VALID = [
    "ADGHTKQNPRA",
    "QEGHSTKNPRA",
    "TQGHDKNPRAA",
    "ADKQGHTNPRA",
    "GHKQADTNPRS",
]


class ManyLLM:
    def __init__(self):
        self.calls = 0

    def call_many(self, prompt, temperature, timeout_s, n):
        self.calls += 1
        self.prompt = prompt
        return [json.dumps([sequence]) for sequence in VALID[:n]]


def _args(**overrides):
    values = {
        "max_retries": 1,
        "temperature": 0.3,
        "timeout_s": 10,
        "history_top_k": 5,
        "fallback_random": False,
        "planner_mode": "choices",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parser_requires_direct_valid_novel_sequences():
    raw = '<think>hidden</think>\n["ADGHTKQNPRA", "BAD", "FFFFFFFFFFF"]'

    assert parse_direct_sequences(
        raw,
        seq_len=11,
        observed=set(),
        max_sequences=5,
    ) == ["ADGHTKQNPRA"]


def test_independent_generation_uses_one_multi_choice_request():
    llm = ManyLLM()
    candidates, record = propose_direct_batch(
        llm=llm,
        rng=random.Random(2),
        antigen="SMOKE",
        seq_len=11,
        n=5,
        observed=set(),
        rows=[],
        antigen_context=None,
        args=_args(),
        independent=True,
    )

    assert llm.calls == 1
    assert [candidate["sequence"] for candidate in candidates] == VALID
    assert record["generation_mode"] == "independent_choices"
    assert "candidate_pool" not in llm.prompt
    assert '"GGGGGGGGGGG"' in llm.prompt


def test_independent_transport_uses_separate_calls():
    class SequentialLLM:
        def __init__(self):
            self.index = 0

        def call(self, prompt, temperature, timeout_s):
            sequence = VALID[self.index]
            self.index += 1
            return json.dumps([sequence])

    candidates, record = propose_direct_batch(
        llm=SequentialLLM(),
        rng=random.Random(2),
        antigen="SMOKE",
        seq_len=11,
        n=3,
        observed=set(),
        rows=[],
        antigen_context=None,
        args=_args(planner_mode="independent"),
        independent=True,
    )

    assert [candidate["sequence"] for candidate in candidates] == VALID[:3]
    assert record["generation_mode"] == "independent_calls"


def test_generation_reports_candidate_admission_rejections():
    class RejectingLLM:
        def call_many(self, prompt, temperature, timeout_s, n):
            return [json.dumps(["GYYGYGYGYGY"])] * n

    with pytest.raises(RuntimeError) as exc_info:
        propose_direct_batch(
            llm=RejectingLLM(),
            rng=random.Random(2),
            antigen="SMOKE",
            seq_len=11,
            n=5,
            observed=set(),
            rows=[],
            antigen_context=None,
            args=_args(),
            independent=True,
        )

    payload = json.loads(str(exc_info.value))
    assert len(payload) == 5
    assert payload[0]["error"] == "no candidates passed admission"
    assert payload[0]["rejections"] == [
        {
            "item_index": 0,
            "sequence": "GYYGYGYGYGY",
            "reasons": ["max_aromatic_FWY"],
        }
    ]
    assert payload[0]["raw_response"] == '["GYYGYGYGYGY"]'
