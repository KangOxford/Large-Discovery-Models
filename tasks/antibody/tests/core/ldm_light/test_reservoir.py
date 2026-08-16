from __future__ import annotations

import json
from types import SimpleNamespace

from tasks.antibody.core.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
)
from tasks.antibody.core.ldm_light.reservoir import (
    cap_strategy_budget,
    parse_policy_strategy,
    policy_representatives,
    propose_policy_reservoir,
)


class PolicyLLM:
    def call_many(self, prompt, temperature, timeout_s, n):
        self.prompt = prompt
        return [
            json.dumps({
                "rationale": f"policy {index}",
                "trust_region": (
                    "NeighborSampling('ARDYGNYWYFD', radius=2, "
                    f"mut_pr=0.{index + 2}, budget=99)"
                ),
                "update_bias": None,
            })
            for index in range(n)
        ]


def _args(**overrides):
    values = {
        "n_strategies": 5,
        "history_top_k": 10,
        "max_retries": 1,
        "planner_mode": "choices",
        "temperature": 0.25,
        "timeout_s": 10,
        "sample_timeout_s": 1.0,
        "fallback_random": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_planner_returns_five_independent_budget_capped_policies():
    llm = PolicyLLM()
    strategies, record = propose_policy_reservoir(
        llm=llm,
        antigen="SMOKE",
        rows=[],
        antigen_context=None,
        strategy_budget=7,
        args=_args(),
    )

    assert len(strategies) == 5
    assert all(strategy.atom.budget == 7 for strategy in strategies)
    assert record["n_fallback"] == 0
    assert "sample_one_antibody_search_policy" in llm.prompt
    assert "ADGHTKQNPRA" in llm.prompt


def test_missing_or_invalid_policies_are_filled_deterministically():
    class PartialLLM:
        def call_many(self, prompt, temperature, timeout_s, n):
            return ['{"bad": true}'] * n

    strategies, record = propose_policy_reservoir(
        llm=PartialLLM(),
        antigen="SMOKE",
        rows=[],
        antigen_context=None,
        strategy_budget=3,
        args=_args(),
    )

    assert len(strategies) == 5
    assert record["n_fallback"] == 5
    assert all(strategy.fallback for strategy in strategies)


def test_representatives_preserve_one_unique_candidate_per_policy():
    records = [
        [{"seq": [0], "ei": 3.0}, {"seq": [1], "ei": 2.0}],
        [{"seq": [0], "ei": 4.0}, {"seq": [2], "ei": 1.0}],
    ]

    representatives = policy_representatives(records, score_key="ei")

    assert [record["seq"] for record in representatives] == [[0], [2]]
    assert [record["strategy_index"] for record in representatives] == [0, 1]


def test_cap_local_or_sampling_budget():
    atom = NeighborSampling("ARDYGNYWYFD", radius=2, budget=100)
    assert cap_strategy_budget(atom, 9).budget == 9

    local = LocalSearch("ADGHTKQNPRA", restart=5, steps=100)
    assert cap_strategy_budget(local, 9).budget <= 9
    assert cap_strategy_budget(local, 1).budget == 1


def test_composite_policy_is_preserved_and_capped_as_one_strategy():
    raw = json.dumps({
        "rationale": "search globally and around the incumbent",
        "trust_region": (
            "LatinHyperCubeSampling(num=20) | "
            "NeighborSampling('ARDYGNYWYFD', radius=2, budget=20)"
        ),
        "update_bias": None,
    })

    strategy = parse_policy_strategy(raw, sample_timeout_s=1.0)
    capped = cap_strategy_budget(strategy.atom, 9)

    assert isinstance(strategy.atom, Or)
    assert isinstance(capped, Or)
    assert capped.budget == 9
    assert isinstance(capped.children[0], LatinHyperCubeSampling)
    assert isinstance(capped.children[1], NeighborSampling)


def test_composite_policy_with_unit_budget_degrades_to_one_valid_branch():
    atom = Or(
        LatinHyperCubeSampling(num=20),
        NeighborSampling("ARDYGNYWYFD", radius=2, budget=20),
    )

    capped = cap_strategy_budget(atom, 1)

    assert not isinstance(capped, Or)
    assert capped.budget == 1
