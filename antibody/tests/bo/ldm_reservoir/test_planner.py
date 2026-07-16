from __future__ import annotations

from bo.ldm.llm.client import LLMClient
from bo.ldm.dsl.search_space import NeighborSampling
from bo.ldm_reservoir import ReservoirLDMConfig, ReservoirPlanner


class FakeClient(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response

    def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
        self.last_prompt = prompt
        return self.response


def test_planner_parses_five_strategy_json():
    response = """
    {
      "rationale": "diverse anchors",
      "strategies": [
        {"name": "s0", "trust_region": "NeighborSampling('AAAAAAAAAAA', radius=0, budget=1)"},
        {"name": "s1", "trust_region": "NeighborSampling('CAAAAAAAAAA', radius=0, budget=1)"},
        {"name": "s2", "trust_region": "NeighborSampling('DAAAAAAAAAA', radius=0, budget=1)"},
        {"name": "s3", "trust_region": "NeighborSampling('EAAAAAAAAAA', radius=0, budget=1)"},
        {"name": "s4", "trust_region": "NeighborSampling('FAAAAAAAAAA', radius=0, budget=1)"}
      ],
      "update_bias": "MaxCysteine(1) + MaxHydrophobicRun(4) + NoNGlycosylation()"
    }
    """
    planner = ReservoirPlanner(FakeClient(response), ReservoirLDMConfig(n_strategies=5))
    plan = planner.propose({"history": [], "best_sequence": "AAAAAAAAAAA"})

    assert len(plan.strategies) == 5
    assert all(isinstance(s, NeighborSampling) for s in plan.strategies)
    assert plan.bias_dsl is not None
    assert plan.rationale == "diverse anchors"



def test_independent_parallel_planner_makes_five_calls():
    import threading

    responses = [
        '{"rationale":"z0","trust_region":"NeighborSampling(\'AAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z1","trust_region":"NeighborSampling(\'CAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z2","trust_region":"NeighborSampling(\'DAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z3","trust_region":"NeighborSampling(\'EAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z4","trust_region":"NeighborSampling(\'FAAAAAAAAAA\', radius=0, budget=1)"}',
    ]

    class ManyResponseClient(LLMClient):
        def __init__(self):
            self.calls = 0
            self._lock = threading.Lock()

        def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
            with self._lock:
                idx = self.calls
                self.calls += 1
            return responses[idx]

    client = ManyResponseClient()
    planner = ReservoirPlanner(client, ReservoirLDMConfig(n_strategies=5, max_retries=1))
    plan = planner.propose_independent({"history": [], "best_sequence": "AAAAAAAAAAA"}, n=5)

    assert client.calls == 5
    assert len(plan.strategies) == 5
    assert all(isinstance(s, NeighborSampling) for s in plan.strategies)
    assert '"mode": "independent_parallel"' in plan.raw


def test_n_choices_planner_uses_one_request_for_five_choices():
    responses = [
        '{"rationale":"z0","trust_region":"NeighborSampling(\'AAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z1","trust_region":"NeighborSampling(\'CAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z2","trust_region":"NeighborSampling(\'DAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z3","trust_region":"NeighborSampling(\'EAAAAAAAAAA\', radius=0, budget=1)"}',
        '{"rationale":"z4","trust_region":"NeighborSampling(\'FAAAAAAAAAA\', radius=0, budget=1)"}',
    ]

    class ManyChoicesClient(LLMClient):
        def __init__(self):
            self.calls = 0
            self.completions = 0

        def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
            raise AssertionError("n_choices mode should use call_many, not call")

        def call_many(self, prompt: str, temperature: float, timeout_s: int, n: int) -> list[str]:
            self.calls += 1
            self.completions += n
            self.last_prompt = prompt
            return responses[:n]

    client = ManyChoicesClient()
    planner = ReservoirPlanner(client, ReservoirLDMConfig(n_strategies=5, max_retries=1))
    plan = planner.propose_n_choices({"history": [], "best_sequence": "AAAAAAAAAAA"}, n=5)

    assert client.calls == 1
    assert client.completions == 5
    assert len(plan.strategies) == 5
    assert all(isinstance(s, NeighborSampling) for s in plan.strategies)
    assert '"mode": "n_choices"' in plan.raw
    assert '"transport": "call_many"' in plan.raw

