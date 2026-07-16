"""scripts/smoke/run_ldm_smoke.py — minimal LDM-only smoke test.

Boots an Orchestrator with a MockLLM, runs 5 iterations, verifies decisions
are applied and logged. Does NOT call Absolut! or run full BO.

Usage (from repo root):
    python scripts/smoke/run_ldm_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bo.ldm import DSLConfig, Orchestrator, SearchSpaceAtom, BiasAtom
from bo.ldm.llm.client import LLMClient
from bo.ldm.orchestrator.status import OrchestratorStatus


class MockLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list = []
        self._idx = 0

    def call(self, prompt, temperature=0.25, timeout_s=30):
        self.calls.append(1)
        if self._idx < len(self.responses):
            r = self.responses[self._idx]
            self._idx += 1
            return r
        return self.responses[-1]


def main() -> None:
    # Pre-canned LLM responses. We make all TRs small so validation succeeds.
    # The orchestrator may retry on failure; the mock cycles through this list.
    responses = [
        "{}",  # 0: pass
        '{"update_trust_region": "HammingDistanceTo(\'ARYYGSYWYFD\', 1)"}',  # 1: TR (Hamming 1 ~ 200 seqs)
        '{"update_bias": "MaxCysteine(1)"}',  # 2: bias
        '{"update_trust_region": "HammingDistanceTo(\'VRGYYSDWYMD\', 1)", "update_bias": "MaxHydrophobicRun(4)"}',  # 3: both
        "{}",  # 4: pass
    ]
    client = MockLLM(responses)
    config = DSLConfig(
        llm_init_enabled=True,
        llm_loop_enabled=True,
        llm_call_max_retries=2,
        sample_timeout_s=2.0,
    )

    with tempfile.TemporaryDirectory(prefix="ldm_smoke_") as td:
        log_path = Path(td) / "decisions.json"
        orch = Orchestrator(config=config, llm_client=client, decision_log_path=log_path)

        print(f"Running 5 iterations, log={log_path}")
        for i in range(5):
            status = OrchestratorStatus(
                iteration=i,
                antigen_id="1ADQ_A",
                antigen_seed=42,
                iter_seed=i,
                best_value=-73.6 - i,
                best_sequence=[0] * 11,
                full_history=[(f"ARYY{i}GSYWYFD", -73.6 - j, j) for j in range(i + 1)],
            )
            d = orch.step(status)
            tr = type(d.search_dsl).__name__ if d.search_dsl else "None"
            bias = type(d.bias_dsl).__name__ if d.bias_dsl else "None"
            print(f"  iter {i}: source={d.source}, search={tr}, bias={bias}, fallback={d.fallback_used}")

        # Verify state changes (after iter 3, both should be set).
        # Some iterations may have retried; we look at the final state.
        # Both should eventually be set after successful iterations.
        assert isinstance(orch.current_search_dsl, (SearchSpaceAtom, type(None)))
        assert isinstance(orch.current_bias_dsl, (BiasAtom, type(None)))
        # At least one of the 5 iterations set a search_dsl.
        # Check the log for evidence.
        import json
        data = json.loads(log_path.read_text())
        any_search_applied = any(
            e.get("field_outcomes", {}).get("update_trust_region", {}).get("applied")
            for e in data["decisions"]
        )
        any_bias_applied = any(
            e.get("field_outcomes", {}).get("update_bias", {}).get("applied")
            for e in data["decisions"]
        )
        assert any_search_applied, "At least one update_trust_region should be applied"
        assert any_bias_applied, "At least one update_bias should be applied"
        assert len(data["decisions"]) == 5
        print(f"Log written with {len(data['decisions'])} decisions "
              f"({len(client.calls)} LLM calls including retries)")

    print("LDM SMOKE OK")


if __name__ == "__main__":
    main()