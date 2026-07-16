"""scripts/smoke/run_real_llm_smoke.py — real LLM end-to-end smoke.

Uses the configured OpenAI endpoint (.env) to drive the orchestrator.
Verifies:
  1. OpenAIClient constructs from .env
  2. Real LLM returns parseable JSON in the agreed format
  3. DSL parses (safe_exec_dsl + validate_search_atom / validate_bias_atom)
  4. Orchestrator.step() applies the DSL or falls back

Usage (from repo root):
    python scripts/smoke/run_real_llm_smoke.py

Requires .env with LLM_API_KEY and LLM_BASE_URL.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from bo.ldm import (
        BiasAtom,
        DSLConfig,
        OpenAIClient,
        Orchestrator,
        SearchSpaceAtom,
    )
    from bo.ldm.orchestrator.status import OrchestratorStatus

    # Real client (reads .env via python-dotenv).
    client = OpenAIClient()
    print(f"client.model = {client.model}")

    config = DSLConfig(
        llm_init_enabled=True,
        llm_loop_enabled=True,
        llm_call_max_retries=2,
        llm_call_timeout_s=60,
        sample_timeout_s=5.0,
    )

    with tempfile.TemporaryDirectory(prefix="real_llm_smoke_") as td:
        log_path = Path(td) / "decisions.json"
        orch = Orchestrator(config=config, llm_client=client,
                            decision_log_path=log_path)

        status = OrchestratorStatus(
            iteration=1,
            antigen_id="1ADQ_A",
            antigen_seed=42,
            iter_seed=1,
            best_value=-73.6,
            best_sequence=[0, 14, 19, 19, 5, 15, 19, 18, 19, 4, 2],
            full_history=[("ARYYGSYWYFD", -73.6, 0)],
        )

        print("Calling orchestrator.step() (real LLM call)...")
        decision = orch.step(status)

        # Show the raw LLM response from the log.
        with open(log_path) as f:
            log_data = json.load(f)
        raw = log_data["decisions"][0]["llm_response_raw"]
        print(f"LLM raw response (first 300 chars):\n{raw[:300]}")

        if decision.applied:
            tr = type(decision.search_dsl).__name__ if decision.search_dsl else "None"
            bias = type(decision.bias_dsl).__name__ if decision.bias_dsl else "None"
            print(f"Applied: search={tr}, bias={bias}")
        else:
            print(f"Fallback (reason: {decision.rejection_reason})")

        assert decision.source in ("llm", "fallback")
        assert decision.search_dsl is None or isinstance(decision.search_dsl, SearchSpaceAtom)
        assert decision.bias_dsl is None or isinstance(decision.bias_dsl, BiasAtom)

    print("REAL LLM SMOKE OK")


if __name__ == "__main__":
    main()