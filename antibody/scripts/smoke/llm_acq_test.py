#!/usr/bin/env python3
"""Smoke test for the LLM + LDM parallel acquisition baseline.

This test avoids external services:
  - fake LLM responses for both warmup selection and LDM DSL updates
  - bbox.tool=random instead of Absolut

It verifies that the new baseline reaches the GP + execute_atoms + acquisition argmax
path after the warmup evaluations.
"""
from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bo.ldm_light.ldm_acq as baseline


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
        self.calls.append(prompt)
        if '"candidate_pool"' in prompt:
            return json.dumps({"selected": [{"id": 0, "sequence": "", "score": 1.0}]})
        return json.dumps({
            "rationale": "smoke test LDM search",
            "update_trust_region": "LatinHyperCubeSampling(num=8)",
        })

    def close(self) -> None:
        pass


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llm_acq_smoke_") as td:
        root = Path(td)
        cfg_path = root / "config.yaml"
        antigen_path = root / "antigens.txt"
        out_root = root / "outputs"
        config = {
            "seq_len": 11,
            "tabular_search_csv": None,
            "bbox": {
                "antigen": "SMOKE_ANTIGEN",
                "tool": "random",
                "path": "/tmp",
                "process": 1,
                "startTask": 0,
            },
        }
        cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        antigen_path.write_text("SMOKE_ANTIGEN\n", encoding="utf-8")

        fake_llm = FakeLLM()
        baseline.make_llm_client = lambda: fake_llm
        args = Namespace(
            config=str(cfg_path),
            antigens_file=str(antigen_path),
            seed=123,
            n_trials=1,
            n_evals=4,
            batch_size=1,
            out_root=str(out_root),
            temperature=0.0,
            timeout_s=10,
            max_retries=1,
            history_top_k=3,
            parallel_budget=8,
            n_init=3,
            acq="ei",
            acq_beta=1.0,
            include_antigen_context=False,
            fallback_random=False,
        )

        run_dir = baseline.run_one(config, "SMOKE_ANTIGEN", 123, args)
        results_path = run_dir / "results.csv"
        decisions_path = run_dir / "llm_acq_decisions.jsonl"
        ldm_log_path = run_dir / "ldm_parallel_decisions.json"

        if not results_path.exists():
            raise RuntimeError(f"results.csv missing: {results_path}")
        if not decisions_path.exists():
            raise RuntimeError(f"decision log missing: {decisions_path}")
        if not ldm_log_path.exists():
            raise RuntimeError(f"LDM decision log missing: {ldm_log_path}")

        df = pd.read_csv(results_path)
        if len(df) != 4:
            raise RuntimeError(f"Expected 4 evaluated rows, got {len(df)}")
        if "AcquisitionScore" not in df.columns:
            raise RuntimeError("results.csv missing AcquisitionScore column")
        expected_source = f"ldm_parallel_{args.acq}_argmax"
        if df.iloc[-1]["Source"] != expected_source:
            raise RuntimeError(f"Last row did not use LDM parallel {args.acq.upper()}: {df['Source'].tolist()}")
        if pd.isna(df.iloc[-1]["AcquisitionScore"]):
            raise RuntimeError("Last row has no acquisition score")

        lines = decisions_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) != 4:
            raise RuntimeError(f"Expected 4 decision rows, got {len(lines)}")
        last = json.loads(lines[-1])
        acquisition = last["acquisition"]
        if acquisition["used"] is not True:
            raise RuntimeError(f"Acquisition path was not used: {acquisition}")
        expected_executor = "bo.ldm.acquisition.parallel_search.execute_atoms"
        if acquisition["parallel_executor"] != expected_executor:
            raise RuntimeError(f"Wrong executor: {acquisition['parallel_executor']}")
        if not acquisition["selected_candidates"]:
            raise RuntimeError("No selected candidates recorded")

        print("LLM ACQ SMOKE OK")
        print(f"run_dir={run_dir}")
        print(df[[
            "Index",
            "LastValue",
            "BestValue",
            "AcquisitionScore",
            "LastProtein",
            "Source",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
