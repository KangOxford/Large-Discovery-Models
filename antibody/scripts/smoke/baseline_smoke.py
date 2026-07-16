#!/usr/bin/env python3
"""Smoke test for the LLM-only baseline using the real LLM.

This intentionally uses bbox.tool=random so the test is fast and does not need
Absolut, but it does call the real OpenAI-compatible endpoint through
bo.ldm.OpenAIClient. It verifies:

  - .env / LLM endpoint works
  - reasoning/thinking env controls can be sent
  - the LLM returns parseable selected sequence+score JSON
  - results.csv and llm_only_decisions.jsonl are produced
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    out_root = Path(tempfile.mkdtemp(prefix="antbo_llm_only_real_llm_smoke_"))
    cfg_path = out_root / "smoke_config.yaml"
    antigen_file = out_root / "smoke_antigens.txt"

    config = {
        "seq_len": 11,
        "tabular_search_csv": None,
        "llm_antigen_context_timeout_s": 1,
        "bbox": {
            "antigen": "SMOKE_ANTIGEN",
            "tool": "random",
            "path": "/tmp",
            "process": 1,
            "startTask": 0,
        },
    }
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    antigen_file.write_text("SMOKE_ANTIGEN\n", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("LLM_ENABLE_THINKING", "true")
    env.setdefault("LLM_REASONING_EFFORT", "high")

    cmd = [
        sys.executable,
        str(ROOT / "bo" / "ldm" / "llm" / "LLM_baseline.py"),
        "--config",
        str(cfg_path),
        "--antigens_file",
        str(antigen_file),
        "--seed",
        "42",
        "--n_trials",
        "1",
        "--n_evals",
        "2",
        "--batch_size",
        "1",
        "--llm_pool_size",
        "20",
        "--out_root",
        str(out_root / "outputs"),
        "--timeout_s",
        "120",
        "--max_retries",
        "3",
        "--no-include_antigen_context",
        "--no-fallback_random",
    ]

    print("Running real-LLM smoke command:")
    print(" ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    result_files = sorted((out_root / "outputs").glob("**/results.csv"))
    decision_files = sorted((out_root / "outputs").glob("**/llm_only_decisions.jsonl"))
    if len(result_files) != 1:
        raise RuntimeError(f"Expected 1 results.csv, found {len(result_files)} under {out_root}")
    if len(decision_files) != 1:
        raise RuntimeError(f"Expected 1 decisions log, found {len(decision_files)} under {out_root}")

    df = pd.read_csv(result_files[0])
    required = {"Index", "LastValue", "BestValue", "LLMScore", "LastProtein", "BestProtein", "Source"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"results.csv missing columns: {sorted(missing)}")
    if len(df) != 2:
        raise RuntimeError(f"Expected 2 evaluated rows, got {len(df)}")
    if not (df["Source"] == "llm").all():
        raise RuntimeError(f"Expected all rows Source=llm, got {df['Source'].tolist()}")

    print(f"REAL LLM-ONLY SMOKE OK")
    print(f"results.csv: {result_files[0]}")
    print(f"decisions: {decision_files[0]}")
    print(f"temporary output root kept at: {out_root}")


if __name__ == "__main__":
    main()
