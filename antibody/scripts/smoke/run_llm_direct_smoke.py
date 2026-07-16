#!/usr/bin/env python3
"""Smoke test for direct LLM generation baselines.

No external LLM or Absolut binary is required:
  - --mock_llm generates deterministic valid antibody strings
  - bbox.tool=random provides a cheap oracle

The smoke run uses 1 antigen, 1 seed, 2 LLM init observations, and 2
acquisition-guided iterations.
"""
from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_llm_direct_absolut import run_one


def _run_method(method: str, root: Path, config: dict) -> None:
    out_root = root / f"outputs_{method}"
    args = Namespace(
        method=method,
        config=str(root / "config.yaml"),
        antigens_file="",
        seed=123,
        n_trials=1,
        n_evals=4,
        batch_size=1,
        out_root=str(out_root),
        n_init=2,
        gen_m=5,
        softmax_eta=1.0,
        temperature=0.0,
        timeout_s=10,
        max_retries=1,
        history_top_k=3,
        gp_train_steps=30,
        acq_device="cpu",
        include_antigen_context=False,
        fallback_random=False,
        mock_llm=True,
    )

    run_dir = run_one(config, "SMOKE_ANTIGEN", 123, args)
    results_path = run_dir / "results.csv"
    decisions_path = run_dir / "llm_direct_decisions.jsonl"
    if not results_path.exists():
        raise RuntimeError(f"results.csv missing: {results_path}")
    if not decisions_path.exists():
        raise RuntimeError(f"decision log missing: {decisions_path}")

    df = pd.read_csv(results_path)
    if len(df) != 4:
        raise RuntimeError(f"Expected 4 evaluated rows, got {len(df)}")
    init_source = f"{method}_init"
    if df["Source"].tolist()[:2] != [init_source, init_source]:
        raise RuntimeError(f"First two rows should be LLM init: {df['Source'].tolist()}")
    if df["Source"].tolist()[2:] != [method, method]:
        raise RuntimeError(f"Last two rows should use acquisition: {df['Source'].tolist()}")
    if df["AcquisitionUsed"].tolist() != [False, False, True, True]:
        raise RuntimeError(f"Unexpected AcquisitionUsed flags: {df['AcquisitionUsed'].tolist()}")
    if df.iloc[2:]["AcquisitionScore"].isna().any():
        raise RuntimeError("Acquisition iterations have missing acquisition scores")

    lines = decisions_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 4:
        raise RuntimeError(f"Expected 4 decision rows, got {len(lines)}")

    print(f"LLM DIRECT SMOKE OK: {method}")
    print(f"run_dir={run_dir}")
    print(df[[
        "Index",
        "LastValue",
        "BestValue",
        "AcquisitionScore",
        "LastProtein",
        "Source",
        "AcquisitionUsed",
    ]].to_string(index=False))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llm_direct_smoke_") as td:
        root = Path(td)
        cfg_path = root / "config.yaml"
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
        _run_method("LDM_gen_softmax", root, config)
        _run_method("LDM_gen_argmax", root, config)


if __name__ == "__main__":
    main()
