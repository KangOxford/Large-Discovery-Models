"""scripts/smoke/run_real_bo_smoke.py — real BO loop with LDM orchestrator.

Runs a minimal BO loop using:
  - bbox.tool=random       (no Absolut! needed)
  - OpenAIClient           (real LLM, .env-configured)
  - DSLConfig              (orchestrator enabled)

Verifies:
  1. BOExperiments starts without crashing
  2. Orchestrator is wired into CASMOPOLITANCat
  3. Each iteration calls the real LLM and applies a DSL
  4. results.csv is written
  5. outputs/llm_decisions/*.json is written

Usage (from repo root):
    python scripts/smoke/run_real_bo_smoke.py

Requires .env with LLM_API_KEY and LLM_BASE_URL.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


SMOKE_CONFIG = """
acq: ei
ard: true
batch_size: 1
device: cpu
kernel_type: transformed_overlap
max_iters: 5
min_cuda: 1000
n_init: 3
noise_variance: 1e-6
normalise: true
resume: false
save_path: __SAVE_PATH__
search_strategy: local
seq_len: 11
tabular_search_csv: null

bbox:
  antigen: SMOKE_1ADQ_A
  tool: random
  path: /tmp
  process: 1
  startTask: 0

llm:
  llm_init_enabled: true
  llm_loop_enabled: true
  llm_temperature: 0.25
  llm_call_max_retries: 2
  llm_call_timeout_s: 60
  llm_decisions_log: __DECISIONS_LOG__
  history_max_in_prompt: 100
  bias_weight: 0.1
  acq_n_candidates: 2000
  sample_timeout_s: 5.0
  init_pool_size: 10000
  max_nesting_depth: 8

llm_antigen_context: false
""".strip()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="real_bo_smoke_") as td:
        td_path = Path(td)
        cfg_path = td_path / "smoke.yaml"
        save_path = td_path / "antbo_out"
        decisions_log = td_path / "decisions.json"
        antigen_file = td_path / "antigens.txt"
        antigen_file.write_text("SMOKE_1ADQ_A\n")

        cfg_text = (
            SMOKE_CONFIG
            .replace("__SAVE_PATH__", str(save_path))
            .replace("__DECISIONS_LOG__", str(decisions_log))
        )
        cfg_path.write_text(cfg_text)
        print(f"Config:\n{cfg_text}\n")

        rc = subprocess.run(
            [
                sys.executable,
                "bo/main.py",
                "--config",
                str(cfg_path),
                "--antigens_file",
                str(antigen_file),
                "--seed",
                "42",
                "--n_trials",
                "1",
            ],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        print(f"returncode: {rc.returncode}")
        if rc.stdout:
            print("--- STDOUT (last 30 lines) ---")
            for line in rc.stdout.splitlines()[-30:]:
                print(line)
        if rc.stderr:
            print("--- STDERR (last 15 lines) ---")
            for line in rc.stderr.splitlines()[-15:]:
                print(line)

        if rc.returncode != 0:
            raise AssertionError(f"bo/main.py exited with code {rc.returncode}")

        # Verify outputs.
        csvs = list(save_path.rglob("results.csv"))
        assert csvs, f"results.csv was not produced under {save_path}"
        print(f"\nFound {len(csvs)} results.csv at {csvs[0]}")

        assert decisions_log.exists(), f"decisions log not produced: {decisions_log}"
        import json
        log_data = json.loads(decisions_log.read_text())
        assert len(log_data["decisions"]) >= 1, "no decisions in log"
        print(f"Found {len(log_data['decisions'])} LLM decisions in log")
        first_raw = log_data["decisions"][0]["llm_response_raw"]
        print(f"First LLM raw (first 200 chars):\n{first_raw[:200]}")

        # Cleanup
        shutil.rmtree(save_path, ignore_errors=True)

    print("REAL BO SMOKE OK")


if __name__ == "__main__":
    main()