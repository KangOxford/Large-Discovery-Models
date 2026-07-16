"""BO main-entry smoke test.

Validates that, after repository reorganisation, the `bo.main` module can:
  1. Be imported (with all LDM dependencies resolved through `bo.ldm.*`).
  2. Construct a `BOExperiments` from a self-contained minimal config.
  3. Run a single observe iteration under `bbox.tool=random` (no Absolut!).
  4. Persist `results.csv` to disk.

Usage (from repo root):
    python scripts/smoke/run_bo_smoke.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def preflight_imports() -> None:
    """Static check that the import chain resolves without touching Absolut!."""
    from bo.main import BOExperiments  # noqa: F401
    import bo.ldm  # noqa: F401
    from bo.ldm import (  # noqa: F401
        DSLConfig, SearchSpaceAtom, BiasAtom, Orchestrator,
        OrchestratorStatus, OrchestratorDecision, LLMClient,
    )
    from bo.ldm.antigen_context import validate_policy_json  # noqa: F401


def run_mini_experiment(save_path: str) -> None:
    """Drive bo/main.py with the smoke config for 1 trial, n_init=5, max_iters=3."""
    cfg_path = ROOT / "scripts" / "smoke" / "smoke_config.yaml"
    antigen_file = ROOT / "scripts" / "smoke" / "smoke_config.yaml"  # any path works
    # We need a one-line antigen file
    tmp_antigen = Path(tempfile.mkstemp(suffix=".txt", prefix="smoke_antigens_")[1])
    tmp_antigen.write_text("SMOKE_1ADQ_A\n")
    rc = subprocess.run(
        [
            sys.executable,
            "bo/main.py",
            "--config",
            str(cfg_path),
            "--antigens_file",
            str(tmp_antigen),
            "--seed",
            "42",
            "--n_trials",
            "1",
        ],
        cwd=str(ROOT),
        check=False,
    )
    tmp_antigen.unlink(missing_ok=True)
    if rc.returncode != 0:
        raise AssertionError(f"bo/main.py exited with code {rc.returncode}")


def main() -> None:
    preflight_imports()
    print("PRE-FLIGHT: imports OK")

    with tempfile.TemporaryDirectory(prefix="antbo_smoke_") as td:
        save_path = os.path.join(td, "smoke_results")
        # Patch save_path inside the smoke config via env-var or direct edit?
        # Simpler: pass via tempfile by writing a copy of the smoke config with
        # a unique save_path. To avoid that, we just rely on the hardcoded
        # save_path in smoke_config.yaml and clean it up afterwards.
        run_mini_experiment(save_path)

    # The smoke config writes to /tmp/antbo_smoke_outputs/BO_transformed_overlap/.
    run_root = Path("/tmp/antbo_smoke_outputs/BO_transformed_overlap")
    csvs = list(run_root.rglob("results.csv"))
    assert csvs, f"results.csv was not produced under {run_root}"
    print(f"SMOKE OK: produced {len(csvs)} results.csv at {csvs[0]}")

    # Clean up smoke artefacts so re-runs are idempotent.
    shutil.rmtree(run_root, ignore_errors=True)
    shutil.rmtree("/tmp/antbo_smoke_outputs", ignore_errors=True)


if __name__ == "__main__":
    main()