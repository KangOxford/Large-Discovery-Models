import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

from scripts import run_case2_m1_experiment as runner
from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_case2_m1_experiment.py"


def run_cli(tmp_path, method, *extra):
    out_dir = tmp_path / method
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--mock",
        "--method",
        method,
        "--budget",
        "8",
        "--init-size",
        "3",
        "--gp-fit-itersteps",
        "5",
        "--trajectory-dir",
        str(out_dir),
        *extra,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return out_dir


def test_cli_mock_m1_runs(tmp_path):
    out_dir = run_cli(tmp_path, "m1_direct_llm_sir", "--m1-k-direct-llm", "16")
    assert (out_dir / "summary.json").exists()


def test_cli_mock_m1_stratified_runs(tmp_path):
    out_dir = run_cli(
        tmp_path,
        "m1_stratified_direct_llm_sir",
        "--m1-k-direct-llm",
        "16",
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["method"] == "m1_stratified_direct_llm_sir"
    assert summary["llm_call_count"] > 0


def test_cli_mock_m1_oversample_runs(tmp_path):
    out_dir = run_cli(
        tmp_path,
        "m1_stratified_direct_llm_oversample_sir",
        "--m1-k-direct-llm",
        "16",
        "--max-candidates-per-round",
        "8",
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["method"] == "m1_stratified_direct_llm_oversample_sir"


def test_cli_mock_m1_one_step_runs(tmp_path):
    out_dir = run_cli(tmp_path, "m1_llm_one_step", "--m1-k-direct-llm", "16")
    first = json.loads((out_dir / "rounds.jsonl").read_text().splitlines()[0])
    assert first["method"] == "m1_llm_one_step"
    assert len(first["candidates"]) == 1


def test_cli_mock_m1_seed_analog_runs(tmp_path):
    out_dir = run_cli(
        tmp_path,
        "m1_llm_seed_analog_oversample_sir",
        "--m1-analog-n-llm-seeds",
        "2",
        "--m1-analog-k-total",
        "8",
        "--max-candidates-per-round",
        "4",
    )
    first = json.loads((out_dir / "rounds.jsonl").read_text().splitlines()[0])
    assert first["method"] == "m1_llm_seed_analog_oversample_sir"
    assert first["pool_maintenance"]["maintained_candidate_count"] <= 4


def test_cli_rejects_invalid_method(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--mock", "--method", "bad", "--trajectory-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode != 0


def test_trajectory_dir_defaults_to_output_dir(tmp_path):
    out_dir = tmp_path / "result"
    args = runner.parse_args([
        "--method",
        "m1_direct_llm_sir",
        "--output-dir",
        str(out_dir),
    ])

    runner.apply_output_defaults(args)

    assert args.output_dir == str(out_dir)
    assert args.trajectory_dir == str(out_dir)


def test_trajectory_dir_can_override_output_dir(tmp_path):
    out_dir = tmp_path / "result"
    trace_dir = tmp_path / "trace"
    args = runner.parse_args([
        "--method",
        "m1_direct_llm_sir",
        "--output-dir",
        str(out_dir),
        "--trajectory-dir",
        str(trace_dir),
    ])

    runner.apply_output_defaults(args)

    assert args.output_dir == str(out_dir)
    assert args.trajectory_dir == str(trace_dir)


def test_output_dir_has_method_seed_default():
    args = runner.parse_args([
        "--method",
        "m1_direct_llm_sir",
        "--seed",
        "7",
    ])

    runner.apply_output_defaults(args)

    assert Path(args.output_dir).as_posix().endswith(
        "output/case2/m1_direct_llm_sir_seed=7"
    )
    assert args.trajectory_dir == args.output_dir


def test_nn_model_path_defaults_to_g12d_model():
    args = runner.parse_args([
        "--method",
        "m1_direct_llm_sir",
    ])

    assert args.nn_model_path == DEFAULT_NN_MODEL_PATH


def test_init_strategy_auto_resolves_to_llm_cold_start():
    args = runner.parse_args([
        "--method",
        "m1_direct_llm_sir",
    ])

    assert runner.resolve_init_strategy(args.init_strategy) == "llm_cold_start"


def test_init_strategy_can_preserve_seed_smiles_initialization():
    args = runner.parse_args([
        "--method",
        "m1_direct_llm_sir",
        "--init-strategy",
        "seed_smiles",
    ])

    assert runner.resolve_init_strategy(args.init_strategy) == "seed_smiles"


def test_build_real_analog_fn_passes_reasyn_time_limit(tmp_path):
    args = runner.parse_args([
        "--method",
        "m1_llm_seed_analog_oversample_sir",
        "--trajectory-dir",
        str(tmp_path),
        "--reasyn-time-limit",
        "1800",
    ])

    with mock.patch("strbo_v1.analog.ReasynConfig") as config_cls:
        runner.build_real_analog_fn(args)

    assert config_cls.call_args.kwargs["time_limit"] == 1800
