from pathlib import Path

from scripts.run_case2_main_four_experiment import (
    baseline_command,
    case2_command,
    parse_args,
    selected_runs,
)
from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH


def test_main_runner_expands_four_methods_per_seed(tmp_path: Path):
    args = parse_args([
        "--output-root",
        str(tmp_path),
        "--seeds",
        "13,17",
    ])

    runs = selected_runs(args)

    assert [(run["name"], run["seed"]) for run in runs] == [
        ("pure_llm_one_step", 13),
        ("ldm_llm_bo_sk", 13),
        ("ldm_llm_reasyn_bo_sk", 13),
        ("bo_baseline_sk", 13),
        ("pure_llm_one_step", 17),
        ("ldm_llm_bo_sk", 17),
        ("ldm_llm_reasyn_bo_sk", 17),
        ("bo_baseline_sk", 17),
    ]


def test_run_name_filter_rejects_removed_legacy_methods(tmp_path: Path):
    args = parse_args([
        "--output-root",
        str(tmp_path),
        "--run-names",
        "m1_stratified_sk",
    ])

    try:
        selected_runs(args)
    except SystemExit as exc:
        assert "unknown run names" in str(exc)
    else:
        raise AssertionError("legacy run name was accepted")


def test_pure_llm_one_step_command(tmp_path: Path):
    args = parse_args([
        "--output-root",
        str(tmp_path),
        "--seeds",
        "13",
    ])
    run = selected_runs(args)[0]

    command, run_dir = case2_command(run, args, tmp_path)

    assert run["name"] == "pure_llm_one_step"
    assert run_dir == tmp_path / "pure_llm_one_step" / "seed_13"
    assert Path(command[1]).name == "run_case2_m1_experiment.py"
    assert command[command.index("--method") + 1] == "m1_llm_one_step"
    assert command[command.index("--kernel") + 1] == "sk"
    assert command[command.index("--m1-k-direct-llm") + 1] == "1"
    assert command[command.index("--init-strategy") + 1] == "llm_cold_start"


def test_ldm_llm_bo_command_uses_sk_and_oversampling(tmp_path: Path):
    args = parse_args([
        "--output-root",
        str(tmp_path),
        "--run-names",
        "ldm_llm_bo_sk",
    ])
    run = selected_runs(args)[0]

    command, run_dir = case2_command(run, args, tmp_path)

    assert run_dir == tmp_path / "ldm_llm_bo_sk" / "seed_0"
    assert command[command.index("--method") + 1] == "m1_stratified_direct_llm_oversample_sir"
    assert command[command.index("--kernel") + 1] == "sk"
    assert command[command.index("--m1-k-direct-llm") + 1] == "512"
    assert command[command.index("--max-candidates-per-round") + 1] == "256"


def test_ldm_llm_reasyn_command_uses_analog_oversampling(tmp_path: Path):
    args = parse_args([
        "--output-root",
        str(tmp_path),
        "--run-names",
        "ldm_llm_reasyn_bo_sk",
    ])
    run = selected_runs(args)[0]

    command, run_dir = case2_command(run, args, tmp_path)

    assert run_dir == tmp_path / "ldm_llm_reasyn_bo_sk" / "seed_0"
    assert command[command.index("--method") + 1] == "m1_llm_seed_analog_oversample_sir"
    assert command[command.index("--m1-analog-n-llm-seeds") + 1] == "16"
    assert command[command.index("--m1-analog-k-total") + 1] == "1024"
    assert command[command.index("--reasyn-time-limit") + 1] == "1800"


def test_bo_baseline_command_uses_shared_g12d_and_pool_config(tmp_path: Path):
    args = parse_args([
        "--output-root",
        str(tmp_path),
        "--run-names",
        "bo_baseline_sk",
    ])
    run = selected_runs(args)[0]

    command, run_dir = baseline_command(run, args, tmp_path)

    assert run_dir == tmp_path / "bo_baseline_sk" / "seed_0"
    assert command[command.index("--method") + 1] == "bo-strkernel"
    assert command[command.index("--nn-model-path") + 1] == DEFAULT_NN_MODEL_PATH
    assert command[command.index("--max-pool-size") + 1] == "256"
    assert command[command.index("--acq-budget") + 1] == "256"
    assert command[command.index("--smiles-max-len") + 1] == "80"
