from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POOL_SIZE = 256
DEFAULT_SMILES_MAX_LEN = 80
DEFAULT_REASYN_TIME_LIMIT = 1800
DEFAULT_DIRECT_LLM_OVERSAMPLE = 512
DEFAULT_ANALOG_LLM_SEEDS = 16
DEFAULT_ANALOG_POOL_SIZE = 1024

MAIN_RUNS = [
    {
        "kind": "case2",
        "name": "pure_llm_one_step",
        "method": "m1_llm_one_step",
        "extra": {"--m1-k-direct-llm": "1"},
    },
    {
        "kind": "case2",
        "name": "ldm_llm_bo_sk",
        "method": "m1_stratified_direct_llm_oversample_sir",
        "extra": {"--m1-k-direct-llm": str(DEFAULT_DIRECT_LLM_OVERSAMPLE)},
    },
    {
        "kind": "case2",
        "name": "ldm_llm_reasyn_bo_sk",
        "method": "m1_llm_seed_analog_oversample_sir",
        "extra": {
            "--m1-analog-n-llm-seeds": str(DEFAULT_ANALOG_LLM_SEEDS),
            "--m1-analog-k-total": str(DEFAULT_ANALOG_POOL_SIZE),
        },
    },
    {
        "kind": "baseline",
        "name": "bo_baseline_sk",
        "method": "bo-strkernel",
        "extra": {},
    },
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    manifest_path = output_root / args.manifest_name
    manifest = {"config": vars(args), "runs": []}
    for run in selected_runs(args):
        if run["kind"] == "baseline":
            command, run_dir = baseline_command(run, args, output_root)
        else:
            command, run_dir = case2_command(run, args, output_root)
        if args.skip_existing and run_is_complete(run, run_dir):
            manifest["runs"].append({"name": run["name"], "seed": run["seed"], "status": "skipped", "path": str(run_dir)})
            continue
        if args.dry_run:
            print(" ".join(shlex.quote(str(part)) for part in command))
            continue
        record = execute(command, run_dir, env)
        record.update({"name": run["name"], "seed": run["seed"], "kind": run["kind"], "path": str(run_dir)})
        manifest["runs"].append(record)
        write_json(manifest_path, manifest)
        if record["returncode"] != 0 and not args.keep_going:
            return record["returncode"]
    write_json(manifest_path, manifest)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--seed-smiles", default="CCO,CCN,CCC,CCCN,CCCC")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--init-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--smiles-max-len", type=int, default=DEFAULT_SMILES_MAX_LEN)
    parser.add_argument("--env-file", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vina-bin", default="")
    parser.add_argument("--reasyn-repo", default="")
    parser.add_argument("--reasyn-model-path", default="")
    parser.add_argument("--reasyn-devices", default="0")
    parser.add_argument("--reasyn-time-limit", type=int, default=DEFAULT_REASYN_TIME_LIMIT)
    parser.add_argument("--nn-model-path", default=DEFAULT_NN_MODEL_PATH)
    parser.add_argument("--gp-device", default="cpu")
    parser.add_argument("--gp-fit-itersteps", type=int, default=20)
    parser.add_argument("--ehvi-n-samples", type=int, default=128)
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=2,
        help="Retry attempts after the initial LLM JSON attempt fails.",
    )
    parser.add_argument(
        "--llm-retry-wait-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between failed LLM JSON attempts.",
    )
    parser.add_argument("--ldm-sys-prompt", default="")
    parser.add_argument("--m1-q0-smoothing", type=float, default=None)
    parser.add_argument("--run-names", default="")
    parser.add_argument("--manifest-name", default="experiment_manifest.json")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.env_file:
        load_env_file(Path(args.env_file), env)
    if args.reasyn_repo:
        env["REASYN_HOME"] = args.reasyn_repo
        env["REASYN_REPO"] = args.reasyn_repo
    if args.reasyn_model_path:
        env["REASYN_MODEL_PATH"] = args.reasyn_model_path
    if args.vina_bin:
        env["VINA_BIN"] = args.vina_bin
    return env


def load_env_file(path: Path, env: dict[str, str]) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def selected_runs(args: argparse.Namespace) -> list[dict]:
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    runs = [with_seed(run, seed) for seed in seeds for run in MAIN_RUNS]
    requested = {name.strip() for name in args.run_names.split(",") if name.strip()}
    if not requested:
        return runs
    filtered = [run for run in runs if run["name"] in requested]
    missing = requested - {run["name"] for run in filtered}
    if missing:
        raise SystemExit(f"--run-names contains unknown run names: {sorted(missing)}")
    return filtered


def with_seed(run: dict, seed: int) -> dict:
    return {
        "kind": run["kind"],
        "name": run["name"],
        "method": run["method"],
        "extra": dict(run["extra"]),
        "seed": seed,
    }


def baseline_command(run: dict, args: argparse.Namespace, root: Path) -> tuple[list[str], Path]:
    run_dir = root / run["name"] / f"seed_{run['seed']}"
    out_json = run_dir / "result.json"
    command = [
        args.python,
        str(REPO_ROOT / "run_search.py"),
        "--objective",
        "vina+nn",
        "--method",
        run["method"],
        "--seed",
        str(run["seed"]),
        "--num-evaluations",
        str(args.budget),
        "--init-size",
        str(args.init_size),
        "--batch-size",
        str(args.batch_size),
        "--seed-smiles",
        args.seed_smiles,
        "--gp-device",
        args.gp_device,
        "--gp-fit-itersteps",
        str(args.gp_fit_itersteps),
        "--ehvi-n-samples",
        str(args.ehvi_n_samples),
        "--max-pool-size",
        str(args.pool_size),
        "--acq-budget",
        str(args.pool_size),
        "--smiles-max-len",
        str(args.smiles_max_len),
        "--llm-timeout",
        str(args.llm_timeout),
        "--ref-point",
        "0,5",
        "--nn-model-path",
        args.nn_model_path,
        "--output",
        str(out_json),
    ]
    append_common_optional_flags(command, args)
    if args.llm_model:
        command.extend(["--llm-model", args.llm_model])
    if args.ldm_sys_prompt:
        command.extend(["--ldm-sys-prompt", args.ldm_sys_prompt])
    return command, run_dir


def case2_command(run: dict, args: argparse.Namespace, root: Path) -> tuple[list[str], Path]:
    run_dir = root / run["name"] / f"seed_{run['seed']}"
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "run_case2_m1_experiment.py"),
        "--method",
        run["method"],
        "--seed",
        str(run["seed"]),
        "--budget",
        str(args.budget),
        "--init-size",
        str(args.init_size),
        "--init-strategy",
        "llm_cold_start",
        "--batch-size",
        str(args.batch_size),
        "--seed-smiles",
        args.seed_smiles,
        "--kernel",
        "sk",
        "--gp-device",
        args.gp_device,
        "--gp-fit-itersteps",
        str(args.gp_fit_itersteps),
        "--ehvi-n-samples",
        str(args.ehvi_n_samples),
        "--llm-timeout",
        str(args.llm_timeout),
        "--llm-max-retries",
        str(args.llm_max_retries),
        "--llm-retry-wait-seconds",
        str(args.llm_retry_wait_seconds),
        "--max-candidates-per-round",
        str(args.pool_size),
        "--trajectory-dir",
        str(run_dir),
        "--nn-model-path",
        args.nn_model_path,
        "--reasyn-time-limit",
        str(args.reasyn_time_limit),
    ]
    if args.m1_q0_smoothing is not None:
        command.extend(["--m1-q0-smoothing", str(args.m1_q0_smoothing)])
    for key, value in run["extra"].items():
        command.extend([key, value])
    append_common_optional_flags(command, args)
    if args.llm_model:
        command.extend(["--llm-model", args.llm_model])
    return command, run_dir


def append_common_optional_flags(command: list[str], args: argparse.Namespace) -> None:
    if args.vina_bin:
        command.extend(["--vina-bin", args.vina_bin])
    if args.reasyn_repo:
        command.extend(["--reasyn-repo", args.reasyn_repo])
    if args.reasyn_model_path:
        command.extend(["--reasyn-model-path", args.reasyn_model_path])
    if args.reasyn_devices:
        command.extend(["--reasyn-devices", args.reasyn_devices])


def run_is_complete(run: dict, path: Path) -> bool:
    if run["kind"] == "baseline":
        return (path / "result.json").exists()
    return (path / "summary.json").exists() and (path / "history.json").exists()


def execute(command: list[str], run_path: Path, env: dict[str, str]) -> dict:
    run_path.mkdir(parents=True, exist_ok=True)
    log_path = run_path / "run.log"
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - t0
    record = {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
    }
    write_json(run_path / "manifest.json", record)
    return record


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
