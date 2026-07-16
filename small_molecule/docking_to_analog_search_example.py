#!/usr/bin/env python3
"""
Convert extracted seed SMILES into ReaSyn hit-expansion inputs.

The intended pipeline is PDF extraction -> seed SMILES/literature activity ->
ReaSyn analog generation -> AutoDock Vina ranking. ReaSyn takes seed molecules
as SMILES and generates synthesizable analogs plus pathways in inference mode,
so this bridge writes a SMILES text file and a reproducible command script for
ReaSyn's scripts/sample.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


SCORE_COLUMNS = ("score", "vina_score_kcal_mol", "best_score_kcal_mol", "best_score", "affinity", "affinity_kcal_mol")
ACTIVITY_COLUMNS = ("primary_activity_value_nM", "Activity_nM", "activity_nM", "IC50_nM", "EC50_nM", "Ki_nM", "Kd_nM")
SMILES_COLUMNS = ("canonical_smiles", "selected_smiles", "SMILES", "smiles")
STATUS_COLUMNS = ("status", "docking_status")
SUCCESS_STATUSES = {"", "ok", "success", "completed"}
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SEED_CSV = REPO_ROOT / "examples" / "reasyn_demo" / "autodock_markush_demo_docking_results.csv"
DEFAULT_DOCKING_RESULTS = DEFAULT_SEED_CSV
DEFAULT_REASYN_MODEL_PATHS = (
    "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,"
    "data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
)
DEFAULT_REASYN_INPUT_NAME = "reasyn_input.txt"
DEFAULT_REASYN_OUTPUT_NAME = "reasyn_analogs.csv"


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_value(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def docking_row_score(row: dict[str, Any]) -> Optional[float]:
    for column in SCORE_COLUMNS:
        score = safe_float(row.get(column))
        if score is not None:
            return score
    return None


def seed_row_activity_nM(row: dict[str, Any]) -> Optional[float]:
    for column in ACTIVITY_COLUMNS:
        activity = safe_float(row.get(column))
        if activity is not None:
            return activity
    return None


def docking_row_is_success(row: dict[str, Any]) -> bool:
    status = row_value(row, STATUS_COLUMNS).lower()
    return status in SUCCESS_STATUSES


def seed_sort_key(row: dict[str, Any]) -> tuple[int, float, int]:
    activity = seed_row_activity_nM(row)
    score = docking_row_score(row)
    input_order = int(str(row.get("_input_order", "0") or "0"))
    if activity is not None:
        return (0, activity, input_order)
    if score is not None:
        return (1, score, input_order)
    return (2, float("inf"), input_order)


def canonicalize_smiles(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    try:
        from rdkit import Chem
    except ImportError:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def read_seed_smiles(path: Path, top_n: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    score_columns_present = any(column in fieldnames for column in SCORE_COLUMNS)
    activity_columns_present = any(column in fieldnames for column in ACTIVITY_COLUMNS)
    seeds: list[dict[str, Any]] = []
    for row in rows:
        score = docking_row_score(row)
        activity_nM = seed_row_activity_nM(row)
        smiles = row_value(row, SMILES_COLUMNS)
        if not docking_row_is_success(row) or not smiles:
            continue
        if score_columns_present and not activity_columns_present and score is None:
            continue
        normalized = dict(row)
        normalized["compound_id"] = row_value(row, ("compound_id", "Compound", "compound")) or f"seed_{len(seeds) + 1}"
        normalized["canonical_smiles"] = canonicalize_smiles(smiles)
        normalized["score"] = "" if score is None else str(score)
        normalized["activity_nM"] = "" if activity_nM is None else str(activity_nM)
        normalized["_input_order"] = str(len(seeds))
        seeds.append(normalized)
    seeds.sort(key=seed_sort_key)
    return seeds if top_n <= 0 else seeds[:top_n]


def read_top_docking_hits(path: Path, top_n: int) -> list[dict[str, Any]]:
    """Backward-compatible alias for older examples that passed docking CSVs."""
    return read_seed_smiles(path, top_n)


def write_reasyn_input_txt(path: Path, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with path.open("w", encoding="utf-8") as handle:
        handle.write("SMILES\n")
        for index, hit in enumerate(hits, start=1):
            row = {
                "compound_id": str(hit.get("compound_id", f"seed_{index}")),
                "SMILES": str(hit.get("canonical_smiles", "")),
                "dock_score": str(hit.get("score", "")),
                "activity_nM": str(hit.get("activity_nM", "")),
                "activity_assay": str(hit.get("primary_activity_assay", hit.get("Activity_Assay", ""))),
                "activity_target": str(hit.get("primary_activity_target", hit.get("Activity_Target", ""))),
                "source_rank": str(index),
            }
            handle.write(row["SMILES"] + "\n")
            rows.append(row)
    return rows


def resolve_reasyn_repo(value: Optional[str]) -> Optional[Path]:
    if value:
        return Path(value).expanduser().resolve()
    env_home = os.getenv("REASYN_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return None


def discover_reasyn_entrypoint(reasyn_repo: Optional[Path], entrypoint: Optional[str]) -> Path:
    if entrypoint:
        return Path(entrypoint).expanduser().resolve()
    if reasyn_repo:
        return reasyn_repo / "scripts" / "sample.py"
    return Path("scripts/sample.py")


def resolve_reasyn_model_paths(model_paths: str, reasyn_repo: Optional[Path]) -> tuple[list[Path], str]:
    resolved: list[Path] = []
    for item in str(model_paths).split(","):
        path_text = item.strip()
        if not path_text:
            continue
        path = Path(path_text).expanduser()
        if reasyn_repo and not path.is_absolute():
            path = (reasyn_repo / path).resolve()
        resolved.append(path)
    return resolved, ",".join(str(path) for path in resolved)


def shell_command(cmd: list[str], *, cwd: Optional[Path] = None) -> str:
    joined = shlex.join(cmd)
    if cwd:
        return f"(cd {shlex.quote(str(cwd))} && {joined})"
    return joined


def reasyn_command(
    *,
    python_bin: str,
    entrypoint: Path,
    model_paths_csv: str,
    input_txt: Path,
    output_csv: Path,
    search_width: int,
    exhaustiveness: int,
    num_gpus: int,
    num_workers_per_gpu: int,
    task_qsize: int,
    result_qsize: int,
    time_limit: int,
    add_bb_path: Optional[str],
    no_exact_break: bool,
    num_cycles: int,
    num_editflow_samples: int,
    num_editflow_steps: int,
    mols_to_filter: Optional[str],
    filter_sim: float,
) -> list[str]:
    cmd = [
        python_bin,
        str(entrypoint),
        "-m",
        model_paths_csv,
        "-i",
        str(input_txt),
        "-o",
        str(output_csv),
        "--search_width",
        str(search_width),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_gpus",
        str(num_gpus),
        "--num_workers_per_gpu",
        str(num_workers_per_gpu),
        "--task_qsize",
        str(task_qsize),
        "--result_qsize",
        str(result_qsize),
        "--time_limit",
        str(time_limit),
        "--num_cycles",
        str(num_cycles),
        "--num_editflow_samples",
        str(num_editflow_samples),
        "--num_editflow_steps",
        str(num_editflow_steps),
        "--filter_sim",
        str(filter_sim),
    ]
    if add_bb_path:
        cmd.extend(["--add_bb_path", add_bb_path])
    if no_exact_break:
        cmd.append("--no_exact_break")
    if mols_to_filter:
        cmd.extend(["--mols_to_filter", mols_to_filter])
    return cmd


def output_summary(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"analog_count": 0, "target_count": 0, "max_similarity": None}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores = [score for score in (safe_float(row.get("score")) for row in rows) if score is not None]
    return {
        "analog_count": len(rows),
        "target_count": len({row.get("target", "") for row in rows if row.get("target", "")}),
        "unique_analog_count": len({row.get("smiles", "") for row in rows if row.get("smiles", "")}),
        "max_similarity": max(scores) if scores else None,
    }


def run_logged_command(
    cmd: list[str],
    log_path: Path,
    timeout: int,
    *,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False, cwd=str(cwd) if cwd else None)
    log_path.write_text(
        "$ "
        + shell_command(cmd, cwd=cwd)
        + "\n\n[stdout]\n"
        + proc.stdout
        + "\n[stderr]\n"
        + proc.stderr
        + "\n",
        encoding="utf-8",
    )
    return proc


def command_error_tail(proc: subprocess.CompletedProcess[str], max_chars: int = 3000) -> str:
    text = (proc.stderr or proc.stdout or "").strip()
    return text[-max_chars:] if text else f"command exited with code {proc.returncode}"


def unique_candidate_smiles(values: Any) -> list[str]:
    raw: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                raw.append(text)
            return
        if isinstance(value, dict):
            for key in ("smiles", "SMILES", "canonical_smiles", "product_smiles"):
                if key in value:
                    collect(value[key])
                    return
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                collect(nested)

    collect(values)
    seen: set[str] = set()
    smiles: list[str] = []
    for item in raw:
        canonical = canonicalize_smiles(item)
        if canonical and canonical not in seen:
            seen.add(canonical)
            smiles.append(canonical)
    return smiles


def filter_smiles_by_mw(smiles: list[str], mw_cutoff: Optional[float]) -> list[str]:
    if mw_cutoff is None:
        return smiles
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return smiles
    filtered: list[str] = []
    for item in smiles:
        mol = Chem.MolFromSmiles(item)
        if mol is not None and Descriptors.ExactMolWt(mol) <= mw_cutoff:
            filtered.append(item)
    return filtered


def sample_smiles_for_docking(smiles: list[str], *, sample_size: int, seed: int, strategy: str) -> list[str]:
    if sample_size <= 0 or len(smiles) <= sample_size:
        return list(smiles)
    if strategy == "first":
        return smiles[:sample_size]
    rng = random.Random(seed)
    return rng.sample(smiles, sample_size)


def aggregate_docking_scores(scores: list[float], *, objective: str, top_k: int) -> float:
    if not scores:
        raise RuntimeError("Cannot aggregate empty docking scores.")
    ordered = sorted(scores)
    if objective == "dock_mean_score":
        return float(sum(ordered) / len(ordered))
    if objective == "dock_topk_mean_score":
        k = max(1, min(top_k, len(ordered)))
        return float(sum(ordered[:k]) / k)
    return float(ordered[0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Example bridge from extracted seed SMILES to ReaSyn analog generation.")
    parser.add_argument(
        "--smiles-path",
        "--docking-results",
        dest="seed_csv",
        default=str(DEFAULT_SEED_CSV),
        help=(
            "Seed CSV from PDF extraction, or a legacy docking_results/joint_score CSV. "
            f"Defaults to {DEFAULT_SEED_CSV}."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of seed molecules to send to ReaSyn; 0 means all. Lower activity_nM ranks first when available.",
    )
    parser.add_argument("--output-dir", default="output/reasyn_example")
    parser.add_argument("--input-name", default=DEFAULT_REASYN_INPUT_NAME, help="Prepared ReaSyn SMILES input filename.")
    parser.add_argument("--reasyn-output-name", default=DEFAULT_REASYN_OUTPUT_NAME, help="ReaSyn analog output CSV filename.")
    parser.add_argument(
        "--reasyn-repo",
        help="Path to a local NVIDIA-BioNeMo/ReaSyn checkout. Falls back to REASYN_HOME when set.",
    )
    parser.add_argument(
        "--reasyn-entrypoint",
        help="Path to ReaSyn's sample script. Defaults to scripts/sample.py under --reasyn-repo.",
    )
    parser.add_argument(
        "--model-paths",
        "--model-path",
        dest="model_paths",
        default=DEFAULT_REASYN_MODEL_PATHS,
        help="Comma-separated ReaSyn AR and Edit Bridge checkpoints. Relative paths are resolved under --reasyn-repo.",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for ReaSyn.")
    parser.add_argument("--search-width", type=int, default=12)
    parser.add_argument("--exhaustiveness", type=int, default=128)
    parser.add_argument("--num-gpus", dest="reasyn_num_gpus", type=int, default=-1)
    parser.add_argument("--num-workers-per-gpu", type=int, default=8)
    parser.add_argument("--task-qsize", type=int, default=0)
    parser.add_argument("--result-qsize", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=10000)
    parser.add_argument("--add-bb-path", default=None, help="Optional ReaSyn --add_bb_path value.")
    parser.add_argument("--allow-exact-break", dest="no_exact_break", action="store_false")
    parser.set_defaults(no_exact_break=True)
    parser.add_argument("--num-cycles", type=int, default=12)
    parser.add_argument("--num-editflow-samples", type=int, default=100)
    parser.add_argument("--num-editflow-steps", type=int, default=100)
    parser.add_argument("--mols-to-filter", default=None)
    parser.add_argument("--filter-sim", type=float, default=0.8)
    parser.add_argument("--run-reasyn", action="store_true", help="Run ReaSyn after writing the input SMILES file.")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.top_n < 0:
        raise RuntimeError("--top-n must be non-negative.")
    if args.search_width < 1:
        raise RuntimeError("--search-width must be at least 1.")
    if args.exhaustiveness < 1:
        raise RuntimeError("--exhaustiveness must be at least 1.")
    if args.num_workers_per_gpu < 1:
        raise RuntimeError("--num-workers-per-gpu must be at least 1.")
    if args.task_qsize < 0 or args.result_qsize < 0:
        raise RuntimeError("Queue sizes must be non-negative.")
    if args.time_limit < 1:
        raise RuntimeError("--time-limit must be at least 1.")
    if args.num_cycles < 1:
        raise RuntimeError("--num-cycles must be at least 1.")
    if args.num_editflow_samples < 1 or args.num_editflow_steps < 1:
        raise RuntimeError("EditFlow sample and step counts must be at least 1.")
    if args.timeout < 1:
        raise RuntimeError("--timeout must be at least 1.")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    seed_csv = Path(args.seed_csv)
    hits = read_seed_smiles(seed_csv, args.top_n)
    if not hits:
        raise RuntimeError(f"No seed SMILES found in {args.seed_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_txt = (output_dir / args.input_name).resolve()
    output_csv = (output_dir / args.reasyn_output_name).resolve()
    input_rows = write_reasyn_input_txt(input_txt, hits)

    reasyn_repo = resolve_reasyn_repo(args.reasyn_repo)
    entrypoint = discover_reasyn_entrypoint(reasyn_repo, args.reasyn_entrypoint)
    model_paths, model_paths_csv = resolve_reasyn_model_paths(args.model_paths, reasyn_repo)
    cmd = reasyn_command(
        python_bin=args.python_bin,
        entrypoint=entrypoint,
        model_paths_csv=model_paths_csv,
        input_txt=input_txt,
        output_csv=output_csv,
        search_width=args.search_width,
        exhaustiveness=args.exhaustiveness,
        num_gpus=args.reasyn_num_gpus,
        num_workers_per_gpu=args.num_workers_per_gpu,
        task_qsize=args.task_qsize,
        result_qsize=args.result_qsize,
        time_limit=args.time_limit,
        add_bb_path=args.add_bb_path,
        no_exact_break=args.no_exact_break,
        num_cycles=args.num_cycles,
        num_editflow_samples=args.num_editflow_samples,
        num_editflow_steps=args.num_editflow_steps,
        mols_to_filter=args.mols_to_filter,
        filter_sim=args.filter_sim,
    )
    command_text = shell_command(cmd, cwd=reasyn_repo)

    records = [
        {
            "compound_id": row["compound_id"],
            "dock_score": row["dock_score"],
            "activity_nM": row["activity_nM"],
            "activity_assay": row["activity_assay"],
            "activity_target": row["activity_target"],
            "seed_smiles": row["SMILES"],
            "source_rank": row["source_rank"],
            "run_status": "prepared",
        }
        for row in input_rows
    ]
    run_status = "prepared_only"
    run_outputs: dict[str, Any] = {}
    manifest_warnings: list[str] = []

    if args.run_reasyn:
        if not entrypoint.exists():
            raise RuntimeError(f"ReaSyn entrypoint not found: {entrypoint}. Pass --reasyn-repo or --reasyn-entrypoint.")
        missing_models = [str(path) for path in model_paths if not path.exists()]
        if len(model_paths) != 2 or missing_models:
            raise RuntimeError(
                "ReaSyn inference requires comma-separated AR and Edit Bridge checkpoints. "
                f"Missing/invalid model paths: {missing_models or model_paths_csv}"
            )
        log_path = output_dir / "reasyn.log"
        proc = run_logged_command(
            cmd,
            log_path,
            timeout=args.timeout,
            cwd=reasyn_repo,
        )
        run_status = "ok" if proc.returncode == 0 else "failed"
        for record in records:
            record["run_status"] = run_status
        run_outputs = {
            "log": str(log_path),
            "returncode": proc.returncode,
            "output_summary": output_summary(output_csv) if proc.returncode == 0 else {},
        }
        if proc.returncode != 0:
            run_outputs["message"] = command_error_tail(proc)
    elif reasyn_repo is None:
        manifest_warnings.append("No --reasyn-repo or REASYN_HOME was provided; the command script uses scripts/sample.py.")

    manifest_path = output_dir / "reasyn_manifest.json"
    commands_path = output_dir / "run_reasyn_commands.sh"
    manifest = {
        "engine": "reasyn",
        "source_seed_csv": str(seed_csv),
        "input_format": "SMILES text file with a SMILES header",
        "input_txt": str(input_txt),
        "output_csv": str(output_csv),
        "top_n": args.top_n,
        "run_status": run_status,
        "run_reasyn": bool(args.run_reasyn),
        "reasyn_repo": str(reasyn_repo) if reasyn_repo else "",
        "reasyn_entrypoint": str(entrypoint),
        "model_paths": [str(path) for path in model_paths],
        "reasyn_params": {
            "search_width": args.search_width,
            "exhaustiveness": args.exhaustiveness,
            "num_gpus": args.reasyn_num_gpus,
            "num_workers_per_gpu": args.num_workers_per_gpu,
            "task_qsize": args.task_qsize,
            "result_qsize": args.result_qsize,
            "time_limit": args.time_limit,
            "add_bb_path": args.add_bb_path or "",
            "no_exact_break": bool(args.no_exact_break),
            "num_cycles": args.num_cycles,
            "num_editflow_samples": args.num_editflow_samples,
            "num_editflow_steps": args.num_editflow_steps,
            "mols_to_filter": args.mols_to_filter or "",
            "filter_sim": args.filter_sim,
        },
        "reasyn_command": command_text,
        "reasyn_outputs": run_outputs,
        "records": records,
        "warnings": manifest_warnings,
        "notes": [
            "ReaSyn consumes PDF-extracted seed molecules directly as SMILES and writes analog/pathway candidates as CSV.",
            "The default sampling settings follow the ReaSyn README hit-expansion example.",
            "Trained AR/Edit Bridge checkpoints and processed building-block indexes are not bundled with PDF2Dock.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command_text + "\n", encoding="utf-8")
    os.chmod(commands_path, 0o755)
    print(f"Wrote {input_txt}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {commands_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
