#!/usr/bin/env python3
"""Prepare frozen inputs for a standard KRAS G12C retrospective experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen G12C standard benchmark splits and docking inputs.")
    parser.add_argument("--run-dir", default="activity_modeling/runs/g12c_expanded_20260612_122903")
    parser.add_argument("--benchmark-csv", default="", help="Defaults to <run-dir>/g12c_docking_benchmark_binary.csv")
    parser.add_argument("--experiment-dir", default="", help="Defaults to <run-dir>/standard_experiment")
    parser.add_argument("--pdb-id", default="8UN5")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--work-dir", default="output/docking_work")
    parser.add_argument("--docking-output-root", default="output/docking_work/g12c_standard")
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_inputs(run_dir: Path, benchmark_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark_path = Path(benchmark_csv) if benchmark_csv else run_dir / "g12c_docking_benchmark_binary.csv"
    predictions_path = run_dir / "predictions_all_splits.csv"
    metrics_path = run_dir / "metrics.csv"
    if not benchmark_path.exists():
        raise RuntimeError(f"Benchmark CSV not found: {benchmark_path}")
    if not predictions_path.exists():
        raise RuntimeError(f"Predictions file not found: {predictions_path}")
    if not metrics_path.exists():
        raise RuntimeError(f"Metrics file not found: {metrics_path}")
    benchmark = pd.read_csv(benchmark_path)
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    return benchmark, predictions, metrics


def best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.sort_values(["split", "rmse", "mae"], ascending=[True, True, True])
        .groupby("split", as_index=False)
        .first()[["split", "model", "rmse", "mae", "r2", "spearman"]]
        .sort_values("split")
    )


def split_manifest(benchmark: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    bench = benchmark.copy()
    bench["canonical_smiles"] = bench["canonical_smiles"].astype(str)
    rows: list[pd.DataFrame] = []
    for split, group in predictions.groupby("split"):
        test_smiles = sorted(set(group["canonical_smiles"].astype(str)))
        split_rows = bench[bench["canonical_smiles"].isin(test_smiles)].copy()
        split_rows.insert(0, "split", split)
        rows.append(split_rows)
    if not rows:
        return pd.DataFrame()
    manifest = pd.concat(rows, ignore_index=True)
    manifest["active_label"] = (manifest["benchmark_label"] == "active").astype(int)
    return manifest.sort_values(["split", "benchmark_label", "pIC50", "compound_id"], ascending=[True, True, False, True])


def docking_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "compound_id",
        "SMILES",
        "canonical_smiles",
        "Activity_nM",
        "pIC50",
        "benchmark_label",
        "active_pIC50_ge_7",
        "inactive_pIC50_le_6",
        "primary_assay_chembl_id",
        "primary_document_chembl_id",
        "assay_family",
        "assay_family_broad",
        "Activity_Target",
        "Activity_Endpoint",
        "Activity_Value",
        "Activity_Evidence",
        "Note",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def shell_quote(path: str) -> str:
    return "'" + path.replace("'", "'\"'\"'") + "'"


def docking_command(
    csv_path: Path,
    output_dir: Path,
    *,
    pdb_id: str,
    chain_id: str,
    work_dir: str,
    exhaustiveness: int,
    num_modes: int,
    seed: int,
) -> str:
    parts = [
        "python3",
        "extract_and_dock.py",
        "dock",
        "--csv",
        shell_quote(str(csv_path)),
        "--allow-unreviewed",
        "--pdb-id",
        pdb_id,
        "--chain-id",
        chain_id,
        "--work-dir",
        shell_quote(work_dir),
        "--output-dir",
        shell_quote(str(output_dir)),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num-modes",
        str(num_modes),
        "--seed",
        str(seed),
    ]
    return " ".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    experiment_dir = ensure_dir(Path(args.experiment_dir) if args.experiment_dir else run_dir / "standard_experiment")
    docking_input_dir = ensure_dir(experiment_dir / "docking_inputs")
    split_output_root = Path(args.docking_output_root)

    benchmark, predictions, metrics = load_inputs(run_dir, args.benchmark_csv)
    best = best_models(metrics)
    manifest = split_manifest(benchmark, predictions)
    if manifest.empty:
        raise RuntimeError("No split rows were generated.")

    manifest.to_csv(experiment_dir / "frozen_split_manifest.csv", index=False)
    best.to_csv(experiment_dir / "best_models_by_split.csv", index=False)

    split_summary_rows: list[dict[str, Any]] = []
    command_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run from the repository root after activating the markush-dock environment.",
        "# conda activate markush-dock",
        "",
    ]
    for split, group in manifest.groupby("split", sort=True):
        split_csv = docking_input_dir / f"{split}_test.csv"
        docking_columns(group).to_csv(split_csv, index=False)
        output_dir = split_output_root / f"{split}_test"
        command_lines.append(
            docking_command(
                split_csv,
                output_dir,
                pdb_id=args.pdb_id,
                chain_id=args.chain_id,
                work_dir=args.work_dir,
                exhaustiveness=args.exhaustiveness,
                num_modes=args.num_modes,
                seed=args.seed,
            )
        )
        counts = group["benchmark_label"].value_counts().to_dict()
        split_summary_rows.append(
            {
                "split": split,
                "rows": int(len(group)),
                "actives": int(counts.get("active", 0)),
                "inactives": int(counts.get("inactive", 0)),
                "active_rate": float(counts.get("active", 0) / len(group)) if len(group) else 0.0,
                "docking_input_csv": str(split_csv),
                "docking_output_dir": str(output_dir),
            }
        )

    unique = (
        manifest.sort_values(["benchmark_label", "pIC50", "compound_id"], ascending=[True, False, True])
        .drop_duplicates("canonical_smiles")
        .reset_index(drop=True)
    )
    all_unique_csv = docking_input_dir / "all_unique_test.csv"
    docking_columns(unique).to_csv(all_unique_csv, index=False)
    all_unique_output = split_output_root / "all_unique_test"
    command_lines.append("")
    command_lines.append("# Recommended full benchmark command: dock each unique test molecule once, then evaluate by split.")
    command_lines.append(
        docking_command(
            all_unique_csv,
            all_unique_output,
            pdb_id=args.pdb_id,
            chain_id=args.chain_id,
            work_dir=args.work_dir,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            seed=args.seed,
        )
    )

    commands_path = experiment_dir / "run_docking_commands.sh"
    commands_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    split_summary = pd.DataFrame(split_summary_rows).sort_values("split")
    split_summary.to_csv(experiment_dir / "split_summary.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "experiment_dir": str(experiment_dir),
        "benchmark_rows": int(len(benchmark)),
        "manifest_rows": int(len(manifest)),
        "unique_test_molecules": int(len(unique)),
        "splits": split_summary.to_dict(orient="records"),
        "all_unique_docking_input_csv": str(all_unique_csv),
        "all_unique_docking_output_dir": str(all_unique_output),
        "commands_file": str(commands_path),
        "docking_parameters": {
            "pdb_id": args.pdb_id,
            "chain_id": args.chain_id,
            "work_dir": args.work_dir,
            "output_root": args.docking_output_root,
            "exhaustiveness": args.exhaustiveness,
            "num_modes": args.num_modes,
            "seed": args.seed,
        },
    }
    write_json(experiment_dir / "standard_experiment_manifest.json", summary)
    print(f"Wrote standard experiment files to {experiment_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
