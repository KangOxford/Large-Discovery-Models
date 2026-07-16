#!/usr/bin/env python3
"""Generate diagnostic case-study tables for a G12C QSAR run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from train_g12c_qsar import HAS_RDKIT, Chem, make_splits


def latest_run_dir(root: Path) -> Path:
    candidates = [path for path in root.glob("g12c_*") if path.is_dir() and (path / "metrics.csv").exists()]
    if not candidates:
        raise RuntimeError(f"No G12C run directories with metrics.csv found under {root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def best_by_split(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for split, group in metrics.groupby("split"):
        best_rmse = group.sort_values(["rmse", "mae", "spearman"], ascending=[True, True, False]).iloc[0].copy()
        best_rmse["criterion"] = "best_rmse"
        rows.append(best_rmse)
        best_spearman = group.sort_values(["spearman", "rmse"], ascending=[False, True]).iloc[0].copy()
        best_spearman["criterion"] = "best_spearman"
        rows.append(best_spearman)
    return pd.DataFrame(rows).reset_index(drop=True)


def prediction_cases(predictions: pd.DataFrame, best: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    cases: list[pd.DataFrame] = []
    for _, row in best[best["criterion"] == "best_rmse"].iterrows():
        split = str(row["split"])
        model = str(row["model"])
        subset = predictions[(predictions["split"] == split) & (predictions["prediction_model"] == model)].copy()
        if subset.empty:
            continue
        subset["case_type"] = "largest_abs_error"
        cases.append(subset.sort_values("abs_prediction_error", ascending=False).head(top_n))
        good = subset.copy()
        good["case_type"] = "smallest_abs_error"
        cases.append(good.sort_values("abs_prediction_error", ascending=True).head(top_n))
    if not cases:
        return pd.DataFrame()
    columns = [
        "split",
        "prediction_model",
        "case_type",
        "canonical_smiles",
        "p_activity",
        "predicted_p_activity",
        "prediction_error",
        "abs_prediction_error",
        "n_records",
        "p_activity_std",
        "primary_document_chembl_id",
        "primary_assay_chembl_id",
        "molecule_chembl_ids",
        "min_standard_value_nm",
        "max_standard_value_nm",
    ]
    out = pd.concat(cases, ignore_index=True)
    return out[[column for column in columns if column in out.columns]]


def consensus_cases(predictions: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    group_cols = ["split", "canonical_smiles"]
    rows = []
    for (split, smiles), group in predictions.groupby(group_cols):
        preds = group["predicted_p_activity"].astype(float).to_numpy()
        true_value = float(group["p_activity"].iloc[0])
        rows.append(
            {
                "split": split,
                "canonical_smiles": smiles,
                "p_activity": true_value,
                "prediction_mean": float(np.mean(preds)),
                "prediction_std": float(np.std(preds)),
                "prediction_min": float(np.min(preds)),
                "prediction_max": float(np.max(preds)),
                "consensus_error": float(np.mean(preds) - true_value),
                "abs_consensus_error": float(abs(np.mean(preds) - true_value)),
                "n_models": int(len(group)),
                "primary_document_chembl_id": group["primary_document_chembl_id"].iloc[0],
                "primary_assay_chembl_id": group["primary_assay_chembl_id"].iloc[0],
                "molecule_chembl_ids": group["molecule_chembl_ids"].iloc[0],
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    high_disagreement = summary.sort_values("prediction_std", ascending=False).head(top_n).copy()
    high_disagreement["case_type"] = "highest_model_disagreement"
    consensus_error = summary.sort_values("abs_consensus_error", ascending=False).head(top_n).copy()
    consensus_error["case_type"] = "largest_consensus_error"
    return pd.concat([high_disagreement, consensus_error], ignore_index=True)


def morgan_fp(smiles: str, generator: Any) -> Any:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return generator.GetFingerprint(mol)


def nearest_neighbor_table(dataset: pd.DataFrame, *, random_seed: int, test_size: float, top_n: int) -> pd.DataFrame:
    if not HAS_RDKIT:
        return pd.DataFrame()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    splits = make_splits(dataset, test_size=test_size, random_seed=random_seed)
    rows: list[dict[str, Any]] = []
    smiles_values = dataset["canonical_smiles"].astype(str).tolist()
    fps = [morgan_fp(smiles, generator) for smiles in smiles_values]
    labels = dataset["p_activity"].astype(float).to_numpy()
    for split in splits:
        train_fps = [fps[index] for index in split.train_index]
        train_indices = [index for index, fp in zip(split.train_index, train_fps) if fp is not None]
        train_fps = [fp for fp in train_fps if fp is not None]
        if not train_fps:
            continue
        for test_index in split.test_index:
            fp = fps[test_index]
            if fp is None:
                continue
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=float)
            best_pos = int(np.argmax(sims))
            nn_index = int(train_indices[best_pos])
            rows.append(
                {
                    "split": split.name,
                    "canonical_smiles": smiles_values[test_index],
                    "p_activity": float(labels[test_index]),
                    "nearest_neighbor_smiles": smiles_values[nn_index],
                    "nearest_neighbor_p_activity": float(labels[nn_index]),
                    "nearest_neighbor_similarity": float(sims[best_pos]),
                    "activity_delta_vs_neighbor": float(labels[test_index] - labels[nn_index]),
                    "abs_activity_delta_vs_neighbor": float(abs(labels[test_index] - labels[nn_index])),
                    "primary_document_chembl_id": dataset["primary_document_chembl_id"].iloc[test_index],
                    "primary_assay_chembl_id": dataset["primary_assay_chembl_id"].iloc[test_index],
                    "nearest_neighbor_document_chembl_id": dataset["primary_document_chembl_id"].iloc[nn_index],
                    "nearest_neighbor_assay_chembl_id": dataset["primary_assay_chembl_id"].iloc[nn_index],
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    top_delta = table.sort_values("abs_activity_delta_vs_neighbor", ascending=False).head(top_n).copy()
    top_delta["case_type"] = "largest_activity_delta_vs_nearest_neighbor"
    low_similarity = table.sort_values("nearest_neighbor_similarity", ascending=True).head(top_n).copy()
    low_similarity["case_type"] = "lowest_nearest_neighbor_similarity"
    return pd.concat([top_delta, low_similarity], ignore_index=True)


def write_report(
    path: Path,
    *,
    run_dir: Path,
    dataset: pd.DataFrame,
    best: pd.DataFrame,
    metrics: pd.DataFrame,
    nn_cases: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        text_df = df.copy()
        for column in text_df.columns:
            if pd.api.types.is_float_dtype(text_df[column]):
                text_df[column] = text_df[column].map(lambda value: f"{float(value):.3f}")
            else:
                text_df[column] = text_df[column].astype(str)
        header = "| " + " | ".join(text_df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in text_df.to_numpy()
        ]
        return "\n".join([header, sep, *rows])

    lines = [
        "# G12C QSAR Case Study",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Dataset Concentration",
        "",
        f"- Rows: {len(dataset):,}",
        f"- Primary documents: {dataset['primary_document_chembl_id'].nunique():,}",
        f"- Primary assays: {dataset['primary_assay_chembl_id'].nunique():,}",
        f"- Largest document rows: {int(dataset['primary_document_chembl_id'].value_counts().iloc[0]):,}",
        f"- Largest assay rows: {int(dataset['primary_assay_chembl_id'].value_counts().iloc[0]):,}",
        "",
        "## Best Models by Split",
        "",
        markdown_table(best[best["criterion"] == "best_rmse"][["split", "model", "rmse", "mae", "r2", "spearman"]]),
        "",
        "## Diagnostic Readout",
        "",
        "- Random split is easiest and nearest-neighbor SAR is highly competitive.",
        "- Document holdout exposes patent/document-series transfer loss.",
        "- Assay holdout is the hardest split; poor R2 there indicates mixed assay labels are a dominant failure mode.",
        "- Compounds with low nearest-neighbor similarity or large activity deltas against close neighbors should be down-weighted or flagged for human review.",
        "",
    ]
    if not nn_cases.empty:
        by_split = nn_cases.groupby("split")["nearest_neighbor_similarity"].describe()[["mean", "min", "max"]].reset_index()
        lines.extend(
            [
                "## Nearest-Neighbor Case Notes",
                "",
                markdown_table(by_split),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate G12C QSAR diagnostic case-study outputs.")
    parser.add_argument("--run-dir", default="", help="Run directory with metrics.csv and predictions_all_splits.csv.")
    parser.add_argument("--runs-root", default="activity_modeling/runs")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=714)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(Path(args.runs_root))
    metrics = pd.read_csv(run_dir / "metrics.csv")
    dataset = pd.read_csv(run_dir / "g12c_ic50_dataset.csv")
    predictions_path = run_dir / "predictions_all_splits.csv"
    if not predictions_path.exists():
        raise RuntimeError(f"{predictions_path} not found. Re-run train_g12c_qsar.py with current script.")
    predictions = pd.read_csv(predictions_path)

    best = best_by_split(metrics)
    best.to_csv(run_dir / "best_by_split.csv", index=False)

    cases = prediction_cases(predictions, best, top_n=args.top_n)
    cases.to_csv(run_dir / "case_study_prediction_cases.csv", index=False)

    consensus = consensus_cases(predictions, top_n=args.top_n)
    consensus.to_csv(run_dir / "case_study_consensus_cases.csv", index=False)

    nn_cases = nearest_neighbor_table(dataset, random_seed=args.random_seed, test_size=args.test_size, top_n=args.top_n)
    nn_cases.to_csv(run_dir / "case_study_nearest_neighbor_cases.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "rows": int(len(dataset)),
        "documents": int(dataset["primary_document_chembl_id"].nunique()),
        "assays": int(dataset["primary_assay_chembl_id"].nunique()),
        "best_by_split": best.to_dict(orient="records"),
        "outputs": {
            "best_by_split_csv": str(run_dir / "best_by_split.csv"),
            "prediction_cases_csv": str(run_dir / "case_study_prediction_cases.csv"),
            "consensus_cases_csv": str(run_dir / "case_study_consensus_cases.csv"),
            "nearest_neighbor_cases_csv": str(run_dir / "case_study_nearest_neighbor_cases.csv"),
            "report_md": str(run_dir / "CASE_STUDY.md"),
        },
    }
    (run_dir / "case_study_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(run_dir / "CASE_STUDY.md", run_dir=run_dir, dataset=dataset, best=best, metrics=metrics, nn_cases=nn_cases)
    print(f"Wrote case-study outputs under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
