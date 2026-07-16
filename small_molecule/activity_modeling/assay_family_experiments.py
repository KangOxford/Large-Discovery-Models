#!/usr/bin/env python3
"""Assay-family stratification and focused G12C QSAR experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from train_g12c_qsar import (
    HAS_RDKIT,
    SplitData,
    build_model_pipelines,
    regression_metrics,
    scaffold_train_test_split,
)


SELECTED_MODELS = [
    "char_tfidf_ridge",
    "rdkit_desc_elastic_net",
    "morgan_tanimoto_knn_k5",
]


def latest_run_dir(root: Path) -> Path:
    candidates = [path for path in root.glob("g12c_*") if path.is_dir() and (path / "g12c_ic50_dataset.csv").exists()]
    if not candidates:
        raise RuntimeError(f"No G12C run directories found under {root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def classify_assay(description: Any) -> tuple[str, str]:
    text = str(description or "").lower()
    if any(term in text for term in ("nci-h358", "cell", "erk phosphorylation", "in-cell")):
        return "cellular_pERK", "cellular"
    if "scintillation proximity" in text or " spa" in text or "covalent probe" in text:
        return "competition_spa", "competition"
    if "covalently label" in text or "covalent label" in text or "label nucleotide-loaded" in text:
        return "covalent_labeling", "covalent_labeling"
    if "nucleotide exchange" in text or "sos-mediated guanine" in text or "sos1-mediated guanine" in text:
        if "20 hours" in text or "20 hour" in text:
            return "nucleotide_exchange_20h", "nucleotide_exchange"
        if "18 hours" in text or "18 hour" in text or "htrf" in text:
            return "nucleotide_exchange_htrf_18h", "nucleotide_exchange"
        if ("either 5 min or 2 hour" in text) or ("either 5 min or 2 hours" in text):
            return "nucleotide_exchange_mixed_5min_2h", "nucleotide_exchange"
        if "2 hours" in text or "2 hour" in text:
            return "nucleotide_exchange_2h", "nucleotide_exchange"
        if "5 minutes" in text or "5 min" in text:
            return "nucleotide_exchange_5min", "nucleotide_exchange"
        if "tr-fret" in text or "fret" in text:
            return "nucleotide_exchange_trfret", "nucleotide_exchange"
        return "nucleotide_exchange_other", "nucleotide_exchange"
    if "binding affinity" in text or " kd" in text:
        return "binding_affinity", "binding"
    if "inhibition" in text:
        return "biochemical_inhibition_other", "biochemical_other"
    return "other", "other"


def load_family_dataset(run_dir: Path) -> pd.DataFrame:
    dataset = pd.read_csv(run_dir / "g12c_ic50_dataset.csv")
    raw = pd.read_csv(run_dir / "raw_g12c_chembl_activities.csv")
    descriptions = (
        raw.dropna(subset=["assay_chembl_id"])
        .assign(assay_chembl_id=lambda df: df["assay_chembl_id"].astype(str))
        .drop_duplicates("assay_chembl_id")
        .set_index("assay_chembl_id")["assay_description"]
        .to_dict()
    )
    dataset["primary_assay_chembl_id"] = dataset["primary_assay_chembl_id"].astype(str)
    dataset["primary_document_chembl_id"] = dataset["primary_document_chembl_id"].astype(str)
    dataset["primary_assay_description"] = dataset["primary_assay_chembl_id"].map(descriptions).fillna("")
    families = dataset["primary_assay_description"].map(classify_assay)
    dataset["assay_family"] = [item[0] for item in families]
    dataset["assay_family_broad"] = [item[1] for item in families]
    for group_col in ("primary_assay_chembl_id", "primary_document_chembl_id", "assay_family", "assay_family_broad"):
        median = dataset.groupby(group_col)["p_activity"].transform("median")
        mean = dataset.groupby(group_col)["p_activity"].transform("mean")
        std = dataset.groupby(group_col)["p_activity"].transform("std").replace(0, np.nan)
        dataset[f"p_activity_centered_by_{group_col}"] = dataset["p_activity"] - median
        dataset[f"p_activity_mean_centered_by_{group_col}"] = dataset["p_activity"] - mean
        dataset[f"p_activity_z_by_{group_col}"] = ((dataset["p_activity"] - mean) / std).fillna(0.0)
    return dataset


def summary_tables(dataset: pd.DataFrame, out_dir: Path) -> None:
    family_summary = (
        dataset.groupby(["assay_family_broad", "assay_family"])
        .agg(
            n=("canonical_smiles", "count"),
            median_pIC50=("p_activity", "median"),
            mean_pIC50=("p_activity", "mean"),
            std_pIC50=("p_activity", "std"),
            n_assays=("primary_assay_chembl_id", "nunique"),
            n_documents=("primary_document_chembl_id", "nunique"),
            acrylamide_fraction=("warhead_acrylamide_like", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    family_summary.to_csv(out_dir / "assay_family_summary.csv", index=False)

    assay_summary = (
        dataset.groupby(["primary_assay_chembl_id", "assay_family_broad", "assay_family", "primary_document_chembl_id"])
        .agg(
            n=("canonical_smiles", "count"),
            median_pIC50=("p_activity", "median"),
            mean_pIC50=("p_activity", "mean"),
            std_pIC50=("p_activity", "std"),
            acrylamide_fraction=("warhead_acrylamide_like", "mean"),
            description=("primary_assay_description", "first"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    assay_summary.to_csv(out_dir / "assay_summary.csv", index=False)

    document_summary = (
        dataset.groupby(["primary_document_chembl_id", "assay_family_broad"])
        .agg(
            n=("canonical_smiles", "count"),
            median_pIC50=("p_activity", "median"),
            mean_pIC50=("p_activity", "mean"),
            std_pIC50=("p_activity", "std"),
            n_assays=("primary_assay_chembl_id", "nunique"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    document_summary.to_csv(out_dir / "document_family_summary.csv", index=False)


def median_baseline_metrics(dataset: pd.DataFrame, out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    y = dataset["p_activity"].astype(float).to_numpy()
    global_median = float(np.median(y))
    for name, pred in [("global_median", np.full_like(y, global_median, dtype=float))]:
        metrics = regression_metrics(y, pred)
        rows.append({"baseline": name, **metrics})
    for group_col in ("assay_family_broad", "assay_family", "primary_document_chembl_id", "primary_assay_chembl_id"):
        pred = dataset.groupby(group_col)["p_activity"].transform("median").astype(float).to_numpy()
        metrics = regression_metrics(y, pred)
        rows.append({"baseline": f"in_sample_median_by_{group_col}", **metrics})
    pd.DataFrame(rows).to_csv(out_dir / "median_baselines.csv", index=False)


def train_models_for_label(
    dataset: pd.DataFrame,
    *,
    label_col: str,
    experiment_name: str,
    models: dict[str, Any],
    selected_models: list[str],
    out_rows: list[dict[str, Any]],
    random_seed: int,
    test_size: float,
) -> None:
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)
    splits = [SplitData("random", np.asarray(train_idx), np.asarray(test_idx))]
    if HAS_RDKIT:
        scaffold_train, scaffold_test = scaffold_train_test_split(dataset, test_size=test_size, random_seed=random_seed)
        splits.append(SplitData("scaffold", scaffold_train, scaffold_test))
    X = dataset["canonical_smiles"].astype(str)
    y = dataset[label_col].astype(float).to_numpy()
    for split in splits:
        for model_name in selected_models:
            if model_name not in models:
                continue
            print(f"{experiment_name} | {label_col} | {split.name} | {model_name} | n={len(dataset)}", flush=True)
            model = clone(models[model_name])
            model.fit(X.iloc[split.train_index], y[split.train_index])
            pred = model.predict(X.iloc[split.test_index])
            metrics = regression_metrics(y[split.test_index], pred)
            out_rows.append(
                {
                    "experiment": experiment_name,
                    "label": label_col,
                    "model": model_name,
                    "split": split.name,
                    "n_rows": int(len(dataset)),
                    "n_train": int(len(split.train_index)),
                    "n_test": int(len(split.test_index)),
                    "n_assays": int(dataset["primary_assay_chembl_id"].nunique()),
                    "n_documents": int(dataset["primary_document_chembl_id"].nunique()),
                    "n_families": int(dataset["assay_family"].nunique()),
                    **metrics,
                }
            )


def run_experiments(dataset: pd.DataFrame, out_dir: Path, *, random_seed: int, test_size: float, min_subset_size: int) -> pd.DataFrame:
    models = build_model_pipelines(random_seed)
    selected = [name for name in SELECTED_MODELS if name in models]
    rows: list[dict[str, Any]] = []

    train_models_for_label(
        dataset,
        label_col="p_activity",
        experiment_name="all_absolute",
        models=models,
        selected_models=selected,
        out_rows=rows,
        random_seed=random_seed,
        test_size=test_size,
    )
    for label_col in (
        "p_activity_centered_by_primary_assay_chembl_id",
        "p_activity_centered_by_assay_family_broad",
        "p_activity_z_by_primary_assay_chembl_id",
    ):
        train_models_for_label(
            dataset,
            label_col=label_col,
            experiment_name=f"all_{label_col}",
            models=models,
            selected_models=selected,
            out_rows=rows,
            random_seed=random_seed,
            test_size=test_size,
        )

    subset_specs: list[tuple[str, pd.DataFrame]] = []
    for family, group in dataset.groupby("assay_family_broad"):
        if len(group) >= min_subset_size:
            subset_specs.append((f"family_broad={family}", group.copy()))
    for family, group in dataset.groupby("assay_family"):
        if len(group) >= min_subset_size:
            subset_specs.append((f"family={family}", group.copy()))
    for document, group in dataset.groupby("primary_document_chembl_id"):
        if len(group) >= min_subset_size:
            subset_specs.append((f"document={document}", group.copy()))
    for assay, group in dataset.groupby("primary_assay_chembl_id"):
        if len(group) >= min_subset_size:
            subset_specs.append((f"assay={assay}", group.copy()))

    for experiment_name, subset in subset_specs:
        train_models_for_label(
            subset.reset_index(drop=True),
            label_col="p_activity",
            experiment_name=experiment_name,
            models=models,
            selected_models=selected,
            out_rows=rows,
            random_seed=random_seed,
            test_size=test_size,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "assay_family_experiment_metrics.csv", index=False)
    best_rows = []
    for (experiment, split, label), group in metrics.groupby(["experiment", "split", "label"]):
        best = group.sort_values(["rmse", "mae", "spearman"], ascending=[True, True, False]).iloc[0].copy()
        best_rows.append(best)
    best = pd.DataFrame(best_rows).sort_values(["experiment", "split", "rmse"])
    best.to_csv(out_dir / "assay_family_best_by_experiment.csv", index=False)
    return metrics


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda value: f"{float(value):.3f}")
        else:
            text_df[col] = text_df[col].astype(str)
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in text_df.to_numpy(dtype=str)]
    return "\n".join([header, sep, *body])


def write_report(dataset: pd.DataFrame, out_dir: Path) -> None:
    family_summary = pd.read_csv(out_dir / "assay_family_summary.csv")
    best = pd.read_csv(out_dir / "assay_family_best_by_experiment.csv")
    baselines = pd.read_csv(out_dir / "median_baselines.csv")

    all_abs = best[(best["experiment"] == "all_absolute") & (best["label"] == "p_activity")]
    subset_focus = best[
        best["experiment"].str.startswith(("family_broad=", "family=", "document=", "assay="))
        & best["split"].isin(["random", "scaffold"])
    ].sort_values(["rmse", "n_rows"], ascending=[True, False]).head(20)
    centered = best[best["experiment"].str.startswith("all_p_activity_centered")]
    zscore = best[best["experiment"].str.startswith("all_p_activity_z")]

    lines = [
        "# G12C Assay-Family Follow-Up",
        "",
        "## Assay Family Distribution",
        "",
        markdown_table(family_summary[["assay_family_broad", "assay_family", "n", "median_pIC50", "std_pIC50", "n_assays", "n_documents"]].head(12)),
        "",
        "## Median Baselines",
        "",
        markdown_table(baselines[["baseline", "rmse", "mae", "r2", "spearman"]]),
        "",
        "## Best All-Data Absolute Models",
        "",
        markdown_table(all_abs[["split", "model", "rmse", "mae", "r2", "spearman"]]),
        "",
        "## Best Centered-Label Models",
        "",
        markdown_table(centered[["experiment", "split", "model", "rmse", "mae", "r2", "spearman"]].head(20)),
        "",
        "## Best Z-Score Models",
        "",
        markdown_table(zscore[["experiment", "split", "model", "rmse", "mae", "r2", "spearman"]].head(20)),
        "",
        "## Focused Subset Winners",
        "",
        markdown_table(subset_focus[["experiment", "split", "model", "n_rows", "n_assays", "n_documents", "rmse", "mae", "r2", "spearman"]]),
        "",
        "## Interpretation",
        "",
        "- The major families are covalent labeling and nucleotide-exchange assays; treating them as one label is noisy.",
        "- In-sample assay/document median baselines quantify how much of the apparent signal is assay/document offset.",
        "- Focused subset metrics are the most relevant for analog work inside one assay family or one patent series.",
        "- Centered labels are useful as a diagnostic for within-assay SAR, but they are not directly deployable for prospective absolute IC50 unless the target assay baseline is known.",
        "",
    ]
    (out_dir / "ASSAY_FAMILY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assay-family stratified G12C QSAR experiments.")
    parser.add_argument("--run-dir", default="", help="Existing run directory with raw and cleaned G12C CSVs.")
    parser.add_argument("--runs-root", default="activity_modeling/runs")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <run-dir>/assay_family.")
    parser.add_argument("--random-seed", type=int, default=714)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-subset-size", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(Path(args.runs_root))
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "assay_family"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_family_dataset(run_dir)
    dataset.to_csv(out_dir / "g12c_ic50_assay_family_dataset.csv", index=False)
    summary_tables(dataset, out_dir)
    median_baseline_metrics(dataset, out_dir)
    run_experiments(dataset, out_dir, random_seed=args.random_seed, test_size=args.test_size, min_subset_size=args.min_subset_size)
    write_report(dataset, out_dir)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "rows": int(len(dataset)),
        "assay_families": int(dataset["assay_family"].nunique()),
        "broad_assay_families": int(dataset["assay_family_broad"].nunique()),
        "outputs": {
            "dataset": str(out_dir / "g12c_ic50_assay_family_dataset.csv"),
            "family_summary": str(out_dir / "assay_family_summary.csv"),
            "metrics": str(out_dir / "assay_family_experiment_metrics.csv"),
            "best": str(out_dir / "assay_family_best_by_experiment.csv"),
            "report": str(out_dir / "ASSAY_FAMILY_REPORT.md"),
        },
    }
    (out_dir / "assay_family_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote assay-family outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
