#!/usr/bin/env python3
"""Evaluate the frozen KRAS G12C standard experiment with bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_g12c_pipeline import (  # noqa: E402
    TOP_SPECS,
    add_property_columns,
    add_train_only_nearest_neighbors,
    load_dataset,
    markdown_table,
    ranking_metric_rows,
)
from predict_g12c_activity import applicability_flag  # noqa: E402
from rank_g12c_candidates import fixed_scale, robust_scale, to_numeric  # noqa: E402


PRIMARY_QSAR_MODEL = "ensemble_nn_ridge_lgbm"
ACTIVE_THRESHOLD = 7.0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen G12C standard experiment.")
    parser.add_argument("--run-dir", default="activity_modeling/runs/g12c_expanded_20260612_122903")
    parser.add_argument("--experiment-dir", default="", help="Defaults to <run-dir>/standard_experiment.")
    parser.add_argument("--output-dir", default="", help="Defaults to <experiment-dir>/evaluation.")
    parser.add_argument("--docking-csv", default="", help="Optional docking_results.csv for frozen split molecules.")
    parser.add_argument("--docking-score-column", default="score")
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=714)
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")
    return pd.read_csv(path)


def best_models(metrics: pd.DataFrame) -> dict[str, str]:
    best = (
        metrics.sort_values(["split", "rmse", "mae"], ascending=[True, True, True])
        .groupby("split", as_index=False)
        .first()[["split", "model"]]
    )
    return dict(zip(best["split"], best["model"]))


def prediction_map(predictions: pd.DataFrame, split: str, model: str) -> pd.Series:
    subset = predictions[(predictions["split"] == split) & (predictions["prediction_model"] == model)]
    return subset.drop_duplicates("canonical_smiles").set_index("canonical_smiles")["predicted_p_activity"]


def load_docking_scores(path: str, score_column: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    docking_path = Path(path)
    if not docking_path.exists():
        raise RuntimeError(f"Docking CSV not found: {docking_path}")
    docking = pd.read_csv(docking_path)
    if score_column not in docking.columns:
        raise RuntimeError(f"Docking score column {score_column!r} not found in {docking_path}")
    smiles_col = next((column for column in ("canonical_smiles", "SMILES", "smiles") if column in docking.columns), "")
    keep = [column for column in ["compound_id", smiles_col, score_column, "status", "pose_ref", "message"] if column]
    out = docking[keep].copy()
    out = out.rename(columns={score_column: "vina_score"})
    if smiles_col and smiles_col != "canonical_smiles":
        out = out.rename(columns={smiles_col: "canonical_smiles"})
    out["vina_score"] = to_numeric(out["vina_score"])
    return out


def merge_docking(frame: pd.DataFrame, docking: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if docking.empty:
        out["vina_score"] = np.nan
        out["docking_status"] = ""
        return out
    if "compound_id" in docking.columns:
        docking_by_id = docking.drop_duplicates("compound_id")
        out = out.merge(
            docking_by_id[["compound_id", "vina_score", "status", "pose_ref"] if "pose_ref" in docking_by_id.columns else ["compound_id", "vina_score", "status"]],
            on="compound_id",
            how="left",
            suffixes=("", "_dock"),
        )
    elif "canonical_smiles" in docking.columns:
        docking_by_smiles = docking.drop_duplicates("canonical_smiles")
        out = out.merge(
            docking_by_smiles[["canonical_smiles", "vina_score", "status", "pose_ref"] if "pose_ref" in docking_by_smiles.columns else ["canonical_smiles", "vina_score", "status"]],
            on="canonical_smiles",
            how="left",
            suffixes=("", "_dock"),
        )
    else:
        out["vina_score"] = np.nan
        out["status"] = ""
    out = out.rename(columns={"status": "docking_status"})
    if "vina_score" not in out.columns:
        out["vina_score"] = np.nan
    if "docking_status" not in out.columns:
        out["docking_status"] = ""
    return out


def build_scores(
    run_dir: Path,
    split_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    docking: pd.DataFrame,
) -> pd.DataFrame:
    dataset = load_dataset(run_dir)
    best_by_split = best_models(metrics)
    rows: list[pd.DataFrame] = []
    for split, group in split_manifest.groupby("split", sort=True):
        split_test_smiles = set(predictions.loc[predictions["split"] == split, "canonical_smiles"].astype(str))
        train_df = dataset[~dataset["canonical_smiles"].astype(str).isin(split_test_smiles)].copy()
        scored = group.copy().reset_index(drop=True)
        scored["p_activity"] = scored["pIC50"].astype(float)
        scored = add_property_columns(scored)

        nn = add_train_only_nearest_neighbors(scored, train_df)
        for column in nn.columns:
            scored[column] = nn[column].to_numpy()
        scored["g12c_applicability_domain"] = scored["nearest_neighbor_similarity"].map(applicability_flag)

        primary_pred = prediction_map(predictions, split, PRIMARY_QSAR_MODEL)
        scored["qsar_primary_pIC50"] = scored["canonical_smiles"].map(primary_pred)
        oracle_model = best_by_split.get(split, PRIMARY_QSAR_MODEL)
        oracle_pred = prediction_map(predictions, split, oracle_model)
        scored["qsar_oracle_best_rmse_pIC50"] = scored["canonical_smiles"].map(oracle_pred)
        scored["qsar_oracle_best_rmse_model"] = oracle_model

        scored = merge_docking(scored, docking)
        scored["activity_component"] = fixed_scale(scored["qsar_primary_pIC50"], 5.5, 8.5)
        scored["nn_component"] = fixed_scale(scored["nearest_neighbor_pIC50"], 5.5, 8.5) * fixed_scale(
            scored["nearest_neighbor_similarity"], 0.30, 0.75
        )
        scored["docking_component"] = robust_scale(scored["vina_score"], lower_is_better=True)
        scored["property_score"] = scored["property_component"]

        scored["score_random"] = np.nan
        scored["score_qsar_primary"] = scored["qsar_primary_pIC50"]
        scored["score_nn_train_only"] = scored["nearest_neighbor_pIC50"]
        scored["score_qsar_plus_nn"] = 0.70 * scored["activity_component"] + 0.30 * scored["nn_component"]
        scored["score_docking_only"] = scored["docking_component"]
        scored["score_qsar_plus_docking"] = 0.65 * scored["activity_component"] + 0.35 * scored["docking_component"]
        scored["score_qsar_nn_docking"] = (
            0.50 * scored["activity_component"] + 0.25 * scored["nn_component"] + 0.25 * scored["docking_component"]
        )
        scored["score_balanced_qsar_nn_docking_property"] = (
            0.40 * scored["activity_component"]
            + 0.20 * scored["nn_component"]
            + 0.25 * scored["docking_component"]
            + 0.15 * scored["property_component"]
        )
        scored["score_oracle_best_rmse_diagnostic"] = scored["qsar_oracle_best_rmse_pIC50"]
        rows.append(scored)
    return pd.concat(rows, ignore_index=True)


def strategy_columns() -> dict[str, str]:
    return {
        "qsar_primary_fixed": "score_qsar_primary",
        "nn_train_only": "score_nn_train_only",
        "qsar_plus_nn_fixed": "score_qsar_plus_nn",
        "docking_only": "score_docking_only",
        "qsar_plus_docking_fixed": "score_qsar_plus_docking",
        "qsar_nn_docking_fixed": "score_qsar_nn_docking",
        "balanced_qsar_nn_docking_property": "score_balanced_qsar_nn_docking_property",
        "oracle_best_rmse_diagnostic": "score_oracle_best_rmse_diagnostic",
    }


def point_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in scores.groupby("split", sort=True):
        rows.extend(random_expected_rows_for_group(group, split))
        for strategy, column in strategy_columns().items():
            rows.extend(
                ranking_metric_rows(
                    group,
                    score_column=column,
                    strategy=strategy,
                    split=split,
                    source="standard_experiment",
                )
            )
    return pd.DataFrame(rows)


def random_expected_rows_for_group(group: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(group)
    active_count = int((group["p_activity"] >= ACTIVE_THRESHOLD).sum())
    base_rate = float(active_count / n) if n else np.nan
    for top_name, kind, value in TOP_SPECS:
        k = min(n, max(1, int(value) if kind == "absolute" else int(np.ceil(n * value))))
        rows.append(
            {
                "source": "standard_experiment",
                "split": split,
                "strategy": "random_expected",
                "score_column": "",
                "active_threshold": ACTIVE_THRESHOLD,
                "top_spec": top_name,
                "n": n,
                "active_count": active_count,
                "base_rate": base_rate,
                "k": k,
                "hits": float(k * base_rate) if np.isfinite(base_rate) else np.nan,
                "precision": base_rate,
                "recall": float(k * base_rate / active_count) if active_count else np.nan,
                "enrichment": 1.0 if active_count else np.nan,
                "roc_auc": 0.5 if 0 < active_count < n else np.nan,
                "average_precision": base_rate if 0 < active_count < n else np.nan,
            }
        )
    return rows


def top_k_value(n: int, top_spec: str) -> int:
    spec = {name: (kind, value) for name, kind, value in TOP_SPECS}[top_spec]
    kind, value = spec
    return min(n, max(1, int(value) if kind == "absolute" else int(np.ceil(n * value))))


def metric_once(group: pd.DataFrame, score_column: str, top_spec: str) -> dict[str, float]:
    frame = group[["p_activity", score_column]].copy()
    frame["p_activity"] = to_numeric(frame["p_activity"])
    frame[score_column] = to_numeric(frame[score_column])
    frame = frame[np.isfinite(frame["p_activity"]) & np.isfinite(frame[score_column])].copy()
    n = len(frame)
    if n == 0:
        return {"precision": np.nan, "recall": np.nan, "enrichment": np.nan, "roc_auc": np.nan, "average_precision": np.nan}
    y = (frame["p_activity"].to_numpy(dtype=float) >= ACTIVE_THRESHOLD).astype(int)
    scores = frame[score_column].to_numpy(dtype=float)
    active_count = int(y.sum())
    base_rate = float(active_count / n) if n else np.nan
    ranked = frame.assign(active=y).sort_values(score_column, ascending=False).reset_index(drop=True)
    k = top_k_value(n, top_spec)
    hits = int(ranked["active"].iloc[:k].sum())
    precision = float(hits / k)
    recall = float(hits / active_count) if active_count else np.nan
    enrichment = float(precision / base_rate) if base_rate else np.nan
    if len(np.unique(y)) < 2:
        roc_auc = np.nan
        ap = np.nan
    else:
        roc_auc = float(roc_auc_score(y, scores))
        ap = float(average_precision_score(y, scores))
    return {
        "precision": precision,
        "recall": recall,
        "enrichment": enrichment,
        "roc_auc": roc_auc,
        "average_precision": ap,
    }


def bootstrap_ci(
    scores: pd.DataFrame,
    *,
    n_iters: int,
    random_seed: int,
    top_specs: tuple[str, ...] = ("top_5pct", "top_10pct", "top_20"),
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for split, split_group in scores.groupby("split", sort=True):
        for strategy, column in strategy_columns().items():
            valid = split_group[np.isfinite(to_numeric(split_group[column]))].copy()
            if valid.empty:
                continue
            for top_spec in top_specs:
                point = metric_once(valid, column, top_spec)
                boot_values: dict[str, list[float]] = {metric: [] for metric in point}
                indices = np.arange(len(valid))
                for _ in range(n_iters):
                    sample_indices = rng.choice(indices, size=len(indices), replace=True)
                    sample = valid.iloc[sample_indices]
                    values = metric_once(sample, column, top_spec)
                    for metric, value in values.items():
                        if np.isfinite(value):
                            boot_values[metric].append(float(value))
                for metric, values in boot_values.items():
                    if values:
                        low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
                    else:
                        low, high = np.nan, np.nan
                    rows.append(
                        {
                            "split": split,
                            "strategy": strategy,
                            "score_column": column,
                            "active_threshold": ACTIVE_THRESHOLD,
                            "top_spec": top_spec,
                            "metric": metric,
                            "point": point[metric],
                            "ci_low": float(low) if np.isfinite(low) else np.nan,
                            "ci_high": float(high) if np.isfinite(high) else np.nan,
                            "bootstrap_iters": n_iters,
                            "n": int(len(valid)),
                            "active_count": int((valid["p_activity"] >= ACTIVE_THRESHOLD).sum()),
                        }
                    )
    return pd.DataFrame(rows)


def coverage_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in scores.groupby("split", sort=True):
        docked = np.isfinite(to_numeric(group["vina_score"]))
        rows.append(
            {
                "split": split,
                "rows": int(len(group)),
                "actives": int((group["benchmark_label"] == "active").sum()),
                "inactives": int((group["benchmark_label"] == "inactive").sum()),
                "docked_rows": int(docked.sum()),
                "docking_coverage": float(docked.mean()) if len(group) else 0.0,
                "docked_actives": int(((group["benchmark_label"] == "active") & docked).sum()),
                "docked_inactives": int(((group["benchmark_label"] == "inactive") & docked).sum()),
            }
        )
    return pd.DataFrame(rows)


def selected_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    keep_strategies = [
        "random_expected",
        "qsar_primary_fixed",
        "nn_train_only",
        "qsar_plus_nn_fixed",
        "docking_only",
        "qsar_plus_docking_fixed",
        "qsar_nn_docking_fixed",
        "balanced_qsar_nn_docking_property",
    ]
    return metrics[
        (metrics["active_threshold"] == ACTIVE_THRESHOLD)
        & (metrics["top_spec"].isin(["top_5pct", "top_10pct", "top_20"]))
        & (metrics["strategy"].isin(keep_strategies))
    ].copy()


def build_report(
    output_dir: Path,
    coverage: pd.DataFrame,
    metrics: pd.DataFrame,
    ci: pd.DataFrame,
    docking_csv: str,
) -> None:
    top5 = selected_metric_table(metrics)
    ci_focus = ci[
        (ci["top_spec"] == "top_5pct")
        & (ci["metric"].isin(["precision", "enrichment", "roc_auc", "average_precision"]))
        & (ci["strategy"].isin(["qsar_primary_fixed", "qsar_plus_nn_fixed", "docking_only", "qsar_nn_docking_fixed"]))
    ].copy()
    report = [
        "# Standard G12C Experiment Evaluation",
        "",
        "## Scope",
        "",
        "- Frozen binary benchmark: active pIC50 >= 7, inactive pIC50 <= 6.",
        "- Frozen splits are derived from the existing random/scaffold/document/assay holdouts.",
        f"- Primary QSAR model is fixed to `{PRIMARY_QSAR_MODEL}`.",
        "- `oracle_best_rmse_diagnostic` is diagnostic only and should not be used as final evidence.",
        "",
        "## Docking Coverage",
        "",
        f"- Docking CSV: {docking_csv or 'not provided'}",
        "",
        markdown_table(coverage, ["split", "rows", "actives", "inactives", "docked_rows", "docking_coverage", "docked_actives", "docked_inactives"], max_rows=20),
        "",
        "## Point Metrics",
        "",
        markdown_table(
            top5,
            ["split", "strategy", "top_spec", "n", "active_count", "k", "hits", "precision", "recall", "enrichment", "roc_auc", "average_precision"],
            max_rows=120,
        ),
        "",
        "## Bootstrap 95% CI",
        "",
        markdown_table(
            ci_focus,
            ["split", "strategy", "top_spec", "metric", "point", "ci_low", "ci_high", "n", "active_count"],
            max_rows=120,
        ),
        "",
        "## Interpretation Rules",
        "",
        "- Final scientific claims should use fixed strategies, not split-oracle model selection.",
        "- Docking claims require high docking coverage on each frozen split.",
        "- Small pilot docking runs can validate plumbing and signal direction, but do not establish final benchmark performance.",
        "- Report confidence intervals for top-k enrichment and AUC/AP before claiming improvement over QSAR-only.",
        "",
    ]
    (output_dir / "STANDARD_EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else run_dir / "standard_experiment"
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else experiment_dir / "evaluation")

    split_manifest = load_required_csv(experiment_dir / "frozen_split_manifest.csv")
    predictions = load_required_csv(run_dir / "predictions_all_splits.csv")
    metrics = load_required_csv(run_dir / "metrics.csv")
    docking = load_docking_scores(args.docking_csv, args.docking_score_column)

    scores = build_scores(run_dir, split_manifest, predictions, metrics, docking)
    scores.to_csv(output_dir / "standard_scores.csv", index=False)
    coverage = coverage_summary(scores)
    coverage.to_csv(output_dir / "docking_coverage.csv", index=False)
    metrics_df = point_metrics(scores)
    metrics_df.to_csv(output_dir / "standard_point_metrics.csv", index=False)
    ci = bootstrap_ci(scores, n_iters=args.bootstrap_iters, random_seed=args.random_seed)
    ci.to_csv(output_dir / "standard_bootstrap_ci.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "primary_qsar_model": PRIMARY_QSAR_MODEL,
        "docking_csv": args.docking_csv,
        "bootstrap_iters": args.bootstrap_iters,
        "splits": coverage.to_dict(orient="records"),
    }
    write_json(output_dir / "standard_evaluation_summary.json", summary)
    build_report(output_dir, coverage, metrics_df, ci, args.docking_csv)
    print(f"Wrote standard evaluation outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
