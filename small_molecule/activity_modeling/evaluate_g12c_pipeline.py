#!/usr/bin/env python3
"""Retrospective evaluation for the KRAS G12C screening workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from predict_g12c_activity import applicability_flag, canonicalize_smiles, nearest_neighbors  # noqa: E402
from rank_g12c_candidates import (  # noqa: E402
    DOMAIN_COMPONENTS,
    build_risk_flags,
    first_existing,
    fixed_scale,
    property_component,
    rdkit_candidate_properties,
    robust_scale,
    to_numeric,
    weighted_score,
)
from train_g12c_qsar import HAS_RDKIT, Chem  # noqa: E402


ACTIVE_THRESHOLDS = (6.0, 7.0)
TOP_SPECS = (
    ("top_10", "absolute", 10.0),
    ("top_20", "absolute", 20.0),
    ("top_50", "absolute", 50.0),
    ("top_5pct", "fraction", 0.05),
    ("top_10pct", "fraction", 0.10),
)
ID_COLUMNS = ("compound_id", "Compound", "compound", "Cmpd", "id", "ID")
DOCKING_SCORE_COLUMNS = (
    "vina_score_kcal_mol",
    "vina_score",
    "docking_score",
    "docking_affinity",
    "affinity",
    "gnina_score",
    "score",
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate G12C QSAR/ranking workflow on retrospective holdouts.")
    parser.add_argument("--run-dir", default="activity_modeling/runs/g12c_expanded_20260612_122903")
    parser.add_argument("--output-dir", default="", help="Defaults to <run-dir>/evaluation.")
    parser.add_argument(
        "--candidate-csv",
        default="output/formula_workflow_outputs/reversible_kras_g13d_inhibitors_formula_to_smiles.csv",
        help="Optional current workflow candidate CSV for data-link QA.",
    )
    parser.add_argument("--docking-csv", default="", help="Optional retrospective docking CSV with SMILES and scores.")
    parser.add_argument("--docking-score-column", default="")
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [column for column in columns if column in df.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    widths = {column: max(len(str(column)), *(len(str(value)) for value in view[column].tolist())) for column in view.columns}
    header = "| " + " | ".join(f"{column:<{widths[column]}}" for column in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[column] for column in view.columns) + " |"
    rows = [
        "| " + " | ".join(f"{str(row[column]):<{widths[column]}}" for column in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def top_k_value(n_rows: int, spec_kind: str, value: float) -> int:
    if spec_kind == "absolute":
        return min(n_rows, max(1, int(value)))
    return min(n_rows, max(1, int(math.ceil(n_rows * value))))


def safe_auc(y_binary: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(y_binary) == 0 or len(np.unique(y_binary)) < 2:
        return np.nan, np.nan
    try:
        roc_auc = float(roc_auc_score(y_binary, scores))
    except Exception:
        roc_auc = np.nan
    try:
        avg_precision = float(average_precision_score(y_binary, scores))
    except Exception:
        avg_precision = np.nan
    return roc_auc, avg_precision


def regression_summary(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    y = to_numeric(y_true)
    score = to_numeric(y_score)
    mask = np.isfinite(y) & np.isfinite(score)
    if int(mask.sum()) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan}
    yv = y.loc[mask].to_numpy(dtype=float)
    sv = score.loc[mask].to_numpy(dtype=float)
    if len(yv) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            rho = spearmanr(yv, sv).statistic
    else:
        rho = np.nan
    return {
        "rmse": float(np.sqrt(mean_squared_error(yv, sv))),
        "mae": float(mean_absolute_error(yv, sv)),
        "r2": float(r2_score(yv, sv)) if len(yv) > 1 else np.nan,
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def ranking_metric_rows(
    frame: pd.DataFrame,
    *,
    score_column: str,
    strategy: str,
    split: str,
    source: str,
) -> list[dict[str, Any]]:
    if score_column not in frame.columns:
        return []
    y = to_numeric(frame["p_activity"])
    score = to_numeric(frame[score_column])
    mask = np.isfinite(y) & np.isfinite(score)
    data = frame.loc[mask, ["canonical_smiles", "p_activity", score_column]].copy()
    data["p_activity"] = to_numeric(data["p_activity"])
    data[score_column] = to_numeric(data[score_column])
    n = len(data)
    if n == 0:
        return []
    rows: list[dict[str, Any]] = []
    for threshold in ACTIVE_THRESHOLDS:
        active = data["p_activity"].to_numpy(dtype=float) >= threshold
        active_count = int(active.sum())
        base_rate = float(active.mean()) if n else np.nan
        roc_auc, avg_precision = safe_auc(active.astype(int), data[score_column].to_numpy(dtype=float))
        ranked = data.sort_values([score_column, "canonical_smiles"], ascending=[False, True]).reset_index(drop=True)
        ranked_active = ranked["p_activity"].to_numpy(dtype=float) >= threshold
        for top_name, top_kind, top_value in TOP_SPECS:
            k = top_k_value(n, top_kind, top_value)
            top_active = ranked_active[:k]
            hits = int(top_active.sum())
            precision = float(hits / k) if k else np.nan
            recall = float(hits / active_count) if active_count else np.nan
            enrichment = float(precision / base_rate) if base_rate and np.isfinite(base_rate) else np.nan
            rows.append(
                {
                    "source": source,
                    "split": split,
                    "strategy": strategy,
                    "score_column": score_column,
                    "active_threshold": threshold,
                    "top_spec": top_name,
                    "n": n,
                    "active_count": active_count,
                    "base_rate": base_rate,
                    "k": k,
                    "hits": hits,
                    "precision": precision,
                    "recall": recall,
                    "enrichment": enrichment,
                    "roc_auc": roc_auc,
                    "average_precision": avg_precision,
                }
            )
    return rows


def random_expected_rows(frame: pd.DataFrame, *, split: str, source: str) -> list[dict[str, Any]]:
    y = to_numeric(frame["p_activity"])
    y = y[np.isfinite(y)]
    n = len(y)
    if n == 0:
        return []
    rows: list[dict[str, Any]] = []
    for threshold in ACTIVE_THRESHOLDS:
        active_count = int((y >= threshold).sum())
        base_rate = float(active_count / n)
        for top_name, top_kind, top_value in TOP_SPECS:
            k = top_k_value(n, top_kind, top_value)
            expected_hits = float(k * base_rate)
            rows.append(
                {
                    "source": source,
                    "split": split,
                    "strategy": "random_expected",
                    "score_column": "",
                    "active_threshold": threshold,
                    "top_spec": top_name,
                    "n": n,
                    "active_count": active_count,
                    "base_rate": base_rate,
                    "k": k,
                    "hits": expected_hits,
                    "precision": base_rate,
                    "recall": float(expected_hits / active_count) if active_count else np.nan,
                    "enrichment": 1.0 if active_count else np.nan,
                    "roc_auc": 0.5 if 0 < active_count < n else np.nan,
                    "average_precision": base_rate if 0 < active_count < n else np.nan,
                }
            )
    return rows


def load_dataset(run_dir: Path) -> pd.DataFrame:
    assay_family = run_dir / "assay_family" / "g12c_ic50_assay_family_dataset.csv"
    path = assay_family if assay_family.exists() else run_dir / "g12c_ic50_dataset.csv"
    df = pd.read_csv(path)
    df["canonical_smiles_norm"] = df["canonical_smiles"].map(canonicalize_smiles)
    return df[df["canonical_smiles_norm"] != ""].reset_index(drop=True)


def dataset_quality(dataset: pd.DataFrame, raw_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "raw_activity_rows": int(len(raw_df)),
        "cleaned_unique_smiles": int(len(dataset)),
        "valid_canonical_smiles": int((dataset["canonical_smiles_norm"] != "").sum()),
        "primary_assays": int(dataset["primary_assay_chembl_id"].nunique()),
        "primary_documents": int(dataset["primary_document_chembl_id"].nunique()),
        "p_activity_min": float(dataset["p_activity"].min()),
        "p_activity_median": float(dataset["p_activity"].median()),
        "p_activity_max": float(dataset["p_activity"].max()),
        "active_rate_pIC50_ge_6": float((dataset["p_activity"] >= 6.0).mean()),
        "active_rate_pIC50_ge_7": float((dataset["p_activity"] >= 7.0).mean()),
        "multi_record_fraction": float((dataset["n_records"] > 1).mean()),
        "conflicting_record_fraction_std_ge_0_5": float((dataset["p_activity_std"] >= 0.5).mean()),
        "largest_document_fraction": float(dataset["primary_document_chembl_id"].value_counts(normalize=True).iloc[0]),
        "largest_assay_fraction": float(dataset["primary_assay_chembl_id"].value_counts(normalize=True).iloc[0]),
    }
    for column in (
        "warhead_acrylamide_like",
        "warhead_chloroacetamide_like",
        "warhead_vinyl_sulfonamide_like",
        "warhead_cyanoacrylamide_like",
    ):
        if column in dataset.columns:
            summary[f"{column}_fraction"] = float(to_numeric(dataset[column]).fillna(0).mean())
    if "assay_family" in dataset.columns:
        summary["assay_families"] = int(dataset["assay_family"].nunique())
    return summary


def evaluate_candidate_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        return pd.DataFrame(), {"candidate_csv": str(path), "status": "missing"}
    df = pd.read_csv(path)
    smiles_col = first_existing(df.columns, ("SMILES", "smiles", "canonical_smiles", "selected_smiles", "full_smiles"))
    id_col = first_existing(df.columns, ID_COLUMNS)
    rows: list[dict[str, Any]] = []
    formula_matches = 0
    valid = 0
    duplicate_canonical = 0
    canon_values: list[str] = []
    for _, row in df.iterrows():
        smiles = str(row.get(smiles_col, "") or "")
        canonical = canonicalize_smiles(smiles)
        formula = str(row.get("Neutral_formula", "") or "")
        calc_formula = ""
        formula_match = False
        if HAS_RDKIT and canonical:
            from rdkit.Chem import rdMolDescriptors

            mol = Chem.MolFromSmiles(canonical)
            if mol is not None:
                calc_formula = rdMolDescriptors.CalcMolFormula(mol)
                formula_match = bool(formula and calc_formula == formula)
        if canonical:
            valid += 1
            canon_values.append(canonical)
        if formula_match:
            formula_matches += 1
        rows.append(
            {
                "compound_id": row.get(id_col, "") if id_col else "",
                "input_smiles": smiles,
                "canonical_smiles": canonical,
                "valid_smiles": bool(canonical),
                "neutral_formula": formula,
                "rdkit_formula": calc_formula,
                "formula_match": formula_match,
                "bindingdb_id_present": bool(str(row.get("BindingDB_ID", "") or "").strip()),
                "activity_nM_present": pd.notna(row.get("Activity_nM")) and str(row.get("Activity_nM", "")).strip() != "",
                "note": row.get("Note", ""),
            }
        )
    duplicate_canonical = int(pd.Series(canon_values).duplicated().sum()) if canon_values else 0
    summary = {
        "candidate_csv": str(path),
        "rows": int(len(df)),
        "smiles_column": smiles_col,
        "valid_smiles": valid,
        "valid_smiles_rate": float(valid / len(df)) if len(df) else np.nan,
        "duplicate_canonical_smiles": duplicate_canonical,
        "formula_match_rate_among_rows": float(formula_matches / len(df)) if len(df) else np.nan,
        "bindingdb_id_present_rate": float(pd.Series([r["bindingdb_id_present"] for r in rows]).mean()) if rows else np.nan,
        "activity_nM_present_rate": float(pd.Series([r["activity_nM_present"] for r in rows]).mean()) if rows else np.nan,
        "corrected_note_fraction": float(df.get("Note", pd.Series(dtype=str)).fillna("").astype(str).str.contains("corrected", case=False).mean())
        if "Note" in df.columns
        else np.nan,
    }
    return pd.DataFrame(rows), summary


def add_train_only_nearest_neighbors(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    train = train_df.copy()
    train["canonical_smiles_norm"] = train["canonical_smiles"].map(canonicalize_smiles)
    nn = nearest_neighbors(test_df["canonical_smiles"].astype(str).tolist(), train)
    return nn.reset_index(drop=True)


def add_property_columns(frame: pd.DataFrame) -> pd.DataFrame:
    props = rdkit_candidate_properties(frame["canonical_smiles"].astype(str))
    out = frame.reset_index(drop=True).copy()
    for column in props.columns:
        out[column] = props[column].to_numpy()
    out["property_component"] = property_component(out)
    out["warhead_component"] = to_numeric(out.get("has_covalent_warhead_like", pd.Series(0, index=out.index))).fillna(0.0)
    return out


def prediction_lookup(predictions: pd.DataFrame, split: str, model: str) -> pd.Series:
    subset = predictions[(predictions["split"] == split) & (predictions["prediction_model"] == model)]
    return subset.drop_duplicates("canonical_smiles").set_index("canonical_smiles")["predicted_p_activity"]


def make_ranker_scores(frame: pd.DataFrame, qsar_col: str, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out[f"{prefix}_activity_component"] = fixed_scale(out[qsar_col], 5.5, 8.5)
    out[f"{prefix}_nearest_neighbor_component"] = fixed_scale(out["nearest_neighbor_pIC50"], 5.5, 8.5) * fixed_scale(
        out["nearest_neighbor_similarity"], 0.30, 0.75
    )
    out[f"{prefix}_applicability_component"] = out["g12c_applicability_domain"].map(DOMAIN_COMPONENTS).fillna(0.0)
    out[f"{prefix}_qsar_nn_score"] = (
        0.70 * out[f"{prefix}_activity_component"] + 0.30 * out[f"{prefix}_nearest_neighbor_component"]
    )
    out[f"{prefix}_qsar_nn_domain_score"] = (
        0.55 * out[f"{prefix}_activity_component"]
        + 0.25 * out[f"{prefix}_nearest_neighbor_component"]
        + 0.20 * out[f"{prefix}_applicability_component"]
    )
    temp = out.copy()
    temp["activity_component"] = out[f"{prefix}_activity_component"]
    temp["nearest_neighbor_component"] = out[f"{prefix}_nearest_neighbor_component"]
    temp["applicability_component"] = out[f"{prefix}_applicability_component"]
    temp["g12c_predicted_pIC50"] = out[qsar_col]
    weights = {
        "activity": 0.35,
        "docking": 0.25,
        "applicability": 0.15,
        "property": 0.15,
        "nearest_neighbor": 0.10,
        "warhead": 0.05,
    }
    temp["raw_multi_objective_score"] = weighted_score(temp, weights)
    _flags, penalty = build_risk_flags(temp, "", prefer_covalent=True)
    out[f"{prefix}_ranker_no_docking_score"] = (temp["raw_multi_objective_score"] - penalty).clip(0.0, 1.0)
    return out


def evaluate_model_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, model), group in predictions.groupby(["split", "prediction_model"]):
        rows.extend(
            ranking_metric_rows(
                group,
                score_column="predicted_p_activity",
                strategy=model,
                split=str(split),
                source="model_predictions",
            )
        )
    return pd.DataFrame(rows)


def evaluate_metadata_baselines(dataset: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_smiles = {
        split: set(group["canonical_smiles"].astype(str).unique())
        for split, group in predictions.groupby("split")
    }
    group_columns = [
        ("global_train_median", ""),
        ("train_median_by_assay_family_broad", "assay_family_broad"),
        ("train_median_by_assay_family", "assay_family"),
        ("train_median_by_primary_document", "primary_document_chembl_id"),
        ("train_median_by_primary_assay", "primary_assay_chembl_id"),
    ]
    for split, test_smiles in split_smiles.items():
        test = dataset[dataset["canonical_smiles"].isin(test_smiles)].copy()
        train = dataset[~dataset["canonical_smiles"].isin(test_smiles)].copy()
        global_median = float(train["p_activity"].median())
        for strategy, group_col in group_columns:
            scored = test[["canonical_smiles", "p_activity"]].copy()
            if not group_col or group_col not in train.columns or group_col not in test.columns:
                scored["score"] = global_median
            else:
                medians = train.groupby(group_col)["p_activity"].median()
                scored["score"] = test[group_col].map(medians).fillna(global_median).to_numpy()
            reg = regression_summary(scored["p_activity"], scored["score"])
            base = {
                "source": "metadata_baseline",
                "split": split,
                "strategy": strategy,
                "group_column": group_col,
                **reg,
            }
            rows.append(base)
    return pd.DataFrame(rows)


def evaluate_ablation(dataset: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_by_rmse = (
        metrics.sort_values(["split", "rmse", "mae"], ascending=[True, True, True])
        .groupby("split", as_index=False)
        .first()[["split", "model"]]
        .set_index("split")["model"]
        .to_dict()
    )
    split_smiles = {
        split: set(group["canonical_smiles"].astype(str).unique())
        for split, group in predictions.groupby("split")
    }
    score_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fixed_models = ["ensemble_nn_ridge_lgbm", "rdkit_desc_elastic_net", "morgan_tanimoto_knn_k5"]

    for split, test_smiles in split_smiles.items():
        test = dataset[dataset["canonical_smiles"].isin(test_smiles)].copy().reset_index(drop=True)
        train = dataset[~dataset["canonical_smiles"].isin(test_smiles)].copy().reset_index(drop=True)
        scored = add_property_columns(test[["canonical_smiles", "p_activity"]].merge(dataset, on=["canonical_smiles", "p_activity"], how="left"))
        nn = add_train_only_nearest_neighbors(scored, train)
        for column in nn.columns:
            scored[column] = nn[column].to_numpy()
        scored["g12c_applicability_domain"] = scored["nearest_neighbor_similarity"].map(applicability_flag)
        scored["nearest_neighbor_score"] = scored["nearest_neighbor_pIC50"]
        scored["nearest_neighbor_component_score"] = fixed_scale(scored["nearest_neighbor_pIC50"], 5.5, 8.5) * fixed_scale(
            scored["nearest_neighbor_similarity"], 0.30, 0.75
        )
        scored["applicability_score"] = to_numeric(scored["nearest_neighbor_similarity"])
        scored["property_score"] = scored["property_component"]
        scored["warhead_score"] = scored["warhead_component"]
        metric_rows.extend(random_expected_rows(scored, split=str(split), source="ablation"))

        model_map: dict[str, str] = {}
        best_model = best_by_rmse.get(split)
        if best_model:
            model_map["qsar_best_rmse"] = best_model
        for model in fixed_models:
            if model in set(predictions.loc[predictions["split"] == split, "prediction_model"]):
                model_map[f"qsar_{model}"] = model

        for strategy_prefix, model in model_map.items():
            pred = prediction_lookup(predictions, str(split), model)
            scored[f"{strategy_prefix}_pIC50"] = scored["canonical_smiles"].map(pred)
            scored = make_ranker_scores(scored, f"{strategy_prefix}_pIC50", strategy_prefix)
            strategy_columns = {
                strategy_prefix: f"{strategy_prefix}_pIC50",
                f"{strategy_prefix}_plus_nn": f"{strategy_prefix}_qsar_nn_score",
                f"{strategy_prefix}_plus_nn_domain": f"{strategy_prefix}_qsar_nn_domain_score",
                f"{strategy_prefix}_ranker_no_docking": f"{strategy_prefix}_ranker_no_docking_score",
            }
            for strategy, score_col in strategy_columns.items():
                metric_rows.extend(
                    ranking_metric_rows(
                        scored,
                        score_column=score_col,
                        strategy=strategy,
                        split=str(split),
                        source="ablation",
                    )
                )

        for strategy, score_col in {
            "nearest_neighbor_pIC50_train_only": "nearest_neighbor_score",
            "nearest_neighbor_component_train_only": "nearest_neighbor_component_score",
            "applicability_similarity_only": "applicability_score",
            "property_only": "property_score",
            "warhead_only": "warhead_score",
        }.items():
            metric_rows.extend(
                ranking_metric_rows(
                    scored,
                    score_column=score_col,
                    strategy=strategy,
                    split=str(split),
                    source="ablation",
                )
            )
        scored["split"] = split
        score_frames.append(scored)

    return pd.concat(score_frames, ignore_index=True), pd.DataFrame(metric_rows)


def evaluate_docking_csv(path: Path, dataset: pd.DataFrame, score_column: str = "") -> tuple[pd.DataFrame, dict[str, Any]]:
    if not str(path) or path.is_dir() or not path.exists():
        return pd.DataFrame(), {"status": "missing", "docking_csv": str(path) if path else ""}
    docking = pd.read_csv(path)
    smiles_col = first_existing(docking.columns, ("canonical_smiles", "SMILES", "smiles", "selected_smiles"))
    if not smiles_col:
        return pd.DataFrame(), {"status": "no_smiles_column", "docking_csv": str(path)}
    score_col = score_column or first_existing(docking.columns, DOCKING_SCORE_COLUMNS)
    if not score_col:
        return pd.DataFrame(), {"status": "no_score_column", "docking_csv": str(path)}
    docking["_canonical_smiles_norm"] = docking[smiles_col].map(canonicalize_smiles)
    merged = docking.merge(
        dataset[["canonical_smiles_norm", "canonical_smiles", "p_activity"]],
        left_on="_canonical_smiles_norm",
        right_on="canonical_smiles_norm",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), {"status": "no_overlap_with_g12c_dataset", "docking_csv": str(path), "score_column": score_col}
    if "canonical_smiles" not in merged.columns:
        if "canonical_smiles_y" in merged.columns:
            merged["canonical_smiles"] = merged["canonical_smiles_y"]
        elif "canonical_smiles_x" in merged.columns:
            merged["canonical_smiles"] = merged["canonical_smiles_x"]
    merged["docking_rank_score"] = robust_scale(merged[score_col], lower_is_better=True)
    rows = ranking_metric_rows(
        merged,
        score_column="docking_rank_score",
        strategy="docking_only",
        split="provided_docking_csv",
        source="docking",
    )
    summary = {
        "status": "evaluated",
        "docking_csv": str(path),
        "score_column": score_col,
        "rows": int(len(docking)),
        "overlap_with_g12c_dataset": int(len(merged)),
    }
    return pd.DataFrame(rows), summary


def summarize_key_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    subset = metrics_df[
        (metrics_df["active_threshold"] == 7.0)
        & (metrics_df["top_spec"].isin(["top_20", "top_5pct", "top_10pct"]))
    ].copy()
    if subset.empty:
        return subset
    order = {
        "random_expected": 0,
        "qsar_best_rmse": 1,
        "qsar_best_rmse_plus_nn": 2,
        "qsar_best_rmse_plus_nn_domain": 3,
        "qsar_best_rmse_ranker_no_docking": 4,
        "nearest_neighbor_pIC50_train_only": 5,
        "property_only": 6,
    }
    subset["strategy_order"] = subset["strategy"].map(order).fillna(20)
    return subset.sort_values(["split", "top_spec", "strategy_order", "strategy"]).drop(columns=["strategy_order"])


def build_report(
    output_dir: Path,
    dataset_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    best_by_split: pd.DataFrame,
    model_topk: pd.DataFrame,
    metadata_baselines: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    docking_metrics: pd.DataFrame,
    docking_summary: dict[str, Any],
) -> None:
    docking_interpretation = (
        "- Docking-only metrics were computed for the provided docking CSV; treat small pilot runs as sanity checks, not final benchmarks."
        if docking_summary.get("status") == "evaluated"
        else "- Without a retrospective docking CSV, docking-only and QSAR+docking claims remain unmeasured."
    )
    report = [
        "# G12C Pipeline Evaluation",
        "",
        "## Dataset QA",
        "",
        f"- Raw ChEMBL activity rows: {dataset_summary.get('raw_activity_rows')}",
        f"- Cleaned unique G12C IC50 molecules: {dataset_summary.get('cleaned_unique_smiles')}",
        f"- Primary assays/documents: {dataset_summary.get('primary_assays')} assays / {dataset_summary.get('primary_documents')} documents",
        f"- Active rate pIC50 >= 7: {dataset_summary.get('active_rate_pIC50_ge_7', float('nan')):.3f}",
        f"- Largest document fraction: {dataset_summary.get('largest_document_fraction', float('nan')):.3f}",
        f"- Largest assay fraction: {dataset_summary.get('largest_assay_fraction', float('nan')):.3f}",
        f"- Multi-record molecule fraction: {dataset_summary.get('multi_record_fraction', float('nan')):.3f}",
        f"- Conflicting-record fraction, pIC50 std >= 0.5: {dataset_summary.get('conflicting_record_fraction_std_ge_0_5', float('nan')):.3f}",
        "",
        "## Current Candidate CSV QA",
        "",
        f"- Candidate CSV: {candidate_summary.get('candidate_csv')}",
        f"- Rows: {candidate_summary.get('rows', 0)}",
        f"- Valid SMILES rate: {candidate_summary.get('valid_smiles_rate', float('nan')):.3f}",
        f"- Formula match rate: {candidate_summary.get('formula_match_rate_among_rows', float('nan')):.3f}",
        f"- Activity nM present rate: {candidate_summary.get('activity_nM_present_rate', float('nan')):.3f}",
        f"- Duplicate canonical SMILES: {candidate_summary.get('duplicate_canonical_smiles', 0)}",
        "",
        "## Regression Holdout Summary",
        "",
        markdown_table(best_by_split, ["split", "model", "n_test", "rmse", "mae", "r2", "spearman"]),
        "",
        "## Top-K Ranking Summary",
        "",
        "Threshold is pIC50 >= 7 unless noted. Enrichment is relative to random selection in the same holdout.",
        "",
        markdown_table(
            ablation_summary[
                (ablation_summary["active_threshold"] == 7.0)
                & (ablation_summary["top_spec"] == "top_5pct")
                & (
                    ablation_summary["strategy"].isin(
                        [
                            "random_expected",
                            "qsar_best_rmse",
                            "qsar_best_rmse_plus_nn",
                            "qsar_best_rmse_plus_nn_domain",
                            "qsar_best_rmse_ranker_no_docking",
                            "nearest_neighbor_pIC50_train_only",
                            "property_only",
                        ]
                    )
                )
            ],
            ["split", "strategy", "n", "active_count", "k", "hits", "precision", "recall", "enrichment", "roc_auc", "average_precision"],
            max_rows=80,
        ),
        "",
        "## Metadata Baselines",
        "",
        markdown_table(
            metadata_baselines.sort_values(["split", "rmse"]),
            ["split", "strategy", "rmse", "mae", "r2", "spearman"],
            max_rows=30,
        ),
        "",
        "## Docking Evaluation",
        "",
        f"- Status: {docking_summary.get('status')}",
        f"- CSV: {docking_summary.get('docking_csv', '')}",
        f"- Overlap with G12C dataset: {docking_summary.get('overlap_with_g12c_dataset', '')}",
        "",
        markdown_table(
            docking_metrics[docking_metrics["active_threshold"] == 7.0] if not docking_metrics.empty else docking_metrics,
            ["strategy", "active_threshold", "top_spec", "n", "active_count", "k", "hits", "precision", "recall", "enrichment", "roc_auc", "average_precision"],
            max_rows=20,
        ),
        "",
        "## Interpretation",
        "",
        "- Random/scaffold performance is usable for coarse triage, but assay holdout remains the main risk.",
        "- The strongest retrospective question is top-k enrichment, not only RMSE.",
        docking_interpretation,
        "- Current candidate CSV has valid structures, but lacks extracted activity values, so it can be ranked but not validated as hits.",
        "",
    ]
    (output_dir / "PIPELINE_EVALUATION.md").write_text("\n".join(report), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else run_dir / "evaluation")

    dataset = load_dataset(run_dir)
    raw_df = pd.read_csv(run_dir / "raw_g12c_chembl_activities.csv")
    predictions = pd.read_csv(run_dir / "predictions_all_splits.csv")
    metrics = pd.read_csv(run_dir / "metrics.csv")

    dataset_summary = dataset_quality(dataset, raw_df)
    write_json(output_dir / "dataset_quality_summary.json", dataset_summary)

    candidate_rows, candidate_summary = evaluate_candidate_csv(Path(args.candidate_csv))
    candidate_rows.to_csv(output_dir / "candidate_csv_quality.csv", index=False)
    write_json(output_dir / "candidate_csv_quality_summary.json", candidate_summary)

    best_by_split = (
        metrics.sort_values(["split", "rmse", "mae"], ascending=[True, True, True])
        .groupby("split", as_index=False)
        .first()
        .sort_values("split")
    )
    best_by_split.to_csv(output_dir / "best_regression_by_split.csv", index=False)

    model_topk = evaluate_model_predictions(predictions)
    model_topk.to_csv(output_dir / "model_topk_metrics.csv", index=False)

    metadata_baselines = evaluate_metadata_baselines(dataset, predictions)
    metadata_baselines.to_csv(output_dir / "metadata_baseline_regression.csv", index=False)

    ablation_scores, ablation_metrics = evaluate_ablation(dataset, predictions, metrics)
    ablation_scores.to_csv(output_dir / "ablation_holdout_scores.csv", index=False)
    ablation_metrics.to_csv(output_dir / "ablation_topk_metrics.csv", index=False)
    ablation_summary = summarize_key_rows(ablation_metrics)
    ablation_summary.to_csv(output_dir / "ablation_summary_key_topk.csv", index=False)

    if args.docking_csv:
        docking_metrics, docking_summary = evaluate_docking_csv(
            Path(args.docking_csv),
            dataset,
            score_column=args.docking_score_column,
        )
    else:
        docking_metrics = pd.DataFrame()
        docking_summary = {"status": "missing", "docking_csv": ""}
    docking_metrics.to_csv(output_dir / "docking_topk_metrics.csv", index=False)
    write_json(output_dir / "docking_evaluation_summary.json", docking_summary)

    build_report(
        output_dir,
        dataset_summary,
        candidate_summary,
        best_by_split,
        model_topk,
        metadata_baselines,
        ablation_summary,
        docking_metrics,
        docking_summary,
    )
    print(f"Wrote evaluation outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
