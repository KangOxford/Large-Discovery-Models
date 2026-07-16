#!/usr/bin/env python3
"""Rank KRAS G12C candidates with QSAR, docking, property, and SAR-domain terms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from predict_g12c_activity import (  # noqa: E402
    applicability_flag,
    canonicalize_smiles,
    find_smiles_column,
    load_training_dataset,
    nearest_neighbors,
)
from train_g12c_qsar import HAS_RDKIT, Chem  # noqa: E402


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
DOCKING_STATUS_COLUMNS = ("docking_status", "status", "vina_status")
DOMAIN_COMPONENTS = {
    "in_domain_close_analog": 1.0,
    "moderate_domain": 0.70,
    "weak_domain": 0.35,
    "out_of_domain": 0.10,
    "invalid_smiles_or_no_neighbor": 0.0,
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a multi-objective KRAS G12C candidate ranking table."
    )
    parser.add_argument("--input-csv", required=True, help="Candidate, QSAR, docking, or joint CSV.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--run-dir", default="activity_modeling/runs/g12c_expanded_20260612_122903")
    parser.add_argument("--model-path", default="", help="Defaults to <run-dir>/best_model.joblib")
    parser.add_argument("--docking-csv", default="", help="Optional separate docking CSV to merge.")
    parser.add_argument("--merge-on", default="", help="Column used to merge --docking-csv. Defaults to shared ID or canonical SMILES.")
    parser.add_argument("--docking-score-column", default="", help="Override docking score column.")
    parser.add_argument("--higher-docking-better", action="store_true", help="Use when docking score is already a positive desirability score.")
    parser.add_argument("--prefer-covalent-warhead", action="store_true", help="Add a small reward for acrylamide/chloroacetamide/vinyl-sulfone-like warheads.")
    parser.add_argument("--activity-weight", type=float, default=0.35)
    parser.add_argument("--docking-weight", type=float, default=0.25)
    parser.add_argument("--applicability-weight", type=float, default=0.15)
    parser.add_argument("--property-weight", type=float, default=0.15)
    parser.add_argument("--neighbor-weight", type=float, default=0.10)
    parser.add_argument("--warhead-weight", type=float, default=0.0)
    return parser.parse_args(argv)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing(columns: list[str] | pd.Index, candidates: tuple[str, ...]) -> str:
    present = set(columns)
    return next((column for column in candidates if column in present), "")


def fixed_scale(series: pd.Series, low: float, high: float) -> pd.Series:
    values = to_numeric(series)
    scaled = (values - low) / (high - low)
    return scaled.clip(0.0, 1.0)


def robust_scale(series: pd.Series, *, lower_is_better: bool) -> pd.Series:
    values = to_numeric(series)
    finite = values[np.isfinite(values)]
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(finite) == 0:
        return out
    if len(finite) == 1:
        out.loc[finite.index] = 0.5
        return out
    low, high = np.nanpercentile(finite.to_numpy(dtype=float), [5, 95])
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
        out.loc[finite.index] = 0.5
        return out
    if lower_is_better:
        scaled = (high - values) / (high - low)
    else:
        scaled = (values - low) / (high - low)
    return scaled.clip(0.0, 1.0)


def range_preference(value: Any, good_low: float, good_high: float, hard_low: float, hard_high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    if good_low <= number <= good_high:
        return 1.0
    if number < good_low:
        if number <= hard_low:
            return 0.0
        return (number - hard_low) / (good_low - hard_low)
    if number >= hard_high:
        return 0.0
    return (hard_high - number) / (hard_high - good_high)


def upper_preference(value: Any, good_high: float, hard_high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    if number <= good_high:
        return 1.0
    if number >= hard_high:
        return 0.0
    return (hard_high - number) / (hard_high - good_high)


def absolute_charge_preference(value: Any) -> float:
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    if number <= 1.0:
        return 1.0
    if number >= 3.0:
        return 0.0
    return (3.0 - number) / 2.0


def rdkit_candidate_properties(smiles_values: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not HAS_RDKIT:
        return pd.DataFrame(index=smiles_values.index)

    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    smarts = {
        "warhead_acrylamide_like": "[#6]=[#6][CX3](=[OX1])[NX3]",
        "warhead_chloroacetamide_like": "[Cl][CH2][CX3](=[OX1])[NX3]",
        "warhead_vinyl_sulfonamide_like": "[#6]=[#6]S(=O)(=O)[#6,#7]",
        "warhead_cyanoacrylamide_like": "N#C[#6]=[#6][CX3](=[OX1])[NX3]",
    }
    queries = {name: Chem.MolFromSmarts(pattern) for name, pattern in smarts.items()}

    for smiles in smiles_values.astype(str):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            rows.append(
                {
                    "valid_rdkit_mol": False,
                    "mol_wt": np.nan,
                    "clogp": np.nan,
                    "tpsa": np.nan,
                    "hbd": np.nan,
                    "hba": np.nan,
                    "rotatable_bonds": np.nan,
                    "heavy_atoms": np.nan,
                    "formal_charge": np.nan,
                    "fraction_csp3": np.nan,
                    "ring_count": np.nan,
                    "aromatic_ring_count": np.nan,
                    "warhead_acrylamide_like": 0,
                    "warhead_chloroacetamide_like": 0,
                    "warhead_vinyl_sulfonamide_like": 0,
                    "warhead_cyanoacrylamide_like": 0,
                    "has_covalent_warhead_like": 0,
                }
            )
            continue
        row = {
            "valid_rdkit_mol": True,
            "mol_wt": float(Descriptors.MolWt(mol)),
            "clogp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "heavy_atoms": float(mol.GetNumHeavyAtoms()),
            "formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        }
        for name, query in queries.items():
            row[name] = int(query is not None and mol.HasSubstructMatch(query))
        row["has_covalent_warhead_like"] = int(
            row["warhead_acrylamide_like"]
            or row["warhead_chloroacetamide_like"]
            or row["warhead_vinyl_sulfonamide_like"]
            or row["warhead_cyanoacrylamide_like"]
        )
        rows.append(row)
    return pd.DataFrame(rows, index=smiles_values.index)


def property_component(df: pd.DataFrame) -> pd.Series:
    if "mol_wt" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    parts = pd.DataFrame(
        {
            "mw": df["mol_wt"].map(lambda value: range_preference(value, 300.0, 600.0, 150.0, 750.0)),
            "clogp": df["clogp"].map(lambda value: range_preference(value, 1.0, 4.5, -1.0, 6.5)),
            "tpsa": df["tpsa"].map(lambda value: range_preference(value, 35.0, 130.0, 0.0, 190.0)),
            "hbd": df["hbd"].map(lambda value: upper_preference(value, 3.0, 6.0)),
            "hba": df["hba"].map(lambda value: upper_preference(value, 10.0, 14.0)),
            "rotb": df["rotatable_bonds"].map(lambda value: upper_preference(value, 8.0, 15.0)),
            "charge": df["formal_charge"].map(absolute_charge_preference),
        },
        index=df.index,
    )
    return parts.mean(axis=1, skipna=True)


def property_alerts(row: pd.Series) -> list[str]:
    alerts: list[str] = []
    checks = (
        ("mol_wt_high", row.get("mol_wt"), lambda value: value > 750),
        ("clogp_high", row.get("clogp"), lambda value: value > 6.5),
        ("tpsa_high", row.get("tpsa"), lambda value: value > 190),
        ("hbd_high", row.get("hbd"), lambda value: value > 6),
        ("hba_high", row.get("hba"), lambda value: value > 14),
        ("rotatable_bonds_high", row.get("rotatable_bonds"), lambda value: value > 15),
        ("large_formal_charge", row.get("formal_charge"), lambda value: abs(value) > 2),
    )
    for name, value, predicate in checks:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and predicate(number):
            alerts.append(name)
    return alerts


def ensure_activity_predictions(df: pd.DataFrame, run_dir: Path, model_path: Path) -> tuple[pd.DataFrame, str, bool]:
    smiles_column = "g12c_model_input_smiles" if "g12c_model_input_smiles" in df.columns else find_smiles_column(df)
    if "g12c_model_input_smiles" not in df.columns:
        df["g12c_model_input_smiles"] = df[smiles_column].map(canonicalize_smiles)
    else:
        df["g12c_model_input_smiles"] = df["g12c_model_input_smiles"].map(canonicalize_smiles)
    valid_mask = df["g12c_model_input_smiles"] != ""

    needed_prediction = "g12c_predicted_pIC50" not in df.columns or df["g12c_predicted_pIC50"].isna().any()
    needed_neighbors = "nearest_neighbor_similarity" not in df.columns or "nearest_neighbor_pIC50" not in df.columns
    predicted_now = False

    if needed_prediction:
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_path}")
        model = joblib.load(model_path)
        predictions = (
            to_numeric(df["g12c_predicted_pIC50"]).to_numpy(dtype=float)
            if "g12c_predicted_pIC50" in df.columns
            else np.full(len(df), np.nan, dtype=float)
        )
        if valid_mask.any():
            missing_mask = valid_mask.to_numpy() & ~np.isfinite(predictions)
            if missing_mask.any():
                predictions[missing_mask] = model.predict(
                    df.loc[missing_mask, "g12c_model_input_smiles"].astype(str)
                )
        df["g12c_predicted_pIC50"] = predictions
        predicted_now = True

    if needed_neighbors:
        train_df = load_training_dataset(run_dir)
        nn = nearest_neighbors(df["g12c_model_input_smiles"].astype(str).tolist(), train_df)
        for column in nn.columns:
            if column not in df.columns:
                df[column] = nn[column].to_numpy()
    if "g12c_applicability_domain" not in df.columns:
        df["g12c_applicability_domain"] = df["nearest_neighbor_similarity"].map(applicability_flag)
    return df, smiles_column, predicted_now


def merge_docking(df: pd.DataFrame, docking_csv: str, merge_on: str) -> tuple[pd.DataFrame, str]:
    if not docking_csv:
        return df, ""
    docking = pd.read_csv(docking_csv)
    key = merge_on
    if key:
        if key not in df.columns or key not in docking.columns:
            raise RuntimeError(f"Merge key {key!r} must exist in both input and docking CSVs.")
    else:
        key = first_existing(df.columns, tuple(column for column in ID_COLUMNS if column in docking.columns))
    if not key:
        left_key = "g12c_model_input_smiles"
        if left_key not in df.columns:
            raise RuntimeError("Canonical SMILES must be prepared before docking merge.")
        docking_smiles = find_smiles_column(docking)
        docking["_merge_canonical_smiles"] = docking[docking_smiles].map(canonicalize_smiles)
        df["_merge_canonical_smiles"] = df[left_key].astype(str)
        key = "_merge_canonical_smiles"
    merged = df.merge(docking, how="left", on=key, suffixes=("", "_docking"))
    return merged, key


def choose_docking_score_column(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise RuntimeError(f"Requested docking score column not found: {requested}")
        return requested
    return first_existing(df.columns, DOCKING_SCORE_COLUMNS)


def docking_component(df: pd.DataFrame, score_column: str, higher_is_better: bool) -> pd.Series:
    if not score_column:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return robust_scale(df[score_column], lower_is_better=not higher_is_better)


def build_risk_flags(df: pd.DataFrame, docking_score_column: str, prefer_covalent: bool) -> tuple[pd.Series, pd.Series]:
    flags_out: list[str] = []
    penalty_out: list[float] = []
    status_column = first_existing(df.columns, DOCKING_STATUS_COLUMNS)
    for _index, row in df.iterrows():
        flags: list[str] = []
        penalty = 0.0
        domain = str(row.get("g12c_applicability_domain", "") or "")
        if domain in ("weak_domain", "out_of_domain", "invalid_smiles_or_no_neighbor"):
            flags.append(f"domain={domain}")
            penalty += {"weak_domain": 0.12, "out_of_domain": 0.22, "invalid_smiles_or_no_neighbor": 0.25}.get(domain, 0.0)
        elif domain == "moderate_domain":
            penalty += 0.04

        pred = row.get("g12c_predicted_pIC50")
        nn = row.get("nearest_neighbor_pIC50")
        sim = row.get("nearest_neighbor_similarity")
        try:
            pred_f, nn_f, sim_f = float(pred), float(nn), float(sim)
        except (TypeError, ValueError):
            pred_f = nn_f = sim_f = np.nan
        if np.isfinite(pred_f) and np.isfinite(nn_f) and np.isfinite(sim_f) and sim_f >= 0.50:
            delta = abs(pred_f - nn_f)
            if delta >= 1.0:
                flags.append("model_nn_disagreement")
                penalty += min(0.18, 0.05 + 0.05 * (delta - 1.0))

        prop_flags = property_alerts(row)
        if prop_flags:
            flags.extend(prop_flags)
            penalty += min(0.15, 0.03 * len(prop_flags))

        if prefer_covalent and int(row.get("has_covalent_warhead_like", 0) or 0) == 0:
            flags.append("no_covalent_warhead_like")
            penalty += 0.05

        if docking_score_column:
            score = row.get(docking_score_column)
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = np.nan
            if not np.isfinite(score_f):
                flags.append("no_docking_score")
                penalty += 0.10
        if status_column:
            status = str(row.get(status_column, "") or "").lower()
            if status and status not in ("ok", "success", "done", "cached"):
                flags.append(f"docking_status={status}")
                penalty += 0.08

        flags_out.append(";".join(flags))
        penalty_out.append(float(min(0.45, penalty)))
    return pd.Series(flags_out, index=df.index), pd.Series(penalty_out, index=df.index, dtype=float)


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    numerator = pd.Series(0.0, index=df.index, dtype=float)
    denominator = pd.Series(0.0, index=df.index, dtype=float)
    for name, weight in weights.items():
        column = f"{name}_component"
        if weight <= 0 or column not in df.columns:
            continue
        values = to_numeric(df[column])
        valid = np.isfinite(values)
        numerator.loc[valid] += values.loc[valid] * weight
        denominator.loc[valid] += weight
    score = numerator / denominator.replace(0.0, np.nan)
    return score


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    model_path = Path(args.model_path) if args.model_path else run_dir / "best_model.joblib"

    df = pd.read_csv(args.input_csv)
    df, smiles_column, predicted_now = ensure_activity_predictions(df, run_dir, model_path)
    df, merge_key = merge_docking(df, args.docking_csv, args.merge_on)

    properties = rdkit_candidate_properties(df["g12c_model_input_smiles"].astype(str))
    for column in properties.columns:
        if column not in df.columns:
            df[column] = properties[column].to_numpy()

    docking_score_column = choose_docking_score_column(df, args.docking_score_column)
    df["activity_component"] = fixed_scale(df["g12c_predicted_pIC50"], 5.5, 8.5)
    df["nearest_neighbor_component"] = fixed_scale(df["nearest_neighbor_pIC50"], 5.5, 8.5) * fixed_scale(
        df["nearest_neighbor_similarity"], 0.30, 0.75
    )
    df["applicability_component"] = df["g12c_applicability_domain"].map(DOMAIN_COMPONENTS).fillna(0.0)
    df["docking_component"] = docking_component(df, docking_score_column, args.higher_docking_better)
    df["property_component"] = property_component(df)
    df["warhead_component"] = to_numeric(df.get("has_covalent_warhead_like", pd.Series(0, index=df.index))).fillna(0.0)

    warhead_weight = args.warhead_weight
    if args.prefer_covalent_warhead and warhead_weight == 0.0:
        warhead_weight = 0.05
    weights = {
        "activity": args.activity_weight,
        "docking": args.docking_weight,
        "applicability": args.applicability_weight,
        "property": args.property_weight,
        "nearest_neighbor": args.neighbor_weight,
        "warhead": warhead_weight,
    }
    df["raw_multi_objective_score"] = weighted_score(df, weights)
    df["risk_flags"], df["risk_penalty"] = build_risk_flags(df, docking_score_column, args.prefer_covalent_warhead)
    df["multi_objective_score"] = (df["raw_multi_objective_score"] - df["risk_penalty"]).clip(0.0, 1.0)

    if docking_score_column and "heavy_atoms" in df.columns:
        docking_values = to_numeric(df[docking_score_column])
        heavy_atoms = to_numeric(df["heavy_atoms"]).replace(0.0, np.nan)
        df["docking_ligand_efficiency"] = -docking_values / heavy_atoms

    id_col = first_existing(df.columns, ID_COLUMNS)
    sort_columns = ["multi_objective_score", "activity_component", "applicability_component"]
    df = df.sort_values(sort_columns, ascending=[False, False, False]).reset_index(drop=True)
    df["multi_objective_rank"] = np.arange(1, len(df) + 1)

    preferred = [
        "multi_objective_rank",
        id_col,
        smiles_column,
        "g12c_model_input_smiles",
        "multi_objective_score",
        "raw_multi_objective_score",
        "risk_penalty",
        "risk_flags",
        "g12c_predicted_pIC50",
        "nearest_neighbor_similarity",
        "nearest_neighbor_pIC50",
        "g12c_applicability_domain",
        docking_score_column,
        "docking_ligand_efficiency",
        "activity_component",
        "docking_component",
        "nearest_neighbor_component",
        "applicability_component",
        "property_component",
        "warhead_component",
        "mol_wt",
        "clogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "has_covalent_warhead_like",
    ]
    ordered = []
    for column in preferred:
        if column and column in df.columns and column not in ordered:
            ordered.append(column)
    ordered.extend([column for column in df.columns if column not in ordered])

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(output_path, index=False)
    metadata = {
        "input_csv": args.input_csv,
        "output_csv": str(output_path),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "smiles_column": smiles_column,
        "predicted_activity_in_ranker": predicted_now,
        "docking_csv": args.docking_csv,
        "docking_merge_key": merge_key,
        "docking_score_column": docking_score_column,
        "higher_docking_better": bool(args.higher_docking_better),
        "prefer_covalent_warhead": bool(args.prefer_covalent_warhead),
        "weights": weights,
        "rows": int(len(df)),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
