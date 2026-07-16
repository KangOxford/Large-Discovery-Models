#!/usr/bin/env python3
"""Score SMILES/analog CSVs with the trained KRAS G12C activity model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_g12c_qsar import HAS_RDKIT, Chem


SMILES_COLUMNS = (
    "canonical_smiles",
    "selected_smiles",
    "SMILES",
    "smiles",
    "full_smiles",
    "seed_smiles",
)


def find_smiles_column(df: pd.DataFrame) -> str:
    for column in SMILES_COLUMNS:
        if column in df.columns:
            return column
    raise RuntimeError(f"No SMILES column found. Tried: {', '.join(SMILES_COLUMNS)}")


def canonicalize_smiles(smiles: Any) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    if not HAS_RDKIT:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def load_training_dataset(run_dir: Path) -> pd.DataFrame:
    assay_family_path = run_dir / "assay_family" / "g12c_ic50_assay_family_dataset.csv"
    dataset_path = assay_family_path if assay_family_path.exists() else run_dir / "g12c_ic50_dataset.csv"
    if not dataset_path.exists():
        raise RuntimeError(f"Training dataset not found under {run_dir}")
    dataset = pd.read_csv(dataset_path)
    dataset["canonical_smiles_norm"] = dataset["canonical_smiles"].map(canonicalize_smiles)
    return dataset[dataset["canonical_smiles_norm"] != ""].reset_index(drop=True)


def nearest_neighbors(query_smiles: list[str], train_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not HAS_RDKIT:
        return pd.DataFrame(
            {
                "nearest_neighbor_smiles": [""] * len(query_smiles),
                "nearest_neighbor_similarity": [np.nan] * len(query_smiles),
                "nearest_neighbor_pIC50": [np.nan] * len(query_smiles),
                "nearest_neighbor_document": [""] * len(query_smiles),
                "nearest_neighbor_assay": [""] * len(query_smiles),
            }
        )
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = []
    train_indices = []
    for index, smiles in enumerate(train_df["canonical_smiles_norm"].astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        train_fps.append(generator.GetFingerprint(mol))
        train_indices.append(index)
    for smiles in query_smiles:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or not train_fps:
            rows.append(
                {
                    "nearest_neighbor_smiles": "",
                    "nearest_neighbor_similarity": np.nan,
                    "nearest_neighbor_pIC50": np.nan,
                    "nearest_neighbor_document": "",
                    "nearest_neighbor_assay": "",
                }
            )
            continue
        fp = generator.GetFingerprint(mol)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=float)
        pos = int(np.argmax(sims))
        train_index = int(train_indices[pos])
        nn = train_df.iloc[train_index]
        rows.append(
            {
                "nearest_neighbor_smiles": nn["canonical_smiles_norm"],
                "nearest_neighbor_similarity": float(sims[pos]),
                "nearest_neighbor_pIC50": float(nn["p_activity"]),
                "nearest_neighbor_document": str(nn.get("primary_document_chembl_id", "")),
                "nearest_neighbor_assay": str(nn.get("primary_assay_chembl_id", "")),
                "nearest_neighbor_family": str(nn.get("assay_family", "")),
            }
        )
    return pd.DataFrame(rows)


def applicability_flag(similarity: Any) -> str:
    try:
        sim = float(similarity)
    except (TypeError, ValueError):
        return "invalid_smiles_or_no_neighbor"
    if not np.isfinite(sim):
        return "invalid_smiles_or_no_neighbor"
    if sim >= 0.65:
        return "in_domain_close_analog"
    if sim >= 0.45:
        return "moderate_domain"
    if sim >= 0.30:
        return "weak_domain"
    return "out_of_domain"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict KRAS G12C pIC50 for analog/docking CSV SMILES.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--run-dir", default="activity_modeling/runs/g12c_expanded_20260612_122903")
    parser.add_argument("--model-path", default="", help="Defaults to <run-dir>/best_model.joblib")
    parser.add_argument("--compound-id-column", default="", help="Optional ID column to keep first in output.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    model_path = Path(args.model_path) if args.model_path else run_dir / "best_model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    df = pd.read_csv(args.input_csv)
    smiles_column = find_smiles_column(df)
    df["g12c_model_input_smiles"] = df[smiles_column].map(canonicalize_smiles)
    valid_mask = df["g12c_model_input_smiles"] != ""

    model = joblib.load(model_path)
    predictions = np.full(len(df), np.nan, dtype=float)
    if valid_mask.any():
        predictions[valid_mask.to_numpy()] = model.predict(df.loc[valid_mask, "g12c_model_input_smiles"].astype(str))
    df["g12c_predicted_pIC50"] = predictions

    train_df = load_training_dataset(run_dir)
    nn = nearest_neighbors(df["g12c_model_input_smiles"].astype(str).tolist(), train_df)
    df = pd.concat([df.reset_index(drop=True), nn.reset_index(drop=True)], axis=1)
    df["g12c_applicability_domain"] = df["nearest_neighbor_similarity"].map(applicability_flag)
    df["g12c_prediction_note"] = np.where(
        df["g12c_applicability_domain"].isin(["weak_domain", "out_of_domain", "invalid_smiles_or_no_neighbor"]),
        "Down-weight QSAR score; analog is outside or near edge of G12C training domain.",
        "",
    )

    preferred = []
    id_col = args.compound_id_column or next((col for col in ("compound_id", "Compound", "compound", "Cmpd") if col in df.columns), "")
    if id_col:
        preferred.append(id_col)
    preferred.extend(
        [
            smiles_column,
            "g12c_model_input_smiles",
            "g12c_predicted_pIC50",
            "nearest_neighbor_similarity",
            "nearest_neighbor_pIC50",
            "g12c_applicability_domain",
            "nearest_neighbor_document",
            "nearest_neighbor_assay",
            "nearest_neighbor_family",
            "g12c_prediction_note",
        ]
    )
    ordered = [col for col in preferred if col in df.columns]
    ordered.extend([col for col in df.columns if col not in ordered])
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(output_path, index=False)
    metadata = {
        "input_csv": args.input_csv,
        "output_csv": str(output_path),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "smiles_column": smiles_column,
        "valid_smiles": int(valid_mask.sum()),
        "rows": int(len(df)),
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
