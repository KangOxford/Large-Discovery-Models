#!/usr/bin/env python3
"""Create a KRAS G12C docking benchmark CSV from the cleaned ChEMBL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a G12C docking benchmark CSV from cleaned activity data.")
    parser.add_argument("--run-dir", default="activity_modeling/runs/g12c_expanded_20260612_122903")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--active-threshold", type=float, default=7.0, help="pIC50 threshold for active label.")
    parser.add_argument("--inactive-threshold", type=float, default=6.0, help="pIC50 threshold for inactive label.")
    parser.add_argument("--exclude-intermediate", action="store_true", help="Keep only active/inactive binary labels.")
    parser.add_argument("--assay-family", action="append", default=[], help="Optional assay family filter; can be repeated.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap after sorting. 0 keeps all rows.")
    parser.add_argument("--random-seed", type=int, default=714)
    return parser.parse_args(argv)


def load_dataset(run_dir: Path) -> pd.DataFrame:
    assay_family_path = run_dir / "assay_family" / "g12c_ic50_assay_family_dataset.csv"
    dataset_path = assay_family_path if assay_family_path.exists() else run_dir / "g12c_ic50_dataset.csv"
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def label_class(p_activity: float, active_threshold: float, inactive_threshold: float) -> str:
    if p_activity >= active_threshold:
        return "active"
    if p_activity <= inactive_threshold:
        return "inactive"
    return "intermediate"


def pactivity_to_nm(p_activity: float) -> float:
    return float(10 ** (9.0 - p_activity))


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    df = load_dataset(run_dir).copy()
    if args.assay_family:
        keep = set(args.assay_family)
        if "assay_family" not in df.columns:
            raise RuntimeError("Assay-family filter requested, but assay_family column is missing.")
        df = df[df["assay_family"].isin(keep)].copy()
    if df.empty:
        raise RuntimeError("No benchmark rows after filtering.")

    df["benchmark_label"] = df["p_activity"].map(
        lambda value: label_class(float(value), args.active_threshold, args.inactive_threshold)
    )
    if args.exclude_intermediate:
        df = df[df["benchmark_label"].isin(["active", "inactive"])].copy()
    if df.empty:
        raise RuntimeError("No benchmark rows after intermediate filtering.")

    # Stable order with actives first, then inactives, then intermediates. Within
    # each class, keep stronger pIC50 first so optional max-rows is deterministic.
    label_order = {"active": 0, "inactive": 1, "intermediate": 2}
    df["_label_order"] = df["benchmark_label"].map(label_order).fillna(9)
    df = df.sort_values(["_label_order", "p_activity", "canonical_smiles"], ascending=[True, False, True]).reset_index(drop=True)
    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        rng = np.random.default_rng(args.random_seed)
        sampled = []
        for label, group in df.groupby("benchmark_label", sort=False):
            frac = len(group) / len(df)
            take = max(1, int(round(args.max_rows * frac)))
            sampled.append(group.sample(n=min(take, len(group)), random_state=int(rng.integers(0, 2**31 - 1))))
        df = pd.concat(sampled, ignore_index=True).head(args.max_rows)
        df = df.sort_values(["_label_order", "p_activity", "canonical_smiles"], ascending=[True, False, True]).reset_index(drop=True)

    out = pd.DataFrame()
    out["compound_id"] = [f"G12C_{index:05d}" for index in range(1, len(df) + 1)]
    out["SMILES"] = df["canonical_smiles"].astype(str)
    out["canonical_smiles"] = df["canonical_smiles"].astype(str)
    out["Activity_nM"] = df["p_activity"].map(pactivity_to_nm)
    out["pIC50"] = df["p_activity"].astype(float)
    out["benchmark_label"] = df["benchmark_label"].astype(str)
    out["active_pIC50_ge_7"] = (df["p_activity"].astype(float) >= args.active_threshold).astype(int)
    out["inactive_pIC50_le_6"] = (df["p_activity"].astype(float) <= args.inactive_threshold).astype(int)
    out["primary_assay_chembl_id"] = df.get("primary_assay_chembl_id", "")
    out["primary_document_chembl_id"] = df.get("primary_document_chembl_id", "")
    out["assay_family"] = df.get("assay_family", "")
    out["assay_family_broad"] = df.get("assay_family_broad", "")
    out["n_records"] = df.get("n_records", "")
    out["p_activity_std"] = df.get("p_activity_std", "")
    out["molecule_chembl_ids"] = df.get("molecule_chembl_ids", "")
    out["activity_ids"] = df.get("activity_ids", "")
    out["Activity_Target"] = "KRAS G12C"
    out["Activity_Endpoint"] = "IC50"
    out["Activity_Value"] = out["Activity_nM"].map(lambda value: f"{float(value):.4g} nM")
    out["Activity_Evidence"] = "ChEMBL cleaned exact IC50; benchmark label derived from pIC50 thresholds"
    out["Note"] = (
        f"active if pIC50 >= {args.active_threshold}; inactive if pIC50 <= {args.inactive_threshold}; "
        "intermediate kept unless --exclude-intermediate"
    )

    output_path = Path(args.output_csv) if args.output_csv else run_dir / "g12c_docking_benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = {
        "output_csv": str(output_path),
        "run_dir": str(run_dir),
        "rows": int(len(out)),
        "active_threshold": args.active_threshold,
        "inactive_threshold": args.inactive_threshold,
        "exclude_intermediate": bool(args.exclude_intermediate),
        "assay_family_filter": args.assay_family,
        "label_counts": out["benchmark_label"].value_counts().to_dict(),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
