#!/usr/bin/env python3
"""
String-kernel Bayesian optimization loop for ReaSyn inference parameters.

Each trial samples ReaSyn inference settings, generates analogs from the current
active seed pool, docks those analogs with AutoDock Vina, and returns the best
Vina score from that trial. Lower Vina kcal/mol is better, so the optimizer
uses direction="minimize".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import docking_to_analog_search_example as bridge
from strbo import Dimension, SearchSpace, StrBOConfig, create_study


DEFAULT_OUTPUT_DIR = "output/reasyn_strbo"
DEFAULT_TOP_K = 3
DEFAULT_FAILURE_SCORE = 1_000_000.0


@dataclass
class Candidate:
    root_id: str
    root_smiles: str
    compound_id: str
    smiles: str
    score: Optional[float] = None
    source_trial: int = -1
    parent_smiles: str = ""
    reasyn_score: Optional[float] = None
    synthesis: str = ""
    num_steps: str = ""
    activity_nM: str = ""
    activity_assay: str = ""
    activity_target: str = ""


def safe_float(value: Any) -> Optional[float]:
    return bridge.safe_float(value)


def candidate_sort_key(candidate: Candidate) -> tuple[float, int, str]:
    return (
        candidate.score if candidate.score is not None else float("inf"),
        candidate.source_trial,
        candidate.compound_id,
    )


def load_initial_candidates(path: Path, top_n: int) -> list[Candidate]:
    seeds = bridge.read_seed_smiles(path, top_n)
    candidates: list[Candidate] = []
    for seed in seeds:
        root_id = str(seed.get("compound_id") or f"seed_{len(candidates) + 1}")
        smiles = str(seed.get("canonical_smiles") or "")
        candidates.append(
            Candidate(
                root_id=root_id,
                root_smiles=smiles,
                compound_id=root_id,
                smiles=smiles,
                activity_nM=str(seed.get("activity_nM", "")),
                activity_assay=str(seed.get("primary_activity_assay", seed.get("Activity_Assay", ""))),
                activity_target=str(seed.get("primary_activity_target", seed.get("Activity_Target", ""))),
            )
        )
    return candidates


def group_candidates(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.root_id, []).append(candidate)
    return grouped


def truncate_candidate_archive(candidates: list[Candidate], max_candidates: int) -> list[Candidate]:
    best_by_smiles: dict[str, Candidate] = {}
    for candidate in candidates:
        key = bridge.canonicalize_smiles(candidate.smiles)
        existing = best_by_smiles.get(key)
        if existing is None or candidate_sort_key(candidate) < candidate_sort_key(existing):
            best_by_smiles[key] = candidate
    ordered = sorted(best_by_smiles.values(), key=candidate_sort_key)
    return ordered[:max_candidates]


def write_reasyn_input(path: Path, active_candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("SMILES\n")
        for candidate in active_candidates:
            handle.write(candidate.smiles + "\n")


def write_seed_map(path: Path, active_candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "root_id",
        "root_smiles",
        "compound_id",
        "smiles",
        "score",
        "source_trial",
        "parent_smiles",
        "activity_nM",
        "activity_assay",
        "activity_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in active_candidates:
            writer.writerow({field: getattr(candidate, field) for field in fields})


def write_dict_csv(path: Path, rows: list[dict[str, Any]], fallback_fieldnames: list[str]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or fallback_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def canonicalize_smiles_no_stereo(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        from rdkit import Chem
    except ImportError:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def smiles_match_keys(smiles: str) -> list[str]:
    keys: list[str] = []
    for key in (bridge.canonicalize_smiles(smiles), canonicalize_smiles_no_stereo(smiles)):
        if key and key not in keys:
            keys.append(key)
    return keys


def parent_seed_map(active_candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    mapping: dict[str, list[Candidate]] = {}
    for candidate in active_candidates:
        for key in smiles_match_keys(candidate.smiles):
            bucket = mapping.setdefault(key, [])
            if all(item.compound_id != candidate.compound_id for item in bucket):
                bucket.append(candidate)
    return mapping


def lookup_parent_candidates(mapping: dict[str, list[Candidate]], smiles: str) -> list[Candidate]:
    parents: list[Candidate] = []
    for key in smiles_match_keys(smiles):
        for candidate in mapping.get(key, []):
            if all(item.compound_id != candidate.compound_id for item in parents):
                parents.append(candidate)
    return parents


def annotate_reasyn_analog_csv(
    input_csv: Path,
    output_csv: Path,
    active_candidates: list[Candidate],
    *,
    trial_number: int,
) -> list[dict[str, Any]]:
    parent_map = parent_seed_map(active_candidates)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    annotated: list[dict[str, Any]] = []
    per_root_counts: dict[str, int] = {}
    for row in rows:
        analog_smiles = str(row.get("smiles") or row.get("SMILES") or row.get("product_smiles") or "").strip()
        parent_smiles = str(row.get("target") or row.get("seed_smiles") or row.get("parent_smiles") or "").strip()
        if not analog_smiles or not parent_smiles:
            continue
        parents = lookup_parent_candidates(parent_map, parent_smiles)
        if not parents:
            continue
        for parent in parents:
            per_root_counts[parent.root_id] = per_root_counts.get(parent.root_id, 0) + 1
            out = dict(row)
            out["compound_id"] = f"{parent.root_id}_trial{trial_number}_analog_{per_root_counts[parent.root_id]}"
            out["smiles"] = analog_smiles
            out["target"] = parent_smiles
            out["parent_smiles"] = parent_smiles
            out["analog_group_id"] = parent.root_id
            out["root_seed_smiles"] = parent.root_smiles
            out["Activity_nM"] = parent.activity_nM
            out["Activity_Assay"] = parent.activity_assay
            out["Activity_Target"] = parent.activity_target
            out["reasyn_score"] = out.get("reasyn_score") or out.get("score", "")
            annotated.append(out)

    write_dict_csv(output_csv, annotated, ["target", "smiles"])
    return annotated


def analog_row_rank(row: dict[str, Any]) -> tuple[float, float, str]:
    score = safe_float(row.get("reasyn_score") or row.get("score"))
    num_steps = safe_float(row.get("num_steps"))
    return (
        -score if score is not None else float("inf"),
        num_steps if num_steps is not None else float("inf"),
        str(row.get("smiles") or ""),
    )


def limit_annotated_analogs(rows: list[dict[str, Any]], max_per_seed: int) -> list[dict[str, Any]]:
    if max_per_seed <= 0:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        root_id = str(row.get("analog_group_id") or row.get("target") or "")
        grouped.setdefault(root_id, []).append(row)

    limited: list[dict[str, Any]] = []
    for root_id in sorted(grouped):
        best_by_smiles: dict[str, dict[str, Any]] = {}
        for row in grouped[root_id]:
            smiles = str(row.get("smiles") or "")
            key = bridge.canonicalize_smiles(smiles) or smiles
            existing = best_by_smiles.get(key)
            if existing is None or analog_row_rank(row) < analog_row_rank(existing):
                best_by_smiles[key] = row
        limited.extend(sorted(best_by_smiles.values(), key=analog_row_rank)[:max_per_seed])
    return limited


def read_topk_candidates(topk_csv: Path, root_smiles_by_id: dict[str, str], *, trial_number: int) -> list[Candidate]:
    if not topk_csv.exists():
        return []
    with topk_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[Candidate] = []
    for row in rows:
        root_id = str(row.get("group_id") or "").strip()
        smiles = str(row.get("analog_smiles") or row.get("smiles") or "").strip()
        if not root_id or not smiles:
            continue
        candidates.append(
            Candidate(
                root_id=root_id,
                root_smiles=root_smiles_by_id.get(root_id, ""),
                compound_id=str(row.get("compound_id") or f"{root_id}_trial{trial_number}_topk_{len(candidates) + 1}"),
                smiles=smiles,
                score=safe_float(row.get("vina_score_kcal_mol")),
                source_trial=trial_number,
                parent_smiles=str(row.get("seed_smiles") or ""),
                reasyn_score=safe_float(row.get("reasyn_score")),
                synthesis=str(row.get("synthesis") or ""),
                num_steps=str(row.get("num_steps") or ""),
                activity_nM=str(row.get("seed_activity_nM") or ""),
                activity_assay=str(row.get("seed_activity_assay") or ""),
                activity_target=str(row.get("seed_activity_target") or ""),
            )
        )
    return candidates


def update_candidate_state(
    archive: dict[str, list[Candidate]],
    previous_active: list[Candidate],
    new_topk: list[Candidate],
    *,
    top_k: int,
    max_candidates_per_seed: int,
) -> tuple[dict[str, list[Candidate]], list[Candidate]]:
    previous_by_root = group_candidates(previous_active)
    new_by_root = group_candidates(new_topk)
    updated_archive: dict[str, list[Candidate]] = {}
    next_active: list[Candidate] = []
    for root_id in sorted(set(archive) | set(previous_by_root) | set(new_by_root)):
        merged = list(archive.get(root_id, [])) + list(new_by_root.get(root_id, []))
        updated_archive[root_id] = truncate_candidate_archive(merged, max_candidates_per_seed)
        active_for_root = sorted(new_by_root.get(root_id, []), key=candidate_sort_key)[:top_k]
        if not active_for_root:
            active_for_root = sorted(previous_by_root.get(root_id, []), key=candidate_sort_key)[:top_k]
        next_active.extend(active_for_root)
    return updated_archive, next_active


def sample_reasyn_params(trial: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "search_width": trial.suggest_int("search_width", args.search_width_range[0], args.search_width_range[1]),
        "exhaustiveness": trial.suggest_int("exhaustiveness", args.exhaustiveness_range[0], args.exhaustiveness_range[1]),
        "num_cycles": trial.suggest_int("num_cycles", args.num_cycles_range[0], args.num_cycles_range[1]),
        "num_editflow_samples": trial.suggest_int(
            "num_editflow_samples",
            args.num_editflow_samples_range[0],
            args.num_editflow_samples_range[1],
        ),
        "num_editflow_steps": trial.suggest_int(
            "num_editflow_steps",
            args.num_editflow_steps_range[0],
            args.num_editflow_steps_range[1],
        ),
        "filter_sim": trial.suggest_float("filter_sim", args.filter_sim_range[0], args.filter_sim_range[1]),
        "no_exact_break": trial.suggest_categorical("no_exact_break", [True, False]),
    }


def best_vina_score_from_summary(summary: dict[str, Any], failure_score: float) -> float:
    group_best = summary.get("group_best") or []
    scores = [score for score in (safe_float(row.get("vina_score_kcal_mol")) for row in group_best) if score is not None]
    if not scores:
        return failure_score
    return min(scores)


def write_best_analog_per_seed_csv(
    path: Path,
    initial_candidates: list[Candidate],
    archive: dict[str, list[Candidate]],
    seed_results_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = [
        "root_id",
        "root_smiles",
        "seed_vina_score_kcal_mol",
        "seed_docking_status",
        "seed_pose_ref",
        "seed_activity_nM",
        "seed_activity_assay",
        "seed_activity_target",
        "best_analog_compound_id",
        "best_analog_smiles",
        "best_analog_vina_score_kcal_mol",
        "delta_vina_analog_minus_seed",
        "improvement_kcal_mol",
        "analog_better_than_seed",
        "best_analog_source_trial",
        "best_analog_parent_smiles",
        "best_analog_reasyn_score",
        "best_analog_synthesis",
        "best_analog_num_steps",
    ]
    rows: list[dict[str, Any]] = []
    for seed in initial_candidates:
        seed_result = seed_results_by_id.get(seed.root_id)
        seed_score = safe_float(getattr(seed_result, "score", None))
        seed_status = getattr(seed_result, "status", "") if seed_result else ""
        seed_pose_ref = getattr(seed_result, "pose_ref", "") if seed_result else ""
        analog_candidates = [
            candidate
            for candidate in archive.get(seed.root_id, [])
            if candidate.source_trial >= 0 and candidate.score is not None
        ]
        best_analog = sorted(analog_candidates, key=candidate_sort_key)[0] if analog_candidates else None
        analog_score = best_analog.score if best_analog else None
        delta = analog_score - seed_score if analog_score is not None and seed_score is not None else None
        improvement = -delta if delta is not None else None
        rows.append(
            {
                "root_id": seed.root_id,
                "root_smiles": seed.root_smiles,
                "seed_vina_score_kcal_mol": seed_score if seed_score is not None else "",
                "seed_docking_status": seed_status,
                "seed_pose_ref": seed_pose_ref or "",
                "seed_activity_nM": seed.activity_nM,
                "seed_activity_assay": seed.activity_assay,
                "seed_activity_target": seed.activity_target,
                "best_analog_compound_id": best_analog.compound_id if best_analog else "",
                "best_analog_smiles": best_analog.smiles if best_analog else "",
                "best_analog_vina_score_kcal_mol": analog_score if analog_score is not None else "",
                "delta_vina_analog_minus_seed": delta if delta is not None else "",
                "improvement_kcal_mol": improvement if improvement is not None else "",
                "analog_better_than_seed": (analog_score < seed_score)
                if analog_score is not None and seed_score is not None
                else "",
                "best_analog_source_trial": best_analog.source_trial if best_analog else "",
                "best_analog_parent_smiles": best_analog.parent_smiles if best_analog else "",
                "best_analog_reasyn_score": best_analog.reasyn_score if best_analog and best_analog.reasyn_score is not None else "",
                "best_analog_synthesis": best_analog.synthesis if best_analog else "",
                "best_analog_num_steps": best_analog.num_steps if best_analog else "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def validate_range(name: str, values: list[Any]) -> None:
    if len(values) != 2 or values[0] > values[1]:
        raise RuntimeError(f"{name} must be two values in ascending order.")


class ReaSynBOObjective:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        args.output_dir = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        initial = load_initial_candidates(Path(args.smiles_path), args.top_n_seeds)
        if not initial:
            raise RuntimeError(f"No initial seed SMILES found in {args.smiles_path}")
        self.initial_candidates = list(initial)
        self.root_smiles_by_id = {candidate.root_id: candidate.root_smiles for candidate in initial}
        self.archive = {candidate.root_id: [candidate] for candidate in initial}
        self.active_candidates = list(initial)
        self.max_candidates_per_seed = args.candidate_cap_per_seed or args.candidate_cap_multiplier * args.top_k
        self.receptor = None

    def _receptor_config(self) -> Any:
        if self.receptor is None:
            from extract_and_dock import prepare_receptor

            self.receptor = prepare_receptor(
                self.args.pdb_id,
                chain_id=self.args.chain_id,
                ligand_resname=self.args.ligand_resname,
                work_dir=self.args.work_dir,
                allow_zero_charge_fallback=self.args.allow_zero_charge_fallback,
            )
        return self.receptor

    def dock_initial_seeds(self) -> dict[str, Any]:
        from extract_and_dock import ExtractedCompound, dock_batch, write_docking_results_csv, write_joint_score_csv

        seed_dir = self.output_dir / "initial_seed_docking"
        seed_dir.mkdir(parents=True, exist_ok=True)
        compounds = [
            ExtractedCompound(
                compound_id=candidate.root_id,
                full_smiles=candidate.root_smiles,
                activity_nM=safe_float(candidate.activity_nM),
                activity_assay=candidate.activity_assay,
                activity_target=candidate.activity_target,
                needs_review=False,
            )
            for candidate in self.initial_candidates
        ]
        results = dock_batch(
            compounds,
            self._receptor_config(),
            exhaustiveness=self.args.docking_exhaustiveness,
            n_poses=self.args.num_modes,
            seed=self.args.docking_seed,
            use_cache=not self.args.no_cache,
            allow_debug_receptor=self.args.allow_debug_receptor,
            max_workers=self.args.docking_workers,
        )
        results_csv = seed_dir / "seed_docking_results.csv"
        joint_csv = seed_dir / "seed_docking_activity_joint_score.csv"
        write_docking_results_csv(results_csv, results)
        write_joint_score_csv(joint_csv, compounds, results)
        return {
            "results_by_id": {result.compound_id: result for result in results},
            "results_csv": str(results_csv),
            "joint_score_csv": str(joint_csv),
        }

    def __call__(self, trial: Any) -> float:
        from extract_and_dock import (
            compounds_from_csv,
            dock_batch,
            write_analog_group_score_csvs,
            write_docking_results_csv,
            write_joint_score_csv,
        )

        trial_dir = self.output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        input_txt = trial_dir / "reasyn_input.txt"
        seed_map_csv = trial_dir / "seed_map.csv"
        raw_analogs_csv = trial_dir / "reasyn_analogs.csv"
        annotated_analogs_csv = trial_dir / "reasyn_analogs_annotated.csv"
        docking_results_csv = trial_dir / "docking_results.csv"
        joint_score_csv = trial_dir / "docking_activity_joint_score.csv"
        analog_topk_csv = trial_dir / "analog_group_topk.csv"
        analog_overall_best_csv = trial_dir / "analog_overall_best.csv"
        trial_summary_path = trial_dir / "trial_summary.json"

        write_reasyn_input(input_txt, self.active_candidates)
        write_seed_map(seed_map_csv, self.active_candidates)

        params = sample_reasyn_params(trial, self.args)
        reasyn_repo = bridge.resolve_reasyn_repo(self.args.reasyn_repo)
        entrypoint = bridge.discover_reasyn_entrypoint(reasyn_repo, self.args.reasyn_entrypoint)
        model_paths, model_paths_csv = bridge.resolve_reasyn_model_paths(self.args.model_paths, reasyn_repo)
        missing_models = [str(path) for path in model_paths if not path.exists()]
        if len(model_paths) != 2 or missing_models:
            raise RuntimeError(
                "ReaSyn inference requires comma-separated AR and Edit Bridge checkpoints. "
                f"Missing/invalid model paths: {missing_models or model_paths_csv}"
            )
        if not entrypoint.exists():
            raise RuntimeError(f"ReaSyn entrypoint not found: {entrypoint}. Pass --reasyn-repo or --reasyn-entrypoint.")

        cmd = bridge.reasyn_command(
            python_bin=self.args.python_bin,
            entrypoint=entrypoint,
            model_paths_csv=model_paths_csv,
            input_txt=input_txt,
            output_csv=raw_analogs_csv,
            search_width=params["search_width"],
            exhaustiveness=params["exhaustiveness"],
            num_gpus=self.args.reasyn_num_gpus,
            num_workers_per_gpu=self.args.num_workers_per_gpu,
            task_qsize=self.args.task_qsize,
            result_qsize=self.args.result_qsize,
            time_limit=self.args.time_limit,
            add_bb_path=self.args.add_bb_path,
            no_exact_break=bool(params["no_exact_break"]),
            num_cycles=params["num_cycles"],
            num_editflow_samples=params["num_editflow_samples"],
            num_editflow_steps=params["num_editflow_steps"],
            mols_to_filter=self.args.mols_to_filter,
            filter_sim=params["filter_sim"],
        )
        proc = bridge.run_logged_command(cmd, trial_dir / "reasyn.log", timeout=self.args.timeout, cwd=reasyn_repo)
        if proc.returncode != 0:
            objective = self.args.failure_score
            trial.set_user_attr("reasyn_failed", True)
            trial.set_user_attr("message", bridge.command_error_tail(proc))
            trial_summary_path.write_text(
                json.dumps(
                    {
                        "trial_number": trial.number,
                        "objective": objective,
                        "params": params,
                        "active_seed_count": len(self.active_candidates),
                        "message": bridge.command_error_tail(proc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return objective

        annotated_rows = annotate_reasyn_analog_csv(
            raw_analogs_csv,
            annotated_analogs_csv,
            self.active_candidates,
            trial_number=trial.number,
        )
        raw_analog_count = len(annotated_rows)
        if self.args.max_generated_analogs_per_seed:
            annotated_rows = limit_annotated_analogs(annotated_rows, self.args.max_generated_analogs_per_seed)
            write_dict_csv(annotated_analogs_csv, annotated_rows, ["target", "smiles"])
        try:
            compounds = compounds_from_csv(annotated_analogs_csv, approved_ids=[], allow_unreviewed=True)
        except RuntimeError as exc:
            objective = self.args.failure_score
            trial.set_user_attr("message", str(exc))
            trial.set_user_attr("raw_analog_count", raw_analog_count)
            trial.set_user_attr("docked_analog_count", 0)
            trial_summary_path.write_text(
                json.dumps(
                    {
                        "trial_number": trial.number,
                        "objective": objective,
                        "params": params,
                        "active_seed_count": len(self.active_candidates),
                        "raw_analog_count": raw_analog_count,
                        "docked_analog_count": 0,
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return objective
        results = dock_batch(
            compounds,
            self._receptor_config(),
            exhaustiveness=self.args.docking_exhaustiveness,
            n_poses=self.args.num_modes,
            seed=self.args.docking_seed,
            use_cache=not self.args.no_cache,
            allow_debug_receptor=self.args.allow_debug_receptor,
            max_workers=self.args.docking_workers,
        )
        write_docking_results_csv(docking_results_csv, results)
        write_joint_score_csv(joint_score_csv, compounds, results)
        analog_summary = write_analog_group_score_csvs(
            analog_topk_csv,
            analog_overall_best_csv,
            compounds,
            results,
            top_k=self.args.top_k,
        )
        objective = best_vina_score_from_summary(analog_summary, self.args.failure_score)
        topk_candidates = read_topk_candidates(analog_topk_csv, self.root_smiles_by_id, trial_number=trial.number)
        self.archive, self.active_candidates = update_candidate_state(
            self.archive,
            self.active_candidates,
            topk_candidates,
            top_k=self.args.top_k,
            max_candidates_per_seed=self.max_candidates_per_seed,
        )

        trial.set_user_attr("best_vina_score_kcal_mol", objective)
        trial.set_user_attr("active_seed_count", len(self.active_candidates))
        trial.set_user_attr("raw_analog_count", raw_analog_count)
        trial.set_user_attr("docked_analog_count", len(compounds))
        trial.set_user_attr("archive_counts", {root_id: len(items) for root_id, items in self.archive.items()})
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial_summary = {
            "trial_number": trial.number,
            "objective": objective,
            "params": params,
            "active_seed_count": len(self.active_candidates),
            "candidate_cap_per_seed": self.max_candidates_per_seed,
            "max_generated_analogs_per_seed": self.args.max_generated_analogs_per_seed,
            "raw_analog_count": raw_analog_count,
            "docked_analog_count": len(compounds),
            "archive": {root_id: [asdict(candidate) for candidate in items] for root_id, items in self.archive.items()},
            "analog_summary": analog_summary,
            "paths": {
                "input_txt": str(input_txt),
                "seed_map_csv": str(seed_map_csv),
                "raw_analogs_csv": str(raw_analogs_csv),
                "annotated_analogs_csv": str(annotated_analogs_csv),
                "docking_results_csv": str(docking_results_csv),
                "analog_topk_csv": str(analog_topk_csv),
                "analog_overall_best_csv": str(analog_overall_best_csv),
            },
        }
        trial_summary_path.write_text(json.dumps(trial_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Trial {trial.number}: best Vina score {objective}")
        for row in analog_summary.get("group_best", []):
            print(f"  {row['group_id']}: {row['compound_id']} score={row['vina_score_kcal_mol']}")
        return objective


ReaSynOptunaObjective = ReaSynBOObjective


def build_reasyn_search_space(args: argparse.Namespace) -> SearchSpace:
    return SearchSpace(
        (
            Dimension.integer("search_width", args.search_width_range[0], args.search_width_range[1]),
            Dimension.integer("exhaustiveness", args.exhaustiveness_range[0], args.exhaustiveness_range[1]),
            Dimension.integer("num_cycles", args.num_cycles_range[0], args.num_cycles_range[1]),
            Dimension.integer(
                "num_editflow_samples",
                args.num_editflow_samples_range[0],
                args.num_editflow_samples_range[1],
            ),
            Dimension.integer(
                "num_editflow_steps",
                args.num_editflow_steps_range[0],
                args.num_editflow_steps_range[1],
            ),
            Dimension.floating("filter_sim", args.filter_sim_range[0], args.filter_sim_range[1]),
            Dimension.categorical("no_exact_break", (True, False)),
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize ReaSyn inference parameters with StrBO and Vina feedback.")
    parser.add_argument("--smiles-path", default=str(bridge.DEFAULT_SEED_CSV), help="PDF-extracted seed CSV with SMILES and activity.")
    parser.add_argument("--top-n-seeds", type=int, default=3, help="Initial seed count selected from the extraction CSV; 0 means all.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top K analogs per initial seed used for the next iteration.")
    parser.add_argument(
        "--candidate-cap-multiplier",
        type=int,
        default=2,
        help="Per-initial-seed archive cap multiplier; default keeps 2 * top_k candidates.",
    )
    parser.add_argument("--candidate-cap-per-seed", type=int, default=0, help="Explicit archive cap per initial seed; overrides multiplier.")
    parser.add_argument(
        "--max-generated-analogs-per-seed",
        type=int,
        default=0,
        help="Optional pre-docking ReaSyn analog cap per initial seed; 0 docks all generated analogs.",
    )
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--study-name", default="reasyn-vina-optimization")
    parser.add_argument("--storage", default=None, help="Deprecated compatibility option; StrBO writes JSON summaries instead.")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--strbo-initial-random", type=int, default=3, help="Random warm-up trials before fitting the string GP.")
    parser.add_argument("--strbo-candidate-pool-size", type=int, default=512, help="Generated candidate configs scored by the acquisition each BO step.")
    parser.add_argument("--strbo-acquisition", choices=["ei", "lcb", "ei_lcb"], default="ei_lcb")
    parser.add_argument("--strbo-xi", type=float, default=0.01, help="Expected-improvement exploration offset.")
    parser.add_argument("--strbo-beta", type=float, default=1.96, help="LCB exploration weight for lcb/ei_lcb acquisitions.")
    parser.add_argument("--strbo-kernel-max-ngram", type=int, default=5, help="Maximum character n-gram length for the string kernel.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--failure-score", type=float, default=DEFAULT_FAILURE_SCORE)

    parser.add_argument("--reasyn-repo", help="Path to a local NVIDIA-BioNeMo/ReaSyn checkout. Falls back to REASYN_HOME.")
    parser.add_argument("--reasyn-entrypoint", help="Path to ReaSyn scripts/sample.py.")
    parser.add_argument("--model-paths", default=bridge.DEFAULT_REASYN_MODEL_PATHS)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--num-gpus", dest="reasyn_num_gpus", type=int, default=-1)
    parser.add_argument("--num-workers-per-gpu", type=int, default=8)
    parser.add_argument("--task-qsize", type=int, default=0)
    parser.add_argument("--result-qsize", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=10000)
    parser.add_argument("--add-bb-path", default=None)
    parser.add_argument("--mols-to-filter", default=None)
    parser.add_argument("--timeout", type=int, default=3600)

    parser.add_argument("--search-width-range", type=int, nargs=2, default=[4, 24])
    parser.add_argument("--exhaustiveness-range", type=int, nargs=2, default=[32, 256])
    parser.add_argument("--num-cycles-range", type=int, nargs=2, default=[4, 16])
    parser.add_argument("--num-editflow-samples-range", type=int, nargs=2, default=[25, 150])
    parser.add_argument("--num-editflow-steps-range", type=int, nargs=2, default=[25, 150])
    parser.add_argument("--filter-sim-range", type=float, nargs=2, default=[0.6, 0.95])

    parser.add_argument("--pdb-id", default="8UN5")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--ligand-resname", default=None)
    parser.add_argument("--work-dir", default="docking_work")
    parser.add_argument("--docking-exhaustiveness", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=3)
    parser.add_argument("--docking-seed", type=int, default=42)
    parser.add_argument("--docking-workers", type=int, default=1, help="Parallel Vina subprocess workers.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-zero-charge-fallback", action="store_true")
    parser.add_argument("--allow-debug-receptor", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.top_n_seeds < 0:
        raise RuntimeError("--top-n-seeds must be non-negative.")
    if args.top_k < 1:
        raise RuntimeError("--top-k must be at least 1.")
    if args.candidate_cap_multiplier < 1:
        raise RuntimeError("--candidate-cap-multiplier must be at least 1.")
    if args.candidate_cap_per_seed < 0:
        raise RuntimeError("--candidate-cap-per-seed must be non-negative.")
    if args.candidate_cap_per_seed and args.candidate_cap_per_seed < args.top_k:
        raise RuntimeError("--candidate-cap-per-seed must be at least --top-k.")
    if args.max_generated_analogs_per_seed < 0:
        raise RuntimeError("--max-generated-analogs-per-seed must be non-negative.")
    if args.n_trials < 1:
        raise RuntimeError("--n-trials must be at least 1.")
    if args.strbo_initial_random < 1:
        raise RuntimeError("--strbo-initial-random must be at least 1.")
    if args.strbo_candidate_pool_size < 1:
        raise RuntimeError("--strbo-candidate-pool-size must be at least 1.")
    if args.strbo_kernel_max_ngram < 1:
        raise RuntimeError("--strbo-kernel-max-ngram must be at least 1.")
    if args.docking_workers < 1:
        raise RuntimeError("--docking-workers must be at least 1.")
    validate_range("--search-width-range", args.search_width_range)
    validate_range("--exhaustiveness-range", args.exhaustiveness_range)
    validate_range("--num-cycles-range", args.num_cycles_range)
    validate_range("--num-editflow-samples-range", args.num_editflow_samples_range)
    validate_range("--num-editflow-steps-range", args.num_editflow_steps_range)
    validate_range("--filter-sim-range", args.filter_sim_range)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.storage:
        print("Ignoring --storage: StrBO writes JSON summaries and does not use Optuna storage.", file=sys.stderr)

    objective = ReaSynBOObjective(args)
    seed_docking = objective.dock_initial_seeds()
    strbo_config = StrBOConfig(
        seed=args.sampler_seed,
        n_initial=min(args.strbo_initial_random, args.n_trials),
        candidate_pool_size=args.strbo_candidate_pool_size,
        acquisition=args.strbo_acquisition,
        xi=args.strbo_xi,
        beta=args.strbo_beta,
        kernel_max_ngram=args.strbo_kernel_max_ngram,
    )
    study = create_study(
        study_name=args.study_name,
        space=build_reasyn_search_space(args),
        direction="minimize",
        config=strbo_config,
    )
    study.optimize(objective, n_trials=args.n_trials)

    output_dir = Path(args.output_dir)
    best_per_seed_csv = output_dir / "best_analog_per_seed.csv"
    best_per_seed_rows = write_best_analog_per_seed_csv(
        best_per_seed_csv,
        objective.initial_candidates,
        objective.archive,
        seed_docking["results_by_id"],
    )
    summary = {
        "study_name": study.study_name,
        "optimizer": "strbo",
        "direction": "minimize",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "strbo": {
            "initial_random": strbo_config.n_initial,
            "candidate_pool_size": strbo_config.candidate_pool_size,
            "acquisition": strbo_config.acquisition,
            "xi": strbo_config.xi,
            "beta": strbo_config.beta,
            "kernel": {
                "type": "normalized_character_ngram",
                "max_ngram": strbo_config.kernel_max_ngram,
            },
            "storage": args.storage or "",
        },
        "top_k": args.top_k,
        "candidate_cap_per_seed": objective.max_candidates_per_seed,
        "max_generated_analogs_per_seed": args.max_generated_analogs_per_seed,
        "archive": {root_id: [asdict(candidate) for candidate in items] for root_id, items in objective.archive.items()},
        "initial_seed_docking": {
            "results_csv": seed_docking["results_csv"],
            "joint_score_csv": seed_docking["joint_score_csv"],
        },
        "best_analog_per_seed_csv": str(best_per_seed_csv),
        "best_analog_per_seed": best_per_seed_rows,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "state": trial.state,
            }
            for trial in study.trials
        ],
    }
    summary_path = output_dir / "strbo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Best StrBO trial {study.best_trial.number}: value={study.best_value}, params={study.best_params}")
    print(f"Wrote {best_per_seed_csv}")
    print(f"Wrote {summary_path}")
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
