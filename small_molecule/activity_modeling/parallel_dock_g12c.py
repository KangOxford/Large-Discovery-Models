#!/usr/bin/env python3
"""Parallel AutoDock Vina runner for the frozen G12C benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extract_and_dock import (  # noqa: E402
    DockingResult,
    canonicalize_smiles,
    command_error_tail,
    compounds_from_csv,
    find_local_tool,
    hash_text,
    model_to_plain_dict,
    parse_vina_score_from_pose,
    parse_vina_score_from_text,
    prepare_ligand,
    prepare_receptor,
    require_production_receptor,
    run_logged_command,
    work_dir_for_receptor,
    write_docking_results_csv,
    write_joint_score_csv,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel AutoDock Vina for G12C benchmark CSVs.")
    parser.add_argument(
        "--csv",
        default="activity_modeling/runs/g12c_expanded_20260612_122903/standard_experiment/docking_inputs/all_unique_test.csv",
    )
    parser.add_argument("--output-dir", default="output/docking_work/g12c_standard/all_unique_test")
    parser.add_argument("--work-dir", default="output/docking_work")
    parser.add_argument("--pdb-id", default="8UN5")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--ligand-resname", default=None)
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--vina-cpus", type=int, default=1, help="CPUs passed to each Vina process.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-zero-charge-fallback", action="store_true")
    parser.add_argument("--allow-debug-receptor", action="store_true")
    parser.add_argument("--write-every", type=int, default=1, help="Rewrite partial CSV every N completed molecules.")
    return parser.parse_args(argv)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cache_key_for(
    canonical: str,
    receptor: Any,
    *,
    exhaustiveness: int,
    n_poses: int,
    seed: int,
) -> str:
    return hash_text(
        json.dumps(
            {
                "engine": "vina",
                "smiles": canonical,
                "receptor": str(Path(receptor.receptor_pdbqt).resolve()),
                "box_center": receptor.box_center,
                "box_size": receptor.box_size,
                "prep_method": receptor.prep_method,
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "seed": seed,
            },
            sort_keys=True,
        ),
        20,
    )


def pose_key_for(
    canonical: str,
    receptor: Any,
    *,
    exhaustiveness: int,
    n_poses: int,
    seed: int,
) -> str:
    return hash_text(
        json.dumps(
            {
                "smiles": canonical,
                "receptor": str(Path(receptor.receptor_pdbqt).resolve()),
                "box_center": receptor.box_center,
                "box_size": receptor.box_size,
                "prep_method": receptor.prep_method,
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "seed": seed,
            },
            sort_keys=True,
        ),
        20,
    )


def cached_result(cache_path: Path, compound_id: str) -> Optional[DockingResult]:
    if not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        result = DockingResult(**cached)
    except Exception:
        return None
    if result.status == "ok" and result.pose_ref and Path(result.pose_ref).exists():
        result.compound_id = compound_id
        result.cached = True
        return result
    return None


def dock_one_parallel(
    *,
    compound_id: str,
    smiles: str,
    receptor: Any,
    use_cache: bool,
    exhaustiveness: int,
    n_poses: int,
    seed: int,
    vina_cpus: int,
    allow_debug_receptor: bool,
) -> DockingResult:
    require_production_receptor(receptor, allow_debug_receptor=allow_debug_receptor)
    canonical = canonicalize_smiles(smiles)
    result_id = compound_id or hash_text(canonical, 12)
    work_path = work_dir_for_receptor(receptor)
    cache_dir = work_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not canonical:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles="",
            score=None,
            pose_ref=None,
            status="prep_failed",
            message="Compound has no SMILES.",
        )
    cache_key = cache_key_for(canonical, receptor, exhaustiveness=exhaustiveness, n_poses=n_poses, seed=seed)
    cache_path = cache_dir / f"docking_{cache_key}.json"
    if use_cache:
        result = cached_result(cache_path, result_id)
        if result is not None:
            return result

    ligand_pdbqt = prepare_ligand(canonical, work_dir=work_path, seed=seed)
    if not ligand_pdbqt:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=None,
            status="prep_failed",
            message="Ligand PDBQT preparation failed.",
        )
    vina_bin = find_local_tool("vina", work_path)
    if not vina_bin:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=None,
            status="dock_failed",
            message="AutoDock Vina executable not found.",
        )

    pose_dir = work_path / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    pose_key = pose_key_for(canonical, receptor, exhaustiveness=exhaustiveness, n_poses=n_poses, seed=seed)
    pose_path = pose_dir / f"{result_id}_{pose_key}.pdbqt"
    log_path = pose_dir / f"{result_id}_{pose_key}.log"
    cmd = [
        vina_bin,
        "--receptor",
        receptor.receptor_pdbqt,
        "--ligand",
        ligand_pdbqt,
        "--center_x",
        f"{receptor.box_center[0]:.3f}",
        "--center_y",
        f"{receptor.box_center[1]:.3f}",
        "--center_z",
        f"{receptor.box_center[2]:.3f}",
        "--size_x",
        f"{receptor.box_size[0]:.3f}",
        "--size_y",
        f"{receptor.box_size[1]:.3f}",
        "--size_z",
        f"{receptor.box_size[2]:.3f}",
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(n_poses),
        "--seed",
        str(seed),
        "--cpu",
        str(max(1, vina_cpus)),
        "--out",
        str(pose_path),
    ]
    try:
        proc = run_logged_command(cmd, log_path, timeout=3600)
    except Exception as exc:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=str(log_path),
            status="dock_failed",
            message=f"Vina execution failed: {exc}",
        )
    if proc.returncode != 0:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=str(log_path),
            status="dock_failed",
            message=command_error_tail(proc),
        )
    score = parse_vina_score_from_text(proc.stdout or "")
    if score is None:
        score = parse_vina_score_from_pose(pose_path)
    if score is None:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=str(pose_path) if pose_path.exists() else str(log_path),
            status="dock_failed",
            message="Vina finished but no score could be parsed.",
        )
    result = DockingResult(
        compound_id=result_id,
        canonical_smiles=canonical,
        score=score,
        pose_ref=str(pose_path),
        status="ok",
    )
    atomic_write_json(cache_path, model_to_plain_dict(result))
    return result


def write_outputs(output_dir: Path, compounds: list[Any], results_by_index: dict[int, DockingResult]) -> None:
    ordered_results = [results_by_index[index] for index in sorted(results_by_index)]
    write_docking_results_csv(output_dir / "docking_results.partial.csv", ordered_results)
    if len(ordered_results) == len(compounds):
        write_docking_results_csv(output_dir / "docking_results.csv", ordered_results)
        write_joint_score_csv(output_dir / "docking_activity_joint_score.csv", compounds, ordered_results)


def summarize(results: list[DockingResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        key = "cached_ok" if result.cached and result.status == "ok" else result.status
        counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compounds = compounds_from_csv(Path(args.csv), approved_ids=[], allow_unreviewed=True)
    receptor = prepare_receptor(
        args.pdb_id,
        chain_id=args.chain_id,
        ligand_resname=args.ligand_resname,
        work_dir=args.work_dir,
        allow_zero_charge_fallback=args.allow_zero_charge_fallback,
    )
    require_production_receptor(receptor, allow_debug_receptor=args.allow_debug_receptor)

    start = time.time()
    results_by_index: dict[int, DockingResult] = {}
    futures: dict[concurrent.futures.Future[DockingResult], int] = {}
    print(
        f"[{dt.datetime.now().isoformat(timespec='seconds')}] starting parallel docking: "
        f"molecules={len(compounds)} workers={args.workers} vina_cpus={args.vina_cpus}",
        flush=True,
    )
    print(f"output_dir={output_dir}", flush=True)
    print(f"receptor={receptor.receptor_pdbqt}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, compound in enumerate(compounds):
            future = executor.submit(
                dock_one_parallel,
                compound_id=compound.compound_id,
                smiles=compound.full_smiles or "",
                receptor=receptor,
                use_cache=not args.no_cache,
                exhaustiveness=args.exhaustiveness,
                n_poses=args.num_modes,
                seed=args.seed,
                vina_cpus=args.vina_cpus,
                allow_debug_receptor=args.allow_debug_receptor,
            )
            futures[future] = index

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            compound = compounds[index]
            try:
                result = future.result()
            except Exception as exc:
                result = DockingResult(
                    compound_id=compound.compound_id,
                    canonical_smiles=canonicalize_smiles(compound.full_smiles or ""),
                    score=None,
                    pose_ref=None,
                    status="dock_failed",
                    message=f"Unhandled worker exception: {exc}",
                )
            results_by_index[index] = result
            completed += 1
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = len(compounds) - completed
            eta_min = remaining / rate / 60.0 if rate > 0 else 0.0
            print(
                f"[{dt.datetime.now().isoformat(timespec='seconds')}] "
                f"{completed}/{len(compounds)} {result.compound_id} status={result.status} "
                f"score={result.score} cached={result.cached} eta_min={eta_min:.1f}",
                flush=True,
            )
            if args.write_every > 0 and completed % args.write_every == 0:
                write_outputs(output_dir, compounds, results_by_index)

    write_outputs(output_dir, compounds, results_by_index)
    ordered_results = [results_by_index[index] for index in sorted(results_by_index)]
    metadata = {
        "input_csv": args.csv,
        "output_dir": str(output_dir),
        "work_dir": args.work_dir,
        "receptor": model_to_plain_dict(receptor),
        "parameters": {
            "pdb_id": args.pdb_id,
            "chain_id": args.chain_id,
            "exhaustiveness": args.exhaustiveness,
            "num_modes": args.num_modes,
            "seed": args.seed,
            "workers": args.workers,
            "vina_cpus": args.vina_cpus,
            "use_cache": not args.no_cache,
        },
        "elapsed_seconds": round(time.time() - start, 3),
        "counts": summarize(ordered_results),
    }
    atomic_write_json(output_dir / "docking_metadata.json", metadata)
    print(f"Wrote {output_dir / 'docking_results.csv'}", flush=True)
    print(f"Wrote {output_dir / 'docking_activity_joint_score.csv'}", flush=True)
    print(f"Wrote {output_dir / 'docking_metadata.json'}", flush=True)
    print(json.dumps(metadata["counts"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
