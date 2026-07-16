from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from strbo_v1.acquisition import hypervolume


MINIMIZE = (True, False)
REF_POINT = (0.0, 5.0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(input_root)
    if not runs:
        raise SystemExit(f"No experiment outputs found under {input_root}")
    rows, curves = summarize_runs(runs, args.budget)
    write_metrics_csv(output_dir / "metrics_summary.csv", rows)
    write_curves_csv(output_dir / "hypervolume_curves.csv", curves)
    plot_hv(output_dir / "hypervolume_curves.png", curves)
    plot_final_hv(output_dir / "final_hypervolume.png", rows)
    plot_diagnostics(output_dir / "diagnostics.png", rows)
    write_report(output_dir / "analysis_report.md", rows)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, default=80)
    return parser.parse_args(argv)


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted((root / "baselines").glob("*_seed=*.json")):
        if path.name.endswith(".manifest.json"):
            continue
        runs.append(load_baseline_run(path))
    for summary in sorted((root / "tilted").glob("*_seed=*/summary.json")):
        runs.append(load_tilted_run(summary.parent))
    return runs


def load_baseline_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    method = path.stem.rsplit("_seed=", 1)[0]
    seed = int(path.stem.rsplit("_seed=", 1)[1])
    history = [
        (item["smiles"], tuple(item.get("scores", [])))
        for item in payload.get("history", [])
        if "scores" in item
    ]
    manifest = load_optional_json(path.with_suffix(".manifest.json"))
    llm_count = count_llm_attempts(payload.get("llm_trajectory"))
    return {
        "kind": "baseline",
        "method": method,
        "seed": seed,
        "history": history,
        "llm_call_count": llm_count,
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "path": str(path),
        "config": config,
    }


def load_tilted_run(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    history_payload = json.loads((path / "history.json").read_text(encoding="utf-8"))
    manifest = load_optional_json(path / "manifest.json")
    method = path.name.rsplit("_seed=", 1)[0]
    seed = int(path.name.rsplit("_seed=", 1)[1])
    rounds = load_rounds(path / "rounds.jsonl")
    history = [(item["smiles"], tuple(item.get("scores", []))) for item in history_payload]
    return {
        "kind": "tilted",
        "method": method,
        "seed": seed,
        "history": history,
        "llm_call_count": summary.get("llm_call_count"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "q0_entropy_final": summary.get("q0_entropy"),
        "prob_ess_final": summary.get("prob_effective_sample_size"),
        "drop_counts": summary.get("drop_counts", {}),
        "early_stop_reason": summary.get("early_stop_reason"),
        "rounds": rounds,
        "path": str(path),
    }


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_rounds(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_runs(runs: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, list[np.ndarray]] = defaultdict(list)
    for run in runs:
        curve = hv_curve(run["history"], budget)
        curves[run["method"]].append(curve)
        rows.append({
            "method": run["method"],
            "kind": run["kind"],
            "seed": run["seed"],
            "history_size": len(run["history"]),
            "final_hypervolume": float(curve[-1]) if len(curve) else 0.0,
            "llm_call_count": run.get("llm_call_count"),
            "elapsed_seconds": run.get("elapsed_seconds"),
            "q0_entropy_final": run.get("q0_entropy_final"),
            "prob_ess_final": run.get("prob_ess_final"),
            "early_stop_reason": run.get("early_stop_reason"),
            "path": run.get("path"),
        })
    rows.extend(aggregate_rows(curves, runs))
    return rows, curves


def hv_curve(history: list[tuple[str, tuple]], budget: int) -> np.ndarray:
    values: list[float] = []
    finite: list[tuple[float, float]] = []
    for _smiles, scores in history[:budget]:
        if len(scores) == 2 and all(score is not None for score in scores):
            finite.append((float(scores[0]), float(scores[1])))
        values.append(float(hypervolume(finite, REF_POINT, minimize=MINIMIZE)) if finite else 0.0)
    while len(values) < budget:
        values.append(values[-1] if values else 0.0)
    return np.asarray(values, dtype=float)


def aggregate_rows(curves: dict[str, list[np.ndarray]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_method[run["method"]].append(run)
    for method, method_curves in curves.items():
        arr = np.stack(method_curves, axis=0)
        elapsed = [r.get("elapsed_seconds") for r in by_method[method] if r.get("elapsed_seconds") is not None]
        llm = [r.get("llm_call_count") for r in by_method[method] if r.get("llm_call_count") is not None]
        rows.append({
            "method": method,
            "kind": "aggregate",
            "seed": "mean",
            "history_size": "",
            "final_hypervolume": float(arr[:, -1].mean()),
            "final_hypervolume_std": float(arr[:, -1].std()),
            "llm_call_count": float(np.mean(llm)) if llm else "",
            "elapsed_seconds": float(np.mean(elapsed)) if elapsed else "",
            "q0_entropy_final": "",
            "prob_ess_final": "",
            "early_stop_reason": "",
            "path": "",
        })
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_curves_csv(path: Path, curves: dict[str, list[np.ndarray]]) -> None:
    methods = sorted(curves)
    max_len = max(len(curve) for method_curves in curves.values() for curve in method_curves)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation"] + [f"{method}_mean" for method in methods] + [f"{method}_std" for method in methods])
        for idx in range(max_len):
            means = []
            stds = []
            for method in methods:
                arr = np.stack(curves[method], axis=0)
                means.append(float(arr[:, idx].mean()))
                stds.append(float(arr[:, idx].std()))
            writer.writerow([idx] + means + stds)


def plot_hv(path: Path, curves: dict[str, list[np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(curves):
        arr = np.stack(curves[method], axis=0)
        x = np.arange(arr.shape[1])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x, mean, label=method)
        if arr.shape[0] > 1:
            ax.fill_between(x, mean - std, mean + std, alpha=0.12)
    ax.set_xlabel("Expensive evaluations")
    ax.set_ylabel("Cumulative hypervolume")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def plot_final_hv(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = [row for row in rows if row["kind"] == "aggregate"]
    aggregate.sort(key=lambda row: row["method"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([row["method"] for row in aggregate], [row["final_hypervolume"] for row in aggregate])
    ax.set_ylabel("Final hypervolume")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def plot_diagnostics(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = [row for row in rows if row["kind"] == "aggregate"]
    aggregate.sort(key=lambda row: row["method"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar([row["method"] for row in aggregate], [float(row["llm_call_count"] or 0) for row in aggregate])
    axes[0].set_ylabel("Mean LLM calls")
    axes[1].bar([row["method"] for row in aggregate], [float(row["elapsed_seconds"] or 0) for row in aggregate])
    axes[1].set_ylabel("Mean elapsed seconds")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    aggregate = [row for row in rows if row["kind"] == "aggregate"]
    aggregate.sort(key=lambda row: float(row["final_hypervolume"]), reverse=True)
    lines = ["# Case2 Full Experiment Analysis", "", "| Method | Final HV mean | Final HV std | Mean LLM calls | Mean elapsed s |", "|---|---:|---:|---:|---:|"]
    for row in aggregate:
        lines.append(
            f"| {row['method']} | {float(row['final_hypervolume']):.6g} | "
            f"{float(row.get('final_hypervolume_std') or 0):.6g} | "
            f"{row.get('llm_call_count', '')} | {row.get('elapsed_seconds', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_llm_attempts(trajectory: Any) -> int:
    if not trajectory:
        return 0
    if isinstance(trajectory, dict):
        attempts = trajectory.get("attempts")
        if isinstance(attempts, list):
            return len(attempts)
        return sum(count_llm_attempts(value) for value in trajectory.values())
    if isinstance(trajectory, list):
        return sum(count_llm_attempts(value) for value in trajectory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
