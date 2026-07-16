#!/usr/bin/env python3
"""Plot Qwen3 9B Pareto hypervolume mean +/- std curves from TTS logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


GROUPS = [
    "qwen3_9B_proposer16_bo16",
    "qwen3_9B_proposer32_bo16",
    "qwen3_9B_proposer32_bo32",
    "qwen3_9B_proposer64_bo32",
    "qwen3_9B_proposer64_bo64",
]
REF_POINT = (0.0, 5.0)
MINIMIZE = (True, False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = Path(args.runs_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    curves_by_group, run_rows = load_group_curves(runs_root, args.budget)
    if not curves_by_group:
        raise SystemExit(f"no score-bearing qwen3_9B logs found under {runs_root}")

    write_curve_csv(output.with_suffix(".csv"), curves_by_group)
    write_run_summary(output.with_name(output.stem + "_runs.csv"), run_rows)
    plot_curves(output, curves_by_group)
    pdf_output = output.with_suffix(".pdf")
    plot_curves(pdf_output, curves_by_group)

    print(f"wrote {output.resolve()}")
    print(f"wrote {pdf_output.resolve()}")
    print(f"wrote {output.with_suffix('.csv').resolve()}")
    print(f"wrote {output.with_name(output.stem + '_runs.csv').resolve()}")
    for row in run_rows:
        note = " padded" if row["evaluations"] < args.budget else ""
        print(
            f"{row['group']} {row['run_dir']}: "
            f"{row['evaluations']} evaluations, final HV={row['final_hypervolume']:.6f}{note}"
        )
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Pareto hypervolume mean +/- std for selected Qwen3 9B TTS runs. "
            "Each run is normalized to include an initial x=0, HV=0 point."
        )
    )
    parser.add_argument("--runs-root", default="TTS/runs")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument(
        "--output",
        default="TTS/runs/qwen3_9B_pareto_hypervolume_mean_std.png",
    )
    return parser.parse_args(argv)


def load_group_curves(
    runs_root: Path,
    budget: int,
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, Any]]]:
    curves_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    run_rows: list[dict[str, Any]] = []
    for group in GROUPS:
        for run_dir in sorted(runs_root.glob(f"{group}_run*")):
            scores = load_scores(run_dir)
            if not scores:
                print(f"skipping {run_dir}: no selected scores in history or rounds")
                continue
            curve = hypervolume_curve(scores, budget)
            curves_by_group[group].append(curve)
            run_rows.append(
                {
                    "group": group,
                    "run_dir": run_dir.name,
                    "evaluations": min(len(scores), budget),
                    "raw_scores": len(scores),
                    "final_hypervolume": float(curve[-1]),
                }
            )
    return dict(curves_by_group), run_rows


def load_scores(run_dir: Path) -> list[tuple[float, float]]:
    history_scores = load_history_scores(run_dir / "history.json")
    if history_scores:
        return history_scores
    return load_round_scores(run_dir / "rounds.jsonl")


def load_history_scores(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    scores: list[tuple[float, float]] = []
    for item in payload:
        parsed = parse_score_pair(item.get("scores"))
        if parsed is not None:
            scores.append(parsed)
    return scores


def load_round_scores(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    scores: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            selection = record.get("selection_results", {})
            selected_scores = selection.get("selected_scores") or []
            if not selected_scores:
                continue
            parsed = parse_score_pair(selected_scores[0])
            if parsed is not None:
                scores.append(parsed)
    return scores


def parse_score_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or len(value) != 2:
        return None
    if value[0] is None or value[1] is None:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def hypervolume_curve(scores: Sequence[tuple[float, float]], budget: int) -> np.ndarray:
    values = [0.0]
    finite: list[tuple[float, float]] = []
    for score in scores[:budget]:
        finite.append(score)
        pareto = pareto_front_2d(finite, MINIMIZE)
        values.append(hypervolume_2d(pareto, REF_POINT, MINIMIZE))
    while len(values) < budget + 1:
        values.append(values[-1])
    return np.asarray(values[: budget + 1], dtype=float)


def pareto_front_2d(
    points: Sequence[tuple[float, float]],
    minimize: Sequence[bool],
) -> list[tuple[float, float]]:
    front = []
    for idx, point in enumerate(points):
        if not any(
            other_idx != idx and dominates_2d(other, point, minimize)
            for other_idx, other in enumerate(points)
        ):
            front.append(point)
    return front


def dominates_2d(
    left: tuple[float, float],
    right: tuple[float, float],
    minimize: Sequence[bool],
) -> bool:
    better_or_equal = []
    strictly_better = []
    for left_value, right_value, is_minimize in zip(left, right, minimize):
        if is_minimize:
            better_or_equal.append(left_value <= right_value)
            strictly_better.append(left_value < right_value)
        else:
            better_or_equal.append(left_value >= right_value)
            strictly_better.append(left_value > right_value)
    return all(better_or_equal) and any(strictly_better)


def hypervolume_2d(
    points: Sequence[tuple[float, float]],
    ref_point: Sequence[float],
    minimize: Sequence[bool],
) -> float:
    rectangles = []
    ref_vina, ref_activity = float(ref_point[0]), float(ref_point[1])
    for vina, activity in points:
        x = ref_vina - vina if minimize[0] else vina - ref_vina
        y = ref_activity - activity if minimize[1] else activity - ref_activity
        if x > 0.0 and y > 0.0:
            rectangles.append((x, y))
    if not rectangles:
        return 0.0
    rectangles.sort(key=lambda item: item[0])
    suffix_y = [0.0 for _ in rectangles]
    current = 0.0
    for idx in range(len(rectangles) - 1, -1, -1):
        current = max(current, rectangles[idx][1])
        suffix_y[idx] = current
    area = 0.0
    prev_x = 0.0
    for (x, _y), height in zip(rectangles, suffix_y):
        area += max(0.0, x - prev_x) * height
        prev_x = x
    return float(area)


def write_curve_csv(path: Path, curves_by_group: dict[str, list[np.ndarray]]) -> None:
    groups = [group for group in GROUPS if group in curves_by_group]
    max_len = max(len(curve) for curves in curves_by_group.values() for curve in curves)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["evaluation"]
            + [f"{group}_mean" for group in groups]
            + [f"{group}_std" for group in groups]
            + [f"{group}_n" for group in groups]
        )
        for idx in range(max_len):
            means = []
            stds = []
            ns = []
            for group in groups:
                arr = np.stack(curves_by_group[group], axis=0)
                means.append(float(arr[:, idx].mean()))
                stds.append(float(arr[:, idx].std()))
                ns.append(int(arr.shape[0]))
            writer.writerow([idx] + means + stds + ns)


def write_run_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = ["group", "run_dir", "evaluations", "raw_scores", "final_hypervolume"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, curves_by_group: dict[str, list[np.ndarray]]) -> None:
    cache_dir = path.parent / ".plot_cache"
    mpl_cache = cache_dir / "matplotlib"
    xdg_cache = cache_dir / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for group in GROUPS:
        curves = curves_by_group.get(group)
        if not curves:
            continue
        arr = np.stack(curves, axis=0)
        x = np.arange(arr.shape[1])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        label = group.replace("qwen3_9B_", "")
        ax.plot(x, mean, linewidth=2.0, label=f"{label}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.16, linewidth=0)

    ax.set_xlabel("Expensive evaluations")
    ax.set_ylabel("Pareto hypervolume")
    ax.set_title("LDM-TTS (Qwen3.5-9B) for Molecular Design: Pareto Hypervolume Mean +/- Std")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
