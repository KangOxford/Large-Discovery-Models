#!/usr/bin/env python3
"""Plot all experiment setups under TTS/logs.

Top-level directories named like ``setup``, ``setup_run2``, ``setup_run3`` are
grouped as one setup. Within each setup, per-antigen ``BestValue`` curves are
aggregated across seeds as mean +/- std when multiple runs are present.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(r"antigen_(?P<antigen>.+?)_seed_(?P<seed>\d+)")
SETUP_RUN_RE = re.compile(r"^(?P<setup>.+?)(?:_run(?P<run>\d+))?$")
COLORS = [
    "#2f6fbb",
    "#c43c39",
    "#2f8f5b",
    "#8d5cc2",
    "#c9822e",
    "#555555",
]


def setup_name(root: Path) -> str:
    match = SETUP_RUN_RE.match(root.name)
    if not match:
        return root.name
    return match.group("setup")


def run_number(root: Path) -> int:
    match = SETUP_RUN_RE.match(root.name)
    if not match or match.group("run") is None:
        return 1
    return int(match.group("run"))


def parse_run_dir(run_dir: Path) -> tuple[str, int]:
    match = RUN_RE.search(run_dir.name)
    if not match:
        raise ValueError(f"could not parse antigen/seed from {run_dir}")
    return match.group("antigen"), int(match.group("seed"))


def discover_setup_roots(log_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for root in sorted(path for path in log_root.iterdir() if path.is_dir()):
        if not any(root.glob("*/results.csv")):
            continue
        grouped.setdefault(setup_name(root), []).append(root)
    for setup, roots in grouped.items():
        roots.sort(key=lambda path: (run_number(path), path.name))
    if not grouped:
        raise FileNotFoundError(f"no setup directories with results.csv found under {log_root}")
    return dict(sorted(grouped.items()))


def load_setup(roots: list[Path]) -> dict[str, list[tuple[int, pd.DataFrame, Path]]]:
    curves: dict[str, list[tuple[int, pd.DataFrame, Path]]] = {}
    for root in roots:
        for csv_path in sorted(root.glob("*/results.csv")):
            antigen, seed = parse_run_dir(csv_path.parent)
            df = pd.read_csv(csv_path)
            required = {"Index", "BestValue", "BestProtein"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            curves.setdefault(antigen, []).append((seed, df, csv_path))
    for antigen in curves:
        curves[antigen].sort(key=lambda item: item[0])
    return curves


def read_antigen_order(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None or not path.exists():
        return fallback
    antigens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ordered = [antigen for antigen in antigens if antigen in fallback]
    return ordered or fallback


def stack_best(items: list[tuple[int, pd.DataFrame, Path]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    min_len = min(len(df) for _, df, _ in items)
    x = pd.to_numeric(items[0][1]["Index"].iloc[:min_len], errors="coerce").to_numpy(dtype=float)
    y = np.stack(
        [
            pd.to_numeric(df["BestValue"].iloc[:min_len], errors="coerce").to_numpy(dtype=float)
            for _, df, _ in items
        ],
        axis=0,
    )
    seeds = [seed for seed, _, _ in items]
    return x, y, seeds


def best_seed_summary(items: list[tuple[int, pd.DataFrame, Path]]) -> tuple[int, float, int, str]:
    best_record: tuple[float, int, int, str] | None = None
    for seed, df, _ in items:
        values = pd.to_numeric(df["BestValue"], errors="coerce")
        final_value = float(values.iloc[-1])
        best_idx = int(df.loc[values.idxmin(), "Index"])
        best_protein = str(df.loc[values.idxmin(), "BestProtein"])
        record = (final_value, seed, best_idx, best_protein)
        if best_record is None or record[0] < best_record[0]:
            best_record = record
    assert best_record is not None
    final_value, seed, best_idx, best_protein = best_record
    return seed, final_value, best_idx, best_protein


def plot_all_setups(
    setups: dict[str, dict[str, list[tuple[int, pd.DataFrame, Path]]]],
    antigens: list[str],
    out_path: Path,
    summary_path: Path,
) -> None:
    fig_height = max(2.45 * len(antigens), 8)
    fig, axes = plt.subplots(len(antigens), 1, figsize=(13, fig_height), sharex=True, squeeze=False)
    axes_list = axes[:, 0]
    colors = {setup: COLORS[i % len(COLORS)] for i, setup in enumerate(setups)}
    summary_rows = []

    for ax, antigen in zip(axes_list, antigens):
        annotation_lines = []
        for setup, curves in setups.items():
            if antigen not in curves:
                continue
            x, y, seeds = stack_best(curves[antigen])
            mean = y.mean(axis=0)
            std = y.std(axis=0)
            color = colors[setup]
            label = f"{setup} (n={len(seeds)})"
            ax.plot(x, mean, color=color, lw=2.3, label=label)
            if len(seeds) > 1:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            else:
                ax.scatter([x[-1]], [mean[-1]], color=color, s=24, zorder=3)

            best_seed, best_final, best_idx, best_protein = best_seed_summary(curves[antigen])
            summary_rows.append(
                {
                    "Setup": setup,
                    "Antigen": antigen,
                    "Seeds": ",".join(map(str, seeds)),
                    "NSeeds": len(seeds),
                    "FinalMean": mean[-1],
                    "FinalStd": std[-1],
                    "BestFinalSeed": best_seed,
                    "BestFinalValue": best_final,
                    "BestFinalIndex": best_idx,
                    "BestProtein": best_protein,
                }
            )
            annotation_lines.append(f"{setup}: {mean[-1]:.2f} +/- {std[-1]:.2f} n={len(seeds)}")

        ax.axvspan(20, 199, color="#f4e2bd", alpha=0.11, lw=0)
        ax.axvline(20, color="#9c6a00", lw=1.0, ls="--", alpha=0.48)
        ax.text(
            0.99,
            0.08,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            family="monospace",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
        )
        ax.set_title(antigen, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Best-so-far energy")
        ax.grid(True, alpha=0.25)

    axes_list[-1].set_xlabel("Evaluation index")
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle("TTS/logs setup comparison", y=0.998, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"saved {out_path}")
    print(f"saved {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("TTS/logs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TTS/logs/all_setups_mean_std_comparison_5subplots.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("TTS/logs/all_setups_mean_std_summary.csv"),
    )
    parser.add_argument("--antigens", type=Path, default=Path("test_5_antigens.txt"))
    args = parser.parse_args()

    setup_roots = discover_setup_roots(args.log_root)
    setups = {setup: load_setup(roots) for setup, roots in setup_roots.items()}
    all_antigens = sorted(set().union(*(curves.keys() for curves in setups.values())))
    antigens = read_antigen_order(args.antigens, all_antigens)

    for setup, roots in setup_roots.items():
        seed_text = []
        for antigen, items in sorted(setups[setup].items()):
            seed_text.append(f"{antigen}:{','.join(str(seed) for seed, _, _ in items)}")
        print(f"{setup}: roots={[str(root) for root in roots]}")
        print(f"  {', '.join(seed_text)}")

    plot_all_setups(setups, antigens, args.output, args.summary)


if __name__ == "__main__":
    main()
