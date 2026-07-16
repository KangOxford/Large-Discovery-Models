#!/usr/bin/env python3
"""Plot mean +/- std convergence for the parallel_1200 setup."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(r"antigen_(?P<antigen>.+?)_seed_(?P<seed>\d+)")


def natural_key(path: Path) -> tuple[str, int]:
    match = re.search(r"run(\d+)$", path.name)
    return (path.name, int(match.group(1)) if match else 1)


def discover_roots(log_root: Path) -> list[Path]:
    roots = [path for path in log_root.glob("parallel_1200*") if path.is_dir()]
    roots = sorted(roots, key=natural_key)
    if not roots:
        raise FileNotFoundError(f"no parallel_1200* directories found under {log_root}")
    return roots


def parse_run_name(run_dir: Path) -> tuple[str, int]:
    match = RUN_RE.search(run_dir.name)
    if not match:
        raise ValueError(f"could not parse antigen/seed from {run_dir}")
    return match.group("antigen"), int(match.group("seed"))


def load_curves(roots: list[Path]) -> dict[str, list[tuple[int, pd.DataFrame, Path]]]:
    curves: dict[str, list[tuple[int, pd.DataFrame, Path]]] = {}
    for root in roots:
        for csv_path in sorted(root.glob("*/results.csv")):
            antigen, seed = parse_run_name(csv_path.parent)
            df = pd.read_csv(csv_path)
            required = {"Index", "BestValue", "BestProtein"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            curves.setdefault(antigen, []).append((seed, df, csv_path))
    if not curves:
        raise FileNotFoundError(f"no results.csv files found under {', '.join(map(str, roots))}")
    for antigen in curves:
        curves[antigen].sort(key=lambda item: item[0])
    return curves


def read_antigen_order(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None or not path.exists():
        return fallback
    antigens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [antigen for antigen in antigens if antigen in fallback] or fallback


def stack_best_values(items: list[tuple[int, pd.DataFrame, Path]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    min_len = min(len(df) for _, df, _ in items)
    first_df = items[0][1].iloc[:min_len]
    x = pd.to_numeric(first_df["Index"], errors="coerce").to_numpy(dtype=float)
    values = []
    seeds = []
    for seed, df, _ in items:
        values.append(pd.to_numeric(df["BestValue"].iloc[:min_len], errors="coerce").to_numpy(dtype=float))
        seeds.append(seed)
    return x, np.stack(values, axis=0), seeds


def best_seed_summary(items: list[tuple[int, pd.DataFrame, Path]]) -> tuple[int, float, int, str]:
    best = None
    for seed, df, _ in items:
        values = pd.to_numeric(df["BestValue"], errors="coerce")
        final_value = float(values.iloc[-1])
        best_idx = int(df.loc[values.idxmin(), "Index"])
        best_protein = str(df.loc[values.idxmin(), "BestProtein"])
        record = (final_value, seed, best_idx, best_protein)
        if best is None or record[0] < best[0]:
            best = record
    assert best is not None
    final_value, seed, best_idx, best_protein = best
    return seed, final_value, best_idx, best_protein


def plot(
    roots: list[Path],
    out_path: Path,
    summary_path: Path,
    antigen_file: Path | None,
) -> None:
    curves = load_curves(roots)
    antigens = read_antigen_order(antigen_file, sorted(curves))

    height = max(2.35 * len(antigens), 8)
    fig, axes = plt.subplots(len(antigens), 1, figsize=(12.5, height), sharex=True, squeeze=False)
    axes_list = axes[:, 0]
    summary_rows = []

    for ax, antigen in zip(axes_list, antigens):
        items = curves[antigen]
        x, y, seeds = stack_best_values(items)
        mean = y.mean(axis=0)
        std = y.std(axis=0)
        color = "#c43c39"

        for seed, run_y in zip(seeds, y):
            ax.plot(x, run_y, color="#8a8a8a", lw=1.0, alpha=0.34, label="individual seeds" if seed == seeds[0] else None)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.22, label="+/- 1 std")
        ax.plot(x, mean, color=color, lw=2.4, label="mean BestValue")
        ax.axvspan(20, float(x.max()), color="#f4e2bd", alpha=0.13, lw=0)
        ax.axvline(20, color="#9c6a00", lw=1.0, ls="--", alpha=0.55)

        best_seed, best_final, best_idx, best_protein = best_seed_summary(items)
        final_values = y[:, -1]
        summary_rows.append(
            {
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

        ax.text(
            0.99,
            0.08,
            (
                f"seeds: {','.join(map(str, seeds))}\n"
                f"final mean: {mean[-1]:7.2f} +/- {std[-1]:.2f}\n"
                f"best seed: {best_seed} ({best_final:.2f} @ {best_idx})"
            ),
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

        print(
            f"{antigen}: n={len(seeds)} seeds={seeds}, "
            f"final mean={mean[-1]:.2f}, std={std[-1]:.2f}, "
            f"finals={', '.join(f'{v:.2f}' for v in final_values)}"
        )

    axes_list[-1].set_xlabel("Evaluation index")
    handles, labels = axes_list[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=max(1, len(by_label)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle("parallel_1200 mean +/- std across 3 seeds", y=0.998, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"saved {out_path}")
    print(f"saved {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("TTS/logs"))
    parser.add_argument("--roots", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TTS/logs/parallel_1200/parallel1200_mean_std_3seeds_5subplots.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("TTS/logs/parallel_1200/parallel1200_mean_std_3seeds_summary.csv"),
    )
    parser.add_argument("--antigens", type=Path, default=Path("test_5_antigens.txt"))
    args = parser.parse_args()

    roots = args.roots if args.roots else discover_roots(args.log_root)
    plot(roots, args.output, args.summary, args.antigens)


if __name__ == "__main__":
    main()
