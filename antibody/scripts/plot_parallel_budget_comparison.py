#!/usr/bin/env python3
"""Compare parallel AntBO runs with different acquisition search budgets."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(r"antigen_(?P<antigen>.+?)_seed_(?P<seed>\d+)")
COLORS = {
    "budget 600": "#2f6fbb",
    "budget 1200": "#c43c39",
}


def antigen_from_path(csv_path: Path) -> str:
    match = RUN_RE.search(csv_path.parent.name)
    if not match:
        return csv_path.parent.name
    return match.group("antigen")


def load_curves(root: Path) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(root.glob("*/results.csv")):
        antigen = antigen_from_path(csv_path)
        df = pd.read_csv(csv_path)
        required = {"Index", "LastValue", "BestValue", "BestProtein"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        curves[antigen] = df
    if not curves:
        raise FileNotFoundError(f"no results.csv files found under {root}")
    return curves


def read_antigen_order(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None or not path.exists():
        return fallback
    antigens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return antigens or fallback


def final_summary(df: pd.DataFrame) -> tuple[float, int, str]:
    best = pd.to_numeric(df["BestValue"], errors="coerce")
    best_pos = int(best.idxmin())
    row = df.loc[best_pos]
    return float(best.iloc[-1]), int(row["Index"]), str(row["BestProtein"])


def plot(
    budget600_root: Path,
    budget1200_root: Path,
    out_path: Path,
    antigen_file: Path | None,
) -> None:
    methods = {
        "budget 600": load_curves(budget600_root),
        "budget 1200": load_curves(budget1200_root),
    }
    common = sorted(set(methods["budget 600"]) & set(methods["budget 1200"]))
    antigens = [ag for ag in read_antigen_order(antigen_file, common) if ag in common]
    if not antigens:
        raise ValueError("no shared antigens found between the two roots")

    height = max(2.35 * len(antigens), 8)
    fig, axes = plt.subplots(len(antigens), 1, figsize=(12.5, height), sharex=True, squeeze=False)
    axes_list = axes[:, 0]

    for ax, antigen in zip(axes_list, antigens):
        summaries: dict[str, tuple[float, int, str]] = {}
        for label, curves in methods.items():
            df = curves[antigen]
            x = pd.to_numeric(df["Index"], errors="coerce")
            last = pd.to_numeric(df["LastValue"], errors="coerce")
            best = pd.to_numeric(df["BestValue"], errors="coerce")
            color = COLORS[label]

            ax.scatter(x, last, s=11, color=color, alpha=0.16, edgecolors="none")
            ax.plot(x, best, color=color, lw=2.1, label=label)
            ax.scatter([x.iloc[-1]], [best.iloc[-1]], s=28, color=color, zorder=3)
            summaries[label] = final_summary(df)

        delta = summaries["budget 1200"][0] - summaries["budget 600"][0]
        winner = "1200 better" if delta < 0 else "600 better" if delta > 0 else "tie"
        summary_text = (
            f"600:  {summaries['budget 600'][0]:7.2f} @ {summaries['budget 600'][1]:3d}\n"
            f"1200: {summaries['budget 1200'][0]:7.2f} @ {summaries['budget 1200'][1]:3d}\n"
            f"delta: {delta:+7.2f} ({winner})"
        )
        ax.text(
            0.99,
            0.08,
            summary_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            family="monospace",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 3},
        )
        ax.axvspan(20, 199, color="#f4e2bd", alpha=0.14, lw=0)
        ax.axvline(20, color="#9c6a00", lw=1.0, ls="--", alpha=0.55)
        ax.set_title(antigen, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Best-so-far energy")
        ax.grid(True, alpha=0.25)

    axes_list[-1].set_xlabel("Evaluation index")
    axes_list[0].legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.26))
    fig.suptitle("parallel acquisition budget comparison: 600 vs 1200", y=0.998, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget600-root", type=Path, default=Path("TTS/logs/parallel600"))
    parser.add_argument("--budget1200-root", type=Path, default=Path("TTS/logs/parallel_1200"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TTS/logs/parallel_1200/parallel600_vs_parallel1200_5subplots.png"),
    )
    parser.add_argument("--antigens", type=Path, default=Path("test_5_antigens.txt"))
    args = parser.parse_args()
    plot(args.budget600_root, args.budget1200_root, args.output, args.antigens)


if __name__ == "__main__":
    main()
