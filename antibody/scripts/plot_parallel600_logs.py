#!/usr/bin/env python3
"""Plot the five parallel600 AntBO logs as per-antigen subplots."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(r"antigen_(?P<antigen>.+?)_seed_(?P<seed>\d+)")


SOURCE_COLORS = {
    "llm": "#2f6fbb",
    "ldm_parallel_ei_argmax": "#d28b18",
    "fallback_random": "#777777",
}


def antigen_from_run_dir(run_dir: Path) -> str:
    match = RUN_RE.search(run_dir.name)
    if not match:
        return run_dir.name
    return match.group("antigen")


def decision_iterations(run_dir: Path) -> list[int]:
    path = run_dir / "ldm_parallel_decisions.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    iterations = []
    for item in data.get("decisions", []):
        iteration = item.get("iteration")
        if isinstance(iteration, int):
            iterations.append(iteration)
    return iterations


def load_runs(root: Path) -> list[tuple[str, Path, pd.DataFrame, list[int]]]:
    runs = []
    for csv_path in sorted(root.glob("*/results.csv")):
        run_dir = csv_path.parent
        df = pd.read_csv(csv_path)
        required = {"Index", "LastValue", "BestValue", "Source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        antigen = antigen_from_run_dir(run_dir)
        runs.append((antigen, run_dir, df, decision_iterations(run_dir)))
    if not runs:
        raise FileNotFoundError(f"no results.csv files found under {root}")
    return runs


def plot(root: Path, out_path: Path) -> None:
    runs = load_runs(root)
    height = max(2.25 * len(runs), 8)
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, height), sharex=True, squeeze=False)
    axes_list = axes[:, 0]

    seen_sources: set[str] = set()
    for ax, (antigen, _run_dir, df, decisions) in zip(axes_list, runs):
        x = pd.to_numeric(df["Index"], errors="coerce")
        best = pd.to_numeric(df["BestValue"], errors="coerce")
        last = pd.to_numeric(df["LastValue"], errors="coerce")

        first_decision = min(decisions) if decisions else None
        if first_decision is not None:
            ax.axvspan(first_decision, float(x.max()), color="#f4e2bd", alpha=0.22, lw=0)
            ax.axvline(first_decision, color="#a36a00", lw=1.0, ls="--", alpha=0.65)

        for source, group in df.groupby("Source", dropna=False):
            source_name = str(source)
            label = source_name if source_name not in seen_sources else None
            color = SOURCE_COLORS.get(source_name, "#555555")
            ax.scatter(
                pd.to_numeric(group["Index"], errors="coerce"),
                pd.to_numeric(group["LastValue"], errors="coerce"),
                s=12,
                alpha=0.34,
                color=color,
                edgecolors="none",
                label=label,
            )
            seen_sources.add(source_name)

        ax.plot(x, best, color="#111111", lw=2.0, label="BestValue" if antigen == runs[0][0] else None)
        final_best = float(best.iloc[-1])
        best_idx = int(df.loc[best.idxmin(), "Index"])
        best_protein = str(df.loc[best.idxmin(), "BestProtein"])
        ax.text(
            0.99,
            0.08,
            f"final best {final_best:.2f} @ {best_idx}\n{best_protein}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3},
        )
        ax.set_title(antigen, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Binding energy")
        ax.grid(True, alpha=0.25)

    axes_list[-1].set_xlabel("Evaluation index")
    handles, labels = axes_list[0].get_legend_handles_labels()
    for ax in axes_list[1:]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
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
    fig.suptitle("parallel600 runs: last evaluations and best-so-far curves", y=0.998, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("TTS/logs/parallel600"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TTS/logs/parallel600/parallel600_5subplots.png"),
    )
    args = parser.parse_args()
    plot(args.root, args.output)


if __name__ == "__main__":
    main()
