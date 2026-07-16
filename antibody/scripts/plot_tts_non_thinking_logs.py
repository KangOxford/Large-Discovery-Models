#!/usr/bin/env python3
"""Plot experiment setups under TTS/logs_non_thinking.

Top-level directories named like ``setup_run1`` and ``setup_run2`` are grouped
as one setup. Curves are aggregated across completed run folders, so repeated
seed IDs still count as separate runs.
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
DEFAULT_LOG_SOURCES = [
    ("EI", Path("TTS/logs_non_thinking")),
    ("UCB", Path("TTS/log_non_thinking_ucb")),
    ("Thinking", Path("TTS/logs_thinking")),
]


def combined_setup_name(acquisition: str, setup: str) -> str:
    return f"{acquisition}:{setup}"


def split_combined_setup(name: str) -> tuple[str | None, str]:
    if ":" not in name:
        return None, name
    acquisition, setup = name.split(":", 1)
    return acquisition, setup


def setup_sort_key(name: str) -> tuple[int, str, str]:
    acquisition, setup = split_combined_setup(name)
    match = re.search(r"(\d+)", setup)
    if match:
        return int(match.group(1)), acquisition or "", setup
    return 10**9, acquisition or "", setup


def setup_name(root: Path) -> str:
    match = SETUP_RUN_RE.match(root.name)
    return match.group("setup") if match else root.name


def display_setup_name(setup: str) -> str:
    acquisition, base_setup = split_combined_setup(setup)
    match = re.search(r"(\d+)", base_setup)
    if match:
        budget_label = f"Budget {match.group(1)}"
        return f"{acquisition} {budget_label}" if acquisition else budget_label
    return f"{acquisition} {base_setup}" if acquisition else base_setup


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
    for roots in grouped.values():
        roots.sort(key=lambda path: (run_number(path), path.name))
    if not grouped:
        raise FileNotFoundError(f"no setup directories with results.csv found under {log_root}")
    return dict(sorted(grouped.items(), key=lambda item: setup_sort_key(item[0])))


def discover_all_setup_roots(
    log_sources: list[tuple[str, Path]],
    excluded_setups: set[str],
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for acquisition, log_root in log_sources:
        if not log_root.exists():
            print(f"warning: skipped missing log root {log_root}")
            continue
        for setup, roots in discover_setup_roots(log_root).items():
            if setup in excluded_setups or combined_setup_name(acquisition, setup) in excluded_setups:
                continue
            grouped[combined_setup_name(acquisition, setup)] = roots
    if not grouped:
        raise SystemExit("no setup directories remain after applying log roots and --exclude-setups")
    return dict(sorted(grouped.items(), key=lambda item: setup_sort_key(item[0])))


def load_setup(roots: list[Path]) -> dict[str, list[dict[str, object]]]:
    curves: dict[str, list[dict[str, object]]] = {}
    for root in roots:
        run_id = run_number(root)
        for csv_path in sorted(root.glob("*/results.csv")):
            antigen, seed = parse_run_dir(csv_path.parent)
            df = pd.read_csv(csv_path)
            required = {"Index", "BestValue", "BestProtein"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            curves.setdefault(antigen, []).append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "df": df,
                    "path": csv_path,
                }
            )
    for antigen in curves:
        curves[antigen].sort(key=lambda item: (int(item["run_id"]), str(item["path"])))
    return curves


def read_antigen_order(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None or not path.exists():
        return fallback
    antigens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ordered = [antigen for antigen in antigens if antigen in fallback]
    return ordered or fallback


def stack_best(items: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    min_len = min(len(item["df"]) for item in items)
    first_df = items[0]["df"]
    x = pd.to_numeric(first_df["Index"].iloc[:min_len], errors="coerce").to_numpy(dtype=float)
    y = np.stack(
        [
            pd.to_numeric(item["df"]["BestValue"].iloc[:min_len], errors="coerce").to_numpy(dtype=float)
            for item in items
        ],
        axis=0,
    )
    labels = [f"run{item['run_id']}/seed{item['seed']}" for item in items]
    return x, y, labels


def best_run_summary(items: list[dict[str, object]]) -> tuple[str, float, int, str]:
    best_record: tuple[float, str, int, str] | None = None
    for item in items:
        df = item["df"]
        values = pd.to_numeric(df["BestValue"], errors="coerce")
        final_value = float(values.iloc[-1])
        best_idx = int(df.loc[values.idxmin(), "Index"])
        best_protein = str(df.loc[values.idxmin(), "BestProtein"])
        label = f"run{item['run_id']}/seed{item['seed']}"
        record = (final_value, label, best_idx, best_protein)
        if best_record is None or record[0] < best_record[0]:
            best_record = record
    assert best_record is not None
    final_value, label, best_idx, best_protein = best_record
    return label, final_value, best_idx, best_protein


def plot_all_setups(
    setups: dict[str, dict[str, list[dict[str, object]]]],
    antigens: list[str],
    out_path: Path,
    pdf_path: Path,
    summary_path: Path,
) -> None:
    ncols = min(3, len(antigens))
    nrows = int(np.ceil(len(antigens) / ncols))
    if len(antigens) == 5 and ncols == 3:
        fig = plt.figure(figsize=(5.4 * ncols, 3.9 * nrows))
        grid = fig.add_gridspec(nrows, 6)
        axes_list = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 1:3]),
            fig.add_subplot(grid[1, 3:5]),
        ]
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.9 * nrows), sharex=True, squeeze=False)
        axes_list = list(axes.flat[: len(antigens)])
        for ax in axes.flat[len(antigens):]:
            ax.axis("off")
    colors = {setup: COLORS[i % len(COLORS)] for i, setup in enumerate(setups)}
    summary_rows = []

    for ax, antigen in zip(axes_list, antigens):
        annotation_lines = []
        for setup, curves in setups.items():
            if antigen not in curves:
                continue
            x, y, run_labels = stack_best(curves[antigen])
            mean = y.mean(axis=0)
            std = y.std(axis=0)
            color = colors[setup]
            label = display_setup_name(setup)

            ax.plot(x, mean, color=color, lw=2.3, label=label)
            if len(run_labels) > 1:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            else:
                ax.scatter([x[-1]], [mean[-1]], color=color, s=24, zorder=3)

            best_label, best_final, best_idx, best_protein = best_run_summary(curves[antigen])
            summary_rows.append(
                {
                    "Acquisition": split_combined_setup(setup)[0],
                    "Setup": setup,
                    "Antigen": antigen,
                    "Runs": ",".join(run_labels),
                    "NRuns": len(run_labels),
                    "FinalMean": mean[-1],
                    "FinalStd": std[-1],
                    "BestFinalRun": best_label,
                    "BestFinalValue": best_final,
                    "BestFinalIndex": best_idx,
                    "BestProtein": best_protein,
                }
            )
            annotation_lines.append(f"{display_setup_name(setup)}: {mean[-1]:.2f} +/- {std[-1]:.2f}")

        ax.axvspan(20, 199, color="#f4e2bd", alpha=0.11, lw=0)
        ax.axvline(20, color="#9c6a00", lw=1.0, ls="--", alpha=0.48)
        ax.text(
            0.99,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            family="monospace",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
        )
        ax.set_title(antigen, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Best-so-far ΔG (lower = better)")
        ax.grid(True, alpha=0.25)

    bottom_start = 3 if len(antigens) == 5 and ncols == 3 else (nrows - 1) * ncols
    for ax in axes_list[bottom_start:]:
        ax.set_xlabel("Evaluation index")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "LDM-TTS (Qwen3.5-9B) for Antibody CDRH3 design across five antigens",
        y=0.998,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"saved {out_path}")
    print(f"saved {pdf_path}")
    print(f"saved {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("TTS/logs_non_thinking"))
    parser.add_argument(
        "--ucb-log-root",
        type=Path,
        default=Path("TTS/log_non_thinking_ucb"),
        help="Optional UCB log root to include alongside --log-root.",
    )
    parser.add_argument(
        "--thinking-log-root",
        type=Path,
        default=Path("TTS/logs_thinking"),
        help="Optional thinking log root to include alongside non-thinking logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TTS/logs_non_thinking/non_thinking_setups_mean_std_comparison_5subplots.png"),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="PDF output path. Defaults to --output with a .pdf suffix.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("TTS/logs_non_thinking/non_thinking_setups_mean_std_summary.csv"),
    )
    parser.add_argument("--antigens", type=Path, default=Path("test_5_antigens.txt"))
    parser.add_argument(
        "--exclude-setups",
        nargs="*",
        default=[],
        help=(
            "Setup names to omit, for example: parallel_600 parallel_1200. "
            "Use EI:parallel_600 to exclude only one acquisition source."
        ),
    )
    args = parser.parse_args()

    log_sources = [("EI", args.log_root)]
    if args.ucb_log_root is not None:
        log_sources.append(("UCB", args.ucb_log_root))
    if args.thinking_log_root is not None:
        log_sources.append(("Thinking", args.thinking_log_root))
    excluded = set(args.exclude_setups)
    setup_roots = discover_all_setup_roots(log_sources, excluded)
    setups = {setup: load_setup(roots) for setup, roots in setup_roots.items()}
    all_antigens = sorted(set().union(*(curves.keys() for curves in setups.values())))
    antigens = read_antigen_order(args.antigens, all_antigens)

    for setup, roots in setup_roots.items():
        run_text = []
        for antigen, items in sorted(setups[setup].items()):
            labels = [f"run{item['run_id']}/seed{item['seed']}" for item in items]
            run_text.append(f"{antigen}:{','.join(labels)}")
        print(f"{setup}: roots={[str(root) for root in roots]}")
        print(f"  {', '.join(run_text)}")

    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    plot_all_setups(setups, antigens, args.output, pdf_output, args.summary)


if __name__ == "__main__":
    main()
