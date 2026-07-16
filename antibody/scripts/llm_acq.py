#!/usr/bin/env python3
"""Recreate the all-methods figure and add the LLM+Acq curve.

This script is intentionally fixed to the local result layout under
``outputs/`` so the figure can be regenerated with one command.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANTIGENS_FILE = Path("test_5_antigens.txt")
OUT_PATH = Path("outputs/comparison/results/plots/all_methods_with_llm_acq.png")

EXPERIMENT_METHODS = [
    ("LDM", Path("outputs/ldm/ldm_ninit20_iter200")),
    ("Pure LLM", Path("outputs/llm_baseline/llm_baseline_5x5_200")),
    ("LLM+Acq", Path("outputs/llm_acq_5antigen_5seed_200eval")),
]

REPRODUCTION_METHODS = [
    ("AntBO", Path("outputs/reproduction/BO_transformed_overlap_optim_res.csv")),
    ("HEBO", Path("outputs/reproduction/HEBO_optim_res.csv")),
    ("TURBO", Path("outputs/reproduction/TURBO_optim_res.csv")),
    ("COMBO", Path("outputs/reproduction/BO_COMBO_optim_res.csv")),
    ("RS", Path("outputs/reproduction/RS_optim_res.csv")),
]

COLORS = {
    "LDM": "red",
    "Pure LLM": "blue",
    "LLM+Acq": "black",
    "AntBO": "green",
    "HEBO": "cyan",
    "TURBO": "orange",
    "COMBO": "purple",
    "RS": "gray",
}

RUN_RE = re.compile(r"antigen_(?P<rest>.+?)_seed_(?P<seed>\d+)")


def read_antigens(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def truncate_and_stack(curves: list[np.ndarray]) -> np.ndarray:
    min_len = min(len(curve) for curve in curves)
    return np.stack([curve[:min_len] for curve in curves], axis=0)


def antigen_from_results_path(path: Path) -> str | None:
    match = RUN_RE.search(path.parent.name)
    if not match:
        return None
    antigen = match.group("rest")
    if "_kernel_" in antigen:
        antigen = antigen.split("_kernel_", 1)[0]
    return antigen


def load_experiment_root(root: Path, antigens: set[str]) -> dict[str, np.ndarray]:
    if not root.exists():
        raise FileNotFoundError(root)

    grouped: dict[str, list[np.ndarray]] = {}
    for csv_path in sorted(root.rglob("results.csv")):
        antigen = antigen_from_results_path(csv_path)
        if antigen is None or antigen not in antigens:
            continue
        df = pd.read_csv(csv_path)
        if "BestValue" not in df.columns:
            raise ValueError(f"{csv_path} missing BestValue")
        grouped.setdefault(antigen, []).append(
            pd.to_numeric(df["BestValue"], errors="coerce").to_numpy(dtype=float)
        )
    return {antigen: truncate_and_stack(curves) for antigen, curves in grouped.items()}


def load_reproduction_csv(csv_path: Path, antigens: set[str]) -> dict[str, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    grouped: dict[str, list[np.ndarray]] = {}
    for antigen, ag_df in df.groupby("Antigen"):
        antigen = str(antigen)
        if antigen not in antigens:
            continue
        for _, seed_df in ag_df.groupby("Seed"):
            seed_df = seed_df.sort_values("Num BB Evals")
            grouped.setdefault(antigen, []).append(
                pd.to_numeric(seed_df["Best Binding Energy"], errors="coerce").to_numpy(dtype=float)
            )
    return {antigen: truncate_and_stack(curves) for antigen, curves in grouped.items()}


def load_all_methods(antigens: list[str]) -> dict[str, dict[str, np.ndarray]]:
    antigen_set = set(antigens)
    methods: dict[str, dict[str, np.ndarray]] = {}
    for label, root in EXPERIMENT_METHODS:
        methods[label] = load_experiment_root(root, antigen_set)
    for label, csv_path in REPRODUCTION_METHODS:
        methods[label] = load_reproduction_csv(csv_path, antigen_set)
    return methods


def plot(methods: dict[str, dict[str, np.ndarray]], antigens: list[str], out_path: Path) -> None:
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16.5, 7.6), squeeze=False)

    for ax, antigen in zip(axes.flat, antigens):
        ann_lines = []
        for label, curves_by_antigen in methods.items():
            if antigen not in curves_by_antigen:
                continue
            y = curves_by_antigen[antigen]
            x = np.arange(1, y.shape[1] + 1)
            mean = y.mean(axis=0)
            std = y.std(axis=0) if y.shape[0] > 1 else np.zeros_like(mean)
            color = COLORS[label]
            ax.plot(x, mean, color=color, lw=2.0, label=f"{label} ({y.shape[0]} seeds)")
            if y.shape[0] > 1:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            ann_lines.append(f"{label}: {mean[-1]:.1f} ± {std[-1]:.1f}")

        ax.text(
            0.98,
            0.97,
            "\n".join(ann_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            family="monospace",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax.set_title(antigen)
        ax.set_xlabel("Evaluation index")
        ax.set_ylabel("Best-so-far ΔG (lower = better)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85)

    for ax in axes.flat[len(antigens):]:
        ax.axis("off")

    fig.suptitle("All methods comparison (5 antigens)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


def main() -> None:
    antigens = read_antigens(ANTIGENS_FILE)
    methods = load_all_methods(antigens)
    for label, curves in methods.items():
        loaded = ", ".join(f"{ag}:{curves[ag].shape[0]}" for ag in antigens if ag in curves)
        print(f"{label}: {loaded}")
    plot(methods, antigens, OUT_PATH)


if __name__ == "__main__":
    main()
