#!/usr/bin/env python3
"""Plot best-so-far curves from extracted per-antigen JSON files.

Default input:
  outputs/comparisons/best_seed_json_by_antigen

Default output:
  outputs/comparisons/plots/best_seed_json_by_antigen_subplots.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_IN_DIR = Path("outputs/comparisons/best_seed_json_by_antigen")
TYPO_IN_DIR = Path("outputs/comparision/best_seed_json_by_antigen")
DEFAULT_OUT = Path("outputs/comparisons/plots/best_seed_json_by_antigen_subplots.png")
DEFAULT_EXCLUDE = "BO_ssk,BO_transformed_overlap_ntr"
DEFAULT_LEGEND_COLUMNS = 5
LABEL_MAP = {
    "BO+LLM": "LDM",
    "LDM": "LDM",
    "LLM baseline": "LLM Only",
    "LLM Only": "LLM Only",
    "BObert": "AntBO ProtBERT",
    "BO_ssk": "AntBO SSK",
    "BO_transformed_overlap": "AntBO TK",
    "BO_transformed_overlap_ntr": "AntBO NT",
    "BO_COMBO": "COMBO",
    "HEBO": "HEBO",
    "TURBO": "TuRBO",
    "GA": "Genetic Algorithm",
    "RS": "Random Search",
    "LamBO": "LamBO",
}
PREFERRED_STYLES = {
    "LDM": {"linestyle": "-", "linewidth": 3.0, "zorder": 5},
    "LLM Only": {"linestyle": "--", "linewidth": 2.3, "zorder": 4},
    "AntBO TK": {"linestyle": "-", "linewidth": 2.2, "zorder": 3},
}
STYLE_CYCLE = [
    {"linestyle": "-", "linewidth": 1.7},
    {"linestyle": "-", "linewidth": 1.7},
    {"linestyle": "-", "linewidth": 1.7},
    {"linestyle": "-", "linewidth": 1.7},
    {"linestyle": ":", "linewidth": 1.9},
    {"linestyle": "-", "linewidth": 1.6},
    {"linestyle": "-", "linewidth": 1.6},
    {"linestyle": "-", "linewidth": 1.6},
    {"linestyle": "-.", "linewidth": 1.8},
    {"linestyle": "-", "linewidth": 1.6},
]


def resolve_input_dir(path: Path) -> Path:
    if path.exists():
        return path
    if "comparision" in str(path):
        fallback = Path(str(path).replace("comparision", "comparisons"))
        if fallback.exists():
            return fallback
    if path == TYPO_IN_DIR and DEFAULT_IN_DIR.exists():
        return DEFAULT_IN_DIR
    return path


def load_antigen_jsons(in_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    paths = sorted(p for p in in_dir.glob("*.json") if p.name != "manifest.json")
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        antigen = path.stem
        rows.append((antigen, data))
    return rows


def cumulative_best(values: list[float], minimize: bool = True) -> list[float]:
    out: list[float] = []
    best = math.inf if minimize else -math.inf
    for value in values:
        best = min(best, value) if minimize else max(best, value)
        out.append(best)
    return out


def trajectory_curve(result: dict[str, Any], minimize: bool = True) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for pos, item in enumerate(result.get("trajectory", []), start=1):
        objective = item.get("objective")
        if objective is None:
            continue
        if isinstance(objective, list):
            if not objective:
                continue
            objective = objective[0]
        try:
            y = float(objective)
        except (TypeError, ValueError):
            continue
        xs.append(pos)
        ys.append(y)
    return xs, cumulative_best(ys, minimize=minimize)


def method_label(method: str) -> str:
    raw = method.removeprefix("paper:")
    return LABEL_MAP.get(raw, raw)


def parse_name_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def method_matches(method: str, names: set[str]) -> bool:
    raw = method.removeprefix("paper:")
    label = method_label(method)
    return method in names or raw in names or label in names


def plot_curves(
    antigen_data: list[tuple[str, dict[str, Any]]],
    out_path: Path,
    methods_filter: set[str] | None,
    exclude_methods: set[str],
    max_eval: int | None,
    dpi: int,
) -> None:
    if not antigen_data:
        raise ValueError("No antigen JSON files found.")

    n = len(antigen_data)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)

    all_handles = []
    all_labels = []
    style_by_label: dict[str, dict[str, Any]] = {}
    for ax, (antigen, data) in zip(axes.ravel(), antigen_data):
        direction = data.get("direction", "minimize")
        minimize = direction != "maximize"
        for result in data.get("results", []):
            method = str(result.get("method", "unknown"))
            label = method_label(method)
            if method_matches(method, exclude_methods):
                continue
            if methods_filter and not method_matches(method, methods_filter):
                continue
            xs, ys = trajectory_curve(result, minimize=minimize)
            if max_eval is not None:
                filtered = [(x, y) for x, y in zip(xs, ys) if x <= max_eval]
                if filtered:
                    xs, ys = map(list, zip(*filtered))
                else:
                    xs, ys = [], []
            if not xs:
                continue
            if label not in style_by_label:
                style_by_label[label] = PREFERRED_STYLES.get(
                    label,
                    STYLE_CYCLE[len(style_by_label) % len(STYLE_CYCLE)],
                )
            style = style_by_label[label]
            line, = ax.plot(xs, ys, alpha=0.92, label=label, **style)
            if label not in all_labels:
                all_handles.append(line)
                all_labels.append(label)

        ax.set_title(antigen)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Best binding energy")
        ax.grid(True, alpha=0.25, linewidth=0.8)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            ncol=min(DEFAULT_LEGEND_COLUMNS, len(all_labels)),
            fontsize=9,
            frameon=False,
        )
    fig.suptitle("Best-so-far binding energy by antigen", fontsize=16)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--methods",
        help="Comma-separated method names to plot. Accepts either full names like 'paper:BO_COMBO' or labels like 'BO_COMBO'. Defaults to all methods.",
    )
    parser.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help=f"Comma-separated method names to exclude. Default: {DEFAULT_EXCLUDE!r}. Use '' to disable.",
    )
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = resolve_input_dir(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.in_dir}")

    methods_filter = None
    if args.methods:
        methods_filter = parse_name_set(args.methods)
    exclude_methods = parse_name_set(args.exclude)

    antigen_data = load_antigen_jsons(in_dir)
    plot_curves(
        antigen_data=antigen_data,
        out_path=args.out,
        methods_filter=methods_filter,
        exclude_methods=exclude_methods,
        max_eval=args.max_eval,
        dpi=args.dpi,
    )
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
