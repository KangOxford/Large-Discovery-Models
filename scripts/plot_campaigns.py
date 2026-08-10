#!/usr/bin/env python3
"""Plot comparable trajectories from persisted antibody, molecule, and nanoGPT runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.scripts.plot_pareto_hv import (  # noqa: E402
    hypervolume_curve,
    load_run_scores,
)


COLORS = {
    "antibody": "#007C83",
    "molecule": "#D97706",
    "nanogpt": "#C2413B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot real LDM trajectories from their persisted artifacts."
    )
    parser.add_argument("--antibody-results", type=Path, required=True)
    parser.add_argument("--molecule-run", type=Path, required=True)
    parser.add_argument("--nanogpt-buffer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nanogpt-warmup-successes",
        type=int,
        default=None,
        help=(
            "Finite warm-up observations before LCB. By default this is inferred "
            "from metrics.model_based_iteration, falling back to 20."
        ),
    )
    parser.add_argument(
        "--nanogpt-total-iterations",
        type=int,
        default=None,
        help="Expected LCB iterations; when incomplete, label the nanoGPT plot interim.",
    )
    return parser.parse_args()


def read_antibody(path: Path) -> tuple[list[int], list[float], list[float]]:
    evaluations: list[int] = []
    observed: list[float] = []
    best: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            value = float(row["LastValue"])
            best_value = float(row.get("BestValue", min([*observed, value])))
            if not math.isfinite(value) or not math.isfinite(best_value):
                continue
            evaluations.append(row_index)
            observed.append(value)
            best.append(best_value)
    return evaluations, observed, best


def read_molecule(path: Path) -> tuple[list[int], list[float]]:
    _run_dir, scores, _summary = load_run_scores(path)
    curve = hypervolume_curve(
        [row.scores for row in scores],
        len(scores),
        ref_point=(0.0, 5.0),
        minimize=(True, False),
    )
    return list(range(len(curve))), [float(value) for value in curve]


def read_nanogpt(
    path: Path,
) -> tuple[list[int], list[float], list[float], int, int]:
    scores: list[float] = []
    inferred_warmup = 0
    model_iterations: list[int] = []
    model_search_started = False

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            score = float(row["score"])
            if not math.isfinite(score):
                continue
            scores.append(score)
            model_iteration = row.get("metrics", {}).get("model_based_iteration")
            if model_iteration is None and not model_search_started:
                inferred_warmup += 1
            elif model_iteration is not None:
                model_search_started = True
                model_iterations.append(int(model_iteration))

    best: list[float] = []
    incumbent = math.inf
    for score in scores:
        incumbent = min(incumbent, score)
        best.append(incumbent)

    completed_iterations = max(model_iterations, default=0)
    return (
        list(range(1, len(scores) + 1)),
        scores,
        best,
        inferred_warmup,
        completed_iterations,
    )


def write_curve(path: Path, header: list[str], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def configure_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D6D9DC", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_antibody(
    evaluations: list[int],
    observed: list[float],
    best: list[float],
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.scatter(evaluations, observed, s=14, color="#9AA0A6", alpha=0.5, label="Observed")
    axis.step(
        evaluations,
        best,
        where="post",
        linewidth=2.2,
        color=COLORS["antibody"],
        label="Best so far",
    )
    axis.set(
        title="Antibody LDM: UCB on 1ADQ_A",
        xlabel="Absolut evaluation",
        ylabel="Binding energy (lower is better)",
    )
    configure_axis(axis)
    axis.legend(frameon=False)
    save_figure(figure, output_dir / "antibody_ucb_trajectory")


def plot_molecule(evaluations: list[int], hypervolume: list[float], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.step(evaluations, hypervolume, where="post", linewidth=2.2, color=COLORS["molecule"])
    axis.set(
        title="Small-Molecule LDM: EHVI",
        xlabel="Vina/activity evaluation",
        ylabel="Pareto hypervolume (higher is better)",
    )
    configure_axis(axis)
    save_figure(figure, output_dir / "small_molecule_ehvi_trajectory")


def nanogpt_title(completed: int, total: int | None) -> str:
    if total is not None and completed < total:
        return f"NanoGPT LDM: LCB N4H4 ({completed}/{total} interim)"
    return "NanoGPT LDM: LCB N4H4"


def plot_nanogpt(
    evaluations: list[int],
    scores: list[float],
    best: list[float],
    warmup: int,
    completed: int,
    total: int | None,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.scatter(evaluations, scores, s=14, color="#9AA0A6", alpha=0.5, label="Observed")
    axis.step(
        evaluations,
        best,
        where="post",
        linewidth=2.2,
        color=COLORS["nanogpt"],
        label="Best so far",
    )
    if len(evaluations) > warmup:
        axis.axvline(
            warmup + 0.5,
            color="#4B5563",
            linestyle="--",
            linewidth=1.1,
            label="LCB search starts",
        )
    axis.set(
        title=nanogpt_title(completed, total),
        xlabel="Finite real 300-second training evaluation",
        ylabel="Validation bits per byte (lower is better)",
    )
    configure_axis(axis)
    axis.legend(frameon=False)
    save_figure(figure, output_dir / "nanogpt_lcb_trajectory")


def plot_combined(
    antibody: tuple[list[int], list[float], list[float]],
    molecule: tuple[list[int], list[float]],
    nanogpt: tuple[list[int], list[float], list[float]],
    warmup: int,
    completed: int,
    total: int | None,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    antibody_x, _antibody_observed, antibody_best = antibody
    molecule_x, molecule_hv = molecule
    nano_x, _nano_scores, nano_best = nanogpt

    axes[0].step(antibody_x, antibody_best, where="post", linewidth=2.2, color=COLORS["antibody"])
    axes[0].set(title="Antibody / UCB", xlabel="Evaluation", ylabel="Best binding energy")
    axes[1].step(molecule_x, molecule_hv, where="post", linewidth=2.2, color=COLORS["molecule"])
    axes[1].set(title="Small molecule / EHVI", xlabel="Evaluation", ylabel="Pareto hypervolume")
    axes[2].step(nano_x, nano_best, where="post", linewidth=2.2, color=COLORS["nanogpt"])
    if len(nano_x) > warmup:
        axes[2].axvline(warmup + 0.5, color="#4B5563", linestyle="--", linewidth=1.0)
    axes[2].set(
        title=nanogpt_title(completed, total).replace("NanoGPT LDM: ", "NanoGPT / "),
        xlabel="Finite evaluation",
        ylabel="Best val_bpb",
    )
    for axis in axes:
        configure_axis(axis)
    figure.suptitle("Real LDM Campaign Examples", fontsize=14)
    figure.tight_layout()
    save_figure(figure, output_dir / "ldm_three_tasks")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    antibody = read_antibody(args.antibody_results)
    molecule = read_molecule(args.molecule_run)
    nano_x, nano_scores, nano_best, inferred_warmup, completed = read_nanogpt(
        args.nanogpt_buffer
    )
    nanogpt = (nano_x, nano_scores, nano_best)
    warmup = (
        inferred_warmup
        if args.nanogpt_warmup_successes is None
        else args.nanogpt_warmup_successes
    )

    if not antibody[0] or len(molecule[0]) <= 1 or not nanogpt[0]:
        raise SystemExit("Each campaign must contain at least one finite evaluation before plotting.")

    plot_antibody(*antibody, args.output_dir)
    plot_molecule(*molecule, args.output_dir)
    plot_nanogpt(
        *nanogpt,
        warmup,
        completed,
        args.nanogpt_total_iterations,
        args.output_dir,
    )
    plot_combined(
        antibody,
        molecule,
        nanogpt,
        warmup,
        completed,
        args.nanogpt_total_iterations,
        args.output_dir,
    )

    write_curve(
        args.output_dir / "antibody_ucb_trajectory.csv",
        ["evaluation", "observed_energy", "best_energy"],
        list(zip(*antibody)),
    )
    write_curve(
        args.output_dir / "small_molecule_ehvi_trajectory.csv",
        ["evaluation", "hypervolume"],
        list(zip(*molecule)),
    )
    write_curve(
        args.output_dir / "nanogpt_lcb_trajectory.csv",
        ["evaluation", "observed_val_bpb", "best_val_bpb"],
        list(zip(*nanogpt)),
    )

    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "antibody_evaluations": len(antibody[0]),
        "molecule_evaluations": len(molecule[0]) - 1,
        "nanogpt_finite_evaluations": len(nanogpt[0]),
        "nanogpt_warmup_successes": warmup,
        "nanogpt_completed_model_iterations": completed,
        "nanogpt_expected_model_iterations": args.nanogpt_total_iterations,
    }
    (args.output_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
