#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


FAILURE_SCORE_CUTOFF = 1.0e8


@dataclass(frozen=True)
class IterationPoint:
    iteration: int
    iteration_score: float | None
    best_after: float
    state_id: str
    status: str = ""
    description: str = ""


@dataclass(frozen=True)
class FeatureExpansion:
    iteration: int
    active_feature_count: int | None
    name: str
    state_id: str = ""


@dataclass(frozen=True)
class FeatureExpansionAnnotation:
    iteration: int
    from_count: int | None
    to_count: int | None
    names: list[str]


@dataclass(frozen=True)
class PlotData:
    run_name: str
    metric: str
    minimize: bool
    points: list[IterationPoint]
    warmup_best: float | None
    source: Path
    feature_expansions: list[FeatureExpansion]


SERIES_COLORS = [
    "#0f766e",
    "#2563eb",
    "#b45309",
    "#be123c",
    "#4f46e5",
    "#64748b",
]
PRIMARY_SERIES_COLORS = [
    "#0f766e",
    "#be123c",
    "#7c3aed",
    "#0891b2",
    "#ca8a04",
]
RESULTS_TSV_COLOR = "#2563eb"
BASELINE_REACT_COLOR = "#b45309"

FAILURE_STATUSES = {
    "crash",
    "failed",
    "failure",
    "error",
    "oom",
    "timeout",
    "cancelled",
    "canceled",
}

DEFAULT_COMPARISON_RUNS = [
    Path("TTS/ablation_runs/baseline_tool_call_real_train_b1_i100_e1_20260706_163546"),
]
DEFAULT_INITIAL_ANCHOR_TSV = Path("results.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot best validation metric by search iteration. "
            "Input can be a run directory, model_based_summary.json, baseline_summary.json, "
            "model_based.log, or baseline.log."
        ),
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more run directories, summary JSON files, or run logs.",
    )
    parser.add_argument(
        "--metric",
        default="val_bpb",
        help="Metric key to plot. Default: val_bpb.",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Treat higher metric values as better. Default is lower-is-better.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output SVG path or output prefix. Default: "
            "<run>/val_bpb_best_by_iteration.svg."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path. Default: same prefix as SVG with .csv.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help=(
            "Optional PNG path. Default: same prefix as SVG with .png when "
            "rsvg-convert is available."
        ),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help=(
            "Optional PDF path. Default: same prefix as SVG with .pdf when "
            "rsvg-convert is available."
        ),
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Do not attempt PNG conversion.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not attempt PDF conversion.",
    )
    parser.add_argument(
        "--comparison-tsv",
        "--results-tsv",
        dest="comparison_tsvs",
        action="append",
        type=Path,
        default=None,
        help=(
            "Optional comparison results.tsv file. Can be repeated. "
            "When omitted, no TSV comparison is added."
        ),
    )
    parser.add_argument(
        "--no-comparisons",
        action="store_true",
        help="Do not auto-load ./results.tsv or any comparison runs.",
    )
    parser.add_argument(
        "--comparison-run",
        dest="comparison_runs",
        action="append",
        type=Path,
        default=None,
        help=(
            "Optional comparison run directory, summary JSON, or log. Can be repeated. "
            "When omitted, known ablation comparison runs are used automatically if present."
        ),
    )
    parser.add_argument(
        "--no-default-comparison-runs",
        action="store_true",
        help="Do not auto-load built-in comparison run directories.",
    )
    parser.add_argument(
        "--comparison-max-iteration",
        type=int,
        default=149,
        help=(
            "Keep comparison TSV rows through this iteration/row number. "
            "Use 0 or a negative value to disable the cap. Default: 149, "
            "which means before iteration 150."
        ),
    )
    parser.add_argument(
        "--no-selected",
        action="store_true",
        help="Hide the per-iteration selected/evaluated metric line.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Hide the warmup-best reference line when warmup data exists.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1100,
        help="SVG width in pixels. Default: 1100.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=680,
        help="SVG height in pixels. Default: 680.",
    )
    parser.add_argument(
        "--xlim",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional x-axis iteration limits, e.g. --xlim 0 100.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional y-axis metric limits, e.g. --ylim 0.96 1.08.",
    )
    parser.add_argument(
        "--zoom-inset",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Add an inset zoom panel over this iteration window, e.g. "
            "--zoom-inset 40 100."
        ),
    )
    parser.add_argument(
        "--zoom-ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Optional y-axis limits for --zoom-inset. When omitted, limits are "
            "computed from best-so-far values in the zoom window."
        ),
    )
    parser.add_argument(
        "--trial-label",
        "--trial-name",
        dest="trial_labels",
        action="append",
        default=None,
        help=(
            "Legend/name override for a primary input trial. Repeat once per input path, "
            "in the same order as the positional paths."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [resolve_source(path) for path in args.paths]
    primary_series = [
        load_plot_data(source, metric=args.metric, minimize=not args.maximize)
        for source in sources
    ]
    primary_series = apply_trial_labels(primary_series, args.trial_labels)
    comparison_sources = resolve_comparison_sources(
        args.comparison_tsvs,
        sources=sources,
        metric=args.metric,
        disabled=args.no_comparisons,
    )
    comparison_run_sources = resolve_comparison_run_sources(
        args.comparison_runs,
        sources=sources,
        disabled=args.no_comparisons,
        disable_defaults=args.no_default_comparison_runs,
    )
    comparisons = [
        load_comparison_tsv(
            comparison_source,
            metric=args.metric,
            minimize=not args.maximize,
            max_iteration=args.comparison_max_iteration,
        )
        for comparison_source in comparison_sources
    ]
    comparisons.extend(
        load_plot_data(
            resolve_source(comparison_run_source),
            metric=args.metric,
            minimize=not args.maximize,
        )
        for comparison_run_source in comparison_run_sources
    )
    if args.no_warmup:
        primary_series = [
            PlotData(
                run_name=series.run_name,
                metric=series.metric,
                minimize=series.minimize,
                points=series.points,
                warmup_best=None,
                source=series.source,
                feature_expansions=series.feature_expansions,
            )
            for series in primary_series
        ]

    data = primary_series[0]
    all_series = [*primary_series, *comparisons]
    svg_path = resolve_svg_path(args.out, sources[0], data.metric, multi=len(primary_series) > 1)
    csv_path = args.csv or svg_path.with_suffix(".csv")
    png_path = args.png or svg_path.with_suffix(".png")
    pdf_path = args.pdf or svg_path.with_suffix(".pdf")

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_series, csv_path, metric=data.metric)
    write_svg(
        primary_series,
        svg_path,
        comparisons=comparisons,
        width=max(700, int(args.width)),
        height=max(460, int(args.height)),
        show_selected=not args.no_selected,
        xlim=None if args.xlim is None else (int(args.xlim[0]), int(args.xlim[1])),
        ylim=None if args.ylim is None else (float(args.ylim[0]), float(args.ylim[1])),
        zoom_inset=None if args.zoom_inset is None else (int(args.zoom_inset[0]), int(args.zoom_inset[1])),
        zoom_ylim=None if args.zoom_ylim is None else (float(args.zoom_ylim[0]), float(args.zoom_ylim[1])),
    )

    converted_png: Path | None = None
    if not args.no_png:
        converted_png = convert_svg_to_png(svg_path, png_path)
    converted_pdf: Path | None = None
    if not args.no_pdf:
        converted_pdf = convert_svg_to_pdf(svg_path, pdf_path)

    result = {
        "source": str(sources[0]),
        "sources": [str(source) for source in sources],
        "svg": str(svg_path),
        "csv": str(csv_path),
        "png": None if converted_png is None else str(converted_png),
        "pdf": None if converted_pdf is None else str(converted_pdf),
        "metric": data.metric,
        "mode": "minimize" if data.minimize else "maximize",
        "xlim": args.xlim,
        "ylim": args.ylim,
        "zoom_inset": args.zoom_inset,
        "zoom_ylim": args.zoom_ylim,
        "trials": [
            {
                "name": series.run_name,
                "source": str(series.source),
                "points": [
                    {
                        "iteration": point.iteration,
                        "iteration_score": point.iteration_score,
                        "best_after": point.best_after,
                        "state_id": point.state_id,
                        "status": point.status,
                        "description": point.description,
                    }
                    for point in series.points
                ],
            }
            for series in primary_series
        ],
        "comparisons": [
            {
                "name": comparison.run_name,
                "source": str(comparison.source),
                "points": [
                    {
                        "iteration": point.iteration,
                        "iteration_score": point.iteration_score,
                        "best_after": point.best_after,
                        "state_id": point.state_id,
                        "status": point.status,
                        "description": point.description,
                    }
                    for point in comparison.points
                ],
            }
            for comparison in comparisons
        ],
        "iterations": [
            {
                "iteration": point.iteration,
                "iteration_score": point.iteration_score,
                "best_after": point.best_after,
                "state_id": point.state_id,
            }
            for point in data.points
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


def resolve_source(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        for name in ("model_based_summary.json", "baseline_summary.json", "summary.json"):
            summary_path = path / name
            if summary_path.exists():
                return summary_path
        for name in ("model_based.log", "baseline.log"):
            log_path = path / name
            if log_path.exists():
                return log_path
        raise SystemExit(
            "No supported summary/log found under "
            f"{path}. Expected model_based_summary.json, baseline_summary.json, summary.json, "
            "model_based.log, or baseline.log."
        )
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")
    return path


def apply_trial_labels(series_list: list[PlotData], labels: list[str] | None) -> list[PlotData]:
    if not labels:
        return series_list
    if len(labels) != len(series_list):
        raise SystemExit(
            f"Expected {len(series_list)} --trial-label value(s), got {len(labels)}."
        )
    return [
        replace(series, run_name=label.strip() or series.run_name)
        for series, label in zip(series_list, labels)
    ]


def resolve_comparison_sources(
    explicit_paths: list[Path] | None,
    *,
    sources: list[Path],
    metric: str,
    disabled: bool,
) -> list[Path]:
    if disabled:
        return []

    explicit = explicit_paths or []
    if explicit:
        return dedupe_paths(resolve_tsv_path(path) for path in explicit)
    return []


def resolve_comparison_run_sources(
    explicit_paths: list[Path] | None,
    *,
    sources: list[Path],
    disabled: bool,
    disable_defaults: bool,
) -> list[Path]:
    if disabled:
        return []

    if explicit_paths:
        candidates = explicit_paths
    elif disable_defaults:
        return []
    else:
        candidates = DEFAULT_COMPARISON_RUNS

    resolved: list[Path] = []
    source_keys = resolved_path_keys(sources)
    for candidate in dedupe_paths(candidates):
        candidate = candidate.expanduser()
        if not candidate.exists():
            continue
        try:
            candidate_source = resolve_source(candidate)
        except SystemExit:
            continue
        candidate_key = (
            str(candidate_source.resolve()) if candidate_source.exists() else str(candidate_source)
        )
        if candidate_key in source_keys:
            continue
        resolved.append(candidate)
    return resolved


def resolve_tsv_path(path: Path) -> Path:
    resolved = path.expanduser()
    if resolved.is_dir():
        resolved = resolved / "results.tsv"
    if not resolved.exists():
        raise SystemExit(f"Comparison TSV does not exist: {resolved}")
    return resolved


def dedupe_paths(paths: Any) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def path_key(path: Path) -> str:
    return str(path.resolve()) if path.exists() else str(path)


def resolved_path_keys(paths: list[Path]) -> set[str]:
    return {path_key(path) for path in paths}


def tsv_has_metric(path: Path, metric: str) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames or []
    except OSError:
        return False
    return metric_column(fieldnames, metric) is not None


def resolve_svg_path(out_arg: Path | None, source: Path, metric: str, *, multi: bool = False) -> Path:
    default_name = f"{metric}_{'multi_trial_' if multi else ''}best_by_iteration.svg"
    if out_arg is None:
        return source.parent / default_name
    out_path = out_arg.expanduser()
    if out_path.suffix.lower() == ".svg":
        return out_path
    if out_path.exists() and out_path.is_dir():
        return out_path / default_name
    if out_path.suffix:
        return out_path.with_suffix(".svg")
    return out_path.with_suffix(".svg")


def load_plot_data(source: Path, metric: str, minimize: bool) -> PlotData:
    if source.suffix.lower() == ".json":
        return load_summary_data(source, metric=metric, minimize=minimize)
    return load_log_data(source, metric=metric, minimize=minimize)


def load_comparison_tsv(
    path: Path,
    *,
    metric: str,
    minimize: bool,
    max_iteration: int,
) -> PlotData:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames or []
            metric_col = metric_column(fieldnames, metric)
            if metric_col is None:
                available = ", ".join(fieldnames)
                raise SystemExit(
                    f"Comparison TSV {path} does not contain metric column {metric!r}. "
                    f"Available columns: {available}"
                )
            rows = list(reader)
    except OSError as exc:
        raise SystemExit(f"Could not read comparison TSV {path}: {exc}") from exc

    points: list[IterationPoint] = []
    running_best: float | None = None
    for row_number, row in enumerate(rows, start=1):
        iteration = row_iteration(row, default=row_number)
        if max_iteration > 0 and iteration > max_iteration:
            continue

        state_id = first_nonempty(row, ("commit", "state_id", "run_id", "id"))
        status = first_nonempty(row, ("status",))
        description = first_nonempty(row, ("description", "notes", "comment"))
        score = as_float(row.get(metric_col))
        score_is_usable = score is not None and usable_score(score)
        if is_failure_status(status) or not score_is_usable:
            if running_best is not None:
                points.append(
                    IterationPoint(
                        iteration=iteration,
                        iteration_score=None,
                        best_after=running_best,
                        state_id=state_id,
                        status=status,
                        description=description,
                    )
                )
            continue

        running_best = better_score(running_best, score, minimize=minimize)
        points.append(
            IterationPoint(
                iteration=iteration,
                iteration_score=score,
                best_after=running_best,
                state_id=state_id,
                status=status,
                description=description,
            )
        )

    if not points:
        raise SystemExit(f"No usable {metric} points found in comparison TSV {path}.")

    return PlotData(
        run_name=comparison_name(path, max_iteration=max_iteration),
        metric=metric,
        minimize=minimize,
        points=points,
        warmup_best=None,
        source=path,
        feature_expansions=[],
    )


def metric_column(fieldnames: list[str], metric: str) -> str | None:
    aliases = metric_aliases(metric)
    for alias in aliases:
        if alias in fieldnames:
            return alias
    return None


def row_iteration(row: dict[str, str], *, default: int) -> int:
    for name in ("iteration", "iter", "row", "step"):
        value = as_int(row.get(name))
        if value is not None:
            return value
    return default


def first_nonempty(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def is_failure_status(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized in FAILURE_STATUSES


def comparison_name(path: Path, *, max_iteration: int) -> str:
    cwd = Path.cwd().resolve()
    resolved = path.resolve()
    if resolved.parent == cwd:
        label = path.name
    elif path.name == "results.tsv":
        label = path.parent.name
    else:
        label = path.name
    if max_iteration > 0:
        label = f"{label} <= {max_iteration}"
    return label


def load_summary_data(path: Path, metric: str, minimize: bool) -> PlotData:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise SystemExit(f"Expected a JSON object in {path}.")

    iterations = summary.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise SystemExit(f"No iterations found in {path}.")

    points: list[IterationPoint] = []
    running_best: float | None = None
    for item in iterations:
        if not isinstance(item, dict):
            continue
        iteration = as_int(item.get("iteration"))
        state_id = str(
            item.get("selected_state_id")
            or first_string(item.get("actual_state_ids"))
            or ""
        )
        iteration_score = iteration_metric_from_summary_item(
            item,
            metric=metric,
            run_dir=path.parent,
            state_id=state_id,
        )
        best_after = None
        if item.get("score_key") in (None, metric):
            best_after = as_float(item.get("best_score_after_iteration"))
            if best_after is not None and not usable_score(best_after):
                best_after = None
        if iteration is None:
            continue

        if iteration_score is None or not usable_score(iteration_score):
            if best_after is None or not usable_score(best_after):
                continue
            running_best = best_after
            iteration_score = None
        elif best_after is None or not usable_score(best_after):
            running_best = better_score(
                running_best,
                iteration_score,
                minimize=minimize,
            )
            best_after = running_best
        else:
            running_best = best_after

        points.append(
            IterationPoint(
                iteration=iteration,
                iteration_score=iteration_score,
                best_after=best_after,
                state_id=state_id,
            )
        )

    points = sorted(points, key=lambda point: point.iteration)
    if not points:
        raise_no_metric_error(path.parent, metric, source=path)

    warmup_best = summary_warmup_best(summary, metric=metric, minimize=minimize, run_dir=path.parent)
    return PlotData(
        run_name=path.parent.name,
        metric=metric,
        minimize=minimize,
        points=points,
        warmup_best=warmup_best,
        source=path,
        feature_expansions=load_feature_expansions(path.parent),
    )


def load_log_data(path: Path, metric: str, minimize: bool) -> PlotData:
    result_re = re.compile(
        rf"\biteration=(?P<iteration>\d+)\b.*?\bresult\b.*?"
        rf"(?:\bselected=(?P<state>\S+)\b.*?)?"
        rf"\b{re.escape(metric)}=(?P<score>[-+0-9.eE]+)\b.*?"
        rf"(?:\bbest_after=(?P<best>[-+0-9.eE]+)\b)?"
    )
    eval_re = re.compile(
        rf"\biteration=(?P<iteration>\d+)\b.*?\bevaluated_selected=(?P<state>\S+)\b.*?"
        rf"\b{re.escape(metric)}=(?P<score>[-+0-9.eE]+)\b"
    )
    selected_re = re.compile(
        rf"\biteration=(?P<iteration>\d+)\b.*?\bselected=(?P<state>\S+)\b.*?"
        rf"\b{re.escape(metric)}=(?P<score>[-+0-9.eE]+)\b"
    )
    warmup_re = re.compile(
        rf"\bwarmup(?:_root)?\b.*?\b{re.escape(metric)}=(?P<score>[-+0-9.eE]+)\b"
    )
    generic_result_re = re.compile(
        r"\biteration=(?P<iteration>\d+)\b.*?\bresult\b.*?\bselected=(?P<state>\S+)\b"
    )
    generic_eval_re = re.compile(
        r"\biteration=(?P<iteration>\d+)\b.*?\bevaluated_selected=(?P<state>\S+)\b"
    )
    generic_warmup_re = re.compile(r"\bwarmup(?:_root)?\b.*?\bstate=(?P<state>\S+)\b")

    by_iteration: dict[int, tuple[float, float | None, str]] = {}
    warmup_scores: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"Could not read log {path}: {exc}") from exc

    for line in lines:
        warmup_match = warmup_re.search(line)
        if warmup_match:
            score = as_float(warmup_match.group("score"))
            if score is not None and usable_score(score):
                warmup_scores.append(score)
        else:
            generic_warmup_match = generic_warmup_re.search(line)
            if generic_warmup_match:
                score = metric_from_state(path.parent, generic_warmup_match.group("state"), metric)
                if score is not None and usable_score(score):
                    warmup_scores.append(score)

        result_match = result_re.search(line)
        if result_match:
            iteration = int(result_match.group("iteration"))
            score = as_float(result_match.group("score"))
            best_after = as_float(result_match.group("best"))
            state_id = result_match.group("state") or ""
            if (score is None or not usable_score(score)) and state_id:
                score = metric_from_state(path.parent, state_id, metric)
                best_after = None
            if score is not None and usable_score(score):
                by_iteration[iteration] = (score, best_after, state_id)
            continue

        eval_match = eval_re.search(line)
        if eval_match:
            iteration = int(eval_match.group("iteration"))
            score = as_float(eval_match.group("score"))
            state_id = eval_match.group("state") or ""
            if (score is None or not usable_score(score)) and state_id:
                score = metric_from_state(path.parent, state_id, metric)
            if score is not None and usable_score(score) and iteration not in by_iteration:
                by_iteration[iteration] = (score, None, state_id)
            continue

        selected_match = selected_re.search(line)
        if selected_match:
            iteration = int(selected_match.group("iteration"))
            score = as_float(selected_match.group("score"))
            state_id = selected_match.group("state") or ""
            if (score is None or not usable_score(score)) and state_id:
                score = metric_from_state(path.parent, state_id, metric)
            if score is not None and usable_score(score) and iteration not in by_iteration:
                by_iteration[iteration] = (score, None, state_id)
            continue

        generic_result_match = generic_result_re.search(line)
        if generic_result_match:
            iteration = int(generic_result_match.group("iteration"))
            state_id = generic_result_match.group("state") or ""
            score = metric_from_state(path.parent, state_id, metric)
            if score is not None and usable_score(score):
                by_iteration[iteration] = (score, None, state_id)
            continue

        generic_eval_match = generic_eval_re.search(line)
        if generic_eval_match:
            iteration = int(generic_eval_match.group("iteration"))
            state_id = generic_eval_match.group("state") or ""
            score = metric_from_state(path.parent, state_id, metric)
            if score is not None and usable_score(score) and iteration not in by_iteration:
                by_iteration[iteration] = (score, None, state_id)

    if not by_iteration:
        raise_no_metric_error(path.parent, metric, source=path)

    points: list[IterationPoint] = []
    running_best: float | None = None
    for iteration in sorted(by_iteration):
        iteration_score, best_after, state_id = by_iteration[iteration]
        if best_after is None or not usable_score(best_after):
            running_best = better_score(running_best, iteration_score, minimize=minimize)
            best_after = running_best
        else:
            running_best = best_after
        points.append(
            IterationPoint(
                iteration=iteration,
                iteration_score=iteration_score,
                best_after=best_after,
                state_id=state_id,
            )
        )

    warmup_best = best_of(warmup_scores, minimize=minimize)
    return PlotData(
        run_name=path.parent.name,
        metric=metric,
        minimize=minimize,
        points=points,
        warmup_best=warmup_best,
        source=path,
        feature_expansions=load_feature_expansions(path.parent),
    )


def iteration_metric_from_summary_item(
    item: dict[str, Any],
    *,
    metric: str,
    run_dir: Path,
    state_id: str,
) -> float | None:
    if item.get("score_key") in (None, metric):
        value = as_float(item.get("iteration_best_score", item.get("selected_real_score")))
        if value is not None and usable_score(value):
            return value
    if state_id:
        return metric_from_state(run_dir, state_id, metric)
    return None


def summary_warmup_best(
    summary: dict[str, Any],
    metric: str,
    minimize: bool,
    run_dir: Path,
) -> float | None:
    warmup = summary.get("warmup")
    if not isinstance(warmup, dict):
        return None
    scores: list[float] = []
    if warmup.get("score_key") in (None, metric):
        raw_scores = warmup.get("scores")
        if isinstance(raw_scores, list):
            scores.extend(
                score
                for score in (as_float(item) for item in raw_scores)
                if score is not None and usable_score(score)
            )
    if not scores:
        state_ids = warmup.get("state_ids")
        if isinstance(state_ids, list):
            for state_id in state_ids:
                if not isinstance(state_id, str):
                    continue
                score = metric_from_state(run_dir, state_id, metric)
                if score is not None and usable_score(score):
                    scores.append(score)
    return best_of(scores, minimize=minimize)


def metric_from_state(run_dir: Path, state_id: str, metric: str) -> float | None:
    state_dir = run_dir / "states" / state_id
    meta = load_json(state_dir / "meta.json")
    if meta is not None:
        value = metric_from_meta(meta, metric)
        if value is not None and usable_score(value):
            return value
    return metric_from_stdout(state_dir / "stdout.log", metric)


def load_feature_expansions(run_dir: Path) -> list[FeatureExpansion]:
    path = run_dir / "operation_feature_expansions.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []

    expansions: list[FeatureExpansion] = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        iteration = as_int(item.get("iteration"))
        if iteration is None:
            continue
        expansions.append(
            FeatureExpansion(
                iteration=iteration,
                active_feature_count=as_int(item.get("active_feature_count")),
                name=str(item.get("name") or ""),
                state_id=str(item.get("state_id") or ""),
            )
        )
    return sorted(expansions, key=lambda expansion: (expansion.iteration, expansion.active_feature_count or 0))


def metric_from_meta(meta: dict[str, Any], metric: str) -> float | None:
    aliases = metric_aliases(metric)
    metrics = meta.get("metrics")
    if isinstance(metrics, dict):
        for name in aliases:
            value = as_float(metrics.get(name))
            if value is not None:
                return value
    for name in aliases:
        value = as_float(meta.get(name))
        if value is not None:
            return value
    return None


def metric_from_stdout(path: Path, metric: str) -> float | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    aliases = metric_aliases(metric)
    if metric in {"train_loss", "loss", "smooth_train_loss"}:
        losses = extract_colon_metric_values(text, "loss")
        return losses[-1] if losses else None

    for name in aliases:
        values = extract_colon_metric_values(text, name)
        if values:
            return values[-1]
    return None


def extract_colon_metric_values(text: str, metric: str) -> list[float]:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(metric)}\s*:\s*([-+0-9.eE]+)"
    )
    values: list[float] = []
    for match in pattern.finditer(text):
        value = as_float(match.group(1))
        if value is not None and usable_score(value):
            values.append(value)
    return values


def metric_aliases(metric: str) -> list[str]:
    aliases = [metric]
    alias_map = {
        "train_loss": ["loss", "smooth_train_loss", "debiased_smooth_loss"],
        "loss": ["train_loss", "smooth_train_loss", "debiased_smooth_loss"],
    }
    aliases.extend(alias_map.get(metric, []))
    deduped: list[str] = []
    for alias in aliases:
        if alias not in deduped:
            deduped.append(alias)
    return deduped


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def raise_no_metric_error(run_dir: Path, metric: str, *, source: Path) -> None:
    available = available_metric_names(run_dir)
    suffix = ""
    if available:
        suffix = "\nAvailable saved metrics include: " + ", ".join(available[:40])
        if len(available) > 40:
            suffix += f", ... ({len(available)} total)"
        if metric == "train_loss":
            suffix += "\nFor train_loss, the script also checks the final printed 'loss:' in each state's stdout.log."
    raise SystemExit(f"No usable {metric} iteration points found in {source}.{suffix}")


def available_metric_names(run_dir: Path) -> list[str]:
    names: set[str] = set()
    for meta_path in sorted((run_dir / "states").glob("*/meta.json")):
        meta = load_json(meta_path)
        if not meta:
            continue
        metrics = meta.get("metrics")
        if isinstance(metrics, dict):
            names.update(key for key, value in metrics.items() if as_float(value) is not None)
        names.update(key for key, value in meta.items() if as_float(value) is not None)
    return sorted(names)


def write_csv(series_list: list[PlotData], path: Path, *, metric: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "series",
                "iteration",
                f"iteration_{metric}",
                f"best_{metric}_after_iteration",
                "state_id",
                "status",
                "description",
                "source",
            ],
        )
        writer.writeheader()
        for series in series_list:
            for point in series.points:
                writer.writerow(
                    {
                        "series": series.run_name,
                        "iteration": point.iteration,
                        f"iteration_{metric}": "" if point.iteration_score is None else format_float(point.iteration_score),
                        f"best_{metric}_after_iteration": format_float(point.best_after),
                        "state_id": point.state_id,
                        "status": point.status,
                        "description": point.description,
                        "source": str(series.source),
                    }
                )


def write_svg(
    primary_series: list[PlotData],
    path: Path,
    *,
    comparisons: list[PlotData],
    width: int,
    height: int,
    show_selected: bool,
    xlim: tuple[int, int] | None,
    ylim: tuple[float, float] | None,
    zoom_inset: tuple[int, int] | None,
    zoom_ylim: tuple[float, float] | None,
) -> None:
    margin_left = 96
    margin_right = 48
    margin_top = 62
    margin_bottom = 92
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    plot_y1 = plot_y0 + plot_h

    if not primary_series:
        raise SystemExit("No primary series to plot.")
    data = primary_series[0]
    primary_count = len(primary_series)
    series_list = [*primary_series, *comparisons]
    shared_initial = shared_initial_point(metric=data.metric, comparisons=comparisons)
    display_points = {
        series_index: [
            (
                display_iteration(
                    point,
                    series=series,
                    series_index=series_index,
                    primary_count=primary_count,
                    shared_initial=shared_initial,
                ),
                point.best_after,
            )
            for point in series.points
        ]
        for series_index, series in enumerate(series_list)
    }
    x_values = [
        x_value
        for points in display_points.values()
        for x_value, _ in points
    ]
    if shared_initial is not None:
        x_values.extend(
            shared_initial[0]
            for index, series in enumerate(series_list)
            if needs_shared_initial(series, series_index=index, primary_count=primary_count)
        )
    if xlim is not None:
        x_min, x_max = sorted((int(xlim[0]), int(xlim[1])))
    elif x_values:
        x_min, x_max = min(x_values), max(x_values)
    else:
        x_min, x_max = 0, 1
    zoom_range = normalized_zoom_range(zoom_inset, x_min=x_min, x_max=x_max)
    y_values = [point.best_after for series in series_list for point in series.points]
    if shared_initial is not None:
        y_values.extend(
            shared_initial[1]
            for index, series in enumerate(series_list)
            if needs_shared_initial(series, series_index=index, primary_count=primary_count)
        )
    if show_selected:
        y_values.extend(
            point.iteration_score
            for series in primary_series
            for point in series.points
            if point.iteration_score is not None
        )
    y_values.extend(series.warmup_best for series in primary_series if series.warmup_best is not None)

    if ylim is not None:
        y_min, y_max = sorted((float(ylim[0]), float(ylim[1])))
        if math.isclose(y_min, y_max):
            y_min, y_max = padded_range([y_min])
    else:
        y_min, y_max = padded_range(y_values)
    y_ticks = make_y_ticks(y_min, y_max)

    def sx(x_value: int) -> float:
        if x_min == x_max:
            return plot_x0 + plot_w / 2.0
        return plot_x0 + (x_value - x_min) / (x_max - x_min) * plot_w

    def sy(y_value: float) -> float:
        return plot_y1 - (y_value - y_min) / (y_max - y_min) * plot_h

    def polyline(points: list[tuple[int, float]]) -> str:
        return " ".join(
            f"{sx(x):.2f},{sy(y):.2f}"
            for x, y in points
            if x_min <= x <= x_max and y_min <= y <= y_max
        )

    lines: list[str] = []
    append = lines.append
    append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
    )
    append(f"<title id=\"title\">Best {escape(data.metric)} by iteration</title>")
    append(
        "<desc id=\"desc\">Line chart showing the running best metric by "
        "model-based iteration.</desc>"
    )
    append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    append(
        svg_text(
            52,
            42,
            f"Best {data.metric} by LDM-TTS (Qwen3-Coder-30B-A3B) iteration",
            size=25,
            weight="700",
            fill="#111827",
        )
    )
    append(
        f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" '
        'fill="#ffffff" stroke="#d6d3ca" stroke-width="1" rx="6"/>'
    )
    if zoom_range is not None:
        zoom_x0, zoom_x1 = zoom_range
        append(
            f'<rect x="{sx(zoom_x0):.2f}" y="{plot_y0}" '
            f'width="{max(1.0, sx(zoom_x1) - sx(zoom_x0)):.2f}" height="{plot_h}" '
            'fill="#f8fafc" stroke="#94a3b8" stroke-width="1" '
            'stroke-dasharray="5 5" opacity="0.38"/>'
        )

    for tick in y_ticks:
        y = sy(tick)
        append(
            f'<line x1="{plot_x0}" y1="{y:.2f}" x2="{plot_x0 + plot_w}" '
            f'y2="{y:.2f}" stroke="#ece8df" stroke-width="1"/>'
        )
        append(
            svg_text(
                plot_x0 - 12,
                y + 5,
                format_axis_float(tick),
                size=13,
                fill="#6b7280",
                anchor="end",
            )
        )

    for x_value in make_x_ticks(x_min, x_max):
        x = sx(x_value)
        append(
            f'<line x1="{x:.2f}" y1="{plot_y0}" x2="{x:.2f}" y2="{plot_y1}" '
            'stroke="#f3f0e8" stroke-width="1"/>'
        )
        append(svg_text(x, plot_y1 + 30, str(x_value), size=14, fill="#374151", anchor="middle"))

    if len(primary_series) == 1:
        for series_index, series in enumerate(primary_series):
            if series.warmup_best is None:
                continue
            y = sy(series.warmup_best)
            color = series_color(series, series_index=series_index, primary_count=primary_count)
            append(
                f'<line x1="{plot_x0}" y1="{y:.2f}" x2="{plot_x0 + plot_w}" y2="{y:.2f}" '
                f'stroke="{color}" stroke-width="1.8" stroke-dasharray="7 7" opacity="0.55"/>'
            )
            label = f"warmup best {series.warmup_best:.6f}"
            label_width = max(150, min(260, len(label) * 7 + 26))
            label_x = plot_x0 + plot_w - label_width - 10
            append(
                f'<rect x="{label_x:.2f}" y="{y - 25:.2f}" width="{label_width}" height="22" '
                'fill="#fbfbf8" stroke="#d6d3ca" rx="4"/>'
            )
            append(
                svg_text(
                    label_x + label_width / 2.0,
                    y - 9,
                    label,
                    size=12,
                    fill="#6b4f2a",
                    anchor="middle",
                )
            )

    for index, series in enumerate(series_list):
        color = series_color(series, series_index=index, primary_count=primary_count)
        stroke_width = 4 if index < primary_count else 3
        opacity = "1" if index < primary_count else "0.92"
        dasharray = series_dasharray(series)
        dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        best_points: list[tuple[int, float]] = []
        if shared_initial is not None and needs_shared_initial(series, series_index=index, primary_count=primary_count):
            best_points.append(shared_initial)
        best_points.extend(display_points[index])
        append(
            f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash_attr} '
            f'points="{polyline(best_points)}"/>'
        )

    # append(
    #     render_feature_expansion_annotations(
    #         primary_series=primary_series,
    #         primary_count=primary_count,
    #         shared_initial=shared_initial,
    #         x_min=x_min,
    #         x_max=x_max,
    #         plot_y0=plot_y0,
    #         plot_y1=plot_y1,
    #         sx=sx,
    #     )
    # )

    if show_selected:
        for series_index, series in enumerate(primary_series):
            fill = series_color(series, series_index=series_index, primary_count=primary_count)
            for point in series.points:
                if point.iteration_score is None:
                    continue
                display_x = display_iteration(
                    point,
                    series=series,
                    series_index=series_index,
                    primary_count=primary_count,
                    shared_initial=shared_initial,
                )
                if not (x_min <= display_x <= x_max):
                    continue
                if not (y_min <= point.iteration_score <= y_max):
                    continue
                x = sx(display_x)
                y_selected = sy(point.iteration_score)
                append(
                    f'<circle cx="{x:.2f}" cy="{y_selected:.2f}" r="3.2" fill="{fill}" '
                    'stroke="#ffffff" stroke-width="1" opacity="0.5"/>'
                )

    if zoom_range is not None:
        append(
            render_zoom_inset(
                series_list=series_list,
                display_points=display_points,
                primary_count=primary_count,
                zoom_range=zoom_range,
                zoom_ylim=zoom_ylim,
                main_x_min=x_min,
                main_x_max=x_max,
                width=width,
                plot_x0=plot_x0,
                plot_y0=plot_y0,
                plot_w=plot_w,
                plot_h=plot_h,
            )
        )

    append(
        f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x0 + plot_w}" y2="{plot_y1}" '
        'stroke="#4b5563" stroke-width="1.4"/>'
    )
    append(
        f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" '
        'stroke="#4b5563" stroke-width="1.4"/>'
    )
    append(
        svg_text(
            plot_x0 + plot_w / 2.0,
            height - 24,
            "Iteration",
            size=15,
            weight="600",
            fill="#111827",
            anchor="middle",
        )
    )
    append(
        f'<g transform="translate(28,{plot_y0 + plot_h / 2.0:.2f}) rotate(-90)">'
        f'{svg_text(0, 0, data.metric, size=15, weight="600", fill="#111827", anchor="middle")}'
        "</g>"
    )
    append(render_legend(series_list=series_list, width=width, show_selected=show_selected, primary_count=primary_count))
    append("</svg>")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalized_zoom_range(
    zoom_inset: tuple[int, int] | None,
    *,
    x_min: int,
    x_max: int,
) -> tuple[int, int] | None:
    if zoom_inset is None:
        return None
    zoom_x0, zoom_x1 = sorted((int(zoom_inset[0]), int(zoom_inset[1])))
    zoom_x0 = max(x_min, zoom_x0)
    zoom_x1 = min(x_max, zoom_x1)
    if zoom_x0 >= zoom_x1:
        return None
    return zoom_x0, zoom_x1


def render_zoom_inset(
    *,
    series_list: list[PlotData],
    display_points: dict[int, list[tuple[int, float]]],
    primary_count: int,
    zoom_range: tuple[int, int],
    zoom_ylim: tuple[float, float] | None,
    main_x_min: int,
    main_x_max: int,
    width: int,
    plot_x0: float,
    plot_y0: float,
    plot_w: float,
    plot_h: float,
) -> str:
    zoom_x0, zoom_x1 = zoom_range
    inset_w = min(410.0, max(320.0, plot_w * 0.42))
    inset_h = min(245.0, max(190.0, plot_h * 0.45))
    inset_x = plot_x0 + plot_w - inset_w - 24
    inset_y = plot_y0 + plot_h - inset_h - 28
    if width < 860:
        inset_w = max(280.0, plot_w * 0.56)
        inset_h = max(170.0, plot_h * 0.42)
        inset_x = plot_x0 + plot_w - inset_w - 14
        inset_y = plot_y0 + plot_h - inset_h - 18
    inset_x = max(plot_x0 + 10, inset_x)
    inset_y = max(plot_y0 + 10, inset_y)

    inset_margin_left = 58.0
    inset_margin_right = 14.0
    inset_margin_top = 34.0
    inset_margin_bottom = 38.0
    ix0 = inset_x + inset_margin_left
    iy0 = inset_y + inset_margin_top
    iw = inset_w - inset_margin_left - inset_margin_right
    ih = inset_h - inset_margin_top - inset_margin_bottom
    iy1 = iy0 + ih

    zoom_values = [
        y_value
        for index, points in display_points.items()
        if not hide_from_zoom_inset(series_list[index])
        for x_value, y_value in points
        if zoom_x0 <= x_value <= zoom_x1
    ]
    if not zoom_values:
        return ""
    if zoom_ylim is not None:
        y_min, y_max = sorted((float(zoom_ylim[0]), float(zoom_ylim[1])))
        if math.isclose(y_min, y_max):
            y_min, y_max = padded_range([y_min])
    else:
        y_min, y_max = padded_range(zoom_values, pad_fraction=0.18, min_pad=0.0002)
    y_ticks = make_y_ticks(y_min, y_max)

    def zx(x_value: int) -> float:
        if zoom_x0 == zoom_x1:
            return ix0 + iw / 2.0
        return ix0 + (x_value - zoom_x0) / (zoom_x1 - zoom_x0) * iw

    def zy(y_value: float) -> float:
        return iy1 - (y_value - y_min) / (y_max - y_min) * ih

    def zoom_polyline(points: list[tuple[int, float]]) -> str:
        return " ".join(
            f"{zx(x):.2f},{zy(y):.2f}"
            for x, y in points
            if zoom_x0 <= x <= zoom_x1 and y_min <= y <= y_max
        )

    lines: list[str] = [
        f'<rect x="{inset_x:.2f}" y="{inset_y:.2f}" width="{inset_w:.2f}" height="{inset_h:.2f}" '
        'fill="#ffffff" stroke="#94a3b8" stroke-width="1.2" rx="6"/>',
        svg_text(
            inset_x + 14,
            inset_y + 22,
            f"Converged region ({zoom_x0}-{zoom_x1})",
            size=12,
            weight="700",
            fill="#111827",
        ),
    ]

    for tick in y_ticks[:: max(1, math.ceil(len(y_ticks) / 4))]:
        if not (y_min <= tick <= y_max):
            continue
        y = zy(tick)
        lines.append(
            f'<line x1="{ix0:.2f}" y1="{y:.2f}" x2="{ix0 + iw:.2f}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            svg_text(
                ix0 - 8,
                y + 4,
                format_axis_float(tick),
                size=10,
                fill="#64748b",
                anchor="end",
            )
        )

    for tick in make_x_ticks(zoom_x0, zoom_x1, max_ticks=4):
        x = zx(tick)
        lines.append(
            f'<line x1="{x:.2f}" y1="{iy0:.2f}" x2="{x:.2f}" y2="{iy1:.2f}" '
            'stroke="#f1f5f9" stroke-width="1"/>'
        )
        lines.append(svg_text(x, iy1 + 18, str(tick), size=10, fill="#64748b", anchor="middle"))

    label_items: list[tuple[float, str, str, str]] = []
    for index, series in enumerate(series_list):
        if hide_from_zoom_inset(series):
            continue
        points = display_points.get(index, [])
        visible_points = [
            (x_value, y_value)
            for x_value, y_value in points
            if zoom_x0 <= x_value <= zoom_x1 and y_min <= y_value <= y_max
        ]
        polyline_points = zoom_polyline(visible_points)
        if not polyline_points:
            continue
        color = series_color(series, series_index=index, primary_count=primary_count)
        stroke_width = 2.7 if index < primary_count else 2.2
        dasharray = series_dasharray(series)
        dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} points="{polyline_points}"/>'
        )
        if visible_points:
            label_items.append(
                (
                    zy(visible_points[-1][1]),
                    zoom_inset_label(series, index=index, primary_count=primary_count),
                    color,
                    dasharray,
                )
            )

    lines.extend(
        render_zoom_inset_labels(
            label_items=label_items,
            ix0=ix0,
            iy0=iy0,
            iw=iw,
            iy1=iy1,
        )
    )

    lines.extend(
        [
            f'<line x1="{ix0:.2f}" y1="{iy1:.2f}" x2="{ix0 + iw:.2f}" y2="{iy1:.2f}" '
            'stroke="#334155" stroke-width="1"/>',
            f'<line x1="{ix0:.2f}" y1="{iy0:.2f}" x2="{ix0:.2f}" y2="{iy1:.2f}" '
            'stroke="#334155" stroke-width="1"/>',
        ]
    )
    return "\n".join(lines)


def hide_from_zoom_inset(series: PlotData) -> bool:
    return series.source.name == "results.tsv" or is_qwen_react_baseline(series)


def zoom_inset_label(series: PlotData, *, index: int, primary_count: int) -> str:
    label = legend_label(series, index=index, primary_count=primary_count)
    replacements = (
        ("LDM-TTS BoN ", ""),
        ("LDM-TTS with ", ""),
        ("Acquisition: ", ""),
        (" (Expanded features)", " expanded"),
        (" (Fixed features)", " fixed"),
        (" (Qwen3-Coder-30B-A3B)", ""),
    )
    for old, new in replacements:
        label = label.replace(old, new)
    label = " ".join(label.split())
    if len(label) <= 24:
        return label
    return label[:21].rstrip() + "..."


def render_zoom_inset_labels(
    *,
    label_items: list[tuple[float, str, str, str]],
    ix0: float,
    iy0: float,
    iw: float,
    iy1: float,
) -> list[str]:
    if not label_items:
        return []
    label_h = 15.0
    gap = 16.0
    label_y_min = iy0 + label_h
    label_y_max = iy1 - 5
    sorted_items = sorted(enumerate(label_items), key=lambda item: item[1][0])
    adjusted: list[list[float | int]] = []
    last_y = label_y_min - gap
    for original_index, (target_y, *_rest) in sorted_items:
        label_y = max(target_y, label_y_min, last_y + gap)
        adjusted.append([original_index, label_y])
        last_y = label_y
    if adjusted and adjusted[-1][1] > label_y_max:
        shift = float(adjusted[-1][1]) - label_y_max
        for item in adjusted:
            item[1] = float(item[1]) - shift
    if adjusted and adjusted[0][1] < label_y_min:
        shift = label_y_min - float(adjusted[0][1])
        for item in adjusted:
            item[1] = float(item[1]) + shift

    y_by_index = {int(original_index): float(label_y) for original_index, label_y in adjusted}
    label_x = ix0 + iw - 6
    lines: list[str] = []
    for original_index, (_target_y, label, color, dasharray) in enumerate(label_items):
        label_y = y_by_index[original_index]
        label_w = max(80.0, min(138.0, len(label) * 5.8 + 34.0))
        label_left = max(ix0 + 4.0, label_x - label_w)
        swatch_x0 = label_left + 7.0
        swatch_x1 = swatch_x0 + 18.0
        dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        lines.extend(
            [
                f'<rect x="{label_left:.2f}" y="{label_y - label_h + 2:.2f}" '
                f'width="{label_w:.2f}" height="{label_h:.2f}" fill="#ffffff" '
                'stroke="#e5e7eb" stroke-width="0.8" rx="4" opacity="0.94"/>',
                f'<line x1="{swatch_x0:.2f}" y1="{label_y - 5:.2f}" '
                f'x2="{swatch_x1:.2f}" y2="{label_y - 5:.2f}" stroke="{color}" '
                f'stroke-width="2.4" stroke-linecap="round"{dash_attr}/>',
                svg_text(
                    swatch_x1 + 5.0,
                    label_y - 1.0,
                    label,
                    size=9,
                    weight="700",
                    fill="#111827",
                ),
            ]
        )
    return lines


def render_feature_expansion_annotations(
    *,
    primary_series: list[PlotData],
    primary_count: int,
    shared_initial: tuple[int, float] | None,
    x_min: int,
    x_max: int,
    plot_y0: float,
    plot_y1: float,
    sx: Any,
) -> str:
    lines: list[str] = []
    expansion_slots: dict[int, int] = {}
    for series_index, series in enumerate(primary_series):
        if not series.feature_expansions:
            continue
        color = series_color(series, series_index=series_index, primary_count=primary_count)
        for annotation in feature_expansion_annotations(series.feature_expansions):
            display_x = annotation.iteration - series_x_shift(series, shared_initial=shared_initial)
            if not (x_min <= display_x <= x_max):
                continue
            slot = expansion_slots.get(display_x, 0)
            expansion_slots[display_x] = slot + 1
            x = sx(display_x)
            label = format_feature_expansion_label(annotation)
            label_y = plot_y0 + 32 + slot * 22
            label_x = min(max(x + 8, 102), sx(x_max) - 8)
            anchor = "start"
            if label_x > sx(x_max) - 170:
                label_x = max(sx(x_min) + 8, x - 8)
                anchor = "end"
            lines.extend(
                [
                    f'<line x1="{x:.2f}" y1="{plot_y0}" x2="{x:.2f}" y2="{plot_y1}" '
                    f'stroke="{color}" stroke-width="1.4" stroke-dasharray="4 5" opacity="0.58"/>',
                    f'<circle cx="{x:.2f}" cy="{plot_y0 + 13:.2f}" r="4.5" fill="{color}" '
                    'stroke="#ffffff" stroke-width="1.2" opacity="0.92"/>',
                    svg_text(
                        label_x,
                        label_y,
                        label,
                        size=11,
                        weight="600",
                        fill=color,
                        anchor=anchor,
                    ),
                ]
            )
    return "\n".join(lines)


def feature_expansion_annotations(expansions: list[FeatureExpansion]) -> list[FeatureExpansionAnnotation]:
    grouped: dict[int, list[FeatureExpansion]] = {}
    for expansion in expansions:
        grouped.setdefault(expansion.iteration, []).append(expansion)

    annotations: list[FeatureExpansionAnnotation] = []
    for iteration in sorted(grouped):
        items = grouped[iteration]
        counts = [
            item.active_feature_count
            for item in items
            if item.active_feature_count is not None and item.active_feature_count > 0
        ]
        to_count = max(counts) if counts else None
        from_count = min(counts) - 1 if counts else None
        names = [item.name for item in items if item.name]
        annotations.append(
            FeatureExpansionAnnotation(
                iteration=iteration,
                from_count=from_count,
                to_count=to_count,
                names=names,
            )
        )
    return annotations


def format_feature_expansion_label(annotation: FeatureExpansionAnnotation) -> str:
    if annotation.from_count is not None and annotation.to_count is not None:
        return f"feature dim {annotation.from_count} -> {annotation.to_count}"
    if annotation.names:
        return f"feature added: {annotation.names[0]}"
    return "feature dim increased"


def render_legend(
    *,
    series_list: list[PlotData],
    width: int,
    show_selected: bool,
    primary_count: int,
) -> str:
    row_count = len(series_list) + (1 if show_selected else 0)
    longest_label = max(
        [
            len(legend_label(series, index=index, primary_count=primary_count))
            for index, series in enumerate(series_list)
        ]
        + ([len("all data points")] if show_selected else [0])
    )
    legend_width = max(300, min(520, longest_label * 7 + 100))
    legend_height = 24 + row_count * 25
    legend_x = width - legend_width - 52
    legend_y = 70
    lines = [
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" '
        'fill="#ffffff" stroke="#d6d3ca" rx="6"/>',
    ]
    row_y = legend_y + 23
    for index, series in enumerate(series_list):
        color = series_color(series, series_index=index, primary_count=primary_count)
        stroke_width = 4 if index < primary_count else 3
        dasharray = series_dasharray(series)
        dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        lines.extend(
            [
                f'<line x1="{legend_x + 18}" y1="{row_y}" x2="{legend_x + 58}" y2="{row_y}" '
                f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round"{dash_attr}/>',
                svg_text(
                    legend_x + 70,
                    row_y + 5,
                    legend_label(series, index=index, primary_count=primary_count),
                    size=12,
                    fill="#111827",
                ),
            ]
        )
        row_y += 25
    if show_selected:
        lines.extend(
            [
                f'<circle cx="{legend_x + 28}" cy="{row_y}" r="3.5" fill="#9ca3af" '
                'stroke="#ffffff" stroke-width="1"/>',
                f'<circle cx="{legend_x + 38}" cy="{row_y}" r="3.5" fill="#9ca3af" '
                'stroke="#ffffff" stroke-width="1"/>',
                f'<circle cx="{legend_x + 48}" cy="{row_y}" r="3.5" fill="#9ca3af" '
                'stroke="#ffffff" stroke-width="1"/>',
                svg_text(
                    legend_x + 70,
                    row_y + 5,
                    "all data points",
                    size=12,
                    fill="#111827",
                ),
            ]
        )
    return "\n".join(lines)


def legend_label(series: PlotData, *, index: int, primary_count: int) -> str:
    if index < primary_count:
        if series.run_name != series.source.parent.name:
            return series.run_name
        if primary_count == 1:
            return "LDM-TTS with BoN-N4H4"
        return series.source.parent.name
    if series.source.name == "results.tsv":
        return "ReAct (CodeX)"
    if is_qwen_react_baseline(series):
        return "ReAct"
    return series.run_name


def series_color(series: PlotData, *, series_index: int, primary_count: int) -> str:
    if series_index < primary_count:
        return PRIMARY_SERIES_COLORS[series_index % len(PRIMARY_SERIES_COLORS)]
    if series.source.name == "results.tsv":
        return RESULTS_TSV_COLOR
    if is_qwen_react_baseline(series):
        return BASELINE_REACT_COLOR
    comparison_index = max(0, series_index - primary_count)
    return SERIES_COLORS[(comparison_index + 3) % len(SERIES_COLORS)]


def series_dasharray(series: PlotData) -> str:
    if series.source.name == "results.tsv":
        return "8 6"
    if is_qwen_react_baseline(series):
        return "8 6"
    return ""


def shared_initial_point(
    *,
    metric: str,
    comparisons: list[PlotData],
) -> tuple[int, float] | None:
    initial = initial_anchor_from_tsv(DEFAULT_INITIAL_ANCHOR_TSV, metric=metric)
    if initial is not None:
        return initial
    if not comparisons or not comparisons[0].points:
        return None
    first_point = comparisons[0].points[0]
    return first_point.iteration, first_point.best_after


def initial_anchor_from_tsv(path: Path, *, metric: str) -> tuple[int, float] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames or []
            metric_col = metric_column(fieldnames, metric)
            if metric_col is None:
                return None
            for row_number, row in enumerate(reader, start=1):
                status = first_nonempty(row, ("status",))
                score = as_float(row.get(metric_col))
                if is_failure_status(status) or score is None or not usable_score(score):
                    continue
                return row_iteration(row, default=row_number), score
    except OSError:
        return None
    return None


def needs_shared_initial(series: PlotData, *, series_index: int, primary_count: int) -> bool:
    if series_index < primary_count:
        return True
    return is_qwen_react_baseline(series)


def is_qwen_react_baseline(series: PlotData) -> bool:
    return series.source.parent.name in {
        "baseline_tool_call_real_train_b1_i20_e1_20260630_110732",
        "baseline_tool_call_real_train_b1_i100_e1_20260706_163546",
    }


def series_x_shift(
    series: PlotData,
    *,
    shared_initial: tuple[int, float] | None,
) -> int:
    if not series.points or shared_initial is None:
        return 0
    return series.points[0].iteration - (shared_initial[0] + 1)


def display_iteration(
    point: IterationPoint,
    *,
    series: PlotData,
    series_index: int,
    primary_count: int,
    shared_initial: tuple[int, float] | None,
) -> int:
    if series_index < primary_count:
        return point.iteration - series_x_shift(series, shared_initial=shared_initial)
    if shared_initial is not None and needs_shared_initial(series, series_index=series_index, primary_count=primary_count):
        return point.iteration + 1
    return point.iteration


def convert_svg_to_png(svg_path: Path, png_path: Path) -> Path | None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return None
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [converter, str(svg_path), "-o", str(png_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: could not convert SVG to PNG: {exc}", file=sys.stderr)
        return None
    return png_path


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> Path | None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return None
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [converter, "--format=pdf", str(svg_path), "-o", str(pdf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: could not convert SVG to PDF: {exc}", file=sys.stderr)
        return None
    return pdf_path


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    weight: str = "400",
    fill: str,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" '
        'font-family="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def padded_range(
    values: list[float],
    *,
    pad_fraction: float = 0.12,
    min_pad: float = 0.02,
) -> tuple[float, float]:
    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        pad = max(abs(y_min) * pad_fraction, min_pad)
    else:
        pad = max((y_max - y_min) * pad_fraction, min_pad)
    return y_min - pad, y_max + pad


def make_y_ticks(y_min: float, y_max: float) -> list[float]:
    raw_step = (y_max - y_min) / 7.0
    if raw_step <= 0:
        return [y_min]
    magnitude = 10 ** math.floor(math.log10(raw_step))
    candidates = [1, 2, 2.5, 5, 10]
    step = min(candidates, key=lambda value: abs(value * magnitude - raw_step)) * magnitude
    tick_start = math.floor(y_min / step) * step
    tick_end = math.ceil(y_max / step) * step
    ticks: list[float] = []
    tick = tick_start
    while tick <= tick_end + step * 0.5:
        ticks.append(round(tick, 10))
        tick += step
    return ticks


def make_x_ticks(x_min: int, x_max: int, *, max_ticks: int = 8) -> list[int]:
    if x_min == x_max:
        return [x_min]
    span = x_max - x_min
    if span <= max_ticks - 1:
        return list(range(x_min, x_max + 1))

    raw_step = span / max(1, max_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    candidates = [1, 2, 5, 10]
    step = 10 * magnitude
    for candidate in candidates:
        candidate_step = candidate * magnitude
        if candidate_step >= raw_step:
            step = candidate_step
            break
    step = max(1, int(step))

    tick_start = int(math.ceil(x_min / step) * step)
    tick_end = int(math.floor(x_max / step) * step)
    ticks = list(range(tick_start, tick_end + 1, step))
    if not ticks:
        return [x_min, x_max]
    if ticks[0] != x_min and len(ticks) < max_ticks:
        ticks.insert(0, x_min)
    if ticks[-1] != x_max and len(ticks) < max_ticks:
        ticks.append(x_max)
    return ticks


def better_score(current: float | None, candidate: float, *, minimize: bool) -> float:
    if current is None:
        return candidate
    if minimize:
        return min(current, candidate)
    return max(current, candidate)


def best_of(values: list[float], *, minimize: bool) -> float | None:
    if not values:
        return None
    return min(values) if minimize else max(values)


def usable_score(value: float) -> bool:
    return math.isfinite(value) and abs(value) < FAILURE_SCORE_CUTOFF


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def first_string(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                return item
    return None


def format_float(value: float) -> str:
    return f"{value:.12g}"


def format_axis_float(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
