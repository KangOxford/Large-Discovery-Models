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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PlotData:
    run_name: str
    metric: str
    minimize: bool
    points: list[IterationPoint]
    warmup_best: float | None
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot best validation metric by search iteration. "
            "Input can be a run directory, model_based_summary.json, baseline_summary.json, "
            "model_based.log, or baseline.log."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Run directory, summary JSON, or run log.",
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
        "--no-png",
        action="store_true",
        help="Do not attempt PNG conversion.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = resolve_source(args.path)
    data = load_plot_data(source, metric=args.metric, minimize=not args.maximize)
    if args.no_warmup:
        data = PlotData(
            run_name=data.run_name,
            metric=data.metric,
            minimize=data.minimize,
            points=data.points,
            warmup_best=None,
            source=data.source,
        )

    svg_path = resolve_svg_path(args.out, source, data.metric)
    csv_path = args.csv or svg_path.with_suffix(".csv")
    png_path = args.png or svg_path.with_suffix(".png")

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(data, csv_path)
    write_svg(
        data,
        svg_path,
        width=max(700, int(args.width)),
        height=max(460, int(args.height)),
        show_selected=not args.no_selected,
    )

    converted_png: Path | None = None
    if not args.no_png:
        converted_png = convert_svg_to_png(svg_path, png_path)

    result = {
        "source": str(source),
        "svg": str(svg_path),
        "csv": str(csv_path),
        "png": None if converted_png is None else str(converted_png),
        "metric": data.metric,
        "mode": "minimize" if data.minimize else "maximize",
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


def resolve_svg_path(out_arg: Path | None, source: Path, metric: str) -> Path:
    default_name = f"{metric}_best_by_iteration.svg"
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


def write_csv(data: PlotData, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                f"iteration_{data.metric}",
                f"best_{data.metric}_after_iteration",
                "state_id",
            ],
        )
        writer.writeheader()
        for point in data.points:
            writer.writerow(
                {
                    "iteration": point.iteration,
                    f"iteration_{data.metric}": "" if point.iteration_score is None else format_float(point.iteration_score),
                    f"best_{data.metric}_after_iteration": format_float(point.best_after),
                    "state_id": point.state_id,
                }
            )


def write_svg(
    data: PlotData,
    path: Path,
    *,
    width: int,
    height: int,
    show_selected: bool,
) -> None:
    margin_left = 96
    margin_right = 48
    margin_top = 94
    margin_bottom = 92
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    plot_y1 = plot_y0 + plot_h

    x_values = [point.iteration for point in data.points]
    y_values = [point.best_after for point in data.points]
    if show_selected:
        y_values.extend(point.iteration_score for point in data.points if point.iteration_score is not None)
    if data.warmup_best is not None:
        y_values.append(data.warmup_best)

    y_min, y_max = padded_range(y_values)
    y_ticks = make_y_ticks(y_min, y_max)

    def sx(x_value: int) -> float:
        if len(x_values) == 1 or min(x_values) == max(x_values):
            return plot_x0 + plot_w / 2.0
        return plot_x0 + (x_value - min(x_values)) / (max(x_values) - min(x_values)) * plot_w

    def sy(y_value: float) -> float:
        return plot_y1 - (y_value - y_min) / (y_max - y_min) * plot_h

    def polyline(points: list[tuple[int, float]]) -> str:
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)

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
            f"Best {data.metric} by model-based searching iteration",
            size=25,
            weight="700",
            fill="#111827",
        )
    )
    append(svg_text(52, 68, f"Run: {data.run_name}", size=13, fill="#4b5563"))
    direction = "Lower is better" if data.minimize else "Higher is better"
    append(
        svg_text(
            52,
            88,
            f"{direction}. Values are from {data.source.name}.",
            size=13,
            fill="#4b5563",
        )
    )
    append(
        f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" '
        'fill="#ffffff" stroke="#d6d3ca" stroke-width="1" rx="6"/>'
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

    for x_value in make_x_ticks(min(x_values), max(x_values)):
        x = sx(x_value)
        append(
            f'<line x1="{x:.2f}" y1="{plot_y0}" x2="{x:.2f}" y2="{plot_y1}" '
            'stroke="#f3f0e8" stroke-width="1"/>'
        )
        append(svg_text(x, plot_y1 + 30, str(x_value), size=14, fill="#374151", anchor="middle"))

    if data.warmup_best is not None:
        y = sy(data.warmup_best)
        append(
            f'<line x1="{plot_x0}" y1="{y:.2f}" x2="{plot_x0 + plot_w}" y2="{y:.2f}" '
            'stroke="#8b6f47" stroke-width="2" stroke-dasharray="7 7" opacity="0.75"/>'
        )
        label = f"warmup best {data.warmup_best:.6f}"
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

    best_points = [(point.iteration, point.best_after) for point in data.points]
    append(
        f'<polyline fill="none" stroke="#0f766e" stroke-width="4" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{polyline(best_points)}"/>'
    )

    for point in data.points:
        x = sx(point.iteration)
        y_best = sy(point.best_after)
        if show_selected and point.iteration_score is not None:
            y_selected = sy(point.iteration_score)
            append(
                f'<circle cx="{x:.2f}" cy="{y_selected:.2f}" r="3.5" fill="#9ca3af" '
                'stroke="#ffffff" stroke-width="1"/>'
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
    append(render_legend(width=width, show_selected=show_selected))
    append("</svg>")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_legend(*, width: int, show_selected: bool) -> str:
    legend_width = 350 if show_selected else 250
    legend_height = 64 if show_selected else 39
    legend_x = width - legend_width - 52
    legend_y = 32
    lines = [
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" '
        'fill="#ffffff" stroke="#d6d3ca" rx="6"/>',
        f'<line x1="{legend_x + 18}" y1="{legend_y + 22}" x2="{legend_x + 58}" y2="{legend_y + 22}" '
        'stroke="#0f766e" stroke-width="4" stroke-linecap="round"/>',
        svg_text(legend_x + 70, legend_y + 27, "best after iteration", size=10, fill="#111827"),
    ]
    if show_selected:
        lines.extend(
            [
                f'<circle cx="{legend_x + 28}" cy="{legend_y + 47}" r="3.5" fill="#9ca3af" '
                'stroke="#ffffff" stroke-width="1"/>',
                f'<circle cx="{legend_x + 38}" cy="{legend_y + 47}" r="3.5" fill="#9ca3af" '
                'stroke="#ffffff" stroke-width="1"/>',
                f'<circle cx="{legend_x + 48}" cy="{legend_y + 47}" r="3.5" fill="#9ca3af" '
                'stroke="#ffffff" stroke-width="1"/>',
                svg_text(
                    legend_x + 70,
                    legend_y + 52,
                    "per-iteration selected metric",
                    size=13,
                    fill="#111827",
                ),
            ]
        )
    return "\n".join(lines)


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


def padded_range(values: list[float]) -> tuple[float, float]:
    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 0.02)
    else:
        pad = max((y_max - y_min) * 0.12, 0.02)
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
