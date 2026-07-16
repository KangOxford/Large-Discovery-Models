#!/usr/bin/env python3
"""Render an HTML progress report for tilted case2 TTS trajectories."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Sequence


PALETTE = [
    "#1d4ed8",
    "#c2410c",
    "#047857",
    "#7c3aed",
    "#be123c",
    "#0f766e",
    "#a16207",
    "#4338ca",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_paths = [Path(path) for path in args.paths]
    run_dirs = discover_run_dirs(input_paths)
    if len(run_dirs) == 1 and not args.compare:
        run_dir = run_dirs[0]
        report_path = Path(args.output) if args.output else run_dir / "progress_report.html"
        report_path.write_text(render_report(run_dir, load_run(run_dir)), encoding="utf-8")
    else:
        report_path = Path(args.output) if args.output else default_comparison_path(input_paths, run_dirs)
        runs = [
            {
                "run_dir": run_dir,
                "name": run_dir.name,
                "color": PALETTE[idx % len(PALETTE)],
                "data": load_run(run_dir),
            }
            for idx, run_dir in enumerate(run_dirs)
        ]
        report_path.write_text(render_comparison_report(runs), encoding="utf-8")
    print(str(report_path.resolve()))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize tilted case2 TTS trajectory directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories, or parent directories whose children contain rounds.jsonl.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Force comparison layout even when only one run is discovered.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output HTML path. Defaults to progress_report.html for one run, comparison_report.html for multiple.",
    )
    return parser.parse_args(argv)


def discover_run_dirs(paths: Sequence[Path]) -> list[Path]:
    run_dirs = []
    seen = set()
    for path in paths:
        if (path / "rounds.jsonl").exists():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                child for child in path.iterdir()
                if child.is_dir() and (child / "rounds.jsonl").exists()
            )
        else:
            raise SystemExit(f"missing run directory: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dirs.append(candidate)
    if not run_dirs:
        searched = ", ".join(str(path) for path in paths)
        raise SystemExit(f"no run directories with rounds.jsonl found in: {searched}")
    return run_dirs


def default_comparison_path(input_paths: Sequence[Path], run_dirs: Sequence[Path]) -> Path:
    if len(input_paths) == 1 and input_paths[0].is_dir() and not (input_paths[0] / "rounds.jsonl").exists():
        return input_paths[0] / "comparison_report.html"
    return run_dirs[0].parent / "comparison_report.html"


def load_run(run_dir: Path) -> dict[str, Any]:
    rounds_path = run_dir / "rounds.jsonl"
    history_path = run_dir / "history.json"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if not rounds_path.exists():
        raise SystemExit(f"missing {rounds_path}")
    rounds = [
        json.loads(line)
        for line in rounds_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history = _load_json(history_path, [])
    summary = _load_json(summary_path, {})
    config = _load_json(config_path, {})
    return {
        "rounds": rounds,
        "history": history,
        "summary": summary,
        "config": config,
        "series": build_series(rounds, history, config),
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_series(rounds: Sequence[dict[str, Any]], history: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    best_vina = None
    best_activity = None
    hv_points = []
    ref_point = tuple(config.get("ref_point", [0.0, 5.0]))
    minimize = tuple(config.get("minimize", [True, False]))
    for idx, record in enumerate(rounds):
        selection = record.get("selection_results", {})
        smiles_list = selection.get("selected_smiles", [])
        scores_list = selection.get("selected_scores", [])
        smiles = smiles_list[0] if smiles_list else ""
        scores = scores_list[0] if scores_list else [None, None]
        vina = _float_or_none(scores[0] if len(scores) > 0 else None)
        activity = _float_or_none(scores[1] if len(scores) > 1 else None)
        if vina is not None:
            best_vina = vina if best_vina is None else min(best_vina, vina)
        if activity is not None:
            best_activity = activity if best_activity is None else max(best_activity, activity)
        if vina is not None and activity is not None:
            hv_points.append((vina, activity))
        pareto_points = pareto_front_2d(hv_points, minimize)
        rows.append({
            "round": record.get("round_idx", idx),
            "evaluation_count": len(hv_points),
            "smiles": smiles,
            "vina": vina,
            "activity": activity,
            "best_vina": best_vina,
            "best_activity": best_activity,
            "hypervolume": hypervolume_2d(hv_points, ref_point, minimize),
            "pareto_front_hypervolume": hypervolume_2d(pareto_points, ref_point, minimize),
            "pareto_front_size": len(pareto_points),
            "candidate_count": len(record.get("candidates", [])),
            "drop_duplicate": int(record.get("drop_counts", {}).get("duplicate", 0)),
            "drop_invalid": int(record.get("drop_counts", {}).get("invalid", 0)),
            "drop_evaluated": int(record.get("drop_counts", {}).get("evaluated", 0)),
            "fallback": record.get("fallback_reason"),
            "selection_mode": record.get("selection_mode"),
            "q0_effective_support": _float_or_none(record.get("q0_effective_support")),
            "prob_effective_sample_size": _float_or_none(record.get("prob_effective_sample_size")),
            "selected_probability": _first_float(selection.get("selected_probabilities")),
            "selected_ehvi": _first_float(selection.get("selected_ehvi")),
            "llm_attempts": len(record.get("llm_attempts", [])),
            "source_count": len(record.get("sources", [])),
        })
    return {
        "rows": rows,
        "history_points": [
            {
                "smiles": item.get("smiles", ""),
                "vina": _float_or_none((item.get("scores") or [None, None])[0]),
                "activity": _float_or_none((item.get("scores") or [None, None])[1]),
            }
            for item in history
        ],
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(values: Any) -> float | None:
    if not values:
        return None
    return _float_or_none(values[0])


def pareto_front_2d(points: Sequence[tuple[float, float]], minimize: Sequence[bool]) -> list[tuple[float, float]]:
    front = []
    for idx, point in enumerate(points):
        dominated = False
        for other_idx, other in enumerate(points):
            if idx == other_idx:
                continue
            if dominates_2d(other, point, minimize):
                dominated = True
                break
        if not dominated:
            front.append(point)
    return front


def dominates_2d(left: tuple[float, float], right: tuple[float, float], minimize: Sequence[bool]) -> bool:
    better_or_equal = []
    strictly_better = []
    for left_value, right_value, is_minimize in zip(left, right, minimize):
        if is_minimize:
            better_or_equal.append(left_value <= right_value)
            strictly_better.append(left_value < right_value)
        else:
            better_or_equal.append(left_value >= right_value)
            strictly_better.append(left_value > right_value)
    return all(better_or_equal) and any(strictly_better)


def hypervolume_2d(points: Sequence[tuple[float, float]], ref_point: Sequence[float], minimize: Sequence[bool]) -> float:
    rectangles = []
    ref_vina, ref_activity = float(ref_point[0]), float(ref_point[1])
    for vina, activity in points:
        x = ref_vina - vina if minimize[0] else vina - ref_vina
        y = ref_activity - activity if minimize[1] else activity - ref_activity
        if x > 0.0 and y > 0.0:
            rectangles.append((x, y))
    if not rectangles:
        return 0.0
    rectangles.sort(key=lambda item: item[0])
    suffix_y = [0.0 for _ in rectangles]
    current = 0.0
    for idx in range(len(rectangles) - 1, -1, -1):
        current = max(current, rectangles[idx][1])
        suffix_y[idx] = current
    area = 0.0
    prev_x = 0.0
    for (x, _y), height in zip(rectangles, suffix_y):
        area += max(0.0, x - prev_x) * height
        prev_x = x
    return area


def render_report(run_dir: Path, data: dict[str, Any]) -> str:
    summary = data["summary"]
    config = data["config"]
    rows = data["series"]["rows"]
    history_points = data["series"]["history_points"]
    title = f"Tilted Case2 Progress - {run_dir.name}"
    cards = [
        ("Rounds", summary.get("round_count", len(rows))),
        ("History Size", summary.get("history_size", len(history_points))),
        ("Final Hypervolume", _fmt(summary.get("final_hypervolume"))),
        ("LLM Calls", summary.get("llm_call_count", sum(row["llm_attempts"] for row in rows))),
        ("Early Stop", summary.get("early_stop_reason") or "none"),
        ("Method", summary.get("method") or config.get("method") or "-"),
    ]
    sections = [
        render_cards(cards),
        render_plot_grid(rows, history_points),
        render_table(rows),
        render_config_summary(config, summary),
    ]
    return "\n".join([
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{escape(title)}</title>",
        "<style>",
        CSS,
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{escape(title)}</h1>",
        f"<p class=\"subtitle\">Run directory: <code>{escape(str(run_dir))}</code></p>",
        *sections,
        "</main>",
        "</body>",
        "</html>",
    ])


def render_comparison_report(runs: Sequence[dict[str, Any]]) -> str:
    title = "Tilted Case2 Progress Comparison"
    groups = group_runs_by_prefix(runs)
    averaged_runs = build_average_runs(groups)
    final_hvs = [(run, final_metric(run, "pareto_front_hypervolume")) for run in runs]
    best_run, best_hv = max(final_hvs, key=lambda item: item[1] if item[1] is not None else float("-inf"))
    early_stops = [
        run["name"] for run in runs
        if run["data"]["summary"].get("early_stop_reason")
    ]
    cards = [
        ("Runs", len(runs)),
        ("Groups", len(groups)),
        ("Total Rounds", sum(len(run["data"]["series"]["rows"]) for run in runs)),
        ("Best Final HV", _fmt(best_hv)),
        ("Best Run", best_run["name"]),
        ("Early Stops", len(early_stops)),
    ]
    sections = [
        render_cards(cards),
        render_visibility_controls(runs, averaged_runs),
        render_comparison_plot_grid(runs),
        render_average_section(groups, averaged_runs),
        render_comparison_table(runs),
        render_comparison_settings(runs),
    ]
    return "\n".join([
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{escape(title)}</title>",
        "<style>",
        CSS,
        "</style>",
        "<script>",
        JS,
        "</script>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{escape(title)}</h1>",
        "<p class=\"subtitle\">Compared runs: "
        + ", ".join(f"<code>{escape(run['name'])}</code>" for run in runs)
        + "</p>",
        *sections,
        "</main>",
        "</body>",
        "</html>",
    ])


def group_runs_by_prefix(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        prefix, run_number = split_run_name(run["name"])
        grouped.setdefault(prefix, []).append({**run, "run_number": run_number})
    groups = []
    for idx, (prefix, members) in enumerate(sorted(grouped.items())):
        members.sort(key=lambda run: (run["run_number"] is None, run["run_number"] or 0, run["name"]))
        groups.append({
            "name": prefix,
            "color": PALETTE[idx % len(PALETTE)],
            "runs": members,
        })
    return groups


def split_run_name(name: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<prefix>.+)_run(?P<number>\d+)$", name)
    if not match:
        return name, None
    return match.group("prefix"), int(match.group("number"))


def build_average_runs(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": group["name"],
            "color": group["color"],
            "member_count": len(group["runs"]),
            "data": {
                "summary": build_group_summary(group["runs"]),
                "config": representative_config(group["runs"]),
                "series": {
                    "rows": average_rows_by_evaluation(group["runs"]),
                    "history_points": [],
                },
            },
        }
        for group in groups
    ]


def build_group_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    final_hvs = [_float_or_none(final_metric(run, "pareto_front_hypervolume")) for run in runs]
    final_vinas = [_float_or_none(final_metric(run, "best_vina")) for run in runs]
    final_activities = [_float_or_none(final_metric(run, "best_activity")) for run in runs]
    round_counts = [len(run["data"]["series"]["rows"]) for run in runs]
    return {
        "run_count": len(runs),
        "round_count_mean": mean_value(round_counts),
        "final_hv_mean": mean_value(final_hvs),
        "final_hv_std": std_value(final_hvs),
        "best_vina_mean": mean_value(final_vinas),
        "best_vina_std": std_value(final_vinas),
        "best_activity_mean": mean_value(final_activities),
        "best_activity_std": std_value(final_activities),
    }


def representative_config(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    return runs[0]["data"]["config"]


def average_rows_by_evaluation(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "best_vina",
        "best_activity",
        "pareto_front_hypervolume",
        "pareto_front_size",
    ]
    by_eval: dict[int, dict[str, list[float]]] = {}
    common_eval_limit = common_x_limit(runs, "evaluation_count")
    for run in runs:
        latest_by_eval: dict[int, dict[str, Any]] = {}
        for row in run["data"]["series"]["rows"]:
            evaluation_count = row.get("evaluation_count")
            if evaluation_count is not None:
                latest_by_eval[int(evaluation_count)] = row
        for evaluation_count, row in latest_by_eval.items():
            if common_eval_limit is not None and evaluation_count > common_eval_limit:
                continue
            bucket = by_eval.setdefault(evaluation_count, {metric: [] for metric in metrics})
            for metric in metrics:
                value = _float_or_none(row.get(metric))
                if value is not None:
                    bucket[metric].append(value)
    rows = []
    for evaluation_count in sorted(by_eval):
        values_by_metric = by_eval[evaluation_count]
        row: dict[str, Any] = {
            "round": evaluation_count,
            "evaluation_count": evaluation_count,
            "n": max(len(values) for values in values_by_metric.values()) if values_by_metric else 0,
        }
        for metric, values in values_by_metric.items():
            row[metric] = mean_value(values)
            row[f"{metric}_std"] = std_value(values)
        rows.append(row)
    return rows


def common_x_limit(runs: Sequence[dict[str, Any]], x_key: str) -> float | None:
    maxima = []
    for run in runs:
        values = [
            _float_or_none(row.get(x_key))
            for row in run["data"]["series"]["rows"]
            if row.get(x_key) is not None
        ]
        values = [value for value in values if value is not None]
        if values:
            maxima.append(max(values))
    if not maxima:
        return None
    return min(maxima)


def mean_value(values: Sequence[Any]) -> float | None:
    finite = [_float_or_none(value) for value in values]
    finite = [value for value in finite if value is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def std_value(values: Sequence[Any]) -> float | None:
    finite = [_float_or_none(value) for value in values]
    finite = [value for value in finite if value is not None]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return variance ** 0.5


def render_cards(cards: Sequence[tuple[str, Any]]) -> str:
    items = []
    for label, value in cards:
        items.append(
            "<div class=\"metric\">"
            f"<div class=\"metric-label\">{escape(label)}</div>"
            f"<div class=\"metric-value\">{escape(str(value))}</div>"
            "</div>"
        )
    return "<section class=\"metrics\">" + "\n".join(items) + "</section>"


def render_visibility_controls(runs: Sequence[dict[str, Any]], averaged_runs: Sequence[dict[str, Any]]) -> str:
    return "\n".join([
        "<section class=\"visibility-panel\">",
        render_toggle_group("Raw Runs", "run", runs),
        render_toggle_group("Averaged Setups", "setup", averaged_runs),
        "</section>",
    ])


def render_toggle_group(title: str, scope: str, series: Sequence[dict[str, Any]]) -> str:
    items = []
    for item in series:
        items.append(
            "<label class=\"toggle-item\">"
            f"<input class=\"series-toggle\" type=\"checkbox\" checked data-scope=\"{escape(scope)}\" "
            f"data-series=\"{escape(item['name'])}\">"
            f"<span class=\"swatch\" style=\"background:{escape(item['color'])}\"></span>"
            f"<span>{escape(short_run_name(item['name']))}</span>"
            "</label>"
        )
    return "\n".join([
        "<div class=\"toggle-group\">",
        "<div class=\"toggle-header\">",
        f"<h2>{escape(title)}</h2>",
        "<div class=\"toggle-actions\">",
        f"<button type=\"button\" data-toggle-scope=\"{escape(scope)}\" data-toggle-state=\"show\">Show all</button>",
        f"<button type=\"button\" data-toggle-scope=\"{escape(scope)}\" data-toggle-state=\"hide\">Hide all</button>",
        "</div>",
        "</div>",
        "<div class=\"toggle-list\">",
        *items,
        "</div>",
        "</div>",
    ])


def render_plot_grid(rows: Sequence[dict[str, Any]], history_points: Sequence[dict[str, Any]]) -> str:
    plots = [
        ("Objective Scores", line_svg(rows, [
            ("vina", "Vina", "#0f766e"),
            ("activity", "Activity", "#b45309"),
        ], y_label="score")),
        ("Best So Far", line_svg(rows, [
            ("best_vina", "Best Vina", "#0f766e"),
            ("best_activity", "Best Activity", "#b45309"),
        ], y_label="score")),
        ("Hypervolume of Pareto Front by Number of Evaluations", line_svg(rows, [
            ("pareto_front_hypervolume", "Pareto HV", "#1d4ed8"),
        ], y_label="hypervolume", x_key="evaluation_count", x_label="evaluations")),
        ("Pareto Front Size", line_svg(rows, [
            ("pareto_front_size", "Pareto size", "#7c3aed"),
        ], y_label="molecules", x_key="evaluation_count", x_label="evaluations")),
        ("Reservoir Health", line_svg(rows, [
            ("candidate_count", "Candidates", "#1d4ed8"),
            ("drop_duplicate", "Duplicates", "#be123c"),
            ("drop_invalid", "Invalid", "#7c2d12"),
            ("drop_evaluated", "Evaluated", "#6d28d9"),
        ], y_label="count")),
        ("Selection Concentration", line_svg(rows, [
            ("q0_effective_support", "q0 eff support", "#047857"),
            ("prob_effective_sample_size", "prob ESS", "#7c3aed"),
        ], y_label="effective size")),
        ("Objective Scatter", scatter_svg(history_points)),
    ]
    blocks = [
        "<div class=\"plot-card\">"
        f"<h2>{escape(title)}</h2>"
        f"{svg}"
        "</div>"
        for title, svg in plots
    ]
    return "<section class=\"plot-grid\">" + "\n".join(blocks) + "</section>"


def render_comparison_plot_grid(runs: Sequence[dict[str, Any]]) -> str:
    plots = [
        ("Pareto Hypervolume by Evaluation", multi_line_svg(
            runs,
            "pareto_front_hypervolume",
            y_label="hypervolume",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(runs, "evaluation_count"),
        ), ""),
        ("Best Vina by Evaluation", multi_line_svg(
            runs,
            "best_vina",
            y_label="Vina, lower is better",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(runs, "evaluation_count"),
        ), ""),
        ("Best Activity by Evaluation", multi_line_svg(
            runs,
            "best_activity",
            y_label="activity, higher is better",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(runs, "evaluation_count"),
        ), ""),
        ("Pareto Front Size", multi_line_svg(
            runs,
            "pareto_front_size",
            y_label="molecules",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(runs, "evaluation_count"),
        ), ""),
        ("Candidate Reservoir Size", multi_line_svg(
            runs,
            "candidate_count",
            y_label="candidates",
            x_max_limit=common_x_limit(runs, "round"),
        ), ""),
        ("Selection Probability ESS", multi_line_svg(
            runs,
            "prob_effective_sample_size",
            y_label="effective size",
            x_max_limit=common_x_limit(runs, "round"),
        ), ""),
        ("Objective Scatter", comparison_scatter_svg(runs), " wide"),
    ]
    blocks = [
        "<div class=\"plot-card" + extra_class + "\">"
        f"<h2>{escape(title)}</h2>"
        f"{svg}"
        "</div>"
        for title, svg, extra_class in plots
    ]
    return "<section class=\"plot-grid\">" + "\n".join(blocks) + "</section>"


def render_average_section(groups: Sequence[dict[str, Any]], averaged_runs: Sequence[dict[str, Any]]) -> str:
    if not averaged_runs:
        return ""
    return "\n".join([
        "<section class=\"section-block\">",
        "<h2>Mean And Std By Run Prefix</h2>",
        "<p class=\"subtitle compact\">Groups are formed from names matching <code>&lt;prefix&gt;_run&lt;number&gt;</code>. "
        "Curves are clipped to the common evaluation length; shaded bands show one standard deviation across replicated runs.</p>",
        render_average_plot_grid(averaged_runs),
        render_average_table(groups, averaged_runs),
        "</section>",
    ])


def render_average_plot_grid(averaged_runs: Sequence[dict[str, Any]]) -> str:
    plots = [
        ("Pareto Hypervolume Mean +/- Std", mean_std_line_svg(
            averaged_runs,
            "pareto_front_hypervolume",
            "pareto_front_hypervolume_std",
            y_label="hypervolume",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(averaged_runs, "evaluation_count"),
        ), ""),
        ("Best Vina Mean +/- Std", mean_std_line_svg(
            averaged_runs,
            "best_vina",
            "best_vina_std",
            y_label="Vina, lower is better",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(averaged_runs, "evaluation_count"),
        ), ""),
        ("Best Activity Mean +/- Std", mean_std_line_svg(
            averaged_runs,
            "best_activity",
            "best_activity_std",
            y_label="activity, higher is better",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(averaged_runs, "evaluation_count"),
        ), ""),
        ("Pareto Front Size Mean +/- Std", mean_std_line_svg(
            averaged_runs,
            "pareto_front_size",
            "pareto_front_size_std",
            y_label="molecules",
            x_key="evaluation_count",
            x_label="evaluations",
            x_max_limit=common_x_limit(averaged_runs, "evaluation_count"),
        ), ""),
    ]
    blocks = [
        "<div class=\"plot-card\">"
        f"<h2>{escape(title)}</h2>"
        f"{svg}"
        "</div>"
        for title, svg, _extra_class in plots
    ]
    return "<section class=\"plot-grid\">" + "\n".join(blocks) + "</section>"


def render_average_table(groups: Sequence[dict[str, Any]], averaged_runs: Sequence[dict[str, Any]]) -> str:
    averaged_by_name = {run["name"]: run for run in averaged_runs}
    headers = [
        "Prefix", "Runs", "Members", "Mean Final HV", "HV Std",
        "Mean Best Vina", "Vina Std", "Mean Best Activity", "Activity Std",
    ]
    body = []
    for group in sorted(groups, key=lambda item: item["name"]):
        averaged = averaged_by_name[group["name"]]
        summary = averaged["data"]["summary"]
        body.append(
            f"<tr data-scope=\"setup\" data-series=\"{escape(group['name'])}\">"
            + "".join([
            td(group["name"], "run-name"),
            td(len(group["runs"])),
            td(", ".join(run["name"] for run in group["runs"]), "members"),
            td(_fmt(summary.get("final_hv_mean"))),
            td(_fmt(summary.get("final_hv_std"))),
            td(_fmt(summary.get("best_vina_mean"))),
            td(_fmt(summary.get("best_vina_std"))),
            td(_fmt(summary.get("best_activity_mean"))),
            td(_fmt(summary.get("best_activity_std"))),
        ]) + "</tr>")
    return (
        "<section class=\"table-section\"><h2>Average Summary</h2>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table></div></section>"
    )


def line_svg(
    rows: Sequence[dict[str, Any]],
    series: Sequence[tuple[str, str, str]],
    y_label: str,
    *,
    x_key: str = "round",
    x_label: str = "round",
) -> str:
    width, height = 720, 320
    left, right, top, bottom = 56, 22, 28, 48
    x_values = [float(row[x_key]) for row in rows if row.get(x_key) is not None]
    values = [
        float(row[key])
        for key, _label, _color in series
        for row in rows
        if row.get(key) is not None
    ]
    if not rows or not values:
        return empty_svg(width, height)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def sx(x: float) -> float:
        if x_max == x_min:
            return (left + width - right) / 2
        return left + (x - x_min) * (width - left - right) / (x_max - x_min)

    def sy(y: float) -> float:
        return height - bottom - (y - y_min) * (height - top - bottom) / (y_max - y_min)

    elements = axes(width, height, left, right, top, bottom, x_min, x_max, y_min, y_max, y_label, x_label=x_label)
    for key, label, color in series:
        points = [
            (sx(float(row[x_key])), sy(float(row[key])))
            for row in rows
            if row.get(key) is not None and row.get(x_key) is not None
        ]
        if not points:
            continue
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        elements.append(f"<polyline class=\"line\" points=\"{path}\" stroke=\"{color}\"/>")
        for x, y in points:
            elements.append(f"<circle cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"3\" fill=\"{color}\"/>")
        elements.append(f"<text class=\"legend\" x=\"{left}\" y=\"{18 + 18 * len(elements) % 70}\" fill=\"{color}\">{escape(label)}</text>")
    return svg_wrap(width, height, elements)


def multi_line_svg(
    runs: Sequence[dict[str, Any]],
    y_key: str,
    y_label: str,
    *,
    x_key: str = "round",
    x_label: str = "round",
    x_max_limit: float | None = None,
    scope: str = "run",
) -> str:
    width, height = 760, 340
    left, right, top, bottom = 64, 28, 48, 50

    def row_points(rows: Sequence[dict[str, Any]]) -> list[tuple[float, float, dict[str, Any]]]:
        points = []
        for row in rows:
            if row.get(x_key) is None or row.get(y_key) is None:
                continue
            if x_max_limit is not None and float(row[x_key]) > x_max_limit:
                continue
            point = (float(row[x_key]), float(row[y_key]), row)
            if x_key == "evaluation_count" and points and point[0] == points[-1][0]:
                points[-1] = point
            else:
                points.append(point)
        return points

    all_points = []
    for run in runs:
        points = [(x, y) for x, y, _row in row_points(run["data"]["series"]["rows"])]
        if points:
            all_points.extend(points)
    if not all_points:
        return empty_svg(width, height)
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) * (width - left - right) / (x_max - x_min)

    def sy(y: float) -> float:
        return height - bottom - (y - y_min) * (height - top - bottom) / (y_max - y_min)

    elements = axes(
        width, height, left, right, top, bottom, x_min, x_max, y_min, y_max, y_label, x_label=x_label
    )
    elements.extend(legend_items(runs, left, 18, scope=scope))
    for run in runs:
        points = [
            (sx(x_value), sy(y_value), row)
            for x_value, y_value, row in row_points(run["data"]["series"]["rows"])
        ]
        if not points:
            continue
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y, _row in points)
        elements.append(
            f"<polyline class=\"line\" data-scope=\"{escape(scope)}\" data-series=\"{escape(run['name'])}\" "
            f"points=\"{path}\" stroke=\"{run['color']}\">"
            f"<title>{escape(run['name'])}</title></polyline>"
        )
        for x, y, row in points:
            replicate_count = row.get("n")
            replicate_label = f", n={replicate_count}" if replicate_count is not None else ""
            elements.append(
                f"<circle data-scope=\"{escape(scope)}\" data-series=\"{escape(run['name'])}\" "
                f"cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"2.6\" fill=\"{run['color']}\">"
                f"<title>{escape(run['name'])}: {x_label} {escape(row.get(x_key))}, "
                f"{y_label} {_fmt(row.get(y_key))}{replicate_label}</title>"
                "</circle>"
            )
    return svg_wrap(width, height, elements)


def mean_std_line_svg(
    runs: Sequence[dict[str, Any]],
    mean_key: str,
    std_key: str,
    y_label: str,
    *,
    x_key: str = "round",
    x_label: str = "round",
    x_max_limit: float | None = None,
    scope: str = "setup",
) -> str:
    width, height = 760, 340
    left, right, top, bottom = 64, 28, 48, 50

    def row_points(rows: Sequence[dict[str, Any]]) -> list[tuple[float, float, float | None, dict[str, Any]]]:
        points = []
        for row in rows:
            if row.get(x_key) is None or row.get(mean_key) is None:
                continue
            if x_max_limit is not None and float(row[x_key]) > x_max_limit:
                continue
            std = _float_or_none(row.get(std_key))
            point = (float(row[x_key]), float(row[mean_key]), std, row)
            if x_key == "evaluation_count" and points and point[0] == points[-1][0]:
                points[-1] = point
            else:
                points.append(point)
        return points

    all_points = []
    for run in runs:
        for x_value, mean, std, _row in row_points(run["data"]["series"]["rows"]):
            all_points.append((x_value, mean))
            if std is not None:
                all_points.append((x_value, mean - std))
                all_points.append((x_value, mean + std))
    if not all_points:
        return empty_svg(width, height)
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) * (width - left - right) / (x_max - x_min)

    def sy(y: float) -> float:
        return height - bottom - (y - y_min) * (height - top - bottom) / (y_max - y_min)

    elements = axes(
        width, height, left, right, top, bottom, x_min, x_max, y_min, y_max, y_label, x_label=x_label
    )
    elements.extend(legend_items(runs, left, 18, scope=scope))

    for run in runs:
        points = row_points(run["data"]["series"]["rows"])
        band_points = [(x, mean, std) for x, mean, std, _row in points if std is not None]
        if len(band_points) >= 2:
            upper = [(sx(x), sy(mean + (std or 0.0))) for x, mean, std in band_points]
            lower = [(sx(x), sy(mean - (std or 0.0))) for x, mean, std in reversed(band_points)]
            polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in [*upper, *lower])
            elements.append(
                f"<polygon class=\"std-band\" data-scope=\"{escape(scope)}\" data-series=\"{escape(run['name'])}\" "
                f"points=\"{polygon}\" fill=\"{run['color']}\">"
                f"<title>{escape(run['name'])}: mean +/- std</title></polygon>"
            )

    for run in runs:
        points = [
            (sx(x_value), sy(mean), std, row)
            for x_value, mean, std, row in row_points(run["data"]["series"]["rows"])
        ]
        if not points:
            continue
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y, _std, _row in points)
        elements.append(
            f"<polyline class=\"line\" data-scope=\"{escape(scope)}\" data-series=\"{escape(run['name'])}\" "
            f"points=\"{path}\" stroke=\"{run['color']}\">"
            f"<title>{escape(run['name'])}: mean</title></polyline>"
        )
        for x, y, std, row in points:
            replicate_count = row.get("n")
            replicate_label = f", n={replicate_count}" if replicate_count is not None else ""
            std_label = f", std {_fmt(std)}" if std is not None else ""
            elements.append(
                f"<circle data-scope=\"{escape(scope)}\" data-series=\"{escape(run['name'])}\" "
                f"cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"2.8\" fill=\"{run['color']}\">"
                f"<title>{escape(run['name'])}: {x_label} {escape(row.get(x_key))}, "
                f"mean {_fmt(row.get(mean_key))}{std_label}{replicate_label}</title>"
                "</circle>"
            )
    return svg_wrap(width, height, elements)


def scatter_svg(points: Sequence[dict[str, Any]]) -> str:
    width, height = 720, 320
    left, right, top, bottom = 64, 26, 24, 54
    finite = [
        point for point in points
        if point.get("vina") is not None and point.get("activity") is not None
    ]
    if not finite:
        return empty_svg(width, height)
    xs = [float(point["vina"]) for point in finite]
    ys = [float(point["activity"]) for point in finite]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) * (width - left - right) / (x_max - x_min)

    def sy(y: float) -> float:
        return height - bottom - (y - y_min) * (height - top - bottom) / (y_max - y_min)

    elements = axes(
        width,
        height,
        left,
        right,
        top,
        bottom,
        x_min,
        x_max,
        y_min,
        y_max,
        "activity",
        x_label="Vina, lower is better",
    )
    for idx, point in enumerate(finite):
        x = sx(float(point["vina"]))
        y = sy(float(point["activity"]))
        color = "#1d4ed8" if idx < len(finite) - 1 else "#be123c"
        elements.append(f"<circle cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"5\" fill=\"{color}\"><title>round {idx}: {escape(point['smiles'])}</title></circle>")
        elements.append(f"<text class=\"point-label\" x=\"{x + 7:.2f}\" y=\"{y - 7:.2f}\">{idx}</text>")
    return svg_wrap(width, height, elements)


def comparison_scatter_svg(runs: Sequence[dict[str, Any]]) -> str:
    width, height = 1100, 420
    left, right, top, bottom = 72, 30, 50, 58
    finite = []
    for run in runs:
        for idx, point in enumerate(run["data"]["series"]["history_points"]):
            if point.get("vina") is not None and point.get("activity") is not None:
                finite.append((run, idx, point))
    if not finite:
        return empty_svg(width, height)
    xs = [float(point["vina"]) for _run, _idx, point in finite]
    ys = [float(point["activity"]) for _run, _idx, point in finite]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) * (width - left - right) / (x_max - x_min)

    def sy(y: float) -> float:
        return height - bottom - (y - y_min) * (height - top - bottom) / (y_max - y_min)

    elements = axes(
        width,
        height,
        left,
        right,
        top,
        bottom,
        x_min,
        x_max,
        y_min,
        y_max,
        "activity, higher is better",
        x_label="Vina, lower is better",
    )
    elements.extend(legend_items(runs, left, 18, scope="run"))
    for run, idx, point in finite:
        x = sx(float(point["vina"]))
        y = sy(float(point["activity"]))
        radius = 5 if idx == len(run["data"]["series"]["history_points"]) - 1 else 3.4
        elements.append(
            f"<circle class=\"scatter-point\" data-scope=\"run\" data-series=\"{escape(run['name'])}\" "
            f"cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"{radius}\" "
            f"fill=\"{run['color']}\">"
            f"<title>{escape(run['name'])} point {idx}: Vina {_fmt(point['vina'])}, "
            f"Activity {_fmt(point['activity'])}, {escape(point['smiles'])}</title></circle>"
        )
    return svg_wrap(width, height, elements)


def legend_items(runs: Sequence[dict[str, Any]], x: int, y: int, *, scope: str = "run") -> list[str]:
    elements = []
    for idx, run in enumerate(runs):
        col = idx % 3
        row = idx // 3
        item_x = x + col * 220
        item_y = y + row * 17
        elements.append(
            f"<circle data-scope=\"{escape(scope)}\" data-series=\"{escape(run['name'])}\" "
            f"cx=\"{item_x}\" cy=\"{item_y - 4}\" r=\"4\" fill=\"{run['color']}\"/>"
        )
        elements.append(
            f"<text class=\"legend\" data-scope=\"{escape(scope)}\" data-series=\"{escape(run['name'])}\" "
            f"x=\"{item_x + 8}\" y=\"{item_y}\" fill=\"{run['color']}\">"
            f"{escape(short_run_name(run['name']))}</text>"
        )
    return elements


def axes(
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    y_label: str,
    *,
    x_label: str = "round",
) -> list[str]:
    plot_right = width - right
    plot_bottom = height - bottom
    elements = [
        f"<line class=\"axis\" x1=\"{left}\" y1=\"{plot_bottom}\" x2=\"{plot_right}\" y2=\"{plot_bottom}\"/>",
        f"<line class=\"axis\" x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{plot_bottom}\"/>",
        f"<text class=\"axis-label\" x=\"{width / 2:.1f}\" y=\"{height - 12}\">{escape(x_label)}</text>",
        f"<text class=\"axis-label\" transform=\"translate(14 {height / 2:.1f}) rotate(-90)\">{escape(y_label)}</text>",
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + frac * (plot_right - left)
        y = plot_bottom - frac * (plot_bottom - top)
        x_val = x_min + frac * (x_max - x_min)
        y_val = y_min + frac * (y_max - y_min)
        elements.append(f"<line class=\"grid\" x1=\"{x:.2f}\" y1=\"{top}\" x2=\"{x:.2f}\" y2=\"{plot_bottom}\"/>")
        elements.append(f"<line class=\"grid\" x1=\"{left}\" y1=\"{y:.2f}\" x2=\"{plot_right}\" y2=\"{y:.2f}\"/>")
        elements.append(f"<text class=\"tick\" x=\"{x:.2f}\" y=\"{plot_bottom + 18}\">{_fmt(x_val)}</text>")
        elements.append(f"<text class=\"tick\" x=\"{left - 8}\" y=\"{y + 4:.2f}\" text-anchor=\"end\">{_fmt(y_val)}</text>")
    return elements


def svg_wrap(width: int, height: int, elements: Sequence[str]) -> str:
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" "
        "xmlns=\"http://www.w3.org/2000/svg\">"
        + "\n".join(elements)
        + "</svg>"
    )


def empty_svg(width: int, height: int) -> str:
    return svg_wrap(width, height, [
        f"<text x=\"{width / 2}\" y=\"{height / 2}\" text-anchor=\"middle\">No data</text>"
    ])


def render_table(rows: Sequence[dict[str, Any]]) -> str:
    headers = [
        "Round", "Selected SMILES", "Vina", "Activity", "Best Vina",
        "Best Activity", "Pareto HV", "Pareto Size", "Candidates", "Dup",
        "Invalid", "Evaluated", "Fallback", "ESS", "LLM Calls",
    ]
    body = []
    for row in rows:
        body.append("<tr>" + "".join([
            td(row["round"]),
            td(row["smiles"], "smiles"),
            td(_fmt(row["vina"])),
            td(_fmt(row["activity"])),
            td(_fmt(row["best_vina"])),
            td(_fmt(row["best_activity"])),
            td(_fmt(row["pareto_front_hypervolume"])),
            td(row["pareto_front_size"]),
            td(row["candidate_count"]),
            td(row["drop_duplicate"]),
            td(row["drop_invalid"]),
            td(row["drop_evaluated"]),
            td(row["fallback"] or ""),
            td(_fmt(row["prob_effective_sample_size"])),
            td(row["llm_attempts"]),
        ]) + "</tr>")
    return (
        "<section class=\"table-section\"><h2>Round Details</h2>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table></div></section>"
    )


def render_comparison_table(runs: Sequence[dict[str, Any]]) -> str:
    headers = [
        "Run", "Rounds", "History", "Final HV", "Best Vina", "Best Activity",
        "Pareto Size", "LLM Calls", "Avg Candidates", "Dup Drops",
        "Invalid Drops", "Evaluated Drops", "Early Stop", "Settings",
    ]
    body = []
    for run in sorted(runs, key=lambda item: final_metric(item, "pareto_front_hypervolume") or -1, reverse=True):
        data = run["data"]
        rows = data["series"]["rows"]
        summary = data["summary"]
        config = data["config"]
        last = rows[-1] if rows else {}
        drops = summary.get("drop_counts", {})
        avg_candidates = sum(row["candidate_count"] for row in rows) / len(rows) if rows else None
        settings = ", ".join([
            f"k={config.get('m1_k_direct_llm', '-')}",
            f"bo={config.get('max_candidates_per_round', '-')}",
            f"seed={config.get('seed', '-')}",
        ])
        body.append(
            f"<tr data-scope=\"run\" data-series=\"{escape(run['name'])}\">"
            + "".join([
            td(run["name"], "run-name"),
            td(len(rows)),
            td(summary.get("history_size", len(data["series"]["history_points"]))),
            td(_fmt(last.get("pareto_front_hypervolume"))),
            td(_fmt(last.get("best_vina"))),
            td(_fmt(last.get("best_activity"))),
            td(last.get("pareto_front_size", "")),
            td(summary.get("llm_call_count", sum(row["llm_attempts"] for row in rows))),
            td(_fmt(avg_candidates)),
            td(drops.get("duplicate", 0)),
            td(drops.get("invalid", 0)),
            td(drops.get("evaluated", 0)),
            td(summary.get("early_stop_reason") or ""),
            td(settings),
        ]) + "</tr>")
    return (
        "<section class=\"table-section\"><h2>Run Summary</h2>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table></div></section>"
    )


def render_comparison_settings(runs: Sequence[dict[str, Any]]) -> str:
    keys = [
        "method", "init_strategy", "budget", "batch_size", "m1_k_direct_llm",
        "max_candidates_per_round", "ehvi_n_samples", "alpha_base_measure",
        "eta_ehvi_tilt", "llm_max_retries", "seed",
    ]
    headers = ["Run", *keys]
    body = []
    for run in runs:
        config = run["data"]["config"]
        body.append(
            f"<tr data-scope=\"run\" data-series=\"{escape(run['name'])}\">"
            + "".join([
            td(run["name"], "run-name"),
            *[td(config.get(key, "")) for key in keys],
        ]) + "</tr>")
    return (
        "<section class=\"table-section\"><h2>Run Settings</h2>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table></div></section>"
    )


def render_config_summary(config: dict[str, Any], summary: dict[str, Any]) -> str:
    keys = [
        "method", "init_strategy", "budget", "batch_size", "m1_k_direct_llm",
        "max_candidates_per_round", "alpha_base_measure", "eta_ehvi_tilt",
        "ehvi_n_samples", "trajectory_dir",
    ]
    rows = []
    for key in keys:
        if key in config:
            rows.append(f"<tr><th>{escape(key)}</th><td>{escape(str(config[key]))}</td></tr>")
    rows.append(f"<tr><th>drop_counts_total</th><td>{escape(json.dumps(summary.get('drop_counts', {}), sort_keys=True))}</td></tr>")
    return (
        "<section class=\"config\"><h2>Run Settings</h2><table>"
        + "\n".join(rows)
        + "</table></section>"
    )


def td(value: Any, class_name: str = "") -> str:
    attr = f" class=\"{class_name}\"" if class_name else ""
    return f"<td{attr}>{escape(str(value))}</td>"


def final_metric(run: dict[str, Any], key: str) -> float | None:
    rows = run["data"]["series"]["rows"]
    if not rows:
        return None
    return _float_or_none(rows[-1].get(key))


def short_run_name(name: str) -> str:
    prefix = "case2_qwen3_coder_longer_temp07_"
    return name.removeprefix(prefix)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --ink: #172033;
  --muted: #667085;
  --line: #d7dce5;
  --soft: #eef2f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  width: min(1480px, calc(100vw - 40px));
  margin: 28px auto 48px;
}
h1 {
  margin: 0;
  font-size: 28px;
  font-weight: 750;
}
h2 {
  margin: 0 0 12px;
  font-size: 16px;
}
code {
  background: var(--soft);
  padding: 2px 5px;
  border-radius: 4px;
}
.subtitle {
  margin: 6px 0 20px;
  color: var(--muted);
}
.subtitle.compact {
  margin-bottom: 12px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.section-block {
  margin-top: 18px;
}
.section-block > h2 {
  margin: 0 0 4px;
  font-size: 19px;
}
.visibility-panel {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}
.toggle-group {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.toggle-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.toggle-header h2 {
  margin: 0;
}
.toggle-actions {
  display: flex;
  gap: 8px;
}
.toggle-actions button {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #f9fafb;
  color: var(--ink);
  cursor: pointer;
  font: inherit;
  padding: 4px 8px;
}
.toggle-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 12px;
}
.toggle-item {
  align-items: center;
  color: #344054;
  display: grid;
  font-size: 12px;
  gap: 7px;
  grid-template-columns: auto auto minmax(0, 1fr);
  min-width: 0;
}
.toggle-item span:last-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.swatch {
  border-radius: 999px;
  display: inline-block;
  height: 10px;
  width: 10px;
}
.hidden-series {
  display: none !important;
}
.metric, .plot-card, .table-section, .config {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 1px 2px rgb(16 24 40 / 0.04);
}
.metric {
  padding: 14px;
}
.metric-label {
  color: var(--muted);
  font-size: 12px;
}
.metric-value {
  margin-top: 4px;
  font-size: 18px;
  font-weight: 720;
  overflow-wrap: anywhere;
}
.plot-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.plot-card {
  padding: 14px;
}
.plot-card.wide {
  grid-column: 1 / -1;
}
svg {
  display: block;
  width: 100%;
  height: auto;
}
.axis {
  stroke: #667085;
  stroke-width: 1.2;
}
.grid {
  stroke: #e4e7ec;
  stroke-width: 1;
}
.line {
  fill: none;
  stroke-width: 2.3;
}
.std-band {
  opacity: 0.18;
  stroke: none;
}
.tick, .axis-label {
  fill: #667085;
  font-size: 11px;
}
.legend {
  font-size: 12px;
  font-weight: 650;
}
.point-label {
  fill: #344054;
  font-size: 11px;
  font-weight: 650;
}
.scatter-point {
  opacity: 0.72;
}
.table-section, .config {
  margin-top: 16px;
  padding: 14px;
}
.table-wrap {
  overflow: auto;
  max-height: 620px;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 8px 9px;
  text-align: right;
  vertical-align: top;
  white-space: nowrap;
}
th {
  position: sticky;
  top: 0;
  background: #f9fafb;
  color: #475467;
  font-size: 12px;
  font-weight: 700;
}
th:nth-child(2), td.smiles {
  text-align: left;
}
td.run-name {
  color: var(--ink);
  font-weight: 650;
  text-align: left;
}
td.members {
  max-width: 520px;
  text-align: left;
  white-space: normal;
}
td.smiles {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.config table th {
  text-align: left;
  width: 240px;
}
.config table td {
  text-align: left;
  white-space: normal;
}
@media (max-width: 1100px) {
  .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .visibility-panel { grid-template-columns: 1fr; }
  .plot-grid { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  main { width: min(100vw - 24px, 1480px); }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .toggle-list { grid-template-columns: 1fr; }
}
"""


JS = """
document.addEventListener("DOMContentLoaded", () => {
  const toggles = Array.from(document.querySelectorAll(".series-toggle"));

  function applyToggle(toggle) {
    const scope = toggle.dataset.scope;
    const series = toggle.dataset.series;
    const visible = toggle.checked;
    document.querySelectorAll("[data-scope][data-series]").forEach((node) => {
      if (node.classList.contains("series-toggle")) {
        return;
      }
      if (node.dataset.scope === scope && node.dataset.series === series) {
        node.classList.toggle("hidden-series", !visible);
      }
    });
  }

  toggles.forEach((toggle) => {
    toggle.addEventListener("change", () => applyToggle(toggle));
    applyToggle(toggle);
  });

  document.querySelectorAll("[data-toggle-scope][data-toggle-state]").forEach((button) => {
    button.addEventListener("click", () => {
      const scope = button.dataset.toggleScope;
      const checked = button.dataset.toggleState === "show";
      toggles
        .filter((toggle) => toggle.dataset.scope === scope)
        .forEach((toggle) => {
          toggle.checked = checked;
          applyToggle(toggle);
        });
    });
  });
});
"""


if __name__ == "__main__":
    raise SystemExit(main())
