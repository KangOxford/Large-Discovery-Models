#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import importlib
import json
import math
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAILURE_STATUSES = {
    "crash",
    "evaluation_error",
    "generation_error",
    "score_missing",
    "timeout",
}


@dataclass
class RunData:
    path: Path
    name: str
    method: str
    score_key: str
    minimize: bool
    failure_score: float | None
    best_state_id: str | None
    best_score: float | None
    states: list[dict[str, Any]]
    source: str


@dataclass(frozen=True)
class ColumnWidths:
    state: int
    parent: int
    depth: int
    status: int
    score: int
    surrogate: int
    logprob: int


@dataclass
class PrettyNode:
    value: str
    children: list["PrettyNode"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render TTS search runs as a compact terminal tree.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="TTS/runs",
        type=Path,
        help="Run directory, summary.json, manifest.jsonl, or parent directory containing runs.",
    )
    parser.add_argument(
        "--run",
        default="",
        help="Only show run directories whose name contains this text.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Only show the most recently modified run.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum number of runs to display after filtering. 0 means all.",
    )
    parser.add_argument(
        "--score-key",
        default="",
        help="Override the metric name used when displaying scores.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="For model-based runs, render only one outer iteration tree. 0 means all iterations.",
    )
    parser.add_argument(
        "--max-action-chars",
        type=int,
        default=120,
        help="Maximum characters shown for each action summary.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Append saved state errors to each tree line when present.",
    )
    parser.add_argument(
        "--no-logprobs",
        action="store_true",
        help="Hide saved LLM logprob summaries in tree nodes.",
    )
    parser.add_argument(
        "--no-surrogate",
        action="store_true",
        help="Hide saved model-based surrogate summaries in tree nodes.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Render the tree shape with PrettyPrintTree instead of the aligned ASCII table tree.",
    )
    parser.add_argument(
        "--pretty-trim",
        type=int,
        default=0,
        help=(
            "Optional PrettyPrintTree trim length for each rendered line. "
            "0 leaves wrapping/trimming to --pretty-action-width."
        ),
    )
    parser.add_argument(
        "--pretty-action-width",
        type=int,
        default=48,
        help="Wrap action/error text to this width inside each PrettyPrintTree node.",
    )
    parser.add_argument(
        "--pretty-orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="PrettyPrintTree layout orientation. Vertical is usually clearer for multi-line state nodes.",
    )
    parser.add_argument(
        "--pretty-border",
        action="store_true",
        help="Draw PrettyPrintTree borders around multi-line state nodes.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Refresh the terminal view every SECONDS for an active run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pretty:
        pretty_tree_class, import_error = import_pretty_print_tree()
        if pretty_tree_class is None:
            print(
                "error: --pretty requires PrettyPrintTree. Install it with: pip install PrettyPrintTree",
                file=sys.stderr,
            )
            if import_error:
                print(f"import error: {import_error}", file=sys.stderr)
            return 2
        args.pretty_tree_class = pretty_tree_class
        args.pretty_supports_options = detect_pretty_options(pretty_tree_class)

    if args.watch and args.watch < 0.2:
        args.watch = 0.2

    while True:
        output = render_path(args)
        if args.watch:
            clear_terminal()
        print(output, end="" if output.endswith("\n") else "\n")
        sys.stdout.flush()
        if not args.watch:
            break
        time.sleep(args.watch)
    return 0


def clear_terminal() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def import_pretty_print_tree() -> tuple[Any | None, str]:
    try:
        from PrettyPrint import PrettyPrintTree
    except ModuleNotFoundError as exc:
        if missing_cmd2_ansi(exc):
            install_cmd2_ansi_shim()
            clear_prettyprint_modules()
            try:
                from PrettyPrint import PrettyPrintTree
            except Exception as retry_exc:
                return None, f"{type(retry_exc).__name__}: {retry_exc}"
            return PrettyPrintTree, ""
        return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return PrettyPrintTree, ""


def detect_pretty_options(pretty_tree_class: Any) -> set[str]:
    import inspect

    try:
        signature = inspect.signature(pretty_tree_class.__init__)
    except (TypeError, ValueError):
        return set()
    return set(signature.parameters)


def missing_cmd2_ansi(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", "") or ""
    return missing_name in {"cmd2", "cmd2.ansi"} or "cmd2.ansi" in str(exc)


def install_cmd2_ansi_shim() -> None:
    import types

    ansi_module = types.ModuleType("cmd2.ansi")
    ansi_module.style_aware_wcswidth = style_aware_wcswidth
    sys.modules["cmd2.ansi"] = ansi_module

    try:
        cmd2_module = importlib.import_module("cmd2")
    except ModuleNotFoundError:
        cmd2_module = types.ModuleType("cmd2")
        cmd2_module.__path__ = []
        sys.modules["cmd2"] = cmd2_module
    setattr(cmd2_module, "ansi", ansi_module)


def clear_prettyprint_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "PrettyPrint" or module_name.startswith("PrettyPrint."):
            del sys.modules[module_name]


def style_aware_wcswidth(text: str) -> int:
    return len(strip_ansi(text))


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def render_path(args: argparse.Namespace) -> str:
    run_paths = discover_runs(args.path)
    if args.run:
        run_paths = [path for path in run_paths if args.run in path.name]
    run_paths = sorted(run_paths, key=run_mtime, reverse=True)
    if args.latest and run_paths:
        run_paths = run_paths[:1]
    if args.max_runs > 0:
        run_paths = run_paths[: args.max_runs]

    if not run_paths:
        return f"No TTS search runs found under {args.path}\n"

    rendered: list[str] = []
    for index, run_path in enumerate(run_paths):
        run = load_run(run_path)
        if run is None:
            continue
        if args.score_key:
            run.score_key = args.score_key
        if index:
            rendered.append("")
        rendered.extend(render_run(run, args))
    if not rendered:
        return f"No readable TTS search runs found under {args.path}\n"
    return "\n".join(rendered) + "\n"


def discover_runs(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file() and path.name in {"summary.json", "manifest.jsonl"}:
        return [path.parent]
    if is_run_dir(path):
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    return [child for child in path.iterdir() if child.is_dir() and is_run_dir(child)]


def is_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (
            (path / "summary.json").exists()
            or (path / "manifest.jsonl").exists()
            or (path / "states").is_dir()
        )
    )


def run_mtime(path: Path) -> float:
    candidates = [path / "summary.json", path / "manifest.jsonl", path]
    return max(candidate.stat().st_mtime for candidate in candidates if candidate.exists())


def load_run(path: Path) -> RunData | None:
    summary_path = path / "summary.json"
    manifest_path = path / "manifest.jsonl"
    summary = load_json(summary_path) if summary_path.exists() else None

    if summary is not None:
        args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
        states = summary.get("states") if isinstance(summary.get("states"), list) else []
        return RunData(
            path=path,
            name=path.name,
            method=str(summary.get("method") or args.get("method") or "unknown"),
            score_key=str(summary.get("score_key") or args.get("score_key") or "score"),
            minimize=bool(summary.get("minimize", not bool(args.get("maximize", False)))),
            failure_score=as_float(args.get("failure_score")),
            best_state_id=as_optional_str(summary.get("best_state_id")),
            best_score=as_float(summary.get("best_score")),
            states=[state for state in states if isinstance(state, dict)],
            source="summary.json",
        )

    states = load_manifest(manifest_path) if manifest_path.exists() else load_state_metas(path)
    if not states:
        return None
    score_key = infer_score_key(states)
    minimize = True
    best_state_id = find_best_state_id(states, minimize=minimize)
    best_score = state_score(next((state for state in states if state.get("state_id") == best_state_id), {}), score_key)
    return RunData(
        path=path,
        name=path.name,
        method=infer_method(path.name),
        score_key=score_key,
        minimize=minimize,
        failure_score=None,
        best_state_id=best_state_id,
        best_score=best_score,
        states=states,
        source="manifest.jsonl" if manifest_path.exists() else "state meta files",
    )


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def load_manifest(path: Path) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(state, dict):
            continue
        state_id = as_optional_str(state.get("state_id"))
        if state_id is None:
            continue
        if state_id not in by_id:
            order.append(state_id)
        by_id[state_id] = state
    return [by_id[state_id] for state_id in order]


def load_state_metas(path: Path) -> list[dict[str, Any]]:
    states_dir = path / "states"
    if not states_dir.exists():
        return []
    states = []
    for meta_path in sorted(states_dir.glob("state_*/meta.json")):
        state = load_json(meta_path)
        if state is not None:
            states.append(state)
    return states


def infer_method(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else "unknown"


def find_best_state_id(states: list[dict[str, Any]], *, minimize: bool) -> str | None:
    scored = [
        (as_float(state.get("score")), as_optional_str(state.get("state_id")))
        for state in states
    ]
    scored = [(score, state_id) for score, state_id in scored if score is not None and state_id]
    if not scored:
        return None
    score, state_id = sorted(scored, key=lambda item: item[0], reverse=not minimize)[0]
    return state_id


def infer_score_key(states: list[dict[str, Any]]) -> str:
    preferred = ["val_bpb", "accuracy", "loss", "score"]
    metric_counts: dict[str, int] = {}
    for state in states:
        metrics = state.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if as_float(value) is not None:
                metric_counts[str(key)] = metric_counts.get(str(key), 0) + 1
    for key in preferred:
        if key in metric_counts:
            return key
    if metric_counts:
        return sorted(metric_counts, key=lambda key: (-metric_counts[key], key))[0]
    return "score"


def filter_model_based_iteration(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    iteration: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    record = find_iteration_record(run.path, iteration)
    if record is None:
        return filter_iteration_by_metric(state_by_id, iteration), None
    root_id = as_optional_str(record.get("root_state_id"))
    if root_id is None or root_id not in state_by_id:
        return {}, record
    keep_ids = descendants_of(root_id, state_by_id)
    return {state_id: state_by_id[state_id] for state_id in keep_ids if state_id in state_by_id}, record


def find_iteration_record(path: Path, iteration: int) -> dict[str, Any] | None:
    summary = load_json(path / "model_based_summary.json")
    if summary is None:
        return None
    records = summary.get("iterations")
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and int(record.get("iteration") or -1) == iteration:
            return record
    return None


def filter_iteration_by_metric(
    state_by_id: dict[str, dict[str, Any]],
    iteration: int,
) -> dict[str, dict[str, Any]]:
    filtered = {
        state_id: state
        for state_id, state in state_by_id.items()
        if state_iteration(state) == iteration
    }
    root_ids = {
        as_optional_str(state.get("parent_id"))
        for state in filtered.values()
        if as_optional_str(state.get("parent_id")) in state_by_id
    }
    for root_id in root_ids:
        if root_id is not None:
            filtered[root_id] = state_by_id[root_id]
    return filtered


def state_iteration(state: dict[str, Any]) -> int | None:
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("model_based_iteration")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def descendants_of(root_id: str, state_by_id: dict[str, dict[str, Any]]) -> list[str]:
    children: dict[str, list[str]] = {}
    for state_id, state in state_by_id.items():
        parent_id = as_optional_str(state.get("parent_id"))
        if parent_id is not None:
            children.setdefault(parent_id, []).append(state_id)
    keep: list[str] = []
    stack = [root_id]
    seen: set[str] = set()
    while stack:
        state_id = stack.pop()
        if state_id in seen:
            continue
        seen.add(state_id)
        keep.append(state_id)
        stack.extend(reversed(sorted(children.get(state_id, []), key=state_id_sort_key)))
    return keep


def iteration_scoped_run(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    record: dict[str, Any] | None,
) -> RunData:
    best_state_id = None
    best_score = None
    if record is not None:
        best_state_id = as_optional_str(record.get("selected_state_id"))
        best_score = as_float(record.get("selected_real_score"))
    if best_state_id is None or best_state_id not in state_by_id or best_score is None:
        best_state_id = find_best_state_id(list(state_by_id.values()), minimize=run.minimize)
        best_state = state_by_id.get(best_state_id or "")
        best_score = None if best_state is None else state_score(best_state, run.score_key)
    return RunData(
        path=run.path,
        name=run.name,
        method=run.method,
        score_key=run.score_key,
        minimize=run.minimize,
        failure_score=run.failure_score,
        best_state_id=best_state_id,
        best_score=best_score,
        states=list(state_by_id.values()),
        source=run.source,
    )


def format_iteration_header(
    iteration: int,
    record: dict[str, Any] | None,
    state_by_id: dict[str, dict[str, Any]],
    run: RunData,
) -> str:
    if record is None:
        return f"  iteration={iteration} states={len(state_by_id)} source=state metrics"
    parts = [
        f"iteration={iteration}",
        f"root={record.get('root_state_id')}",
        f"selected={record.get('selected_state_id')}",
        f"generated={record.get('generated_count')}",
    ]
    selected_score = as_float(record.get("selected_real_score"))
    if selected_score is not None:
        parts.append(f"{run.score_key}={format_float(selected_score)}")
    best_after = as_float(record.get("best_score_after_iteration"))
    if best_after is not None:
        parts.append(f"best_after={format_float(best_after)}")
    improved = record.get("selected_improved_previous_best")
    if improved is not None:
        parts.append(f"improved={bool(improved)}")
    return "  " + " ".join(parts)


def render_run(run: RunData, args: argparse.Namespace) -> list[str]:
    states = sorted(run.states, key=state_sort_key)
    state_by_id = {
        str(state["state_id"]): state
        for state in states
        if "state_id" in state and state.get("state_id") is not None
    }
    iteration_record: dict[str, Any] | None = None
    display_run = run
    if args.iteration > 0:
        state_by_id, iteration_record = filter_model_based_iteration(run, state_by_id, args.iteration)
        display_run = iteration_scoped_run(run, state_by_id, iteration_record)
    children: dict[str | None, list[str]] = {}
    for state in states:
        state_id = as_optional_str(state.get("state_id"))
        if state_id is None or state_id not in state_by_id:
            continue
        parent_id = as_optional_str(state.get("parent_id"))
        if parent_id is not None and parent_id not in state_by_id:
            parent_id = None
        children.setdefault(parent_id, []).append(state_id)
    for child_ids in children.values():
        child_ids.sort(key=state_id_sort_key)

    direction = "minimize" if run.minimize else "maximize"
    best_score = format_best_score(display_run)
    lines = [
        f"Run: {run.name}",
        (
            f"  method={display_run.method} source={display_run.source} states={len(state_by_id)} "
            f"score_key={display_run.score_key} direction={direction} best={best_score}"
        ),
    ]
    if args.iteration > 0:
        lines.append(format_iteration_header(args.iteration, iteration_record, state_by_id, display_run))
    if args.pretty:
        lines.extend(render_pretty_forest(display_run, state_by_id, children, args))
        lines.extend(render_optimal_path(display_run, state_by_id, args))
        return lines

    widths = measure_columns(display_run, state_by_id, args)

    visited: set[str] = set()
    roots = children.get(None, [])
    for index, root_id in enumerate(roots):
        render_tree_node(
            display_run,
            state_by_id,
            children,
            root_id,
            prefix="",
            is_last=index == len(roots) - 1,
            is_root=True,
            visited=visited,
            args=args,
            widths=widths,
            lines=lines,
        )
    for state_id in sorted(set(state_by_id) - visited, key=state_id_sort_key):
        lines.append(format_state_line(display_run, state_by_id, state_by_id[state_id], args, widths))
    lines.extend(render_optimal_path(display_run, state_by_id, args))
    return lines


def render_optimal_path(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    best_state_id = run.best_state_id
    if best_state_id is None or best_state_id not in state_by_id:
        return ["", "Optimal path: no evaluated best state available."]

    path = trace_state_path(best_state_id, state_by_id)
    if not path:
        return ["", f"Optimal path: best state {best_state_id} is not connected to the displayed tree."]

    best_state = path[-1]
    lines = [
        "",
        (
            f"Optimal path: from root state to {best_state_id} "
            f"with {format_state_score(run, best_state)}."
        ),
    ]
    if len(path) == 1:
        lines.append("  No actions needed; the root state is the best state.")
        return lines

    for index, state in enumerate(path[1:], start=1):
        action = summarize_action(run, state_by_id, state, max_chars=args.max_action_chars)
        state_id = str(state.get("state_id", "unknown"))
        score = format_state_score(run, state)
        surrogate = "" if args.no_surrogate else format_state_surrogate_summary(state)
        logprob = "" if args.no_logprobs else format_state_logprob_summary(state)
        metadata = f"reach {state_id} with {score}"
        if surrogate:
            metadata += f", {surrogate}"
        if logprob:
            metadata += f", {logprob}"
        lines.extend(format_optimal_path_step(index, action, metadata))
    return lines


def trace_state_path(best_state_id: str, state_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    state_id: str | None = best_state_id
    while state_id is not None:
        if state_id in seen:
            return []
        seen.add(state_id)
        state = state_by_id.get(state_id)
        if state is None:
            return []
        path.append(state)
        state_id = as_optional_str(state.get("parent_id"))
    path.reverse()
    return path


def format_optimal_path_step(index: int, action: str, metadata: str) -> list[str]:
    prefix = f"  {index}. "
    width = 100
    wrapped = textwrap.wrap(
        f"Perform action {index}: {action}",
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        wrapped = [prefix.rstrip()]
    wrapped.append(" " * len(prefix) + f"=> {metadata}")
    return wrapped


def render_pretty_forest(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    children: dict[str | None, list[str]],
    args: argparse.Namespace,
) -> list[str]:
    pretty_tree_class = args.pretty_tree_class
    options: dict[str, Any] = {
        "return_instead_of_print": True,
        "color": "",
        "show_newline_literal": False,
    }
    if "border" in args.pretty_supports_options:
        options["border"] = args.pretty_border
    if "orientation" in args.pretty_supports_options:
        options["orientation"] = pretty_orientation(pretty_tree_class, args.pretty_orientation)
    if args.pretty_trim > 0:
        options["trim"] = args.pretty_trim
    printer = pretty_tree_class(
        lambda node: node.children,
        lambda node: node.value,
        **options,
    )

    visited: set[str] = set()
    roots = children.get(None, [])
    rendered: list[str] = []
    for root_id in roots:
        root = build_pretty_node(run, state_by_id, children, root_id, args, visited)
        tree = printer(root)
        rendered.extend(str(tree).rstrip("\n").splitlines())

    for state_id in sorted(set(state_by_id) - visited, key=state_id_sort_key):
        root = build_pretty_node(run, state_by_id, children, state_id, args, visited)
        tree = printer(root)
        rendered.extend(str(tree).rstrip("\n").splitlines())
    return rendered


def pretty_orientation(pretty_tree_class: Any, name: str) -> Any:
    if name == "horizontal":
        return getattr(pretty_tree_class, "Horizontal", True)
    return getattr(pretty_tree_class, "Vertical", False)


def build_pretty_node(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    children: dict[str | None, list[str]],
    state_id: str,
    args: argparse.Namespace,
    visited: set[str],
) -> PrettyNode:
    visited.add(state_id)
    state = state_by_id[state_id]
    child_nodes = [
        build_pretty_node(run, state_by_id, children, child_id, args, visited)
        for child_id in children.get(state_id, [])
        if child_id not in visited
    ]
    return PrettyNode(
        value=format_pretty_node_value(run, state_by_id, state, args),
        children=child_nodes,
    )


def format_pretty_node_value(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    state: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    header = " ".join(
        [
            format_state_label(run, state),
            format_depth_label(state),
            format_status_label(state),
        ]
    )
    score = format_state_score(run, state)
    surrogate = "" if args.no_surrogate else format_state_surrogate_summary(state)
    logprob = "" if args.no_logprobs else format_state_logprob_summary(state)
    action = summarize_action(run, state_by_id, state, max_chars=args.max_action_chars)
    lines = [header, score]
    if surrogate:
        lines.append(surrogate)
    if logprob:
        lines.append(logprob)
    lines.extend(wrap_pretty_field("action", action, args.pretty_action_width))
    if args.show_errors and state.get("error"):
        error = shorten(str(state.get("error")), args.max_action_chars)
        lines.extend(wrap_pretty_field("error", error, args.pretty_action_width))
    return "\n".join(lines)


def wrap_pretty_field(label: str, text: str, width: int) -> list[str]:
    width = max(20, int(width))
    prefix = f"{label}: "
    wrapped = textwrap.wrap(
        text,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [prefix.rstrip()]


def render_tree_node(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    children: dict[str | None, list[str]],
    state_id: str,
    *,
    prefix: str,
    is_last: bool,
    is_root: bool,
    visited: set[str],
    args: argparse.Namespace,
    widths: ColumnWidths,
    lines: list[str],
) -> None:
    if state_id in visited:
        return
    visited.add(state_id)
    connector = "" if is_root else ("`-- " if is_last else "|-- ")
    lines.append(prefix + connector + format_state_line(run, state_by_id, state_by_id[state_id], args, widths))

    child_ids = children.get(state_id, [])
    child_prefix = prefix if is_root else prefix + ("    " if is_last else "|   ")
    for index, child_id in enumerate(child_ids):
        render_tree_node(
            run,
            state_by_id,
            children,
            child_id,
            prefix=child_prefix,
            is_last=index == len(child_ids) - 1,
            is_root=False,
            visited=visited,
            args=args,
            widths=widths,
            lines=lines,
        )


def format_state_line(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    state: dict[str, Any],
    args: argparse.Namespace,
    widths: ColumnWidths,
) -> str:
    state_label = format_state_label(run, state)
    parent = format_parent_label(state)
    depth = format_depth_label(state)
    status = format_status_label(state)
    score = format_state_score(run, state)
    surrogate_part = ""
    if widths.surrogate > 0 and not args.no_surrogate:
        surrogate = format_state_surrogate_summary(state) or "sg=-"
        surrogate_part = f"{surrogate:<{widths.surrogate}}  "
    logprob_part = ""
    if widths.logprob > 0 and not args.no_logprobs:
        logprob = format_state_logprob_summary(state) or "lp=-"
        logprob_part = f"{logprob:<{widths.logprob}}  "
    action = summarize_action(run, state_by_id, state, max_chars=args.max_action_chars)
    line = (
        f"{state_label:<{widths.state}}  "
        f"{parent:<{widths.parent}}  "
        f"{depth:<{widths.depth}}  "
        f"{status:<{widths.status}}  "
        f"{score:<{widths.score}}  "
        f"{surrogate_part}"
        f"{logprob_part}"
        f"action: {action}"
    )
    if args.show_errors and state.get("error"):
        line += f" error: {shorten(str(state.get('error')), args.max_action_chars)}"
    return line


def measure_columns(run: RunData, state_by_id: dict[str, dict[str, Any]], args: argparse.Namespace) -> ColumnWidths:
    states = list(state_by_id.values())
    state_width = max([len(format_state_label(run, state)) for state in states] or [len("state")])
    parent_width = max([len(format_parent_label(state)) for state in states] or [len("parent=-")])
    depth_width = max([len(format_depth_label(state)) for state in states] or [len("d=?")])
    status_width = max([len(format_status_label(state)) for state in states] or [len("[status]")])
    score_width = max([len(format_state_score(run, state)) for state in states] or [len(run.score_key) + 2])
    surrogate_values = [] if args.no_surrogate else [format_state_surrogate_summary(state) for state in states]
    surrogate_values = [value for value in surrogate_values if value]
    surrogate_width = max([len("sg=-"), *(len(value) for value in surrogate_values)] if surrogate_values else [0])
    logprob_values = [] if args.no_logprobs else [format_state_logprob_summary(state) for state in states]
    logprob_values = [value for value in logprob_values if value]
    logprob_width = max([len("lp=-"), *(len(value) for value in logprob_values)] if logprob_values else [0])
    return ColumnWidths(
        state=state_width,
        parent=parent_width,
        depth=depth_width,
        status=status_width,
        score=score_width,
        surrogate=surrogate_width,
        logprob=logprob_width,
    )


def format_state_label(run: RunData, state: dict[str, Any]) -> str:
    state_id = str(state.get("state_id", "unknown"))
    return state_id + ("*" if state_id == run.best_state_id else "")


def format_parent_label(state: dict[str, Any]) -> str:
    return f"parent={as_optional_str(state.get('parent_id')) or '-'}"


def format_depth_label(state: dict[str, Any]) -> str:
    depth = state.get("depth")
    return f"d={depth if depth is not None else '?'}"


def format_status_label(state: dict[str, Any]) -> str:
    return f"[{state.get('status') or 'unknown'}]"


def format_best_score(run: RunData) -> str:
    if run.best_state_id is None:
        return "none"
    best_score = run.best_score
    if best_score is None:
        state = next((item for item in run.states if item.get("state_id") == run.best_state_id), None)
        best_score = None if state is None else as_float(state.get("score"))
    if best_score is None:
        return run.best_state_id
    return f"{run.best_state_id} {run.score_key}={format_float(best_score)}"


def format_state_score(run: RunData, state: dict[str, Any]) -> str:
    score = state_score(state, run.score_key)
    if score is None:
        return f"{run.score_key}=-"
    text = f"{run.score_key}={format_float(score)}"
    status = str(state.get("status") or "")
    if (
        run.failure_score is not None
        and status in FAILURE_STATUSES
        and math.isclose(score, run.failure_score, rel_tol=0.0, abs_tol=max(1e-9, abs(run.failure_score) * 1e-12))
    ):
        text += " (failure)"
    return text


def format_state_surrogate_summary(state: dict[str, Any]) -> str:
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        return ""
    parts = []
    score = as_float(metrics.get("surrogate_score"))
    pred = as_float(metrics.get("surrogate_pred"))
    std = as_float(metrics.get("surrogate_std"))
    ei = as_float(metrics.get("surrogate_ei"))
    if score is not None:
        parts.append(f"sg={score:.4g}")
    if pred is not None:
        parts.append(f"pred={pred:.4g}")
    if std is not None:
        parts.append(f"std={std:.3g}")
    if ei is not None:
        parts.append(f"ei={ei:.3g}")
    return " ".join(parts)


def format_state_logprob_summary(state: dict[str, Any]) -> str:
    summary = state_logprob_summary(state)
    if not summary:
        return ""
    parts = []
    mean_logprob = summary.get("mean_logprob")
    perplexity = summary.get("perplexity")
    token_count = summary.get("token_count")
    if as_float(mean_logprob) is not None:
        parts.append(f"lp={float(mean_logprob):.4g}")
    if as_float(perplexity) is not None:
        parts.append(f"ppl={float(perplexity):.4g}")
    if as_float(token_count) is not None:
        parts.append(f"lptok={int(float(token_count))}")
    return " ".join(parts)


def state_logprob_summary(state: dict[str, Any]) -> dict[str, Any]:
    summary = state.get("logprob_summary")
    if isinstance(summary, dict) and summary:
        return summary

    edits = state.get("edits")
    if not isinstance(edits, list):
        return {}
    summaries = [
        edit.get("logprob_summary")
        for edit in edits
        if isinstance(edit, dict) and isinstance(edit.get("logprob_summary"), dict) and edit.get("logprob_summary")
    ]
    if not summaries:
        return {}
    return aggregate_logprob_summaries(summaries) or summaries[-1]


def aggregate_logprob_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) == 1:
        return summaries[0]

    token_count = 0.0
    sum_logprob = 0.0
    min_logprobs = []
    max_logprobs = []
    for summary in summaries:
        count = as_float(summary.get("token_count"))
        if count is None or count <= 0:
            return {}
        total = as_float(summary.get("sum_logprob"))
        if total is None:
            mean = as_float(summary.get("mean_logprob"))
            if mean is None:
                return {}
            total = mean * count
        token_count += count
        sum_logprob += total
        min_logprob = as_float(summary.get("min_logprob"))
        max_logprob = as_float(summary.get("max_logprob"))
        if min_logprob is not None:
            min_logprobs.append(min_logprob)
        if max_logprob is not None:
            max_logprobs.append(max_logprob)

    if token_count <= 0:
        return {}
    mean_logprob = sum_logprob / token_count
    aggregate: dict[str, Any] = {
        "token_count": int(token_count),
        "sum_logprob": sum_logprob,
        "mean_logprob": mean_logprob,
        "mean_probability": math.exp(mean_logprob),
        "perplexity": math.exp(-mean_logprob),
    }
    if min_logprobs:
        aggregate["min_logprob"] = min(min_logprobs)
    if max_logprobs:
        aggregate["max_logprob"] = max(max_logprobs)
    return aggregate


def state_score(state: dict[str, Any], score_key: str) -> float | None:
    metrics = state.get("metrics")
    if isinstance(metrics, dict):
        metric_score = as_float(metrics.get(score_key))
        if metric_score is not None:
            return metric_score
    return as_float(state.get("score"))


def summarize_action(
    run: RunData,
    state_by_id: dict[str, dict[str, Any]],
    state: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    parent_id = as_optional_str(state.get("parent_id"))
    if parent_id is None:
        return "seed train.py"

    operation_summary = summarize_operations(state)
    if operation_summary:
        return shorten(operation_summary, max_chars)

    description = clean_text(state.get("description"))
    if description:
        return shorten(description, max_chars)

    edit_summary = summarize_edits(state)
    if edit_summary:
        return shorten(edit_summary, max_chars)

    response_summary = summarize_response_artifacts(run, state)
    if response_summary:
        return shorten(response_summary, max_chars)

    patch_summary = summarize_patch_artifacts(run, state)
    if patch_summary:
        return shorten(patch_summary, max_chars)

    parent = state_by_id.get(parent_id)
    diff_summary = summarize_parent_child_diff(parent, state)
    if diff_summary:
        return shorten(diff_summary, max_chars)

    if str(state.get("status")) == "generation_error":
        return "generation failed before applying an edit"
    return "no code changes detected"


def summarize_operations(state: dict[str, Any]) -> str:
    metrics = state.get("metrics")
    operations = None
    if isinstance(metrics, dict) and isinstance(metrics.get("operations"), list):
        operations = metrics.get("operations")
    if operations is None and isinstance(state.get("edits"), list):
        operations = []
        for edit in state.get("edits", []):
            if isinstance(edit, dict) and isinstance(edit.get("applied_operations"), list):
                operations.extend(edit["applied_operations"])
    if not isinstance(operations, list) or not operations:
        return ""
    parts = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        name = clean_text(operation.get("name"))
        if not name:
            continue
        old_value = operation.get("old_value")
        new_value = operation.get("new_value")
        parts.append(f"set {name} {old_value!r} -> {new_value!r}")
    return "; ".join(parts)


def summarize_edits(state: dict[str, Any]) -> str:
    edits = state.get("edits")
    if not isinstance(edits, list):
        return ""

    descriptions = [
        clean_text(edit.get("description"))
        for edit in edits
        if isinstance(edit, dict) and clean_text(edit.get("description"))
    ]
    if descriptions:
        return "; ".join(descriptions)

    parts = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        detected = edit.get("detected_edits")
        if isinstance(detected, list):
            for block in detected:
                if isinstance(block, dict):
                    parts.append(summarize_line_delta(block.get("search"), block.get("replace")))
        if len(parts) >= 3:
            break
    return "; ".join(part for part in parts if part)


def summarize_response_artifacts(run: RunData, state: dict[str, Any]) -> str:
    paths = []
    response_path = as_path(state.get("response_path"))
    if response_path is not None:
        paths.append(response_path)
    workdir = state_workdir(run, state)
    paths.extend(sorted(workdir.glob("response*.md")))

    seen: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        summary = extract_response_summary(text)
        if summary:
            return summary
    return ""


def extract_response_summary(text: str) -> str:
    for pattern in (r"(?im)^summary:\s*(.+)$", r"(?im)^description:\s*(.+)$"):
        match = re.search(pattern, text)
        if match:
            return clean_text(match.group(1))
    return ""


def summarize_patch_artifacts(run: RunData, state: dict[str, Any]) -> str:
    paths = []
    patch_path = as_path(state.get("patch_path"))
    if patch_path is not None:
        paths.append(patch_path)
    workdir = state_workdir(run, state)
    paths.extend(sorted(workdir.glob("patch*.diff")))

    seen: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        summary = summarize_patch_text(text)
        if summary:
            return summary
    return ""


def summarize_parent_child_diff(parent: dict[str, Any] | None, state: dict[str, Any]) -> str:
    if parent is None:
        return ""
    parent_train = as_path(parent.get("train_path"))
    child_train = as_path(state.get("train_path"))
    if parent_train is None or child_train is None or not parent_train.exists() or not child_train.exists():
        return ""
    try:
        parent_text = parent_train.read_text(encoding="utf-8", errors="replace")
        child_text = child_train.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if parent_text == child_text:
        return ""
    diff_lines = difflib.unified_diff(
        parent_text.splitlines(),
        child_text.splitlines(),
        fromfile=f"{parent.get('state_id')}/train.py",
        tofile=f"{state.get('state_id')}/train.py",
        lineterm="",
    )
    return summarize_patch_text("\n".join(diff_lines))


def summarize_patch_text(text: str) -> str:
    removed: list[str] = []
    added: list[str] = []
    for line in text.splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
    return summarize_line_delta("\n".join(removed), "\n".join(added))


def summarize_line_delta(old: Any, new: Any) -> str:
    old_lines = compact_code_lines(str(old or ""))
    new_lines = compact_code_lines(str(new or ""))
    if not old_lines and not new_lines:
        return ""
    if old_lines == new_lines:
        return ""
    if len(old_lines) == 1 and len(new_lines) == 1:
        return f"change {old_lines[0]} -> {new_lines[0]}"
    if not old_lines:
        return f"add {brief_lines(new_lines)}"
    if not new_lines:
        return f"remove {brief_lines(old_lines)}"
    return f"change {len(old_lines)} line(s) -> {len(new_lines)} line(s): {brief_lines(new_lines)}"


def compact_code_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(re.sub(r"\s+", " ", stripped))
    return lines


def brief_lines(lines: list[str], *, limit: int = 3) -> str:
    shown = lines[:limit]
    suffix = "" if len(lines) <= limit else f"; ... +{len(lines) - limit} more"
    return "; ".join(shown) + suffix


def state_workdir(run: RunData, state: dict[str, Any]) -> Path:
    workdir = as_path(state.get("workdir"))
    if workdir is not None:
        return workdir
    state_id = str(state.get("state_id", "unknown"))
    return run.path / "states" / state_id


def state_sort_key(state: dict[str, Any]) -> tuple[int, str]:
    return state_id_sort_key(str(state.get("state_id") or ""))


def state_id_sort_key(state_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", state_id)
    if not match:
        return (10**12, state_id)
    return (int(match.group(1)), state_id)


def as_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def format_float(value: float) -> str:
    return f"{value:.6g}"


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def shorten(text: str, limit: int) -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
