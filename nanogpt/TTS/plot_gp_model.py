#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TTS.run_model_based_search import (
    BufferEntry,
    GPSurrogate,
    OperationSchema,
    as_float,
    choice_values_equal,
    feature_dim,
    finite_score,
    load_buffer,
    load_operation_schema,
    normalize_operation_numeric,
    operation_feature_dim,
    operation_feature_version,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the final learned GP surrogate from a model-based TTS run. "
            "Writes prediction diagnostics, a 2D PCA embedding, and optional schema-parameter slices."
        ),
    )
    parser.add_argument("path", type=Path, help="Model-based run directory, buffer JSONL, or model_based_summary.json.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for PNG/JSON outputs. Default: <run>/gp_plots.")
    parser.add_argument("--schema", type=Path, default=None, help="Operation schema JSON. Defaults to <run>/operation_schema.json.")
    parser.add_argument("--hash-dims", type=int, default=48, help="Feature hash dims for non-operation buffers.")
    parser.add_argument("--score-key", default="val_bpb", help="Score name for labels.")
    parser.add_argument("--maximize", action="store_true", help="Treat score as higher-is-better.")
    parser.add_argument("--gp-lengthscale", type=float, default=1.5)
    parser.add_argument("--gp-noise", type=float, default=1.0e-4)
    parser.add_argument("--prior-score", type=float, default=1.0)
    parser.add_argument("--prior-std", type=float, default=0.15)
    parser.add_argument(
        "--slice-params",
        default="",
        help=(
            "Comma-separated schema parameters for 1D GP slices. "
            "Default: first four schema parameters that can be varied."
        ),
    )
    parser.add_argument("--slice-points", type=int, default=80, help="Points per 1D slice for numeric parameters.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of observed points to list in summary.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir, buffer_path = resolve_input_path(args.path)
    schema = resolve_schema(args.schema, run_dir)
    if schema is not None:
        expected_dim = operation_feature_dim(schema)
        expected_version = operation_feature_version(schema)
    else:
        expected_dim = feature_dim(args.hash_dims)
        expected_version = None
    entries = load_buffer(buffer_path, expected_dim, expected_feature_version=expected_version)
    if not entries:
        raise SystemExit(f"No compatible buffer entries found in {buffer_path}.")

    gp = GPSurrogate(
        entries,
        lengthscale=args.gp_lengthscale,
        noise=args.gp_noise,
        prior_score=args.prior_score,
        prior_std=args.prior_std,
        minimize=not args.maximize,
    )
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (run_dir / "gp_plots") if run_dir is not None else buffer_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = [gp.predict(entry.feature_vector, mode="mean", beta=1.0, xi=0.001) for entry in entries]
    plot_observed_vs_predicted(entries, predictions, gp, out_dir / "observed_vs_predicted.png", args.score_key)
    plot_pca(entries, predictions, out_dir / "feature_pca.png", args.score_key)

    slice_outputs: list[str] = []
    if schema is not None:
        base_params = best_entry_params(entries, minimize=not args.maximize)
        for name in select_slice_params(schema, args.slice_params):
            output_path = out_dir / f"slice_{name}.png"
            plot_schema_slice(
                schema,
                name,
                base_params,
                gp,
                output_path,
                score_key=args.score_key,
                minimize=not args.maximize,
                points=max(2, int(args.slice_points)),
            )
            slice_outputs.append(str(output_path))

    summary = {
        "input": str(args.path),
        "run_dir": None if run_dir is None else str(run_dir),
        "buffer": str(buffer_path),
        "schema": None if schema is None or schema.path is None else str(schema.path),
        "feature_version": expected_version,
        "entries": len(entries),
        "gp": gp.summary(),
        "outputs": {
            "observed_vs_predicted": str(out_dir / "observed_vs_predicted.png"),
            "feature_pca": str(out_dir / "feature_pca.png"),
            "slices": slice_outputs,
        },
        "best_observed": summarize_top_entries(entries, top_k=max(1, int(args.top_k)), minimize=not args.maximize),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary": str(summary_path), "entries": len(entries)}, indent=2))
    return 0


def resolve_input_path(path: Path) -> tuple[Path | None, Path]:
    path = path.expanduser()
    if path.is_dir():
        candidates = [path / "model_based_buffer.jsonl", path / "buffer.jsonl"]
        for candidate in candidates:
            if candidate.exists():
                return path, candidate
        summary = load_json(path / "model_based_summary.json")
        if summary is not None and isinstance(summary.get("run_buffer"), str):
            buffer_path = Path(summary["run_buffer"])
            if buffer_path.exists():
                return path, buffer_path
        raise SystemExit(f"No model_based_buffer.jsonl found under {path}.")
    if path.name == "model_based_summary.json":
        summary = load_json(path)
        if summary is None:
            raise SystemExit(f"Could not read {path}.")
        buffer_value = summary.get("run_buffer") or summary.get("buffer")
        if not isinstance(buffer_value, str):
            raise SystemExit(f"{path} does not contain a buffer path.")
        return path.parent, Path(buffer_value)
    return None, path


def resolve_schema(schema_arg: Path | None, run_dir: Path | None) -> OperationSchema | None:
    schema_path = schema_arg
    if schema_path is None and run_dir is not None and (run_dir / "operation_schema.json").exists():
        schema_path = run_dir / "operation_schema.json"
    if schema_path is None:
        return None
    return load_operation_schema(schema_path, Path.cwd())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def plot_observed_vs_predicted(
    entries: list[BufferEntry],
    predictions: list[Any],
    gp: GPSurrogate,
    path: Path,
    score_key: str,
) -> None:
    observed = np.array([entry.score for entry in entries], dtype=float)
    mean = np.array([pred.mean for pred in predictions], dtype=float)
    std = np.array([pred.std for pred in predictions], dtype=float)
    order = np.argsort(observed)
    lo = float(min(observed.min(), mean.min()))
    hi = float(max(observed.max(), mean.max()))
    pad = max((hi - lo) * 0.08, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(observed, mean, yerr=std, fmt="o", capsize=3, alpha=0.85)
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
    axes[0].set_xlabel(f"Observed {score_key}")
    axes[0].set_ylabel("GP predicted mean")
    axes[0].set_title("Observed vs GP Prediction")
    axes[0].grid(alpha=0.25)

    axes[1].plot(range(1, len(entries) + 1), observed[order], "o-", label="observed")
    axes[1].plot(range(1, len(entries) + 1), mean[order], "s--", label="predicted")
    axes[1].fill_between(
        range(1, len(entries) + 1),
        mean[order] - std[order],
        mean[order] + std[order],
        alpha=0.2,
        label="+/- 1 std",
    )
    axes[1].set_xlabel("Observed points sorted by score")
    axes[1].set_ylabel(score_key)
    axes[1].set_title(format_gp_title(gp.summary()))
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca(entries: list[BufferEntry], predictions: list[Any], path: Path, score_key: str) -> None:
    X = np.array([entry.feature_vector for entry in entries], dtype=float)
    y = np.array([entry.score for entry in entries], dtype=float)
    if len(entries) == 1:
        coords = np.zeros((1, 2), dtype=float)
        explained = [1.0, 0.0]
    else:
        Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        _, singular_values, vt = np.linalg.svd(Xz, full_matrices=False)
        components = vt[:2]
        coords = Xz @ components.T
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
        total_var = float(np.sum(singular_values * singular_values))
        explained = ((singular_values[:2] * singular_values[:2]) / total_var).tolist() if total_var > 0 else [0.0, 0.0]
        if len(explained) == 1:
            explained.append(0.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, s=70, cmap="viridis_r", edgecolor="black", linewidth=0.4)
    for index, entry in enumerate(entries):
        ax.annotate(entry.state_id or str(index + 1), (coords[index, 0], coords[index, 1]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.set_title("Feature PCA Colored by Observed Score")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(score_key)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_schema_slice(
    schema: OperationSchema,
    parameter_name: str,
    base_params: dict[str, Any],
    gp: GPSurrogate,
    path: Path,
    *,
    score_key: str,
    minimize: bool,
    points: int,
) -> None:
    parameter = schema.parameters[parameter_name]
    xs = slice_values(parameter, points)
    means = []
    stds = []
    for value in xs:
        params = dict(base_params)
        params[parameter_name] = value
        vector = vector_from_schema_params(schema, params)
        pred = gp.predict(vector, mode="mean", beta=1.0, xi=0.001)
        means.append(pred.mean)
        stds.append(pred.std)
    means_arr = np.array(means, dtype=float)
    stds_arr = np.array(stds, dtype=float)

    observed_x = []
    observed_y = []
    for entry in gp.entries:
        value = entry.params.get(parameter_name)
        if value is None:
            continue
        observed_x.append(value)
        observed_y.append(entry.score)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if parameter.kind == "choice":
        positions = np.arange(len(xs))
        ax.plot(positions, means_arr, "o-", label="GP mean")
        ax.fill_between(positions, means_arr - stds_arr, means_arr + stds_arr, alpha=0.2, label="+/- 1 std")
        observed_positions = [choice_index(xs, value) for value in observed_x]
        ax.scatter(observed_positions, observed_y, color="black", s=35, alpha=0.75, label="observed")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(value) for value in xs], rotation=25, ha="right")
    else:
        x_arr = np.array(xs, dtype=float)
        ax.plot(x_arr, means_arr, "-", label="GP mean")
        ax.fill_between(x_arr, means_arr - stds_arr, means_arr + stds_arr, alpha=0.2, label="+/- 1 std")
        ax.scatter(observed_x, observed_y, color="black", s=35, alpha=0.75, label="observed")
        if parameter.scale == "log":
            ax.set_xscale("log")
    base_value = base_params.get(parameter_name)
    if base_value is not None:
        if parameter.kind == "choice":
            ax.axvline(choice_index(xs, base_value), color="tab:red", linestyle="--", linewidth=1, label="base")
        else:
            ax.axvline(float(base_value), color="tab:red", linestyle="--", linewidth=1, label="base")
    ax.set_xlabel(parameter_name)
    ax.set_ylabel(f"Predicted {score_key}")
    direction = "lower is better" if minimize else "higher is better"
    ax.set_title(f"GP 1D Slice Around Best Observed Point ({direction})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def slice_values(parameter: Any, points: int) -> list[Any]:
    if parameter.kind == "choice":
        return list(parameter.choices)
    lo = float(parameter.min_value)
    hi = float(parameter.max_value)
    if parameter.kind == "int":
        count = min(max(2, int(points)), int(hi - lo + 1))
        values = np.linspace(lo, hi, count)
        return sorted(set(int(round(value)) for value in values))
    if parameter.scale == "log":
        return np.exp(np.linspace(math.log(lo), math.log(hi), points)).tolist()
    return np.linspace(lo, hi, points).tolist()


def vector_from_schema_params(schema: OperationSchema, params: dict[str, Any]) -> list[float]:
    vector: list[float] = []
    for name, parameter in schema.parameters.items():
        value = params.get(name)
        if value is None:
            value = default_schema_value(parameter)
        if parameter.kind == "choice":
            vector.extend(1.0 if choice_values_equal(value, choice) else 0.0 for choice in parameter.choices)
            vector.append(1.0)
        else:
            number = as_float(value)
            vector.append(0.0 if number is None else normalize_operation_numeric(number, parameter))
            vector.append(1.0 if number is not None else 0.0)
    return vector


def default_schema_value(parameter: Any) -> Any:
    if parameter.kind == "choice":
        return parameter.choices[0]
    lo = float(parameter.min_value)
    hi = float(parameter.max_value)
    if parameter.kind == "int":
        return int(round((lo + hi) / 2.0))
    if parameter.scale == "log":
        return math.exp((math.log(lo) + math.log(hi)) / 2.0)
    return (lo + hi) / 2.0


def choice_index(choices: list[Any], value: Any) -> int:
    for index, choice in enumerate(choices):
        if str(choice) == str(value):
            return index
        left = as_float(choice)
        right = as_float(value)
        if left is not None and right is not None and abs(left - right) <= 1e-9:
            return index
    return 0


def best_entry_params(entries: list[BufferEntry], *, minimize: bool) -> dict[str, Any]:
    entry = sorted(entries, key=lambda item: item.score, reverse=not minimize)[0]
    return dict(entry.params)


def select_slice_params(schema: OperationSchema, requested: str) -> list[str]:
    if requested.strip():
        names = []
        for name in requested.split(","):
            canonical = name.strip().upper()
            if canonical in schema.parameters:
                names.append(canonical)
        return names
    return list(schema.parameters)[:4]


def summarize_top_entries(entries: list[BufferEntry], *, top_k: int, minimize: bool) -> list[dict[str, Any]]:
    ranked = sorted(entries, key=lambda entry: entry.score, reverse=not minimize)[:top_k]
    return [
        {
            "state_id": entry.state_id,
            "score": entry.score,
            "iteration": entry.iteration,
            "train_path": entry.train_path,
            "params": entry.params,
        }
        for entry in ranked
        if finite_score(entry.score)
    ]


def format_gp_title(summary: dict[str, Any]) -> str:
    parts = [f"GP {summary.get('fit_status', 'unknown')} n={summary.get('history_size')}"]
    nlml = as_float(summary.get("nlml_z"))
    rmse = as_float(summary.get("train_rmse"))
    if nlml is not None:
        parts.append(f"nlml={nlml:.4g}")
    if rmse is not None:
        parts.append(f"rmse={rmse:.4g}")
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
