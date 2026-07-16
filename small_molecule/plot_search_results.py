"""Aggregate ``*_seed=*.json`` files and plot summary curves.

Reads every JSON file produced by ``run_search.py`` in ``--input-dir``,
groups them by method (filename pattern: ``<method>_seed=<seed>.json``),
computes a per-seed summary curve (best-so-far for single-objective,
cumulative hypervolume for 2-objective, per-objective best-so-far
for 3+-objective), aggregates mean / std across seeds, and writes:

    * ``--output`` (figure, default ``<input-dir>/summary.<fmt>``)
    * ``<output-stem>.csv`` (combined mean/std CSV)

The plot dispatch is controlled by ``config.n_objectives`` in each
JSON's ``config`` echo:

    * ``n_objectives == 1`` — best-so-far curve (legacy behaviour)
    * ``n_objectives == 2`` — cumulative hypervolume curve only
      (no 2D Pareto scatter; the HV curve is the canonical view for
      2-objective runs)
    * ``n_objectives >= 3`` — per-objective best-so-far curves
      (graceful degradation; HV is not implemented for n_obj >= 3)

Usage::

    python plot_search_results.py --input-dir output/bo \\
        --ref-point 0,5 \\
        --output output/bo/summary --figure-format png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger("plot_search_results")

METHOD_COLORS = {
    "random": "tab:gray",
    "random-best": "tab:blue",
    "bo-tanimoto": "tab:green",
    "bo-strkernel": "tab:orange",
    "bo-tanimoto-ldm": "tab:red",
    "bo-strkernel-ldm": "tab:purple",
}
METHOD_LABELS = {
    "random": "Random (uniform expansion)",
    "random-best": "Random (Chebyshev-ParEGO expansion)",
    "bo-tanimoto": "BO (Tanimoto fingerprint)",
    "bo-strkernel": "BO (Subsequence string kernel)",
    "bo-tanimoto-ldm": "BO + LDM (Tanimoto fingerprint, LLM advisor)",
    "bo-strkernel-ldm": "BO + LDM (Subsequence string kernel, LLM advisor)",
}

FILENAME_RE = re.compile(r"^(?P<method>.+)_seed=(?P<seed>\d+)\.json$")


# ---------------------------------------------------------------------------
# Loading + aggregation
# ---------------------------------------------------------------------------


def parse_filename(path: Path) -> Optional[tuple]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("method"), int(m.group("seed"))


def _read_history_entry(entry: dict, n_obj: int) -> Optional[Tuple[str, object]]:
    """Read one JSON ``history`` entry. Returns ``(smiles, score_or_scores)``."""
    smi = entry.get("smiles")
    if smi is None:
        return None
    if n_obj == 1:
        sc = entry.get("score")
        if sc is not None and not isinstance(sc, (int, float)):
            sc = None
        elif sc is not None and not np.isfinite(float(sc)):
            sc = None
        return smi, (None if sc is None else float(sc))
    # n_obj >= 2
    scs = entry.get("scores")
    if scs is None:
        # Tolerate the legacy single-obj schema for back-compat.
        sc = entry.get("score")
        if sc is None:
            return smi, None
        scs = [sc]
    if not isinstance(scs, list):
        return smi, None
    out: List[Optional[float]] = []
    for v in scs:
        if v is None or not isinstance(v, (int, float)):
            out.append(None)
        else:
            f = float(v)
            out.append(f if np.isfinite(f) else None)
    while len(out) < n_obj:
        out.append(None)
    return smi, tuple(out[:n_obj])


def load_history(path: Path, n_obj: int) -> list:
    """Load a JSON's history list. ``n_obj`` from the JSON's config echo."""
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    history_raw = payload.get("history", [])
    out = []
    for entry in history_raw:
        parsed = _read_history_entry(entry, n_obj)
        if parsed is None:
            continue
        smi, sc = parsed
        out.append((smi, sc))
    return out


def _read_config(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("config", {}) or {}
    except Exception:
        return {}


def best_so_far_single(
    history: list,
    num_evaluations: int,
    minimize: bool,
) -> np.ndarray:
    """Best-so-far curve for a single-objective history."""
    bsf: list[float] = []
    current = float("inf") if minimize else float("-inf")
    for _, sc in history[:num_evaluations]:
        if sc is not None:
            v = sc if isinstance(sc, (int, float)) else sc[0]
            if v is not None and np.isfinite(float(v)):
                current = min(current, float(v)) if minimize else max(current, float(v))
        bsf.append(current)
    while len(bsf) < num_evaluations:
        bsf.append(current)
    return np.asarray(bsf, dtype=float)


def hypervolume_curve(
    history: list,
    num_evaluations: int,
    ref_point: Sequence[float],
    minimize: Sequence[bool],
) -> np.ndarray:
    """Cumulative 2-objective hypervolume curve w.r.t. ``ref_point``."""
    from strbo_v1.acquisition import hypervolume
    n_obj = len(minimize)
    if n_obj != 2:
        raise ValueError(f"hypervolume_curve only supports n_obj=2; got {n_obj}")
    ref = list(ref_point)
    out: list[float] = []
    for k in range(1, num_evaluations + 1):
        partial = history[:k]
        finite = [
            sc for _, sc in partial
            if sc is not None
            and len(sc) == 2
            and all(v is not None and np.isfinite(float(v)) for v in sc)
        ]
        if not finite:
            out.append(0.0)
            continue
        hv = hypervolume(
            points=[tuple(float(v) for v in sc) for sc in finite],
            ref=ref,
            minimize=tuple(minimize),
        )
        out.append(float(hv))
    while len(out) < num_evaluations:
        out.append(out[-1] if out else 0.0)
    return np.asarray(out, dtype=float)


def per_obj_best_so_far(
    history: list,
    num_evaluations: int,
    n_obj: int,
) -> np.ndarray:
    """Per-objective best-so-far for ``n_obj >= 3``.

    Direction is inferred from the *first* finite value per objective
    (if positive, the value is treated as already in the right
    direction; we use ``np.nanmax`` / ``np.nanmin`` accordingly). For
    n_obj >= 3 we don't have a minimize direction in the JSON; the
    caller should pass per-objective direction explicitly if needed.
    Here we just take the running best (max if positive, else min)
    for visualisation.

    Returns:
        ``np.ndarray`` of shape ``(n_obj, num_evaluations)``.
    """
    arrs = np.full((n_obj, num_evaluations), np.nan, dtype=float)
    for k in range(num_evaluations):
        if k >= len(history):
            break
        _, sc = history[k]
        if sc is None or len(sc) != n_obj:
            continue
        for i in range(n_obj):
            v = sc[i]
            if v is None or not np.isfinite(float(v)):
                continue
            prev = arrs[i, k - 1] if k > 0 else np.nan
            if np.isnan(prev):
                arrs[i, k] = float(v)
            else:
                # For n_obj>=3 we just keep the running best (max).
                # Callers that want a "minimize" view can flip the
                # sign externally. The plotter uses these as-is.
                arrs[i, k] = max(prev, float(v))
    # Forward-fill.
    for i in range(n_obj):
        last = np.nan
        for k in range(num_evaluations):
            if np.isnan(arrs[i, k]):
                arrs[i, k] = last
            else:
                last = arrs[i, k]
    # Pad with last value.
    for i in range(n_obj):
        if num_evaluations > 0 and not np.isnan(arrs[i, num_evaluations - 1]):
            tail = arrs[i, num_evaluations - 1]
        else:
            tail = 0.0
        for k in range(num_evaluations):
            if np.isnan(arrs[i, k]):
                arrs[i, k] = tail
    return arrs


def load_inputs(
    input_dir: Path,
    methods_filter: Optional[set],
) -> Tuple[dict, dict]:
    """Return per-(method, seed) histories and per-method metadata.

    Returns:
        ``(results, meta)`` where ``results`` maps method -> seed ->
        history, and ``meta`` maps method -> ``{n_obj, minimize,
        ref_point}`` (read from the first JSON of that method).
    """
    results: dict = defaultdict(dict)
    meta: dict = {}

    for path in sorted(input_dir.glob("*_seed=*.json")):
        parsed = parse_filename(path)
        if parsed is None:
            LOGGER.warning("skipping file with non-matching name: %s", path.name)
            continue
        method, seed = parsed
        if methods_filter is not None and method not in methods_filter:
            continue
        cfg = _read_config(path)
        n_obj = int(cfg.get("n_objectives", 1))
        history = load_history(path, n_obj=n_obj)
        if not history:
            LOGGER.warning("empty history in %s; skipping", path.name)
            continue
        results[method][seed] = history
        if method not in meta:
            minimize_raw = cfg.get("minimize", True)
            if isinstance(minimize_raw, list):
                minimize_t: Tuple = tuple(bool(x) for x in minimize_raw)
            else:
                minimize_t = (bool(minimize_raw),)
            ref = cfg.get("ref_point")
            ref_t: Optional[Tuple[float, ...]] = (
                tuple(float(x) for x in ref) if isinstance(ref, list) else None
            )
            meta[method] = {
                "n_obj": n_obj,
                "minimize": minimize_t,
                "ref_point": ref_t,
            }
    return results, meta


def aggregate(
    results: dict,
    meta: dict,
    num_evaluations: int,
    *,
    start: int = 0,
    step: int = 1,
    default_ref_point: Optional[Tuple[float, ...]] = None,
) -> dict:
    """Compute per-method mean/std summary curves.

    For ``n_obj == 1``: best-so-far (minimise direction from ``meta``).
    For ``n_obj == 2``: cumulative hypervolume.
    For ``n_obj >= 3``: per-objective best-so-far (shape
    ``(n_obj, num_evaluations)``).

    Slicing is applied per-seed BEFORE mean/std: ``arr[:, start::step]``.

    Returns:
        Dict mapping method -> ``(mean, std, n_obj)``. For n_obj >= 3,
        ``mean`` and ``std`` have an extra leading axis.
    """
    out: dict = {}
    for method, seed_dict in results.items():
        if not seed_dict:
            continue
        method_meta = meta.get(method, {"n_obj": 1, "minimize": (True,), "ref_point": None})
        n_obj = method_meta["n_obj"]
        minimize_t = method_meta["minimize"]
        ref_point = method_meta["ref_point"] or default_ref_point
        curves = []
        for hist in seed_dict.values():
            if n_obj == 1:
                curves.append(best_so_far_single(hist, num_evaluations, minimize=minimize_t[0]))
            elif n_obj == 2:
                if ref_point is None:
                    raise ValueError(
                        f"method {method!r}: n_obj=2 requires ref_point. "
                        "Pass --ref-point on the CLI or write it into the JSON's config."
                    )
                curves.append(
                    hypervolume_curve(
                        hist, num_evaluations,
                        ref_point=ref_point, minimize=minimize_t,
                    )
                )
            else:
                curves.append(per_obj_best_so_far(hist, num_evaluations, n_obj))
        if n_obj <= 2:
            arr = np.stack(curves, axis=0)               # (n_seeds, num_evaluations)
            sliced = arr[:, start::step]                  # (n_seeds, n_slices)
            out[method] = (sliced.mean(axis=0), sliced.std(axis=0), n_obj)
        else:
            arr = np.stack(curves, axis=0)               # (n_seeds, n_obj, num_evaluations)
            sliced = arr[:, :, start::step]               # (n_seeds, n_obj, n_slices)
            out[method] = (sliced.mean(axis=0), sliced.std(axis=0), n_obj)
    return out


def find_num_evaluations(results: dict) -> int:
    """Use the longest history across all (method, seed) files."""
    max_len = 0
    for seed_dict in results.values():
        for history in seed_dict.values():
            max_len = max(max_len, len(history))
    return max_len


# ---------------------------------------------------------------------------
# CSV / plot
# ---------------------------------------------------------------------------


def write_csv(
    mean_std_dir: dict,
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(mean_std_dir.keys())
    if not methods:
        return
    first = next(iter(mean_std_dir.values()))
    if first[2] <= 2:
        num_pts = len(first[0])
    else:
        num_pts = first[0].shape[-1]
    fieldnames = ["evaluation"]
    for m in methods:
        n_obj = mean_std_dir[m][2]
        if n_obj <= 2:
            fieldnames.extend([f"{m}_mean", f"{m}_std"])
        else:
            for i in range(n_obj):
                fieldnames.extend([f"{m}_obj{i}_mean", f"{m}_obj{i}_std"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        for i in range(num_pts):
            row = [i]
            for m in methods:
                mean, std, n_obj = mean_std_dir[m]
                if n_obj <= 2:
                    row.extend([f"{mean[i]:.6g}", f"{std[i]:.6g}"])
                else:
                    for k in range(n_obj):
                        row.extend([f"{mean[k, i]:.6g}", f"{std[k, i]:.6g}"])
            writer.writerow(row)
    LOGGER.info("wrote CSV: %s", csv_path)


def _plot_single_obj(
    mean_std_dir: dict,
    fig_path: Path,
    title: Optional[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install via `pip install matplotlib`."
        ) from exc

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, (mean, std, n_obj) in mean_std_dir.items():
        if n_obj != 1:
            continue
        color = METHOD_COLORS.get(method, "tab:purple")
        label = METHOD_LABELS.get(method, method)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label, color=color, linewidth=2.0)
        if len(mean) > 1 and np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
    ax.set_xlabel("BO iteration (0 = right after init)")
    ax.set_ylabel("Best score so far (lower is better)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_path)
    LOGGER.info("wrote plot: %s", fig_path)


def _plot_2obj(
    mean_std_dir: dict,
    fig_path: Path,
    title: Optional[str],
) -> None:
    """Plot the cumulative hypervolume curve for 2-objective methods.

    Single-subplot: the HV curve is the canonical view for 2-obj runs.
    A second subplot with a 2D Pareto scatter was previously planned
    but never implemented (the placeholder text was misleading), so
    the layout is now a single subplot to keep the figure focused.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, (mean, std, n_obj) in mean_std_dir.items():
        if n_obj != 2:
            continue
        color = METHOD_COLORS.get(method, "tab:purple")
        label = METHOD_LABELS.get(method, method)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label, color=color, linewidth=2.0)
        if len(mean) > 1 and np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Cumulative hypervolume")
    ax.set_title("Hypervolume progression" if title is None else title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_path)
    LOGGER.info("wrote plot: %s", fig_path)


def _plot_per_obj_bsf(
    mean_std_dir: dict,
    fig_path: Path,
    title: Optional[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    # Find the max n_obj across all methods.
    max_n = max((n for _, _, n in mean_std_dir.values()), default=1)
    fig, axes = plt.subplots(1, max_n, figsize=(5 * max_n, 5), sharex=True)
    if max_n == 1:
        axes = [axes]
    for method, (mean, std, n_obj) in mean_std_dir.items():
        if n_obj < 3:
            continue
        color = METHOD_COLORS.get(method, "tab:purple")
        label = METHOD_LABELS.get(method, method)
        for i in range(min(n_obj, max_n)):
            ax = axes[i]
            x = np.arange(mean.shape[-1])
            ax.plot(x, mean[i], label=label, color=color, linewidth=2.0)
            if mean.shape[-1] > 1 and np.any(std[i] > 0):
                ax.fill_between(x, mean[i] - std[i], mean[i] + std[i], color=color, alpha=0.1)
            ax.set_title(f"Objective {i}")
            ax.set_xlabel("Evaluation index")
            ax.grid(alpha=0.3)
    # Add a single legend in the first axis (or use figure-level).
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, fig_path)
    LOGGER.info("wrote plot: %s", fig_path)


def _save_fig(fig, fig_path: Path) -> None:
    if fig_path.suffix.lower() == ".pdf":
        fig.savefig(fig_path, format="pdf")
    else:
        fig.savefig(fig_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_summary(
    mean_std_dir: dict,
    fig_path: Path,
    title: Optional[str] = None,
) -> None:
    """Dispatch to the right plot function based on the n_obj distribution."""
    n_objs = {n for _, _, n in mean_std_dir.values()}
    if not n_objs:
        return
    if n_objs == {1}:
        _plot_single_obj(mean_std_dir, fig_path, title)
        return
    if n_objs == {2}:
        _plot_2obj(mean_std_dir, fig_path, title)
        return
    # Mixed or >=3 → degrade to per-objective (covers >=3 and mixed
    # mixed inputs by routing the >=3 method through the per-obj
    # plotter; single-obj methods are ignored for the per-obj view).
    if 1 in n_objs and 2 not in n_objs and len(n_objs) == 1:
        _plot_single_obj(mean_std_dir, fig_path, title)
        return
    if 2 in n_objs and 1 in n_objs:
        LOGGER.warning(
            "plot_summary: mixed single-obj and 2-obj methods; "
            "falling back to per-obj BSF for 2-obj methods (HV curve skipped)."
        )
    _plot_per_obj_bsf(mean_std_dir, fig_path, title)


def _summary(mean_std_dir: dict) -> str:
    lines = ["=== Summary ==="]
    lines.append(f"{'method':<14} {'n_obj':>6} {'final_mean':>12} {'final_std':>11}")
    for m, (mean, std, n_obj) in mean_std_dir.items():
        if n_obj <= 2:
            lines.append(f"{m:<14} {n_obj:>6} {mean[-1]:>12.4f} {std[-1]:>11.4f}")
        else:
            for i in range(n_obj):
                lines.append(f"{m:<14} {n_obj:>6} obj{i}_mean={mean[i, -1]:.4f} std={std[i, -1]:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate run_search JSON outputs and plot summary curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default="output/bo",
                        help="Directory containing '<method>_seed=<seed>.json' files.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output base path (no extension). Defaults to "
                             "<input-dir>/summary; figure format chosen via --figure-format.")
    parser.add_argument("--methods", type=str, default=None,
                        help="Optional comma-separated method filter (default: auto-detect).")
    parser.add_argument("--ref-point", type=str, default=None,
                        help="Comma-separated default reference point for 2-objective "
                             "runs that did not embed one in their JSON config. "
                             "Silently ignored for single-objective. "
                             "Multi-objective plots (n_obj>=3) do not need a ref_point.")
    parser.add_argument("--num-evaluations", type=int, default=None,
                        help="Pad/truncate curves to this length. Default: longest history.")
    parser.add_argument("--figure-format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-csv", action="store_true", help="Skip writing the combined CSV.")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip first N samples. Slice applied per-seed before mean/std.")
    parser.add_argument("--step", type=int, default=1,
                        help="Subsample every Nth point.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {input_dir}")

    methods_filter: Optional[set] = None
    if args.methods:
        methods_filter = {m.strip() for m in args.methods.split(",") if m.strip()}

    results, meta = load_inputs(input_dir, methods_filter)
    if not results:
        raise SystemExit(f"No matching JSON files in {input_dir}")

    # Default ref_point from --ref-point, used when a JSON didn't embed one.
    default_ref: Optional[Tuple[float, ...]] = None
    if args.ref_point:
        try:
            default_ref = tuple(float(x) for x in args.ref_point.split(","))
        except ValueError as exc:
            raise SystemExit(f"--ref-point invalid: {exc}") from exc

    num_evaluations = args.num_evaluations or find_num_evaluations(results)
    if args.step < 1:
        raise SystemExit(f"--step must be >= 1, got {args.step}")
    if args.start < 0:
        raise SystemExit(f"--start must be >= 0, got {args.start}")
    mean_std_dir = aggregate(
        results, meta, num_evaluations,
        start=args.start, step=args.step,
        default_ref_point=default_ref,
    )

    base = Path(args.output) if args.output else input_dir / "summary"
    fig_path = base.with_suffix(f".{args.figure_format}")
    print(f"[output] writing plot -> {fig_path}")
    plot_summary(mean_std_dir, fig_path, title=args.title)
    if not args.no_csv:
        csv_path = base.with_suffix(".csv")
        print(f"[output] writing combined CSV -> {csv_path}")
        write_csv(mean_std_dir, csv_path)

    print()
    print(_summary(mean_std_dir))
    print(f"\nOutputs:\n  {fig_path}")
    if not args.no_csv:
        print(f"  {base.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
