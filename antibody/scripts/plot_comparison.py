"""Plot N-method BO comparison (LDM, AntBO baselines, reproduction methods).

Each ``--methods`` entry is auto-detected as either:
  * Aggregate reproduction CSV  (``outputs/reproduction/*.csv``)
    — columns: ``Antigen, Seed, Num BB Evals, Best Binding Energy``
  * Experiment directory         (auto-discovered seeds/antigens)
    — multi-seed new:   ``{root}/seed_N/BO_K/antigen_AG_kernel_.../results.csv``
    — multi-seed legacy:``{root}/seed_N/antigen_AG_kernel_.../results.csv``
    — single-seed:      ``{root}/antigen_AG_kernel_.../results.csv``
    — single-seed no-kernel (e.g. ``result1/llm_baseline_5x5_200``):
                       ``{root}/antigen_AG_seed_N_n200_batch1/results.csv``

All methods are plotted on the same per-antigen subplot grid for direct
visual comparison.

Usage (new --methods interface):
    # Two methods (LDM + one reproduction CSV)
    python scripts/plot_comparison.py \
        --methods outputs/ldm_ninit20_iter200,outputs/reproduction/BO_transformed_overlap_optim_res.csv

    # Many methods with custom labels and colors
    python scripts/plot_comparison.py \
        --methods outputs/ldm_ninit20_iter200,outputs/reproduction/HEBO_optim_res.csv,outputs/reproduction/RS_optim_res.csv \
        --labels  LDM-AntBO,HEBO,RS \
        --colors  #1f77b4,#2ca02c,#d62728

    # Partial labels (use '' to fall back to path-name default)
    python scripts/plot_comparison.py \
        --methods a.csv,b.csv,c.csv \
        --labels  LABEL1,,LABEL3

    # Plot every reproduction CSV + the LDM run
    python scripts/plot_comparison.py \
        --methods outputs/reproduction/BObert_optim_res.csv,\
                  outputs/reproduction/BO_COMBO_optim_res.csv,\
                  outputs/reproduction/BO_ssk_optim_res.csv,\
                  outputs/reproduction/BO_transformed_overlap_ntr_optim_res.csv,\
                  outputs/reproduction/BO_transformed_overlap_optim_res.csv,\
                  outputs/reproduction/GA_optim_res.csv,\
                  outputs/reproduction/HEBO_optim_res.csv,\
                  outputs/reproduction/LamBO_optim_res.csv,\
                  outputs/reproduction/RS_optim_res.csv,\
                  outputs/reproduction/TURBO_optim_res.csv,\
                  outputs/ldm_ninit20_iter200 \
        --output  outputs/comparisons/plots/all_methods.png \
        --title   "All methods comparison (5 antigens)"

Usage (legacy interface, still supported):
    python scripts/plot_comparison.py \
        --ldm-dir outputs/ldm_ninit20_iter200 \
        --baseline-dir outputs/baseline_ninit20_iter200
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── auto-discovery helpers ──────────────────────────────────────────

def _antigen_from_dirname(name: str) -> str:
    """Extract antigen id from a directory name.

    Handles four layouts::

        antigen_1ADQ_A_kernel_transformed_overlap_...       -> 1ADQ_A
        antigen_1ADQ_A_seed_42_n200_batch1                  -> 1ADQ_A
        antigen_1ADQ_A_kernel_..._search-strat_..._seed_42   -> 1ADQ_A
        antigen_1ADQ_A                                       -> 1ADQ_A

    The antigen id is the substring between the ``antigen_`` prefix and
    the first ``_kernel_`` or ``_seed_`` token (whichever comes first).
    """
    stripped = name.replace("antigen_", "", 1)
    for marker in ("_kernel_", "_seed_"):
        if marker in stripped:
            return stripped.split(marker, 1)[0]
    return stripped


def detect_layout(root: Path) -> str:
    """Detect directory layout of an experiment root.

    Returns ``"multi_seed"`` if ``root`` contains ``seed_*`` subdirectories
    (layout: ``seed_N/antigen_*/results.csv``), or ``"single_seed"`` if
    ``root`` contains ``antigen_*`` subdirectories directly
    (layout: ``antigen_*/results.csv``).

    Raises ``FileNotFoundError`` if neither pattern is present.
    Multi-seed takes precedence if both somehow coexist at the root level.
    """
    entries = list(root.iterdir())
    if any(p.is_dir() and p.name.startswith("seed_") for p in entries):
        return "multi_seed"
    if any(p.is_dir() and p.name.startswith("antigen_") for p in entries):
        return "single_seed"
    raise FileNotFoundError(
        f"No seed_* or antigen_* subdirectories found in {root}"
    )


def discover_seeds(root: Path) -> list[int]:
    """Find all ``seed_N`` subdirectories and return sorted seed numbers.

    For single-seed layout (no ``seed_*`` dirs), returns ``[1]`` so that
    downstream title generation reports "1 seed" rather than "0 seeds".
    """
    if detect_layout(root) == "single_seed":
        return [1]
    seeds = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("seed_"):
            try:
                seeds.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(seeds)


def _result_csv_patterns(root: Path) -> list[str]:
    """Return glob patterns matching results.csv files under ``root``.

    Single-seed layout matches ``antigen_*/results.csv``.

    Multi-seed layout tries both depths and returns whichever matches:
      * new (3-level): ``seed_*/BO_*/antigen_*/results.csv``
      * legacy (2-level): ``seed_*/antigen_*/results.csv``

    Both are returned if both have matches (mixed layout).
    """
    if detect_layout(root) == "single_seed":
        return ["antigen_*/results.csv"]
    candidates = [
        "seed_*/BO_*/antigen_*/results.csv",
        "seed_*/antigen_*/results.csv",
    ]
    return [p for p in candidates if any(root.glob(p))]


def discover_antigens(root: Path) -> list[str]:
    """Find all antigens present in the experiment root."""
    antigens: set[str] = set()
    for pat in _result_csv_patterns(root):
        for csv in root.glob(pat):
            antigens.add(_antigen_from_dirname(csv.parent.name))
    return sorted(antigens)


# ── data loading ────────────────────────────────────────────────────

def _truncate_and_stack(series_list: List[np.ndarray]) -> np.ndarray:
    """Truncate all series to the shortest length, then stack."""
    min_len = min(len(s) for s in series_list)
    return np.stack([s[:min_len] for s in series_list])

def load_curves(root: Path) -> Dict[str, np.ndarray]:
    """Load best-so-far curves from an experiment directory.

    Supports three auto-detected layouts::

        # multi-seed, new (default for current sweep script)
        {root}/seed_{N}/BO_{kernel}/antigen_{AG}_kernel_.../results.csv

        # multi-seed, legacy
        {root}/seed_{N}/antigen_{AG}_kernel_.../results.csv

        # single-seed
        {root}/antigen_{AG}_kernel_.../results.csv

        # single-seed, no-kernel (e.g. result1/llm_baseline_5x5_200)
        {root}/antigen_{AG}_seed_{N}_n200_batch1/results.csv

    Returns ``{antigen: array(n_runs, T)}`` — ``n_runs == 1`` for
    single-seed layout. Both multi-seed depths are matched if both present.
    """
    out: Dict[str, List[np.ndarray]] = {}
    for pat in _result_csv_patterns(root):
        for csv in sorted(root.glob(pat)):
            ag = _antigen_from_dirname(csv.parent.name)
            out.setdefault(ag, []).append(pd.read_csv(csv)["BestValue"].values)
    return {ag: _truncate_and_stack(v) for ag, v in out.items() if v}


def load_aggregate_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load best-so-far curves from an aggregate reproduction CSV.

    Expected columns: ``Antigen``, ``Seed``, ``Num BB Evals``,
    ``Best Binding Energy``.

    Returns ``{antigen: array(n_seeds, T)}``.
    """
    df = pd.read_csv(csv_path)
    out: Dict[str, List[np.ndarray]] = {}
    for antigen, sub_ag in df.groupby("Antigen"):
        series_list = []
        for _, g in sub_ag.groupby("Seed"):
            g = g.sort_values("Num BB Evals")
            y = pd.to_numeric(g["Best Binding Energy"], errors="coerce").values
            series_list.append(y)
        if series_list:
            out[str(antigen)] = _truncate_and_stack(series_list)
    return out


def load_method(path: Path) -> Dict[str, np.ndarray]:
    """Load curves from either a CSV file or an experiment directory.

    Dispatches to :func:`load_aggregate_csv` for files and
    :func:`load_curves` for directories.

    Returns ``{antigen: array(n_runs, T)}``.
    """
    if path.is_file():
        return load_aggregate_csv(path)
    if path.is_dir():
        return load_curves(path)
    raise FileNotFoundError(
        f"Method path is neither a file nor a directory: {path}"
    )


# ── CLI argument parsing helpers ───────────────────────────────────

def _split_csv_arg(value: str) -> List[str]:
    """Split a comma-separated string, keeping empty entries (``''``)."""
    return [v.strip() for v in value.split(",")]


def _default_label_for_path(path: Path) -> str:
    """Default label = stem of filename (for CSVs) or dir name."""
    if path.is_file():
        return path.stem
    return path.name


# ── plotting ────────────────────────────────────────────────────────

def _seed_word(n: int) -> str:
    """Return singular ``seed`` for n==1, plural ``seeds`` otherwise."""
    return "seed" if n == 1 else "seeds"


def plot_grid(
    methods: Dict[str, Dict[str, np.ndarray]],
    colors: List[str],
    out_path: Path,
    title: str,
    antigen_filter: Optional[set[str]] = None,
    xlabel: str = "Evaluation index",
    ylabel: str = "Best-so-far ΔG (lower = better)",
    n_column: int = 3,
    font_scale: float = 1.0,
) -> None:
    """Plot a grid of per-antigen subplots with all methods overlaid.

    Args:
        methods: ``{label: {antigen: array(n_runs, T)}}`` — one entry per
            method, in the same order as ``colors``.
        colors:  color per method (one entry per method).
        out_path: output PNG path.
        title:   suptitle.
        antigen_filter: if provided, only antigens in this set are plotted
            (intersected with the union of all methods' antigens).
        xlabel:  x-axis label.
        ylabel:  y-axis label.
        n_column: max number of columns in the subplot grid. If
            ``n_column >= n_antigens``, all subplots are placed in a
            single row. Default: 3.
        font_scale: multiplicative scale on default font sizes.
    """
    if not methods:
        print("ERROR: no methods to plot.")
        return

    # Union of all antigens across all methods, preserving sorted order.
    antigen_set: set[str] = set()
    for curves in methods.values():
        antigen_set.update(curves.keys())
    if antigen_filter is not None:
        antigen_set &= antigen_filter
    antigens = sorted(antigen_set)
    n = len(antigens)
    if n == 0:
        print("ERROR: no antigens found across any method (after filter).")
        return

    labels = list(methods.keys())

    cols = max(1, min(n_column, n))
    rows = (n + cols - 1) // cols
    if rows == 1:
        fig_w_per = 4.2 * font_scale
        fig_h = 4.5 * font_scale
    else:
        fig_w_per = 5.5
        fig_h = 3.8 * rows
    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_w_per * cols, fig_h), squeeze=False
    )

    for ax, antigen in zip(axes.flat, antigens):
        # Build per-method text annotation block (top-right).
        ann_lines: List[str] = []
        for i, (label, curves) in enumerate(methods.items()):
            color = colors[i]
            if antigen not in curves:
                continue
            Y = curves[antigen]
            x = np.arange(1, Y.shape[1] + 1)
            mean = Y.mean(0)
            std = Y.std(0) if Y.shape[0] > 1 else np.zeros_like(mean)

            ax.plot(
                x, mean, color=color, lw=2.5 * font_scale,
                label=label,
            )
            if Y.shape[0] > 1:
                ax.fill_between(
                    x, mean - std, mean + std, color=color, alpha=0.15,
                )

            ann_lines.append(
                f"{label}: {mean[-1]:.1f} ± {std[-1]:.1f}"
            )

        # Per-antigen final-value annotation block.
        if ann_lines:
            ann_text = "\n".join(ann_lines)
            ax.text(
                0.98, 0.97, ann_text,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11 * font_scale, color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                family="monospace",
            )

        ax.set_title(antigen, fontsize=16 * font_scale, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=15 * font_scale)
        ax.set_ylabel(ylabel, fontsize=15 * font_scale)
        ax.tick_params(axis="both", labelsize=13 * font_scale)
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[n:]:
        ax.axis("off")

    # Unified legend at center bottom of figure.
    handles, leg_labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        n_cols = min(len(handles), 5)
        fig.legend(
            handles, leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=n_cols,
            fontsize=15 * font_scale,
            frameon=False,
        )

    fig.suptitle(title, fontsize=18 * font_scale, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────

def _parse_methods(
    args: argparse.Namespace,
) -> Tuple[List[Path], List[str]]:
    """Resolve the list of (path, default_label) from ``--methods`` or
    legacy ``--ldm-dir``/``--baseline-dir``/``--baseline-csv`` args.

    Returns:
        (paths, default_labels) — one entry per method, in order.

    Raises:
        SystemExit: if both new and legacy interfaces are used, or if
        neither is supplied.
    """
    legacy_provided = any([
        args.ldm_dir is not None,
        args.baseline_dir is not None,
        args.baseline_csv is not None,
    ])

    if args.methods:
        if legacy_provided:
            raise SystemExit(
                "ERROR: use either --methods (new) or --ldm-dir/--baseline-dir/--baseline-csv (legacy), not both."
            )
        paths = [Path(p) for p in _split_csv_arg(args.methods)]
        labels = [_default_label_for_path(p) for p in paths]
        return paths, labels

    if not legacy_provided:
        raise SystemExit(
            "ERROR: must provide --methods (comma-separated paths) "
            "or legacy --ldm-dir (+ optional --baseline-dir/--baseline-csv)."
        )

    paths: List[Path] = []
    labels: List[str] = []
    if args.ldm_dir is not None:
        paths.append(args.ldm_dir)
        labels.append("LDM AntBO")
    if args.baseline_dir is not None:
        paths.append(args.baseline_dir)
        labels.append("AntBO baseline")
    elif args.baseline_csv is not None:
        paths.append(args.baseline_csv)
        labels.append(args.baseline_label or "AntBO baseline")
    return paths, labels


def _resolve_labels_colors(
    paths: List[Path],
    default_labels: List[str],
    args: argparse.Namespace,
) -> Tuple[List[str], List[str]]:
    """Validate lengths, fill in defaults for empty entries, assign colors.

    Returns:
        (labels, colors) — both lists of the same length as ``paths``.
    """
    n = len(paths)

    # --- labels ---
    if args.labels is None:
        labels = list(default_labels)
    else:
        user_labels = _split_csv_arg(args.labels)
        if len(user_labels) != n:
            raise SystemExit(
                f"ERROR: --labels has {len(user_labels)} entries but --methods has {n}. "
                "Lengths must match."
            )
        labels = [
            user_lbl if user_lbl else default_lbl
            for user_lbl, default_lbl in zip(user_labels, default_labels)
        ]

    # --- colors ---
    if args.colors is None:
        colors = [f"C{i}" for i in range(n)]
    else:
        user_colors = _split_csv_arg(args.colors)
        if len(user_colors) != n:
            raise SystemExit(
                f"ERROR: --colors has {len(user_colors)} entries but --methods has {n}. "
                "Lengths must match."
            )
        # Empty entries fall back to the C{i} default.
        colors = [
            user_clr if user_clr else f"C{i}"
            for i, user_clr in enumerate(user_colors)
        ]
    return labels, colors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot N-method BO comparison (LDM, baselines, reproduction). "
            "All methods are overlaid on the same per-antigen subplot grid."
        ),
    )
    # New unified interface.
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated list of method paths. Each entry is "
             "auto-detected as either an aggregate reproduction CSV "
             "(columns: Antigen, Seed, Num BB Evals, Best Binding Energy) "
             "or an experiment directory (multi-seed or single-seed layout).",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Comma-separated legend labels (one per --methods entry). "
             "Empty entries (e.g. 'L1,,L3') fall back to the path-stem "
             "default. Must have the same length as --methods.",
    )
    parser.add_argument(
        "--colors", type=str, default=None,
        help="Comma-separated matplotlib color specs (one per --methods "
             "entry). Empty entries fall back to the default 'C0, C1, ...' "
             "cycle. Must have the same length as --methods.",
    )
    parser.add_argument(
        "--antigens", type=Path, default=None,
        help="Optional path to a text file listing antigens to plot "
             "(one per line, like test_5_antigens.txt). Only antigens "
             "present in this list AND in at least one method are plotted. "
             "Default: union of all antigens across all methods.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG path. Default: "
             "outputs/comparisons/plots/ldm_vs_baseline.png",
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Plot title. Default: auto-generated.",
    )
    parser.add_argument(
        "--xlabel", type=str, default="Evaluation index",
        help="X-axis label. Default: 'Evaluation index'.",
    )
    parser.add_argument(
        "--ylabel", type=str, default="Best-so-far ΔG (lower = better)",
        help="Y-axis label. Default: 'Best-so-far ΔG (lower = better)'.",
    )
    parser.add_argument(
        "--n-column", type=int, default=3,
        help="Max number of columns in the subplot grid. If >= n_antigens, "
             "all subplots are placed in a single row. Default: 3.",
    )
    parser.add_argument(
        "--font-scale", type=float, default=1.0,
        help="Multiplicative scale on default font sizes. Default: 1.0.",
    )

    # Legacy interface (deprecated; internally mapped to --methods).
    parser.add_argument(
        "--ldm-dir", type=Path, default=None,
        help="[DEPRECATED] LDM experiment root. Use --methods instead.",
    )
    bl_group = parser.add_mutually_exclusive_group()
    bl_group.add_argument(
        "--baseline-dir", type=Path, default=None,
        help="[DEPRECATED] Baseline experiment root. Use --methods instead.",
    )
    bl_group.add_argument(
        "--baseline-csv", type=Path, default=None,
        help="[DEPRECATED] Aggregate reproduction CSV. Use --methods instead.",
    )
    parser.add_argument(
        "--baseline-label", type=str, default="AntBO baseline",
        help="[DEPRECATED] Label for the legacy baseline curve. "
             "Use --labels with --methods instead.",
    )
    args = parser.parse_args()

    # --- Resolve methods (paths + default labels) ---
    paths, default_labels = _parse_methods(args)

    # --- Validate all paths exist and load data ---
    method_data: Dict[str, Dict[str, np.ndarray]] = {}
    for path, default_label in zip(paths, default_labels):
        if not path.exists():
            raise FileNotFoundError(
                f"Method path does not exist: {path}"
            )
        print(f"Loading method: {path}")
        try:
            curves = load_method(path)
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: failed to load {path}: {e}")
        for ag in sorted(curves):
            print(f"  loaded {ag}: shape={curves[ag].shape}")
        method_data[default_label] = curves

    # --- Resolve labels and colors ---
    labels, colors = _resolve_labels_colors(paths, default_labels, args)

    # Re-key method_data by the final (possibly user-overridden) labels,
    # preserving the user-specified order.
    if list(method_data.keys()) != labels:
        method_data = {lbl: method_data[default_lbl] for lbl, default_lbl in zip(labels, default_labels)}

    print("")
    print("Final method order:")
    for i, (lbl, clr) in enumerate(zip(labels, colors)):
        curves_i = method_data[lbl]
        n_runs_total = sum(Y.shape[0] for Y in curves_i.values())
        n_antigens_covered = len(curves_i)
        print(
            f"  [{i}] {lbl}  color={clr}  ← {paths[i]}  "
            f"runs={n_runs_total}  antigens={n_antigens_covered}"
        )

    # --- Output path & title ---
    out_path = args.output or Path(
        "outputs/comparisons/plots/ldm_vs_baseline.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.title:
        title = args.title
    else:
        n_methods = len(labels)
        method_word = "method" if n_methods == 1 else "methods"
        title = f"Comparison: {n_methods} {method_word} on {len(set().union(*[set(c.keys()) for c in method_data.values()]))} antigens"

    # --- Antigen filter ---
    antigen_filter: Optional[set[str]] = None
    if args.antigens is not None:
        if not args.antigens.is_file():
            raise FileNotFoundError(
                f"--antigens file not found: {args.antigens}"
            )
        with open(args.antigens) as f:
            antigen_filter = {
                line.strip() for line in f if line.strip()
            }
        print(f"Antigen filter ({len(antigen_filter)} entries): {sorted(antigen_filter)}")

    plot_grid(method_data, colors, out_path, title, antigen_filter, args.xlabel, args.ylabel, args.n_column, args.font_scale)


if __name__ == "__main__":
    main()
