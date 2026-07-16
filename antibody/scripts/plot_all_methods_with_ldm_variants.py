#!/usr/bin/env python3
"""Plot existing AntBO baselines plus Reservoir-LDM softmax/argmax variants.

This is a small wrapper around scripts/plot_comparison.py. It checks that the
new 25-run experiment directories are complete before plotting, then passes all
method paths, labels, and colors to the existing plotting code.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def count_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def completion_report(root: Path, expected_runs: int, expected_evals: int) -> tuple[int, int, list[str]]:
    csvs = sorted(root.glob("antigen_*_seed_*_n200_batch1/results.csv"))
    complete = 0
    incomplete: list[str] = []
    for csv in csvs:
        rows = count_rows(csv)
        if rows >= expected_evals:
            complete += 1
        else:
            incomplete.append(f"{csv.parent.name}: {rows}/{expected_evals}")
    if len(csvs) < expected_runs:
        incomplete.append(f"missing result files: {len(csvs)}/{expected_runs}")
    return len(csvs), complete, incomplete


def require_complete(name: str, root: Path, expected_runs: int, expected_evals: int, allow_partial: bool) -> None:
    if not root.exists():
        raise SystemExit(f"{name} directory not found: {root}")
    found, complete, incomplete = completion_report(root, expected_runs, expected_evals)
    print(f"{name}: result files={found}/{expected_runs}, complete={complete}/{expected_runs}")
    if complete < expected_runs and not allow_partial:
        preview = "\n".join(incomplete[:12])
        raise SystemExit(
            f"{name} is not complete yet. Re-run with --allow-partial for a preview.\n{preview}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Plot all methods with LDM-parallel softmax and argmax variants.")
    p.add_argument("--softmax-root", type=Path, default=Path("outputs/ldm_parallel"))
    p.add_argument("--argmax-root", type=Path, default=Path("outputs/ldm_parallel_argmax"))
    p.add_argument("--output", type=Path, default=Path("outputs/comparison/plots/all_methods_with_ldm_parallel_variants.png"))
    p.add_argument("--title", default="All methods comparison (5 antigens)")
    p.add_argument("--expected-runs", type=int, default=25)
    p.add_argument("--expected-evals", type=int, default=200)
    p.add_argument("--allow-partial", action="store_true", help="Plot whatever has finished so far.")
    p.add_argument("--skip-argmax", action="store_true", help="Only include the softmax LDM-parallel variant.")
    p.add_argument("--check-only", action="store_true", help="Only print completion status; do not plot.")
    args = p.parse_args()

    require_complete("LDM parallel softmax", args.softmax_root, args.expected_runs, args.expected_evals, args.allow_partial)
    if not args.skip_argmax:
        require_complete("LDM parallel argmax", args.argmax_root, args.expected_runs, args.expected_evals, args.allow_partial)
    if args.check_only:
        return

    methods = [
        Path("outputs/ldm/ldm_ninit20_iter200"),
        args.softmax_root,
        Path("outputs/llm_baseline/llm_baseline_5x5_200"),
        Path("outputs/reproduction/BO_transformed_overlap_optim_res.csv"),
        Path("outputs/reproduction/HEBO_optim_res.csv"),
        Path("outputs/reproduction/TURBO_optim_res.csv"),
        Path("outputs/reproduction/BO_COMBO_optim_res.csv"),
        Path("outputs/reproduction/RS_optim_res.csv"),
    ]
    if not args.skip_argmax:
        methods.insert(2, args.argmax_root)

    labels = [
        "LDM (5 seeds)",
        "LDM+Acq softmax (5 seeds)",
        "Pure LLM (5 seeds)",
        "AntBO (10 seeds)",
        "HEBO (10 seeds)",
        "TURBO (10 seeds)",
        "COMBO (10 seeds)",
        "RS (10 seeds)",
    ]
    if not args.skip_argmax:
        labels.insert(2, "LDM+Acq argmax (5 seeds)")

    colors = [
        "#e41a1c",
        "#000000",
        "#0000cc",
        "#2ca02c",
        "#17becf",
        "#ff9900",
        "#984ea3",
        "#7f7f7f",
    ]
    if not args.skip_argmax:
        colors.insert(2, "#a65628")

    missing = [str(path) for path in methods if not path.exists()]
    if missing:
        raise SystemExit("Missing method paths:\n" + "\n".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/plot_comparison.py",
        "--methods", ",".join(str(path) for path in methods),
        "--labels", ",".join(labels),
        "--colors", ",".join(colors),
        "--antigens", "test_5_antigens.txt",
        "--output", str(args.output),
        "--title", args.title,
        "--n-column", "3",
        "--font-scale", "0.72",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
