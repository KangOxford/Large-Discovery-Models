"""Plot best-so-far convergence for the 5-seed full-LLM run.

For each antigen, aggregate the ``BestValue`` column across all 5 seeds
(42..46), compute the per-iteration mean and std, and draw a line + ±1σ
shaded band.

Inputs : outputs/full_llm_5seeds_init100000_ninit50_iter100/seed_*/antigen_*/results.csv
Outputs: outputs/comparisons/plots/best_so_far_5seeds_subplots.png
"""
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("outputs/full_llm_5seeds_init100000_ninit50_iter100")
SEEDS = [42, 43, 44, 45, 46]
OUT_DIR = Path("outputs/comparisons/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _antigen_from_path(p: Path) -> str:
    return p.parent.name.split("_kernel_")[0].replace("antigen_", "")


def load_seed_curves(root: Path) -> Dict[str, np.ndarray]:
    """Return ``{antigen: array(n_seeds, T)}`` where T is the iteration horizon.

    Skips antigens that have fewer seed CSVs than expected (with a warning).
    """
    out: Dict[str, List[np.ndarray]] = {}
    for csv in sorted(root.glob("seed_*/antigen_*/results.csv")):
        ag = _antigen_from_path(csv)
        out.setdefault(ag, []).append(pd.read_csv(csv)["BestValue"].values)

    curves: Dict[str, np.ndarray] = {}
    for ag, series_list in out.items():
        if len(series_list) != len(SEEDS):
            print(f"warning: antigen {ag} has {len(series_list)} seed CSVs, "
                  f"expected {len(SEEDS)}")
            continue
        curves[ag] = np.stack(series_list)
    return curves


def plot_subplots(curves: Dict[str, np.ndarray], out_path: Path) -> None:
    n = len(curves)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for ax, (ag, Y) in zip(axes.flat, curves.items()):
        mean = Y.mean(0)
        std = Y.std(0)
        x = np.arange(Y.shape[1])
        ax.plot(x, mean, color="C0", lw=2, label=f"mean ({Y.shape[0]} seeds)")
        ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.3,
                        label="±1 std")
        ax.set_title(ag)
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("Best-so-far ΔG (lower = better)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle("AntBO-LDM best-so-far convergence, 5 seeds (mean ± std)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


def main() -> None:
    curves = load_seed_curves(ROOT)
    if not curves:
        raise SystemExit(f"no seed CSVs found under {ROOT}")
    for ag, Y in curves.items():
        print(f"  {ag}: {Y.shape[0]} seeds × {Y.shape[1]} iterations, "
              f"final mean = {Y[:, -1].mean():.3f}")
    plot_subplots(curves, OUT_DIR / "best_so_far_5seeds_subplots.png")


if __name__ == "__main__":
    main()