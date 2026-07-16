"""End-to-end smoke test for strbo_v1 search loops.

Exercises every supported n_obj path with deterministic mock scorers
and a mock analog generator. No GPU, no Vina, no NN model, no
ReaSyn — the script must run on any environment with the strbo_v1
package installed.

Six scenarios are run, one per (method, n_obj) cell:

    n_obj = 1  → single-obj EI on Vina-style mock (minimise)
    n_obj = 1  → single-obj random search
    n_obj = 2  → multi-obj BO with EHVI (Vina + NN mocks)
    n_obj = 2  → multi-obj random search with Chebyshev-ParEGO expansion
    n_obj = 3  → multi-obj BO with Chebyshev ParEGO (Vina + NN + mock3)
    n_obj = 3  → multi-obj random search with Chebyshev-ParEGO expansion

Each scenario:

* Builds a flat ``analog_fn`` (no DataFrame, no ReaSyn).
* Builds a tuple of mock scorers (or a single one for n_obj=1).
* Calls the search loop with deterministic seed.
* Asserts the loop completed (no exception, history length > 0).
* Prints a one-line summary so the script's stdout doubles as a
  manual smoke check.

Exit code 0 if all 6 scenarios pass; non-zero on the first failure.

Usage::

    python search_smoke.py              # all 6 scenarios, CPU
    python search_smoke.py --quick      # smaller n_evaluations (faster)
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from typing import Callable, List, Optional, Tuple

from strbo_v1 import (
    BayesianAnalogSearchConfig,
    DEFAULT_REF,
    Scorers,
    bayesian_analog_search,
    random_analog_search,
    resolve_ref_point,
)
from strbo_v1.gp import GPConfig


# ---------------------------------------------------------------------------
# Mock scoring functions (deterministic, no I/O, no model file)
# ---------------------------------------------------------------------------
#
# The mock scorers are constructed so the multi-objective Pareto front
# is *non-trivial* (the loop should actually have to make trade-offs
# rather than just collapse to a single-objective optimum).
#
# Score ranges (per SMILES):
#   vina_mock:  negative, more C → more negative (i.e. better)
#   nn_mock:    ~5.0, more N → higher pIC50 (i.e. better)
#   mock3:      small, more O → larger (i.e. better)
#
# Vina and NN are intentionally **anti-correlated** in the seed set
# (CCO has more C → better Vina, fewer N → worse NN; CCN has more N
# → better NN, fewer C → worse Vina) so the Pareto front has both
# endpoints, exercising the EHVI / Chebyshev path with real
# candidate trade-offs.


def vina_mock(smis: List[str]) -> List[float]:
    """Mock Vina: kcal/mol, lower is better. Favours more carbons."""
    return [-float(s.count("C")) - 0.1 * float(s.count("N")) for s in smis]


def nn_mock(smis: List[str]) -> List[float]:
    """Mock NN: pIC50, higher is better. Favours more nitrogens."""
    return [
        5.0 + 0.5 * float(s.count("N")) + 0.1 * float(s.count("C"))
        for s in smis
    ]


def mock3(smis: List[str]) -> List[float]:
    """Third mock: arbitrary, higher is better. Favours more oxygens."""
    return [1.0 + 0.3 * float(s.count("O")) + 0.1 * float(s.count("C")) for s in smis]


# ---------------------------------------------------------------------------
# Mock analog generators (deterministic, no ReaSyn)
# ---------------------------------------------------------------------------
#
# Each generator takes a list of input SMILES and returns a flat list
# of analogue SMILES. The signature matches the search-loop contract
# (`Iterable[str] -> Iterable[str]`); no DataFrame is involved.


def chain_analog(smis: List[str]) -> List[str]:
    """For each input SMILES, append 'C' (single-char extension).

    Deterministic, no duplicates across the run (the pool's
    ``FIFOSet`` + ``seen`` set dedup at ingestion).
    """
    return [s + "C" for s in smis]


def branch_analog(smis: List[str]) -> List[str]:
    """For each input SMILES, produce three branches: +C, +N, +O."""
    out: List[str] = []
    for s in smis:
        out.extend([s + "C", s + "N", s + "O"])
    return out


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def _build_scorer_tuple(parts: List[str]) -> Tuple[Callable, ...]:
    """Map ``objective_parts`` to a tuple of mock scorer callables."""
    table = {
        "vina": vina_mock,
        "nn": nn_mock,
        "mock3": mock3,
    }
    return tuple(table[p] for p in parts)


def _print_summary(
    n_obj: int,
    method: str,
    parts: List[str],
    history: list,
    n_evals: int,
) -> None:
    """Print a one-line summary for one scenario."""
    if n_obj == 1:
        scores = [sc for _, sc in history if sc is not None]
        if not scores:
            print(f"  [n_obj={n_obj} {method:>6} parts={parts}] NO FINITE SCORES")
            return
        best = min(scores) if parts[0] == "vina" else max(scores)
        print(
            f"  [n_obj={n_obj} {method:>6} parts={parts}] "
            f"history={len(history):>2} evals (target {n_evals}), best={best:.3f}"
        )
        return
    # n_obj >= 2
    finite = [
        sc for _, sc in history
        if sc is not None and len(sc) == n_obj
        and all(s is not None for s in sc)
    ]
    if not finite:
        print(f"  [n_obj={n_obj} {method:>6} parts={parts}] NO FINITE SCORES")
        return
    # Per-objective best (min for Vina/mock3 in parts[0], max for nn).
    for i, part in enumerate(parts):
        vals = [sc[i] for sc in finite]
        minimize = part in ("vina",)
        best = min(vals) if minimize else max(vals)
        print(
            f"    obj{i} ({part}): best={best:.3f} over {len(finite)} finite evals"
        )
    print(
        f"  [n_obj={n_obj} {method:>6} parts={parts}] "
        f"history={len(history):>2} evals (target {n_evals})"
    )


def _run_scenario(
    *,
    label: str,
    method: str,
    parts: List[str],
    n_evaluations: int,
    batch_size: int,
    init_size: int,
    rng_seed: int = 0,
    expansion: str = "random",
    ref_point: Optional[Tuple[float, ...]] = None,
) -> list:
    """Run one (method, n_obj) scenario with the given objective parts.

    Returns the history list (post-loop). Raises on any error.
    """
    rng = random.Random(rng_seed)
    scorers = _build_scorer_tuple(parts)
    n_obj = len(parts)
    minimize_t = tuple(p in ("vina",) for p in parts)

    if method in ("random", "random-best"):
        expansion_used = "best" if method == "random-best" else "random"
        history = random_analog_search(
            seed_smiles=["CCO", "CCN", "CCC"],
            scorer=scorers,
            analog_fn=branch_analog,
            n_iterations=n_evaluations,
            batch_size=batch_size,
            pool_min_size=1,
            pool_max_size=10,
            smiles_max_len=20,
            expansion=expansion_used,
            minimize=minimize_t,
            rng=rng,
            verbose=False,
        )
    elif method in ("bo-tanimoto", "bo-strkernel"):
        impl = (
            "fingerprint+tanimoto" if method == "bo-tanimoto" else "smiles-strkernel"
        )
        gp_cfg = GPConfig(
            impl=impl,
            device="cpu",
            fit_n_itersteps=5,
            min_jitter=1e-6,
            max_jitter=1e-1,
            standardize_y=True,
            smiles_maxlen=20,
        )
        n_iter = max(0, (n_evaluations - init_size) // batch_size)
        bo_cfg = BayesianAnalogSearchConfig(
            init_size=init_size,
            batch_size=batch_size,
            n_iterations=n_iter,
            warmup=False,
            acquisition="ei",
            xi=0.01,
            kappa=2.0,
            minimize=minimize_t,
            acq_budget=None,
            max_pool_size=20,
            smiles_max_len=20,
            gp_config=gp_cfg,
            ref_point=ref_point,
            ehvi_n_samples=32,
            che_alpha=1.0,
            verbose=False,
        )
        history = bayesian_analog_search(
            seed_smiles=["CCO", "CCN", "CCC"],
            scorer=scorers,
            analog_fn=branch_analog,
            config=bo_cfg,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown method {method!r}")

    # Light assertions — the loop is allowed to produce fewer evals
    # than requested (e.g. if the pool exhausts), but never zero when
    # n_evaluations >= 1.
    assert len(history) > 0, (
        f"[{label}] history is empty (got {len(history)} entries)"
    )
    return history


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end smoke test for strbo_v1 search loops. "
            "Runs BO + random search for n_obj in {1, 2, 3} with "
            "deterministic mock scorers and analog generators."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller n_evaluations for a fast run.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed for reproducibility.")
    args = parser.parse_args(argv)

    n_evaluations = 5 if args.quick else 8
    batch_size = 1
    init_size = 3 if n_evaluations >= 6 else max(1, n_evaluations // 3)

    # Scenario matrix.
    scenarios = [
        # (label, method, parts, ref_point)
        ("single-obj BO vina",    "bo-tanimoto",   ["vina"],            None),
        ("single-obj random",     "random",        ["vina"],            None),
        ("2-obj BO vina+nn",      "bo-tanimoto",   ["vina", "nn"],      (0.0, 5.0)),
        ("2-obj random-best",     "random-best",   ["vina", "nn"],      None),
        ("3-obj BO vina+nn+mock3","bo-tanimoto",   ["vina", "nn", "mock3"], None),
        ("3-obj random-best",     "random-best",   ["vina", "nn", "mock3"], None),
    ]

    print("=" * 78)
    print(f"strbo_v1 search smoke (n_evaluations={n_evaluations}, "
          f"batch_size={batch_size}, init_size={init_size}, seed={args.seed})")
    print("=" * 78)
    print()

    failures = 0
    for i, (label, method, parts, ref_point) in enumerate(scenarios):
        # If user didn't supply a ref_point, fall back to the registry
        # so the JSON / config echo matches the documented default.
        if ref_point is None and len(parts) >= 2:
            try:
                ref_point = resolve_ref_point(parts)
            except ValueError as exc:
                print(f"  [FAIL {label}] ref_point resolution: {exc}")
                failures += 1
                continue
        print(f"--- scenario {i+1}/{len(scenarios)}: {label} ---")
        try:
            history = _run_scenario(
                label=label,
                method=method,
                parts=parts,
                n_evaluations=n_evaluations,
                batch_size=batch_size,
                init_size=init_size,
                rng_seed=args.seed + i,
                ref_point=ref_point,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL {label}] {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stdout)
            failures += 1
            continue
        _print_summary(len(parts), method, parts, history, n_evaluations)
        print(f"  [PASS {label}]")
        print()

    print("=" * 78)
    if failures:
        print(f"FAILED: {failures} / {len(scenarios)} scenarios failed")
        return 1
    print(f"OK: all {len(scenarios)} scenarios passed")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
