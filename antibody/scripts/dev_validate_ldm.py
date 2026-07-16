#!/usr/bin/env python
"""Quick LDM validation with detailed output and side-by-side comparison.

Writes its own BO loop (referencing bo/main.py) for full control over printing.
No state saving — purely for fast iteration and debugging.

Usage:
    # Default: 20 init + 50 BO iterations, LLM init+loop enabled
    python scripts/dev_validate_ldm.py --antigen 1ADQ_A

    # Quick smoke
    python scripts/dev_validate_ldm.py --antigen 1ADQ_A --n-init-iter 2 --n-bo-iter 2

    # LLM init only (plain BO loop)
    python scripts/dev_validate_ldm.py --antigen 1ADQ_A --no-llm-loop

    # Plain LHS init (LLM loop only)
    python scripts/dev_validate_ldm.py --antigen 1ADQ_A --no-llm-init

    # Run live baseline instead of reading reproduction CSV
    python scripts/dev_validate_ldm.py --antigen 1ADQ_A --run-reproduction

    # Init-only comparison (--n-bo-iter 0)
    python scripts/dev_validate_ldm.py --antigen 1ADQ_A --n-bo-iter 0
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Negative variance.*")
warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")

import numpy as np
import pandas as pd
import torch

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from bo.main import BOExperiments
from bo.optimizer import Optimizer
from bo.utils import get_config

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
REPRO_CSV_DEFAULT = "outputs/reproduction/BO_transformed_overlap_optim_res.csv"


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick LDM validation with detailed output."
    )
    p.add_argument("--antigen", type=str, required=True,
                   help="Antigen PDB ID (e.g. 1ADQ_A) or 'SMOKE' for synthetic function")
    p.add_argument("--n-init-iter", type=int, default=20,
                   help="Number of init evaluations (default 20)")
    p.add_argument("--n-bo-iter", type=int, default=50,
                   help="Number of BO iterations after init (default 50; 0 = init-only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batchsize", type=int, default=1,
                   help="Batch size for BO (default 1)")
    p.add_argument("--config", type=str, default=None,
                   help="Config YAML (default: bo/config.yaml)")
    p.add_argument("--llm-init", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable/disable LLM-guided init (default: enabled)")
    p.add_argument("--llm-loop", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable/disable LLM orchestrator during BO loop (default: enabled)")
    p.add_argument("--llm-strategy", type=str, default="ldm-default",
                   choices=["ldm-default", "antbo-mock"],
                   help="LLM strategy prompt (default: ldm-default)")
    p.add_argument("--run-reproduction", action="store_true",
                   help="Run live baseline (no LLM) instead of reading reproduction CSV")
    p.add_argument("--reproduction-csv", type=str, default=REPRO_CSV_DEFAULT,
                   help="Aggregate reproduction CSV (ignored if --run-reproduction)")
    return p.parse_args()


# ── Experiment runner ───────────────────────────────────────────────

def run_experiment(
    args: argparse.Namespace,
    llm_init: bool,
    llm_loop: bool,
    label: str,
) -> list[float]:
    """Run a single BO experiment and return best-so-far curve."""

    n_init = args.n_init_iter
    n_bo_iter = args.n_bo_iter
    max_iters = n_init + n_bo_iter
    seed = args.seed

    # 1. Load config with overrides
    config_path = args.config or os.path.join(ROOT_PROJECT, "bo", "config.yaml")
    config = get_config(config_path)
    config['bbox']['antigen'] = args.antigen
    config['n_init'] = n_init
    config['max_iters'] = max_iters
    config['batch_size'] = args.batchsize
    config['save_path'] = f'/tmp/dev_validate_{label}_{args.antigen}_{seed}'
    os.makedirs(config['save_path'], exist_ok=True)

    # Override LLM flags
    if 'llm' not in config:
        config['llm'] = {}
    config['llm']['llm_init_enabled'] = llm_init
    config['llm']['llm_loop_enabled'] = llm_loop
    config['llm']['strategy'] = args.llm_strategy

    print(f"\n{'='*70}")
    print(f"  [{label}] llm_init={llm_init} llm_loop={llm_loop}")
    print(f"  {n_init} init + {n_bo_iter} BO = {max_iters} total evals")
    print(f"{'='*70}\n")

    # 2. Create BOExperiments (handles BOTask, antigen context, etc.)
    boexp = BOExperiments(config, cdr_constraints=True, seed=seed)

    # 3. Run custom BO loop
    best_so_far = _run_dev_loop(boexp, n_init, n_bo_iter, label)

    # 4. Cleanup
    shutil.rmtree(config['save_path'], ignore_errors=True)

    return best_so_far


def _run_dev_loop(
    boexp: BOExperiments, n_init: int, n_bo_iter: int, label: str,
) -> list[float]:
    """Custom BO loop with detailed LLM decision printing."""

    max_iters = n_init + n_bo_iter
    seed = boexp.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Build orchestrator (returns None if both flags are False)
    orchestrator = boexp._build_orchestrator()

    # 2. Build LLM initial points (returns None if llm_init_enabled is False)
    boexp.llm_initial_points = boexp._build_llm_initial_points(orchestrator)

    # 3. Build kwargs for Optimizer
    _llm_cfg = boexp.config.get('llm', {})
    kwargs = {
        'length_max_discrete': boexp.config['seq_len'],
        'device': boexp.config['device'],
        'seed': seed,
        'search_strategy': boexp.search_strat,
        'BERT_model_path': boexp.config.get('BERT_model_path', 'Rostlab/prot_bert_bfd'),
        'BERT_tokeniser_path': boexp.config.get('BERT_tokenizer_path', 'Rostlab/prot_bert_bfd'),
        'BERT_batchsize': boexp.config.get('BERT_batchsize', 128),
        'antigen_id': boexp.config['bbox']['antigen'],
        'antigen_seed': seed,
        'antigen_context': boexp.antigen_context or {},
        # LLM config keys read by localbo_cat.py DSLConfig construction
        'acq_search_budget': _llm_cfg.get('acq_search_budget', 600),
        'acq_max_rounds': _llm_cfg.get('acq_max_rounds', 3),
        'num_llm_review': _llm_cfg.get('num_llm_review', 10),
        'max_retries': _llm_cfg.get('max_retries', 3),
        'bias_weight': _llm_cfg.get('bias_weight', 0.05),
        'sample_timeout_s': _llm_cfg.get('sample_timeout_s', 5.0),
        'llm_loop_enabled': _llm_cfg.get('llm_loop_enabled', True),
        'strategy': _llm_cfg.get('strategy', 'ldm-default'),
    }
    # Only pass orchestrator if llm_loop_enabled
    if _llm_cfg.get('llm_loop_enabled', True):
        kwargs['orchestrator'] = orchestrator
    else:
        kwargs['orchestrator'] = None

    # 4. Construct Optimizer
    optim = Optimizer(
        config=boexp.n_categories,
        min_cuda=boexp.config['min_cuda'],
        n_init=n_init,
        use_ard=boexp.config['ard'],
        acq=boexp.config['acq'],
        cdr_constraints=boexp.cdr_constraints,
        normalise=boexp.config['normalise'],
        kernel_type=boexp.config['kernel_type'],
        noise_variance=float(boexp.config['noise_variance']),
        alphabet_size=boexp.nb_aas,
        table_of_candidates=None,
        table_of_candidate_embeddings=None,
        embedding_from_array_dict=None,
        initial_points=boexp.llm_initial_points,
        **kwargs,
    )

    # 5. Run loop
    best_so_far: list[float] = []

    for itern in range(max_iters):
        start = time.time()
        x_next = optim.suggest(n_suggestions=1)
        y_next = boexp.f_obj.compute(x=x_next)
        optim.observe(x=x_next, y=y_next)
        elapsed = time.time() - start

        y_val = float(y_next[0].item() if hasattr(y_next[0], 'item') else y_next[0])
        seq_str = ''.join(AA_ALPHABET[int(x)] for x in x_next[0])
        cumul_y = np.array(optim.casmopolitan.fx).flatten()
        best_val = float(cumul_y.min())
        best_so_far.append(best_val)

        phase = "INIT" if itern < n_init else "BO"
        # Determine path from config (clean — not runtime detection)
        _ldm_loop = _llm_cfg.get('llm_loop_enabled', True)
        if _ldm_loop:
            print(f"[{label} {phase} {itern + 1}/{max_iters}] ({elapsed:.1f}s) "
                  f"seq={seq_str} y={y_val:.4f} best={best_val:.4f} "
                  f"[Path: B - LDM session]", flush=True)
        else:
            print(f"[{label} {phase} {itern + 1}/{max_iters}] ({elapsed:.1f}s) "
                  f"seq={seq_str} y={y_val:.4f} best={best_val:.4f} "
                  f"[Path: A - original AntBO]", flush=True)

        decision = getattr(optim.casmopolitan, '_last_decision', None)
        if decision is not None:
            _print_llm_decision(decision)

        session = getattr(optim.casmopolitan, '_last_session', None)
        if session is not None:
            _print_session(session)

    return best_so_far


def _print_llm_decision(decision) -> None:
    prefix = "    "
    if decision.source == 'cache':
        return
    if decision.fallback_used:
        print(f"{prefix}LLM: FALLBACK - {decision.rejection_reason}", flush=True)
        return
    if decision.source == 'llm':
        parts = []
        if decision.rationale:
            parts.append(f"rationale: {decision.rationale}")
        if decision.search_updated and decision.search_dsl is not None:
            parts.append(f"TR updated: {decision.search_dsl!r}")
        elif decision.search_dsl is not None:
            parts.append(f"TR kept: {decision.search_dsl!r}")
        if decision.bias_updated and decision.bias_dsl is not None:
            parts.append(f"bias updated: {decision.bias_dsl!r}")
        elif decision.bias_dsl is not None:
            parts.append(f"bias kept: {decision.bias_dsl!r}")
        if parts:
            for p in parts:
                print(f"{prefix}{p}", flush=True)
        else:
            print(f"{prefix}LLM: {{}} (no update)", flush=True)


def _print_session(session) -> None:
    """Print detailed session: review prompts, LLM responses, all candidates."""
    prefix = "    "
    for i, rnd in enumerate(session.rounds):
        print(f"{prefix}Round {rnd.round_idx + 1}: budget_used={rnd.budget_used}", flush=True)
        atoms_str = rnd.atoms_repr[:150]
        if len(rnd.atoms_repr) > 150:
            atoms_str += "..."
        print(f"{prefix}  atoms: {atoms_str}", flush=True)
        print(f"{prefix}  evaluated: {rnd.n_evaluated} points", flush=True)

        # Show review prompt (first ~6 lines, truncated)
        if i < len(session.review_prompts):
            prompt = session.review_prompts[i]
            print(f"{prefix}  review prompt (preview):", flush=True)
            for line in prompt.split("\n")[:8]:
                print(f"{prefix}    {line[:120]}", flush=True)

        # Show LLM response
        if i < len(session.llm_review_responses):
            resp = session.llm_review_responses[i]
            print(f"{prefix}  LLM response: {resp[:200]}", flush=True)

        # Show top-k candidates (all of them, not just 5)
        if rnd.review_topk:
            print(f"{prefix}  top-{len(rnd.review_topk)} candidates by bias+{session.acq_name}:", flush=True)
            for idx, r in enumerate(rnd.review_topk):
                seq = r.get("seq_str", "???")
                acq_v = r.get(session.acq_name, 0)
                mu = r.get("mu", 0)
                sig = r.get("sigma", 0)
                bias = r.get("bias", 0)
                comb = r.get(f"bias+{session.acq_name}", 0)
                print(f"{prefix}    [{idx:>3}] {seq} {session.acq_name}={acq_v:.3f} "
                      f"mu={mu:.1f} σ={sig:.1f} bias={bias:.2f} comb={comb:.3f}", flush=True)
        if rnd.taken_ids:
            print(f"{prefix}  → {rnd.llm_action}: ids={rnd.taken_ids}", flush=True)
        if rnd.llm_rationale:
            print(f"{prefix}  rationale: {rnd.llm_rationale[:120]}", flush=True)


# ── Reproduction comparison ─────────────────────────────────────────

def load_reproduction(
    csv_path: str,
    antigen: str,
    max_evals: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if not os.path.exists(csv_path):
        print(f"  (reproduction CSV not found: {csv_path})")
        return None

    df = pd.read_csv(csv_path)
    sub = df[df["Antigen"] == antigen]
    if sub.empty:
        print(f"  (antigen {antigen} not found in reproduction CSV)")
        return None

    series_list = []
    for _, g in sub.groupby("Seed"):
        g = g.sort_values("Num BB Evals")
        y = pd.to_numeric(g["Best Binding Energy"], errors="coerce").values
        series_list.append(y)

    if not series_list:
        return None

    n_seeds = len(series_list)
    min_len = min(len(s) for s in series_list)
    stacked = np.stack([s[:min_len] for s in series_list])
    if max_evals is not None:
        min_len = min(min_len, max_evals)
        stacked = stacked[:, :min_len]

    return stacked.mean(0), stacked.std(0), n_seeds


# ── Comparison tables ───────────────────────────────────────────────

def print_comparison_csv(
    ldm_best: list[float],
    repro: tuple[np.ndarray, np.ndarray, int] | None,
    n_init: int,
    antigen: str,
) -> None:
    n_ldm = len(ldm_best)
    repro_mean = repro[0] if repro is not None else None
    repro_std = repro[1] if repro is not None else None
    repro_n_seeds = repro[2] if repro is not None else 0
    n_repro_evals = len(repro_mean) if repro_mean is not None else 0
    n = max(n_ldm, n_repro_evals)

    w = 12
    header = (
        f"{'Eval':>5} | {'Phase':>5} | {'LDM':>{w}} | "
        f"{'Repro mean':>{w}} | {'±std':>{w-3}} | {'Delta':>8}"
    )
    sep = "-" * len(header)

    seed_label = f"{repro_n_seeds} repro seeds" if repro_n_seeds else "no repro"
    print(f"\n{'='*len(header)}")
    print(f"  Comparison: {antigen}  (LDM 1 seed vs {seed_label})")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for i in range(n):
        phase = "INIT" if i < n_init else "BO"
        ldm_str = f"{ldm_best[i]:.2f}" if i < n_ldm else "---"
        if repro_mean is not None and i < len(repro_mean):
            rm = f"{repro_mean[i]:.2f}"
            rs = f"±{repro_std[i]:.2f}"
            delta = f"{ldm_best[i] - repro_mean[i]:+.2f}" if i < n_ldm else "---"
        else:
            rm, rs, delta = "---", "", "---"
        print(f"{i+1:>5} | {phase:>5} | {ldm_str:>{w}} | {rm:>{w}} | {rs:>{w-3}} | {delta:>8}")

    print(sep)
    if n_ldm > 0 and repro_mean is not None:
        idx = min(n_ldm - 1, len(repro_mean) - 1)
        print(f"  Final (eval {n_ldm}): LDM={ldm_best[-1]:.2f}, "
              f"Repro={repro_mean[idx]:.2f}, "
              f"Delta={ldm_best[-1] - repro_mean[idx]:+.2f}")
    print()


def print_comparison_runs(
    ldm_best: list[float],
    baseline_best: list[float],
    n_init: int,
    antigen: str,
    ldm_label: str,
) -> None:
    n = max(len(ldm_best), len(baseline_best))
    w = 12

    header = (
        f"{'Eval':>5} | {'Phase':>5} | {ldm_label:>{w}} | "
        f"{'Baseline':>{w}} | {'Delta':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  Comparison: {antigen}  ({ldm_label} vs Baseline, same seed)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for i in range(n):
        phase = "INIT" if i < n_init else "BO"
        a = f"{ldm_best[i]:.2f}" if i < len(ldm_best) else "---"
        b = f"{baseline_best[i]:.2f}" if i < len(baseline_best) else "---"
        if i < len(ldm_best) and i < len(baseline_best):
            delta = f"{ldm_best[i] - baseline_best[i]:+.2f}"
        else:
            delta = "---"
        print(f"{i+1:>5} | {phase:>5} | {a:>{w}} | {b:>{w}} | {delta:>8}")

    print(sep)
    if ldm_best and baseline_best:
        idx = min(len(ldm_best), len(baseline_best)) - 1
        print(f"  Final (eval {idx+1}): {ldm_label}={ldm_best[idx]:.2f}, "
              f"Baseline={baseline_best[idx]:.2f}, "
              f"Delta={ldm_best[idx] - baseline_best[idx]:+.2f}")
    print()


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    max_iters = args.n_init_iter + args.n_bo_iter
    if max_iters == 0:
        print("ERROR: n_init_iter + n_bo_iter = 0; nothing to do.")
        return

    # Build label for the main run
    flags = []
    if args.llm_init:
        flags.append("init")
    if args.llm_loop:
        flags.append("loop")
    ldm_label = "LDM(" + "+".join(flags) + ")" if flags else "LDM(none)"

    # 1. Run main experiment
    ldm_best = run_experiment(args, args.llm_init, args.llm_loop, ldm_label)

    # 2. Baseline: live run or CSV
    if args.run_reproduction:
        baseline_best = run_experiment(args, llm_init=False, llm_loop=False, label="BASELINE")
        print_comparison_runs(ldm_best, baseline_best, args.n_init_iter,
                              args.antigen, ldm_label)
    else:
        repro = load_reproduction(args.reproduction_csv, args.antigen, max_evals=max_iters)
        print_comparison_csv(ldm_best, repro, args.n_init_iter, args.antigen)


if __name__ == "__main__":
    main()
