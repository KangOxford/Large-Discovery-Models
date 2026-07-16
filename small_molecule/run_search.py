"""Run ONE search trajectory (one method, one seed) and write a JSON.

Each invocation executes a single ``--method`` trajectory with the given
``--seed``, then writes a JSON file with two top-level keys:

    {
      "config": { ...full input echo... },
      "history": [
        {"index": 0, "smiles": "CCO", "score": -7.5},
        ...
      ]
    }

The ``config`` echo captures every relevant knob so downstream
post-processing (e.g. ``plot_search_results.py``) and audit can
recover how the run was produced without re-parsing CLI / bash history.

Default output path: ``output/bo/<method>_seed=<seed>.json`` when ``--output``
is a directory, or the exact path supplied when ``--output`` ends in
``.json``.

Methods:

    * ``random``      — uniform random analog search.
    * ``random-best`` — random search with Chebyshev-ParEGO expansion.
    * ``bo-tanimoto`` — BO with Morgan fingerprint + Tanimoto kernel.
    * ``bo-strkernel`` — BO with smiles-subsequence-string kernel.

Scorer / objective backends (``--objective``):

    * ``vina``       — single AutoDock Vina scorer.
    * ``nn``         — single G12D pIC50 NN scorer.
    * ``mock``       — single deterministic mock scorer (no compute).
    * ``vina+nn``    — multi-objective: Vina + NN, uses EHVI (n_obj=2).
    * ``vina+nn+mock`` — multi-objective (n_obj=3), uses Chebyshev-ParEGO.

Per-objective ``minimize`` direction is hard-coded by backend
(``vina`` and ``mock`` minimise; ``nn`` maximises); it is **not** a
CLI flag. The JSON ``config.minimize`` echoes the resulting tuple.

Usage::

    python run_search.py --objective mock --method random --seed 0 \\
        --num-evaluations 30 --output output/bo/random_seed=0.json

    python run_search.py --method bo-strkernel --seed 0 \\
        --num-evaluations 100 --batch-size 3 --init-size 12 \\
        --acq-budget 500 --output output/bo/bo-strkernel_seed=0.json

    python run_search.py --objective nn --method bo-tanimoto --seed 0 \\
        --num-evaluations 30 --output output/bo/nn_seed=0.json

    python run_search.py --objective vina+nn --method bo-tanimoto \\
        --seed 0 --num-evaluations 30 \\
        --ref-point 0,5 \\
        --output output/bo/vina_nn_seed=0.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from strbo_v1 import (
    BayesianAnalogSearchConfig,
    Scorer,
    Scorers,
    bayesian_analog_search,
    random_analog_search,
    resolve_ref_point,
)
from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH
from strbo_v1.gp import GPConfig


LOGGER = logging.getLogger("run_search")

VALID_METHODS = ("random", "random-best", "bo-tanimoto", "bo-strkernel",
                 "bo-tanimoto-ldm", "bo-strkernel-ldm")
GP_IMPL = {
    "bo-tanimoto": "fingerprint+tanimoto",
    "bo-strkernel": "smiles-strkernel",
    "bo-tanimoto-ldm": "fingerprint+tanimoto",
    "bo-strkernel-ldm": "smiles-strkernel",
}
LDM_METHODS = ("bo-tanimoto-ldm", "bo-strkernel-ldm")
RANDOM_METHODS = ("random", "random-best")
RANDOM_EXPANSION = {
    "random": "random",
    "random-best": "best",
}

# Per-backend "smaller is better" booleans. Vina scores are kcal/mol
# (more negative = better binding → minimise). NN scores are pIC50
# (higher = more potent → maximise). Mock is a placeholder; minimise.
DEFAULT_MINIMIZE: dict[str, bool] = {
    "vina": True,
    "nn": False,
    "mock": True,
}


# ---------------------------------------------------------------------------
# Mock scorer and analog generator
# ---------------------------------------------------------------------------


def mock_carbon_scorer(smiles_list: list[str]) -> list[float]:
    """Linear in atom counts. Lower = better."""
    out: list[float] = []
    for s in smiles_list:
        c = s.count("C")
        n = s.count("N")
        o = s.count("O")
        out.append(-float(c) - 0.5 * float(n) + 0.3 * float(o))
    return out


def mock_chain_analog_generator(seed_smiles: list[str]) -> list[str]:
    """For each input SMILES, append a single character (C / O / N)."""
    out: list[str] = []
    for s in seed_smiles:
        out.extend([s + "C", s + "O", s + "N"])
    return out


# ---------------------------------------------------------------------------
# Real scorer / analog adapters (Vina + ReaSyn)
# ---------------------------------------------------------------------------


def _build_vina_scorer(args: argparse.Namespace) -> Scorer:
    """Build a :class:`VinaScorer` (already callable: ``scorer(smis) -> list[float]``)."""
    from strbo_v1.objective_vina import VinaScorer, VinaScorerConfig  # local import

    cache_dir = Path(args.vina_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vina_bin = args.vina_bin or os.environ.get("VINA_BIN")
    if vina_bin is None:
        vina_bin = str((Path(__file__).resolve().parent / ".." / "bin" / "vina").resolve())
    cfg = VinaScorerConfig(
        vina_bin=vina_bin,
        cache_dir=cache_dir,
        pdb_id=args.vina_pdb_id,
        chain_id=args.vina_chain_id,
        ligand_resname=args.vina_ligand_resname,
        exhaustiveness=args.vina_exhaustiveness,
        n_poses=args.vina_n_poses,
        seed=args.vina_seed,
        max_workers=args.vina_max_workers,
        allow_debug_receptor=args.vina_allow_debug_receptor,
        use_cache=not args.vina_no_cache,
    )
    return VinaScorer(cfg)


def _build_nn_scorer(args: argparse.Namespace) -> Scorer:
    """Build an :class:`NNScorer` using the configured G12D model path."""
    from strbo_v1.objective_nn import NNScorer, NNScorerConfig  # local import

    cfg = NNScorerConfig(
        model_path=args.nn_model_path,
        metadata_path=args.nn_metadata_path,
        on_error="all_nan",
    )
    return NNScorer(cfg)


def _build_reasyn_analog(args: argparse.Namespace) -> Callable[[list[str]], list[str]]:
    from strbo_v1.analog import ReasynConfig, generate_analogs  # local import

    model_path = args.reasyn_model_path or os.environ.get("REASYN_MODEL_PATH")
    devices = [int(d) for d in args.reasyn_devices.split(",") if d.strip()]
    reasyn_repo = args.reasyn_repo or os.environ.get("REASYN_HOME") or os.environ.get("REASYN_REPO")
    python_bin = args.reasyn_python_bin or os.environ.get("REASYN_PYTHON") or os.environ.get("REASYN_BIN")

    if model_path is None:
        model_path = (
            "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,"
            "data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
        )

    cfg = ReasynConfig(
        model_path=model_path,
        reasyn_repo=reasyn_repo,
        python_bin=python_bin,
        devices=devices,
        search_width=args.reasyn_search_width,
        exhaustiveness=args.reasyn_exhaustiveness,
        num_cycles=args.reasyn_num_cycles,
        num_editflow_samples=args.reasyn_num_editflow_samples,
        num_editflow_steps=args.reasyn_num_editflow_steps,
        time_limit=args.reasyn_time_limit,
        num_workers_per_gpu=args.reasyn_num_workers_per_gpu,
        filter_sim=args.reasyn_filter_sim,
        canonicalize=not args.reasyn_no_canonicalize,
    )

    def analog_fn(smis: list[str]) -> list[str]:
        df = generate_analogs(smis, cfg)
        if df is None or len(df) == 0:
            return []
        return df["smiles"].tolist()

    return analog_fn


def _build_one_scorer(part: str, args: argparse.Namespace) -> Scorer:
    """Build a single scorer for one ``--objective`` part."""
    if part == "vina":
        return _build_vina_scorer(args)
    if part == "nn":
        return _build_nn_scorer(args)
    if part == "mock":
        return mock_carbon_scorer
    raise ValueError(
        f"Unknown objective {part!r}; expected one of 'vina', 'nn', 'mock'"
    )


def _parse_objective(text: str) -> List[str]:
    """Parse ``--objective`` like ``"vina+nn"`` into a list of parts."""
    parts = [p.strip() for p in text.split("+") if p.strip()]
    if not parts:
        raise ValueError("--objective is empty; expected at least one of vina/nn/mock")
    for p in parts:
        if p not in DEFAULT_MINIMIZE:
            raise ValueError(
                f"Unknown objective part {p!r}; expected one of {list(DEFAULT_MINIMIZE)}"
            )
    return parts


def _resolve_ldm_sys_prompt(path_or_text: str) -> str:
    """Resolve ``--ldm-sys-prompt`` to its final string value.

    If ``path_or_text`` is the path of an existing file, the file's
    contents are returned. Otherwise the string is returned verbatim
    (treated as inline text). Empty / unset returns "".

    The resolution is intentionally simple: ``os.path.isfile`` only.
    This means a missing file is *not* an error — the literal string
    is used as the supplement. The user can therefore pass inline
    text by setting ``--ldm-sys-prompt "inline text"``; if the text
    happens to match a real filename, the file's content is used
    instead (rare in practice).
    """
    if not path_or_text:
        return ""
    if os.path.isfile(path_or_text):
        try:
            with open(path_or_text, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError as exc:
            LOGGER.warning(
                "--ldm-sys-prompt: failed to read %s (%s); using literal "
                "string as inline text",
                path_or_text, exc,
            )
            return path_or_text
    return path_or_text


# ---------------------------------------------------------------------------
# Single-trajectory runner
# ---------------------------------------------------------------------------


def _build_scorers_and_minimize(
    args: argparse.Namespace,
) -> Tuple[Scorers, Tuple[bool, ...]]:
    """Build the scorer tuple and the matching ``minimize`` tuple.

    Returns:
        ``(scorers, minimize_t)`` where ``scorers`` is either a single
        :class:`Scorer` (when ``--objective`` is one part) or a tuple
        of scorers (when ``--objective`` contains ``+``), and
        ``minimize_t`` is the matching tuple of per-scorer "smaller
        is better" booleans derived from :data:`DEFAULT_MINIMIZE`.
    """
    parts = _parse_objective(args.objective)
    scorers_list = [_build_one_scorer(p, args) for p in parts]
    minimize_t = tuple(DEFAULT_MINIMIZE[p] for p in parts)
    if len(scorers_list) == 1:
        return scorers_list[0], minimize_t
    return tuple(scorers_list), minimize_t


def run_one(
    method: str,
    seed: int,
    seed_smiles: Sequence[str],
    scorer: Scorers,
    analog_fn: Callable[[list[str]], list[str]],
    args: argparse.Namespace,
    minimize: Tuple[bool, ...],
    ref_point: Optional[Tuple[float, ...]],
) -> Tuple[list, Optional[dict]]:
    """Run one trajectory for ``method`` with the given RNG ``seed``.

    Returns:
        ``(history, trajectory)`` where ``history`` is the list of
        ``(smiles, score_or_scores)`` tuples (same shape as
        ``bayesian_analog_search``'s return value) and ``trajectory``
        is the LLM advisor's per-round trajectory dict, or ``None``
        for non-LDM methods. The trajectory is merged into the
        main run JSON under the top-level ``"llm_trajectory"`` key
        by :func:`write_json`.
    """
    rng = random.Random(seed)
    if method in RANDOM_METHODS:
        return random_analog_search(
            seed_smiles=seed_smiles,
            scorer=scorer,
            analog_fn=analog_fn,
            n_iterations=args.num_evaluations,
            batch_size=args.batch_size,
            pool_min_size=args.pool_min_size,
            pool_max_size=args.pool_max_size,
            smiles_max_len=args.smiles_max_len,
            expansion=RANDOM_EXPANSION[method],
            minimize=minimize,
            rng=rng,
            verbose=args.verbose,
        ), None
    # Check LDM BEFORE GP_IMPL because the LDM methods are also in
    # GP_IMPL (they share the GP impl by suffix). The LDM branch
    # wraps the same GP+acquisition step in a two-phase LLM loop.
    if method in LDM_METHODS:
        return _run_ldm_branch(
            method, seed, seed_smiles, scorer, analog_fn, args,
            minimize=minimize, ref_point=ref_point,
        )
    if method in GP_IMPL:
        n_iter = max(0, (args.num_evaluations - args.init_size) // args.batch_size)
        if n_iter == 0:
            LOGGER.warning(
                "method=%s seed=%d: num_evaluations=%d, init_size=%d, batch_size=%d -> 0 BO rounds",
                method, seed, args.num_evaluations, args.init_size, args.batch_size,
            )
        gp_cfg = GPConfig(
            impl=GP_IMPL[method],
            device=args.gp_device,
            fit_n_itersteps=args.gp_fit_itersteps,
            learning_rate=args.gp_learning_rate,
            min_jitter=args.gp_min_jitter,
            max_jitter=args.gp_max_jitter,
            standardize_y=args.gp_standardize_y,
            fp_radius=args.gp_fp_radius,
            fp_n_bits=args.gp_fp_n_bits,
            smiles_maxlen=args.smiles_max_len,
        )
        bo_cfg = BayesianAnalogSearchConfig(
            init_size=args.init_size,
            batch_size=args.batch_size,
            n_iterations=n_iter,
            warmup=True,
            acquisition=args.acquisition,
            xi=args.xi,
            kappa=args.kappa,
            minimize=minimize,
            acq_budget=args.acq_budget,
            max_pool_size=args.max_pool_size,
            smiles_max_len=args.smiles_max_len,
            gp_config=gp_cfg,
            ref_point=ref_point,
            ehvi_n_samples=args.ehvi_n_samples,
            che_alpha=args.che_alpha,
            verbose=args.verbose,
        )
        return bayesian_analog_search(
            seed_smiles=seed_smiles,
            scorer=scorer,
            analog_fn=analog_fn,
            config=bo_cfg,
            rng=rng,
        ), None
    raise ValueError(f"Unknown method {method!r}; expected one of {VALID_METHODS}")


def _build_llm_advisor(args: argparse.Namespace):
    """Construct the (LLM client, reasyn pool) pair for LDM runs.

    Reads ``LLM_API_KEY`` / ``LLM_BASE_URL`` from ``os.environ`` (the
    ``.env`` file is loaded by :func:`load_env`). ``--llm-api-key``
    and ``--llm-base-url`` override the env values when non-empty.
    The model comes from ``--llm-model`` (default:
    :data:`strbo_v1.llm_advisor.config.DEFAULT_LLM_MODEL`); there
    is no ``LLM_MODEL`` env var.
    """
    from strbo_v1.llm_advisor.config import (
        DEFAULT_LLM_MODEL, LLMClientConfig, load_env,
    )
    from strbo_v1.llm_advisor.client import OpenAIChatClient
    from strbo_v1.llm_advisor.reasyn_pool import ReasynConfigPool

    load_env()
    api_key = (getattr(args, "llm_api_key", "") or "").strip() or (
        os.environ.get("LLM_API_KEY", "") or ""
    ).strip()
    base_url = (getattr(args, "llm_base_url", "") or "").strip() or (
        os.environ.get("LLM_BASE_URL", "") or ""
    ).strip()
    if base_url:
        base_url = base_url.rstrip("/")
    model = (getattr(args, "llm_model", "") or "").strip() or DEFAULT_LLM_MODEL

    if not api_key:
        raise SystemExit(
            "LLM advisor: LLM_API_KEY is empty. Set it in .env, pass "
            "--llm-api-key, or export LLM_API_KEY in the environment."
        )
    if not base_url:
        raise SystemExit(
            "LLM advisor: LLM_BASE_URL is empty. Set it in .env, pass "
            "--llm-base-url, or export LLM_BASE_URL in the environment."
        )

    llm_cfg = LLMClientConfig(api_key=api_key, base_url=base_url, model=model)
    llm = OpenAIChatClient(llm_cfg, timeout=args.llm_timeout)
    pool = ReasynConfigPool.from_env()
    return llm, pool


def _run_ldm_branch(
    method: str,
    seed: int,
    seed_smiles: Sequence[str],
    scorer: Scorers,
    analog_fn: Callable[[list[str]], list[str]],
    args: argparse.Namespace,
    *,
    minimize: Tuple[bool, ...],
    ref_point: Optional[Tuple[float, ...]],
) -> Tuple[list, Optional[dict]]:
    """Run one ``bo-*-ldm`` trajectory.

    Returns ``(history, trajectory)`` in the same shape as
    :func:`run_one`.
    """
    from strbo_v1.bayesian_ldm_search import (
        BayesianLDMSearchConfig, bayesian_ldm_search,
    )

    n_iter = max(0, (args.num_evaluations - args.init_size) // args.batch_size)
    if n_iter == 0:
        LOGGER.warning(
            "method=%s seed=%d: num_evaluations=%d, init_size=%d, batch_size=%d -> 0 BO rounds",
            method, seed, args.num_evaluations, args.init_size, args.batch_size,
        )
    gp_cfg = GPConfig(
        impl=GP_IMPL[method],
        device=args.gp_device,
        fit_n_itersteps=args.gp_fit_itersteps,
        learning_rate=args.gp_learning_rate,
        min_jitter=args.gp_min_jitter,
        max_jitter=args.gp_max_jitter,
        standardize_y=args.gp_standardize_y,
        fp_radius=args.gp_fp_radius,
        fp_n_bits=args.gp_fp_n_bits,
        smiles_maxlen=args.smiles_max_len,
    )
    bo_cfg = BayesianAnalogSearchConfig(
        init_size=args.init_size,
        batch_size=args.batch_size,
        n_iterations=n_iter,
        warmup=True,
        acquisition=args.acquisition,
        xi=args.xi,
        kappa=args.kappa,
        minimize=minimize,
        acq_budget=args.acq_budget,
        max_pool_size=args.max_pool_size,
        smiles_max_len=args.smiles_max_len,
        gp_config=gp_cfg,
        ref_point=ref_point,
        ehvi_n_samples=args.ehvi_n_samples,
        che_alpha=args.che_alpha,
        verbose=args.verbose,
    )
    llm, _pool_unused = _build_llm_advisor(args)
    _ = _pool_unused  # LDM uses its own internal pool via bayesian_ldm_search

    # Build the objective legend from the parts. Helps the LLM
    # interpret the score direction.
    parts = _parse_objective(args.objective)
    objective_legend: list = [
        {"name": p, "minimize": bool(DEFAULT_MINIMIZE[p])}
        for p in parts
    ]

    # Always record a trajectory so the main JSON includes the
    # ``llm_trajectory`` key. If the user didn't pass
    # ``--llm-trajectory-dir``, use a temp dir and clean up after.
    user_traj_dir = getattr(args, "llm_trajectory_dir", "") or None
    cleanup_traj_dir: Optional[str] = None
    if user_traj_dir is None:
        import tempfile
        cleanup_traj_dir = tempfile.mkdtemp(prefix="ldm_traj_")
        trajectory_dir = cleanup_traj_dir
    else:
        trajectory_dir = user_traj_dir

    # Build the LDM config. Reuse the LLMClientConfig we already
    # validated; don't rely on the constructed OpenAIChatClient
    # having a .config attribute (mock clients may not).
    from strbo_v1.llm_advisor.config import LLMClientConfig
    llm_client_cfg = LLMClientConfig(
        api_key=getattr(args, "llm_api_key", "") or os.environ.get("LLM_API_KEY", ""),
        base_url=(getattr(args, "llm_base_url", "") or os.environ.get("LLM_BASE_URL", "")).rstrip("/"),
        model=getattr(args, "llm_model", None) or DEFAULT_LLM_MODEL,
    )

    # Pool-min size: user-supplied value wins, else auto-set to
    # --batch-size for LDM methods. Non-LDM methods never set this
    # so the validator's enforcement is bypassed.
    ldm_pool_min_size = getattr(args, "pool_min_size", None)
    if ldm_pool_min_size is None or ldm_pool_min_size <= 0:
        ldm_pool_min_size = args.batch_size

    ldm_cfg = BayesianLDMSearchConfig(
        init_size=args.init_size,
        batch_size=args.batch_size,
        n_iterations=n_iter,
        smiles_max_len=args.smiles_max_len,
        bo_config=bo_cfg,
        llm_config=llm_client_cfg,
        pool_max_size=args.max_pool_size,
        pool_min_size=ldm_pool_min_size,
        method=method,
        seed=seed,
        minimize=minimize,
        objective_legend=objective_legend,
        trajectory_dir=trajectory_dir,
        verbose=bool(args.verbose),
        guidance=_resolve_ldm_sys_prompt(getattr(args, "ldm_sys_prompt", "") or ""),
    )
    try:
        result = bayesian_ldm_search(
            seed_smiles=seed_smiles,
            scorer=scorer,
            analog_fn=analog_fn,
            config=ldm_cfg,
            rng=random.Random(seed),
            llm=llm,
        )
    finally:
        if cleanup_traj_dir is not None:
            import shutil
            try:
                shutil.rmtree(cleanup_traj_dir, ignore_errors=True)
            except Exception:                                # pragma: no cover
                pass
    return result


# ---------------------------------------------------------------------------
# History summarisation (BSF for n_obj=1, hypervolume curve for n_obj=2,
# per-objective BSF for n_obj>=3 — graceful degradation).
# ---------------------------------------------------------------------------


def _per_obj_best_so_far(
    history: Sequence[Tuple[str, Tuple[Optional[float], ...]]],
    num_evaluations: int,
    obj_idx: int,
) -> np.ndarray:
    """Best-so-far curve for a single objective index, 1D ndarray."""
    bsf: list[float] = []
    current = float("nan")
    seen_any = False
    for _, sc in list(history)[:num_evaluations]:
        v = sc[obj_idx] if obj_idx < len(sc) else None
        if v is not None and np.isfinite(v):
            if not seen_any:
                current = float(v)
                seen_any = True
            else:
                # Direction inferred from the *first* finite value's
                # relation to the current best (we don't know per-obj
                # direction here; for n_obj>=3 the user is expected
                # to consume the per-obj arrays themselves).
                current = float(v)
    while len(bsf) < num_evaluations:
        bsf.append(current)
    return np.asarray(bsf, dtype=float)


def summarize_history(
    history: Sequence,
    *,
    ref_point: Optional[Tuple[float, ...]],
    num_evaluations: int,
    minimize: Tuple[bool, ...],
) -> dict:
    """Reduce a history to per-run summary curves.

    For ``n_obj == 1`` returns ``{"bsf": ndarray}``.
    For ``n_obj == 2`` returns ``{"hypervolume": ndarray}`` (cumulative
    hypervolume w.r.t. ``ref_point``).
    For ``n_obj >= 3`` returns ``{"bsf_per_objective": ndarray of shape
    (n_obj, num_evaluations)}`` and emits a warning (HV is not
    implemented for ``n_obj >= 3``).
    """
    n_obj = len(minimize)
    if n_obj == 1:
        # Single-obj: legacy best-so-far curve.
        from strbo_v1.bayesian_analog_search import (
            _collect_finite_history_n,  # type: ignore[attr-defined]
        )
        _ = _collect_finite_history_n  # silence unused
        bsf: list[float] = []
        current = float("inf") if minimize[0] else float("-inf")
        for _, sc in list(history)[:num_evaluations]:
            v = sc if isinstance(sc, (int, float)) else (sc[0] if sc else None)
            if v is not None and np.isfinite(float(v)):
                current = min(current, float(v)) if minimize[0] else max(current, float(v))
            bsf.append(current)
        while len(bsf) < num_evaluations:
            bsf.append(current)
        return {"bsf": np.asarray(bsf, dtype=float)}

    if n_obj == 2:
        from strbo_v1.acquisition import hypervolume
        if ref_point is None:
            ref_point = (0.0, 0.0)
        # Cumulative hypervolume: ``hv_curve[k]`` = HV of the first
        # ``k + 1`` entries in history, computed exactly via the public
        # 2D HV backend. Padded to ``num_evaluations`` if the actual
        # run produced fewer evaluations.
        hv_curve: list[float] = []
        for k in range(1, num_evaluations + 1):
            partial = list(history)[:k]
            finite = [
                sc for _, sc in partial
                if sc is not None
                and len(sc) == 2
                and all(v is not None and np.isfinite(float(v)) for v in sc)
            ]
            if not finite:
                hv_curve.append(0.0)
                continue
            hv = hypervolume(
                points=[tuple(float(v) for v in sc) for sc in finite],
                ref=list(ref_point),
                minimize=tuple(minimize),
            )
            hv_curve.append(float(hv))
        while len(hv_curve) < num_evaluations:
            hv_curve.append(hv_curve[-1] if hv_curve else 0.0)
        return {"hypervolume": np.asarray(hv_curve, dtype=float)}

    # n_obj >= 3: graceful degradation.
    LOGGER.warning(
        "summarize_history: n_obj=%d; hypervolume not implemented for n_obj>=3. "
        "Returning per-objective best-so-far curves instead.", n_obj,
    )
    per_obj = [
        _per_obj_best_so_far(history, num_evaluations, i) for i in range(n_obj)
    ]
    return {"bsf_per_objective": np.stack(per_obj, axis=0)}


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _config_echo(
    args: argparse.Namespace, method: str, seed: int, *,
    parts: List[str], minimize: Tuple[bool, ...],
    ref_point: Optional[Tuple[float, ...]],
) -> dict:
    """Build the ``config`` echo for the JSON output."""
    gp_keys = (
        "gp_device", "gp_fit_itersteps", "gp_learning_rate",
        "gp_min_jitter", "gp_max_jitter", "gp_standardize_y",
        "gp_fp_radius", "gp_fp_n_bits",
    )
    vina_keys = (
        "vina_bin", "vina_cache_dir", "vina_pdb_id", "vina_chain_id",
        "vina_ligand_resname", "vina_exhaustiveness", "vina_n_poses",
        "vina_seed", "vina_max_workers", "vina_allow_debug_receptor",
        "vina_no_cache",
    )
    reasyn_keys = (
        "reasyn_model_path", "reasyn_devices", "reasyn_repo", "reasyn_python_bin",
        "reasyn_search_width", "reasyn_exhaustiveness", "reasyn_num_cycles",
        "reasyn_num_editflow_samples", "reasyn_num_editflow_steps",
        "reasyn_time_limit", "reasyn_num_workers_per_gpu", "reasyn_filter_sim",
        "reasyn_no_canonicalize",
    )

    n_obj = len(parts)
    cfg: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "seed_smiles": list(args._seed_smiles_list),
        "num_evaluations": args.num_evaluations,
        "batch_size": args.batch_size,
        "init_size": args.init_size,
        "acquisition": args.acquisition,
        "xi": args.xi,
        "kappa": args.kappa,
        "minimize": list(minimize) if n_obj >= 2 else minimize[0],
        "acq_budget": args.acq_budget,
        "max_pool_size": args.max_pool_size,
        "pool_min_size": args.pool_min_size,
        "pool_max_size": args.pool_max_size,
        "smiles_max_len": args.smiles_max_len,
        "objective": args.objective,
        "n_objectives": n_obj,
        "objective_parts": parts,
        "ehvi_n_samples": args.ehvi_n_samples,
        "che_alpha": args.che_alpha,
    }

    gp_dict = {k: getattr(args, k) for k in gp_keys}
    if method in GP_IMPL:
        gp_dict["impl"] = GP_IMPL[method]
        gp_dict["smiles_maxlen"] = args.smiles_max_len
    cfg["gp"] = gp_dict

    if n_obj >= 2 and ref_point is not None:
        cfg["ref_point"] = list(ref_point)

    # Vina / ReaSyn echo only when at least one vina part is requested
    # (avoid leaking config keys for nn-only / mock runs).
    if any(p == "vina" for p in parts):
        cfg["vina"] = {k: getattr(args, k) for k in vina_keys}
    cfg["reasyn"] = {k: getattr(args, k) for k in reasyn_keys}

    # LLM advisor echo (only for bo-*-ldm methods).
    if method in LDM_METHODS:
        from strbo_v1.llm_advisor.config import DEFAULT_LLM_MODEL
        _ldm_pms = getattr(args, "pool_min_size", None)
        if _ldm_pms is None or _ldm_pms <= 0:
            _ldm_pms = args.batch_size
        cfg["llm"] = {
            "model": getattr(args, "llm_model", None) or DEFAULT_LLM_MODEL,
            "base_url": getattr(args, "llm_base_url", None) or "",
            "trajectory_dir": getattr(args, "llm_trajectory_dir", "") or "",
            "pool_min_size": _ldm_pms,
            "ldm_sys_prompt": _resolve_ldm_sys_prompt(
                getattr(args, "ldm_sys_prompt", "") or ""
            ),
        }

    return cfg


def _history_entry(idx: int, smi: str, score) -> dict:
    """Build one JSON history entry; ``score`` is bare float for n_obj=1
    or tuple of floats for n_obj>=2."""
    if isinstance(score, tuple):
        return {"index": idx, "smiles": smi, "scores": list(score)}
    return {"index": idx, "smiles": smi, "score": score}


def write_json(
    config_echo: dict[str, Any],
    history: Sequence[Tuple[str, Any]],
    out_path: Path,
    *,
    llm_trajectory: Optional[Dict[str, Any]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config_echo,
        "history": [
            _history_entry(i, smi, sc) for i, (smi, sc) in enumerate(history)
        ],
    }
    if llm_trajectory is not None:
        payload["llm_trajectory"] = llm_trajectory
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    LOGGER.info("wrote JSON: %s", out_path)


def resolve_output_path(args: argparse.Namespace, method: str, seed: int) -> Path:
    """If ``--output`` ends in ``.json``, use it verbatim; else treat as dir."""
    raw = Path(args.output)
    if raw.suffix.lower() == ".json" or str(args.output).lower().endswith(".json"):
        return raw
    return raw / f"{method}_seed={seed}.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seed_smiles(text: str) -> list[str]:
    """Parse ``--seed-smiles`` into a list of canonical SMILES.

    Detection order (matches the documented CLI behavior):
    1. If ``text`` is a path to an existing file: read it (UTF-8,
       one SMILES per line), auto-filter blank lines, validate +
       canonicalize each via :func:`strbo_v1.utils.canonicalize_smiles_strict`.
    2. Otherwise: split by comma, filter empties, validate +
       canonicalize each.
    """
    from strbo_v1.utils import canonicalize_smiles_strict

    raw = text.strip()
    if not raw:
        return []

    candidate = Path(raw)
    if candidate.is_file():
        source = str(candidate)
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(
                f"--seed-smiles: cannot read file {source}: {exc}"
            ) from exc
        non_blank = [
            (lineno, line.strip())
            for lineno, line in enumerate(lines, start=1)
            if line.strip()
        ]
        out: list[str] = []
        for lineno, smi in non_blank:
            try:
                out.append(canonicalize_smiles_strict(smi))
            except ValueError as exc:
                raise ValueError(
                    f"--seed-smiles: invalid SMILES at line {lineno} of "
                    f"{source}: {smi!r}"
                ) from exc
        return out

    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    out = []
    for i, p in enumerate(parts, start=1):
        try:
            out.append(canonicalize_smiles_strict(p))
        except ValueError as exc:
            raise ValueError(
                f"--seed-smiles: invalid SMILES at position {i}: {p!r}"
            ) from exc
    return out


def _parse_ref_point(text: Optional[str]) -> Optional[Tuple[float, ...]]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return tuple(float(x) for x in text.split(","))


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ONE search trajectory (one method, one seed) and write JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core
    parser.add_argument("--method", type=str, choices=VALID_METHODS,
                        help="Search method.")
    parser.add_argument("--seed", type=int,
                        help="RNG seed.")
    parser.add_argument("--seed-smiles", type=str, default="CCO,CCN,CCC",
                        help="Comma-separated SMILES, OR a path to an existing "
                             "file (one SMILES per line, blank lines filtered). "
                             "All SMILES are validated with RDKit and "
                             "auto-canonicalized; invalid entries raise "
                             "ValueError with file+line or position context.")
    parser.add_argument("--num-evaluations", type=int, default=30,
                        help="Total scorer evaluations.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="BO candidates per round; also batch size for random methods.")
    parser.add_argument("--init-size", type=int, default=10,
                        help="BO initialization size (warm-up + init).")

    # Random
    parser.add_argument("--pool-min-size", type=int, default=1,
                        help="Random-search pool refill trigger.")
    parser.add_argument("--pool-max-size", type=int, default=None,
                        help="Random-search pool FIFO cap (None = unbounded).")

    # Acquisition
    parser.add_argument("--acquisition", type=str, default="ei",
                        choices=["ei", "ucb", "pi"],
                        help="BO acquisition function (single-objective only).")
    parser.add_argument("--xi", type=float, default=0.01,
                        help="Improvement threshold for EI / PI.")
    parser.add_argument("--kappa", type=float, default=2.0,
                        help="Exploration weight for UCB.")
    parser.add_argument("--acq-budget", type=int, default=None,
                        help="Optional pool-subsample size for the BO acquisition step.")
    parser.add_argument("--max-pool-size", type=int, default=None,
                        help="BO pool FIFO cap. None = unbounded.")
    parser.add_argument("--smiles-max-len", type=int, default=50,
                        help="SMILES length cap. Analogue SMILES longer than this are "
                             "dropped at the pool-ingestion step. Also drives the GP "
                             "string kernel's int64 tensor padding. None disables the filter.")
    parser.add_argument("--ehvi-n-samples", type=int, default=128,
                        help="Monte-Carlo samples per candidate in 2-objective EHVI.")
    parser.add_argument("--che-alpha", type=float, default=1.0,
                        help="Concentration parameter for the simplex-weight Beta "
                             "distribution in Chebyshev-ParEGO (n_obj>=3). "
                             "alpha=1 = uniform on simplex; <1 = corners; >1 = center.")
    parser.add_argument("--ref-point", type=str, default=None,
                        help="Comma-separated reference point for HV/EHVI "
                             "(multi-objective). Default: per-backend registry "
                             "(vina=0.0, nn=5.0, mock=0.0). "
                             "Silently ignored for single-objective.")

    # GP config
    parser.add_argument("--gp-device", type=str, default="cuda")
    parser.add_argument("--gp-fit-itersteps", type=int, default=50)
    parser.add_argument("--gp-learning-rate", type=float, default=0.1)
    parser.add_argument("--gp-min-jitter", type=float, default=1e-6)
    parser.add_argument("--gp-max-jitter", type=float, default=1e-1)
    parser.add_argument("--gp-standardize-y", dest="gp_standardize_y", action="store_true", default=True)
    parser.add_argument("--no-gp-standardize-y", dest="gp_standardize_y", action="store_false")
    parser.add_argument("--gp-fp-radius", type=int, default=2)
    parser.add_argument("--gp-fp-n-bits", type=int, default=2048)

    # Scorer / analog mode
    parser.add_argument(
        "--objective", type=str, default="vina",
        help="Scorer backend(s). Comma-free, '+'-separated. "
             "Examples: 'vina', 'nn', 'mock', 'vina+nn', 'vina+nn+mock'. "
             "Per-backend minimize is hard-coded (vina/mock min, nn max); "
             "pass --ref-point to override the default HV reference point."
    )
    parser.add_argument(
        "--nn-model-path", type=str,
        default=DEFAULT_NN_MODEL_PATH,
        help="Path to a joblib model file (used when --objective contains 'nn').")
    parser.add_argument(
        "--nn-metadata-path", type=str, default="",
        help="Path to sidecar .json metadata (default: <model-stem>_metadata.json).")

    # Vina
    parser.add_argument("--vina-bin", type=str, default=None)
    parser.add_argument("--vina-cache-dir", type=str, default="output/bo/vina_cache/")
    parser.add_argument("--vina-pdb-id", type=str, default="8UN5")
    parser.add_argument("--vina-chain-id", type=str, default="A")
    parser.add_argument("--vina-ligand-resname", type=str, default=None)
    parser.add_argument("--vina-exhaustiveness", type=int, default=4)
    parser.add_argument("--vina-n-poses", type=int, default=3)
    parser.add_argument("--vina-seed", type=int, default=42)
    parser.add_argument("--vina-max-workers", type=int, default=1)
    parser.add_argument("--vina-allow-debug-receptor", action="store_true")
    parser.add_argument("--vina-no-cache", action="store_true")

    # ReaSyn
    parser.add_argument("--reasyn-model-path", type=str, default=None)
    parser.add_argument("--reasyn-devices", type=str, default="1,2")
    parser.add_argument("--reasyn-repo", type=str, default=None)
    parser.add_argument("--reasyn-python-bin", type=str, default=None)
    parser.add_argument("--reasyn-search-width", type=int, default=6)
    parser.add_argument("--reasyn-exhaustiveness", type=int, default=16)
    parser.add_argument("--reasyn-num-cycles", type=int, default=4)
    parser.add_argument("--reasyn-num-editflow-samples", type=int, default=20)
    parser.add_argument("--reasyn-num-editflow-steps", type=int, default=100)
    parser.add_argument("--reasyn-time-limit", type=int, default=120)
    parser.add_argument("--reasyn-num-workers-per-gpu", type=int, default=1)
    parser.add_argument("--reasyn-filter-sim", type=float, default=0.8)
    parser.add_argument("--reasyn-no-canonicalize", action="store_true")

    # LLM advisor (bo-*-ldm methods only). Credentials are read from
    # .env (LLM_API_KEY, LLM_BASE_URL) by the LDM branch. The
    # --llm-api-key / --llm-base-url flags override the env values
    # when non-empty. --llm-model has a hardcoded default of
    # DeepSeek-V4-Flash; there is no LLM_MODEL env var.
    from strbo_v1.llm_advisor.config import DEFAULT_LLM_MODEL
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL,
                        help="LLM model name. Default: DeepSeek-V4-Flash "
                             "(hardcoded; no LLM_MODEL env var).")
    parser.add_argument("--llm-base-url", type=str, default="",
                        help="Override LLM_BASE_URL from .env. Empty = use env.")
    parser.add_argument("--llm-api-key", type=str, default="",
                        help="Override LLM_API_KEY from .env. Empty = use env.")
    parser.add_argument("--llm-timeout", type=float, default=60.0,
                        help="Per LLM HTTP request timeout in seconds.")

    parser.add_argument("--llm-trajectory-dir", type=str, default="",
                        help="Directory for per-round LLM trajectory JSONs (bo-*-ldm only). "
                             "The trajectory is also embedded in the main JSON under "
                             "'llm_trajectory'. Empty = no sidecar.")
    parser.add_argument("--ldm-sys-prompt", type=str, default="",
                        help="LDM system-prompt supplement. If the value is a "
                             "path to an existing file, the file's contents "
                             "are read and used; otherwise the value is "
                             "treated as inline text. The text is appended "
                             "to all three LLM system prompts (Stage A1 "
                             "actions, A2 review-analogs, B review-"
                             "suggestions). Empty = no supplement.")


    # Output
    parser.add_argument("--output", type=str, default="output/bo",
                        help="Output directory (auto-named '<method>_seed=<seed>.json') "
                             "or an explicit '*.json' file path.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level.")
    return parser


# ---------------------------------------------------------------------------
# Public driver API (bo_api.py + tests)
# ---------------------------------------------------------------------------


def config_from_dict(d: Dict[str, Any]) -> argparse.Namespace:
    """Build an :class:`argparse.Namespace` from a config dict.

    This is the JSON-in counterpart to :func:`_build_argparser`:
    :func:`bo_api.run_search_trajectory` calls it with a parsed
    JSON object and uses the result as if it had been built by
    ``parser.parse_args(argv)``.

    Args:
        d: A dict whose keys are the long-form CLI flag names (with
            hyphens, e.g. ``"num-evaluations"``) OR the equivalent
            Python attribute names (with underscores, e.g.
            ``"num_evaluations"``). Values are passed through to
            :class:`argparse.ArgumentParser.parse_args` via a
            synthetic argv.

    Returns:
        A fully-populated :class:`argparse.Namespace` with the
        same attribute names as the CLI parser.

    Raises:
        ValueError: If ``d`` is not a dict, or contains keys that
            do not correspond to any known CLI flag, or contains
            values of the wrong type.
    """
    if not isinstance(d, dict):
        raise ValueError(
            f"config_from_dict expects a dict, got {type(d).__name__}"
        )
    parser = _build_argparser()
    # ``parser.error()`` writes a usage message to stderr and calls
    # ``sys.exit(2)``, which loses the error text and is awkward in
    # a JSON API. Override it to raise ``ValueError`` with the same
    # message; the CLI's own ``main()`` uses ``parser.error``
    # directly (not via this code path), so the CLI behavior is
    # unchanged.
    _override_parser_error_to_raise(parser)
    namespace = parser.parse_args([])  # all defaults

    name_to_dest, valid_dests, bool_flag_map, bool_defaults = _introspect_parser(parser)

    argv: List[str] = []
    for key, value in d.items():
        key_str = str(key)
        if key_str in name_to_dest:
            dest = name_to_dest[key_str]
        elif key_str.replace("-", "_") in valid_dests:
            dest = key_str.replace("-", "_")
        else:
            raise ValueError(
                f"config_from_dict: unknown config key {key_str!r}; "
                f"valid keys are the CLI long-form flag names "
                f"(e.g. 'num-evaluations') or their underscore-attribute "
                f"equivalents (e.g. 'num_evaluations')."
            )

        if isinstance(value, bool):
            default = bool_defaults.get(dest, False)
            if value == default:
                continue  # matches default; nothing to add
            true_flag, false_flag = bool_flag_map.get(dest, (None, None))
            if value and true_flag is not None:
                argv.append(true_flag)
            elif not value and false_flag is not None:
                argv.append(false_flag)
            else:
                raise ValueError(
                    f"config_from_dict: cannot pass bool value for {key_str!r}; "
                    f"no flag in the parser toggles this dest to {value}."
                )
            continue

        if value is None:
            continue

        flag = "--" + dest.replace("_", "-")
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.extend([flag, str(value)])

    parser.parse_args(argv, namespace=namespace)
    return namespace


def _override_parser_error_to_raise(parser: argparse.ArgumentParser) -> None:
    """Replace ``parser.error`` with a version that raises ValueError.

    The default ``argparse.ArgumentParser.error()`` writes a usage
    line to stderr and calls ``sys.exit(2)``, which loses the actual
    error message and is hostile to JSON-style callers
    (``bo_api.run_search_trajectory``). The CLI's own
    :func:`main` uses the un-overridden parser (it parses argv
    directly), so CLI behavior is unchanged.
    """

    def _raise_value_error(message: str) -> None:
        raise ValueError(str(message))

    parser.error = _raise_value_error  # type: ignore[method-assign]


def _introspect_parser(parser: argparse.ArgumentParser):
    """Inspect the parser and return lookup tables for :func:`config_from_dict`.

    Returns:
        ``(name_to_dest, valid_dests, bool_flag_map, bool_defaults)``:
        - ``name_to_dest``: maps both long-flag names (``"num-evaluations"``)
          and underscore-attribute names (``"num_evaluations"``) to the
          underlying ``dest``.
        - ``valid_dests``: set of valid dest names.
        - ``bool_flag_map``: for boolean (``store_true`` / ``store_false``)
          dests, maps ``dest → (true_flag, false_flag)`` where each is
          a CLI long-form flag string (e.g. ``"--verbose"``,
          ``"--no-gp-standardize-y"``) or ``None`` if no such flag exists.
        - ``bool_defaults``: maps ``dest → default_value`` (the value
          that is set when no flag is provided).
    """
    name_to_dest: Dict[str, str] = {}
    valid_dests: set = set()
    bool_flag_map: Dict[str, tuple] = {}
    bool_defaults: Dict[str, bool] = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        valid_dests.add(action.dest)
        if action.option_strings:
            canonical_long = max(action.option_strings, key=len)
            name_to_dest[canonical_long.lstrip("-")] = action.dest
            name_to_dest[action.dest] = action.dest
        if type(action).__name__ in ("_StoreTrueAction", "_StoreFalseAction"):
            true_flag, false_flag = bool_flag_map.get(action.dest, (None, None))
            for opt in action.option_strings:
                if type(action).__name__ == "_StoreTrueAction":
                    if true_flag is None:
                        true_flag = opt
                else:
                    if false_flag is None:
                        false_flag = opt
            bool_flag_map[action.dest] = (true_flag, false_flag)
            if action.dest not in bool_defaults:
                bool_defaults[action.dest] = bool(action.default) if action.default is not None else False
    return name_to_dest, valid_dests, bool_flag_map, bool_defaults


def run_one_trajectory(
    args: argparse.Namespace, *, include_summary: bool = True,
) -> Dict[str, Any]:
    """Run one trajectory from a fully-populated namespace.

    This is the JSON-friendly core of :func:`main`: it does all the
    work (build scorers, build analog generator, run loop, compute
    summary) and returns a dict. ``main`` calls it and writes the
    ``{config, history}`` subset to disk; :func:`bo_api.run_search_trajectory`
    calls it and serializes the full ``{config, history, summary}`` to
    JSON.

    Args:
        args: A namespace as built by :func:`_build_argparser` (CLI)
            or :func:`config_from_dict` (JSON). The following fields
            are required: ``method``, ``seed``, ``seed_smiles``,
            ``objective``, ``num_evaluations``, ``batch_size``,
            ``init_size``, ``verbose``, plus the GP / Vina / ReaSyn
            sub-configs.
        include_summary: If True (default), the returned dict has a
            ``"summary"`` key with the bsf / hv / per-obj-bsf curve
            (depending on ``n_obj``). Pass False to skip the
            summary computation (faster, smaller payload).

    Returns:
        ``{"config": <echo dict>, "history": <list of (smiles, score)
        tuples>, "summary": <dict with bsf / hypervolume /
        bsf_per_objective>}`` (the ``"summary"`` key is omitted when
        ``include_summary=False``).

    Raises:
        ValueError: On invalid input (bad method, bad SMILES,
            ref-point / objective mismatch, etc.).
        SystemExit: On errors that the CLI version converts to
            exit code 2; the JSON API re-raises these as
            ``ValueError`` (caught by :func:`bo_api.run_search_trajectory`).
    """
    try:
        seed_smiles = _parse_seed_smiles(args.seed_smiles)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not seed_smiles:
        raise SystemExit("--seed-smiles produced empty SMILES list")
    args._seed_smiles_list = seed_smiles

    method = args.method
    seed = args.seed

    try:
        parts = _parse_objective(args.objective)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    n_obj = len(parts)

    user_ref = _parse_ref_point(args.ref_point)
    if n_obj >= 2:
        try:
            ref_point = resolve_ref_point(parts, user_ref)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        if args.ref_point is not None:
            LOGGER.info("--ref-point ignored for single-objective")
        ref_point = None

    scorer, minimize = _build_scorers_and_minimize(args)

    if n_obj == 1 and parts[0] == "mock":
        analog_generator = mock_chain_analog_generator
    else:
        analog_generator = _build_reasyn_analog(args)

    LOGGER.info(
        "method=%s seed=%d objective=%s n_obj=%d",
        method, seed, args.objective, n_obj,
    )
    LOGGER.info("seed_smiles=%s", seed_smiles)
    LOGGER.info(
        "num_evaluations=%d, batch_size=%d, init_size=%d, minimize=%s",
        args.num_evaluations, args.batch_size, args.init_size,
        list(minimize) if n_obj >= 2 else minimize[0],
    )

    history, trajectory = run_one(
        method, seed, seed_smiles, scorer, analog_generator, args,
        minimize=minimize, ref_point=ref_point,
    )
    if include_summary:
        summary = summarize_history(
            history, ref_point=ref_point,
            num_evaluations=args.num_evaluations, minimize=minimize,
        )
        if len(history) < args.num_evaluations:
            LOGGER.warning(
                "method=%s seed=%d produced %d evaluations (target %d); "
                "summary padded with last value",
                method, seed, len(history), args.num_evaluations,
            )
    else:
        summary = None

    config_echo = _config_echo(
        args, method, seed, parts=parts, minimize=minimize, ref_point=ref_point,
    )
    history_json = [
        {"index": i, "smiles": smi, **_score_to_json_entry(sc)}
        for i, (smi, sc) in enumerate(history)
    ]
    payload: Dict[str, Any] = {
        "config": config_echo,
        "history": list(history),       # in-memory tuples, for write_json
        "history_json": history_json,   # JSON-safe form, for bo_api
    }
    if trajectory is not None:
        payload["llm_trajectory"] = trajectory
    if summary is not None:
        summary_json = _summary_to_json(summary)
        payload["summary"] = summary_json
    return payload


def _score_to_json_entry(score: Any) -> Dict[str, Any]:
    """Build the per-history-entry JSON field for a score.

    Mirrors :func:`_history_entry` in :func:`write_json`: bare float
    becomes ``"score"``; tuple becomes ``"scores"``.
    """
    if isinstance(score, tuple):
        return {"scores": [None if v is None else float(v) for v in score]}
    return {"score": None if score is None else float(score)}


def _summary_to_json(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ``summarize_history()`` result to JSON-safe form.

    The numpy arrays in the summary are converted to plain Python
    lists so :func:`json.dumps` can serialize them.
    """
    out: Dict[str, Any] = {}
    for k, v in summary.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {kk: vv.tolist() if hasattr(vv, "tolist") else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # Manual required-flag checks (we made them optional in the
    # parser so :func:`config_from_dict` can build a defaults-only
    # namespace without errors; the CLI enforces them here).
    missing = [name for name in ("method", "seed") if getattr(args, name) is None]
    if missing:
        parser.error(
            "the following arguments are required: " + ", ".join(f"--{n.replace('_', '-')}" for n in missing)
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    for noisy in ("strbo_v1", "strbo_v1.bayesian_analog_search", "strbo_v1.gp", "strbo_v1.analog"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if args.verbose else logging.WARNING
        )

    print(f"[run] method={args.method} seed={args.seed} objective={args.objective} ...", flush=True)
    try:
        result = run_one_trajectory(args, include_summary=False)
    except SystemExit as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc

    out_path = resolve_output_path(args, args.method, args.seed)
    write_json(
        result["config"], result["history"], out_path,
        llm_trajectory=result.get("llm_trajectory"),
    )
    n_history = len(result["history"])
    print(f"done ({n_history} evals)")
    print(f"[output] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
