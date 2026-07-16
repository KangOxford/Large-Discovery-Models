#!/usr/bin/env python3
"""LLM DSL proposal + parallel acquisition selector baseline for AntBO.

After an initial LLM-only warmup, the LLM proposes search atoms (LocalSearch,
NeighborSampling, LatinHyperCubeSampling, Or). The existing LDM parallel search
executor expands those atoms into candidates, scores them with a GP acquisition
function, and only the argmax/top-k are sent to Absolut.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

def find_repo_root(start: Path) -> Path:
    for path in [start.parent, *start.parents]:
        if (path / "bo").is_dir() and (path / "bo" / "__init__.py").exists():
            return path
    raise RuntimeError(f"Could not find AntBO repo root from {start}")


ROOT = find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WORKSPACE_ROOT = ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ldm_tts.trajectory import JsonlTrajectoryRecorder

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
POSITIVE = set("RKH")
NEGATIVE = set("DE")
N_GLYCO = re.compile("N[^P][ST]")
ACQ_NAME = "ei"
ACQ_CHOICES = ("ei", "lcb", "ucb")
GP_TRAIN_STEPS = 300
WARMUP_POOL_SIZE = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM warmup followed by LDM parallel acquisition selection.")
    p.add_argument("--config", default="bo/config.yaml")
    p.add_argument("--antigens_file", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_trials", type=int, default=1)
    p.add_argument("--n_evals", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--out_root", default="outputs/llm_acq_baseline")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout_s", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--history_top_k", type=int, default=10)
    p.add_argument("--parallel_budget", type=int, default=600,
                   help="Number of LDM-generated candidates scored by the GP acquisition after warmup.")
    p.add_argument("--n_init", type=int, default=20,
                   help="Number of initial oracle observations before fitting a GP acquisition.")
    p.add_argument(
        "--acq",
        "--acquisition",
        dest="acq",
        choices=ACQ_CHOICES,
        default=ACQ_NAME,
        help=(
            "GP acquisition used after warmup. EI is minimization expected improvement; "
            "LCB/UCB use posterior bounds with --acq_beta and are internally scored so "
            "larger is better."
        ),
    )
    p.add_argument(
        "--acq_beta",
        "--acq-beta",
        dest="acq_beta",
        type=float,
        default=1.0,
        help="Exploration coefficient for LCB/UCB acquisitions.",
    )
    p.add_argument("--include_antigen_context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fallback_random", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_antigens(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_candidate_library(path: str | None, seq_len: int) -> list[str]:
    if not path:
        return []
    data = pd.read_csv(path, index_col=None)
    seqs = data.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
    out: list[str] = []
    seen: set[str] = set()
    for seq in seqs:
        if valid_seq(seq, seq_len) and passes_developability(seq) and seq not in seen:
            out.append(seq)
            seen.add(seq)
    return out


def make_llm_client():
    try:
        from bo.ldm import OpenAIClient
        return OpenAIClient()
    except Exception as exc:
        raise RuntimeError(
            "Could not create bo.ldm.OpenAIClient. Check .env, openai package, "
            f"and project dependencies. Original error: {exc}"
        ) from exc


def seqs_to_indices(seqs: list[str]) -> np.ndarray:
    return np.array([[AA_TO_IDX[aa] for aa in seq] for seq in seqs], dtype=np.int32)


def indices_to_seqs(x: np.ndarray) -> list[str]:
    return ["".join(IDX_TO_AA[int(i)] for i in row) for row in np.asarray(x)]


def valid_seq(seq: str, seq_len: int) -> bool:
    return len(seq) == seq_len and all(aa in AA for aa in seq)


def longest_hydrophobic_run(seq: str) -> int:
    best = cur = 0
    for aa in seq:
        cur = cur + 1 if aa in HYDROPHOBIC else 0
        best = max(best, cur)
    return best


def net_charge(seq: str) -> float:
    total = 0.0
    for aa in seq:
        if aa == "H":
            total += 0.1
        elif aa in POSITIVE:
            total += 1.0
        elif aa in NEGATIVE:
            total -= 1.0
    return total


def passes_developability(seq: str) -> bool:
    return (
        seq.count("C") <= 1
        and longest_hydrophobic_run(seq) <= 4
        and sum(1 for aa in seq if aa in AROMATIC) <= 2
        and -1.0 <= net_charge(seq) <= 2.0
        and N_GLYCO.search(seq) is None
    )


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in LLM response")
        text = text[start:end + 1]
    return json.loads(text)


def best_history(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return [
        {"sequence": row["LastProtein"], "score": row["LastValue"]}
        for row in sorted(rows, key=lambda r: r["LastValue"])[:top_k]
    ]


def build_prompt(
    antigen: str,
    seq_len: int,
    batch_size: int,
    observed: set[str],
    rows: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    top_k: int,
) -> str:
    payload = {
        "task": "Pure LLM baseline for CDRH3 sequence proposal.",
        "objective": "Minimize Absolut energy. Lower true score is better.",
        "mode": (
            "LLM proposal stage. Return diverse candidates; a Bayesian "
            "acquisition function may select which candidate(s) are evaluated."
        ),
        "reasoning": "Reason internally from history and antigen context, but do not output reasoning.",
        "antigen": antigen,
        "constraints": {
            "length": seq_len,
            "alphabet": AA,
            "num_sequences": batch_size,
            "choose_only_from_candidate_pool": True,
            "do_not_repeat": sorted(observed)[-200:],
            "candidate_pool_developability_filter": {
                "max_cysteine": 1,
                "max_hydrophobic_run": 4,
                "max_aromatic_FWY": 2,
                "net_charge_range": [-1.0, 2.0],
                "forbid_n_glycosylation_NXS_or_NXT": True,
            },
        },
        "history": {
            "num_observed": len(rows),
            "best": best_history(rows, top_k),
            "recent": rows[-top_k:],
        },
        "candidate_pool": candidate_pool,
        "antigen_context": antigen_context or {},
        "required_output": {
            "selected": [
                {
                    "id": 0,
                    "sequence": "A" * seq_len,
                    "score": "numeric LLM priority score; higher is better",
                }
            ]
        },
        "output_rules": [
            "Return JSON only.",
            "The only top-level key must be selected.",
            "Each selected item must contain only id, sequence, and score.",
            "Select only sequences present in candidate_pool.",
            "Do not include rationale or explanation.",
        ],
    }
    return json.dumps(payload, indent=2)


def parse_selected(
    obj: dict[str, Any],
    seq_len: int,
    observed: set[str],
    candidate_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = obj.get("selected", obj.get("candidates", obj.get("sequences", [])))
    if not isinstance(raw, list):
        raise ValueError("LLM JSON must contain a selected list")

    pool_by_id = {int(item["id"]): item["sequence"] for item in candidate_pool}
    pool_seqs = {item["sequence"] for item in candidate_pool}
    candidates: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            raw_id = item.get("id")
            seq = str(item.get("sequence", "")).strip().upper()
            score = item.get("score")
            if raw_id is not None:
                try:
                    seq = pool_by_id[int(raw_id)]
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            seq, score = str(item).strip().upper(), None

        if (
            seq not in pool_seqs
            or not valid_seq(seq, seq_len)
            or not passes_developability(seq)
            or seq in observed
            or seq in used
        ):
            continue
        try:
            score = None if score is None else float(score)
        except (TypeError, ValueError):
            score = None
        candidates.append({"sequence": seq, "score": score})
        used.add(seq)
    return candidates


def random_candidates(rng: random.Random, n: int, seq_len: int, observed: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    used: set[str] = set()
    attempts = 0
    max_attempts = max(10000, n * 200)
    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        seq = "".join(rng.choice(AA) for _ in range(seq_len))
        if seq not in observed and seq not in used and passes_developability(seq):
            candidates.append({"sequence": seq, "score": None})
            used.add(seq)
    if len(candidates) < n:
        raise RuntimeError(
            f"Could only generate {len(candidates)} developability-filtered random candidates "
            f"after {attempts} attempts; requested {n}."
        )
    return candidates


def make_candidate_pool(
    rng: random.Random,
    library: list[str],
    observed: set[str],
    seq_len: int,
    pool_size: int,
) -> list[dict[str, Any]]:
    if library:
        available = [seq for seq in library if seq not in observed]
        rng.shuffle(available)
        seqs = available[:pool_size]
    else:
        seqs = [item["sequence"] for item in random_candidates(rng, pool_size, seq_len, observed)]
    return [{"id": i, "sequence": seq} for i, seq in enumerate(seqs)]


def propose(
    llm: Any,
    rng: random.Random,
    antigen: str,
    seq_len: int,
    batch_size: int,
    observed: set[str],
    rows: list[dict[str, Any]],
    candidate_library: list[str],
    antigen_context: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors = []
    for attempt in range(1, args.max_retries + 1):
        candidate_pool = make_candidate_pool(
            rng=rng,
            library=candidate_library,
            observed=observed,
            seq_len=seq_len,
            pool_size=max(batch_size, WARMUP_POOL_SIZE),
        )
        if len(candidate_pool) < batch_size:
            raise RuntimeError(f"Candidate pool exhausted: {len(candidate_pool)} available, need {batch_size}")
        prompt = build_prompt(
            antigen=antigen,
            seq_len=seq_len,
            batch_size=batch_size,
            observed=observed,
            rows=rows,
            candidate_pool=candidate_pool,
            antigen_context=antigen_context,
            top_k=args.history_top_k,
        )
        raw = llm.call(prompt, temperature=args.temperature, timeout_s=args.timeout_s)
        try:
            parsed = extract_json(raw)
            candidates = parse_selected(parsed, seq_len, observed, candidate_pool)
            if len(candidates) >= batch_size:
                return candidates[:batch_size], {
                    "source": "llm",
                    "attempt": attempt,
                    "prompt": prompt,
                    "candidate_pool": candidate_pool,
                    "raw_response": raw,
                    "parsed": parsed,
                }
            raise ValueError(f"Only {len(candidates)} valid novel candidates returned")
        except Exception as exc:
            errors.append({"attempt": attempt, "error": str(exc), "raw_response": raw})

    if args.fallback_random:
        candidate_pool = make_candidate_pool(
            rng=rng,
            library=candidate_library,
            observed=observed,
            seq_len=seq_len,
            pool_size=max(batch_size, WARMUP_POOL_SIZE),
        )
        return [{"sequence": item["sequence"], "score": None} for item in candidate_pool[:batch_size]], {
            "source": "fallback_random",
            "errors": errors,
        }
    raise RuntimeError(json.dumps(errors, indent=2))


def build_status_from_rows(
    rows: list[dict[str, Any]],
    antigen: str,
    seed: int,
    iteration: int,
    antigen_context: dict[str, Any] | None,
    orchestrator,
):
    from bo.ldm import OrchestratorStatus

    full_history = [
        (row["LastProtein"], float(row["LastValue"]), int(row["Index"]))
        for row in rows
    ]
    best_row = min(rows, key=lambda row: float(row["BestValue"]))
    best_seq = best_row["BestProtein"]
    best_idx = min(range(len(rows)), key=lambda i: float(rows[i]["LastValue"]))
    n_iters_without_improvement = len(rows) - 1 - best_idx
    return OrchestratorStatus(
        iteration=iteration,
        antigen_id=antigen,
        antigen_seed=seed,
        iter_seed=iteration,
        current_search_dsl=orchestrator.current_search_dsl,
        current_bias_dsl=orchestrator.current_bias_dsl,
        full_history=full_history,
        best_value=float(best_row["BestValue"]),
        best_sequence=seqs_to_indices([best_seq])[0].tolist(),
        n_evals=len(rows),
        n_iters_without_improvement=max(0, n_iters_without_improvement),
        antigen_context=antigen_context or {},
    )


def fit_gp_and_make_acquisition(
    rows: list[dict[str, Any]],
    *,
    acq_name: str = ACQ_NAME,
    beta: float = 1.0,
    gp_train_steps: int = GP_TRAIN_STEPS,
    device: Any = "cpu",
):
    """Fit a GP and return a larger-is-better minimization acquisition."""
    import torch
    from torch.distributions import Normal
    from bo.gp import train_gp

    acq_name = str(acq_name).lower()
    if acq_name not in ACQ_CHOICES:
        raise ValueError(f"Unsupported acquisition {acq_name!r}; choose one of {ACQ_CHOICES}.")
    beta = float(beta)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA acquisition device, but torch.cuda.is_available() is False.")

    train_seqs = [row["LastProtein"] for row in rows]
    train_x = torch.tensor(seqs_to_indices(train_seqs), dtype=torch.float32, device=device)
    y_raw = np.array([float(row["LastValue"]) for row in rows], dtype=np.float32)
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std() + 1e-8)
    train_y = torch.tensor((y_raw - y_mean) / y_std, dtype=torch.float32, device=device).view(-1)

    gp = train_gp(
        train_x=train_x,
        train_y=train_y,
        use_ard=True,
        num_steps=max(30, int(gp_train_steps)),
        kern="transformed_overlap",
        noise_variance=1e-6,
        alphabet_size=len(AA),
        search_strategy="local",
    )
    gp.eval()
    best = train_y.min()

    def f_acq(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device=device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.reshape(1, -1)
        posterior = gp(x)
        mu = posterior.mean.view(-1)
        sigma = posterior.stddev.view(-1).clamp_min(1e-9)
        if acq_name == "lcb":
            # The executor maximizes acquisition scores. For a minimization task,
            # maximizing -LCB is equivalent to minimizing mu - beta * sigma.
            return -(mu - beta * sigma)
        if acq_name == "ucb":
            # For minimization, this favors candidates whose pessimistic bound is
            # still low by maximizing the negated upper confidence bound.
            return -(mu + beta * sigma)
        normal = Normal(
            torch.zeros(1, dtype=mu.dtype, device=mu.device),
            torch.ones(1, dtype=mu.dtype, device=mu.device),
        )
        z = (best.to(mu) - mu) / sigma
        return (best.to(mu) - mu) * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))

    return gp, f_acq


def fallback_search_dsl(rows: list[dict[str, Any]], budget: int):
    from bo.ldm.dsl.search_space import LatinHyperCubeSampling, LocalSearch

    if not rows:
        return LatinHyperCubeSampling(num=max(1, budget))
    best_row = min(rows, key=lambda row: float(row["BestValue"]))
    restart = 3
    steps = max(1, budget // restart - 1)
    return LocalSearch(best_row["BestProtein"], radius=3, restart=restart, steps=steps)


def select_with_parallel_ldm(
    *,
    orchestrator,
    rows: list[dict[str, Any]],
    antigen: str,
    seed: int,
    iteration: int,
    antigen_context: dict[str, Any] | None,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use LDM's parallel executor: LLM DSL -> execute_atoms -> acquisition argmax."""
    import torch
    from bo.ldm.acquisition.parallel_search import execute_atoms

    acq_name = str(getattr(args, "acq", ACQ_NAME)).lower()
    acq_beta = float(getattr(args, "acq_beta", 1.0))
    gp, f_acq = fit_gp_and_make_acquisition(rows, acq_name=acq_name, beta=acq_beta)
    status = build_status_from_rows(
        rows=rows,
        antigen=antigen,
        seed=seed,
        iteration=iteration,
        antigen_context=antigen_context,
        orchestrator=orchestrator,
    )
    decision = orchestrator.step(status)
    search_dsl = decision.search_dsl
    if search_dsl is None:
        search_dsl = fallback_search_dsl(rows, int(args.parallel_budget))
    bias_dsl = decision.bias_dsl

    results = execute_atoms(
        search_dsl=search_dsl,
        gp=gp,
        f_acq=f_acq,
        bias_dsl=bias_dsl,
        bias_weight=0.0,
        config=np.array([len(AA)] * len(rows[0]["LastProtein"]), dtype=int),
        cdr_constraints=True,
        rng=np.random.default_rng(seed + iteration),
        timeout_s=5.0,
        device=torch.device("cpu"),
        acq_name=acq_name,
    )
    if not results:
        raise RuntimeError(f"Parallel LDM search produced no candidates from {search_dsl!r}")

    score_key = f"bias+{acq_name}"
    ranked = sorted(range(len(results)), key=lambda i: results[i][score_key], reverse=True)
    selected_indices = ranked[:batch_size]
    selected_candidates: list[dict[str, Any]] = []
    for idx in selected_indices:
        item = results[idx]
        seq = "".join(IDX_TO_AA[int(v)] for v in item["seq"])
        selected_candidates.append({
            "sequence": seq,
            "score": None,
            "acquisition_score": float(item[score_key]),
            "acquisition_raw": float(item[acq_name]),
            "mu": float(item["mu"]),
            "sigma": float(item["sigma"]),
            "source": item.get("source", repr(search_dsl)),
        })

    parallel_results_json: list[dict[str, Any]] = []
    for item in results:
        parallel_results_json.append({
            "sequence": "".join(IDX_TO_AA[int(v)] for v in item["seq"]),
            acq_name: float(item[acq_name]),
            "mu": float(item["mu"]),
            "sigma": float(item["sigma"]),
            "bias": float(item.get("bias", 0.0)),
            score_key: float(item[score_key]),
            "source": item.get("source", repr(search_dsl)),
        })

    return selected_candidates, {
        "source": f"ldm_parallel_{acq_name}_argmax",
        "acq_name": acq_name,
        "acq_beta": acq_beta,
        "decision": {
            "search_dsl": repr(search_dsl),
            "bias_dsl": repr(bias_dsl) if bias_dsl is not None else None,
            "fallback_used": bool(decision.fallback_used or decision.search_dsl is None),
            "rationale": decision.rationale,
        },
        "parallel_results": parallel_results_json,
        "selected_indices": selected_indices,
        "selected_candidates": selected_candidates,
    }


def run_absolut_info(absolut_path: str, command: str, antigen: str, timeout_s: int) -> str:
    exe = Path(absolut_path).resolve() / "src/bin/Absolut"
    if not exe.exists():
        return ""
    proc = subprocess.run(
        [str(exe), command, antigen],
        cwd=str(Path(absolut_path).resolve()),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return (proc.stdout or "")[:2000]


def collect_antigen_context(config: dict[str, Any], antigen: str) -> dict[str, Any]:
    bbox = dict(config["bbox"])
    bbox["antigen"] = antigen
    timeout_s = int(config.get("llm_antigen_context_timeout_s", 30))
    return {
        "antigen": antigen,
        "source": "Absolut",
        "info_antigen": run_absolut_info(bbox["path"], "info_antigen", antigen, timeout_s),
        "info_filenames": run_absolut_info(bbox["path"], "info_filenames", antigen, timeout_s),
    }


class RandomEvaluator:
    def energy(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        seqs = indices_to_seqs(x)
        return np.random.random(len(seqs)), seqs


class AbsolutEvaluator:
    def __init__(self, bbox: dict[str, Any], run_id: str) -> None:
        self.bbox = bbox
        self.run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)

    def energy(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        x = np.asarray(x, dtype=np.int32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        antigen = self.bbox["antigen"]
        absolut_path = self.bbox["path"]
        n_proc = int(self.bbox["process"])
        start_task = int(self.bbox["startTask"])
        tmp = f"TempCDR3_{antigen}_{self.run_id}.txt"
        out = f"{antigen}FinalBindings_Process_1_Of_1.txt"
        cwd = os.getcwd()

        os.chdir(absolut_path)
        lock_dir = f".antbo_llm_acq_{antigen}.lock"
        self._acquire_lock(lock_dir)
        try:
            self._remove_files(antigen, n_proc, tmp, out)
            seqs = indices_to_seqs(x)
            with open(tmp, "w", encoding="utf-8") as handle:
                for i, seq in enumerate(seqs, start=1):
                    handle.write(f"{i}\t{seq}\n")

            proc = subprocess.run(
                [
                    "taskset",
                    "-c",
                    f"{start_task}-{start_task + n_proc}",
                    "./src/bin/Absolut",
                    "repertoire",
                    antigen,
                    tmp,
                    str(n_proc),
                ],
                capture_output=True,
                text=False,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else "(empty)"
                raise RuntimeError(f"Absolut failed with returncode={proc.returncode}:\n{stderr}")
            if not os.path.exists(out):
                raise FileNotFoundError(f"Absolut output file not found: {out}")

            data = pd.read_csv(out, sep="\t", skiprows=1)
            data["sequence_idx"] = data.ID_slide_Variant.map(lambda value: int(str(value).split("_")[0]))
            values = data.groupby("sequence_idx")[["Energy"]].min()["Energy"].values
            return values, seqs
        finally:
            self._remove_files(antigen, n_proc, tmp, out)
            self._release_lock(lock_dir)
            os.chdir(cwd)

    @staticmethod
    def _remove_files(antigen: str, n_proc: int, tmp: str, out: str) -> None:
        for path in [tmp, out]:
            if os.path.exists(path):
                os.remove(path)
        for i in range(n_proc):
            part = f"TempBindingsFor{antigen}_t{i}_Part1_of_1.txt"
            if os.path.exists(part):
                os.remove(part)

    @staticmethod
    def _acquire_lock(lock_dir: str, timeout_s: int = 600) -> None:
        start = time.time()
        while True:
            try:
                os.mkdir(lock_dir)
                return
            except FileExistsError:
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Timed out waiting for Absolut lock: {lock_dir}")
                time.sleep(1.0)

    @staticmethod
    def _release_lock(lock_dir: str) -> None:
        try:
            os.rmdir(lock_dir)
        except FileNotFoundError:
            pass


def make_evaluator(config: dict[str, Any], antigen: str, run_id: str):
    bbox = dict(config["bbox"])
    bbox["antigen"] = antigen
    if bbox.get("tool", "Absolut") == "random":
        return RandomEvaluator(), bbox
    return AbsolutEvaluator(bbox, run_id), bbox


def append_results(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    seqs: list[str],
    llm_scores: list[float | None],
    acquisition_scores: list[float | None],
    elapsed_s: float,
    source: str,
    start_idx: int,
) -> tuple[int, float, str]:
    best_value = min((row["BestValue"] for row in rows), default=float("inf"))
    best_seq = rows[-1]["BestProtein"] if rows else ""
    idx = start_idx
    for seq, llm_score, acq_score, value in zip(seqs, llm_scores, acquisition_scores, values):
        value = float(value)
        if value < best_value:
            best_value = value
            best_seq = seq
        rows.append({
            "Index": idx,
            "LastValue": value,
            "BestValue": best_value,
            "LLMScore": llm_score,
            "AcquisitionScore": acq_score,
            "Time": elapsed_s,
            "LastProtein": seq,
            "BestProtein": best_seq,
            "Source": source,
        })
        idx += 1
    return idx, best_value, best_seq


def run_one(config: dict[str, Any], antigen: str, seed: int, args: argparse.Namespace) -> Path:
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    run_id = f"{antigen}_seed{seed}_pid{os.getpid()}"
    seq_len = int(config.get("seq_len", 11))
    pool_csv = config.get("tabular_search_csv")
    candidate_library = read_candidate_library(pool_csv, seq_len)
    if candidate_library:
        print(f"[llm-acq] Loaded candidate library: {len(candidate_library)} sequences from {pool_csv}")
    else:
        print("[llm-acq] No candidate library provided; using random temporary candidate pools.")
    llm = make_llm_client()
    evaluator, bbox = make_evaluator(config, antigen, run_id)

    acq_name = str(getattr(args, "acq", ACQ_NAME)).lower()
    acq_beta = float(getattr(args, "acq_beta", 1.0))
    mode = f"llm_parallel_{acq_name}_budget{args.parallel_budget}"
    run_dir = Path(args.out_root) / f"{mode}_antigen_{antigen}_seed_{seed}_n{args.n_evals}_batch{args.batch_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.csv"
    decision_recorder = JsonlTrajectoryRecorder(
        run_dir,
        config_snapshot={
            "algorithm": "LDM-TTS",
            "task": "antibody",
            "antigen": antigen,
            "seed": seed,
            "n_evals": int(args.n_evals),
            "batch_size": int(args.batch_size),
            "parallel_budget": int(args.parallel_budget),
            "acquisition": acq_name,
            "acq_beta": acq_beta,
        },
        rounds_filename="llm_acq_decisions.jsonl",
        reset_rounds_file=True,
    )
    from bo.ldm import DSLConfig, Orchestrator

    ldm_cfg = DSLConfig(
        llm_init_enabled=False,
        llm_loop_enabled=True,
        llm_temperature=float(args.temperature),
        max_retries=int(args.max_retries),
        llm_call_timeout_s=int(args.timeout_s),
        history_max_in_prompt=int(args.history_top_k),
        bias_weight=0.0,
        sample_timeout_s=5.0,
        batch_size=int(args.batch_size),
        acq_search_budget=int(args.parallel_budget),
        acq_max_rounds=1,
        num_llm_review=max(1, min(int(args.parallel_budget), 10)),
        strategy="ldm-default",
    )
    orchestrator = Orchestrator(
        config=ldm_cfg,
        llm_client=llm,
        decision_log_path=run_dir / "ldm_parallel_decisions.json",
    )

    antigen_context = None
    if args.include_antigen_context and bbox.get("tool", "Absolut") == "Absolut":
        antigen_context = collect_antigen_context(config, antigen)
        with open(run_dir / "llm_antigen_context.json", "w", encoding="utf-8") as f:
            json.dump(antigen_context, f, indent=2)

    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    eval_idx = 0

    while eval_idx < args.n_evals:
        batch_size = min(args.batch_size, args.n_evals - eval_idx)
        start = time.time()
        using_parallel_ldm = len(rows) >= int(args.n_init)
        if using_parallel_ldm:
            selected_candidates, decision = select_with_parallel_ldm(
                orchestrator=orchestrator,
                rows=rows,
                antigen=antigen,
                seed=seed,
                iteration=eval_idx,
                antigen_context=antigen_context,
                batch_size=batch_size,
                args=args,
            )
            candidates = decision["parallel_results"]
            selected_indices = decision["selected_indices"]
            acquisition_details = decision["parallel_results"]
            selector_source = decision["source"]
        else:
            candidates, decision = propose(
                llm=llm,
                rng=rng,
                antigen=antigen,
                seq_len=seq_len,
                batch_size=batch_size,
                observed=observed,
                rows=rows,
                candidate_library=candidate_library,
                antigen_context=antigen_context,
                args=args,
            )
            selected_indices = list(range(min(batch_size, len(candidates))))
            selected_candidates = [candidates[i] for i in selected_indices]
            acquisition_details = []
            selector_source = decision.get("source", "llm")

        proposed_seqs = [candidate["sequence"] for candidate in selected_candidates]
        llm_scores = [candidate.get("score") for candidate in selected_candidates]
        selected_acq_scores = [
            candidate.get("acquisition_score") for candidate in selected_candidates
        ]
        values, evaluated_seqs = evaluator.energy(seqs_to_indices(proposed_seqs))
        elapsed = time.time() - start

        old_idx = eval_idx
        eval_idx, best_value, _ = append_results(
            rows=rows,
            values=values,
            seqs=evaluated_seqs,
            llm_scores=llm_scores,
            acquisition_scores=selected_acq_scores,
            elapsed_s=elapsed,
            source=selector_source,
            start_idx=eval_idx,
        )
        observed.update(evaluated_seqs)

        for row in rows[old_idx:eval_idx]:
            print(
                f"[{antigen} seed={seed} eval={row['Index'] + 1}/{args.n_evals}] "
                f"y={row['LastValue']:.4f} best={best_value:.4f} "
                f"seq={row['LastProtein']} llm_score={row['LLMScore']} "
                f"acq={row['AcquisitionScore']} source={row['Source']}",
                flush=True,
            )

        decision_recorder.append_round({
            "eval_start": old_idx,
            "eval_end": eval_idx,
            "antigen": antigen,
            "seed": seed,
            "parallel_budget": int(args.parallel_budget) if using_parallel_ldm else batch_size,
            "candidates": candidates,
            "acquisition": {
                "enabled": True,
                "used": bool(using_parallel_ldm),
                "name": acq_name,
                "beta": acq_beta,
                "n_init": args.n_init,
                "parallel_executor": "bo.ldm.acquisition.parallel_search.execute_atoms" if using_parallel_ldm else None,
                "scores": acquisition_details,
                "selected_indices": selected_indices,
                "selected_candidates": selected_candidates,
            },
            "decision": decision,
            "pool_csv": pool_csv,
        })
        pd.DataFrame(rows).to_csv(results_path, index=False)

    if hasattr(llm, "close"):
        llm.close()
    return run_dir


def main() -> None:
    args = parse_args()
    if args.parallel_budget <= 0:
        raise ValueError("--parallel_budget must be positive")
    if args.n_init < 0:
        raise ValueError("--n_init must be non-negative")
    args.acq = str(args.acq).lower()
    if args.acq not in ACQ_CHOICES:
        raise ValueError(f"--acq must be one of {ACQ_CHOICES}")
    if args.acq_beta < 0:
        raise ValueError("--acq_beta must be non-negative")
    config = read_yaml(os.path.abspath(args.config))
    antigens = read_antigens(args.antigens_file)
    print(f"LLM + parallel {args.acq.upper()} baseline antigens: {antigens}")

    for antigen in antigens:
        for seed in range(args.seed, args.seed + args.n_trials):
            run_dir = run_one(config, antigen, seed, args)
            print(f"Saved LLM + parallel {args.acq.upper()} baseline run to {run_dir}")


if __name__ == "__main__":
    main()
