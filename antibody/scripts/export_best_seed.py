"""Export one required-style comparison JSON per antigen.

The output shape follows the local example ``bo-strkernel-ldm_extracted.json``:

  {
    "num_objectives": 1,
    "direction": "minimize",
    "task": "...",
    "selected_seed": 44,
    "selected_source": ".../results.csv",
    "selection_metric": "bo_llm_final_best",
    "selection_value": -109.3,
    "results": [
      {"method": "BO+LLM", "config": {...}, "trajectory": [...]},
      {"method": "LLM Only", "config": {...}, "trajectory": [...]},
      {"method": "paper:BO_transformed_overlap", "config": {...}, "trajectory": [...]}
    ]
  }

Defaults match the current local result layout:

  python scripts/export_best_seed_json_by_antigen.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BO_LLM_ROOT = Path("outputs/ldm_ninit20_iter200")
DEFAULT_LLM_BASELINE_ROOT = Path("result1/llm_baseline_5x5_200")
DEFAULT_PAPER_ROOT = Path("outputs/reproduction")
DEFAULT_OUT_DIR = Path("outputs/comparisons/best_seed_json_by_antigen")
DEFAULT_LLM_MODEL = "DeepSeek-V4-Flash"
MAX_LDM_SYS_PROMPT_CHARS = 1325
LLM_BASELINE_PROMPT_SOURCE = "bo/ldm/llm/LLM_baseline.py::build_prompt"
ANTBO_LDM_SYS_PROMPT = """1. The trust region is the BO acquisition function's search space over 11-aa CDRH3 sequences. It must not become too narrow too early; a tiny or exhausted trust region hurts convergence. If progress stalls, broaden or replace it with LocalSearch, NeighborSampling, LatinHyperCubeSampling, or Or(...).

2. Prefer conservative local search around strong sequences while BO is improving. During stagnation, add diverse centers from good-but-different history clusters or use soft trust regions (radius=None) so the GP can train on all observations while exploring a wider neighborhood.

3. Optimization objective: Absolut binding energy is minimized. Lower values are better. The BO/GP loop remains the main search engine; the LLM only steers the trust region and acquisition bias when useful.

4. Available search atoms include LocalSearch(center, fixed='***********', radius=None, restart=3, steps=200), NeighborSampling(center, fixed='***********', radius=None, mut_pr=0.5, budget=1000), LatinHyperCubeSampling(num), and Or(...). Keep total search budget within acq_search_budget.

5. Available bias atoms include MaxCysteine, MaxHydrophobicRun, MaxAromatic, NetChargeRange, and NoNGlycosylation. Omit update_trust_region or update_bias when the current setting should be kept.

6. Return only valid JSON with concise rationale and optional update_trust_region/update_bias fields."""

ANTIGEN_KERNEL_RE = re.compile(r"antigen_(?P<antigen>.+?)_kernel_")
ANTIGEN_SEED_RE = re.compile(r"antigen_(?P<antigen>.+?)_seed_")
SEED_RE = re.compile(r"(?:^|_)seed[_=](?P<seed>\d+)(?:_|$)")


@dataclass(frozen=True)
class CsvRun:
    method: str
    antigen: str
    seed: int | None
    path: Path
    final_best: float
    best_protein: str | None
    n_eval: int


def clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: clean_value(v) for k, v in record.items()}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_dotenv_public() -> dict[str, str]:
    """Read non-secret LLM metadata from .env without exposing API keys."""
    values: dict[str, str] = {}
    path = Path(".env")
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"LLM_BASE_URL", "LLM_MODEL"} and value:
            values[key] = value
    return values


def read_dotenv_values(keys: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path(".env")
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in keys:
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_bo_config() -> dict[str, Any]:
    path = Path("bo/config.yaml")
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def antbo_gp_config(config: dict[str, Any]) -> dict[str, Any]:
    noise_variance = as_float(config.get("noise_variance"))
    if noise_variance is None:
        noise_constraint_min = 1e-6
        noise_constraint_max = 0.1
        effective_noise_variance = 0.005
    elif abs(noise_variance) < 1e-6:
        noise_constraint_min = 1e-6
        noise_constraint_max = 0.1
        effective_noise_variance = 0.05
    else:
        noise_constraint_min = 0.99 * noise_variance
        noise_constraint_max = 1.01 * noise_variance
        effective_noise_variance = noise_variance

    return clean_record(
        {
            "device": config.get("device"),
            "kernel_type": config.get("kernel_type"),
            "ard": config.get("ard"),
            "noise_variance": noise_variance,
            "effective_noise_variance": effective_noise_variance,
            "noise_constraint_min": noise_constraint_min,
            "noise_constraint_max": noise_constraint_max,
            "normalise": config.get("normalise"),
            "gp_fit_itersteps": 500,
            "gp_learning_rate": 0.03,
            "optimizer": "Adam",
            "impl": "AntBO categorical GP",
            "seq_len": config.get("seq_len"),
        }
    )


def render_ldm_system_prompt(bo_config: dict[str, Any]) -> str | None:
    prompt = ANTBO_LDM_SYS_PROMPT.strip()
    if len(prompt) > MAX_LDM_SYS_PROMPT_CHARS:
        return prompt[: MAX_LDM_SYS_PROMPT_CHARS - 3].rstrip() + "..."
    return prompt


def ldm_system_prompt_full_length(bo_config: dict[str, Any]) -> int | None:
    template = read_text(Path("bo/ldm/prompts/system.txt"))
    return len(template) if template is not None else None


def llm_runtime_config() -> dict[str, Any]:
    env_values = read_dotenv_public()
    return clean_record(
        {
            "model": env_values.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            "base_url": env_values.get("LLM_BASE_URL"),
        }
    )


def first_sequences(path: Path, n: int) -> list[str]:
    df = pd.read_csv(path)
    if df.empty or "LastProtein" not in df.columns:
        return []
    return [str(x) for x in df["LastProtein"].head(n).tolist()]


def first_paper_sequences(path: Path, antigen: str, method: str, seed: int, n: int) -> list[str]:
    df = pd.read_csv(path)
    sub = df[(df["Antigen"] == antigen) & (df["Method"] == method) & (df["Seed"] == seed)].copy()
    if sub.empty or "Last Protein" not in sub.columns:
        return []
    sub = sub.sort_values("Num BB Evals")
    return [str(x) for x in sub["Last Protein"].head(n).tolist()]


def reference_gp_config(bo_config: dict[str, Any]) -> dict[str, Any]:
    gp = antbo_gp_config(bo_config)
    return clean_record(
        {
            "gp_device": gp.get("device"),
            "gp_fit_itersteps": gp.get("gp_fit_itersteps"),
            "gp_learning_rate": gp.get("gp_learning_rate"),
            "gp_min_jitter": gp.get("noise_constraint_min"),
            "gp_max_jitter": gp.get("noise_constraint_max"),
            "gp_standardize_y": gp.get("normalise"),
            "gp_fp_radius": None,
            "gp_fp_n_bits": None,
            "impl": gp.get("impl"),
            "smiles_maxlen": bo_config.get("seq_len"),
        }
    )


def reference_vina_config(bo_config: dict[str, Any], antigen: str) -> dict[str, Any]:
    bbox = bo_config.get("bbox", {})
    return clean_record(
        {
            "vina_bin": bbox.get("path"),
            "vina_cache_dir": None,
            "vina_pdb_id": antigen,
            "vina_chain_id": None,
            "vina_ligand_resname": None,
            "vina_exhaustiveness": None,
            "vina_n_poses": None,
            "vina_seed": None,
            "vina_max_workers": bbox.get("process"),
            "vina_allow_debug_receptor": None,
            "vina_no_cache": None,
        }
    )


def reference_reasyn_config() -> dict[str, Any]:
    return clean_record(
        {
            "reasyn_model_path": None,
            "reasyn_devices": None,
            "reasyn_repo": None,
            "reasyn_python_bin": None,
            "reasyn_search_width": None,
            "reasyn_exhaustiveness": None,
            "reasyn_num_cycles": None,
            "reasyn_num_editflow_samples": None,
            "reasyn_num_editflow_steps": None,
            "reasyn_time_limit": None,
            "reasyn_num_workers_per_gpu": None,
            "reasyn_filter_sim": None,
            "reasyn_no_canonicalize": None,
        }
    )


def reference_llm_config(run_dir: Path, prompt: str | None, pool_min_size: int | None = None) -> dict[str, Any]:
    runtime = llm_runtime_config()
    return clean_record(
        {
            "model": runtime.get("model"),
            "base_url": runtime.get("base_url"),
            "trajectory_dir": str(run_dir),
            "pool_min_size": pool_min_size,
            "ldm_sys_prompt": prompt,
        }
    )


def reference_config(
    *,
    method: str,
    run: CsvRun,
    bo_config: dict[str, Any],
    seed_smiles: list[str] | None,
    batch_size: int | None,
    init_size: int | None,
    acquisition: str | None,
    acq_budget: int | None,
    llm_prompt: str | None,
) -> dict[str, Any]:
    return clean_record(
        {
            "method": method,
            "seed": run.seed,
            "seed_smiles": seed_smiles,
            "num_evaluations": run.n_eval,
            "batch_size": batch_size,
            "init_size": init_size,
            "acquisition": acquisition,
            "xi": None,
            "kappa": None,
            "minimize": [True],
            "acq_budget": acq_budget,
            "max_pool_size": None,
            "pool_min_size": None,
            "pool_max_size": None,
            "smiles_max_len": bo_config.get("seq_len"),
            "objective": "binding_energy",
            "n_objectives": 1,
            "objective_parts": ["binding_energy"],
            "ehvi_n_samples": None,
            "che_alpha": None,
            "gp": reference_gp_config(bo_config),
            "ref_point": None,
            "vina": reference_vina_config(bo_config, run.antigen),
            "reasyn": reference_reasyn_config(),
            "llm": reference_llm_config(run.path.parent, llm_prompt),
        }
    )


def parse_antigen(path: Path) -> str | None:
    text = str(path)
    for regex in (ANTIGEN_KERNEL_RE, ANTIGEN_SEED_RE):
        match = regex.search(text)
        if match:
            return match.group("antigen")
    return None


def parse_seed(path: Path) -> int | None:
    for part in path.parts:
        match = SEED_RE.search(part)
        if match:
            return int(match.group("seed"))
    match = SEED_RE.search(str(path))
    return int(match.group("seed")) if match else None


def result_csv_summary(path: Path, method: str) -> CsvRun | None:
    antigen = parse_antigen(path)
    if antigen is None:
        return None

    df = pd.read_csv(path)
    if df.empty or "BestValue" not in df.columns:
        return None

    final = df.iloc[-1]
    best_protein = str(final["BestProtein"]) if "BestProtein" in df.columns else None
    return CsvRun(
        method=method,
        antigen=antigen,
        seed=parse_seed(path),
        path=path,
        final_best=float(final["BestValue"]),
        best_protein=best_protein,
        n_eval=int(len(df)),
    )


def discover_runs(root: Path, method: str) -> list[CsvRun]:
    if not root.exists():
        return []
    runs = []
    for path in sorted(root.rglob("results.csv")):
        summary = result_csv_summary(path, method)
        if summary is not None:
            runs.append(summary)
    return runs


def choose_best_runs(runs: list[CsvRun]) -> dict[str, CsvRun]:
    best: dict[str, CsvRun] = {}
    for run in runs:
        current = best.get(run.antigen)
        if current is None:
            best[run.antigen] = run
            continue
        current_seed = current.seed if current.seed is not None else 10**9
        run_seed = run.seed if run.seed is not None else 10**9
        if (run.final_best, run_seed) < (current.final_best, current_seed):
            best[run.antigen] = run
    return best


def decision_label(decision: dict[str, Any]) -> str:
    """Fallback label when no DeepSeek-generated label is available."""
    parsed = decision.get("llm_response_parsed") or decision
    if "stage_a1" in parsed:
        parsed = parsed.get("stage_a1") or {}

    parts = []
    if parsed.get("update_trust_region"):
        parts.append("search policy updated")
    if parsed.get("update_bias"):
        parts.append("developability bias updated")
    if parsed.get("sequences"):
        parts.append("LLM candidate selected")
    if decision.get("fallback_used"):
        parts.append("fallback used")
    if not parts and decision.get("outcome"):
        parts.append(str(decision["outcome"]))
    return ", ".join(parts) if parts else "LLM intervention logged"


def sequence_from_indices(value: Any) -> Any:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    if not isinstance(value, list):
        return value
    try:
        return "".join(alphabet[int(float(i))] for i in value)
    except (TypeError, ValueError, IndexError):
        return value


def compact_bo_llm_reflection(decision):
    parsed = decision.get("llm_response_parsed") or {}

    return clean_record({
        "stage_a1": {
            "type": "update_search_policy",
            "rationale": decision.get("rationale"),
            "update_trust_region": parsed.get("update_trust_region"),
            "update_bias": parsed.get("update_bias"),
            "outcome": decision.get("outcome"),
            "best_value": decision.get("best_value"),
            "best_sequence": sequence_from_indices(decision.get("best_sequence")),
        }
    })


def extract_search_summary(update_trust_region: Any) -> dict[str, Any]:
    """Extract compact, factual search-policy fields from an AntBO LDM update string."""
    if not isinstance(update_trust_region, str):
        return {
            "atom": None,
            "centers": [],
            "radius": None,
            "restart": None,
            "steps": None,
            "budget": None,
            "mut_pr": None,
        }

    atoms = re.findall(
        r"\b(LocalSearch|NeighborSampling|LatinHyperCubeSampling|Or)\s*\(",
        update_trust_region,
    )
    centers = re.findall(r"center=['\"]?([A-Z]{5,})['\"]?", update_trust_region)
    radius_match = re.search(r"radius=([^,\)]+)", update_trust_region)
    restart_match = re.search(r"restart=(\d+)", update_trust_region)
    steps_match = re.search(r"steps=(\d+)", update_trust_region)
    budget_match = re.search(r"budget=(\d+)", update_trust_region)
    mut_pr_match = re.search(r"mut_pr=([0-9.]+)", update_trust_region)

    unique_atoms = []
    for atom in atoms:
        if atom not in unique_atoms:
            unique_atoms.append(atom)

    unique_centers = []
    for center in centers:
        if center not in unique_centers:
            unique_centers.append(center)

    return {
        "atom": "+".join(unique_atoms) if unique_atoms else None,
        "centers": unique_centers[:5],
        "radius": radius_match.group(1).strip() if radius_match else None,
        "restart": int(restart_match.group(1)) if restart_match else None,
        "steps": int(steps_match.group(1)) if steps_match else None,
        "budget": int(budget_match.group(1)) if budget_match else None,
        "mut_pr": float(mut_pr_match.group(1)) if mut_pr_match else None,
    }


def compact_bias_summary(update_bias: Any) -> dict[str, Any] | None:
    """Extract factual bias terms without treating omitted bias as removal."""
    if update_bias is None:
        return None

    text = update_bias if isinstance(update_bias, str) else json.dumps(update_bias, sort_keys=True)
    terms = []
    for name in ["MaxCysteine", "MaxHydrophobicRun", "MaxAromatic", "NetChargeRange", "NoNGlycosylation"]:
        if name in text:
            terms.append(name)

    return {
        "terms": terms,
        "raw": text[:300],
    }


def compact_trajectory_for_llm(path: Path, n_init: int | None = None) -> list[dict[str, Any]]:
    """Compress a full results.csv trajectory into factual rows for DeepSeek."""
    df = pd.read_csv(path)
    rows = []

    for pos, row in df.reset_index(drop=True).iterrows():
        if n_init is not None:
            stage = "init" if pos < n_init else "loop"
            iter_id = 0 if pos < n_init else pos - n_init + 1
        else:
            stage = "loop"
            iter_id = pos

        rows.append(
            clean_record(
                {
                    "eval": pos,
                    "iter": iter_id,
                    "stage": stage,
                    "sequence": row.get("LastProtein"),
                    "objective": row.get("LastValue"),
                    "best_value": row.get("BestValue"),
                    "best_sequence": row.get("BestProtein"),
                }
            )
        )

    return rows


def compact_bo_llm_decisions_for_llm(run_dir: Path) -> list[dict[str, Any]]:
    """Compress llm_decisions.json into factual rows for DeepSeek."""
    data = read_json(run_dir / "llm_decisions.json")
    if not data:
        return []

    rows = []
    for decision in data.get("decisions", []):
        parsed = decision.get("llm_response_parsed") or {}
        update_trust_region = parsed.get("update_trust_region")
        update_bias = parsed.get("update_bias")

        rows.append(
            clean_record(
                {
                    "eval": decision.get("n_evals"),
                    "outcome": decision.get("outcome"),
                    "fallback_used": decision.get("fallback_used"),
                    "retry_count": decision.get("retry_count"),
                    "best_value": decision.get("best_value"),
                    "best_sequence": sequence_from_indices(decision.get("best_sequence")),
                    "search_summary": extract_search_summary(update_trust_region),
                    "bias_summary": compact_bias_summary(update_bias),
                    "update_trust_region": update_trust_region,
                    "update_bias": update_bias,
                    "rationale": decision.get("rationale"),
                }
            )
        )

    return rows


def call_deepseek_chat_json(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout: int = 90,
) -> dict[str, Any] | None:
    """Call a DeepSeek/OpenAI-compatible chat completion endpoint and parse JSON content."""
    dotenv = read_dotenv_values({"DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL"})
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or dotenv.get("DEEPSEEK_API_KEY")
        or dotenv.get("LLM_API_KEY")
        or dotenv.get("OPENAI_API_KEY")
    )
    base_url = os.environ.get("LLM_BASE_URL") or dotenv.get("LLM_BASE_URL") or "https://api.deepseek.com"
    model = os.environ.get("LLM_MODEL") or dotenv.get("LLM_MODEL") or "deepseek-chat"

    if not api_key:
        print("[DeepSeek SKIP] DEEPSEEK_API_KEY/LLM_API_KEY/OPENAI_API_KEY is not set.")
        return None

    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        print(f"[DeepSeek FAIL] HTTP {e.code}: {detail[:500]}")
        return None
    except Exception as e:
        print(f"[DeepSeek FAIL] {type(e).__name__}: {e}")
        return None

    try:
        payload = json.loads(raw)
        content = payload["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception as e:
        print(f"[DeepSeek FAIL] could not parse JSON response: {e}")
        print(raw[:500])
        return None



def build_reflection_event_context(
    *,
    trajectory: list[dict[str, Any]],
    llm_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build explicit previous->current change records so DeepSeek can describe
    how the trust region and bias changed, not only the current setting.
    """
    traj_by_eval = {
        int(row["eval"]): row
        for row in trajectory
        if row.get("eval") is not None
    }

    events = []
    prev_eval = None
    prev_best = None
    prev_best_seq = None
    prev_search = {
        "atom": None,
        "centers": [],
        "radius": None,
        "restart": None,
        "steps": None,
        "budget": None,
        "mut_pr": None,
    }
    prev_bias_terms = []
    prev_bias_raw = None

    for decision in llm_decisions:
        eval_id = decision.get("eval")
        if eval_id is None:
            continue
        eval_id = int(eval_id)

        traj = traj_by_eval.get(eval_id, {})
        curr_best = as_float(traj.get("best_value"))
        curr_best_seq = traj.get("best_sequence")
        curr_obj = as_float(traj.get("objective"))
        curr_seq = traj.get("sequence")

        search = decision.get("search_summary") or {}
        search = {
            "atom": search.get("atom"),
            "centers": search.get("centers") or [],
            "radius": search.get("radius"),
            "restart": search.get("restart"),
            "steps": search.get("steps"),
            "budget": search.get("budget"),
            "mut_pr": search.get("mut_pr"),
        }

        bias = decision.get("bias_summary") or {}
        bias_terms = bias.get("terms") or []
        bias_raw = bias.get("raw")

        best_delta = None
        if prev_best is not None and curr_best is not None:
            best_delta = curr_best - prev_best

        gap = None
        if prev_eval is not None:
            gap = eval_id - prev_eval

        trust_region_changes = {}
        trust_region_change_list = []

        for key in ["atom", "centers", "radius", "restart", "steps", "budget", "mut_pr"]:
            before = prev_search.get(key)
            after = search.get(key)
            if before != after:
                trust_region_changes[key] = {"from": before, "to": after}
                trust_region_change_list.append(f"{key}: {before}->{after}")

        previous_center_count = len(prev_search.get("centers") or [])
        current_center_count = len(search.get("centers") or [])
        center_count_changed = previous_center_count != current_center_count

        bias_change = None
        bias_change_list = []
        if bias_terms and bias_terms != prev_bias_terms:
            bias_change = {"from": prev_bias_terms, "to": bias_terms}
            bias_change_list.append(f"bias_terms: {prev_bias_terms}->{bias_terms}")
        elif not bias_terms and prev_bias_terms:
            # In this codebase, omitted update_bias means no new bias update, not active removal.
            bias_change = "no new bias update"
            bias_change_list.append("bias_terms: no new update")
        elif bias_terms:
            bias_change = "same bias terms"
            bias_change_list.append(f"bias_terms unchanged: {bias_terms}")
        else:
            bias_change = "bias absent"
            bias_change_list.append("bias absent")

        event = clean_record(
            {
                "eval": eval_id,
                "prev_eval": prev_eval,
                "eval_gap_from_previous_decision": gap,
                "current_sequence": curr_seq,
                "current_objective": curr_obj,
                "current_best_value": curr_best,
                "previous_best_value": prev_best,
                "best_delta": best_delta,
                "current_best_sequence": curr_best_seq,
                "previous_best_sequence": prev_best_seq,
                "best_sequence_changed": bool(prev_best_seq and curr_best_seq and prev_best_seq != curr_best_seq),

                # Current trust-region state.
                "trust_region_current": search,
                "trust_region_atom": search.get("atom"),
                "trust_region_center_count": current_center_count,
                "trust_region_centers": search.get("centers") or [],
                "trust_region_radius": search.get("radius"),
                "trust_region_restart": search.get("restart"),
                "trust_region_steps": search.get("steps"),
                "trust_region_budget": search.get("budget"),
                "trust_region_mut_pr": search.get("mut_pr"),

                # Previous trust-region state.
                "trust_region_previous": prev_search,
                "previous_trust_region_atom": prev_search.get("atom"),
                "previous_trust_region_center_count": previous_center_count,
                "previous_trust_region_centers": prev_search.get("centers") or [],
                "previous_trust_region_radius": prev_search.get("radius"),
                "previous_trust_region_restart": prev_search.get("restart"),
                "previous_trust_region_steps": prev_search.get("steps"),
                "previous_trust_region_budget": prev_search.get("budget"),
                "previous_trust_region_mut_pr": prev_search.get("mut_pr"),

                # Explicit change fields DeepSeek should use in labels.
                "trust_region_changes": trust_region_changes,
                "trust_region_change_list": trust_region_change_list,
                "trust_region_changed": bool(trust_region_changes),
                "center_count_changed": center_count_changed,

                # Bias state and change.
                "bias_terms": bias_terms,
                "bias_term_count": len(bias_terms),
                "bias_raw": bias_raw,
                "previous_bias_terms": prev_bias_terms,
                "previous_bias_raw": prev_bias_raw,
                "bias_change": bias_change,
                "bias_change_list": bias_change_list,

                "outcome": decision.get("outcome"),
                "fallback_used": decision.get("fallback_used"),
                "retry_count": decision.get("retry_count"),
            }
        )
        events.append(event)

        prev_eval = eval_id
        if curr_best is not None:
            prev_best = curr_best
        if curr_best_seq:
            prev_best_seq = curr_best_seq

        # Only update previous search if the current decision actually contains a trust-region update.
        if search.get("atom") or search.get("centers") or search.get("radius") is not None or search.get("budget") is not None:
            prev_search = search

        if bias_terms:
            prev_bias_terms = bias_terms
            prev_bias_raw = bias_raw

    return events


def sanitize_llm_label(label: str, max_words: int = 32) -> str:
    """Remove eval wording from labels and enforce a short one-sentence label."""
    label = " ".join(str(label).strip().split())

    # Remove common eval prefixes: "Eval 22->23:", "eval 22:", etc.
    label = re.sub(r"^\s*eval(?:uation)?\s+\d+(?:\s*[-–>]+\s*\d+)?\s*[:;,-]?\s*", "", label, flags=re.I)

    # Remove in-sentence eval spans if the model still includes them.
    label = re.sub(r"\b[Ee]val(?:uation)?\s+\d+(?:\s*[-–>]+\s*\d+)?\s*[:;,-]?\s*", "", label)

    # Clean duplicated separators after deletion.
    label = re.sub(r"^\s*[:;,-]\s*", "", label)
    label = re.sub(r"\s*;\s*;", ";", label)
    label = label.strip(" ;,:")

    words = label.split()
    if len(words) > max_words:
        label = " ".join(words[:max_words]).rstrip(" ,.;:") + "."

    return label


def call_deepseek_reflection_plan(
    *,
    trajectory: list[dict[str, Any]],
    llm_decisions: list[dict[str, Any]],
    max_points: int | None = None,
) -> dict[int, str]:
    """
    Give DeepSeek the whole trajectory and decision list.
    It selects key reflection points and writes concrete, numeric labels.
    """
    event_context = build_reflection_event_context(
        trajectory=trajectory,
        llm_decisions=llm_decisions,
    )

    system_prompt = (
        "You are summarizing an AntBO CDRH3 optimization run. "
        "This is single-objective minimization, but labels should NOT focus on best_value. "
        "Select only key LLM reflection points from the full trajectory; do not label every step. "
        "Use only provided policy fields and numbers; do not invent values. "
        "Do not include eval numbers or the word Eval in labels. "
        "Do not mention best improved, best_value, or objective unless it is secondary and necessary. "
        "Each label must be one factual sentence under 42 words. "
        "The label must mainly describe trust-region and bias changes. "
        "If trust_region_changed is true, explicitly state at least one from->to change from trust_region_change_list. "
        "If trust_region_changed is false, state the current trust-region setting, but do not call it a change. "
        "For trust region, mention concrete values such as search atom, center count/sequence, radius, restart, steps, budget, or mut_pr. "
        "For bias, mention whether update_bias is present, absent, unchanged, or no new update; include named bias terms and values when provided. "
        "Include a short reasoning phrase explaining why this matters, such as 'to widen exploration', 'to keep local search stable', or 'to enforce developability'. "
        "Other trajectory information may be included only after trust-region/bias details. "
        "Omitted update_bias means no new bias update, not bias removal. "
        "Return only valid JSON."
    )

    user_payload = {
        "task": "Select key LLM reflection points for AntBO trajectory JSON.",
        "objective_direction": "minimize",
        "label_requirements": [
            "Use eval indices internally for JSON placement, but do not mention eval numbers in label text.",
            "Do not write best improved, best_value, objective, or energy as the main message.",
            "Prioritize trust-region changes: number of centers/regions, search atom, center sequence, radius, restart, steps, budget, and mut_pr.",
            "When trust_region_changed is true, include an explicit from->to change from trust_region_change_list.",
            "Examples of trust-region changes: LocalSearch->NeighborSampling, centers A->B, radius None->3, budget 500->1000, mut_pr 0.3->0.5.",
            "If trust_region_changed is false, describe the current trust-region setting without saying it changed.",
            "Prioritize bias handling: whether bias was added, changed, unchanged, omitted, or absent; include exact bias terms and values when available.",
            "If update_bias is omitted, say 'no new bias update' only when relevant; never say bias removed.",
            "If multiple centers are present, mention the count and one or two representative centers.",
            "Each label should include at least one trust-region detail and one bias detail, if both exist.",
            "Each label may include a short reason, for example 'to widen exploration', 'to keep local search stable', or 'to enforce developability'.",
            "Other information such as stagnation, fallback, retry, or sequence history can be mentioned after trust-region/bias details.",
            "Keep labels compact, factual, and data-like; avoid broad scientific explanations.",
            "Good: 'Trust region LocalSearch->NeighborSampling, center FFCLFLLVLNL->FFCLFSLVFLL, budget 1000; no new bias update, to widen exploration.'",
            "Good: 'LocalSearch kept 1 center FFCLFSLVFLL, radius None, restart 3; bias unchanged MaxHydrophobicRun, to keep local search stable.'",
            "Good: 'Trust region radius None->3 around FFCLFSLVFLL; bias MaxHydrophobicRun+NoNGlycosylation added, to enforce developability.'",
            "Bad: 'LocalSearch center FFCLFSLVFLL, radius None, restart 3'. It gives current settings but no change when change fields exist.",
            "Bad: 'Search was updated'. It lacks concrete trust-region or bias values.",
        ],
        "output_schema": {
            "reflection_points": [
                {
                    "eval": "integer eval index present in llm_decisions",
                    "label": "one factual sentence under 42 words focused on explicit trust-region changes and bias handling",
                }
            ]
        },
        "max_reflection_points": max_points if max_points is not None else "no fixed maximum; select all genuinely informative reflection points",
        "trajectory": trajectory,
        "llm_decisions": llm_decisions,
        "event_context": event_context,
    }

    parsed = call_deepseek_chat_json(system_prompt=system_prompt, user_payload=user_payload)
    if not parsed:
        return {}

    valid_evals = {int(item["eval"]) for item in llm_decisions if item.get("eval") is not None}
    out: dict[int, str] = {}

    for item in parsed.get("reflection_points", []):
        eval_id = item.get("eval")
        label = item.get("label")

        if not isinstance(eval_id, int):
            continue
        if eval_id not in valid_evals:
            continue
        if not isinstance(label, str) or not label.strip():
            continue

        label = sanitize_llm_label(label, max_words=42)
        if not label:
            continue

        # Guardrail: reject labels that are only eval + best delta and no policy/context detail.
        lower = label.lower()
        if "eval" in lower:
            print(f"[DeepSeek label rejected: contains eval] eval={eval_id}: {label}")
            continue
        if "best improved" in lower:
            print(f"[DeepSeek label rejected: banned wording] eval={eval_id}: {label}")
            continue

        has_trust_detail = any(
            token in lower
            for token in [
                "center",
                "centers",
                "localsearch",
                "neighborsampling",
                "latinhypercube",
                "trust region",
                "radius",
                "restart",
                "steps",
                "budget",
                "mut_pr",
                "mutation",
            ]
        )
        has_bias_handling = any(
            token in lower
            for token in [
                "bias",
                "maxcysteine",
                "maxhydrophobicrun",
                "maxaromatic",
                "netchargerange",
                "nonglycosylation",
                "cysteine",
                "hydrophobic",
                "aromatic",
                "charge",
                "glycosylation",
                "no new bias",
            ]
        )
        has_reasoning = any(
            token in lower
            for token in [
                "to ",
                "for ",
                "after ",
                "keeping",
                "widening",
                "stabil",
                "enforce",
                "filter",
                "avoid",
                "exploration",
                "local search",
                "developability",
                "stagnation",
            ]
        )

        if not has_trust_detail:
            print(f"[DeepSeek label rejected: missing trust-region detail] eval={eval_id}: {label}")
            continue
        if not has_bias_handling:
            print(f"[DeepSeek label rejected: missing bias handling] eval={eval_id}: {label}")
            continue
        if not has_reasoning:
            print(f"[DeepSeek label rejected: missing brief reasoning] eval={eval_id}: {label}")
            continue

        # Avoid score-only labels. Score can appear, but not as the leading claim.
        leading = lower.split(";")[0].strip()
        if leading.startswith("best") or leading.startswith("best_value") or leading.startswith("objective"):
            print(f"[DeepSeek label rejected: score-led label] eval={eval_id}: {label}")
            continue

        out[eval_id] = label

    print(f"[DeepSeek OK] selected {len(out)} reflection points.")
    return out




def fallback_reflection_plan(
    *,
    trajectory: list[dict[str, Any]],
    llm_decisions: list[dict[str, Any]],
    max_points: int | None = None,
) -> dict[int, str]:
    """Rule-based backup if DeepSeek is unavailable."""
    events = build_reflection_event_context(
        trajectory=trajectory,
        llm_decisions=llm_decisions,
    )
    candidates: list[tuple[int, int, str]] = []

    for event in events:
        eval_id = event.get("eval")
        if eval_id is None:
            continue
        eval_id = int(eval_id)

        current = event.get("trust_region_current") or {}
        previous = event.get("trust_region_previous") or {}
        changes = event.get("trust_region_changes") or {}
        change_list = event.get("trust_region_change_list") or []

        atom = current.get("atom")
        centers = current.get("centers") or []
        center = centers[0] if centers else None
        center_count = len(centers)
        radius = current.get("radius")
        restart = current.get("restart")
        steps = current.get("steps")
        budget = current.get("budget")
        mut_pr = current.get("mut_pr")

        bias_terms = event.get("bias_terms") or []
        previous_bias_terms = event.get("previous_bias_terms") or []
        bias_change = event.get("bias_change")
        fallback_used = bool(event.get("fallback_used"))
        retry_count = int(event.get("retry_count") or 0)
        gap = event.get("eval_gap_from_previous_decision")

        has_search = bool(atom or center or radius is not None or budget is not None or mut_pr is not None)
        has_bias = bool(bias_terms or bias_change)
        exceptional = fallback_used or retry_count > 0

        if not (has_search or has_bias or exceptional):
            continue

        parts = []

        if changes:
            # Prioritize explicit from->to changes.
            explicit = []
            if "atom" in changes:
                explicit.append(f"trust region {changes['atom']['from']}->{changes['atom']['to']}")
            if "centers" in changes:
                before = changes["centers"]["from"] or []
                after = changes["centers"]["to"] or []
                before_s = before[0] if before else "None"
                after_s = after[0] if after else "None"
                explicit.append(f"center {before_s}->{after_s}")
            if "radius" in changes:
                explicit.append(f"radius {changes['radius']['from']}->{changes['radius']['to']}")
            if "budget" in changes:
                explicit.append(f"budget {changes['budget']['from']}->{changes['budget']['to']}")
            if "mut_pr" in changes:
                explicit.append(f"mut_pr {changes['mut_pr']['from']}->{changes['mut_pr']['to']}")
            if "restart" in changes:
                explicit.append(f"restart {changes['restart']['from']}->{changes['restart']['to']}")
            if "steps" in changes:
                explicit.append(f"steps {changes['steps']['from']}->{changes['steps']['to']}")
            parts.append(", ".join(explicit[:3]))
        elif atom:
            trust = atom
            if center_count > 1:
                trust += f" kept {center_count} centers"
                if center:
                    trust += f", including {center}"
            elif center:
                trust += f" kept center {center}"
            if radius is not None:
                trust += f", radius {radius}"
            if restart is not None:
                trust += f", restart {restart}"
            if steps is not None:
                trust += f", steps {steps}"
            if budget is not None:
                trust += f", budget {budget}"
            if mut_pr is not None:
                trust += f", mut_pr {mut_pr}"
            parts.append(trust)

        if bias_terms and bias_terms != previous_bias_terms:
            if previous_bias_terms:
                parts.append("bias " + "+".join(previous_bias_terms[:3]) + "->" + "+".join(bias_terms[:3]))
            else:
                parts.append("bias added " + "+".join(bias_terms[:4]))
        elif bias_terms:
            parts.append("bias unchanged " + "+".join(bias_terms[:4]))
        elif bias_change == "no new bias update":
            parts.append("no new bias update")
        else:
            parts.append("bias absent")

        reason = None
        if fallback_used:
            reason = "for fallback recovery"
        elif retry_count > 0:
            reason = f"after {retry_count} retry"
        elif changes.get("atom") and "NeighborSampling" in str(changes["atom"].get("to")):
            reason = "to widen exploration"
        elif atom and "NeighborSampling" in str(atom):
            reason = "to widen exploration"
        elif atom and "LocalSearch" in str(atom):
            reason = "to keep local search stable"
        elif bias_terms:
            reason = "to enforce developability"
        elif gap is not None and gap > 1:
            reason = f"after {gap} decision gap"

        if reason:
            parts.append(reason)

        label = "; ".join(parts[:3])
        label = sanitize_llm_label(label, max_words=42)

        if not label:
            continue

        priority = 0
        if exceptional:
            priority += 50
        if changes:
            priority += 30 + min(len(changes), 5)
        if bias_terms and bias_terms != previous_bias_terms:
            priority += 20
        if event.get("best_sequence_changed"):
            priority += 10
        best_delta = event.get("best_delta")
        if isinstance(best_delta, (int, float)) and best_delta < 0:
            priority += 8
        if gap is not None and gap > 1:
            priority += min(int(gap), 10)

        candidates.append((eval_id, priority, label))

    return select_reflection_points_with_coverage(candidates, max_points=max_points)


def select_reflection_points_with_coverage(
    candidates: list[tuple[int, int, str]],
    max_points: int | None,
) -> dict[int, str]:
    """Select sparse reflection points without collapsing to the first few evals."""
    if not candidates:
        return {}

    by_eval: dict[int, tuple[int, str]] = {}
    for eval_id, priority, label in candidates:
        current = by_eval.get(eval_id)
        if current is None or priority > current[0]:
            by_eval[eval_id] = (priority, label)

    ordered = sorted((eval_id, priority, label) for eval_id, (priority, label) in by_eval.items())
    if max_points is None:
        return {eval_id: label for eval_id, _, label in ordered}
    if max_points <= 0:
        return {}
    if len(ordered) <= max_points:
        return {eval_id: label for eval_id, _, label in ordered}

    selected: set[int] = {ordered[0][0], ordered[-1][0]}

    n_coverage = max(0, max_points // 2 - len(selected))
    if n_coverage > 0:
        last = len(ordered) - 1
        for i in range(1, n_coverage + 1):
            idx = round(i * last / (n_coverage + 1))
            selected.add(ordered[idx][0])

    for eval_id, _, _ in sorted(ordered, key=lambda item: (-item[1], item[0])):
        if len(selected) >= max_points:
            break
        selected.add(eval_id)

    return {
        eval_id: by_eval[eval_id][1]
        for eval_id in sorted(selected)
    }


def compact_llm_baseline_reflection(entry: dict[str, Any]) -> dict[str, Any]:
    decision = entry.get("decision") or {}
    selected = entry.get("candidates") or []
    sequences = [item.get("sequence") for item in selected if isinstance(item, dict) and item.get("sequence")]
    decisions = {seq: "take" for seq in sequences}
    return clean_record(
        {
            "stage_a1": {
                "type": "propose",
                "rationale": "LLM selected antibody sequence(s) from the available candidate pool for evaluation.",
                "sequences": sequences,
                "scores": [
                    item.get("score")
                    for item in selected
                    if isinstance(item, dict) and item.get("score") is not None
                ],
                "source": decision.get("source"),
                "attempt": decision.get("attempt"),
            },
            "stage_b": {
                "type": "review_bo",
                "rationale": "Approve selected LLM baseline sequence(s) for oracle evaluation.",
                "decisions": decisions,
            },
        }
    )


def bo_llm_decisions_by_eval(
    run_dir: Path,
    results_path: Path | None = None,
    use_deepseek_reflection_plan: bool = False,
    n_init: int | None = 20,
    max_reflection_points: int | None = None,
) -> dict[int, dict[str, Any]]:
    data = read_json(run_dir / "llm_decisions.json")
    if not data:
        return {}

    decisions_by_eval: dict[int, dict[str, Any]] = {}
    for decision in data.get("decisions", []):
        n_evals = decision.get("n_evals")
        if n_evals is None:
            continue
        decisions_by_eval[int(n_evals)] = decision

    labels_by_eval: dict[int, str] = {}

    if results_path is not None:
        trajectory_for_llm = compact_trajectory_for_llm(results_path, n_init=n_init)
        decisions_for_llm = compact_bo_llm_decisions_for_llm(run_dir)

        if use_deepseek_reflection_plan:
            labels_by_eval = call_deepseek_reflection_plan(
                trajectory=trajectory_for_llm,
                llm_decisions=decisions_for_llm,
                max_points=max_reflection_points,
            )

        if not labels_by_eval:
            print("[Reflection plan] using rule-based fallback.")
            labels_by_eval = fallback_reflection_plan(
                trajectory=trajectory_for_llm,
                llm_decisions=decisions_for_llm,
                max_points=max_reflection_points,
            )

    if not labels_by_eval:
        eval_ids = sorted(decisions_by_eval)
        if max_reflection_points is not None:
            eval_ids = eval_ids[:max_reflection_points]
        for eval_id in eval_ids:
            labels_by_eval[eval_id] = decision_label(decisions_by_eval[eval_id])

    out: dict[int, dict[str, Any]] = {}
    for eval_id, label in sorted(labels_by_eval.items()):
        decision = decisions_by_eval.get(eval_id)
        if decision is None:
            continue
        reflection = compact_bo_llm_reflection(decision)
        reflection["_llm_label"] = label
        out[eval_id] = reflection

    return out


def llm_baseline_decisions_by_eval(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Keep LLM-only reflections sparse: first point and actual best improvements."""
    decisions = read_jsonl(run_dir / "llm_only_decisions.jsonl")
    results_path = run_dir / "results.csv"
    if not decisions or not results_path.exists():
        return {}

    df = pd.read_csv(results_path)
    selected: dict[int, dict[str, Any]] = {}
    last_best = None

    for i, decision in enumerate(decisions):
        if i >= len(df):
            break

        best_value = as_float(df.iloc[i].get("BestValue"))
        is_first = len(selected) == 0
        best_improved = (
            best_value is not None
            and last_best is not None
            and best_value < last_best
        )

        decision_meta = decision.get("decision") or {}
        source = decision_meta.get("source")
        fallback_used = bool(source and source != "llm")

        if is_first or best_improved or fallback_used:
            selected[i] = decision

        if best_value is not None:
            if last_best is None or best_value < last_best:
                last_best = best_value

    return {i: compact_llm_baseline_reflection(decision) for i, decision in selected.items()}


def antbo_csv_to_trajectory(
    path: Path,
    stage_name: str,
    n_init: int | None = None,
    llm_decisions_by_eval: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    llm_decisions_by_eval = llm_decisions_by_eval or {}
    trajectory = []
    for pos, row in df.reset_index(drop=True).iterrows():
        row_id = int(clean_value(row.get("Index", pos)))
        if n_init is not None:
            stage = "init" if pos < n_init else "loop"
            iter_id = 0 if pos < n_init else pos - n_init + 1
        else:
            stage = stage_name
            iter_id = row_id

        item = {
            "id": row_id,
            "iter": iter_id,
            "decision": clean_value(row.get("LastProtein")),
            "objective": clean_value(row.get("LastValue")),
            "stage": stage,
            "hypervolume": clean_value(row.get("BestValue")),
            "llm_reflection": None,
            "llm_label": None,
        }
        llm_decision = llm_decisions_by_eval.get(pos)
        if llm_decision is not None:
            stage_b = llm_decision.get("stage_b")
            if isinstance(stage_b, dict) and "decisions" not in stage_b and item.get("decision"):
                stage_b["decisions"] = {str(item["decision"]): "take"}
            item["llm_reflection"] = llm_decision
            if "_llm_label" in llm_decision:
                item["llm_label"] = llm_decision.pop("_llm_label")
            elif "stage_a1" in llm_decision:
                item["llm_label"] = decision_label(llm_decision)
            elif "selected" in llm_decision:
                item["llm_label"] = "LLM selected candidate"
        trajectory.append(clean_record(item))
    return trajectory


def load_paper_final_rows(paper_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(paper_root.glob("*_optim_res.csv")):
        df = pd.read_csv(path)
        required = {"Method", "Antigen", "Seed", "Num BB Evals", "Best Binding Energy"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        df["__source_path"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Method", "Antigen", "Seed", "Num BB Evals"])
    return df.groupby(["Method", "Antigen", "Seed"], as_index=False).tail(1)


def paper_stats_and_best_seed(final_rows: pd.DataFrame, antigen: str) -> list[dict[str, Any]]:
    if final_rows.empty:
        return []

    rows = []
    antigen_rows = final_rows[final_rows["Antigen"] == antigen]
    for method, sub in antigen_rows.groupby("Method"):
        best_idx = sub["Best Binding Energy"].idxmin()
        best_row = sub.loc[best_idx]
        rows.append(
            clean_record(
                {
                    "method": str(method),
                    "source_path": str(best_row["__source_path"]),
                    "best_seed": int(best_row["Seed"]),
                    "best_final": float(best_row["Best Binding Energy"]),
                    "best_protein": str(best_row.get("Best Protein", "")),
                    "mean_final": float(sub["Best Binding Energy"].mean()),
                    "std_final": float(sub["Best Binding Energy"].std(ddof=0)),
                    "num_seeds": int(sub["Seed"].nunique()),
                    "max_eval": int(sub["Num BB Evals"].max()),
                }
            )
        )
    return sorted(rows, key=lambda x: x["method"])


def paper_csv_to_trajectory(path: Path, antigen: str, method: str, seed: int) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    sub = df[(df["Antigen"] == antigen) & (df["Method"] == method) & (df["Seed"] == seed)].copy()
    sub = sub.sort_values("Num BB Evals").reset_index(drop=True)

    trajectory = []
    for pos, row in sub.iterrows():
        eval_id = int(row["Num BB Evals"])
        item = {
            "id": eval_id - 1,
            "iter": eval_id - 1,
            "decision": clean_value(row.get("Last Protein")),
            "objective": clean_value(row.get("Last Binding Energy")),
            "stage": "paper",
            "hypervolume": clean_value(row.get("BestValue")),
            "llm_reflection": None,
            "llm_label": None,
        }
        trajectory.append(clean_record(item))
    return trajectory


def parse_antigen_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return {item.strip() for item in value.split(",") if item.strip()}


def bo_llm_config(run: CsvRun) -> dict[str, Any]:
    run_dir = run.path.parent
    decisions_log = read_json(run_dir / "llm_decisions.json") or {}
    bo_config = load_bo_config()
    return reference_config(
        method="LDM",
        run=run,
        bo_config=bo_config,
        seed_smiles=first_sequences(run.path, int(bo_config.get("n_init", 20))),
        batch_size=bo_config.get("batch_size", 1),
        init_size=bo_config.get("n_init", 20),
        acquisition=bo_config.get("acq", "ei"),
        acq_budget=decisions_log.get("config_snapshot", {}).get("acq_search_budget"),
        llm_prompt=render_ldm_system_prompt(bo_config),
    )


def llm_baseline_config(run: CsvRun) -> dict[str, Any]:
    runtime = llm_runtime_config()
    return clean_record(
        {
            "method": "LLM Only",
            "seed": run.seed,
            "seed_smiles": [],
            "num_evaluations": run.n_eval,
            "batch_size": 1,
            "init_size": 0,
            "acquisition": None,
            "xi": None,
            "kappa": None,
            "minimize": [True],
            "acq_budget": None,
            "max_pool_size": None,
            "pool_min_size": None,
            "pool_max_size": None,
            "smiles_max_len": 11,
            "objective": "binding_energy",
            "n_objectives": 1,
            "objective_parts": ["binding_energy"],
            "ehvi_n_samples": None,
            "che_alpha": None,
            "gp": {
                "used": False,
                "impl": None,
                "note": "Not used by pure LLM baseline.",
            },
            "ref_point": None,
            "vina": {
                "name": "Absolut",
                "objective": "binding_energy",
                "antigen": run.antigen,
            },
            "reasyn": None,
            "llm": {
                "model": runtime.get("model"),
                "base_url": runtime.get("base_url"),
                "trajectory_dir": str(run.path.parent),
                "prompt_source": LLM_BASELINE_PROMPT_SOURCE,
                "description": "Pure LLM baseline: no BO, no GP, no acquisition, no trust region.",
            },
            "uses_bo": False,
            "uses_gp": False,
            "uses_acquisition": False,
            "uses_trust_region": False,
            "source_path": str(run.path),
            "selection_policy": "lowest final BestValue among available LLM baseline seeds",
            "final_best": run.final_best,
            "best_protein": run.best_protein,
        }
    )


def method_result_from_run(
    run: CsvRun,
    method_name: str,
    stage_name: str,
    n_init: int | None,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    if method_name == "BO+LLM":
        config = bo_llm_config(run)
        llm_decisions = bo_llm_decisions_by_eval(
            run.path.parent,
            results_path=run.path,
            use_deepseek_reflection_plan=bool(args and args.use_deepseek_reflection_plan),
            n_init=n_init,
            max_reflection_points=args.max_reflection_points if args else None,
        )
    elif method_name == "LLM baseline":
        config = llm_baseline_config(run)
        llm_decisions = llm_baseline_decisions_by_eval(run.path.parent)
    else:
        config = clean_record(
            {
                "source_path": str(run.path),
                "antigen": run.antigen,
                "selected_seed": run.seed,
                "selection_policy": "lowest final BestValue among available seeds",
                "final_best": run.final_best,
                "best_protein": run.best_protein,
                "n_eval": run.n_eval,
            }
        )
        llm_decisions = {}
    display_method_name = {
        "BO+LLM": "LDM",
        "LLM baseline": "LLM Only",
    }.get(method_name, method_name)

    return {
        "method": display_method_name,
        "config": config,
        "trajectory": antbo_csv_to_trajectory(
            run.path,
            stage_name=stage_name,
            n_init=n_init,
            llm_decisions_by_eval=llm_decisions,
        ),
    }


def build_antigen_json(
    antigen: str,
    bo_run: CsvRun,
    llm_run: CsvRun | None,
    paper_infos: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    results = [
        method_result_from_run(
            bo_run,
            method_name="BO+LLM",
            stage_name="loop",
            n_init=args.bo_llm_n_init,
            args=args,
        )
    ]

    if llm_run is not None:
        results.append(
            method_result_from_run(
                llm_run,
                method_name="LLM baseline",
                stage_name="llm_baseline",
                n_init=None,
            )
        )
        

    for info in paper_infos:
        bo_config = load_bo_config()
        paper_run = CsvRun(
            method=info["method"],
            antigen=antigen,
            seed=int(info["best_seed"]),
            path=Path(info["source_path"]),
            final_best=float(info["best_final"]),
            best_protein=info["best_protein"],
            n_eval=int(info["max_eval"]),
        )
        paper_result = {
            "method": f"paper:{info['method']}",
            "config": reference_config(
                method=info["method"],
                run=paper_run,
                bo_config=bo_config,
                seed_smiles=first_paper_sequences(
                    Path(info["source_path"]),
                    antigen=antigen,
                    method=info["method"],
                    seed=int(info["best_seed"]),
                    n=int(bo_config.get("n_init", 20)),
                ),
                batch_size=1,
                init_size=int(bo_config.get("n_init", 20)),
                acquisition=None,
                acq_budget=None,
                llm_prompt=None,
            ),
            "trajectory": paper_csv_to_trajectory(
                Path(info["source_path"]),
                antigen=antigen,
                method=info["method"],
                seed=int(info["best_seed"]),
            ),
        }
        results.append(paper_result)

    return clean_record(
        {
            "num_objectives": 1,
            "direction": "minimize",
            "task": f"AntBO CDRH3 binding energy ({antigen})",
            "selected_seed": bo_run.seed,
            "selected_source": str(bo_run.path),
            "selection_metric": "bo_llm_final_best",
            "selection_value": bo_run.final_best,
            "results": results,
            "ref_point": None,
            "labelling_model": "Absolut binding energy oracle",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bo-llm-root", type=Path, default=DEFAULT_BO_LLM_ROOT)
    parser.add_argument("--llm-baseline-root", type=Path, default=DEFAULT_LLM_BASELINE_ROOT)
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--antigens",
        help="Comma-separated antigen IDs or a text file with one antigen per line. Defaults to BO+LLM antigens.",
    )
    parser.add_argument(
        "--bo-llm-n-init",
        type=int,
        default=20,
        help="Number of BO+LLM initial evaluations to mark as stage='init'.",
    )
    parser.add_argument(
        "--use-deepseek-reflection-plan",
        action="store_true",
        help="Call DeepSeek once per BO+LLM run to select sparse reflection points from the full trajectory.",
    )
    parser.add_argument(
        "--max-reflection-points",
        type=int,
        default=None,
        help="Maximum number of DeepSeek/rule-based reflection points per BO+LLM trajectory. Omit for no fixed limit.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    antigen_filter = parse_antigen_filter(args.antigens)

    bo_best = choose_best_runs(discover_runs(args.bo_llm_root, "BO+LLM"))
    llm_best = choose_best_runs(discover_runs(args.llm_baseline_root, "LLM baseline"))
    paper_final_rows = load_paper_final_rows(args.paper_root)

    antigens = sorted(bo_best)
    if antigen_filter is not None:
        antigens = [antigen for antigen in antigens if antigen in antigen_filter]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for antigen in antigens:
        bo_run = bo_best[antigen]
        payload = build_antigen_json(
            antigen=antigen,
            bo_run=bo_run,
            llm_run=llm_best.get(antigen),
            paper_infos=paper_stats_and_best_seed(paper_final_rows, antigen),
            args=args,
        )
        out_path = args.out_dir / f"{antigen}.json"
        indent = None if args.indent == 0 else args.indent
        out_path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
        manifest.append(
            {
                "antigen": antigen,
                "path": str(out_path),
                "bo_llm_seed": bo_run.seed,
                "bo_llm_final_best": bo_run.final_best,
                "llm_baseline_seed": llm_best.get(antigen).seed if antigen in llm_best else None,
                "llm_baseline_final_best": llm_best.get(antigen).final_best if antigen in llm_best else None,
                "n_methods": len(payload["results"]),
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[Saved] {len(manifest)} antigen JSON files under {args.out_dir}")
    print(f"[Saved] {manifest_path}")
    if manifest:
        print()
        print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
