import json
import os
import re
import subprocess
from typing import Any, Optional

import numpy as np


def split_antigen_id(antigen_id: str) -> tuple[str, Optional[str]]:
    parts = antigen_id.split("_", 1)
    pdb_id = parts[0]
    chain_id = parts[1] if len(parts) > 1 else None
    return pdb_id, chain_id


def run_absolut_info(absolut_path: str, command: str, antigen_id: str, timeout_s: int = 30) -> dict[str, Any]:
    exe = os.path.join(os.path.abspath(absolut_path), "src", "bin", "Absolut")
    if not os.path.exists(exe):
        return {
            "command": command,
            "ok": False,
            "error": f"Absolut executable not found at {exe}",
            "stdout": "",
            "stderr": "",
        }

    proc = subprocess.run(
        [exe, command, antigen_id],
        cwd=os.path.abspath(absolut_path),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {
        "command": command,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_key_values(text: str) -> dict[str, str]:
    out = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
        elif "\t" in line:
            key, value = line.split("\t", 1)
            out[key.strip()] = value.strip()
    return out


def extract_feature_lines(text: str) -> dict[str, list[str]]:
    patterns = {
        "forbidden_positions": re.compile("forbidden", re.IGNORECASE),
        "glycans": re.compile("glycan", re.IGNORECASE),
        "hotspots": re.compile("hotspot", re.IGNORECASE),
        "hotspot_core_residues": re.compile("core", re.IGNORECASE),
        "bound_100_residues": re.compile("bound.*100|100.*bound", re.IGNORECASE),
    }
    features = {key: [] for key in patterns}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for key, pattern in patterns.items():
            if pattern.search(line):
                features[key].append(line)
    return features


def collect_absolut_antigen_context(
    bbox_config: dict[str, Any],
    timeout_s: int = 30,
    include_raw: bool = False,
) -> dict[str, Any]:
    antigen_id = bbox_config["antigen"]
    pdb_id, chain_id = split_antigen_id(antigen_id)
    absolut_path = bbox_config["path"]

    info_antigen = run_absolut_info(absolut_path, "info_antigen", antigen_id, timeout_s=timeout_s)
    info_filenames = run_absolut_info(absolut_path, "info_filenames", antigen_id, timeout_s=timeout_s)

    raw_text = "\n".join(
        result.get("stdout", "") for result in [info_antigen, info_filenames] if result.get("stdout")
    )
    context = {
        "antigen_id": antigen_id,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "source": "Absolut",
        "absolut_path": os.path.abspath(absolut_path),
        "commands": {
            "info_antigen": {key: value for key, value in info_antigen.items() if key not in ["stdout", "stderr"]},
            "info_filenames": {key: value for key, value in info_filenames.items() if key not in ["stdout", "stderr"]},
        },
        "parsed_key_values": parse_key_values(raw_text),
        "features": extract_feature_lines(raw_text),
    }
    if include_raw:
        context["raw_outputs"] = {
            "info_antigen_stdout": info_antigen.get("stdout", ""),
            "info_antigen_stderr": info_antigen.get("stderr", ""),
            "info_filenames_stdout": info_filenames.get("stdout", ""),
            "info_filenames_stderr": info_filenames.get("stderr", ""),
        }
    return context


def idx_to_seq(x: np.ndarray, idx_to_aa: dict[int, str]) -> str:
    return "".join(idx_to_aa[int(i)] for i in x)


def build_bo_history_context(optim, f_obj, top_k: int = 10) -> dict[str, Any]:
    if optim is None or len(optim.casmopolitan.fx) == 0:
        return {
            "num_observations": 0,
            "current_best_sequence": None,
            "current_best_value": None,
            "top_observed_sequences": [],
        }

    x = np.asarray(optim.casmopolitan.x)
    y = np.asarray(optim.casmopolitan.fx).reshape(-1)
    order = np.argsort(y)
    idx_to_aa = f_obj.fbox.idx_to_AA
    top = []
    for rank, idx in enumerate(order[:top_k]):
        top.append(
            {
                "rank": int(rank + 1),
                "sequence": idx_to_seq(x[idx], idx_to_aa),
                "value": float(y[idx]),
            }
        )
    best = top[0]
    return {
        "num_observations": int(len(y)),
        "current_best_sequence": best["sequence"],
        "current_best_value": best["value"],
        "top_observed_sequences": top,
    }


def build_llm_prompt_context(
    antigen_context: dict[str, Any],
    bo_history_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "antigen_context": antigen_context,
        "bo_history": bo_history_context,
        "task": {
            "instruction": (
                "Propose weak, interpretable trust-region centers and mutation preferences. "
                "Return JSON only. Do not predict exact binding energy. "
                "Do not invent antigen facts not present in antigen_context."
            ),
            "expected_json_keys": [
                "trust_region_centers",
                "mutation_policy",
                "soft_constraints",
                "antigen_preferences",
            ],
        },
    }


def save_json(obj: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def save_llm_context_snapshot(
    out_dir: str,
    antigen_context: dict[str, Any],
    optim=None,
    f_obj=None,
    itern: Optional[int] = None,
    top_k: int = 10,
) -> str:
    bo_history = build_bo_history_context(optim=optim, f_obj=f_obj, top_k=top_k)
    prompt_context = build_llm_prompt_context(
        antigen_context=antigen_context,
        bo_history_context=bo_history,
    )
    suffix = "init" if itern is None else f"iter_{int(itern):04d}"
    path = os.path.join(out_dir, f"llm_prompt_context_{suffix}.json")
    save_json(prompt_context, path)
    return path
