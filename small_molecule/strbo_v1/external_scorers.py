"""Structured external scorer interfaces."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from strbo_v1.external_common import item, normalized_request, ok_response, smiles_from
from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH
from strbo_v1.objective_nn import NNScorer, NNScorerConfig
from strbo_v1.objective_vina import VinaScorer, VinaScorerConfig


def score_vina(
    smiles: Optional[Sequence[str]] = None,
    *,
    request: Optional[dict[str, Any]] = None,
    config: Optional[VinaScorerConfig] = None,
    scorer_factory: Callable[[VinaScorerConfig], Any] = VinaScorer,
    vina_bin: Optional[str] = None,
    cache_dir: Optional[str] = None,
    vina_cache_dir: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> dict[str, Any]:
    """Score SMILES with Vina and return structured per-item results."""
    req = normalized_request(request)
    smiles_list = smiles_from(smiles, req)
    cfg = config or build_vina_config(
        req,
        vina_bin=vina_bin,
        cache_dir=cache_dir or vina_cache_dir,
        max_workers=max_workers,
    )
    scorer = scorer_factory(cfg)
    if hasattr(scorer, "_dock_smiles"):
        items = _vina_items_from_docking(smiles_list, scorer._dock_smiles(smiles_list))
    else:
        items = _score_items_from_values(smiles_list, scorer(smiles_list))
    return ok_response(items)


def score_nn(
    smiles: Optional[Sequence[str]] = None,
    *,
    request: Optional[dict[str, Any]] = None,
    config: Optional[NNScorerConfig] = None,
    scorer_factory: Callable[[NNScorerConfig], Any] = NNScorer,
    model_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> dict[str, Any]:
    """Score SMILES with the NN activity model and return structured results."""
    req = normalized_request(request)
    smiles_list = smiles_from(smiles, req)
    cfg = config or build_nn_config(
        req, model_path=model_path, metadata_path=metadata_path
    )
    return ok_response(_score_items_from_values(smiles_list, scorer_factory(cfg)(smiles_list)))


def build_vina_config(
    request: Optional[dict[str, Any]] = None,
    *,
    vina_bin: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> VinaScorerConfig:
    req = normalized_request(request)
    return VinaScorerConfig(
        pdb_id=str(req.get("vina_pdb_id", req.get("pdb_id", "8UN5"))),
        chain_id=str(req.get("vina_chain_id", req.get("chain_id", "A"))),
        ligand_resname=req.get("vina_ligand_resname", req.get("ligand_resname")),
        cache_dir=Path(cache_dir or req.get("cache_dir", "docking_work")),
        vina_bin=vina_bin,
        exhaustiveness=int(req.get("vina_exhaustiveness", 4)),
        n_poses=int(req.get("vina_n_poses", 3)),
        seed=int(req.get("vina_seed", 42)),
        max_workers=int(max_workers if max_workers is not None else 1),
        use_cache=not bool(req.get("vina_no_cache", False)),
        allow_debug_receptor=bool(req.get("vina_allow_debug_receptor", False)),
    )


def build_nn_config(
    request: Optional[dict[str, Any]] = None,
    *,
    model_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> NNScorerConfig:
    req = normalized_request(request)
    return NNScorerConfig(
        model_path=str(model_path or DEFAULT_NN_MODEL_PATH),
        metadata_path=str(metadata_path or ""),
        on_error=str(req.get("nn_on_error", "all_nan")),
    )


def _vina_items_from_docking(
    smiles: list[str], results: Sequence[Any]
) -> list[dict[str, Any]]:
    items = []
    for smi, result in zip(smiles, results):
        status = str(getattr(result, "status", "unknown"))
        score = getattr(result, "score", None)
        ok = status == "ok" and score is not None and math.isfinite(float(score))
        error = None if ok else str(getattr(result, "message", status) or status)
        items.append(item(
            smi,
            ok=ok,
            value=float(score) if ok else None,
            error=error,
            details={
                "status": status,
                "canonical_smiles": getattr(result, "canonical_smiles", smi),
                "pose_ref": getattr(result, "pose_ref", None),
                "cached": bool(getattr(result, "cached", False)),
            },
        ))
    return items


def _score_items_from_values(
    smiles: list[str], values: Sequence[Any]
) -> list[dict[str, Any]]:
    items = []
    for smi, value in zip(smiles, values):
        finite = _is_finite(value)
        items.append(item(
            smi,
            ok=finite,
            value=float(value) if finite else None,
            error=None if finite else "score is non-finite",
            details={"raw_value": value},
        ))
    return items


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
__all__ = [
    "build_nn_config",
    "build_vina_config",
    "score_nn",
    "score_vina",
]
