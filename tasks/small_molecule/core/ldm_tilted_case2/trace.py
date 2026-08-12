"""JSONL trajectory recorder for tilted case2 searches."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from ldm_tts.engine.run_store import JsonlTrajectoryRecorder
from ldm_tts.data import DataCollectionSink, smallmol_irs_from_round_record
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    ProposalSearchSpec,
    SurrogateSpaceSpec,
)
from tasks.small_molecule.core.acquisition import hypervolume
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config


class TiltedTraceRecorder:
    def __init__(
        self,
        trajectory_dir: str | None,
        cfg: TiltedLDMCase2Config,
        existing_rounds: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg
        self._recorder = JsonlTrajectoryRecorder(
            trajectory_dir,
            config_snapshot=_config_dict(cfg),
            existing_rounds=existing_rounds,
            reset_rounds_file=existing_rounds is None,
        )
        self.trajectory_dir = self._recorder.trajectory_dir
        self.rounds = self._recorder.rounds
        self.llm_call_count = sum(len(record.get("llm_attempts", [])) for record in self.rounds)
        default_collection_dir = (
            self.trajectory_dir / "ldm_data" if self.trajectory_dir is not None else None
        )
        self._data_sink = DataCollectionSink.from_env(default_root=default_collection_dir)

    def record_round(self, record: dict[str, Any]) -> None:
        self.llm_call_count += len(record.get("llm_attempts", []))
        self._recorder.append_round(record)
        if self._data_sink.enabled:
            provenance = {
                "task": "small_molecule",
                "method": self.cfg.method,
                "round_idx": record.get("round_idx"),
                "trajectory_dir": None if self.trajectory_dir is None else str(self.trajectory_dir),
            }
            outcome = {
                "selection_results": record.get("selection_results", {}),
                "drop_counts": record.get("drop_counts", {}),
            }
            for ir in smallmol_irs_from_round_record(record):
                self._data_sink.append(ir, provenance=provenance, outcome=outcome)

    def finalize(
        self,
        history: Sequence[tuple[str, Sequence[float | None]]],
        *,
        early_stop_reason: str | None = None,
    ) -> dict[str, Any]:
        summary = self._summary(history, early_stop_reason)
        if self.trajectory_dir:
            self._recorder.write_json("history.json", _history_json(history))
            self._recorder.write_json("summary.json", summary)
        return summary

    def _summary(self, history, early_stop_reason: str | None) -> dict[str, Any]:
        final_hv = _final_hypervolume(history, self.cfg)
        drop_totals: dict[str, int] = {}
        for round_record in self.rounds:
            for key, value in round_record.get("drop_counts", {}).items():
                drop_totals[key] = drop_totals.get(key, 0) + int(value)
        return {
            "method": self.cfg.method,
            "llm_call_count": self.llm_call_count,
            "round_count": len(self.rounds),
            "history_size": len(history),
            "final_hypervolume": final_hv,
            "drop_counts": drop_totals,
            "early_stop_reason": early_stop_reason,
            "q0_entropy": _last_metric(self.rounds, "q0_entropy"),
            "prob_effective_sample_size": _last_metric(self.rounds, "prob_effective_sample_size"),
        }


def _final_hypervolume(history, cfg: TiltedLDMCase2Config) -> float:
    points = [scores for _smiles, scores in history if len(scores) == 2 and None not in scores]
    return float(hypervolume(points, cfg.ref_point, minimize=cfg.minimize)) if points else 0.0


def _history_json(history) -> list[dict[str, Any]]:
    return [{"smiles": smiles, "scores": list(scores)} for smiles, scores in history]


def _last_metric(rounds: list[dict[str, Any]], key: str) -> Any:
    return rounds[-1].get(key) if rounds else None


def _config_dict(cfg: TiltedLDMCase2Config) -> dict[str, Any]:
    out = dict(cfg.__dict__)
    out["gp_config"] = dict(cfg.gp_config.__dict__)
    out["ldm_task_spec"] = _ldm_task_spec(cfg).to_dict()
    return out


def _ldm_task_spec(cfg: TiltedLDMCase2Config) -> LDMTaskSpec:
    gp_impl = str(getattr(cfg.gp_config, "impl", ""))
    gp_feature_dimension = (
        int(getattr(cfg.gp_config, "fp_n_bits", 0) or 0)
        if gp_impl == "fingerprint+tanimoto"
        else None
    )
    if cfg.method in {"m1_stratified_direct_llm_only", "m1_llm_one_step"}:
        acquisition = AcquisitionSpec(
            name="llm_order",
            objective_names=("vina", "activity"),
            score_direction="rank",
            selection_rule="evaluate candidates in LLM/reservoir order",
            parameters={"batch_size": int(cfg.batch_size)},
        )
    else:
        acquisition = AcquisitionSpec(
            name=str(cfg.acquisition),
            objective_names=("vina", "activity"),
            score_direction="sample",
            selection_rule=(
                "sample candidates from q0 base mass tilted by the robust-z shared "
                f"{cfg.acquisition} acquisition score"
            ),
            parameters={
                "alpha_base_measure": float(cfg.alpha_base_measure),
                "eta_acquisition_tilt": float(cfg.eta_ehvi_tilt),
                "acquisition_weights": list(cfg.acquisition_weights),
                "ehvi_n_samples": int(cfg.ehvi_n_samples),
                "batch_size": int(cfg.batch_size),
            },
        )
    direct_only = cfg.method in {"m1_stratified_direct_llm_only", "m1_llm_one_step"}
    if direct_only:
        surrogate = SurrogateSpaceSpec(
            kind="none",
            representation="not used by direct LLM ordering",
            dimension_policy="none",
        )
    elif gp_impl == "fingerprint+tanimoto":
        surrogate = SurrogateSpaceSpec(
            kind="vector",
            representation="fixed-length molecular fingerprint",
            dimension_policy="fixed",
            dimension=int(cfg.gp_config.fp_n_bits),
            encoder="tasks.small_molecule.core.gp.fingerprint_features",
            version=f"molecular_fingerprint_{int(cfg.gp_config.fp_n_bits)}_v1",
        )
    else:
        surrogate = SurrogateSpaceSpec(
            kind="kernel",
            representation="SMILES subsequence string kernel",
            dimension_policy="implicit",
            encoder="tasks.small_molecule.core.gp.SMILESStringKernel",
            version="smiles_strkernel_v1",
        )
    return LDMTaskSpec(
        task="small_molecule",
        candidate_domain=CandidateDomainSpec(
            name="smiles",
            kind="string",
            dimension=None,
            representation="canonical SMILES string",
            constraints={"max_smiles_len": int(cfg.smiles_max_len)},
            metadata={
                "gp_kernel": gp_impl,
                "gp_feature_dimension": gp_feature_dimension,
                "max_candidates_per_round": int(cfg.max_candidates_per_round),
            },
        ),
        objectives=(
            ObjectiveSpec(
                name="vina",
                direction="minimize",
                description="AutoDock Vina binding score; lower is better.",
            ),
            ObjectiveSpec(
                name="activity",
                direction="maximize",
                description="KRAS G12D activity model score; higher is better.",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="direct_smiles",
                output_kind="json",
                parser="tasks.small_molecule.core.ldm_tilted_case2.schemas.parse_m1_direct_smiles",
                description="LLM emits direct candidate SMILES without objective scores.",
            ),
            ResponseSpaceSpec(
                name="seed_plan",
                output_kind="json",
                parser="tasks.small_molecule.core.ldm_tilted_case2.schemas.parse_seed_plan",
                description="LLM emits seed SMILES and per-seed analogue budgets.",
            ),
        ),
        acquisition=acquisition,
        reservoir=ReservoirSpec(
            name="molecular_candidate_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_smiles_generation",
                    action_kind="emit_candidate",
                    response_space="direct_smiles",
                    produces_candidates=True,
                    description="Emit valid molecular candidates as SMILES.",
                ),
                ReservoirExpansionSpec(
                    name="seeded_analogue_generation",
                    action_kind="configure_generator",
                    response_space="seed_plan",
                    produces_candidates=True,
                    description="Choose molecular seeds and budgets for analogue generation.",
                ),
            ),
            candidate_validator="SMILES parse, canonicalization, and molecular constraint checks",
            deduplication_key="canonical SMILES",
            max_size=int(cfg.max_candidates_per_round),
            metadata={"method": cfg.method},
        ),
        surrogate=surrogate,
        proposal_search=ProposalSearchSpec(
            name="single_turn",
            evaluation_policy="outer_loop_acquisition_selection",
            parameters={"proposals_per_round": "task_configured_candidate_reservoir"},
        ),
        metadata={
            "method": cfg.method,
            "init_strategy": cfg.init_strategy,
            "budget": int(cfg.budget),
            "init_size": int(cfg.init_size),
        },
    )
