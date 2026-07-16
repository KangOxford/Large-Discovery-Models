"""JSONL trajectory recorder for tilted case2 searches."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from ldm_tts.trajectory import JsonlTrajectoryRecorder
from strbo_v1.acquisition import hypervolume
from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config


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
        )
        self.trajectory_dir = self._recorder.trajectory_dir
        self.rounds = self._recorder.rounds
        self.llm_call_count = sum(len(record.get("llm_attempts", [])) for record in self.rounds)

    def record_round(self, record: dict[str, Any]) -> None:
        self.llm_call_count += len(record.get("llm_attempts", []))
        self._recorder.append_round(record)

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
    return out
