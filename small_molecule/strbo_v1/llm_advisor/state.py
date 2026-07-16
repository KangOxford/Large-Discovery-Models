"""Round-level read-only data carriers for the LLM advisor.

These are the leaf types referenced by the larger round-state classes
in :mod:`strbo_v1.llm_advisor.round_state`. Keeping them separate
makes the round states easier to read (less noise) and lets the
``__init__`` re-exports stay flat.

* :class:`GPSummary` — minimal GP tag (just ``n_train``; the rich GP
  metadata was dropped per the user's design review).
* :class:`PickRecord` — one BO suggestion, the unit LLM reviews in
  Stage B. ``mu`` / ``sigma`` are ``float`` for n_obj=1 and
  ``list[float]`` (length n_obj) for n_obj>=2.
* :class:`AnalogueRecord` — one ReaSyn analogue awaiting LLM review
  in Stage A2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass(frozen=True)
class GPSummary:
    """Minimal GP tag kept for trajectory / debug.

    The full per-objective GP means / stds were dropped from the LLM-
    facing state (the user deemed them useless for decision making).
    ``n_train`` is just ``len(history)`` at write time; we keep this
    type so the trajectory can serialize it without further plumbing.
    """

    n_train: int = 0

    def to_dict(self) -> dict:
        return {"n_train": self.n_train}


# A score value is a single float for n_obj==1 and a list of floats
# (length n_obj) for n_obj>=2. ``None`` means "scorer failed" and is
# dropped from GP fit / acquisition.
ScoreValue = Union[float, List[float]]


@dataclass(frozen=True)
class PickRecord:
    """One BO suggestion (= one GP top-k entry).

    ``mu`` and ``sigma`` are ``float`` for n_obj==1 and
    ``list[float]`` (length n_obj) for n_obj>=2. The LDM orchestrator
    fits one GP per objective when n_obj>=2, so the GP posterior is
    per-objective.
    """

    smiles: str = ""
    acq_value: float = 0.0
    mu: ScoreValue = 0.0
    sigma: ScoreValue = 0.0

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "acq_value": self.acq_value,
            "mu": list(self.mu) if isinstance(self.mu, (list, tuple)) else self.mu,
            "sigma": list(self.sigma) if isinstance(self.sigma, (list, tuple)) else self.sigma,
        }


@dataclass(frozen=True)
class AnalogueRecord:
    """A ReaSyn analogue awaiting LLM review in Stage A2.

    All fields are produced by ReaSyn except ``seed_smiles`` (which the
    orchestrator tacks on to link the analogue back to the LLM's
    ``analog`` block seed).
    """

    seed_smiles: str = ""
    analogue_smiles: str = ""
    reasyn_score: Optional[float] = None
    synthesis: Optional[str] = None
    num_steps: Optional[int] = None
    scf_sim: Optional[float] = None
    pharm2d_sim: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "seed_smiles": self.seed_smiles,
            "analogue_smiles": self.analogue_smiles,
            "reasyn_score": self.reasyn_score,
            "synthesis": self.synthesis,
            "num_steps": self.num_steps,
            "scf_sim": self.scf_sim,
            "pharm2d_sim": self.pharm2d_sim,
        }


__all__ = ["GPSummary", "PickRecord", "AnalogueRecord", "ScoreValue"]
