"""Six LLM-emitted block dataclasses for the three-stage BO loop.

Each block is an independent JSON object the LLM may emit at most once
per call. Block types are partitioned by stage:

* **Stage A1 — Actions** (pool management): ``propose``, ``reject``,
  ``analog``, ``noop``.  The LLM decides how to change the pool.
* **Stage A2 — Review analogs** (conditional): ``review_analogs``.
  Called synchronously when an ``analog`` action produced non-empty
  results.  The LLM decides which generated analogues to keep.
* **Stage B — Review suggestions** (BO review): ``review_bo``.
  The LLM reviews BO picks and may override or skip.

:data:`PHASE_A_ACTIONS_ALLOWED`, :data:`PHASE_A_REVIEW_ANALOGS_ALLOWED`,
and :data:`PHASE_B_SUGGESTIONS_ALLOWED` are the authoritative
stage-allow sets; :func:`validate_blocks_phase` in ``parser.py`` uses
them to reject blocks that appear in the wrong stage.

Every block has a ``type`` field, a ``rationale`` (free-text string
truncated by the JSON schema), and payload fields. :meth:`to_dict` on
each block returns a JSON-serializable dict with ``type`` always first,
suitable for trajectory serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

AnalogueVerdict = Literal["keep", "reject", "rescore_with_different_params"]
RejectReason = Literal[
    "too_similar_to_history",
    "likely_toxic",
    "synthetically_infeasible",
    "out_of_scope_pharmacophore",
    "no_signal_for_target",
]
GeneratorHint = Literal["conservative", "aggressive", "scaffold_hop"]


# ---------------------------------------------------------------------------
# Phase B block
# ---------------------------------------------------------------------------


@dataclass
class ReviewBOBlock:
    """Phase B: review each BO suggestion.

    Per suggestion verdicts:
        "ok"                  -> use BO's pick, score it
        "override:<SMILES>"   -> replace BO's pick with <SMILES>, score it
        "skip"                -> don't score this slot (batch may be smaller)
    """

    type: Literal["review_bo"] = "review_bo"
    rationale: str = ""
    decisions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "review_bo",
            "rationale": self.rationale,
            "decisions": dict(self.decisions),
        }


# ---------------------------------------------------------------------------
# Phase A blocks
# ---------------------------------------------------------------------------


@dataclass
class ProposeBlock:
    """Phase A: inject new SMILES into the pool.

    The SMILES are added to the pool right after Phase A completes; they
    are *not* auto-evaluated this round (BO will see them next round).
    """

    type: Literal["propose"] = "propose"
    rationale: str = ""
    smiles: List[str] = field(default_factory=list)
    rationale_per_mol: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "propose",
            "rationale": self.rationale,
            "smiles": list(self.smiles),
            "rationale_per_mol": dict(self.rationale_per_mol),
        }


@dataclass
class RejectBlock:
    """Phase A: drop SMILES from the pool.

    Targets must currently live in the pool; out-of-pool targets raise
    SemanticError.
    """

    type: Literal["reject"] = "reject"
    rationale: str = ""
    targets: List[str] = field(default_factory=list)
    reason: RejectReason = "too_similar_to_history"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "reject",
            "rationale": self.rationale,
            "targets": list(self.targets),
            "reason": self.reason,
        }


@dataclass
class AnalogBlock:
    """Actions stage: trigger ReaSyn generation from the listed seeds.

    The resulting analogues are reviewed synchronously via a
    ``review_analogs`` block before entering the pool.  ``generator_hint``
    maps to a ReaSyn config preset; ``reasyn_config_override`` lets the
    LLM tweak individual ReaSyn parameters.
    """

    type: Literal["analog"] = "analog"
    rationale: str = ""
    seeds: List[str] = field(default_factory=list)
    generator_hint: Optional[GeneratorHint] = None
    n_per_seed: int = 5
    reasyn_config_override: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "analog",
            "rationale": self.rationale,
            "seeds": list(self.seeds),
            "generator_hint": self.generator_hint,
            "n_per_seed": self.n_per_seed,
            "reasyn_config_override": (
                dict(self.reasyn_config_override)
                if self.reasyn_config_override is not None
                else None
            ),
        }


@dataclass
class ReviewAnalogsBlock:
    """Review-analogs stage: review generated ReaSyn analogues.

    Called synchronously when an ``analog`` action produced non-empty
    results.  For each analogue the LLM decides ``keep`` (add to pool)
    or ``reject`` (drop).
    """

    type: Literal["review_analogs"] = "review_analogs"
    rationale: str = ""
    decisions: Dict[str, AnalogueVerdict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "review_analogs",
            "rationale": self.rationale,
            "decisions": dict(self.decisions),
        }


@dataclass
class NoopBlock:
    """Phase A: explicit "do nothing" declaration."""

    type: Literal["noop"] = "noop"
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "noop", "rationale": self.rationale}


# ---------------------------------------------------------------------------
# Union and phase allow sets
# ---------------------------------------------------------------------------


LLMBlock = Union[
    ReviewBOBlock,
    ProposeBlock,
    RejectBlock,
    AnalogBlock,
    ReviewAnalogsBlock,
    NoopBlock,
]


# --- Stage-level allow sets ---

PHASE_A_ACTIONS_ALLOWED: Tuple[str, ...] = (
    "propose",
    "reject",
    "analog",
    "noop",
)
PHASE_A_REVIEW_ANALOGS_ALLOWED: Tuple[str, ...] = ("review_analogs",)
PHASE_B_SUGGESTIONS_ALLOWED: Tuple[str, ...] = ("review_bo",)


_BLOCK_REGISTRY: Dict[str, type] = {
    "review_bo": ReviewBOBlock,
    "propose": ProposeBlock,
    "reject": RejectBlock,
    "analog": AnalogBlock,
    "review_analogs": ReviewAnalogsBlock,
    "noop": NoopBlock,
}


def block_from_dict(data: Dict[str, Any]) -> LLMBlock:
    """Construct a block from a parsed-JSON dict.

    Looks up ``data["type"]`` in :data:`_BLOCK_REGISTRY` and dispatches.
    Raises :class:`ValueError` for unknown ``type`` values; missing
    ``type`` also raises.
    """
    if not isinstance(data, dict):
        raise ValueError(f"block payload must be a dict, got {type(data).__name__}")
    block_type = data.get("type")
    if not isinstance(block_type, str):
        raise ValueError(f"block missing 'type' field or type is not str: {data!r}")
    cls = _BLOCK_REGISTRY.get(block_type)
    if cls is None:
        raise ValueError(
            f"unknown block type {block_type!r}; expected one of "
            f"{sorted(_BLOCK_REGISTRY)}"
        )
    kwargs: Dict[str, Any] = {}
    for f in cls.__dataclass_fields__.values():        # type: ignore[attr-defined]
        if f.name == "type":
            continue
        if f.name in data:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


__all__ = [
    "AnalogueVerdict",
    "RejectReason",
    "GeneratorHint",
    "ReviewBOBlock",
    "ProposeBlock",
    "RejectBlock",
    "AnalogBlock",
    "ReviewAnalogsBlock",
    "NoopBlock",
    "LLMBlock",
    "PHASE_A_ACTIONS_ALLOWED",
    "PHASE_A_REVIEW_ANALOGS_ALLOWED",
    "PHASE_B_SUGGESTIONS_ALLOWED",
    "block_from_dict",
]
