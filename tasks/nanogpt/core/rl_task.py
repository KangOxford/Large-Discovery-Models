"""Task declaration, action parser and candidate domain for nanoGPT RL.

An RL *instance* is a bounded tuning problem: which knobs the policy may set
(``free_knobs``), what the rest are held at (``pinned``), and the wall-clock
budget the trainer gets (``time_budget``). One instance drives one episode.

Three things live here because they must agree with each other or the policy is
acting in a different space from the one being measured:

* :func:`describe_rl_task` -- the :class:`LDMTaskSpec` the environment and the
  prompt renderer read;
* :func:`parse_knob_proposals` -- the declared response-space parser;
* :class:`NanogptKnobDomain` -- admission, which is where a proposal becomes a
  candidate with a stable identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateDomainSpec,
    CandidateRejection,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    RawProposal,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)

from tasks.nanogpt.core import rl_encoder, rl_knobs as knobs_mod

OBJECTIVE_NAME = "val_bpb"

#: Keys a proposal may never contain. A proposal states an *action*, never its
#: own outcome: letting the policy emit a score invites it to assert a good one
#: and lets any downstream consumer accidentally trust it.
BANNED_KEYS = frozenset(
    {
        "val_bpb",
        "score",
        "reward",
        "objective_score",
        "acquisition_score",
        "uncertainty",
        "proxy_value",
        "predicted_val_bpb",
    }
)

PROPOSALS_KEY = "proposals"
PARSER_PATH = "tasks.nanogpt.core.rl_task:parse_knob_proposals"


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnobInstance:
    """One tuning problem: free knobs, pinned values, wall-clock budget."""

    free_knobs: tuple[str, ...]
    pinned: dict[str, Any] = field(default_factory=dict)
    time_budget: float = knobs_mod.DEFAULT_TIME_BUDGET

    def __post_init__(self) -> None:
        schema = knobs_mod.load_schema()
        unknown = [name for name in self.free_knobs if name not in schema]
        if unknown:
            raise ValueError(f"unknown free knob(s): {unknown}")
        if not self.free_knobs:
            raise ValueError("instance must have at least one free knob")
        overlap = sorted(set(self.free_knobs) & set(self.pinned))
        if overlap:
            raise ValueError(f"knob(s) both free and pinned: {overlap}")
        if self.time_budget <= 0:
            raise ValueError("time_budget must be positive")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "KnobInstance":
        """Build from an episode's ``real_kwargs`` (all keys optional)."""

        free = kwargs.get("free_knobs") or list(knobs_mod.knob_names())
        if isinstance(free, str):
            free = [item.strip() for item in free.split(",") if item.strip()]
        pinned = dict(kwargs.get("pinned") or {})
        # Anything not free and not explicitly pinned is held at its default,
        # and shown as such: the task is ill-posed if the policy cannot see the
        # values it is tuning around (DEVICE_BATCH_SIZE depends on DEPTH).
        for name in knobs_mod.knob_names():
            if name not in free and name not in pinned:
                pinned[name] = knobs_mod.DEFAULTS[name]
        return cls(
            free_knobs=tuple(free),
            pinned=pinned,
            time_budget=float(kwargs.get("time_budget", knobs_mod.DEFAULT_TIME_BUDGET)),
        )

    def full_config(self, proposed: dict[str, Any]) -> dict[str, Any]:
        return knobs_mod.with_defaults(self.free_knobs, proposed, self.pinned)

    def context(self) -> dict[str, Any]:
        """Episode context rendered into the reset prompt."""

        schema = knobs_mod.load_schema()
        return {
            "time_budget_seconds": self.time_budget,
            "free_knobs": {name: schema[name] for name in self.free_knobs},
            "held_fixed": dict(self.pinned),
            "note": (
                "Only the free knobs are yours to set. The held-fixed values are "
                "given because the free knobs cannot be chosen without them "
                "(e.g. DEVICE_BATCH_SIZE must divide TOTAL_BATCH_SIZE/2048, and "
                "the useful model size depends on DEPTH). Your score is the "
                "measured val_bpb of a real training run, not an estimate."
            ),
        }


# ---------------------------------------------------------------------------
# Response-space parser
# ---------------------------------------------------------------------------


def _unwrap_single_fence(text: str) -> str:
    """Remove one surrounding ``````` fence, if the whole text is one."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    newline = stripped.find("\n")
    if newline < 0 or not stripped.endswith("```"):
        return stripped
    return stripped[newline + 1 : -3].strip()


def parse_knob_proposals(text: str, *, expected_count: int = 1) -> list[dict[str, Any]]:
    """Parse the policy action into a list of knob-value payloads.

    **The entire response must be exactly one JSON object** (optionally wrapped
    in a single fenced block). This is deliberately stricter than
    ``ldm_tts.transport.parsing.load_json_object``, which falls back to
    "everything between the first ``{`` and the last ``}``".

    That fallback is a measured reward hole for this task. With it, a policy
    that reasons at length and never commits still gets scored -- on the last
    draft dictionary inside its unfinished reasoning. Measured over a 32-step
    GRPO run on this task: committed answers fell from 26/512 to 3/512 while
    36% of the "answers" being scored were verbatim restatements of the
    starting config buried in truncated text. Requiring the response to *start*
    with ``{`` is what closes it: prose-then-dict no longer parses, so failing
    to commit costs the round instead of paying out.
    """

    payload_text = _unwrap_single_fence(text or "")
    if not payload_text.startswith("{") or not payload_text.endswith("}"):
        raise ValueError(
            "response must be exactly one JSON object and nothing else "
            "(no reasoning before or after it)"
        )
    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"response must be a JSON object, got {type(data).__name__}")

    _reject_banned(data)

    rows = data.get(PROPOSALS_KEY)
    if not isinstance(rows, list):
        raise ValueError(f"{PROPOSALS_KEY!r} must be a list of knob objects")
    if not rows:
        raise ValueError(f"{PROPOSALS_KEY!r} must not be empty")
    if len(rows) != expected_count:
        raise ValueError(
            f"{PROPOSALS_KEY!r} must contain exactly {expected_count} "
            f"proposal(s), got {len(rows)}"
        )
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{PROPOSALS_KEY!r} entries must be objects")
        out.append({str(key).strip().upper(): value for key, value in row.items()})
    return out


def _reject_banned(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in BANNED_KEYS:
                raise ValueError(f"proposal may not include {key!r}")
            _reject_banned(item)
    elif isinstance(value, list):
        for item in value:
            _reject_banned(item)


# ---------------------------------------------------------------------------
# Candidate domain
# ---------------------------------------------------------------------------


class NanogptKnobDomain:
    """Admit knob-dict proposals for one instance.

    Rejection is graded so the step observation tells the policy *which* rule
    it broke: a missing key, an out-of-range value and a config the trainer
    itself would refuse are three different mistakes.
    """

    def __init__(self, instance: KnobInstance) -> None:
        self.instance = instance
        self._schema = knobs_mod.load_schema()

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        payload = proposal.payload
        if not isinstance(payload, dict):
            return CandidateRejection(
                "invalid_payload",
                "proposal payload must be an object of knob values",
                proposal.source,
            )

        missing = [name for name in self.instance.free_knobs if name not in payload]
        if missing:
            return CandidateRejection(
                "missing_knobs",
                f"proposal does not set free knob(s): {sorted(missing)}",
                proposal.source,
            )
        extra = [
            name
            for name in payload
            if name not in self.instance.free_knobs and name in self._schema
        ]
        if extra:
            return CandidateRejection(
                "not_free",
                f"knob(s) are held fixed for this instance and may not be set: "
                f"{sorted(extra)}",
                proposal.source,
            )

        config = self.instance.full_config(payload)
        try:
            clean = knobs_mod.validate(config, self._schema)
        except knobs_mod.ProposalError as exc:
            return CandidateRejection("out_of_domain", str(exc), proposal.source)
        try:
            knobs_mod.check_hard_constraints(clean)
        except knobs_mod.ProposalError as exc:
            return CandidateRejection("rejected_by_trainer", str(exc), proposal.source)

        key = knobs_mod.canonical_key(clean, self.instance.time_budget)
        return Candidate(
            candidate_id="cfg-" + _short_hash(key),
            payload={
                "config": clean,
                "free_knobs": list(self.instance.free_knobs),
                "time_budget": self.instance.time_budget,
                "summary": knobs_mod.describe_config(clean),
            },
            canonical_key=key,
            source=proposal.source,
            metadata=dict(proposal.metadata),
        )


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------


def _response_schema(instance: KnobInstance, expected_count: int) -> dict[str, Any]:
    schema = knobs_mod.load_schema()
    return {
        "type": "object",
        "required": [PROPOSALS_KEY],
        "additionalProperties": False,
        "properties": {
            PROPOSALS_KEY: {
                "type": "array",
                "minItems": expected_count,
                "maxItems": expected_count,
                "items": {
                    "type": "object",
                    "required": list(instance.free_knobs),
                    "additionalProperties": False,
                    "properties": {
                        name: dict(schema[name]) for name in instance.free_knobs
                    },
                },
            }
        },
        "example": {
            PROPOSALS_KEY: [
                {
                    name: knobs_mod.DEFAULTS[name]
                    for name in instance.free_knobs
                }
                for _ in range(expected_count)
            ]
        },
    }


def describe_rl_task(
    instance: KnobInstance,
    *,
    reservoir_size: int = 1,
    surrogate: bool = True,
) -> LDMTaskSpec:
    """The task declaration the RL environment validates itself against."""

    response_space = ResponseSpaceSpec(
        name="knob_proposals",
        output_kind="json_object",
        schema=_response_schema(instance, reservoir_size),
        parser=PARSER_PATH,
        description=(
            f"Emit exactly one JSON object with a {PROPOSALS_KEY!r} array of "
            f"{reservoir_size} distinct hyperparameter setting(s). Set every "
            "free knob explicitly. Emit no text before or after the object."
        ),
        metadata={"banned_score_keys": sorted(BANNED_KEYS)},
    )

    return LDMTaskSpec(
        task="nanogpt",
        candidate_domain=CandidateDomainSpec(
            name="nanogpt_train_knobs",
            kind="mixed_hyperparameters",
            dimension=len(instance.free_knobs),
            representation=(
                "top-level hyperparameter assignments of nanoGPT's train.py "
                f"(schema {knobs_mod.schema_version()})"
            ),
            constraints={
                "divisibility": (
                    "TOTAL_BATCH_SIZE must be divisible by "
                    f"DEVICE_BATCH_SIZE * {knobs_mod.MAX_SEQ_LEN}; the trainer "
                    "asserts this at startup. Note this makes "
                    "DEVICE_BATCH_SIZE=96 unusable with every available "
                    "TOTAL_BATCH_SIZE."
                ),
                "wall_clock_budget_seconds": instance.time_budget,
                "max_seq_len": knobs_mod.MAX_SEQ_LEN,
                "vocab_size": knobs_mod.VOCAB_SIZE,
                "gpu": "one H100 80GB; a config that exceeds it fails as OOM",
            },
        ),
        objectives=(
            ObjectiveSpec(
                name=OBJECTIVE_NAME,
                direction="minimize",
                description=(
                    "validation bits-per-byte after training for exactly "
                    f"{instance.time_budget:.0f}s of wall clock on one H100. "
                    "Because the budget is wall clock and not a token count, a "
                    "larger model is not automatically better: it costs more "
                    "FLOPs per token, so it completes fewer optimizer steps and "
                    "sees fewer tokens. Measured by a real training run."
                ),
            ),
        ),
        response_spaces=(response_space,),
        acquisition=AcquisitionSpec(
            name="gp_ucb" if surrogate else "reservoir_order",
            objective_names=(OBJECTIVE_NAME,),
            score_direction="minimize" if surrogate else "sample",
            selection_rule=(
                "fit an RBF GP on the measured history and evaluate the "
                "highest-UCB proposal first"
                if surrogate
                else "evaluate in reservoir order"
            ),
        ),
        reservoir=ReservoirSpec(
            name="nanogpt_knob_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="propose_knobs",
                    action_kind="emit_candidate",
                    response_space=response_space.name,
                    produces_candidates=True,
                    description="propose hyperparameter settings to measure",
                ),
            ),
            candidate_validator="tasks.nanogpt.core.rl_knobs:admit",
            deduplication_key="tasks.nanogpt.core.rl_knobs:canonical_key",
            max_size=max(1, reservoir_size),
        ),
        surrogate=(
            rl_encoder.surrogate_spec()
            if surrogate
            else SurrogateSpaceSpec(
                kind="none",
                representation="no surrogate (reservoir-order evaluation)",
                dimension_policy="none",
            )
        ),
        proposal_search=ProposalSearchSpec(
            name="single_turn", breadth=reservoir_size, depth=1
        ),
        metadata={
            "schema_version": knobs_mod.schema_version(),
            "free_knobs": list(instance.free_knobs),
            "reward_note": (
                "objective values are measured, never estimated; there is no "
                "analytic val_bpb surrogate in this task"
            ),
        },
    )


__all__ = [
    "BANNED_KEYS",
    "OBJECTIVE_NAME",
    "PARSER_PATH",
    "PROPOSALS_KEY",
    "KnobInstance",
    "NanogptKnobDomain",
    "describe_rl_task",
    "parse_knob_proposals",
]
