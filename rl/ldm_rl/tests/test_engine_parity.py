"""Budget-semantics parity between LDMEnv and LDMEngine.

Locks two things against future drift:

1. the ``EnvConfig`` budget fields are a superset of the engine's
   ``LDMEngineConfig`` budget fields, so a new engine stop/evaluation knob
   cannot silently slip past the RL environment;
2. for identical proposal streams and budget configs, the env and the engine
   stop with the same reason and record the same observations (count,
   successful count, and canonical-key order).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateDomainSpec,
    CandidateEvaluator,
    CandidateRejection,
    EvaluationResult,
    LDMTaskSpec,
    ObjectiveSpec,
    RawProposal,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.engine import LDMEngine, LDMEngineConfig
from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
from ldm_tts.engine.run_store import CampaignRuntime

from ldm_rl import EnvConfig, LDMEnv

# Engine budget fields the RL environment must mirror. Kept explicit so a new
# engine knob fails this test until EnvConfig grows the same field.
ENGINE_BUDGET_FIELDS = {
    "target_observations",
    "target_successful_evaluations",
    "max_evaluation_attempts",
    "max_evaluation_attempts_per_round",
    "replace_failed_evaluations",
}


class _BoundedDomain:
    """Admits {"x": float} payloads with x in [0, 10]."""

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        payload = proposal.payload
        if not isinstance(payload, dict) or "x" not in payload:
            return CandidateRejection(
                "missing_x", "payload must be an object with x", proposal.source
            )
        x = payload["x"]
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return CandidateRejection("non_numeric_x", "x must be numeric", proposal.source)
        x = float(x)
        if not 0.0 <= x <= 10.0:
            return CandidateRejection("out_of_bounds", "x must be within [0, 10]", proposal.source)
        return Candidate(
            candidate_id=f"cand-{x}",
            payload={"x": x},
            canonical_key=f"x={x}",
            source=proposal.source,
        )


class _ScaledEvaluator:
    """score = 2x; x == 5 raises to exercise failure wrapping."""

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        x = float(candidate.payload["x"])
        if x == 5.0:
            raise RuntimeError("synthetic evaluator failure")
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"score": 2.0 * x},
            resource_usage={"benchmark_jobs": 1},
        )


def _spec() -> LDMTaskSpec:
    return LDMTaskSpec(
        task="synthetic",
        candidate_domain=CandidateDomainSpec(
            name="bounded scalar", kind="scalar", dimension=1, constraints={"bounds": [0, 10]}
        ),
        objectives=(ObjectiveSpec(name="score", direction="maximize"),),
        response_spaces=(
            ResponseSpaceSpec(
                name="candidate_list",
                output_kind="json",
                schema={"type": "object", "required": ["candidates"]},
                parser="ldm_rl.parsing:parse_candidate_list",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="none", objective_names=("score",), score_direction="maximize", selection_rule=""
        ),
        reservoir=ReservoirSpec(
            name="scalar_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="emit",
                    action_kind="emit_candidate",
                    response_space="candidate_list",
                    produces_candidates=True,
                ),
            ),
            candidate_validator="bounds",
            deduplication_key="x",
            max_size=2,
        ),
        surrogate=SurrogateSpaceSpec(kind="none", representation="none", dimension_policy="none"),
    )


def _action(*xs: float) -> str:
    return json.dumps({"candidates": [{"x": x} for x in xs]})


def test_env_config_mirrors_engine_budget_fields() -> None:
    engine_fields = {field.name for field in dataclasses.fields(LDMEngineConfig)}
    env_fields = {field.name for field in dataclasses.fields(EnvConfig)}
    missing = ENGINE_BUDGET_FIELDS - env_fields
    assert not missing, (
        f"EnvConfig is missing engine budget field(s): {sorted(missing)}; "
        "mirror them so RL episodes honor the same campaign semantics"
    )
    # The engine fields must exist on the engine too (guards against the test
    # itself going stale when the engine is refactored).
    assert ENGINE_BUDGET_FIELDS <= engine_fields


def _run_env(
    config: EnvConfig,
    proposals_per_round: list[list[dict[str, float]]],
) -> tuple[str, list[Any]]:
    env = LDMEnv(
        task_spec=_spec(),
        domain=_BoundedDomain(),
        evaluator=_ScaledEvaluator(),
        config=config,
    )
    env.reset()
    for round_idx, proposals in enumerate(proposals_per_round):
        action = _action(*[float(item["x"]) for item in proposals])
        step = env.step(action)
        if step.done:
            break
    assert env._done, "env did not reach a terminal state within the proposal stream"
    return str(step.info["stop_reason"]), list(env.history)


def _run_engine(
    config: LDMEngineConfig,
    proposals_per_round: list[list[dict[str, float]]],
    tmp_path: Any,
) -> tuple[str, list[Any]]:
    def expand(request: Any) -> ExpansionResult:
        if request.round_idx >= len(proposals_per_round):
            return ExpansionResult(proposals=())
        return ExpansionResult(
            proposals=tuple(
                RawProposal(payload, source="policy", metadata={"round_idx": request.round_idx})
                for payload in proposals_per_round[request.round_idx]
            )
        )

    runtime = CampaignRuntime.open(
        tmp_path / "parity_run",
        task="synthetic",
        task_spec=_spec(),
        config={},
        budget_limits={
            "outer_iterations": config.iterations,
            "external_evaluations": 10_000,
            "successful_evaluations": 10_000,
        },
    )
    engine = LDMEngine(
        task_spec=_spec(),
        expander=CallableReservoirExpander(expand),
        candidate_domain=_BoundedDomain(),
        evaluator=_ScaledEvaluator(),
        runtime=runtime,
    )
    result = engine.run(config)
    return result.stop_reason, list(result.state.observations)


def _assert_same_outcome(
    config: dict[str, Any],
    proposals_per_round: list[list[dict[str, float]]],
    tmp_path: Any,
) -> None:
    env_stop, env_observations = _run_env(EnvConfig(**config), proposals_per_round)
    engine_stop, engine_observations = _run_engine(
        LDMEngineConfig(**config), proposals_per_round, tmp_path
    )
    assert env_stop == engine_stop, (
        f"stop_reason mismatch: env={env_stop!r} engine={engine_stop!r} "
        f"(config={config})"
    )
    assert len(env_observations) == len(engine_observations)
    assert [item.canonical_key for item in env_observations] == [
        item.canonical_key for item in engine_observations
    ]
    assert [item.evaluation.status for item in env_observations] == [
        item.evaluation.status for item in engine_observations
    ]


def test_target_observations_parity(tmp_path: Any) -> None:
    _assert_same_outcome(
        {
            "iterations": 10,
            "reservoir_size": 1,
            "evaluations_per_round": 1,
            "target_observations": 3,
        },
        [
            [{"x": 1.0}],
            [{"x": 3.0}],
            [{"x": 0.5}],
            [{"x": 2.0}],  # should never be reached
        ],
        tmp_path,
    )


def test_target_successful_evaluations_replaces_failed_parity(tmp_path: Any) -> None:
    """Failed evaluations must not consume the successful-result budget."""

    _assert_same_outcome(
        {
            "iterations": 10,
            "reservoir_size": 2,
            "evaluations_per_round": 1,
            "target_successful_evaluations": 2,
            "replace_failed_evaluations": True,
        },
        [
            [{"x": 5.0}, {"x": 1.0}],  # x=5 fails, x=1 succeeds
            [{"x": 3.0}, {"x": 4.0}],  # x=3 reaches the target
        ],
        tmp_path,
    )


def test_max_evaluation_attempts_parity(tmp_path: Any) -> None:
    _assert_same_outcome(
        {
            "iterations": 10,
            "reservoir_size": 1,
            "evaluations_per_round": 1,
            "max_evaluation_attempts": 2,
        },
        [
            [{"x": 1.0}],
            [{"x": 2.0}],
            [{"x": 3.0}],  # attempt budget exhausted before this round
        ],
        tmp_path,
    )


def test_env_config_validation_matches_engine() -> None:
    """The mutual-exclusion and replace_failed guards fail identically."""

    with pytest.raises(ValueError, match="not both"):
        EnvConfig(
            iterations=1,
            target_observations=1,
            target_successful_evaluations=1,
        )
    with pytest.raises(ValueError, match="replace_failed_evaluations requires"):
        EnvConfig(
            iterations=1,
            replace_failed_evaluations=True,
        )
    with pytest.raises(ValueError, match="must be positive"):
        EnvConfig(iterations=1, max_evaluation_attempts_per_round=0)
