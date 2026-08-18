"""Environment semantics tests on synthetic task adapters.

These tests keep the environment task-neutral: the domain adapter and
evaluator are tiny local stubs, and the response-space parser is the shared
``parse_candidate_list`` declared the same way a real task declares it.
"""

from __future__ import annotations

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


class _BoundedDomain:
    """Admits {"x": float} payloads with x in [0, 10]."""

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        payload = proposal.payload
        if not isinstance(payload, dict) or "x" not in payload:
            return CandidateRejection("missing_x", "payload must be an object with x", proposal.source)
        x = payload["x"]
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return CandidateRejection("non_numeric_x", "x must be numeric", proposal.source)
        x = float(x)
        if not 0.0 <= x <= 10.0:
            return CandidateRejection("out_of_bounds", "x must be within [0, 10]", proposal.source)
        key = f"x={x}"
        return Candidate(
            candidate_id=f"cand-{x}",
            payload={"x": x},
            canonical_key=key,
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


def _spec(objectives: tuple[ObjectiveSpec, ...] | None = None) -> LDMTaskSpec:
    objective = objectives or (ObjectiveSpec(name="score", direction="maximize"),)
    return LDMTaskSpec(
        task="synthetic",
        candidate_domain=CandidateDomainSpec(
            name="bounded scalar", kind="scalar", dimension=1, constraints={"bounds": [0, 10]}
        ),
        objectives=objective,
        response_spaces=(
            ResponseSpaceSpec(
                name="candidate_list",
                output_kind="json",
                schema={"type": "object", "required": ["candidates"]},
                parser="ldm_rl.parsing:parse_candidate_list",
            ),
        ),
        acquisition=AcquisitionSpec(name="none", objective_names=("score",), score_direction="maximize", selection_rule=""),
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


def _env(**config: Any) -> LDMEnv:
    return LDMEnv(
        task_spec=_spec(),
        domain=_BoundedDomain(),
        evaluator=_ScaledEvaluator(),
        config=EnvConfig(**config),
    )


def _surrogate_spec() -> LDMTaskSpec:
    """The synthetic task with a vector surrogate enabled."""

    spec = _spec()
    return LDMTaskSpec(
        task=spec.task,
        candidate_domain=spec.candidate_domain,
        objectives=spec.objectives,
        response_spaces=spec.response_spaces,
        acquisition=spec.acquisition,
        reservoir=spec.reservoir,
        surrogate=SurrogateSpaceSpec(
            kind="vector",
            representation="scalar feature",
            dimension_policy="fixed",
            dimension=1,
            version="synthetic_v1",
        ),
        proposal_search=spec.proposal_search,
        metadata=spec.metadata,
    )


class _ScalarEncoder:
    """Encodes payload x as a 1-D feature vector."""

    def describe(self) -> Any:
        return SurrogateSpaceSpec(
            kind="vector",
            representation="scalar feature",
            dimension_policy="fixed",
            dimension=1,
            version="synthetic_v1",
        )

    def encode(self, candidate: Candidate) -> Any:
        from ldm_tts.optimization.records import SurrogateVector

        return SurrogateVector(
            values=(float(candidate.payload["x"]),),
            version="synthetic_v1",
            source_id=candidate.candidate_id,
        )


def _gp_env(*, beta: float = 1.0, **config: Any) -> LDMEnv:
    from ldm_tts.optimization.gp import RBFGPUCBSelector

    return LDMEnv(
        task_spec=_surrogate_spec(),
        domain=_BoundedDomain(),
        evaluator=_ScaledEvaluator(),
        config=EnvConfig(**config),
        selector=RBFGPUCBSelector(
            objective_name="score",
            beta=beta,
            feature_version="synthetic_v1",
        ),
        surrogate_encoder=_ScalarEncoder(),
    )


def _action(*xs: float) -> str:
    return json.dumps({"candidates": [{"x": x} for x in xs]})


def test_reset_renders_contract() -> None:
    observation = _env(iterations=2, reservoir_size=1).reset()
    assert "synthetic" in observation
    assert "MAXIMIZE" in observation
    assert "score" in observation
    assert "candidate_list" in observation or "candidates" in observation


def test_step_evaluates_and_rewards_improvement() -> None:
    env = _env(iterations=3, reservoir_size=1, reward="improvement")
    env.reset()
    step = env.step(_action(1.0))
    assert step.reward == pytest.approx(2.0)  # first step: baseline 0.0
    assert not step.done
    assert "score" in step.observation
    assert step.info["evaluated"][0]["evaluation"]["status"] == "succeeded"

    second = env.step(_action(3.0))
    assert second.reward == pytest.approx(4.0)  # 6.0 - incumbent 2.0

    worse = env.step(_action(0.5))
    assert worse.reward == 0.0  # no improvement over incumbent
    assert worse.truncated  # budget exhausted after the last round


def test_step_duplicate_is_rejected() -> None:
    env = _env(iterations=2, reservoir_size=1)
    env.reset()
    env.step(_action(2.0))
    step = env.step(_action(2.0))
    assert step.reward == 0.0
    reasons = [item["reason"] for item in step.info["rejections"]]
    assert "already_evaluated" in reasons
    assert "already_evaluated" in step.observation


def test_invalid_action_feedback_and_termination() -> None:
    env = _env(iterations=10, reservoir_size=1, max_empty_reservoir_rounds=2)
    env.reset()
    first = env.step("this is not json")
    assert first.reward == 0.0
    assert first.info["parse_error"] is not None
    assert "could not be parsed" in first.observation
    assert not first.done
    second = env.step("still not json")
    assert second.terminated
    assert second.done
    with pytest.raises(RuntimeError):
        env.step(_action(1.0))


def test_out_of_bounds_rejection_counts_as_empty_round() -> None:
    env = _env(iterations=10, reservoir_size=1, max_empty_reservoir_rounds=1)
    env.reset()
    step = env.step(_action(99.0))
    assert step.info["rejections"][0]["reason"] == "out_of_bounds"
    assert step.terminated
    assert step.info["reward_components"]["kind"] == "all_rejected"


def test_evaluation_failure_reward() -> None:
    env = _env(iterations=2, reservoir_size=1, reward_failure=-1.0)
    env.reset()
    step = env.step(_action(5.0))  # evaluator raises
    assert step.info["evaluated"][0]["evaluation"]["status"] == "failed"
    assert step.reward == -1.0
    assert "FAILED" in step.observation


def test_raw_and_binary_reward_policies() -> None:
    env = _env(iterations=2, reservoir_size=1, reward="raw")
    env.reset()
    assert env.step(_action(4.0)).reward == pytest.approx(8.0)

    env = _env(iterations=4, reservoir_size=1, reward="binary")
    env.reset()
    assert env.step(_action(4.0)).reward == 1.0
    assert env.step(_action(4.5)).reward == 1.0
    assert env.step(_action(4.6)).reward == 1.0


def test_multi_objective_improvement_sums_components() -> None:
    class _BiEvaluator:
        def evaluate(self, candidate: Candidate) -> EvaluationResult:
            x = float(candidate.payload["x"])
            return EvaluationResult(
                candidate.candidate_id,
                "succeeded",
                metrics={"a": x, "b": 10.0 - x},
            )

    env = LDMEnv(
        task_spec=_spec(
            (
                ObjectiveSpec(name="a", direction="maximize"),
                ObjectiveSpec(name="b", direction="maximize"),
            )
        ),
        domain=_BoundedDomain(),
        evaluator=_BiEvaluator(),
        config=EnvConfig(iterations=2, reservoir_size=1, reward="improvement"),
    )
    env.reset()
    first = env.step(_action(3.0))  # oriented (3, 7) -> 10 over zero baseline
    assert first.reward == pytest.approx(10.0)
    second = env.step(_action(4.0))  # oriented (4, 6): a improves by 1, b worsens
    assert second.reward == pytest.approx(1.0)


def test_run_drives_full_episode() -> None:
    env = _env(iterations=3, reservoir_size=1)
    actions = iter([_action(1.0), _action(2.0), _action(4.0)])

    result = env.run(lambda _obs: next(actions))
    assert result.rounds == 3
    assert result.stop_reason == "iteration_budget"
    assert result.best_metrics == {"score": 8.0}
    assert result.total_reward == pytest.approx(8.0)  # 2 + 2 + 4
    assert len(result.history) == 3


def test_acquisition_reward_requires_selector() -> None:
    with pytest.raises(ValueError, match="acquisition"):
        _env(iterations=2, reservoir_size=1, reward="acquisition")


def test_selector_and_encoder_must_pair() -> None:
    with pytest.raises(ValueError, match="configured together"):
        LDMEnv(
            task_spec=_surrogate_spec(),
            domain=_BoundedDomain(),
            evaluator=_ScaledEvaluator(),
            config=EnvConfig(iterations=1),
            surrogate_encoder=_ScalarEncoder(),  # selector missing
        )


def test_gp_selection_follows_acquisition_argmax() -> None:
    env = _gp_env(iterations=3, reservoir_size=2, evaluations_per_round=1)
    env.reset()
    env.step(_action(0.0, 0.5))
    env.step(_action(1.0, 8.0))
    step = env.step(_action(0.2, 9.0))
    # with a fitted GP, x=9.0 (near the high-scoring x=8.0) beats x=0.2
    predictions = step.info["selection"]["predictions"]
    best_id = max(
        predictions,
        key=lambda item: (item["acquisition_score"], item["candidate_id"]),
    )["candidate_id"]
    evaluated_ids = [item["candidate"]["candidate_id"] for item in step.info["evaluated"]]
    assert evaluated_ids == [best_id] == ["cand-9.0"]
    assert step.info["selection"]["metadata"]["surrogate"]["fit_status"] == "fitted"


def test_acquisition_reward_matches_selection_predictions() -> None:
    env = _gp_env(
        iterations=3,
        reservoir_size=2,
        evaluations_per_round=1,
        reward="acquisition",
    )
    env.reset()
    env.step(_action(0.0, 0.5))
    env.step(_action(1.0, 8.0))
    step = env.step(_action(0.2, 9.0))
    evaluated_ids = {item["candidate"]["candidate_id"] for item in step.info["evaluated"]}
    expected = max(
        float(prediction["acquisition_score"])
        for prediction in step.info["selection"]["predictions"]
        if prediction["candidate_id"] in evaluated_ids
    )
    assert step.reward == pytest.approx(expected)
    assert step.info["reward_components"]["kind"] == "acquisition"
    assert step.info["reward_components"]["scores"] == [expected]


def test_observation_stores_surrogate_representation() -> None:
    env = _gp_env(iterations=2, reservoir_size=1)
    env.reset()
    env.step(_action(3.0))
    observation = env.history[-1]
    assert observation.surrogate is not None
    assert observation.surrogate.values == (3.0,)
    assert observation.surrogate.version == "synthetic_v1"


def test_engine_parity_same_proposals_same_observations(tmp_path) -> None:
    """Env steps must produce engine-identical observations for equal inputs."""

    proposals = [{"x": 1.0}, {"x": 3.0}, {"x": 0.5}]

    def policy_actions() -> list[str]:
        return [_action(1.0), _action(3.0), _action(0.5)]

    env = _env(iterations=3, reservoir_size=1)
    env.reset()
    env_observations = []
    for action in policy_actions():
        env.step(action)
        env_observations.extend(env.history[-1:])

    expander = CallableReservoirExpander(
        lambda request: ExpansionResult(
            proposals=tuple(
                RawProposal(payload, source="policy", metadata={"round_idx": request.round_idx})
                for payload in proposals[request.round_idx : request.round_idx + 1]
            )
        )
    )
    runtime = CampaignRuntime.open(
        tmp_path / "parity_run",
        task="synthetic",
        task_spec=_spec(),
        config={},
        budget_limits={"outer_iterations": 3, "external_evaluations": 3},
    )
    engine = LDMEngine(
        task_spec=_spec(),
        expander=expander,
        candidate_domain=_BoundedDomain(),
        evaluator=_ScaledEvaluator(),
        runtime=runtime,
    )
    result = engine.run(LDMEngineConfig(iterations=3, reservoir_size=1))

    assert result.stop_reason == "iteration_budget"
    engine_observations = result.state.observations
    assert [item.candidate.canonical_key for item in engine_observations] == [
        item.candidate.canonical_key for item in env_observations
    ]
    assert [item.metrics for item in engine_observations] == [
        item.metrics for item in env_observations
    ]
    assert [item.round_idx for item in engine_observations] == [
        item.round_idx for item in env_observations
    ]
