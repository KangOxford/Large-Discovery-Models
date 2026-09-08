"""Environment semantics: parsing, admission, reward invariants, selection."""

from __future__ import annotations

import json

import pytest

from ldm_rl import EnvConfig, LDMEnv
from ldm_rl.factories import build_env
from ldm_tts.contracts import Candidate, EvaluationResult, RawProposal
from ldm_tts.optimization.records import BOObservation, SurrogateVector

from tasks.nanogpt.core import (
    rl_adapter,
    rl_encoder,
    rl_knobs as K,
    rl_real,
    rl_task,
)

REFERENCE_BPB = 0.9961


def _instance(free=("DEPTH",)):
    return rl_task.KnobInstance.from_kwargs(free_knobs=list(free))


def _action(*rows):
    return json.dumps({"proposals": list(rows)})


# --- parser ---------------------------------------------------------------


def test_parser_accepts_a_bare_object_and_a_single_fence():
    payload = {"proposals": [{"DEPTH": 8}]}
    assert rl_task.parse_knob_proposals(json.dumps(payload)) == [{"DEPTH": 8}]
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert rl_task.parse_knob_proposals(fenced) == [{"DEPTH": 8}]


def test_parser_rejects_reasoning_around_the_object():
    # This is the measured reward hole in the shared permissive parser: it
    # takes everything between the first "{" and the last "}", so a policy that
    # never commits still gets scored on a draft inside its own reasoning.
    text = 'Let me think. Maybe {"proposals": [{"DEPTH": 8}]} looks right.'
    with pytest.raises(ValueError, match="exactly one JSON object"):
        rl_task.parse_knob_proposals(text)
    # ...and the shared parser would indeed have accepted it.
    from ldm_tts.transport.parsing import load_json_object

    assert load_json_object(text)["proposals"] == [{"DEPTH": 8}]


def test_parser_rejects_truncated_reasoning_that_ends_in_a_dict():
    text = 'Reasoning about depth... my current draft is {"proposals": [{"DEPTH": 8}]}'
    with pytest.raises(ValueError):
        rl_task.parse_knob_proposals(text)


def test_parser_enforces_the_proposal_count():
    with pytest.raises(ValueError, match="exactly 2"):
        rl_task.parse_knob_proposals(_action({"DEPTH": 8}), expected_count=2)


def test_parser_rejects_self_reported_scores():
    # A proposal states an action, never its own outcome.
    for banned in ("val_bpb", "score", "predicted_val_bpb"):
        with pytest.raises(ValueError, match="may not include"):
            rl_task.parse_knob_proposals(_action({"DEPTH": 8, banned: 0.5}))


def test_parser_upper_cases_knob_names():
    assert rl_task.parse_knob_proposals(_action({"depth": 8})) == [{"DEPTH": 8}]


# --- domain ---------------------------------------------------------------


def _admit(payload, free=("DEPTH",)):
    domain = rl_task.NanogptKnobDomain(_instance(free))
    return domain.admit(RawProposal(payload, source="test"))


def test_domain_admits_and_assigns_stable_identity():
    first = _admit({"DEPTH": 12})
    second = _admit({"DEPTH": 12.0})
    assert isinstance(first, Candidate)
    assert first.canonical_key == second.canonical_key
    assert first.candidate_id == second.candidate_id
    assert first.payload["config"]["DEPTH"] == 12


@pytest.mark.parametrize(
    ("payload", "free", "reason"),
    [
        ("not an object", ("DEPTH",), "invalid_payload"),
        ({}, ("DEPTH",), "missing_knobs"),
        ({"DEPTH": 8, "HEAD_DIM": 64}, ("DEPTH",), "not_free"),
        ({"DEPTH": 99}, ("DEPTH",), "out_of_domain"),
        ({"DEVICE_BATCH_SIZE": 96}, ("DEVICE_BATCH_SIZE",), "rejected_by_trainer"),
    ],
)
def test_domain_rejections_are_graded(payload, free, reason):
    # The step observation reports the reason, so these must stay distinct:
    # "you forgot a knob" and "the trainer refuses this" are different lessons.
    rejection = _admit(payload, free)
    assert rejection.reason == reason
    assert rejection.message


# --- encoder --------------------------------------------------------------


def test_encoder_dimension_and_range():
    for config in (
        K.validate(K.DEFAULTS),
        K.validate({**K.DEFAULTS, "DEPTH": 16, "ASPECT_RATIO": 128, "HEAD_DIM": 64,
                    "WINDOW_PATTERN": "LLLL", "TOTAL_BATCH_SIZE": 1048576,
                    "DEVICE_BATCH_SIZE": 32}),
        K.validate({**K.DEFAULTS, "DEPTH": 2, "ASPECT_RATIO": 32, "HEAD_DIM": 128,
                    "WINDOW_PATTERN": "SSSS", "TOTAL_BATCH_SIZE": 262144,
                    "DEVICE_BATCH_SIZE": 128}),
    ):
        vector = rl_encoder.encode_config(config)
        assert len(vector) == rl_encoder.FEATURE_DIM == len(rl_encoder.FEATURE_NAMES)
        # A single RBF lengthscale spans the whole vector, so any coordinate
        # escaping [0, 1] would quietly dominate the kernel.
        assert all(0.0 <= value <= 1.0 for value in vector), vector


def test_encoder_is_deterministic_and_discriminating():
    base = K.validate(K.DEFAULTS)
    assert rl_encoder.encode_config(base) == rl_encoder.encode_config(dict(base))
    assert rl_encoder.encode_config(base) != rl_encoder.encode_config(
        K.validate({**base, "DEPTH": 9})
    )


def test_total_batch_size_is_not_inert_in_feature_space():
    # The previous surrogate scored a TOTAL_BATCH_SIZE change as exactly 0
    # while it cost 0.023 bpb on the real trainer. It moves two coordinates
    # here: its own, and log_grad_accum_steps.
    base = K.validate(K.DEFAULTS)
    other = K.validate({**base, "TOTAL_BATCH_SIZE": 1048576})
    before = rl_encoder.encode_config(base)
    after = rl_encoder.encode_config(other)
    moved = {
        name for name, a, b in zip(rl_encoder.FEATURE_NAMES, before, after) if a != b
    }
    assert moved == {"total_batch_size", "log_grad_accum_steps"}


def test_task_spec_and_encoder_declarations_agree():
    # LDMEnv refuses to build if they disagree; assert it directly so the
    # failure is a clear message rather than a construction error.
    spec = rl_task.describe_rl_task(_instance()).surrogate
    described = rl_encoder.NanogptSurrogateEncoder().describe()
    for field in ("kind", "dimension_policy", "dimension", "version"):
        assert getattr(spec, field) == getattr(described, field)


# --- reward invariants ----------------------------------------------------


class ScriptedEvaluator:
    """Returns a val_bpb chosen per DEPTH, so a test can script the outcomes."""

    def __init__(self, by_depth: dict[int, float]):
        self.by_depth = by_depth

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        depth = candidate.payload["config"]["DEPTH"]
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"val_bpb": float(self.by_depth[depth])},
        )


def _scripted_env(by_depth, *, ref_point, iterations=3):
    instance = _instance()
    return LDMEnv(
        task_spec=rl_task.describe_rl_task(instance, surrogate=False),
        domain=rl_task.NanogptKnobDomain(instance),
        evaluator=ScriptedEvaluator(by_depth),
        config=EnvConfig(
            iterations=iterations,
            reservoir_size=1,
            evaluations_per_round=1,
            reward="improvement",
            reward_ref_point=ref_point,
        ),
        parse_action=lambda text: rl_task.parse_knob_proposals(text, expected_count=1),
    )


#: depth -> measured val_bpb. 4 is bad, 8 is mediocre, 12 is the best.
SCRIPT = {4: 1.30, 8: 1.05, 12: 0.97}
HONEST_ORDER = [8, 12, 4]
SANDBAG_ORDER = [4, 8, 12]


def _play(order, ref_point):
    env = _scripted_env(SCRIPT, ref_point=ref_point)
    env.reset()
    return sum(env.step(_action({"DEPTH": depth})).reward for depth in order)


def test_improvement_reward_is_path_independent_with_a_fixed_reference():
    # THE invariant that makes this reward safe: the episode total depends only
    # on the best config found, never on the order it was found in. Without it
    # a policy is paid for opening badly and then "recovering".
    honest = _play(HONEST_ORDER, (-REFERENCE_BPB,))
    sandbagged = _play(SANDBAG_ORDER, (-REFERENCE_BPB,))
    assert honest == pytest.approx(sandbagged)
    # ...and the total is exactly how far the best config beat the reference.
    assert honest == pytest.approx(REFERENCE_BPB - min(SCRIPT.values()))


def test_without_a_reference_sandbagging_pays_more():
    # Regression guard: this is the failure the reference point removes. If
    # this ever stops holding, the clamp in LDMEnv._reward has been undone and
    # the invariant above is passing for the wrong reason.
    assert _play(SANDBAG_ORDER, None) > _play(HONEST_ORDER, None)


def test_a_config_worse_than_the_reference_earns_zero():
    env = _scripted_env(SCRIPT, ref_point=(-REFERENCE_BPB,))
    env.reset()
    assert env.step(_action({"DEPTH": 4})).reward == pytest.approx(0.0)


def test_repeating_a_measured_config_is_rejected_as_a_duplicate():
    # Re-proposing costs a full training budget for a number already known, so
    # the reservoir must drop it rather than pay for it again.
    env = _scripted_env(SCRIPT, ref_point=(-REFERENCE_BPB,))
    env.reset()
    env.step(_action({"DEPTH": 12}))
    step = env.step(_action({"DEPTH": 12}))
    assert step.info["evaluated"] == []
    assert step.info["rejections"]


# --- selection direction --------------------------------------------------


def _bo_observation(depth, val_bpb):
    config = K.validate({**K.DEFAULTS, "DEPTH": depth})
    candidate_id = f"cfg-{depth}"
    return BOObservation(
        candidate_id=candidate_id,
        objectives=(float(val_bpb),),
        feature=SurrogateVector(
            rl_encoder.encode_config(config),
            rl_encoder.FEATURE_VERSION,
            source_id=candidate_id,
        ),
    )


def _candidates_and_reps(history_depths):
    candidates, reps = [], {}
    for depth in history_depths:
        config = K.validate({**K.DEFAULTS, "DEPTH": depth})
        candidate_id = f"cfg-{depth}"
        candidates.append(
            Candidate(candidate_id=candidate_id, payload={"config": config},
                      canonical_key=K.canonical_key(config))
        )
        reps[candidate_id] = SurrogateVector(
            rl_encoder.encode_config(config), rl_encoder.FEATURE_VERSION,
            source_id=candidate_id,
        )
    return candidates, reps


def test_selector_prefers_the_lower_predicted_val_bpb():
    # Regression test for a direction bug that is silent by construction:
    # BOObservation.from_observation stores the RAW metric and
    # RBFGPUCBSelector hardcodes minimize=(False,), so an unwrapped selector
    # spends the budget on the candidate it thinks is WORST.
    depths = [2, 4, 6, 8, 10, 12]
    scores = {depth: 1.30 - 0.03 * depth for depth in depths}  # deeper is better
    history = [_bo_observation(depth, scores[depth]) for depth in depths]
    candidates, reps = _candidates_and_reps(depths)
    best_depth = min(depths, key=lambda depth: scores[depth])
    worst_depth = max(depths, key=lambda depth: scores[depth])

    wrapped = rl_real.build_selector(gp_beta=0.0)
    wrapped.fit(history)
    assert wrapped.select(candidates, reps, count=1).selected_candidate_ids == (
        f"cfg-{best_depth}",
    )

    raw = wrapped.inner
    raw.fit(history)
    assert raw.select(candidates, reps, count=1).selected_candidate_ids == (
        f"cfg-{worst_depth}",
    ), "unwrapped selector should still exhibit the bug this wrapper fixes"


def test_selector_negates_objectives_on_the_way_in():
    wrapped = rl_real.build_selector()
    wrapped.fit([_bo_observation(8, 0.99)])
    assert wrapped.inner.history[0].objectives == (pytest.approx(-0.99),)


# --- factory --------------------------------------------------------------


def test_build_env_mock_runs_a_full_episode():
    env = build_env(
        "nanogpt",
        mode="mock",
        config=EnvConfig(iterations=3, reservoir_size=2, evaluations_per_round=1,
                         reward="improvement", reward_ref_point=(-REFERENCE_BPB,)),
        free_knobs=["DEPTH", "MATRIX_LR"],
        reservoir_size=2,
    )
    observation = env.reset()
    assert "val_bpb" in observation and "MATRIX_LR" in observation
    step = env.step(_action({"DEPTH": 8, "MATRIX_LR": 0.04},
                            {"DEPTH": 12, "MATRIX_LR": 0.05}))
    assert step.info["evaluated"]
    assert step.info["selection"]["selected_candidate_ids"]


def test_real_mode_needs_no_search_engine_import():
    # The task previously looked incompatible with the shared environment
    # because its candidates were state references into an OperationSearchEngine
    # that a task-local expander had to materialise first. A full knob dict
    # needs no such indirection, and the RL path must not drag in the campaign
    # search machinery (which pulls the whole task runtime with it).
    import ast
    import pathlib

    heavy = {"workflow", "engine_adapters", "search_core", "single_search"}
    for module in (rl_real, rl_adapter, rl_task, rl_encoder, K):
        tree = ast.parse(pathlib.Path(module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
        leaves = {name.rsplit(".", 1)[-1] for name in imported}
        assert not (leaves & heavy), f"{module.__name__} imports {leaves & heavy}"


def test_reference_config_holds_free_knobs_at_their_defaults():
    instance = rl_task.KnobInstance.from_kwargs(free_knobs=["DEPTH", "MATRIX_LR"])
    reference = rl_real.reference_config(instance)
    assert reference["DEPTH"] == K.DEFAULTS["DEPTH"]
    assert reference["MATRIX_LR"] == K.DEFAULTS["MATRIX_LR"]
    assert reference == K.validate(K.DEFAULTS)
