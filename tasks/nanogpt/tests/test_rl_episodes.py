"""Episode generation: instance sampling, reference wiring, real-mode build."""

from __future__ import annotations

import json

import pytest

from ldm_rl.episodes import EpisodeSpec
from ldm_rl.factories import build_env
from tasks.nanogpt.core import rl_knobs as K
from tasks.nanogpt.scripts import gen_rl_episodes as G

REFERENCES = {"150": 1.0712, "300": 0.9961}


def _generate(tmp_path, *extra, count=12, iterations=4):
    reference_file = tmp_path / "ref.json"
    reference_file.write_text(json.dumps(REFERENCES))
    output = tmp_path / "ep.jsonl"
    real_kwargs = json.dumps(
        {"output_dir": str(tmp_path / "work"), "eval_gpus": "0,1"}
    )
    assert (
        G.main(
            [
                "--output", str(output),
                "--count", str(count),
                "--iterations", str(iterations),
                "--budgets", "150,300",
                "--references", str(reference_file),
                "--seed-offset", "7",
                "--real-kwargs", real_kwargs,
                *extra,
            ]
        )
        == 0
    )
    return [
        EpisodeSpec.from_json(json.loads(line)["prompt"])
        for line in output.read_text().splitlines()
    ]


def test_batching_alone_has_too_small_a_space_to_search():
    # 3 TOTAL_BATCH_SIZE x 4 DEVICE_BATCH_SIZE = 12 combinations, but
    # DEVICE_BATCH_SIZE=96 divides none of them, leaving 9.
    schema = K.load_schema()
    assert G.legal_space_size(G.KNOB_GROUPS["batching"], schema) == 9
    # A free continuous knob makes the space unbounded.
    assert G.legal_space_size(G.KNOB_GROUPS["schedule"], schema) == float("inf")
    assert G.legal_space_size(G.KNOB_GROUPS["architecture"], schema) > 1000


def test_generator_never_emits_an_exhaustible_instance(tmp_path):
    schema = K.load_schema()
    specs = _generate(tmp_path, "--free-knob-mode", "groups", count=40)
    assert specs
    min_space = 4 * 2 + 4
    for spec in specs:
        free = list(spec.real["free_knobs"])
        assert G.legal_space_size(free, schema) >= min_space
    # The batching knobs must still appear -- merged, not dropped, or the
    # policy never learns to set them.
    assert any(
        "TOTAL_BATCH_SIZE" in spec.real["free_knobs"] for spec in specs
    )


def test_reference_is_negated_into_oriented_space(tmp_path):
    for spec in _generate(tmp_path):
        budget = f"{spec.real['time_budget']:.0f}"
        expected = -REFERENCES[budget]
        assert spec.reward_ref_point == (pytest.approx(expected),)
        assert spec.real["reference_val_bpb"] == pytest.approx(REFERENCES[budget])
        assert spec.reward == "improvement"


def test_missing_reference_is_refused_by_default(tmp_path):
    output = tmp_path / "ep.jsonl"
    with pytest.raises(SystemExit):
        G.main(["--output", str(output), "--count", "2", "--budgets", "300"])


def test_missing_reference_can_be_forced_but_leaves_no_ref_point(tmp_path):
    output = tmp_path / "ep.jsonl"
    assert (
        G.main(
            [
                "--output", str(output), "--count", "2", "--budgets", "300",
                "--allow-missing-reference",
            ]
        )
        == 0
    )
    spec = EpisodeSpec.from_json(
        json.loads(output.read_text().splitlines()[0])["prompt"]
    )
    assert spec.reward_ref_point is None


def test_evaluations_per_round_cannot_exceed_the_reservoir(tmp_path):
    with pytest.raises(SystemExit):
        _generate(tmp_path, "--evaluations-per-round", "3", "--reservoir-size", "2")


def test_randomised_pinned_stays_legal_and_differs_from_defaults(tmp_path):
    specs = _generate(tmp_path, "--randomise-pinned", count=20)
    checked = 0
    for spec in specs:
        pinned = spec.real.get("pinned") or {}
        if len(spec.real["free_knobs"]) == len(K.knob_names()):
            # An "all knobs free" instance holds nothing, so there is nothing
            # to randomise.
            assert not pinned
            continue
        assert pinned, "an instance with held knobs must have them randomised"
        # A pinned assignment must never make the instance unsatisfiable.
        config = K.with_defaults(spec.real["free_knobs"], {}, pinned)
        K.check_hard_constraints(K.validate(config))
        assert any(pinned[name] != K.DEFAULTS[name] for name in pinned)
        checked += 1
    assert checked, "expected at least one instance with held knobs"


def test_generated_real_rows_build_an_environment(tmp_path):
    # Proves the row -> EpisodeSpec -> build_env(mode="real") path is wired
    # without needing a GPU: constructing the evaluator does not run anything.
    specs = _generate(tmp_path, count=4)
    for spec in specs:
        env = build_env(
            spec.task,
            mode=spec.mode,
            config=spec.to_env_config(),
            context=spec.context,
            seed=spec.seed,
            **spec.real,
        )
        observation = env.reset()
        assert "val_bpb" in observation
        for name in spec.real["free_knobs"]:
            assert name in observation
        assert env.config.reward_ref_point is not None
        # The prompt must state the measured bar the reward is relative to.
        assert f"{spec.real['reference_val_bpb']}" in observation


def test_episode_prompt_does_not_leak_an_objective_estimate(tmp_path):
    spec = _generate(tmp_path, count=1)[0]
    env = build_env(
        spec.task, mode=spec.mode, config=spec.to_env_config(),
        context=spec.context, seed=spec.seed, **spec.real,
    )
    observation = env.reset()
    # The reference is a measurement and is meant to be shown. What must not
    # appear is any per-proposal prediction the policy could optimise against
    # instead of the real objective.
    assert "surrogate" not in observation.lower()
    assert "predicted" not in observation.lower()
