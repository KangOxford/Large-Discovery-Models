"""EpisodeSpec serialization and the prompt-data generator CLI."""

from __future__ import annotations

import json

import pytest

from ldm_rl.episodes import EpisodeSpec, main, make_prompt_rows


def test_make_prompt_rows_renders_spec_json() -> None:
    rows = make_prompt_rows(
        [
            EpisodeSpec(task="small_molecule", mode="mock", iterations=4, seed=7),
            EpisodeSpec(task="small_molecule", mode="mock", iterations=4, seed=8),
        ]
    )
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"prompt", "label"}
        assert row["label"] == ""
        spec = EpisodeSpec.from_json(row["prompt"])
        assert spec.task == "small_molecule"
        assert spec.iterations == 4
    assert [EpisodeSpec.from_json(r["prompt"]).seed for r in rows] == [7, 8]


def test_main_writes_jsonl_with_seed_offset(tmp_path) -> None:
    output = tmp_path / "episodes.jsonl"
    rc = main(
        [
            "--output", str(output),
            "--task", "small_molecule",
            "--mode", "mock",
            "--count", "3",
            "--iterations", "5",
            "--seed-offset", "100",
        ]
    )
    assert rc == 0
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    specs = [EpisodeSpec.from_json(json.loads(line)["prompt"]) for line in lines]
    assert [spec.seed for spec in specs] == [100, 101, 102]
    assert [spec.iterations for spec in specs] == [5, 5, 5]


def test_main_real_kwargs_are_serialized(tmp_path) -> None:
    output = tmp_path / "episodes.jsonl"
    rc = main(
        [
            "--output", str(output),
            "--task", "ai4bio_mutation_effect_prediction",
            "--mode", "real",
            "--count", "1",
            "--real-kwargs",
            '{"upstream_root": "/data/u", "data_dir": "/data/d", "cv_dir": "/data/c"}',
        ]
    )
    assert rc == 0
    line = output.read_text(encoding="utf-8").strip().splitlines()[0]
    spec = EpisodeSpec.from_json(json.loads(line)["prompt"])
    assert spec.mode == "real"
    assert spec.real == {
        "upstream_root": "/data/u",
        "data_dir": "/data/d",
        "cv_dir": "/data/c",
    }


def test_main_invalid_real_kwargs_json_fails(tmp_path) -> None:
    output = tmp_path / "episodes.jsonl"
    with pytest.raises(SystemExit):
        main(
            [
                "--output", str(output),
                "--task", "small_molecule",
                "--mode", "real",
                "--real-kwargs", "not-json",
            ]
        )


def test_main_mock_mode_rejects_real_kwargs(tmp_path) -> None:
    output = tmp_path / "episodes.jsonl"
    with pytest.raises(ValueError, match="must not carry real-evaluation kwargs"):
        main(
            [
                "--output", str(output),
                "--task", "small_molecule",
                "--mode", "mock",
                "--real-kwargs", '{"vina_bin": "/x"}',
            ]
        )


def test_episode_spec_round_trip_preserves_all_fields() -> None:
    spec = EpisodeSpec(
        task="small_molecule",
        mode="mock",
        iterations=6,
        reservoir_size=3,
        evaluations_per_round=2,
        reward="improvement",
        max_empty_reservoir_rounds=5,
        target_successful_evaluations=80,
        max_evaluation_attempts=640,
        max_evaluation_attempts_per_round=8,
        replace_failed_evaluations=True,
        seed=42,
        context={"assays": ["a", "b"]},
    )
    restored = EpisodeSpec.from_json(spec.to_json())
    assert restored == spec
