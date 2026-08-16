from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from finetune.prepare_dataset import (
    DATASET_INFO_FILENAME,
    EVAL_DATASET,
    EVAL_FILENAME,
    SPLIT_SUMMARY_FILENAME,
    TRAIN_DATASET,
    TRAIN_FILENAME,
    prepare_dataset,
)
from ldm_tts.data import make_complete_design_ir


def _ir(run_id: str, design: str, *, reasoning="Supported rationale.", available=True):
    row = make_complete_design_ir(
        task_id="small_molecule",
        domain="molecule",
        task_description="Propose one molecule.",
        objectives=[
            {
                "name": "vina",
                "direction": "minimize",
                "description": "Lower is better.",
            }
        ],
        design_space_description="Single-component organic SMILES.",
        observations=[],
        candidates=[{"design": design, "rationale": "Nearby candidate."}],
        request_description="Generate one valid candidate.",
        reasoning=reasoning,
        reasoning_available=available,
    )
    row["collection"] = {"provenance": {"run_id": run_id}}
    return row


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _designs(path):
    return {
        json.loads(row["output"])["payload"]["candidates"][0]["design"]
        for row in _read_jsonl(path)
    }


def test_prepare_dataset_writes_group_disjoint_train_and_eval_shards(tmp_path):
    source = tmp_path / "augmented_ir.jsonl"
    destination = tmp_path / "prepared"
    rows = [
        _ir("run-a", "CCN"),
        _ir("run-a", "CCO"),
        _ir("run-b", "CCC"),
        _ir("run-c", "CCCl"),
        _ir("run-d", "CCBr"),
    ]
    _write_jsonl(source, rows)

    report = prepare_dataset([source], destination, eval_fraction=0.25, seed=7)

    train_designs = _designs(destination / TRAIN_FILENAME)
    eval_designs = _designs(destination / EVAL_FILENAME)
    assert train_designs.isdisjoint(eval_designs)
    run_a_is_train = "run_id:run-a" not in report.eval_group_keys
    assert ({"CCN", "CCO"} <= train_designs) == run_a_is_train
    assert report.train_rows + report.eval_rows == len(rows)
    assert report.train_groups == 3
    assert report.eval_groups == 1

    info = json.loads((destination / DATASET_INFO_FILENAME).read_text(encoding="utf-8"))
    assert info[TRAIN_DATASET]["file_name"] == TRAIN_FILENAME
    assert info[EVAL_DATASET]["file_name"] == EVAL_FILENAME
    assert (
        json.loads((destination / SPLIT_SUMMARY_FILENAME).read_text(encoding="utf-8"))[
            "eval_groups"
        ]
        == 1
    )


def test_prepare_dataset_filters_rows_without_supported_reasoning(tmp_path):
    source = tmp_path / "augmented_ir.jsonl"
    destination = tmp_path / "prepared"
    _write_jsonl(
        source,
        [
            _ir("run-a", "CCN"),
            _ir("run-b", "CCC"),
            _ir("run-c", "CCO", reasoning=None),
            _ir("run-d", "CCCl", available=False),
        ],
    )

    report = prepare_dataset([source], destination, eval_fraction=0.5)

    assert report.train_rows == 1
    assert report.eval_rows == 1
    assert report.skipped_missing_reasoning == 1
    assert report.skipped_reasoning_unavailable == 1


def test_prepare_dataset_requires_two_groups_and_refuses_overwrite(tmp_path):
    source = tmp_path / "augmented_ir.jsonl"
    destination = tmp_path / "prepared"
    _write_jsonl(source, [_ir("run-a", "CCN")])

    with pytest.raises(ValueError, match="at least two eligible provenance groups"):
        prepare_dataset([source], destination)

    _write_jsonl(source, [_ir("run-a", "CCN"), _ir("run-b", "CCC")])
    prepare_dataset([source], destination)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_dataset([source], destination)


def test_prepare_dataset_cli_writes_default_artifacts(tmp_path):
    source = tmp_path / "augmented_ir.jsonl"
    destination = tmp_path / "prepared"
    _write_jsonl(source, [_ir("run-a", "CCN"), _ir("run-b", "CCC")])
    repo_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "finetune/prepare_dataset.py"),
            "--input",
            str(source),
            "--output-dir",
            str(destination),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["train_rows"] == 1
    assert report["eval_rows"] == 1
    assert (destination / DATASET_INFO_FILENAME).is_file()


def test_full_sft_config_matches_prepared_dataset_contract():
    repo_root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (repo_root / "finetune/config/full_sft_rationale.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["dataset"] == TRAIN_DATASET
    assert config["eval_dataset"] == EVAL_DATASET
    assert config["dataset_dir"] == "../data/generated/full_sft"
    assert config["template"] == "qwen3_5_nothink"
    assert config["enable_thinking"] is False
    assert "val_size" not in config
    assert config["overwrite_output_dir"] is False
