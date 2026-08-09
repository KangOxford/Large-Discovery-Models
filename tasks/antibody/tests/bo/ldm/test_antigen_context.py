from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bo.ldm.antigen_context import (
    build_bo_history_context,
    build_llm_prompt_context,
    collect_absolut_antigen_context,
    extract_feature_lines,
    parse_key_values,
    run_absolut_info,
    save_llm_context_snapshot,
    split_antigen_id,
)


def test_antigen_ids_split_optional_chain():
    assert split_antigen_id("1ADQ_A") == ("1ADQ", "A")
    assert split_antigen_id("1ADQ") == ("1ADQ", None)
    assert split_antigen_id("complex_chain_A") == ("complex", "chain_A")


def test_absolut_info_reports_missing_executable(tmp_path: Path):
    result = run_absolut_info(str(tmp_path), "info_antigen", "1ADQ_A")

    assert result["ok"] is False
    assert result["command"] == "info_antigen"
    assert "Absolut executable not found" in result["error"]


def test_absolut_info_returns_process_output(tmp_path: Path, monkeypatch):
    executable = tmp_path / "src" / "bin" / "Absolut"
    executable.parent.mkdir(parents=True)
    executable.touch()
    completed = SimpleNamespace(returncode=2, stdout="details", stderr="warning")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed

    monkeypatch.setattr("bo.ldm.antigen_context.subprocess.run", fake_run)

    result = run_absolut_info(str(tmp_path), "info_filenames", "1ADQ_A", timeout_s=9)

    assert result == {
        "command": "info_filenames",
        "ok": False,
        "returncode": 2,
        "stdout": "details",
        "stderr": "warning",
    }
    assert calls[0][0] == [str(executable), "info_filenames", "1ADQ_A"]
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert calls[0][1]["timeout"] == 9


def test_parsers_extract_values_and_named_features():
    text = """
    Antigen: 1ADQ_A
    Filename\tstructures/1adq.pdb
    forbidden positions: 1, 4
    glycan site: N42
    binding hotspot residues: Y10
    hotspot core residues: W12
    residues bound within 100: A3
    line without delimiter
    """

    values = parse_key_values(text)
    features = extract_feature_lines(text)

    assert values["Antigen"] == "1ADQ_A"
    assert values["Filename"] == "structures/1adq.pdb"
    assert features["forbidden_positions"] == ["forbidden positions: 1, 4"]
    assert features["glycans"] == ["glycan site: N42"]
    assert features["hotspots"] == [
        "binding hotspot residues: Y10",
        "hotspot core residues: W12",
    ]
    assert features["hotspot_core_residues"] == ["hotspot core residues: W12"]
    assert features["bound_100_residues"] == ["residues bound within 100: A3"]


def test_collect_antigen_context_combines_commands_and_raw_output(tmp_path: Path, monkeypatch):
    responses = {
        "info_antigen": {
            "command": "info_antigen", "ok": True, "returncode": 0,
            "stdout": "Antigen: 1ADQ_A\nhotspot: Y10", "stderr": "",
        },
        "info_filenames": {
            "command": "info_filenames", "ok": True, "returncode": 0,
            "stdout": "Filename: antigen.pdb", "stderr": "note",
        },
    }
    monkeypatch.setattr(
        "bo.ldm.antigen_context.run_absolut_info",
        lambda path, command, antigen_id, timeout_s: responses[command],
    )

    context = collect_absolut_antigen_context(
        {"antigen": "1ADQ_A", "path": str(tmp_path)},
        timeout_s=5,
        include_raw=True,
    )

    assert context["pdb_id"] == "1ADQ"
    assert context["chain_id"] == "A"
    assert context["commands"]["info_antigen"] == {
        "command": "info_antigen", "ok": True, "returncode": 0,
    }
    assert context["parsed_key_values"]["Filename"] == "antigen.pdb"
    assert context["features"]["hotspots"] == ["hotspot: Y10"]
    assert context["raw_outputs"]["info_filenames_stderr"] == "note"


def test_collect_antigen_context_omits_raw_output_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "bo.ldm.antigen_context.run_absolut_info",
        lambda *args, **kwargs: {"command": args[1], "ok": False, "stdout": "", "stderr": ""},
    )

    context = collect_absolut_antigen_context(
        {"antigen": "1ADQ", "path": str(tmp_path)},
    )

    assert context["chain_id"] is None
    assert "raw_outputs" not in context
    assert context["parsed_key_values"] == {}


def test_bo_history_reports_empty_and_ranked_observations():
    assert build_bo_history_context(None, None) == {
        "num_observations": 0,
        "current_best_sequence": None,
        "current_best_value": None,
        "top_observed_sequences": [],
    }

    casmopolitan = SimpleNamespace(
        x=np.array([[0, 1, 2], [2, 1, 0], [1, 1, 1]]),
        fx=np.array([[3.0], [1.0], [2.0]]),
    )
    optim = SimpleNamespace(casmopolitan=casmopolitan)
    f_obj = SimpleNamespace(fbox=SimpleNamespace(idx_to_AA={0: "A", 1: "C", 2: "D"}))

    history = build_bo_history_context(optim, f_obj, top_k=2)

    assert history["num_observations"] == 3
    assert history["current_best_sequence"] == "DCA"
    assert history["current_best_value"] == 1.0
    assert history["top_observed_sequences"] == [
        {"rank": 1, "sequence": "DCA", "value": 1.0},
        {"rank": 2, "sequence": "CCC", "value": 2.0},
    ]


def test_prompt_context_and_snapshot_are_json_serializable(tmp_path: Path):
    antigen = {"antigen_id": "1ADQ_A", "features": {"hotspots": ["Y10"]}}
    prompt = build_llm_prompt_context(antigen, {"num_observations": 0})

    assert prompt["antigen_context"] is antigen
    assert prompt["task"]["expected_json_keys"] == [
        "trust_region_centers", "mutation_policy", "soft_constraints", "antigen_preferences",
    ]

    path = save_llm_context_snapshot(str(tmp_path / "nested"), antigen, itern=7)
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    assert Path(path).name == "llm_prompt_context_iter_0007.json"
    assert saved["antigen_context"]["antigen_id"] == "1ADQ_A"
    assert saved["bo_history"]["num_observations"] == 0


def test_snapshot_without_iteration_uses_init_suffix(tmp_path: Path):
    path = save_llm_context_snapshot(str(tmp_path), {"antigen_id": "1ADQ"})
    assert Path(path).name == "llm_prompt_context_init.json"
