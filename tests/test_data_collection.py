from __future__ import annotations

import json

import pytest

from ldm_tts.data_collection import (
    DataCollectionSink,
    LDMDataCollectionError,
    dataset_info_payload,
    make_complete_design_ir,
    make_parameter_edit_ir,
    render_record,
    smallmol_ir_from_prompt_response,
    validate_ir_record,
)


def _example_ir():
    return make_complete_design_ir(
        task_id="small_molecule",
        domain="molecule",
        task_description="Propose molecules for a two-objective search.",
        objectives=[
            {
                "name": "vina",
                "direction": "minimize",
                "description": "Docking score; lower is better.",
            },
            {
                "name": "activity",
                "direction": "maximize",
                "description": "Predicted activity; higher is better.",
            },
        ],
        design_space_description="Single-component organic SMILES.",
        observations=[
            {
                "design": "CCO",
                "results": {"vina": -2.4, "activity": 5.1},
                "roles": ["recent"],
            }
        ],
        candidates=[
            {"design": "CCN", "rationale": "amine neighbor"},
            {"design": "CCC", "rationale": "alkyl neighbor"},
        ],
        request_description="Generate valid, novel SMILES without scores.",
        num_candidates=2,
        num_evaluated=1,
        do_not_repeat=["CCO"],
    )


def test_make_complete_design_ir_normalizes_task_and_renders_action_json():
    ir = _example_ir()

    validate_ir_record(ir)
    assert ir["task"]["id"] == "smallmol"

    row = render_record(ir)
    assert row["source"] == "smallmol"
    assert row["input"] == ""
    assert "Return a single JSON object" in row["instruction"]
    assert "Do not repeat" in row["instruction"]
    action = json.loads(row["output"])
    assert action["type"] == "propose"
    assert action["payload"]["candidates"][0]["design"] == "CCN"


def test_sink_writes_ir_sft_and_keeps_provenance_out_of_prompt(tmp_path):
    sink = DataCollectionSink(tmp_path)
    ir = _example_ir()

    sink.append(
        ir,
        provenance={"run_id": "private-run-marker"},
        outcome={"selected": ["CCN"]},
    )

    ir_rows = [
        json.loads(line)
        for line in (tmp_path / "ldm_ir.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sft_rows = [
        json.loads(line)
        for line in (tmp_path / "ldm_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_info = json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8"))

    assert ir_rows[0]["collection"]["provenance"]["run_id"] == "private-run-marker"
    assert ir_rows[0]["collection"]["outcome"]["selected"] == ["CCN"]
    assert len(sft_rows) == 1
    assert "private-run-marker" not in sft_rows[0]["instruction"]
    assert "collection" not in sft_rows[0]["instruction"]
    assert dataset_info == dataset_info_payload("ldm_sft.jsonl")


def test_validate_rejects_action_not_allowed():
    ir = _example_ir()
    ir["action"]["type"] = "expand_design_space"

    with pytest.raises(LDMDataCollectionError, match="not in request.allowed_actions"):
        validate_ir_record(ir)


def test_smallmol_prompt_response_adapter_builds_ldm2_ir():
    history = {
        "pareto_front": [{"smiles": "CCO", "scores": [-2.4, 5.1]}],
        "top_low_vina": [{"smiles": "CCC", "scores": [-2.8, 5.0]}],
        "top_high_activity": [],
        "balanced_elites": [],
        "recent_selected": [{"smiles": "CCO", "scores": [-2.4, 5.1]}],
        "avoid_exact_smiles": ["CCO", "CCC"],
        "n_evaluated": 2,
        "recent_diversity_alert": {"instruction": "Use a different parent."},
    }
    context = [
        {
            "smiles": "CCO",
            "history_role": "pareto_front",
            "proposal_lesson": "vary heteroatoms",
        }
    ]
    prompt = "\n".join(
        [
            "Task:",
            "Generate up to 2 valid, unique candidate SMILES. Use compact minified JSON.",
            "Target context:",
            "KRAS target context.",
            "Background:",
            "External scorers decide which candidates are evaluated.",
            "Molecule context table:",
            json.dumps(context),
            "How to use the molecule context:",
            "Use descriptors as evidence.",
            "Generation principles:",
            "- Avoid exact repeats.",
            "SMILES hygiene:",
            "Single-component organic SMILES only.",
            "Generation focus:",
            "history-guided local mutation.",
            "History summary:",
            json.dumps(history),
            "JSON output format:",
            '{"direct_smiles":[{"smiles":"CCN","rationale":"amine"}]}',
        ]
    )
    output = json.dumps(
        {
            "direct_smiles": [
                {"smiles": "CCN", "rationale": "amine neighbor"},
                {"smiles": "CCCl", "rationale": "halogen scan"},
            ]
        }
    )

    ir = smallmol_ir_from_prompt_response(prompt, output, round_idx=7, source_id="m1_strategy_0")

    assert ir is not None
    validate_ir_record(ir)
    assert ir["search_state"]["round"] == 7
    assert ir["search_state"]["num_evaluated"] == 2
    assert ir["search_state"]["do_not_repeat"] == ["CCO", "CCC"]
    assert ir["search_state"]["progress"]["stalled"] is True
    assert ir["action"]["payload"]["candidates"][0]["design"] == "CCN"
    assert ir["raw_context"]["source_id"] == "m1_strategy_0"


def test_parameter_edit_builder_allows_design_space_expansion():
    ir = make_parameter_edit_ir(
        task_id="nanogpt",
        domain="training_program",
        task_description="Search over train.py scalar assignments.",
        objectives=[
            {
                "name": "val_bpb",
                "direction": "minimize",
                "description": "Validation bits per byte.",
            }
        ],
        active_parameters=[
            {
                "name": "HEAD_DIM",
                "type": "choice",
                "domain": [64, 96, 128],
                "edit_op": "set_choice",
                "current_value": 128,
            }
        ],
        inactive_parameters=[
            {
                "name": "WARMDOWN_RATIO",
                "type": "float",
                "domain": [0.0, 0.95],
                "current_value": 0.5,
            }
        ],
        action={
            "type": "expand_design_space",
            "reasoning": "Active edits have plateaued.",
            "payload": {"activate": "WARMDOWN_RATIO", "initial_value": 0.5},
            "summary": "Activate warmdown schedule control.",
        },
        request_description="Choose one valid action.",
        max_edits_per_candidate=2,
    )

    validate_ir_record(ir)
    assert "expand_design_space" in ir["request"]["allowed_actions"]
    rendered = render_record(ir)
    assert "Inactive parameters" in rendered["instruction"]
    assert "WARMDOWN_RATIO" in rendered["output"]
