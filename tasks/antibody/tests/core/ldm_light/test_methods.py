from __future__ import annotations

import pytest

from tasks.antibody.core.ldm_light.methods import METHOD_SPECS, normalize_method
from tasks.antibody.core.workflow import describe_ldm_task, parse_args


@pytest.mark.parametrize(
    ("publication_name", "canonical"),
    [
        ("Direct_Max", "direct_max"),
        ("Direct-Softmax", "direct_softmax"),
        ("Policy_Max", "policy_max"),
        ("Policy-Softmax", "policy_softmax"),
        ("Pure_LLM", "llm_gen"),
    ],
)
def test_publication_method_names_are_stable(publication_name, canonical):
    assert normalize_method(publication_name) == canonical


def test_task_spec_records_base_measure_and_reduction():
    args = parse_args([
        "--mock",
        "--antigen",
        "SMOKE",
        "--method",
        "Policy_Softmax",
        "--softmax-eta",
        "0.7",
    ])

    spec = describe_ldm_task(args, {"seq_len": 11}, ["SMOKE"])

    assert METHOD_SPECS[args.method]["base_measure"] == "policy"
    assert spec.acquisition.selection_rule == "softmax over policy reservoir acquisition scores"
    assert spec.acquisition.parameters["softmax_eta"] == pytest.approx(0.7)
    assert spec.proposal_search.name == "single_turn"
    assert spec.proposal_search.parameters["planner_mode"] == "choices"
