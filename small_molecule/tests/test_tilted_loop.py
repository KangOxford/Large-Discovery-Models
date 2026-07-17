import pytest
import json
from types import SimpleNamespace

import numpy as np

from strbo_v1.gp import GPConfig
import strbo_v1.ldm_tilted_case2.loop as loop_mod
from strbo_v1.ldm_tilted_case2.candidate_record import (
    CandidateRecord,
    ReservoirBuildResult,
)
from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config
from strbo_v1.ldm_tilted_case2.loop import _score_smiles, run_tilted_case2_search
from strbo_v1.llm_advisor.client import MockLLMClient
from strbo_v1.rng import RNG


def test_config_validation():
    TiltedLDMCase2Config(method="m1_direct_llm_sir")

    with pytest.raises(ValueError, match="method"):
        TiltedLDMCase2Config(method="unknown")
    with pytest.raises(ValueError, match="batch_size"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", batch_size=0)
    with pytest.raises(ValueError, match="budget"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", init_size=5, budget=4)
    with pytest.raises(ValueError, match="eta"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", eta_ehvi_tilt=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", alpha_base_measure=-1.0)
    with pytest.raises(ValueError, match="minimize"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", minimize=(True,))
    with pytest.raises(ValueError, match="ref_point"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", ref_point=(0.0,))
    with pytest.raises(ValueError, match="max_empty_reservoir_rounds"):
        TiltedLDMCase2Config(method="m1_direct_llm_sir", max_empty_reservoir_rounds=0)


def mock_scorer_vina(smiles_list):
    return [-float(len(smiles)) / 10.0 for smiles in smiles_list]


def mock_scorer_activity(smiles_list):
    return [5.0 + float(smiles.count("N")) for smiles in smiles_list]


def mock_analog_fn(seeds):
    out = []
    for seed in seeds:
        out.extend([seed + "C", seed + "N"])
    return out


def m1_llm():
    return MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "x"}, {"smiles": "CCCN", "rationale": "y"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "x"}, {"smiles": "CCCCO", "rationale": "y"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCNCC", "rationale": "x"}, {"smiles": "CCOCC", "rationale": "y"}]}),
    ])


def run_method(method, llm, tmp_path, **kwargs):
    cfg = TiltedLDMCase2Config(
        method=method,
        init_size=3,
        budget=6,
        m1_k_direct_llm=2,
        trajectory_dir=str(tmp_path),
        ehvi_n_samples=8,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
        **kwargs,
    )
    return run_tilted_case2_search(
        ["CCO", "CCN", "CCC"],
        (mock_scorer_vina, mock_scorer_activity),
        mock_analog_fn,
        config=cfg,
        llm=llm,
    )


def test_loop_runs_m1_mock_two_objectives(tmp_path):
    history, trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    assert len(history) == 6
    assert len(history[0][1]) == 2
    assert trace is not None


def test_loop_does_not_collapse_multi_objective_scores(tmp_path):
    history, _trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    assert all(isinstance(scores, tuple) and len(scores) == 2 for _smiles, scores in history)


def test_loop_eta_zero_base_only(tmp_path):
    history, _trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path, eta_ehvi_tilt=0.0)
    assert len(history) == 6


def test_loop_alpha_zero_ehvi_only(tmp_path):
    history, _trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path, alpha_base_measure=0.0)
    assert len(history) == 6


def test_llm_cold_start_does_not_score_seed_smiles(tmp_path):
    client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "cold"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "cold"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCO", "rationale": "cold"}]}),
    ])
    scored_batches = []

    def vina(smiles_list):
        batch = list(smiles_list)
        scored_batches.append(batch)
        return [-float(len(smiles)) / 10.0 for smiles in batch]

    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=3,
        init_strategy="llm_cold_start",
        budget=3,
        batch_size=1,
        m1_k_direct_llm=1,
        trajectory_dir=str(tmp_path),
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )

    history, summary = run_tilted_case2_search(
        ["CCO", "CCN", "CCC"],
        (vina, mock_scorer_activity),
        mock_analog_fn,
        config=cfg,
        llm=client,
    )

    assert [smiles for smiles, _scores in history] == ["CCCC", "CCCCN", "CCCCO"]
    assert ["CCO", "CCN", "CCC"] not in scored_batches
    first_prompt = client.call_log[0]["user"]
    assert '"n_evaluated": 0' in first_prompt
    assert summary["history_size"] == 3


def test_seed_initialization_strategy_preserves_seed_history(tmp_path):
    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=2,
        init_strategy="seed_smiles",
        budget=2,
        trajectory_dir=str(tmp_path),
    )

    history, _summary = run_tilted_case2_search(
        ["CCO", "CCN", "CCC"],
        (mock_scorer_vina, mock_scorer_activity),
        mock_analog_fn,
        config=cfg,
        llm=m1_llm(),
    )

    assert [smiles for smiles, _scores in history] == ["CCO", "CCN"]


def test_resume_from_rounds_jsonl_continues_partial_cold_start(tmp_path):
    first_client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "first"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "second"}]}),
    ])
    first_cfg = TiltedLDMCase2Config(
        "m1_llm_one_step",
        init_size=1,
        init_strategy="llm_cold_start",
        budget=2,
        batch_size=1,
        m1_k_direct_llm=1,
        trajectory_dir=str(tmp_path),
    )
    first_history, _summary = run_tilted_case2_search(
        ["CCO"],
        (mock_scorer_vina, mock_scorer_activity),
        mock_analog_fn,
        config=first_cfg,
        llm=first_client,
    )
    assert [smiles for smiles, _scores in first_history] == ["CCCC", "CCCCN"]

    (tmp_path / "history.json").unlink()
    (tmp_path / "summary.json").unlink()

    resume_client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCCCO", "rationale": "third"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCNCC", "rationale": "fourth"}]}),
    ])
    resume_cfg = TiltedLDMCase2Config(
        "m1_llm_one_step",
        init_size=1,
        init_strategy="llm_cold_start",
        resume_from_trajectory=True,
        budget=4,
        batch_size=1,
        m1_k_direct_llm=1,
        trajectory_dir=str(tmp_path),
    )

    resumed_history, summary = run_tilted_case2_search(
        ["CCO"],
        (mock_scorer_vina, mock_scorer_activity),
        mock_analog_fn,
        config=resume_cfg,
        llm=resume_client,
    )

    assert [smiles for smiles, _scores in resumed_history] == [
        "CCCC",
        "CCCCN",
        "CCCCO",
        "CCNCC",
    ]
    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert [record["round_idx"] for record in rounds] == [0, 1, 2, 3]
    assert summary["history_size"] == 4
    assert summary["llm_call_count"] == 4


def test_fresh_run_resets_existing_rounds_jsonl(tmp_path):
    (tmp_path / "rounds.jsonl").write_text(
        json.dumps({"round_idx": 999, "stale": True}) + "\n",
        encoding="utf-8",
    )

    _history, _summary = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)

    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert rounds
    assert all(not record.get("stale") for record in rounds)
    assert rounds[0]["round_idx"] == 0


def test_loop_retries_transient_empty_reservoir(tmp_path):
    llm = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": []}),
        json.dumps({"direct_smiles": [{"smiles": "CCCC", "rationale": "x"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCCCN", "rationale": "y"}]}),
    ])
    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=3,
        budget=5,
        m1_k_direct_llm=1,
        trajectory_dir=str(tmp_path),
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    history, trace = run_tilted_case2_search(
        ["CCO", "CCN", "CCC"],
        (mock_scorer_vina, mock_scorer_activity),
        mock_analog_fn,
        config=cfg,
        llm=llm,
    )
    assert len(history) == 5
    assert trace["early_stop_reason"] is None


def test_loop_stops_after_empty_reservoir_limit(tmp_path):
    llm = MockLLMClient(
        scripted_responses=[json.dumps({"direct_smiles": []})] * 6
    )
    cfg = TiltedLDMCase2Config(
        "m1_direct_llm_sir",
        init_size=3,
        budget=5,
        m1_k_direct_llm=1,
        max_empty_reservoir_rounds=2,
        trajectory_dir=str(tmp_path),
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )
    history, trace = run_tilted_case2_search(
        ["CCO", "CCN", "CCC"],
        (mock_scorer_vina, mock_scorer_activity),
        mock_analog_fn,
        config=cfg,
        llm=llm,
    )
    assert len(history) == 3
    assert trace["early_stop_reason"] == "empty_reservoir_limit"


def test_trace_jsonl_contains_required_fields(tmp_path):
    _history, _trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    line = (tmp_path / "rounds.jsonl").read_text().splitlines()[0]
    record = json.loads(line)
    assert "q0_entropy" in record
    assert "prob_effective_sample_size" in record
    assert "candidates" in record


def test_trace_selected_candidate_has_probability_and_scores(tmp_path):
    _history, _trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    selected = [c for r in rounds for c in r["candidates"] if c["selected"]]
    assert selected
    assert all(c["resampling_probability"] is not None for c in selected)
    assert all(c["true_scores"] is not None for c in selected)


def test_trace_round_records_raw_llm_inputs_outputs_and_results(tmp_path):
    _history, _trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    first = json.loads((tmp_path / "rounds.jsonl").read_text().splitlines()[0])

    attempt = first["llm_attempts"][0]
    assert attempt["system_prompt"]
    assert "Generate up to" in attempt["user_prompt"]
    assert attempt["raw_output"] == attempt["raw_text"]
    assert attempt["parsed_json"]["direct_smiles"]

    assert first["selection_results"]["selected_smiles"]
    assert first["selection_results"]["selected_scores"]
    assert first["selection_results"]["ehvi_fallback_reason"] is None


def test_trace_summary_counts_llm_calls(tmp_path):
    _history, _trace = run_method("m1_direct_llm_sir", m1_llm(), tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["llm_call_count"] > 0
    assert summary["final_hypervolume"] is not None


def test_score_smiles_retries_transient_nonfinite_values():
    class TransientVinaFailure:
        def __init__(self):
            self.calls = []

        def __call__(self, smiles_list):
            smiles = list(smiles_list)
            self.calls.append(smiles)
            if smiles == ["CCO", "CCN"]:
                return [float("nan"), -4.2]
            if smiles == ["CCO"]:
                return [-3.7]
            return [-4.2]

    vina = TransientVinaFailure()

    scores = _score_smiles(
        ["CCO", "CCN"],
        (vina, lambda smiles_list: [5.1, 5.2]),
    )

    assert scores == [(-3.7, 5.1), (-4.2, 5.2)]
    assert vina.calls == [["CCO", "CCN"], ["CCO"]]


def test_score_smiles_raises_scorer_exceptions():
    def broken_scorer(_smiles_list):
        raise RuntimeError("receptor prep failed")

    with pytest.raises(RuntimeError, match="broken_scorer failed.*receptor prep failed"):
        _score_smiles(["CCO"], (broken_scorer, lambda smiles_list: [5.1]))


def test_one_step_retries_when_selected_candidate_cannot_be_scored(tmp_path):
    client = MockLLMClient(scripted_responses=[
        json.dumps({"direct_smiles": [{"smiles": "CCN", "rationale": "docking fail"}]}),
        json.dumps({"direct_smiles": [{"smiles": "CCC", "rationale": "scorable"}]}),
    ])

    def vina(smiles_list):
        return [float("nan") if smiles == "CCN" else -3.3 for smiles in smiles_list]

    cfg = TiltedLDMCase2Config(
        "m1_llm_one_step",
        init_size=1,
        budget=2,
        batch_size=1,
        m1_k_direct_llm=1,
        trajectory_dir=str(tmp_path),
    )

    history, summary = run_tilted_case2_search(
        ["CCO"],
        scorer=(vina, lambda smiles: [6.0 for _ in smiles]),
        analog_fn=mock_analog_fn,
        config=cfg,
        llm=client,
    )

    assert [smiles for smiles, _scores in history] == ["CCO", "CCC"]
    assert len(client.call_log) == 2
    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert len(rounds) == 1
    assert rounds[0]["selection_results"]["selected_smiles"] == ["CCC"]
    assert rounds[0]["selection_results"]["selected_scores"] == [[-3.3, 6.0]]
    assert summary["history_size"] == 2


def test_bo_selection_retries_when_selected_candidate_cannot_be_scored(monkeypatch):
    bad = CandidateRecord(
        raw_smiles="CCN",
        canonical_smiles="CCN",
        method="test",
        sources=["s1"],
        occurrence_by_source={"s1": 1},
        q0_base_mass=1.0,
    )
    good = CandidateRecord(
        raw_smiles="CCC",
        canonical_smiles="CCC",
        method="test",
        sources=["s1"],
        occurrence_by_source={"s1": 1},
        q0_base_mass=0.1,
    )
    build_result = ReservoirBuildResult(candidates=[bad, good], sources=[])

    def fake_ehvi(_history, candidates, _cfg, _rng):
        for candidate in candidates:
            candidate.ehvi = 0.0
        return SimpleNamespace(ehvi=np.array([0.0, 0.0]), fallback_reason=None)

    def vina(smiles_list):
        return [float("nan") if smiles == "CCN" else -3.3 for smiles in smiles_list]

    monkeypatch.setattr(loop_mod, "compute_ehvi_for_candidates", fake_ehvi)
    monkeypatch.setattr(loop_mod, "gumbel_top_k", lambda _prob, _k, _rng: [0])

    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_oversample_sir",
        batch_size=1,
        gp_config=GPConfig(device="cpu", fit_n_itersteps=2, fp_n_bits=128),
    )

    selected = loop_mod._select_and_score(
        build_result,
        [("CCO", (-3.0, 6.0))],
        (vina, lambda smiles: [6.0 for _ in smiles]),
        cfg,
        RNG(0),
    )

    assert [candidate.canonical_smiles for candidate in selected] == ["CCC"]
    assert bad.selected is False
    assert bad.true_scores is None
    assert bad.metadata["selection_failure_scores"] == [None, 6.0]
    assert good.selected is True
    assert good.true_scores == [-3.3, 6.0]
    assert build_result.metadata["selection_failed_evaluations"] == [
        {"smiles": "CCN", "scores": [None, 6.0]}
    ]
