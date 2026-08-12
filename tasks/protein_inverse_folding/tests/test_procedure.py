from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_tts.data import validate_ir_record
from tasks.protein_inverse_folding.core import workflow
from tasks.protein_inverse_folding.core.candidate import (
    CandidateValidationError,
    assemble_candidate,
    parse_model_response,
)
from tasks.protein_inverse_folding.core.evaluation import (
    aggregate_mls_score,
    continuous_search_score,
    evaluate_mock,
    parse_test_metrics,
)
from tasks.protein_inverse_folding.core.search import (
    FEATURE_VERSION,
    RBFGPSurrogate,
    SearchObservation,
    encode_candidate,
)
from tasks.protein_inverse_folding.ldm_task.procedure import (
    describe_ldm_task,
    main,
    parse_args,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = TASK_ROOT / "resources" / "seed_design.py"


def _proposal():
    payload = {
        "reasoning": "Keep a stable residual message-passing baseline.",
        "summary": "Validated seed.",
        "code": SEED_PATH.read_text(encoding="utf-8"),
    }
    return parse_model_response(json.dumps(payload))


def test_parse_args_and_description_match_runtime_contract() -> None:
    args = parse_args(["--mock", "--iterations", "0", "--breadth", "3"])
    assert args.iterations == 0
    assert args.generator == "mock"
    assert args.evaluator == "mock"
    spec = describe_ldm_task(args)
    assert spec.task == "protein_inverse_folding"
    assert spec.candidate_domain.kind == "structured_python_program"
    assert spec.reservoir.expansions[0].action_kind == "emit_candidate"
    assert spec.reservoir.deduplication_key == "canonical validated encoder source"
    assert spec.surrogate.kind == "none"
    assert spec.acquisition.objective_names == ("aggregate_score",)
    assert spec.proposal_search.breadth == 3


def test_model_response_accepts_seed_and_rejects_unsafe_import() -> None:
    proposal = _proposal()
    assert proposal.config_overrides == {}
    assert "class StructureEncoder" in proposal.code

    fenced = parse_model_response(
        json.dumps(
            {
                "reasoning": "Fence the code as many chat models do.",
                "summary": "Fenced seed.",
                "code": f"```python\n{proposal.code}```",
            }
        )
    )
    assert fenced.code == proposal.code

    unsafe = proposal.code.replace(
        '"""MLS-Bench starter design for the editable inverse-folding region."""',
        '"""unsafe"""\nimport os',
    )
    with pytest.raises(CandidateValidationError, match="import is not allowed"):
        parse_model_response(
            json.dumps({"reasoning": "", "summary": "", "code": unsafe})
        )


def test_assembly_changes_only_editable_region_and_overrides() -> None:
    scaffold = """FIXED_PREFIX = 1
# EDITABLE SECTION START - StructureEncoder + InverseFoldingModel
class OldEncoder:
    pass
# EDITABLE SECTION END
FIXED_MIDDLE = 2
def main():
    CONFIG_OVERRIDES = {}
    return CONFIG_OVERRIDES
FIXED_SUFFIX = 3
"""
    proposal = _proposal()
    assembled = assemble_candidate(scaffold, proposal)
    assert assembled.startswith("FIXED_PREFIX = 1\n")
    assert "class OldEncoder" not in assembled
    assert "class StructureEncoder" in assembled
    assert "FIXED_MIDDLE = 2\ndef main():\n    CONFIG_OVERRIDES = {}" in assembled
    assert assembled.endswith("    return CONFIG_OVERRIDES\nFIXED_SUFFIX = 3\n")


def test_metric_parser_and_public_score_composition() -> None:
    parsed = parse_test_metrics(
        "noise\nTEST_METRICS recovery=0.4\nTEST_METRICS perplexity=5.0\n",
        "CATH4.2",
    )
    assert parsed == {
        "recovery_CATH4.2": 0.4,
        "perplexity_CATH4.2": 5.0,
    }
    metrics = {
        f"recovery_{label}": 0.4 for label in ("CATH4.2", "CATH4.3", "TS50")
    }
    metrics.update(
        {
            f"perplexity_{label}": 5.0
            for label in ("CATH4.2", "CATH4.3", "TS50")
        }
    )
    assert aggregate_mls_score(metrics) == pytest.approx(0.2420237248)
    assert evaluate_mock(_proposal())["aggregate_score"] > 0


def test_continuous_search_score_retains_signal_below_public_floor() -> None:
    weak = {
        "recovery_CATH4.2": 0.3078,
        "perplexity_CATH4.2": 8.9845,
        "recovery_CATH4.3": 0.3137,
        "perplexity_CATH4.3": 8.7761,
        "recovery_TS50": 0.3227,
        "perplexity_TS50": 8.5511,
    }
    better = dict(weak)
    better["recovery_CATH4.2"] += 0.01
    better["perplexity_CATH4.2"] -= 0.1

    assert aggregate_mls_score(weak) == 0.0
    assert continuous_search_score(weak) < 0.0
    assert continuous_search_score(better) > continuous_search_score(weak)


def test_protein_features_are_versioned_and_gp_ucb_is_finite() -> None:
    proposal = _proposal()
    first = encode_candidate(
        proposal.code,
        config_overrides=proposal.config_overrides,
        parameter_count=370_452,
    )
    changed_code = workflow.replace_config_overrides(
        proposal.code, {"dropout": 0.2, "num_encoder_layers": 4}
    )
    changed = encode_candidate(
        changed_code,
        config_overrides={"dropout": 0.2, "num_encoder_layers": 4},
        parameter_count=500_000,
    )
    assert FEATURE_VERSION
    assert len(first) == len(changed)
    assert first != changed

    surrogate = RBFGPSurrogate(
        [
            SearchObservation("seed", first, -0.4),
            SearchObservation("changed", changed, -0.2),
        ]
    )
    prediction = surrogate.predict(changed, beta=1.0)
    assert surrogate.ready
    assert prediction.acquisition_score == pytest.approx(
        prediction.mean + prediction.std
    )


def test_mock_procedure_collects_valid_ir_without_private_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    collection_dir = tmp_path / "collected"
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(collection_dir))
    assert (
        main(
            [
                "--mock",
                "--iterations",
                "1",
                "--breadth",
                "1",
                "--out-dir",
                str(tmp_path / "runs"),
                "--run-name",
                "collection_test",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 1
    assert summary["evaluated"] == 1

    ir_rows = [
        json.loads(line)
        for line in (collection_dir / "ldm_ir.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(ir_rows) == 1
    validate_ir_record(ir_rows[0])
    assert ir_rows[0]["action"]["payload"]["candidates"][0]["code"]
    assert ir_rows[0]["collection"]["provenance"]["run_id"] == "collection_test"
    assert ir_rows[0]["collection"]["outcome"]["status"] == "evaluated"

    rendered = (collection_dir / "ldm_sft.jsonl").read_text(encoding="utf-8")
    assert "collection_test" not in rendered
    assert '"collection"' not in rendered
    assert '"outcome"' not in rendered


def test_mock_procedure_resumes_at_first_unrecorded_iteration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = [
        "--mock",
        "--breadth",
        "1",
        "--out-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "resume_test",
    ]
    assert main([*common, "--iterations", "1"]) == 0
    capsys.readouterr()
    incomplete_state = tmp_path / "runs" / "resume_test" / "states" / "i002_c001"
    incomplete_state.mkdir(parents=True)
    (incomplete_state / "prompt.txt").write_text("interrupted", encoding="utf-8")

    assert main([*common, "--iterations", "2", "--resume"]) == 0
    summary = json.loads(capsys.readouterr().out)
    manifest = [
        json.loads(line)
        for line in (tmp_path / "runs" / "resume_test" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [record["iteration"] for record in manifest] == [1, 2]
    assert summary["accepted"] == 2
    assert summary["evaluated"] == 2
    assert summary["failures"] == 0


def test_resume_does_not_count_search_error_as_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "runs" / "resume_errors"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "candidate_id": "i001_search_error",
                "status": "search_error",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--mock",
                "--iterations",
                "1",
                "--breadth",
                "1",
                "--out-dir",
                str(tmp_path / "runs"),
                "--run-name",
                "resume_errors",
                "--resume",
            ]
        )
        == 1
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 0
    assert summary["real_evaluation_attempts"] == 0


def test_mock_gp_ucb_n4h4_reranks_one_real_candidate_per_iteration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = [
        "--mock",
        "--method",
        "best_of_n",
        "--breadth",
        "2",
        "--depth",
        "2",
        "--score-key",
        "search_score",
        "--out-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "gp_search",
    ]
    assert main([*common, "--iterations", "1"]) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert first_summary["generated"] == 4
    assert first_summary["evaluated"] == 1
    assert first_summary["real_evaluation_attempts"] == 1

    assert main([*common, "--iterations", "2", "--resume"]) == 0
    summary = json.loads(capsys.readouterr().out)
    run_dir = tmp_path / "runs" / "gp_search"
    manifest = [
        json.loads(line)
        for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    search_manifest = [
        json.loads(line)
        for line in (run_dir / "search_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(manifest) == 2
    assert all(record["status"] == "evaluated" for record in manifest)
    assert len(search_manifest) == 8
    assert summary["evaluated"] == 2
    assert summary["real_evaluation_attempts"] == 2
    assert summary["gp"]["fit_status"] == "fitted"
    assert (run_dir / "iterations" / "i002" / "selection.json").is_file()


def test_openai_timeout_is_reported_as_generator_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parse_args(
        [
            "--generator",
            "openai",
            "--llm-url",
            "https://example.invalid/v1",
            "--llm-model-name",
            "test-model",
        ]
    )

    def raise_timeout(**_kwargs):
        raise workflow.EndpointRequestError("read timed out")

    monkeypatch.setattr(workflow, "request_openai_chat", raise_timeout)
    with pytest.raises(RuntimeError, match="LLM request failed: read timed out"):
        workflow._openai_response(args, "test prompt")


def test_endpoint_preflight_pauses_before_consuming_campaign_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(**_kwargs):
        raise workflow.EndpointRequestError("preflight timed out")

    monkeypatch.setattr(workflow, "preflight_openai_chat", unavailable)
    result = main(
        [
            "--generator",
            "openai",
            "--evaluator",
            "mock",
            "--llm-url",
            "https://example.invalid/v1",
            "--llm-model-name",
            "test-model",
            "--iterations",
            "1",
            "--out-dir",
            str(tmp_path),
            "--run-name",
            "preflight_pause",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "paused_endpoint_unavailable"
    assert payload["details"]["resumable"] is True
    assert payload["budget"]["counters"] == {}
    status = json.loads(
        (tmp_path / "preflight_pause" / "status.json").read_text(encoding="utf-8")
    )
    assert status["phase"] == "endpoint_preflight"
