from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.transport import CallableProposalClient

from tasks.llm_kv_adaptive_quantization.core import evaluator as evaluator_module
from tasks.llm_kv_adaptive_quantization.core import workflow
from tasks.llm_kv_adaptive_quantization.core.candidate import (
    QuantizerCandidateDomain,
    parse_proposal_response,
    validate_candidate_source,
)
from tasks.llm_kv_adaptive_quantization.core.evaluator import (
    MLSBenchEvaluator,
    OFFICIAL_COMMIT,
    OFFICIAL_FIXED_HARNESS_SHA256,
    fixed_harness_sha256,
    parse_test_metrics,
    replace_quantizer_class,
)
from tasks.llm_kv_adaptive_quantization.core.proposals import (
    DeterministicQuantizerExpander,
    EndpointQuantizerExpander,
    parse_quantizer_specs,
)
from tasks.llm_kv_adaptive_quantization.core.surrogate import (
    FEATURE_DIMENSION,
    FEATURE_VERSION,
    QuantizerSourceEncoder,
)
from tasks.llm_kv_adaptive_quantization.ldm_task.procedure import main, parse_args


TASK_ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = TASK_ROOT / "resources" / "seed_quantizer.py"
SEED = SEED_PATH.read_text(encoding="utf-8")


def test_candidate_contract_and_collection_boundary(tmp_path: Path) -> None:
    sink = DataCollectionSink(tmp_path / "collection")
    domain = QuantizerCandidateDomain(sink)
    admitted = domain.admit(
        RawProposal({"code": SEED}, "test_model", {"collectable": True})
    )
    assert admitted.candidate_id.startswith("quantizer-")
    uncollected = domain.admit(RawProposal({"code": SEED}, "official_seed"))
    assert uncollected.candidate_id == admitted.candidate_id
    rejected = domain.admit(
        RawProposal("import os", "test_model", {"collectable": True})
    )
    assert rejected.reason == "invalid_quantizer"

    ir = json.loads((tmp_path / "collection" / "ldm_ir.jsonl").read_text())
    sft = json.loads((tmp_path / "collection" / "ldm_sft.jsonl").read_text())
    assert ir["schema_version"] == "ldm-2.0"
    assert ir["collection"]["provenance"]["candidate_id"] == admitted.candidate_id
    assert "collection" not in sft["instruction"]
    assert len((tmp_path / "collection" / "ldm_ir.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda text: text.replace("def reset_request(self, request_meta: dict, budget_state: dict):", "def reset_request(self, request_meta):"), "must have signature"),
        (lambda text: text.replace("def needs_prefill_qkv_observer(self)", "async def needs_prefill_qkv_observer(self)"), "missing required method"),
        (lambda text: text.replace("return False", "return self.__dict__", 1), "dunder"),
        (lambda text: text.replace("return False", "return eval('False')", 1), "may not call eval"),
        (lambda text: text.replace("return False", "return __import__('os')", 1), "may not call __import__"),
        (lambda text: text.replace("def reset_request(self, request_meta: dict, budget_state: dict):", "def reset_request(self, request_meta: dict, budget_state: dict = None):"), "must have signature"),
        (lambda text: text.replace("def query_observation_position(self)", "def query_observation_position(self, /)"), "must have signature"),
        (lambda text: text + "\nyield 1\n", "exactly one top-level class"),
    ],
)
def test_candidate_rejects_unsafe_or_inexact_source(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_candidate_source(mutator(SEED))


def test_response_and_metric_parsers() -> None:
    assert parse_proposal_response(json.dumps({"code": SEED}))[0]["code"] == SEED
    metrics = parse_test_metrics(
        "TEST_METRICS: final_score=72.5 effective_kv_bits=4 "
        "kv_compression_ratio=4 runtime_seconds=1.25"
    )
    assert metrics["final_score"] == 72.5
    assert metrics["kv_compression_ratio"] == 4.0


def test_deterministic_reservoir_is_distinct_and_gp_scores_every_candidate() -> None:
    proposals = DeterministicQuantizerExpander(SEED).expand(
        ExpansionRequest(round_idx=0, reservoir_size=4)
    ).proposals
    domain = QuantizerCandidateDomain()
    candidates = [domain.admit(item) for item in proposals]
    assert len({item.canonical_key for item in candidates}) == 4

    encoder = QuantizerSourceEncoder()
    assert encoder.describe().dimension == FEATURE_DIMENSION
    assert encoder.describe().version == FEATURE_VERSION
    vectors = {item.candidate_id: encoder.encode(item) for item in candidates}
    assert all(len(item.values) == FEATURE_DIMENSION for item in vectors.values())
    selection = RBFGPUCBSelector(
        objective_name="selection_score", feature_version=FEATURE_VERSION
    ).select(candidates, vectors, count=1)
    assert len(selection.selected_candidate_ids) == 1
    assert {item.candidate_id for item in selection.predictions} == {
        item.candidate_id for item in candidates
    }


def test_endpoint_specs_materialize_four_valid_distinct_candidates() -> None:
    specs = [
        {
            "bit_cap": 4,
            "key_group_size": 32,
            "value_group_size": 64,
            "residual_length": 128,
        }
    ] * 4
    response = json.dumps({"candidates": specs})
    assert len(parse_quantizer_specs(response, expected_count=4)) == 4
    expander = EndpointQuantizerExpander(
        CallableProposalClient(lambda request: response),
        SEED,
    )
    proposals = expander.expand(
        ExpansionRequest(round_idx=0, reservoir_size=4)
    ).proposals
    candidates = [QuantizerCandidateDomain().admit(item) for item in proposals]
    assert len({item.canonical_key for item in candidates}) == 4
    assert all(item.metadata["proposal_spec"] for item in candidates)
    assert sum(item.metadata["proposal_repaired"] for item in candidates) == 3


def test_harness_replacement_preserves_fixed_region_and_official_digest() -> None:
    harness = "before = 1\nclass AdaptiveKVQuantizer:\n    pass\nafter = 2\n"
    original_fixed = fixed_harness_sha256(harness)
    replaced = replace_quantizer_class(harness, SEED)
    assert replaced.startswith("before = 1\nclass AdaptiveKVQuantizer:")
    assert replaced.rstrip().endswith("after = 2")
    assert fixed_harness_sha256(replaced) == original_fixed
    contract = json.loads(
        (TASK_ROOT / "resources" / "upstream_contract.json").read_text()
    )
    assert contract["source_commit"] == OFFICIAL_COMMIT
    assert contract["sha256"]["fixed_harness"] == OFFICIAL_FIXED_HARNESS_SHA256
    assert contract["sha256"]["seed_class"] == _sha256(SEED_PATH)


def test_fake_benchmark_harness_runs_in_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "transformers-kv-lab"
    package.mkdir()
    (package / "src").mkdir()
    harness = package / "custom_quant_eval.py"
    harness.write_text(
        "from __future__ import annotations\n"
        "import argparse\n"
        "class AdaptiveKVQuantizer:\n    pass\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--workload')\n"
        "parser.add_argument('--budget-bits')\n"
        "parser.add_argument('--seed')\n"
        "parser.add_argument('--model-id')\n"
        "parser.add_argument('--max-examples')\n"
        "parser.add_argument('--cpu', action='store_true')\n"
        "parser.parse_args()\n"
        "print('TEST_METRICS: final_score=80 effective_kv_bits=4 kv_compression_ratio=4 runtime_seconds=0.1')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluator_module,
        "OFFICIAL_FIXED_HARNESS_SHA256",
        fixed_harness_sha256(harness.read_text(encoding="utf-8")),
    )
    candidate = Candidate("seed", {"code": SEED}, "seed-key", "test")
    result = MLSBenchEvaluator(
        package_dir=package,
        upstream_root=tmp_path / "upstream",
        run_dir=tmp_path / "run",
        workloads=("longbench_hotpotqa",),
        devices=(),
        model_id="fake/model",
        max_examples=1,
        timeout_seconds=30,
        cpu=True,
        evaluator_python=sys.executable,
    ).evaluate(candidate)
    assert result.succeeded, result.error
    assert result.metrics["selection_score"] > 0
    assert result.resource_usage["benchmark_jobs"] == 1
    assert Path(result.artifacts["evaluation_manifest"]).is_file()


def test_evaluation_environment_preserves_runtime_library_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/runtime/lib")
    environment = evaluator_module._evaluation_environment(
        package_dir=tmp_path,
        task_data=tmp_path / "task",
        output_dir=tmp_path / "output",
        device="0",
    )
    assert environment["LD_LIBRARY_PATH"] == "/runtime/lib"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_mock_procedure_writes_engine_artifacts_and_exact_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    assert parse_args(["--mock"]).mock
    assert main([
        "--mock",
        "--iterations",
        "1",
        "--reservoir-size",
        "4",
        "--out-dir",
        str(tmp_path),
        "--run-name",
        "mock_run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    run_dir = Path(output["run_dir"])
    for name in (
        "budget.json",
        "campaign.json",
        "checkpoint.json",
        "events.jsonl",
        "status.json",
        "summary.json",
        "ldm_task_spec.json",
        "search_manifest.json",
        "selection_record.json",
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "ldm_data" / "ldm_ir.jsonl").is_file()
    assert output["engine_summary"]["successful_evaluation_count"] == 1
    counters = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert counters == {
        "benchmark_jobs": 1,
        "expensive_evaluation_attempts": 1,
        "external_evaluations": 1,
        "outer_iterations": 1,
        "selected_candidates": 1,
        "successful_evaluations": 1,
        "valid_search_candidates": 4,
    }
    selection = json.loads((run_dir / "selection_record.json").read_text())
    predictions = selection["selections"][0]["payload"]["predictions"]
    assert len(predictions) == 4


def test_resume_does_not_repeat_completed_evaluation(tmp_path: Path, capsys) -> None:
    argv = [
        "--mock",
        "--iterations",
        "1",
        "--reservoir-size",
        "4",
        "--out-dir",
        str(tmp_path),
        "--run-name",
        "resume",
    ]
    assert main(argv) == 0
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    before = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert main(argv + ["--resume-from", str(run_dir)]) == 0
    capsys.readouterr()
    after = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert after == before
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert sum(item["event_type"] == "candidate_evaluated" for item in events) == 1


def test_endpoint_preflight_failure_pauses_resumably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    class BrokenClient:
        def __init__(self, **kwargs):
            assert kwargs["extra_body"]["chat_template_kwargs"] == {
                "enable_thinking": False
            }
            schema = kwargs["extra_body"]["response_format"]["json_schema"]["schema"]
            assert schema["properties"]["candidates"]["minItems"] == 4

        def preflight(self):
            raise RuntimeError("endpoint offline")

    monkeypatch.setattr(workflow, "OpenAICompatibleProposalClient", BrokenClient)
    code = main([
        "--mock",
        "--proposal-mode",
        "openai",
        "--llm-url",
        "http://localhost:8000/v1",
        "--llm-model-name",
        "test-model",
        "--out-dir",
        str(tmp_path),
        "--run-name",
        "paused",
    ])
    output = json.loads(capsys.readouterr().out)
    status = json.loads((Path(output["run_dir"]) / "status.json").read_text())
    assert code == 2
    assert status["status"] == "paused_endpoint_unavailable"
    assert status["budget"]["counters"] == {}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
