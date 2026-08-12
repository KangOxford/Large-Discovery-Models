from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_tts.engine.run_store import BudgetExceededError, BudgetLedger, CampaignStatus
from ldm_tts.transport.openai import (
    EndpointCircuitBreaker,
    EndpointCircuitOpen,
    EndpointRequestError,
    call_with_circuit_breaker,
)
from ldm_tts.registration.experiment import (
    ExperimentContractError,
    load_experiment_contract,
    snapshot_experiment_contract,
    validate_profile_args,
)
from ldm_tts.cli.runner import build_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTEIN_CONTRACT = REPO_ROOT / "tasks" / "protein_inverse_folding" / "experiment.json"


def test_qualified_contract_exposes_metric_roles_and_pinned_source() -> None:
    contract = load_experiment_contract(PROTEIN_CONTRACT)

    assert contract.task_id == "protein_inverse_folding"
    assert contract.qualification == "qualified"
    assert contract.benchmark["source_commit"] == "da06dffcc79826dc3d22dec53ead310c430b6535"
    assert [item["name"] for item in contract.metrics["reported"]] == ["aggregate_score"]
    assert [item["name"] for item in contract.metrics["optimized"]] == ["search_score"]
    assert contract.profile("gp_ucb_n4h4_20").budget["expensive_evaluation_attempts"] == 20
    assert contract.profile("official_benchmark").locked_args["iterations"] == 10
    assert contract.profile("official_benchmark").locked_args["breadth"] == 2


@pytest.mark.parametrize(
    ("invalid_field", "error_pattern"),
    [
        ("source_commit", r"must pin benchmark\.source_commit"),
        ("per_candidate_limits", r"must define evaluation\.per_candidate_limits"),
        ("profiles", "must define at least one campaign profile"),
    ],
)
def test_qualified_contract_requires_qualification_evidence(
    tmp_path: Path,
    invalid_field: str,
    error_pattern: str,
) -> None:
    payload = json.loads(PROTEIN_CONTRACT.read_text(encoding="utf-8"))
    if invalid_field == "source_commit":
        payload["benchmark"]["source_commit"] = "unqualified"
    elif invalid_field == "per_candidate_limits":
        payload["evaluation"]["per_candidate_limits"] = {}
    else:
        payload["profiles"] = {}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match=error_pattern):
        load_experiment_contract(path)


def test_contract_profile_rejects_training_budget_drift() -> None:
    contract = load_experiment_contract(PROTEIN_CONTRACT)
    profile = contract.profile("gp_ucb_n4h4_20")
    args = dict(profile.locked_args)
    args["epochs"] = 99

    with pytest.raises(ExperimentContractError, match="--epochs=99, expected 100"):
        validate_profile_args(contract, profile.name, args)


def test_runner_enforces_selected_contract_profile() -> None:
    contract = load_experiment_contract(PROTEIN_CONTRACT)
    profile = contract.profile("gp_ucb_n4h4_20")
    config = {
        "name": "contract_test",
        "task": "protein_inverse_folding",
        "algorithm": "gp_ucb",
        "mode": "real",
        "contract_profile": profile.name,
        "args": dict(profile.locked_args),
    }
    config_path = REPO_ROOT / "config" / "protein_inverse_folding" / "real_gp_ucb_n4h4_20.yaml"

    plan = build_plan(config, config_path)
    assert plan["contract_profile"] == profile.name
    assert plan["contract_sha256"] == contract.digest

    config["args"]["epochs"] = 10
    with pytest.raises(SystemExit, match="violates experiment contract profile"):
        build_plan(config, config_path)


def test_contract_snapshot_records_digest_and_profile(tmp_path: Path) -> None:
    contract = load_experiment_contract(PROTEIN_CONTRACT)
    destination = snapshot_experiment_contract(
        contract, tmp_path, profile="gp_ucb_n4h4_20"
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["snapshot"]["sha256"] == contract.digest
    assert payload["snapshot"]["profile"] == "gp_ucb_n4h4_20"


def test_budget_ledger_persists_and_prevents_overflow(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(
        limits={"expensive_evaluation_attempts": 2},
        path=path,
    )
    ledger.consume("expensive_evaluation_attempts")
    ledger.consume("expensive_evaluation_attempts")

    with pytest.raises(BudgetExceededError, match="would be exceeded"):
        ledger.consume("expensive_evaluation_attempts")

    restored = BudgetLedger.load(path)
    assert restored.counters["expensive_evaluation_attempts"] == 2
    assert restored.remaining("expensive_evaluation_attempts") == 0


def test_campaign_status_embeds_current_budget(tmp_path: Path) -> None:
    ledger = BudgetLedger(limits={"outer_iterations": 3})
    ledger.consume("outer_iterations")
    writer = CampaignStatus(tmp_path / "status.json", "task", "run")
    writer.update("running", phase="evaluation", iteration=1, budget=ledger)

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["phase"] == "evaluation"
    assert payload["budget"]["remaining"]["outer_iterations"] == 2


def test_endpoint_circuit_opens_at_failure_threshold() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=2, recovery_timeout_seconds=60)
    calls = 0

    def unavailable() -> None:
        nonlocal calls
        calls += 1
        raise EndpointRequestError("upstream timeout")

    with pytest.raises(EndpointRequestError, match="upstream timeout"):
        call_with_circuit_breaker(breaker, unavailable)
    with pytest.raises(EndpointCircuitOpen, match="opened after 2 failures"):
        call_with_circuit_breaker(breaker, unavailable)
    with pytest.raises(EndpointCircuitOpen, match="circuit is open"):
        call_with_circuit_breaker(breaker, unavailable)

    assert calls == 2
    assert breaker.snapshot()["state"] == "open"


def test_endpoint_circuit_recovers_from_zero_timestamp() -> None:
    breaker = EndpointCircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=10,
    )
    breaker.record_failure(EndpointRequestError("unavailable"), now=0)

    with pytest.raises(EndpointCircuitOpen, match="circuit is open"):
        breaker.before_request(now=9)
    breaker.before_request(now=10)

    assert breaker.state == "half_open"
