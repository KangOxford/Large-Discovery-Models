"""Search workflow for MLS-Bench protein inverse-folding encoder designs."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import BudgetExceededError, BudgetLedger, CampaignStatus
from ldm_tts.data import DataCollectionSink, make_complete_design_ir
from ldm_tts.transport.openai import (
    EndpointCircuitBreaker,
    EndpointCircuitOpen,
    EndpointRequestError,
    call_with_circuit_breaker,
    preflight_openai_chat,
    request_openai_chat,
)
from ldm_tts.registration.experiment import (
    ExperimentContract,
    load_active_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.optimization.gp import select_max_ucb
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from tasks.protein_inverse_folding.core.candidate import (
    CandidateProposal,
    CandidateValidationError,
    parse_model_response,
    replace_config_overrides,
    validate_candidate_code,
)
from tasks.protein_inverse_folding.core.evaluation import (
    BENCHMARKS,
    EvaluationError,
    continuous_search_score,
    evaluate_benchmarks,
    evaluate_gpu_smoke,
    evaluate_mock,
)
from tasks.protein_inverse_folding.core.search import (
    FEATURE_VERSION,
    GPPrediction,
    RBFGPSurrogate,
    SearchObservation,
    SURROGATE_DIMENSION,
    encode_candidate,
)


TASK_ID = "protein_inverse_folding"
TASK_DESCRIPTION = (
    "Design a GNN structure encoder that maps N/CA/C/O protein backbone "
    "coordinates to amino-acid log probabilities."
)


@dataclass(frozen=True)
class TTSCandidate:
    candidate_id: str
    proposal: CandidateProposal
    candidate_path: Path
    state_dir: Path
    feature_vector: tuple[float, ...]
    parameter_count: float | None
    prediction: GPPrediction
    branch: int
    depth: int


@dataclass
class CampaignState:
    parent_code: str
    observations: list[dict[str, Any]]
    gp_observations: list[SearchObservation]
    accepted_count: int = 0
    evaluated_count: int = 0
    real_attempt_count: int = 0
    generated_count: int = 0
    failure_count: int = 0
    best: dict[str, Any] | None = None
    budget: BudgetLedger | None = None
    status: CampaignStatus | None = None
    endpoint_breaker: EndpointCircuitBreaker | None = None
TASK_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = (
    {
        "name": "aggregate_score",
        "direction": "maximize",
        "description": (
            "Geometric mean across CATH4.2, CATH4.3, and TS50 of equally "
            "weighted baseline-calibrated recovery and perplexity terms."
        ),
    },
    {
        "name": "recovery",
        "direction": "maximize",
        "description": "Fraction of native amino acids recovered per benchmark.",
    },
    {
        "name": "perplexity",
        "direction": "minimize",
        "description": "Per-residue sequence perplexity per benchmark.",
    },
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search MLS-Bench protein inverse-folding encoder designs."
    )
    parser.add_argument("--mock", action="store_true", help="Use mock generation/evaluation.")
    parser.add_argument("--generator", choices=("mock", "openai"), default="mock")
    parser.add_argument(
        "--evaluator", choices=("mock", "gpu_smoke", "benchmark"), default="mock"
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--breadth", type=int, default=2)
    parser.add_argument(
        "--method", choices=("best_observed", "best_of_n"), default="best_observed"
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--select-from", choices=("all", "leaves"), default="all")
    parser.add_argument("--max-real-evaluations", type=int, default=0)
    parser.add_argument("--proposal-retries", type=int, default=2)
    parser.add_argument("--surrogate-mode", choices=("ucb",), default="ucb")
    parser.add_argument("--gp-beta", type=float, default=1.0)
    parser.add_argument("--gp-lengthscale", type=float, default=1.5)
    parser.add_argument("--gp-noise", type=float, default=1.0e-4)
    parser.add_argument("--gp-prior-mean", type=float, default=0.0)
    parser.add_argument("--gp-prior-std", type=float, default=0.25)
    parser.add_argument("--initial-observation", type=Path)
    parser.add_argument("--validate-tts-candidates", action="store_true")
    parser.add_argument("--tts-gpu-device", type=int)
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=TASK_ROOT / "resources" / "seed_design.py",
    )
    parser.add_argument("--out-dir", type=Path, default=TASK_ROOT / "runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-key", default="aggregate_score")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-url", default="")
    parser.add_argument("--llm-model-name", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument(
        "--endpoint-preflight", dest="endpoint_preflight", action="store_true", default=True
    )
    parser.add_argument(
        "--no-endpoint-preflight", dest="endpoint_preflight", action="store_false"
    )
    parser.add_argument("--endpoint-preflight-timeout", type=float, default=30.0)
    parser.add_argument("--endpoint-failure-threshold", type=int, default=3)
    parser.add_argument("--endpoint-recovery-timeout", type=float, default=300.0)
    parser.add_argument("--scaffold-path", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS, default=[])
    parser.add_argument("--gpu-device", action="append", type=int, default=[])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cath-max-train-hours", type=float, default=3.0)
    parser.add_argument("--ts-max-train-hours", type=float, default=6.5)
    parser.add_argument("--cath-job-timeout", type=int, default=4 * 60 * 60)
    parser.add_argument("--ts-job-timeout", type=int, default=8 * 60 * 60)
    parser.add_argument("--eval-timeout", type=int, default=8 * 60 * 60)
    parser.add_argument("--parallel-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 0:
        parser.error("--iterations must be non-negative")
    if args.breadth < 1:
        parser.error("--breadth must be at least 1")
    if args.depth < 1:
        parser.error("--depth must be at least 1")
    if args.max_real_evaluations < 0:
        parser.error("--max-real-evaluations must be non-negative")
    if args.proposal_retries < 0:
        parser.error("--proposal-retries must be non-negative")
    if args.endpoint_preflight_timeout <= 0:
        parser.error("--endpoint-preflight-timeout must be positive")
    if args.endpoint_failure_threshold < 1:
        parser.error("--endpoint-failure-threshold must be at least 1")
    if args.endpoint_recovery_timeout < 0:
        parser.error("--endpoint-recovery-timeout must be non-negative")
    if args.validate_tts_candidates and args.tts_gpu_device is None:
        parser.error("--validate-tts-candidates requires --tts-gpu-device")
    if args.resume and not args.run_name.strip():
        parser.error("--resume requires --run-name")
    if args.mock:
        args.generator = "mock"
        args.evaluator = "mock"
    if not args.benchmark:
        args.benchmark = list(BENCHMARKS)
    return args


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="inverse_folding_encoder_source",
            kind="structured_python_program",
            dimension=None,
            representation=(
                "Complete Python for the editable StructureEncoder and "
                "InverseFoldingModel region, plus literal CONFIG_OVERRIDES."
            ),
            constraints={
                "required_classes": ["StructureEncoder", "InverseFoldingModel"],
                "input_shape": "X=(B,L,4,3), mask=(B,L)",
                "output_shape": "log_probs=(B,L,20)",
                "allowed_overrides": [
                    "learning_rate",
                    "dropout",
                    "num_encoder_layers",
                    "batch_size",
                ],
                "fixed_scaffold_preserved": True,
                "parameter_budget": 4_491_989,
            },
            metadata={
                "source": (
                    "https://github.com/Imbernoulli/MLS-Bench/tree/main/tasks/"
                    "ai4bio-protein-inverse-folding"
                )
            },
        ),
        objectives=tuple(
            ObjectiveSpec(
                name=item["name"],
                direction=item["direction"],
                description=item["description"],
            )
            for item in OBJECTIVES
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="encoder_code_proposal",
                output_kind="json",
                schema={
                    "type": "object",
                    "required": ["reasoning", "summary", "code"],
                    "properties": {
                        "reasoning": {"type": "string"},
                        "summary": {"type": "string"},
                        "code": {"type": "string"},
                    },
                },
                parser=(
                    "JSON parse, Python AST parse, required-class/interface checks, "
                    "import restrictions, and CONFIG_OVERRIDES validation"
                ),
                description="Return one complete editable-region design as JSON.",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="gp_ucb" if args.method == "best_of_n" else "best_observed",
            objective_names=(args.score_key,),
            score_direction="maximize",
            selection_rule=(
                "Score the complete valid TTS reservoir with an RBF GP upper "
                "confidence bound and evaluate only the highest-ranked candidate."
                if args.method == "best_of_n"
                else "Evaluate every accepted proposal and retain the highest finite observed score."
            ),
            parameters={
                "breadth": args.breadth,
                "depth": args.depth,
                "beta": args.gp_beta,
                "feature_version": FEATURE_VERSION,
                "select_from": args.select_from,
            },
        ),
        reservoir=ReservoirSpec(
            name="encoder_candidate_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_encoder_proposal",
                    action_kind="emit_candidate",
                    response_space="encoder_code_proposal",
                    produces_candidates=True,
                    description="Emit one complete editable-region encoder candidate.",
                ),
            ),
            candidate_validator=(
                "Python AST, required-class/interface, import, scaffold, tensor, "
                "and parameter-budget validation"
            ),
            deduplication_key="canonical validated encoder source",
            max_size=(
                args.breadth * args.depth
                if args.method == "best_of_n"
                else args.breadth
            ),
        ),
        surrogate=SurrogateSpaceSpec(
            kind="vector" if args.method == "best_of_n" else "none",
            representation=(
                "normalized configuration, AST structure, module/call counts, "
                "parameter count, and hashed source tokens"
                if args.method == "best_of_n"
                else "not used by best-observed selection"
            ),
            dimension_policy="fixed" if args.method == "best_of_n" else "none",
            dimension=SURROGATE_DIMENSION if args.method == "best_of_n" else None,
            encoder=(
                "tasks.protein_inverse_folding.core.search.encode_candidate"
                if args.method == "best_of_n"
                else ""
            ),
            version=FEATURE_VERSION if args.method == "best_of_n" else "",
        ),
        proposal_search=ProposalSearchSpec(
            name=args.method,
            breadth=args.breadth,
            depth=args.depth,
            beam_width=1,
            evaluation_policy=(
                "one_gp_ucb_selected_candidate_per_outer_iteration"
                if args.method == "best_of_n"
                else "every_accepted_candidate"
            ),
        ),
        metadata={
            "generator": args.generator,
            "evaluator": args.evaluator,
            "mock": bool(args.mock),
            "benchmarks": list(args.benchmark),
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_path = _resolve_path(args.seed_file)
    seed_code = seed_path.read_text(encoding="utf-8")
    validate_candidate_code(seed_code)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "task": TASK_ID,
                    "seed_file": str(seed_path),
                    "ldm_task_spec": describe_ldm_task(args).to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    run_name = args.run_name.strip() or time.strftime("protein_if_%Y%m%d_%H%M%S")
    out_dir = _resolve_path(args.out_dir)
    run_dir = out_dir / run_name if args.resume else _unique_run_dir(out_dir, run_name)
    if args.resume and not run_dir.is_dir():
        raise RuntimeError(f"resume run directory does not exist: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    manifest_path = run_dir / "manifest.jsonl"

    contract, contract_profile = load_active_experiment_contract()
    if contract is not None:
        if contract.task_id != TASK_ID:
            raise RuntimeError(
                f"active experiment contract is for {contract.task_id!r}, expected {TASK_ID!r}"
            )
        snapshot_experiment_contract(contract, run_dir, profile=contract_profile)
    budget = _make_budget_ledger(
        args,
        run_dir=run_dir,
        contract=contract,
        contract_profile=contract_profile,
        resume=args.resume,
    )
    status = CampaignStatus(
        path=run_dir / "status.json",
        task=TASK_ID,
        run_id=run_dir.name,
        contract_sha256="" if contract is None else contract.digest,
        contract_profile=contract_profile,
    )
    breaker = EndpointCircuitBreaker(
        failure_threshold=args.endpoint_failure_threshold,
        recovery_timeout_seconds=args.endpoint_recovery_timeout,
    )
    setattr(args, "_endpoint_breaker", breaker)

    state = CampaignState(
        parent_code=seed_code,
        observations=[],
        gp_observations=[],
        budget=budget,
        status=status,
        endpoint_breaker=breaker,
    )
    start_iteration = 1
    status.update("running", phase="initializing", budget=budget)

    if args.generator == "openai" and args.endpoint_preflight:
        status.update("running", phase="endpoint_preflight", budget=budget)
        try:
            url, model, api_key = _llm_settings(args)
            preflight = preflight_openai_chat(
                url=url,
                model=model,
                api_key=api_key,
                timeout_seconds=args.endpoint_preflight_timeout,
            )
        except (EndpointRequestError, RuntimeError) as exc:
            payload = status.update(
                "paused_endpoint_unavailable",
                phase="endpoint_preflight",
                message=str(exc),
                budget=budget,
                details={"resumable": True},
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 2
        status.update(
            "running",
            phase="endpoint_preflight_complete",
            budget=budget,
            details={"endpoint": preflight},
        )

    if args.initial_observation is not None:
        initial_record, initial_gp, initial_code, initial_candidate_path = _load_initial_observation(
            _resolve_path(args.initial_observation), seed_code, args.score_key
        )
        state.observations.append(initial_record)
        state.gp_observations.append(initial_gp)
        state.best = {
            "candidate_id": initial_record["candidate_id"],
            "score": float(initial_record["metrics"][args.score_key]),
            "metrics": initial_record["metrics"],
            "candidate_path": str(initial_candidate_path or seed_path),
            "summary": initial_record.get("summary", ""),
            "code": initial_code,
        }

    if args.resume:
        existing_records = _read_jsonl(manifest_path)
        if existing_records:
            state.observations.extend(_public_observations(existing_records, args.score_key))
            state.accepted_count = sum(
                record.get("status") in {"evaluated", "evaluation_error"}
                for record in existing_records
            )
            evaluated_records = [
                record
                for record in existing_records
                if record.get("status") == "evaluated"
                and args.score_key in record.get("metrics", {})
            ]
            state.evaluated_count = len(evaluated_records)
            state.real_attempt_count = sum(
                record.get("status") in {"evaluated", "evaluation_error"}
                for record in existing_records
            )
            state.failure_count = sum(
                record.get("status") in {"rejected", "evaluation_error", "search_error"}
                for record in existing_records
            )
            state.generated_count = sum(
                1
                for record in _read_jsonl(run_dir / "search_manifest.jsonl")
                if record.get("status") == "surrogate_scored"
            )
            start_iteration = max(
                int(record.get("iteration", 0)) for record in existing_records
            ) + 1
            for record in evaluated_records:
                gp_observation = _search_observation_from_record(record, args.score_key)
                if gp_observation is not None:
                    state.gp_observations.append(gp_observation)
            if evaluated_records:
                best_record = max(
                    evaluated_records,
                    key=lambda record: float(record["metrics"][args.score_key]),
                )
                candidate_path = Path(str(best_record["candidate_path"]))
                best_code = candidate_path.read_text(encoding="utf-8")
                best_score = float(best_record["metrics"][args.score_key])
                if state.best is None or best_score > float(state.best["score"]):
                    state.parent_code = best_code
                    state.best = {
                        "candidate_id": best_record["candidate_id"],
                        "score": best_score,
                        "metrics": best_record["metrics"],
                        "candidate_path": str(candidate_path),
                        "summary": best_record.get("summary", ""),
                        "code": best_code,
                    }

    _synchronize_budget_from_state(args, state, start_iteration=start_iteration)

    try:
        for iteration in range(start_iteration, args.iterations + 1):
            if args.max_real_evaluations and state.real_attempt_count >= args.max_real_evaluations:
                break
            status.update(
                "running",
                phase="search",
                iteration=iteration,
                budget=budget,
            )
            if args.method == "best_of_n":
                _run_tts_iteration(
                    args,
                    iteration=iteration,
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    sink=sink,
                    state=state,
                )
            else:
                _run_best_observed_iteration(
                    args,
                    iteration=iteration,
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    sink=sink,
                    state=state,
                )
            budget.consume("outer_iterations")
    except EndpointCircuitOpen as exc:
        payload = status.update(
            "paused_endpoint_unavailable",
            phase="search",
            iteration=iteration,
            message=str(exc),
            budget=budget,
            details={"resumable": True, "circuit": breaker.snapshot()},
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    except BudgetExceededError as exc:
        payload = status.update(
            "failed_budget_exceeded",
            phase="budget",
            iteration=iteration,
            message=str(exc),
            budget=budget,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    summary = {
        "task": TASK_ID,
        "run_dir": str(run_dir),
        "generator": args.generator,
        "evaluator": args.evaluator,
        "iterations": args.iterations,
        "method": args.method,
        "breadth": args.breadth,
        "depth": args.depth,
        "accepted": state.accepted_count,
        "generated": state.generated_count,
        "evaluated": state.evaluated_count,
        "real_evaluation_attempts": state.real_attempt_count,
        "failures": state.failure_count,
        "gp": _make_surrogate(args, state.gp_observations).summary(),
        "budget": budget.snapshot(),
        "experiment_contract": {
            "path": "" if contract is None else str(contract.path),
            "sha256": "" if contract is None else contract.digest,
            "profile": contract_profile,
        },
        "best": None
        if state.best is None
        else {key: value for key, value in state.best.items() if key != "code"},
        "ldm_task_spec": describe_ldm_task(args).to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_status = "completed" if state.failure_count == 0 else "completed_with_failures"
    status.update(
        final_status,
        phase="complete",
        iteration=min(args.iterations, int(budget.counters.get("outer_iterations", 0))),
        budget=budget,
        details={"summary_path": str(run_dir / "summary.json")},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.skip_eval or args.iterations == 0:
        return 0
    return 0 if state.evaluated_count > 0 else 1


def _run_best_observed_iteration(
    args: argparse.Namespace,
    *,
    iteration: int,
    run_dir: Path,
    manifest_path: Path,
    sink: DataCollectionSink,
    state: CampaignState,
) -> None:
    round_results: list[dict[str, Any]] = []
    round_parent_code = state.parent_code
    for proposal_index in range(1, args.breadth + 1):
        if args.max_real_evaluations and state.real_attempt_count >= args.max_real_evaluations:
            break
        candidate_id = f"i{iteration:03d}_c{proposal_index:03d}"
        state_dir = run_dir / "states" / candidate_id
        state_dir.mkdir(parents=True, exist_ok=args.resume)
        prompt = build_prompt(
            parent_code=round_parent_code,
            observations=state.observations,
            iteration=iteration,
            proposal_index=proposal_index,
            breadth=args.breadth,
        )
        (state_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        try:
            _record_llm_request(args, state)
            raw_response = generate_response(
                args,
                prompt=prompt,
                parent_code=round_parent_code,
                iteration=iteration,
                proposal_index=proposal_index,
            )
            (state_dir / "response.txt").write_text(raw_response, encoding="utf-8")
            proposal = parse_model_response(raw_response)
            _consume_budget(state, "selected_candidates")
            state.accepted_count += 1
            candidate_path = state_dir / "candidate_design.py"
            candidate_path.write_text(proposal.code, encoding="utf-8")
        except EndpointCircuitOpen:
            raise
        except (CandidateValidationError, RuntimeError) as exc:
            state.failure_count += 1
            record = {
                "candidate_id": candidate_id,
                "iteration": iteration,
                "status": "rejected",
                "error": str(exc),
            }
            _append_jsonl(manifest_path, record)
            round_results.append(record)
            continue

        status = "accepted_unevaluated"
        metrics: dict[str, Any] = {}
        error = ""
        if not args.skip_eval:
            _record_evaluation_attempt(args, state, candidate_id, iteration)
            state.real_attempt_count += 1
            try:
                metrics = evaluate_proposal(args, proposal, state_dir)
                score = float(metrics[args.score_key])
                if not math.isfinite(score):
                    raise EvaluationError(f"non-finite {args.score_key}: {score}")
                status = "evaluated"
                state.evaluated_count += 1
                _consume_budget(state, "completed_evaluations")
            except (EvaluationError, KeyError, TypeError, ValueError) as exc:
                status = "evaluation_error"
                error = str(exc)
                state.failure_count += 1

        record = {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "status": status,
            "summary": proposal.summary,
            "config_overrides": proposal.config_overrides,
            "metrics": metrics,
            "error": error,
            "candidate_path": str(candidate_path),
        }
        _append_jsonl(manifest_path, record)
        round_results.append(record)
        sink.append(
            make_collection_ir(
                proposal=proposal,
                parent_code=round_parent_code,
                observations=state.observations,
                best=state.best,
                iteration=iteration,
                evaluated_count=state.evaluated_count,
                candidate_id=candidate_id,
            ),
            provenance={
                "run_id": run_dir.name,
                "candidate_id": candidate_id,
                "iteration": iteration,
                "source": "runtime_model_action",
            },
            outcome={"status": status, "metrics": metrics, "error": error},
        )
        _update_best(state, record, proposal, args.score_key)

    state.observations.extend(_public_observations(round_results, args.score_key))
    if state.best is not None:
        state.parent_code = str(state.best["code"])


def _run_tts_iteration(
    args: argparse.Namespace,
    *,
    iteration: int,
    run_dir: Path,
    manifest_path: Path,
    sink: DataCollectionSink,
    state: CampaignState,
) -> None:
    search_manifest_path = run_dir / "search_manifest.jsonl"
    surrogate = _make_surrogate(args, state.gp_observations)
    all_candidates: list[TTSCandidate] = []
    leaves: list[TTSCandidate] = []
    seen_codes = {state.parent_code}

    for branch in range(1, args.breadth + 1):
        branch_parent = state.parent_code
        branch_prediction = ""
        branch_leaf: TTSCandidate | None = None
        for depth in range(1, args.depth + 1):
            candidate = _generate_tts_candidate(
                args,
                iteration=iteration,
                branch=branch,
                depth=depth,
                run_dir=run_dir,
                search_manifest_path=search_manifest_path,
                campaign_state=state,
                parent_code=branch_parent,
                acquisition_context=branch_prediction,
                surrogate=surrogate,
                seen_codes=seen_codes,
            )
            if candidate is None:
                continue
            all_candidates.append(candidate)
            branch_leaf = candidate
            branch_parent = candidate.proposal.code
            branch_prediction = (
                f"The GP scored the current branch state with predicted search_score "
                f"{candidate.prediction.mean:.6f}, uncertainty {candidate.prediction.std:.6f}, "
                f"and UCB {candidate.prediction.acquisition_score:.6f}. Refine it toward a "
                "distinct architecture with a stronger exploration/exploitation tradeoff."
            )
        if branch_leaf is not None:
            leaves.append(branch_leaf)

    selection_pool = leaves if args.select_from == "leaves" and leaves else all_candidates
    if not selection_pool:
        state.failure_count += 1
        record = {
            "candidate_id": f"i{iteration:03d}_none",
            "iteration": iteration,
            "status": "search_error",
            "error": "TTS produced no valid surrogate-scored candidates",
        }
        _append_jsonl(manifest_path, record)
        state.observations.extend(_public_observations([record], args.score_key))
        return

    selected_id, selected_prediction = select_max_ucb(
        [(item.candidate_id, item.feature_vector) for item in selection_pool],
        surrogate,
        beta=args.gp_beta,
    )
    selected = next(item for item in selection_pool if item.candidate_id == selected_id)
    selected = TTSCandidate(
        candidate_id=selected.candidate_id,
        proposal=selected.proposal,
        candidate_path=selected.candidate_path,
        state_dir=selected.state_dir,
        feature_vector=selected.feature_vector,
        parameter_count=selected.parameter_count,
        prediction=selected_prediction,
        branch=selected.branch,
        depth=selected.depth,
    )
    selection = {
        "iteration": iteration,
        "method": "best_of_n",
        "breadth": args.breadth,
        "depth": args.depth,
        "select_from": args.select_from,
        "feature_version": FEATURE_VERSION,
        "gp": surrogate.summary(),
        "selected_candidate_id": selected.candidate_id,
        "selected_prediction": _prediction_dict(selected.prediction),
        "candidate_count": len(all_candidates),
        "pool": [
            {
                "candidate_id": item.candidate_id,
                "branch": item.branch,
                "depth": item.depth,
                **_prediction_dict(item.prediction),
            }
            for item in all_candidates
        ],
    }
    iteration_dir = run_dir / "iterations" / f"i{iteration:03d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    (iteration_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    status = "accepted_unevaluated"
    metrics: dict[str, Any] = {}
    error = ""
    _consume_budget(state, "selected_candidates")
    state.accepted_count += 1
    if not args.skip_eval:
        _record_evaluation_attempt(args, state, selected.candidate_id, iteration)
        state.real_attempt_count += 1
        try:
            metrics = evaluate_proposal(args, selected.proposal, selected.state_dir)
            score = float(metrics[args.score_key])
            if not math.isfinite(score):
                raise EvaluationError(f"non-finite {args.score_key}: {score}")
            status = "evaluated"
            state.evaluated_count += 1
            _consume_budget(state, "completed_evaluations")
        except (EvaluationError, KeyError, TypeError, ValueError) as exc:
            status = "evaluation_error"
            error = str(exc)
            state.failure_count += 1

    record = {
        "candidate_id": selected.candidate_id,
        "iteration": iteration,
        "status": status,
        "summary": selected.proposal.summary,
        "config_overrides": selected.proposal.config_overrides,
        "metrics": metrics,
        "error": error,
        "candidate_path": str(selected.candidate_path),
        "feature_version": FEATURE_VERSION,
        "feature_vector": list(selected.feature_vector),
        "parameter_count": selected.parameter_count,
        "surrogate": _prediction_dict(selected.prediction),
        "tts": {
            "method": "best_of_n",
            "breadth": args.breadth,
            "depth": args.depth,
            "generated": len(all_candidates),
            "select_from": args.select_from,
        },
    }
    _append_jsonl(manifest_path, record)
    state.observations.extend(_public_observations([record], args.score_key))
    sink.append(
        make_collection_ir(
            proposal=selected.proposal,
            parent_code=state.parent_code,
            observations=state.observations[:-1],
            best=state.best,
            iteration=iteration,
            evaluated_count=state.evaluated_count,
            candidate_id=selected.candidate_id,
        ),
        provenance={
            "run_id": run_dir.name,
            "candidate_id": selected.candidate_id,
            "iteration": iteration,
            "source": "gp_ucb_selected_model_action",
        },
        outcome={"status": status, "metrics": metrics, "error": error},
    )
    if status == "evaluated":
        state.gp_observations.append(
            SearchObservation(
                candidate_id=selected.candidate_id,
                feature_vector=selected.feature_vector,
                score=float(metrics[args.score_key]),
            )
        )
    _update_best(state, record, selected.proposal, args.score_key)
    if state.best is not None:
        state.parent_code = str(state.best["code"])


def _generate_tts_candidate(
    args: argparse.Namespace,
    *,
    iteration: int,
    branch: int,
    depth: int,
    run_dir: Path,
    search_manifest_path: Path,
    campaign_state: CampaignState,
    parent_code: str,
    acquisition_context: str,
    surrogate: RBFGPSurrogate,
    seen_codes: set[str],
) -> TTSCandidate | None:
    candidate_id = f"i{iteration:03d}_b{branch:02d}_d{depth:02d}"
    state_dir = run_dir / "states" / candidate_id
    state_dir.mkdir(parents=True, exist_ok=True)
    last_error = "proposal generation failed"
    total_budget = args.breadth * args.depth
    proposal_index = (branch - 1) * args.depth + depth

    for attempt in range(1, args.proposal_retries + 2):
        retry_context = "" if attempt == 1 else f"Previous attempt was rejected: {last_error}"
        prompt = build_prompt(
            parent_code=parent_code,
            observations=campaign_state.observations,
            iteration=iteration,
            proposal_index=proposal_index,
            breadth=total_budget,
            search_note=(
                f"N4H4 branch {branch}/{args.breadth}, refinement {depth}/{args.depth}. "
                "Propose a substantive architecture change; returning the parent unchanged is invalid."
            ),
            acquisition_context="\n".join(
                item for item in (acquisition_context, retry_context) if item
            ),
        )
        prompt_path = state_dir / ("prompt.txt" if attempt == 1 else f"prompt_attempt_{attempt}.txt")
        response_path = state_dir / (
            "response.txt" if attempt == 1 else f"response_attempt_{attempt}.txt"
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        try:
            _record_llm_request(args, campaign_state)
            raw_response = generate_response(
                args,
                prompt=prompt,
                parent_code=parent_code,
                iteration=iteration,
                proposal_index=proposal_index,
            )
            response_path.write_text(raw_response, encoding="utf-8")
            proposal = parse_model_response(raw_response)
            if proposal.code in seen_codes:
                raise CandidateValidationError("candidate duplicates its parent or another TTS state")
            candidate_path = state_dir / "candidate_design.py"
            candidate_path.write_text(proposal.code, encoding="utf-8")
            parameter_count = _validate_tts_candidate(args, proposal, state_dir)
            feature_vector = encode_candidate(
                proposal.code,
                config_overrides=proposal.config_overrides,
                parameter_count=parameter_count,
            )
            prediction = surrogate.predict(feature_vector, beta=args.gp_beta)
        except EndpointCircuitOpen:
            raise
        except (CandidateValidationError, EvaluationError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            continue

        seen_codes.add(proposal.code)
        campaign_state.generated_count += 1
        _consume_budget(campaign_state, "valid_search_candidates")
        record = {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "branch": branch,
            "depth": depth,
            "attempt": attempt,
            "status": "surrogate_scored",
            "summary": proposal.summary,
            "candidate_path": str(candidate_path),
            "config_overrides": proposal.config_overrides,
            "parameter_count": parameter_count,
            "feature_version": FEATURE_VERSION,
            "feature_vector": list(feature_vector),
            "surrogate": _prediction_dict(prediction),
        }
        _append_jsonl(search_manifest_path, record)
        return TTSCandidate(
            candidate_id=candidate_id,
            proposal=proposal,
            candidate_path=candidate_path,
            state_dir=state_dir,
            feature_vector=feature_vector,
            parameter_count=parameter_count,
            prediction=prediction,
            branch=branch,
            depth=depth,
        )

    campaign_state.failure_count += 1
    _append_jsonl(
        search_manifest_path,
        {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "branch": branch,
            "depth": depth,
            "status": "rejected",
            "attempts": args.proposal_retries + 1,
            "error": last_error,
        },
    )
    return None


def _validate_tts_candidate(
    args: argparse.Namespace, proposal: CandidateProposal, state_dir: Path
) -> float | None:
    if not args.validate_tts_candidates:
        return None
    metrics = evaluate_gpu_smoke(
        proposal,
        work_dir=state_dir / "contract",
        prelude_path=TASK_ROOT / "resources" / "smoke_prelude.py",
        gpu_devices=[int(args.tts_gpu_device)],
        timeout_seconds=min(args.eval_timeout, 300),
    )
    device = str(args.tts_gpu_device)
    return float(metrics["devices"][device]["parameter_count"])


def _make_budget_ledger(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    contract: ExperimentContract | None,
    contract_profile: str,
    resume: bool,
) -> BudgetLedger:
    expensive_limit = 0
    if not args.skip_eval:
        expensive_limit = args.max_real_evaluations or (
            args.iterations if args.method == "best_of_n" else args.iterations * args.breadth
        )
    limits: dict[str, int | float] = {
        "outer_iterations": args.iterations,
        "llm_requests": (
            args.iterations * args.breadth * args.depth * (args.proposal_retries + 1)
            if args.generator == "openai" and args.method == "best_of_n"
            else args.iterations * args.breadth if args.generator == "openai" else 0
        ),
        "valid_search_candidates": (
            args.iterations * args.breadth * args.depth
            if args.method == "best_of_n"
            else 0
        ),
        "selected_candidates": (
            args.iterations if args.method == "best_of_n" else args.iterations * args.breadth
        ),
        "expensive_evaluation_attempts": expensive_limit,
        "benchmark_jobs": (
            expensive_limit * len(args.benchmark) if args.evaluator == "benchmark" else 0
        ),
    }
    if contract is not None and contract_profile:
        limits.update(contract.profile(contract_profile).budget)

    budget_path = run_dir / "budget.json"
    if resume and budget_path.is_file():
        ledger = BudgetLedger.load(budget_path)
        if ledger.limits != limits:
            if contract_profile:
                raise RuntimeError(
                    "resume budget limits do not match the current contract profile: "
                    f"saved={ledger.limits}, current={limits}"
                )
            too_small = {
                name: (limits.get(name), value)
                for name, value in ledger.counters.items()
                if limits.get(name) is not None and value > limits[name]
            }
            if too_small:
                raise RuntimeError(
                    "resume budget cannot be reduced below consumed counters: "
                    f"{too_small}"
                )
            ledger.limits = limits
            ledger.write()
        return ledger
    ledger = BudgetLedger(
        limits=limits,
        path=budget_path,
        metadata={
            "contract_profile": contract_profile,
            "contract_sha256": "" if contract is None else contract.digest,
        },
    )
    ledger.write()
    return ledger


def _synchronize_budget_from_state(
    args: argparse.Namespace,
    state: CampaignState,
    *,
    start_iteration: int,
) -> None:
    if state.budget is None:
        return
    inferred = {
        "outer_iterations": max(0, start_iteration - 1),
        "valid_search_candidates": state.generated_count if args.method == "best_of_n" else 0,
        "selected_candidates": state.accepted_count,
        "expensive_evaluation_attempts": state.real_attempt_count,
        "completed_evaluations": state.evaluated_count,
        "benchmark_jobs": (
            state.real_attempt_count * len(args.benchmark)
            if args.evaluator == "benchmark"
            else 0
        ),
    }
    for name, value in inferred.items():
        if value > state.budget.counters.get(name, 0):
            state.budget.set_counter(name, value)


def _consume_budget(
    state: CampaignState,
    name: str,
    amount: int | float = 1,
) -> None:
    if state.budget is not None:
        state.budget.consume(name, amount)


def _record_llm_request(args: argparse.Namespace, state: CampaignState) -> None:
    if args.generator == "openai":
        _consume_budget(state, "llm_requests")


def _record_evaluation_attempt(
    args: argparse.Namespace,
    state: CampaignState,
    candidate_id: str,
    iteration: int,
) -> None:
    _consume_budget(state, "expensive_evaluation_attempts")
    if args.evaluator == "benchmark":
        _consume_budget(state, "benchmark_jobs", len(args.benchmark))
    if state.status is not None:
        state.status.update(
            "running",
            phase="evaluation",
            iteration=iteration,
            budget=state.budget,
            details={"candidate_id": candidate_id, "evaluator": args.evaluator},
        )


def _make_surrogate(
    args: argparse.Namespace, observations: list[SearchObservation]
) -> RBFGPSurrogate:
    return RBFGPSurrogate(
        observations,
        lengthscale=args.gp_lengthscale,
        noise=args.gp_noise,
        prior_mean=args.gp_prior_mean,
        prior_std=args.gp_prior_std,
        feature_version=FEATURE_VERSION,
    )


def _load_initial_observation(
    path: Path, seed_code: str, score_key: str
) -> tuple[dict[str, Any], SearchObservation, str, Path | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("best", payload)
    metrics = dict(source.get("metrics", {}))
    if "search_score" not in metrics:
        metrics["search_score"] = continuous_search_score(metrics)
    if score_key not in metrics:
        raise RuntimeError(f"initial observation has no {score_key!r}: {path}")
    code = seed_code
    candidate_path = Path(str(source.get("candidate_path", "")))
    if not candidate_path.is_absolute():
        candidate_path = path.parent / candidate_path
    if candidate_path.is_file():
        code = candidate_path.read_text(encoding="utf-8")
    else:
        candidate_path = Path(source.get("seed_file", ""))
        if not candidate_path.is_absolute():
            candidate_path = path.parent / candidate_path
        if candidate_path.is_file():
            code = candidate_path.read_text(encoding="utf-8")
        else:
            candidate_path = None
    _, overrides = validate_candidate_code(code)
    parameter_count = metrics.get("parameter_count")
    feature_vector = encode_candidate(
        code,
        config_overrides=overrides,
        parameter_count=None if parameter_count is None else float(parameter_count),
    )
    candidate_id = f"initial:{source.get('candidate_id', path.stem)}"
    record = {
        "candidate_id": candidate_id,
        "status": "initialization",
        "summary": source.get("summary", "Shared evaluated seed architecture."),
        "metrics": metrics,
        "selection_score": metrics[score_key],
    }
    observation = SearchObservation(candidate_id, feature_vector, float(metrics[score_key]))
    return record, observation, code, candidate_path


def _search_observation_from_record(
    record: dict[str, Any], score_key: str
) -> SearchObservation | None:
    metrics = record.get("metrics", {})
    try:
        score = float(metrics[score_key])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    raw_vector = record.get("feature_vector")
    if isinstance(raw_vector, list):
        feature_vector = tuple(float(value) for value in raw_vector)
    else:
        candidate_path = Path(str(record.get("candidate_path", "")))
        if not candidate_path.is_file():
            return None
        proposal = parse_model_response(
            json.dumps(
                {
                    "reasoning": "",
                    "summary": record.get("summary", ""),
                    "code": candidate_path.read_text(encoding="utf-8"),
                }
            )
        )
        feature_vector = encode_candidate(
            proposal.code,
            config_overrides=proposal.config_overrides,
            parameter_count=metrics.get("parameter_count"),
        )
    return SearchObservation(str(record["candidate_id"]), feature_vector, score)


def _prediction_dict(prediction: GPPrediction) -> dict[str, float]:
    return {
        "mean": prediction.mean,
        "std": prediction.std,
        "acquisition_score": prediction.acquisition_score,
    }


def _update_best(
    state: CampaignState,
    record: dict[str, Any],
    proposal: CandidateProposal,
    score_key: str,
) -> None:
    if record.get("status") != "evaluated":
        return
    metrics = record["metrics"]
    score = float(metrics[score_key])
    if state.best is None or score > float(state.best["score"]):
        state.best = {
            "candidate_id": record["candidate_id"],
            "score": score,
            "metrics": metrics,
            "candidate_path": record["candidate_path"],
            "summary": proposal.summary,
            "code": proposal.code,
        }


def build_prompt(
    *,
    parent_code: str,
    observations: list[dict[str, Any]],
    iteration: int,
    proposal_index: int,
    breadth: int,
    search_note: str = "",
    acquisition_context: str = "",
) -> str:
    history = json.dumps(observations[-12:], indent=2, sort_keys=True)
    return f"""You are designing a protein inverse-folding structure encoder.

Given backbone coordinates X with shape (B, L, 4, 3) for N, CA, C, O atoms and
a binary mask with shape (B, L), implement StructureEncoder.forward returning
(B, L, hidden_dim). InverseFoldingModel.forward must return normalized amino-acid
log probabilities with shape (B, L, 20).

The fixed MLS-Bench scaffold supplies torch, nn, F, NUM_AA, _rbf, _dihedrals,
_orientations, and knn_graph. Preserve constructor compatibility with
hidden_dim, num_encoder_layers, k_neighbors, dropout, and num_rbf. Keep the
model within a practical parameter budget. CONFIG_OVERRIDES may contain only
learning_rate, dropout, num_encoder_layers, and batch_size.

This is iteration {iteration}, proposal {proposal_index} of {breadth}.
{search_note}

Surrogate-guided refinement context:
{acquisition_context or "No branch-specific GP feedback is available yet."}

Prior evaluated observations:
{history}

Current parent editable code:
```python
{parent_code.rstrip()}
```

Return JSON only with exactly these fields:
{{"reasoning":"concise design rationale","summary":"short change summary","code":"complete Python editable-region code including CONFIG_OVERRIDES"}}
The code field must contain raw Python without Markdown fences. Do not return a
patch. Do not include dataset loading, training loops, file I/O, network access,
subprocesses, or module-level execution.
"""


def generate_response(
    args: argparse.Namespace,
    *,
    prompt: str,
    parent_code: str,
    iteration: int,
    proposal_index: int,
) -> str:
    if args.generator == "mock":
        return _mock_response(parent_code, iteration, proposal_index)
    return _openai_response(args, prompt)


def evaluate_proposal(
    args: argparse.Namespace, proposal: CandidateProposal, state_dir: Path
) -> dict[str, Any]:
    if args.evaluator == "mock":
        return evaluate_mock(proposal)
    gpu_devices = args.gpu_device or _visible_devices_from_env()
    if args.evaluator == "gpu_smoke":
        return evaluate_gpu_smoke(
            proposal,
            work_dir=state_dir / "evaluation",
            prelude_path=TASK_ROOT / "resources" / "smoke_prelude.py",
            gpu_devices=gpu_devices,
            timeout_seconds=args.eval_timeout,
        )
    if args.scaffold_path is None or args.data_root is None:
        raise EvaluationError(
            "benchmark evaluation requires --scaffold-path and --data-root"
        )
    return evaluate_benchmarks(
        proposal,
        work_dir=state_dir / "evaluation",
        scaffold_path=_resolve_path(args.scaffold_path),
        data_root=_resolve_path(args.data_root),
        benchmarks=args.benchmark,
        gpu_devices=gpu_devices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cath_max_train_hours=args.cath_max_train_hours,
        ts_max_train_hours=args.ts_max_train_hours,
        cath_job_timeout_seconds=args.cath_job_timeout,
        ts_job_timeout_seconds=args.ts_job_timeout,
        timeout_seconds=args.eval_timeout,
        parallel=args.parallel_benchmarks,
    )


def make_collection_ir(
    *,
    proposal: CandidateProposal,
    parent_code: str,
    observations: list[dict[str, Any]],
    best: dict[str, Any] | None,
    iteration: int,
    evaluated_count: int,
    candidate_id: str,
) -> dict[str, Any]:
    public_best = None
    if best is not None:
        public_best = {
            "candidate_id": best["candidate_id"],
            "score": best["score"],
            "metrics": best["metrics"],
            "summary": best["summary"],
        }
    return make_complete_design_ir(
        task_id=TASK_ID,
        domain="protein engineering",
        task_description=TASK_DESCRIPTION,
        objectives=OBJECTIVES,
        design_space_description=(
            "Complete code for the benchmark's editable encoder region. The fixed "
            "data, training, and evaluation scaffold is not model-editable."
        ),
        observations=observations[-12:],
        candidates=[
            {
                "candidate_id": candidate_id,
                "code": proposal.code,
                "config_overrides": proposal.config_overrides,
            }
        ],
        request_description=(
            "Propose one complete, interface-compatible encoder design as Python code."
        ),
        num_candidates=1,
        round_idx=iteration,
        num_evaluated=evaluated_count,
        best_so_far=public_best,
        allows_new_parameters=True,
        reasoning_available=bool(proposal.reasoning),
        reasoning=proposal.reasoning or None,
        summary=proposal.summary or None,
        raw_context={"parent_train_py": parent_code},
    )


def _mock_response(parent_code: str, iteration: int, proposal_index: int) -> str:
    dropout = round(0.06 + 0.01 * ((iteration + proposal_index) % 5), 3)
    layers = 3 + ((iteration + proposal_index) % 3)
    code = replace_config_overrides(
        parent_code,
        {"dropout": dropout, "num_encoder_layers": layers},
    )
    payload = {
        "reasoning": (
            "Use the validated residual message-passing baseline while exercising "
            "the permitted regularization and depth controls."
        ),
        "summary": f"Mock proposal with dropout={dropout} and layers={layers}.",
        "code": code,
    }
    return json.dumps(payload, sort_keys=True)


def _openai_response(args: argparse.Namespace, prompt: str) -> str:
    url, model, api_key = _llm_settings(args)
    operation = lambda: request_openai_chat(
        url=url,
        model=model,
        api_key=api_key,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON matching the user's requested schema.",
            },
            {"role": "user", "content": prompt},
        ],
        timeout_seconds=args.request_timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    breaker = getattr(args, "_endpoint_breaker", None)
    try:
        if breaker is None:
            return operation()
        return call_with_circuit_breaker(breaker, operation)
    except EndpointCircuitOpen:
        raise
    except EndpointRequestError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc


def _llm_settings(args: argparse.Namespace) -> tuple[str, str, str]:
    url = args.llm_url.strip() or os.environ.get("TTS_LLM_URL", "").strip()
    model = args.llm_model_name.strip() or os.environ.get("TTS_LLM_MODEL", "").strip()
    api_key = (
        args.api_key.strip()
        or os.environ.get("TTS_LLM_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not url or not model:
        raise RuntimeError("openai generator requires an LLM URL and model name")
    return url, model, api_key


def _public_observations(
    records: list[dict[str, Any]], score_key: str
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for record in records:
        item = {
            "candidate_id": record["candidate_id"],
            "status": record["status"],
            "summary": record.get("summary", ""),
        }
        if record.get("metrics"):
            item["metrics"] = record["metrics"]
            item["selection_score"] = record["metrics"].get(score_key)
        if record.get("error"):
            item["error"] = record["error"]
        observations.append(item)
    return observations


def _visible_devices_from_env() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices: list[int] = []
    for part in raw.split(","):
        try:
            devices.append(int(part.strip()))
        except ValueError:
            continue
    return devices


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (Path.cwd() / expanded).resolve()


def _unique_run_dir(parent: Path, run_name: str) -> Path:
    candidate = parent / run_name
    if not candidate.exists():
        return candidate
    for index in range(1, 10_000):
        suffixed = parent / f"{run_name}_{index:03d}"
        if not suffixed.exists():
            return suffixed
    raise RuntimeError(f"could not allocate a run directory under {parent}")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid resume manifest JSON at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"invalid resume manifest record at line {line_number}: expected object"
            )
        records.append(record)
    return records


__all__ = [
    "build_prompt",
    "describe_ldm_task",
    "evaluate_proposal",
    "main",
    "make_collection_ir",
    "parse_args",
]
