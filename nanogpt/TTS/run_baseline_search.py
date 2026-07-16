#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TTS.run_a_search_nanogpt import make_unique_run_dir, safe_path_tag
from TTS.run_model_based_search import (
    GENERATORS,
    OPERATION_GENERATORS,
    FeedbackMemory,
    OperationSearchEngine,
    RunLogger,
    best_score_from_states,
    finite_score,
    format_optional_float,
    format_score_delta,
    is_better,
    operation_schema_to_json,
    resolve_feedback_path,
    resolve_operation_schema,
    updated_best_score,
    write_state_update,
)
from TTS.search_core import DEFAULT_TASK_CONTEXT, ProgressBar, SearchConfig, SearchEngine, SearchState, jsonable


class BaselineProgress:
    def __init__(self, *, enabled: bool, total: int, width: int, score_key: str, minimize: bool):
        self.enabled = enabled and total > 0
        self.score_key = score_key
        self.minimize = minimize
        self.count = 0
        self.best_score: float | None = None
        self.bar = ProgressBar(total=total, label="baseline", width=width) if self.enabled else None
        if self.bar is not None:
            self.bar.update(0, status="starting")

    def status(self, message: str) -> None:
        if self.bar is not None:
            self.bar.update(self.count, best_score=self.best_score, status=message)

    def generated(self, state: SearchState) -> None:
        self.step(f"generated {state.state_id} {state.status}")

    def evaluated(self, state: SearchState) -> None:
        if state.score is not None and finite_score(state.score):
            score = float(state.score)
            if self.best_score is None or is_better(score, self.best_score, minimize=self.minimize):
                self.best_score = score
            self.step(f"evaluated {state.state_id} {self.score_key}={score:.6g}")
        else:
            self.step(f"evaluated {state.state_id} {state.status}")

    def step(self, message: str) -> None:
        if self.bar is None:
            return
        self.count += 1
        self.bar.update(self.count, best_score=self.best_score, status=message)

    def finish(self, message: str = "done") -> None:
        if self.bar is not None:
            self.bar.finish(self.count, best_score=self.best_score, status=message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GP-free iterative baseline: ask the coder LLM for train.py edits, "
            "execute generated candidates, feed real scores back into later prompts, "
            "and choose the next seed by an actual-score policy."
        ),
    )
    parser.add_argument("--iterations", type=int, default=3, help="Number of LLM-edit/evaluate rounds.")
    parser.add_argument(
        "--breadth",
        type=int,
        default=1,
        help=(
            "Candidates generated and actually evaluated per iteration. "
            "The default 1 is a simple iterative chain/hill-climb."
        ),
    )
    parser.add_argument(
        "--seed-policy",
        choices=["original", "latest", "best"],
        default="best",
        help=(
            "Root train.py for each iteration. original always uses --train-file; "
            "latest uses the most recently evaluated candidate; best uses the best evaluated candidate."
        ),
    )
    parser.add_argument(
        "--max-generated-per-iteration",
        type=int,
        default=128,
        help="Safety cap on generated states per iteration. 0 disables the cap.",
    )
    parser.add_argument("--evaluate-root", action="store_true", help="Evaluate the root state before iteration 1.")
    parser.add_argument("--skip-eval", action="store_true", help="Generate candidates without executing them.")
    parser.add_argument(
        "--max-real-evaluations",
        type=int,
        default=0,
        help="Maximum train.py executions across the whole run. 0 means no cap.",
    )

    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-file", type=Path, default=Path("train.py"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Parent directory for baseline runs. Default: TTS/runs/baseline.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Existing baseline run directory, baseline_summary.json, or summary.json to append to. "
            "--iterations is interpreted as additional iterations when resuming."
        ),
    )
    parser.add_argument("--run-name", default="", help="Optional explicit child run folder name.")
    parser.add_argument("--export-best", type=Path, default=None, help="Optional path to copy the best train.py.")

    parser.add_argument("--eval-command", default="uv run python {train_path}")
    parser.add_argument("--eval-shell", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--score-key", default="val_bpb")
    parser.add_argument("--maximize", action="store_true")
    parser.add_argument("--failure-score", type=float, default=1.0e9)

    parser.add_argument("--generator", choices=sorted(GENERATORS), default="api")
    parser.add_argument("--llm-url", default=os.environ.get("TTS_LLM_URL", "http://127.0.0.1:52307/v1"))
    parser.add_argument("--llm-model-name", default=os.environ.get("TTS_LLM_MODEL", "Qwen3-Coder-30B-A3B-Instruct"))
    parser.add_argument("--api-key", default=os.environ.get("TTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--top-logprobs", type=int, default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-edits-per-step", type=int, default=1)
    parser.add_argument("--prompt-max-chars", type=int, default=100_000)
    parser.add_argument("--context-file", type=Path, default=None)
    parser.add_argument("--context", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument(
        "--feedback-max-rows",
        type=int,
        default=12,
        help="Maximum recent/best real-evaluation rows included in later LLM prompts. 0 disables feedback injection.",
    )
    parser.add_argument(
        "--feedback-tsv",
        type=Path,
        default=None,
        help="Optional path for iteration feedback TSV. Default: run_dir/iteration_feedback.tsv.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-width", type=int, default=28)
    parser.add_argument("--response-log-chars", type=int, default=20_000)

    parser.add_argument(
        "--operation-schema",
        type=Path,
        default=None,
        help=(
            "JSON schema for fixed-dimension operation baselines. Required for "
            "--generator operation_tool or operation_mock unless a default schema is found."
        ),
    )
    parser.add_argument(
        "--operation-retries",
        type=int,
        default=2,
        help="LLM repair retries when operation_tool emits invalid operations.",
    )
    parser.add_argument(
        "--max-operations-per-step",
        type=int,
        default=2,
        help="Maximum structured operations allowed in one generated child.",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    resume_info = resolve_resume_info(args.resume_from, project_root) if args.resume_from is not None else None
    if resume_info is not None:
        apply_resume_defaults(args, resume_info, project_root)
    train_file = args.train_file if args.train_file.is_absolute() else project_root / args.train_file
    if not train_file.exists():
        raise SystemExit(f"train file does not exist: {train_file}")

    if args.max_generated_per_iteration > 0 and max(1, args.breadth) > args.max_generated_per_iteration:
        raise SystemExit(
            f"Refusing to generate {max(1, args.breadth)} states per iteration. "
            "Raise --max-generated-per-iteration or reduce --breadth."
        )

    operation_schema = resolve_operation_schema(args, project_root)
    if args.generator in OPERATION_GENERATORS and operation_schema is None:
        raise SystemExit(
            "--generator operation_tool/operation_mock requires --operation-schema, "
            "or a default TTS/operation_schema_real_train.json / TTS/operation_schema_mock_train.json."
        )
    if operation_schema is not None and args.generator not in OPERATION_GENERATORS:
        raise SystemExit("--operation-schema is only supported with operation_tool or operation_mock generators.")
    args.operation_schema_object = operation_schema

    if resume_info is None:
        out_parent_dir = project_root / "TTS" / "runs" / "baseline" if args.out_dir is None else args.out_dir
        if not out_parent_dir.is_absolute():
            out_parent_dir = project_root / out_parent_dir
        run_name = safe_path_tag(args.run_name) if args.run_name.strip() else default_run_name(args, train_file)
        out_dir = make_unique_run_dir(out_parent_dir, run_name)
        starting_iteration = 1
        previous_iteration_records: list[dict[str, Any]] = []
        previous_real_evaluations = 0
    else:
        out_dir = resume_info["run_dir"]
        out_parent_dir = out_dir.parent
        run_name = safe_path_tag(args.run_name) if args.run_name.strip() else out_dir.name
        starting_iteration = int(resume_info["next_iteration"])
        previous_iteration_records = list(resume_info.get("iterations", []))
        previous_real_evaluations = int(resume_info.get("real_evaluations") or 0)
    log_path = out_dir / "baseline.log"
    logger = RunLogger(log_path)
    logger.write(
        ("resume " if resume_info is not None else "start ")
        +
        f"run={run_name} train_file={train_file} generator={args.generator} "
        f"iterations={max(0, args.iterations)} start_iteration={starting_iteration} "
        f"breadth={max(1, args.breadth)} "
        f"seed_policy={args.seed_policy}"
    )

    if operation_schema is not None:
        schema_out_path = out_dir / "operation_schema.json"
        schema_out_path.write_text(
            json.dumps(operation_schema_to_json(operation_schema), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.write(f"operation_schema version={operation_schema.version} path={operation_schema.path}")

    if "{train_path}" not in args.eval_command:
        warning = "--eval-command does not contain {train_path}; evaluation will not run the generated child state."
        print(f"warning: {warning}", file=sys.stderr)
        logger.write(f"warning {warning}")

    task_context = build_task_context(project_root, args)
    config = SearchConfig(
        project_root=project_root,
        seed_train_path=train_file,
        out_dir=out_dir,
        eval_command=args.eval_command,
        eval_shell=args.eval_shell,
        score_key=args.score_key,
        minimize=not args.maximize,
        failure_score=args.failure_score,
        timeout_seconds=args.timeout_seconds,
        run_evaluation=not args.skip_eval,
        generator=args.generator,
        llm_url=args.llm_url,
        llm_model_name=args.llm_model_name,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_logprobs=args.logprobs or args.top_logprobs is not None,
        top_logprobs=args.top_logprobs,
        disable_thinking=args.disable_thinking,
        concurrency=args.concurrency,
        num_edits_per_step=args.num_edits_per_step,
        prompt_max_chars=args.prompt_max_chars,
        task_context=task_context,
        extra_instruction=args.instruction,
        feedback_context="",
        show_progress=False,
        progress_width=args.progress_width,
        response_log_chars=args.response_log_chars,
    )
    if operation_schema is not None:
        engine: SearchEngine = OperationSearchEngine(config, operation_schema, args)
    else:
        engine = SearchEngine(config)

    feedback_path = resolve_feedback_path(args.feedback_tsv, project_root, out_dir)
    feedback_memory = FeedbackMemory(
        feedback_path,
        score_key=args.score_key,
        minimize=not args.maximize,
        max_rows=max(0, int(args.feedback_max_rows)),
    )
    if resume_info is not None:
        load_feedback_memory_rows(feedback_memory, feedback_path)
    engine.config.feedback_context = feedback_memory.prompt_context()
    logger.write(f"feedback path={feedback_path} max_rows={feedback_memory.max_rows}")

    total_progress = estimate_progress_total(args)
    progress = BaselineProgress(
        enabled=not args.no_progress,
        total=total_progress,
        width=args.progress_width,
        score_key=args.score_key,
        minimize=not args.maximize,
    )

    if resume_info is not None:
        load_existing_states(engine, out_dir)
        best_actual = state_from_id(engine, resume_info.get("best_state_id"))
        latest_actual = latest_evaluated_child_state(engine)
        if best_actual is not None and best_actual.score is not None and finite_score(best_actual.score):
            progress.best_score = float(best_actual.score)
        logger.write(
            f"loaded resume states={len(engine.states)} next_state_counter={engine._counter} "
            f"best_state={None if best_actual is None else best_actual.state_id} "
            f"latest_state={None if latest_actual is None else latest_actual.state_id}"
        )
    else:
        best_actual = None
        latest_actual = None
    engine.evaluation_count = previous_real_evaluations
    iteration_records: list[dict[str, Any]] = []
    real_evaluations = previous_real_evaluations

    for iteration in range(starting_iteration, starting_iteration + max(0, args.iterations)):
        if args.max_real_evaluations > 0 and real_evaluations >= args.max_real_evaluations:
            logger.write(f"stop before iteration={iteration} reason=max_real_evaluations")
            break
        progress.status(
            f"iteration {iteration}/{starting_iteration + max(0, args.iterations) - 1} "
            f"best={None if best_actual is None else best_actual.score}"
        )
        engine.config.seed_train_path = choose_seed_path(args, train_file, best_actual, latest_actual)
        remaining_real_evaluations = (
            None
            if args.max_real_evaluations <= 0
            else max(0, args.max_real_evaluations - real_evaluations)
        )
        record = await run_iteration(
            engine,
            args,
            iteration=iteration,
            run_name=run_name,
            logger=logger,
            progress=progress,
            feedback_memory=feedback_memory,
            previous_best_score=None if best_actual is None else best_actual.score,
            remaining_real_evaluations=remaining_real_evaluations,
        )
        iteration_records.append(record)
        real_evaluations += int(record.get("real_evaluations") or 0)
        actual_states = record.get("actual_states")
        if not isinstance(actual_states, list):
            actual_states = []
        for actual_state in actual_states:
            if not isinstance(actual_state, SearchState):
                continue
            latest_actual = actual_state
            if actual_state.score is None or not finite_score(actual_state.score):
                continue
            if best_actual is None or is_better(
                actual_state.score,
                best_actual.score,
                minimize=engine.config.minimize,
            ):
                best_actual = actual_state

    progress.finish("done")

    all_iteration_records = previous_iteration_records + iteration_records
    summary_args = dict(vars(args))
    summary_args.pop("operation_schema_object", None)
    summary_args["resume_from"] = None if resume_info is None else str(resume_info["run_dir"])
    summary_args["continued_from_iteration"] = None if resume_info is None else starting_iteration - 1
    summary_args["additional_iterations_requested"] = max(0, int(args.iterations))
    if operation_schema is not None:
        summary_args["operation_schema_version"] = operation_schema.version
        summary_args["operation_schema_path"] = None if operation_schema.path is None else str(operation_schema.path)
    summary_args["run_name"] = run_name
    summary_args["run_parent_dir"] = str(out_parent_dir)
    summary_args["run_dir"] = str(out_dir)
    summary_args["log"] = str(log_path)
    summary_args["feedback_tsv"] = str(feedback_path)
    summary_path = engine.write_summary(method="baseline_iterative", args=summary_args, best=best_actual)
    write_baseline_summary(
        out_dir,
        summary_path,
        log_path,
        feedback_path,
        all_iteration_records,
        best_actual,
        real_evaluations,
    )
    logger.write(
        "finish "
        f"new_iterations={len(iteration_records)} total_iterations={len(all_iteration_records)} "
        f"real_evaluations={real_evaluations} "
        f"best_state={None if best_actual is None else best_actual.state_id} "
        f"best_score={None if best_actual is None else best_actual.score}"
    )

    if args.export_best is not None and best_actual is not None:
        export_path = args.export_best if args.export_best.is_absolute() else project_root / args.export_best
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_actual.train_path, export_path)

    payload = {
        "out_dir": str(out_dir.resolve()),
        "summary": str(summary_path.resolve()),
        "baseline_summary": str((out_dir / "baseline_summary.json").resolve()),
        "log": str(log_path.resolve()),
        "feedback_tsv": str(feedback_path.resolve()),
        "best_state_id": None if best_actual is None else best_actual.state_id,
        "best_score": None if best_actual is None else best_actual.score,
        "real_evaluations": real_evaluations,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


async def run_iteration(
    engine: SearchEngine,
    args: argparse.Namespace,
    *,
    iteration: int,
    run_name: str,
    logger: RunLogger,
    progress: BaselineProgress,
    feedback_memory: FeedbackMemory,
    previous_best_score: float | None,
    remaining_real_evaluations: int | None,
) -> dict[str, Any]:
    root = engine.create_seed_state()
    root.metrics.update(
        {
            "baseline_iteration": iteration,
            "baseline_role": "root",
            "baseline_seed_policy": args.seed_policy,
            "baseline_seed_path": str(engine.config.seed_train_path),
        }
    )
    write_state_update(engine, root)
    logger.write(
        f"iteration={iteration} root={root.state_id} seed_path={engine.config.seed_train_path} "
        f"previous_best={previous_best_score}"
    )

    actual_states: list[SearchState] = []
    generated_states: list[SearchState] = []
    selected_state: SearchState | None = None
    real_evaluations = 0
    remaining = remaining_real_evaluations
    best_after = previous_best_score

    if iteration == 1 and args.evaluate_root:
        if args.skip_eval:
            engine.defer_evaluation(root, reason="Baseline root evaluation skipped by --skip-eval.")
        elif remaining == 0:
            engine.defer_evaluation(root, reason="Baseline root evaluation deferred because --max-real-evaluations was reached.")
        else:
            engine.evaluate_state(root)
            progress.evaluated(root)
            real_evaluations += 1
            if remaining is not None:
                remaining = max(0, remaining - 1)
            actual_states.append(root)
            best_after = updated_best_score(best_after, root.score, minimize=engine.config.minimize)
            feedback_memory.record(
                kind="root",
                iteration=iteration,
                state=root,
                root=root,
                selected_surrogate_metrics={},
                previous_best_score=previous_best_score,
                best_score_after=best_after,
            )
            engine.config.feedback_context = feedback_memory.prompt_context()
            logger.write(
                f"iteration={iteration} evaluated_root={root.state_id} "
                f"{engine.config.score_key}={root.score}"
            )

    child_count = max(1, int(args.breadth))
    progress.status(f"iteration {iteration} generating {child_count} candidate(s)")
    children = await engine.expand_state(
        root,
        child_count,
        search_note=(
            f"baseline iteration {iteration}: propose a candidate edit that will be executed immediately. "
            "No GP or surrogate model will select among candidates; only real evaluation scores matter."
        ),
    )
    for index, child in enumerate(children, start=1):
        child.metrics.update(
            {
                "baseline_iteration": iteration,
                "baseline_candidate_index": index,
                "baseline_candidate_count": child_count,
                "baseline_seed_policy": args.seed_policy,
                "baseline_run_name": run_name,
            }
        )
        write_state_update(engine, child)
        generated_states.append(child)
        progress.generated(child)
        if child.status == "generation_error":
            logger.write(f"iteration={iteration} candidate={child.state_id} generation_error={child.error}")
            continue
        if args.skip_eval:
            engine.defer_evaluation(child, reason="Baseline real evaluation skipped by --skip-eval.")
            continue
        if remaining == 0:
            engine.defer_evaluation(child, reason="Baseline real evaluation deferred because --max-real-evaluations was reached.")
            continue

        incumbent_before = best_after
        engine.evaluate_state(child)
        progress.evaluated(child)
        real_evaluations += 1
        if remaining is not None:
            remaining = max(0, remaining - 1)
        actual_states.append(child)
        best_after = updated_best_score(best_after, child.score, minimize=engine.config.minimize)
        feedback_memory.record(
            kind="candidate",
            iteration=iteration,
            state=child,
            root=root,
            selected_surrogate_metrics={},
            previous_best_score=incumbent_before,
            best_score_after=best_after,
        )
        engine.config.feedback_context = feedback_memory.prompt_context()
        logger.write(
            f"iteration={iteration} candidate={child.state_id} status={child.status} "
            f"{engine.config.score_key}={child.score} "
            f"delta={format_score_delta(child.score, incumbent_before, engine.config.minimize)}"
        )

    selectable = [
        state
        for state in actual_states
        if state.parent_id == root.state_id and state.score is not None and finite_score(state.score)
    ]
    if selectable:
        selected_state = sorted(selectable, key=lambda state: float(state.score), reverse=not engine.config.minimize)[0]
        selected_state.metrics["baseline_selected_iteration"] = iteration
        write_state_update(engine, selected_state)
        logger.write(
            f"iteration={iteration} selected={selected_state.state_id} "
            f"{engine.config.score_key}={selected_state.score}"
        )

    iteration_best_score = best_score_from_states(actual_states, minimize=engine.config.minimize)
    best_after_iteration = updated_best_score(
        previous_best_score,
        iteration_best_score,
        minimize=engine.config.minimize,
    )
    selected_real_score = None if selected_state is None else selected_state.score
    score_delta = None
    if selected_real_score is not None and finite_score(selected_real_score) and previous_best_score is not None:
        score_delta = float(selected_real_score) - float(previous_best_score)
    progress.status(
        f"iteration {iteration} result selected={None if selected_state is None else selected_state.state_id} "
        f"{engine.config.score_key}={format_optional_float(selected_real_score)} "
        f"best={format_optional_float(best_after_iteration)}"
    )

    return {
        "iteration": iteration,
        "root_state_id": root.state_id,
        "seed_path": str(engine.config.seed_train_path),
        "seed_policy": args.seed_policy,
        "generated_state_ids": [state.state_id for state in generated_states],
        "actual_state_ids": [state.state_id for state in actual_states],
        "selected_state_id": None if selected_state is None else selected_state.state_id,
        "selected_real_score": selected_real_score,
        "score_key": engine.config.score_key,
        "previous_best_score": previous_best_score,
        "iteration_best_score": iteration_best_score,
        "best_score_after_iteration": best_after_iteration,
        "selected_score_delta_vs_previous_best": score_delta,
        "selected_improved_previous_best": (
            None
            if selected_real_score is None or previous_best_score is None
            else is_better(selected_real_score, previous_best_score, minimize=engine.config.minimize)
        ),
        "generated_count": len(generated_states),
        "real_evaluations": real_evaluations,
        "actual_states": actual_states,
        "selected_state": selected_state,
    }


def choose_seed_path(
    args: argparse.Namespace,
    original: Path,
    best_actual: SearchState | None,
    latest_actual: SearchState | None,
) -> Path:
    if args.seed_policy == "best" and best_actual is not None:
        return best_actual.train_path
    if args.seed_policy == "latest" and latest_actual is not None:
        return latest_actual.train_path
    return original


def build_task_context(project_root: Path, args: argparse.Namespace) -> str:
    parts = [DEFAULT_TASK_CONTEXT]
    if args.context_file is not None:
        context_path = args.context_file if args.context_file.is_absolute() else project_root / args.context_file
        parts.append(context_path.read_text(encoding="utf-8").strip())
    if args.context.strip():
        parts.append(args.context.strip())
    parts.append(
        "Baseline search note: every generated candidate is executed with the real evaluation command. "
        "There is no GP, surrogate model, acquisition function, or predicted-score filtering in this run. "
        "Later prompts may include only observed outcomes from actual executions."
    )
    return "\n\n".join(part for part in parts if part)


def default_run_name(args: argparse.Namespace, train_file: Path) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return "_".join(
        [
            "baseline",
            safe_path_tag(args.generator),
            safe_path_tag(train_file.stem, default="train"),
            f"b{max(1, args.breadth)}",
            f"i{max(0, args.iterations)}",
            f"e{max(1, args.num_edits_per_step)}",
            stamp,
        ]
    )


def estimate_progress_total(args: argparse.Namespace) -> int:
    iterations = max(0, int(args.iterations))
    breadth = max(1, int(args.breadth))
    generated_total = iterations * breadth
    if args.skip_eval:
        return generated_total
    real_total = iterations * breadth
    if args.evaluate_root and iterations > 0:
        real_total += 1
    if args.max_real_evaluations > 0:
        real_total = min(real_total, max(0, int(args.max_real_evaluations)))
    return generated_total + real_total


def resolve_resume_info(resume_arg: Path, project_root: Path) -> dict[str, Any]:
    path = resume_arg if resume_arg.is_absolute() else project_root / resume_arg
    if path.is_file():
        if path.name == "baseline_summary.json":
            baseline_summary_path = path
            run_dir = path.parent
        elif path.name == "summary.json":
            run_dir = path.parent
            baseline_summary_path = run_dir / "baseline_summary.json"
        else:
            raise SystemExit("--resume-from must point to a baseline run directory, baseline_summary.json, or summary.json.")
    else:
        run_dir = path
        baseline_summary_path = run_dir / "baseline_summary.json"
    if not run_dir.exists():
        raise SystemExit(f"resume run directory does not exist: {run_dir}")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"resume run is missing summary.json: {summary_path}")
    summary = load_json_object(summary_path)
    baseline_summary = load_json_object(baseline_summary_path) if baseline_summary_path.exists() else {}
    iterations = baseline_summary.get("iterations")
    if not isinstance(iterations, list):
        iterations = []
    completed_iterations = [
        int(item.get("iteration"))
        for item in iterations
        if isinstance(item, dict) and isinstance(item.get("iteration"), int)
    ]
    if completed_iterations:
        next_iteration = max(completed_iterations) + 1
    else:
        next_iteration = infer_next_iteration_from_states(run_dir)
    best_state_id = baseline_summary.get("best_state_id") or summary.get("best_state_id")
    best_score = baseline_summary.get("best_score", summary.get("best_score"))
    real_evaluations = baseline_summary.get("real_evaluations")
    if real_evaluations is None:
        real_evaluations = int(summary.get("evaluation_count") or 0)
    return {
        "run_dir": run_dir.resolve(),
        "summary": summary,
        "baseline_summary": baseline_summary,
        "iterations": iterations,
        "next_iteration": next_iteration,
        "best_state_id": best_state_id,
        "best_score": best_score,
        "real_evaluations": int(real_evaluations or 0),
    }


def apply_resume_defaults(args: argparse.Namespace, resume_info: dict[str, Any], project_root: Path) -> None:
    summary_args = resume_info.get("summary", {}).get("args")
    if not isinstance(summary_args, dict):
        summary_args = {}
    defaults = {
        "breadth": 1,
        "seed_policy": "best",
        "max_generated_per_iteration": 128,
        "evaluate_root": False,
        "skip_eval": False,
        "max_real_evaluations": 0,
        "eval_command": "uv run python {train_path}",
        "eval_shell": False,
        "timeout_seconds": 900,
        "score_key": "val_bpb",
        "maximize": False,
        "failure_score": 1.0e9,
        "generator": "api",
        "llm_url": os.environ.get("TTS_LLM_URL", "http://127.0.0.1:52307/v1"),
        "llm_model_name": os.environ.get("TTS_LLM_MODEL", "Qwen3-Coder-30B-A3B-Instruct"),
        "api_key": os.environ.get("TTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "max_tokens": 4096,
        "temperature": 0.7,
        "logprobs": False,
        "top_logprobs": None,
        "disable_thinking": False,
        "concurrency": 1,
        "num_edits_per_step": 1,
        "prompt_max_chars": 100_000,
        "context_file": None,
        "context": "",
        "instruction": "",
        "feedback_max_rows": 12,
        "response_log_chars": 20_000,
        "operation_schema": None,
        "operation_retries": 2,
        "max_operations_per_step": 2,
    }
    for name, default in defaults.items():
        if getattr(args, name, default) != default:
            continue
        if name not in summary_args:
            continue
        value = summary_args[name]
        if name in {"train_file", "context_file", "operation_schema"} and value is not None:
            value = Path(value)
            if not value.is_absolute():
                value = project_root / value
        setattr(args, name, value)
    if args.feedback_tsv is None:
        feedback_value = summary_args.get("feedback_tsv") or resume_info.get("baseline_summary", {}).get("feedback_tsv")
        if isinstance(feedback_value, str) and feedback_value:
            args.feedback_tsv = Path(feedback_value)
    if args.operation_schema is None:
        schema_value = summary_args.get("operation_schema") or summary_args.get("operation_schema_path")
        if isinstance(schema_value, str) and schema_value:
            args.operation_schema = Path(schema_value)
    train_value = summary_args.get("train_file")
    if isinstance(train_value, str) and train_value and args.train_file == Path("train.py"):
        args.train_file = Path(train_value)


def load_existing_states(engine: SearchEngine, out_dir: Path) -> None:
    state_paths = sorted((out_dir / "states").glob("state_*/meta.json"), key=lambda path: state_id_sort_key(path.parent.name))
    states: list[SearchState] = []
    max_index = -1
    for meta_path in state_paths:
        data = load_json_object(meta_path)
        state_id = str(data.get("state_id") or meta_path.parent.name)
        state = SearchState(
            state_id=state_id,
            parent_id=data.get("parent_id"),
            depth=int(data.get("depth") or 0),
            workdir=Path(data.get("workdir") or meta_path.parent),
            train_path=Path(data.get("train_path") or meta_path.parent / "train.py"),
            status=str(data.get("status") or "created"),
            score=data.get("score"),
            metrics=data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
            token_usage=int(data.get("token_usage") or 0),
            token_usage_detail=data.get("token_usage_detail") if isinstance(data.get("token_usage_detail"), dict) else {},
            elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
            error=data.get("error"),
            description=str(data.get("description") or ""),
            prompt_path=None if data.get("prompt_path") is None else Path(data["prompt_path"]),
            response_path=None if data.get("response_path") is None else Path(data["response_path"]),
            patch_path=None if data.get("patch_path") is None else Path(data["patch_path"]),
            llm_response=str(data.get("llm_response") or ""),
            llm_response_truncated=bool(data.get("llm_response_truncated")),
            edits=data.get("edits") if isinstance(data.get("edits"), list) else [],
        )
        states.append(state)
        order, _ = state_id_sort_key(state_id)
        if order < 10**12:
            max_index = max(max_index, order)
    engine.states = states
    engine._counter = max_index + 1


def load_feedback_memory_rows(feedback_memory: FeedbackMemory, feedback_path: Path) -> None:
    if not feedback_path.exists():
        return
    lines = feedback_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return
    header = lines[0].split("\t")
    if header != feedback_memory.columns:
        return
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        row = {column: values[index] if index < len(values) else "" for index, column in enumerate(header)}
        rows.append(row)
    feedback_memory.rows = rows


def state_from_id(engine: SearchEngine, state_id: Any) -> SearchState | None:
    if not isinstance(state_id, str) or not state_id:
        return None
    for state in engine.states:
        if state.state_id == state_id:
            return state
    return None


def latest_evaluated_child_state(engine: SearchEngine) -> SearchState | None:
    candidates = [
        state
        for state in engine.states
        if state.parent_id is not None and state.score is not None and finite_score(state.score)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda state: state_id_sort_key(state.state_id))[-1]


def infer_next_iteration_from_states(run_dir: Path) -> int:
    max_iteration = 0
    for meta_path in (run_dir / "states").glob("state_*/meta.json"):
        data = load_json_object(meta_path)
        metrics = data.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("baseline_iteration"), int):
            max_iteration = max(max_iteration, int(metrics["baseline_iteration"]))
    return max_iteration + 1


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def state_id_sort_key(state_id: str) -> tuple[int, str]:
    prefix, _, suffix = state_id.rpartition("_")
    if prefix == "state" and suffix.isdigit():
        return int(suffix), state_id
    return 10**12, state_id


def write_baseline_summary(
    out_dir: Path,
    summary_path: Path,
    log_path: Path,
    feedback_path: Path,
    iteration_records: list[dict[str, Any]],
    best_actual: SearchState | None,
    real_evaluations: int,
) -> None:
    serializable_records = []
    for record in iteration_records:
        item = {
            key: value
            for key, value in record.items()
            if key not in {"actual_states", "selected_state"}
        }
        serializable_records.append(item)
    payload = {
        "summary": str(summary_path),
        "log": str(log_path),
        "feedback_tsv": str(feedback_path),
        "best_state_id": None if best_actual is None else best_actual.state_id,
        "best_score": None if best_actual is None else best_actual.score,
        "real_evaluations": real_evaluations,
        "iterations": serializable_records,
    }
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
