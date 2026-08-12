#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from tasks.nanogpt.core.search_core import DEFAULT_TASK_CONTEXT, SearchConfig, SearchEngine
from ldm_tts.optimization.search import SEARCH_METHOD_ALIASES, run_search_method


METHODS = dict(SEARCH_METHOD_ALIASES)


def safe_path_tag(value: object, *, default: str = "run") -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return tag or default


def make_unique_run_dir(parent: Path, run_name: str) -> Path:
    candidate = parent / run_name
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        indexed = parent / f"{run_name}_{index:02d}"
        if not indexed.exists():
            return indexed
    raise RuntimeError(f"Could not create a unique run directory under {parent}.")


def default_run_name(args: argparse.Namespace, train_file: Path) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [
        safe_path_tag(args.method),
        safe_path_tag(args.generator),
        safe_path_tag(train_file.stem, default="train"),
        f"b{max(1, args.breadth)}",
        f"d{max(1, args.depth)}",
        f"e{max(1, args.num_edits_per_step)}",
        stamp,
    ]
    return "_".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-time search over train.py states using LLM-generated code edits.",
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="beam_search")
    parser.add_argument("--breadth", type=int, default=2, help="Children generated per expanded state.")
    parser.add_argument("--depth", type=int, default=1, help="Search depth. For best_of_n, total N is breadth*depth.")
    parser.add_argument("--beam-width", type=int, default=2, help="Beam width, or MCTS exploration constant.")
    parser.add_argument("--max-evaluations", type=int, default=0, help="Hard cap on evaluated candidates. 0 means no cap.")
    parser.add_argument("--evaluate-root", action="store_true", help="Run the seed train.py before proposing edits.")
    parser.add_argument("--skip-eval", action="store_true", help="Generate states without executing them.")

    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-file", type=Path, default=Path("train.py"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for search runs. A fresh timestamped run folder is "
            "created inside it. Default: runs."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="",
        help=(
            "Optional explicit name for the run folder created inside --out-dir. "
            "Defaults to method/generator/train/breadth/depth/edit-count/timestamp tags."
        ),
    )
    parser.add_argument("--export-best", type=Path, default=None, help="Optional path to copy the best train.py into.")

    parser.add_argument(
        "--eval-command",
        default="uv run python {train_path}",
        help=(
            "Command used to score a state. Placeholders: {train_path}, {workdir}, "
            "{diagnostics_path}, {project_root}."
        ),
    )
    parser.add_argument("--eval-shell", action="store_true", help="Run --eval-command through the shell.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--score-key", default="val_bpb", help="Metric parsed from diagnostics/logs.")
    parser.add_argument("--maximize", action="store_true", help="Treat larger score values as better.")
    parser.add_argument("--failure-score", type=float, default=1.0e9)

    parser.add_argument("--generator", choices=["api", "closed_loop", "harness", "mock", "tool_call"], default="api")
    parser.add_argument(
        "--llm-url",
        default=os.environ.get("TTS_LLM_URL") or os.environ.get("LLM_BASE_URL") or "",
        help="OpenAI-compatible base URL. Set TTS_LLM_URL or LLM_BASE_URL for API generators.",
    )
    parser.add_argument(
        "--llm-model-name",
        default=(
            os.environ.get("TTS_LLM_MODEL")
            or os.environ.get("LLM_MODEL_NAME")
            or os.environ.get("LLM_MODEL")
            or ""
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("TTS_LLM_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--logprobs", action="store_true", help="Request output-token logprobs from the LLM endpoint.")
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=None,
        help="When --logprobs is set, request up to N top alternatives per output token.",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--num-edits-per-step",
        type=int,
        default=1,
        help="Sequential LLM edit passes used to create each child state.",
    )
    parser.add_argument(
        "--eval-each-num-steps",
        type=int,
        default=1,
        help=(
            "Evaluate generated states every N search-depth steps. "
            "1 means evaluate each depth; 2 means expand two steps before evaluation."
        ),
    )
    parser.add_argument("--prompt-max-chars", type=int, default=100_000)
    parser.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help=(
            "Optional Markdown/text file appended to the built-in autoresearch task "
            "context for every coder-LLM prompt."
        ),
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional inline task context appended to the built-in prompt context.",
    )
    parser.add_argument("--instruction", default="", help="Extra instruction appended to every generation prompt.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress visualization.")
    parser.add_argument("--progress-width", type=int, default=28, help="Character width of the progress bar.")
    parser.add_argument(
        "--response-log-chars",
        type=int,
        default=20_000,
        help="Max raw LLM response chars embedded in manifest/summary/meta JSON. Negative means unlimited.",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    train_file = args.train_file
    if not train_file.is_absolute():
        train_file = project_root / train_file
    out_parent_dir = project_root / "runs" if args.out_dir is None else args.out_dir
    if not out_parent_dir.is_absolute():
        out_parent_dir = project_root / out_parent_dir
    run_name = safe_path_tag(args.run_name) if args.run_name.strip() else default_run_name(args, train_file)
    out_dir = make_unique_run_dir(out_parent_dir, run_name)
    if "{train_path}" not in args.eval_command:
        print(
            "warning: --eval-command does not contain {train_path}; evaluation will not run the generated child state.",
            file=sys.stderr,
        )
    task_context_parts = [DEFAULT_TASK_CONTEXT]
    if args.context_file is not None:
        context_path = args.context_file if args.context_file.is_absolute() else project_root / args.context_file
        task_context_parts.append(context_path.read_text(encoding="utf-8").strip())
    if args.context.strip():
        task_context_parts.append(args.context.strip())
    task_context = "\n\n".join(part for part in task_context_parts if part)

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
        eval_each_num_steps=args.eval_each_num_steps,
        prompt_max_chars=args.prompt_max_chars,
        task_context=task_context,
        extra_instruction=args.instruction,
        show_progress=not args.no_progress,
        progress_width=args.progress_width,
        response_log_chars=args.response_log_chars,
    )
    engine = SearchEngine(config)

    best = await run_search_method(
        args.method,
        engine,
        breadth=max(1, args.breadth),
        depth=max(1, args.depth),
        beam_width=max(1, args.beam_width),
        max_evaluations=None if args.max_evaluations <= 0 else args.max_evaluations,
        evaluate_root=args.evaluate_root,
    )

    summary_args = dict(vars(args))
    summary_args["run_name"] = run_name
    summary_args["run_parent_dir"] = out_parent_dir
    summary_args["run_dir"] = out_dir
    summary_path = engine.write_summary(method=args.method, args=summary_args, best=best)
    if args.export_best is not None and best is not None:
        export_path = args.export_best if args.export_best.is_absolute() else project_root / args.export_best
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best.train_path, export_path)

    payload = {
        "run_name": run_name,
        "run_parent_dir": str(out_parent_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "summary": str(summary_path.resolve()),
        "best_state_id": None if best is None else best.state_id,
        "best_score": None if best is None else best.score,
        "best_train": None if best is None else str(best.train_path.resolve()),
        "export_best": None if args.export_best is None else str((project_root / args.export_best).resolve()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
