#!/usr/bin/env python3
"""Measure the reference val_bpb that the RL reward is scored against.

The ``improvement`` reward pays a policy for getting below a **fixed**
reference. That reference has to be measured before training starts: if it is
left unset the episode's own first proposal becomes the bar, which pays the
policy for opening badly (see ``ldm_rl.env.LDMEnv._reward``).

The reference config is the instance's defaults, which -- as long as pinned
knobs stay at their default values -- is the same configuration for every
instance. So this costs **one real run per wall-clock budget**, not one per
episode, and the result is cached like any other evaluation.

    python -m tasks.nanogpt.scripts.rl_reference \
        --budgets 300 --output rl/nanogpt_reference.json \
        --output-dir /scratch/nanogpt_rl --eval-gpus 0,1,2,3

Re-running is free once the cache is warm, so it is safe to put this in a
launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):  # allow `python tasks/nanogpt/scripts/rl_reference.py`
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    )

from tasks.nanogpt.core import rl_knobs as knobs_mod, rl_real, rl_task  # noqa: E402


def add_evaluator_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared with the episode generator and any manual evaluation."""

    parser.add_argument(
        "--output-dir",
        default=os.environ.get("NANOGPT_RL_OUTPUT_DIR", ""),
        help="where run directories, the GPU locks and the cache live",
    )
    parser.add_argument(
        "--eval-cache-file",
        default="",
        help="shared evaluation cache (default: <output-dir>/eval_cache.jsonl)",
    )
    parser.add_argument(
        "--eval-gpus",
        default=os.environ.get("NANOGPT_EVAL_GPUS", ""),
        help="comma-separated GPU indices usable for evaluation, e.g. 4,5,6,7",
    )
    parser.add_argument("--repo-root", default="", help="LDM repo root")
    parser.add_argument(
        "--autoresearch-cache-dir",
        default=os.environ.get("AUTORESEARCH_CACHE_DIR", ""),
        help="prepared shards + tokenizer (prepare.py CACHE_DIR)",
    )
    parser.add_argument(
        "--timeout-slack",
        type=float,
        default=300.0,
        help="wall-clock allowance over the budget before a run is a timeout",
    )


def evaluator_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "output_dir": args.output_dir or None,
        "eval_cache_file": args.eval_cache_file or None,
        "eval_gpus": args.eval_gpus or None,
        "repo_root": args.repo_root or None,
        "autoresearch_cache_dir": args.autoresearch_cache_dir or None,
        "timeout_slack": args.timeout_slack,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--budgets",
        default="300",
        help="comma-separated wall-clock budgets to measure a reference for",
    )
    parser.add_argument("--output", default="", help="write a JSON map here")
    add_evaluator_args(parser)
    args = parser.parse_args(argv)

    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    if not budgets:
        parser.error("--budgets must list at least one budget")

    base = evaluator_kwargs(args)
    references: dict[str, float] = {}
    failures: list[str] = []

    for budget in budgets:
        instance = rl_task.KnobInstance.from_kwargs(time_budget=budget)
        runner = rl_real.build_runner(**{**base, "time_budget": budget})
        config = rl_real.reference_config(instance)
        print(
            f"[reference] budget={budget:.0f}s "
            f"{knobs_mod.describe_config(config)}",
            flush=True,
        )
        outcome = rl_real.measure_reference(instance, runner)
        if not outcome.ok:
            failures.append(
                f"budget {budget:.0f}s: {outcome.failure_kind}: {outcome.error}"
            )
            print(
                f"[reference] budget={budget:.0f}s FAILED "
                f"({outcome.failure_kind}) {outcome.error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        references[f"{budget:.0f}"] = float(outcome.val_bpb)
        print(
            f"[reference] budget={budget:.0f}s val_bpb={outcome.val_bpb:.6f} "
            f"({'cached' if outcome.cached else f'{outcome.wall_seconds:.0f}s'})",
            flush=True,
        )

    if args.output and references:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(references, handle, indent=1, sort_keys=True)
        print(f"[reference] wrote {args.output}", flush=True)

    if failures:
        # A missing reference is fatal for training, not a warning: without it
        # the reward silently reverts to the sandbaggable moving baseline.
        print("\n[reference] could not measure:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
