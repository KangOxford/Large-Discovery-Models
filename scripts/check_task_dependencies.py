#!/usr/bin/env python3
"""Preflight dependency checks for LDM-TTS experiment configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.dependency_checks import checks_to_json, format_checks, has_failures, check_plan
from ldm_tts.runner import apply_override, build_plan, expand_experiments, load_config, resolve_config_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    all_checks = []
    for raw_config in args.config:
        config_path = resolve_config_path(raw_config)
        config = load_config(config_path)
        for raw_override in args.set:
            apply_override(config, raw_override)
        experiments = expand_experiments(config, config_path)
        for experiment, path in experiments:
            plan = build_plan(experiment, path)
            checks = check_plan(plan, include_optional=not args.no_optional)
            all_checks.extend(checks)

    if args.json:
        print(checks_to_json(all_checks))
    else:
        print(format_checks(all_checks))

    return 1 if has_failures(all_checks) else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check environment variables, external binaries, model artifacts, "
            "and task-local data for one or more LDM-TTS configs."
        )
    )
    parser.add_argument("config", nargs="+", help="Experiment or suite config path.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Apply the same dotted override syntax as scripts/run_ldm_tts.py.",
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        help=(
            "Skip optional checks such as ReaSyn, and skip nanoGPT data/tokenizer "
            "checks when the plan also uses --skip-eval."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON results.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
