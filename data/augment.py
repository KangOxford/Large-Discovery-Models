#!/usr/bin/env python3
"""Add expert justifications to ldm-2.0 IR or Alpaca JSON/JSONL data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.data import ExpertJustificationPipeline, OpenAICompatibleExpert


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add expert-model justifications to ldm-2.0 IR or Alpaca data. "
            "The input is read-only and output is JSONL."
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input JSON or JSONL file."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Augmented JSONL file."
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        help="Also render augmented ldm-2.0 IR to this Alpaca JSONL file.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL_NAME", "DeepSeek-V4-Flash"),
        help="Expert model name (default: LLM_MODEL_NAME or DeepSeek-V4-Flash).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL"),
        help="OpenAI-compatible endpoint (default: LLM_BASE_URL).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests.")
    parser.add_argument("--max-retries", type=int, default=3, help="Attempts per record.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, help="Process only the first N records.")
    parser.add_argument("--checkpoint", type=Path, help="Override the checkpoint path.")
    parser.add_argument(
        "--overwrite-reasoning",
        action="store_true",
        help="Replace non-empty reasoning or existing <think> blocks.",
    )
    parser.add_argument(
        "--include-reasoning-unavailable",
        action="store_true",
        help=(
            "Augment records explicitly marked reasoning_available=false. "
            "This can fabricate unsupported rationales and is disabled by default."
        ),
    )
    parser.add_argument(
        "--render",
        choices=("prose", "json"),
        default="prose",
        help="SFT rendering mode when --sft-output is used.",
    )
    parser.add_argument(
        "--strip-parent-artifact",
        action="store_true",
        help="Omit large parent artifacts from rendered SFT instructions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        print("error: LLM_API_KEY must be set in the environment", file=sys.stderr)
        return 2

    try:
        expert = OpenAICompatibleExpert(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
        )
        pipeline = ExpertJustificationPipeline(
            expert,
            workers=args.workers,
            max_retries=args.max_retries,
            overwrite_reasoning=args.overwrite_reasoning,
            include_reasoning_unavailable=args.include_reasoning_unavailable,
        )
        report = pipeline.run(
            args.input,
            args.output,
            limit=args.limit,
            checkpoint_path=args.checkpoint,
            sft_output_path=args.sft_output,
            render_mode=args.render,
            include_parent_artifact=not args.strip_parent_artifact,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {report.written} / {report.total} records to {report.output_path}; "
        f"generated={report.generated}, resumed={report.resumed}, "
        f"existing={report.skipped_existing}, "
        f"reasoning-unavailable={report.skipped_unavailable}, failed={report.failed}"
    )
    if report.sft_output_path is not None:
        print(f"rendered SFT data to {report.sft_output_path}")
    if report.failed_indices:
        print(f"failed record indices: {list(report.failed_indices)}", file=sys.stderr)
        print("rerun the same command to retry only unfinished records", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
