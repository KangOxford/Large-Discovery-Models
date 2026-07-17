#!/usr/bin/env python3
"""Antibody task workflow for the shared LDM-TTS config runner.

This module wires the native AntBO LDM acquisition loop in
``bo.ldm_light.ldm_acq``:

    LLM warmup / strategy proposal -> GP acquisition candidate search -> Absolut scoring

The real run uses the same config/oracle path documented in the root READMEs.
The mock mode swaps in a random evaluator and deterministic fake LLM so the
control flow can be smoke-tested without Absolut or an LLM endpoint.

Example dry run:

    python -m antibody.ldm_task.procedure \
        --antigen 1ADQ_A \
        --budget 40 \
        --dry-run

Use a GP confidence-bound acquisition after warmup:

    python -m antibody.ldm_task.procedure \
        --antigen 1ADQ_A \
        --budget 200 \
        --acq lcb \
        --acq-beta 2.0

Example mock smoke run in the DGM environment:

    python -m antibody.ldm_task.procedure \
        --mock \
        --antigen SMOKE_ANTIGEN \
        --budget 4 \
        --n-init 3 \
        --parallel-budget 8 \
        --out-dir ldm_runs/antbo_tts_mock

Example real run:

    python -m antibody.ldm_task.procedure \
        --config bo/config.yaml \
        --antigens-file test_5_antigens.txt \
        --seed 42 \
        --budget 200 \
        --n-init 20 \
        --parallel-budget 600 \
        --llm-url http://127.0.0.1:52313/v1 \
        --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
        --llm-temperature 0.7 \
        --out-dir outputs/experiments/formal_5ag5seed200/antbo_tts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path) -> Path:
    for path in [start.parent, *start.parents]:
        if (path / "bo").is_dir() and (path / "bo" / "__init__.py").exists():
            return path
    raise RuntimeError(f"Could not find AntBO repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


class DeterministicAntBOTTSLLM:
    """Fake LLM for local TTS control-flow checks."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def call(self, prompt: str, temperature: float = 0.25, timeout_s: int = 30) -> str:
        self.calls.append({
            "temperature": temperature,
            "timeout_s": timeout_s,
            "prompt_prefix": prompt[:200],
        })
        if '"candidate_pool"' in prompt:
            return json.dumps({"selected": [{"id": 0, "sequence": "", "score": 1.0}]})
        return json.dumps({
            "rationale": "deterministic mock TTS search",
            "update_trust_region": "LatinHyperCubeSampling(num=8)",
        })

    def close(self) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AntBO CDRH3 test-time search through the native LDM acquisition loop."
    )
    parser.add_argument("--config", default="bo/config.yaml")
    parser.add_argument("--antigens-file", default="")
    parser.add_argument("--antigen", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--budget", "--n-evals", dest="n_evals", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--parallel-budget", type=int, default=600)
    parser.add_argument(
        "--acq",
        "--acquisition",
        dest="acq",
        choices=["ei", "lcb", "ucb"],
        default="ei",
        help=(
            "GP acquisition used after warmup. EI is minimization expected improvement; "
            "LCB/UCB use posterior bounds with --acq-beta and are internally scored so "
            "larger is better."
        ),
    )
    parser.add_argument(
        "--acq-beta",
        type=float,
        default=1.0,
        help="Exploration coefficient for LCB/UCB acquisitions.",
    )
    parser.add_argument("--out-dir", "--trajectory-dir", dest="out_dir", default="")
    parser.add_argument("--temperature", "--llm-temperature", dest="temperature", type=float, default=0.25)
    parser.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--llm-model-name", default=os.environ.get("LLM_MODEL", ""))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--history-top-k", type=int, default=10)
    parser.add_argument(
        "--disable-thinking",
        "--disable-reasoning",
        dest="disable_thinking",
        action="store_true",
        default=None,
        help="For Qwen-style reasoning models, request hidden thinking/reasoning to be disabled.",
    )
    parser.add_argument(
        "--enable-thinking",
        "--enable-reasoning",
        dest="disable_thinking",
        action="store_false",
        help="Force hidden thinking/reasoning to remain enabled, overriding LLM_DISABLE_THINKING.",
    )
    parser.add_argument("--include-antigen-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_config(path: str) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return resolve_path(args.out_dir)
    suffix = "mock" if args.mock else "real"
    return REPO_ROOT / "ldm_runs" / "antbo_tts" / f"{suffix}_seed={args.seed}"


def resolve_antigens(args: argparse.Namespace) -> list[str]:
    if args.antigen:
        return [args.antigen]
    if not args.antigens_file:
        raise SystemExit("Provide either --antigen or --antigens-file.")
    path = resolve_path(args.antigens_file)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def planned_config_json(args: argparse.Namespace, config: dict[str, Any], antigens: list[str]) -> dict[str, Any]:
    bbox = dict(config.get("bbox", {}))
    if args.mock:
        bbox["tool"] = "random"
        bbox["path"] = "/tmp"
        bbox["process"] = 1
        bbox["startTask"] = 0
    return {
        "repo_root": str(REPO_ROOT),
        "config": str(resolve_path(args.config)),
        "antigens": antigens,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "budget": args.n_evals,
        "batch_size": args.batch_size,
        "n_init": args.n_init,
        "parallel_budget": args.parallel_budget,
        "acq": args.acq,
        "acq_beta": args.acq_beta,
        "out_dir": str(resolve_out_dir(args)),
        "mock": bool(args.mock),
        "bbox": bbox,
        "llm_env": {
            "LLM_API_KEY": bool(args.api_key or os.getenv("LLM_API_KEY") or args.llm_url),
            "LLM_BASE_URL": args.llm_url or os.getenv("LLM_BASE_URL", ""),
            "LLM_MODEL": args.llm_model_name or os.getenv("LLM_MODEL", ""),
            "temperature": args.temperature,
            "disable_thinking": args.disable_thinking,
        },
    }


def configure_llm_environment(args: argparse.Namespace) -> None:
    if args.mock:
        return
    if args.llm_url:
        os.environ["LLM_BASE_URL"] = args.llm_url.rstrip("/")
    if args.llm_model_name:
        os.environ["LLM_MODEL"] = args.llm_model_name
    if args.api_key:
        os.environ["LLM_API_KEY"] = args.api_key
    elif args.llm_url and not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "EMPTY"
    if args.disable_thinking is True:
        os.environ["LLM_DISABLE_THINKING"] = "1"
    elif args.disable_thinking is False:
        os.environ["LLM_DISABLE_THINKING"] = "0"


def make_runner_args(args: argparse.Namespace, antigens_file: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(resolve_path(args.config)),
        antigens_file=str(antigens_file),
        seed=args.seed,
        n_trials=args.n_trials,
        n_evals=args.n_evals,
        batch_size=args.batch_size,
        out_root=str(out_dir),
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        history_top_k=args.history_top_k,
        parallel_budget=args.parallel_budget,
        n_init=args.n_init,
        acq=args.acq,
        acq_beta=args.acq_beta,
        include_antigen_context=args.include_antigen_context,
        fallback_random=args.fallback_random,
        disable_thinking=args.disable_thinking,
    )


def run(args: argparse.Namespace) -> list[str]:
    config = load_config(args.config)
    antigens = resolve_antigens(args)
    out_dir = resolve_out_dir(args)
    configure_llm_environment(args)

    if args.dry_run:
        print(json.dumps(planned_config_json(args, config, antigens), indent=2, sort_keys=True))
        return []

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import bo.ldm_light.ldm_acq as antbo_tts

    if args.mock:
        config = dict(config)
        config["bbox"] = {
            "antigen": "PLACEHOLDER",
            "tool": "random",
            "path": "/tmp",
            "process": 1,
            "startTask": 0,
        }
        config["tabular_search_csv"] = None
        config["seq_len"] = int(config.get("seq_len", 11))
        antbo_tts.make_llm_client = lambda: DeterministicAntBOTTSLLM()

    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="antbo_tts_") as td:
        antigens_path = Path(td) / "antigens.txt"
        antigens_path.write_text("\n".join(antigens) + "\n", encoding="utf-8")
        runner_args = make_runner_args(args, antigens_path, out_dir)
        for antigen in antigens:
            for seed in range(args.seed, args.seed + args.n_trials):
                run_dir = antbo_tts.run_one(config, antigen, seed, runner_args)
                run_dirs.append(str(Path(run_dir).resolve()))

    print(json.dumps({"run_dirs": run_dirs}, indent=2, sort_keys=True))
    return run_dirs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.acq_beta < 0:
        raise ValueError("--acq-beta must be non-negative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
