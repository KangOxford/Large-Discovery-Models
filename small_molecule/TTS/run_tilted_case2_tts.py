#!/usr/bin/env python3
"""Run test-time search with the case2 acquisition-tilted LDM loop.

This is a small TTS-facing wrapper around
``strbo_v1.ldm_tilted_case2.loop.run_tilted_case2_search``.  It is meant
to play the same practical role as ``TTS/example_run_expanded_search.py``
does for code search: provide one command that wires together

    LLM proposal reservoir -> BO/EHVI tilted selection -> environment scoring

for the molecule task.

Example smoke run without external services:

    python TTS/run_tilted_case2_tts.py \
        --mock \
        --method m1_stratified_direct_llm_oversample_sir \
        --budget 8 \
        --m1-k-direct-llm 16 \
        --trajectory-dir TTS/runs/case2_mock

Example real run:

    python TTS/run_tilted_case2_tts.py \
        --method m1_stratified_direct_llm_oversample_sir \
        --init-strategy llm_cold_start \
        --budget 80 \
        --m1-k-direct-llm 512 \
        --max-candidates-per-round 256 \
        --kernel sk \
        --gp-device cpu \
        --llm-url http://127.0.0.1:52307/v1 \
        --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
        --llm-max-retries 20 \
        --llm-retry-wait-seconds 10 \
        --vina-bin /path/to/vina \
        --trajectory-dir TTS/runs/case2_real

Resume an interrupted run:

    python TTS/run_tilted_case2_tts.py \
        --resume-from TTS/runs/case2_real \
        --budget 160
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_NN_MODEL_PATH = "activity_modeling/best_g12d_model.joblib"
QWEN35_DEFAULT_SAMPLING = {
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
VALID_METHODS = {
    "m1_direct_llm_sir",
    "m1_stratified_direct_llm_sir",
    "m1_stratified_direct_llm_oversample_sir",
    "m1_stratified_direct_llm_only",
    "m1_llm_one_step",
    "m1_llm_seed_analog_oversample_sir",
}


class ExpandingMockCase2LLM:
    """Deterministic molecule-emitting mock for local loop smoke tests."""

    def __init__(self) -> None:
        self.model_name = "mock-case2-tts"
        self.call_log: list[dict[str, object]] = []
        self._counter = 0

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self._counter += 1
        base_len = 3 + (self._counter % 10)
        base = "C" * base_len
        if '"seeds"' in user:
            payload = {
                "seeds": [
                    {"smiles": base, "budget": 8, "intent": "mock local seed"},
                    {"smiles": base + "N", "budget": 8, "intent": "mock polar seed"},
                ]
            }
        else:
            payload = {
                "direct_smiles": [
                    {"smiles": base, "rationale": "alkyl"},
                    {"smiles": base + "N", "rationale": "amine"},
                    {"smiles": base + "O", "rationale": "alcohol"},
                    {"smiles": "N" + base + "N", "rationale": "diamine"},
                    {"smiles": "O" + base + "N", "rationale": "hetero"},
                ]
            }
        text = json.dumps(payload)
        self.call_log.append({
            "system": system,
            "user": user,
            "response": text,
            "idx": self._counter - 1,
        })
        return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps({
            "config": planned_config_json(args, output_dir),
            "output_dir": str(output_dir),
            "mock": bool(args.mock),
        }, indent=2, sort_keys=True))
        return 0

    try:
        cfg = build_config(args, output_dir)
        from strbo_v1.ldm_tilted_case2.loop import run_tilted_case2_search

        llm = build_llm(args)
        scorer = build_mock_scorers() if args.mock else build_real_scorers(args, output_dir)
        analog_fn = build_mock_analog_fn() if args.mock else build_real_analog_fn(args, output_dir)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import the tilted case2 runtime. Install the project GP "
            "dependencies first, especially torch, gpytorch, and gauche. "
            f"Original import error: {exc}"
        ) from exc

    history, summary = run_tilted_case2_search(
        parse_seed_smiles(args.seed_smiles),
        scorer,
        analog_fn,
        config=cfg,
        llm=llm,
    )

    result = {
        "output_dir": str(output_dir.resolve()),
        "history_size": len(history),
        "best": best_observed(history, cfg.minimize),
        "summary": summary,
        "history_path": str((output_dir / "history.json").resolve()),
        "summary_path": str((output_dir / "summary.json").resolve()),
        "rounds_path": str((output_dir / "rounds.jsonl").resolve()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tilted case2 molecule test-time search."
    )
    parser.add_argument("--method", choices=sorted(VALID_METHODS), default="m1_stratified_direct_llm_oversample_sir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-smiles", default="CCO,CCN,CCC,CCCN,CCCC")
    parser.add_argument("--init-strategy", choices=["seed_smiles", "llm_cold_start"], default="llm_cold_start")
    parser.add_argument("--init-size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smiles-max-len", type=int, default=80)
    parser.add_argument("--max-candidates-per-round", type=int, default=256)
    parser.add_argument("--max-empty-reservoir-rounds", type=int, default=10)
    parser.add_argument(
        "--allow-early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow the search to stop before budget when repeated empty reservoirs hit "
            "--max-empty-reservoir-rounds. Use --no-allow-early-stop to keep retrying."
        ),
    )
    parser.add_argument("--kernel", choices=["fp", "sk"], default="sk")
    parser.add_argument("--gp-device", default="cpu")
    parser.add_argument("--gp-fit-itersteps", type=int, default=20)
    parser.add_argument("--gp-fp-n-bits", type=int, default=2048)
    parser.add_argument("--ehvi-n-samples", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight on log q0 base measure.")
    parser.add_argument("--eta", type=float, default=3.0, help="Weight on robust-z EHVI tilt.")
    parser.add_argument("--m1-k-direct-llm", type=int, default=128)
    parser.add_argument("--m1-q0-smoothing", type=float, default=None)
    parser.add_argument("--m1-analog-n-llm-seeds", type=int, default=8)
    parser.add_argument("--m1-analog-k-total", type=int, default=1024)
    parser.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--llm-model-name", default="DeepSeek-V4-Flash")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-top-p", type=float, default=None)
    parser.add_argument("--llm-top-k", type=int, default=None)
    parser.add_argument("--llm-min-p", type=float, default=None)
    parser.add_argument("--llm-presence-penalty", type=float, default=None)
    parser.add_argument("--llm-repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--qwen35-sampling-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For Qwen3.5 non-coder reasoning models, apply Qwen-style defaults "
            "for unset sampling passthroughs: top_p=0.95, top_k=20, min_p=0, "
            "presence_penalty=1.5, repetition_penalty=1.0."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Pass chat_template_kwargs.enable_thinking=false through extra_body. "
            "Useful for Qwen3.5 reasoning models served by vLLM/SGLang."
        ),
    )
    parser.add_argument(
        "--llm-extra-body-json",
        default="",
        help=(
            "Raw JSON object merged into the OpenAI SDK extra_body request field "
            "for provider-specific parameters."
        ),
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=2,
        help="Retry attempts after the initial LLM JSON attempt fails.",
    )
    parser.add_argument(
        "--llm-retry-wait-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between failed LLM JSON attempts.",
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock LLM/scorers/analog generator.")
    parser.add_argument("--trajectory-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the selected trajectory directory. Existing history.json is used "
            "when present; otherwise rounds.jsonl is replayed."
        ),
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help=(
            "Resume from an existing trajectory directory, or from a file inside it "
            "(summary.json, history.json, rounds.jsonl, or config.json). Implies --resume."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console progress verbosity. Default hides per-chunk LLM details.",
    )
    parser.add_argument(
        "--debug-llm-chunks",
        action="store_true",
        help="Show individual LLM chunk start/success logs.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vina-bin", default=os.environ.get("VINA_BIN", ""))
    parser.add_argument("--nn-model-path", default=DEFAULT_NN_MODEL_PATH)
    parser.add_argument("--reasyn-repo", default=os.environ.get("REASYN_HOME", os.environ.get("REASYN_REPO", "")))
    parser.add_argument("--reasyn-model-path", default=os.environ.get("REASYN_MODEL_PATH", ""))
    parser.add_argument("--reasyn-devices", default="0")
    parser.add_argument("--reasyn-time-limit", type=int, default=1800)
    return parser.parse_args(argv)


def configure_logging(args: argparse.Namespace) -> None:
    root_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.captureWarnings(True)
    direct_logger = logging.getLogger("strbo_v1.ldm_tilted_case2.methods.direct_llm")
    if args.debug_llm_chunks:
        direct_logger.setLevel(logging.DEBUG)
    elif root_level <= logging.DEBUG:
        direct_logger.setLevel(logging.INFO)
    else:
        direct_logger.setLevel(logging.NOTSET)
    logging.getLogger("numexpr").setLevel(logging.WARNING)


def build_config(args: argparse.Namespace, output_dir: Path) -> TiltedLDMCase2Config:
    from strbo_v1.gp import GPConfig
    from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config

    return TiltedLDMCase2Config(
        method=args.method,
        init_size=args.init_size,
        init_strategy=args.init_strategy,
        budget=args.budget,
        batch_size=args.batch_size,
        smiles_max_len=args.smiles_max_len,
        max_candidates_per_round=args.max_candidates_per_round,
        max_empty_reservoir_rounds=args.max_empty_reservoir_rounds,
        allow_early_stop=bool(args.allow_early_stop),
        gp_config=GPConfig(
            impl="smiles-strkernel" if args.kernel == "sk" else "fingerprint+tanimoto",
            device=args.gp_device,
            fit_n_itersteps=args.gp_fit_itersteps,
            fp_n_bits=args.gp_fp_n_bits,
            smiles_maxlen=args.smiles_max_len,
        ),
        ehvi_n_samples=args.ehvi_n_samples,
        alpha_base_measure=args.alpha,
        eta_ehvi_tilt=args.eta,
        m1_k_direct_llm=args.m1_k_direct_llm,
        m1_q0_smoothing=resolve_q0_smoothing(args),
        m1_analog_n_llm_seeds=args.m1_analog_n_llm_seeds,
        m1_analog_k_total=args.m1_analog_k_total,
        llm_max_retries=args.llm_max_retries,
        llm_retry_wait_seconds=args.llm_retry_wait_seconds,
        trajectory_dir=args.trajectory_dir or str(output_dir),
        resume_from_trajectory=bool(args.resume),
        seed=args.seed,
        verbose=bool(args.verbose),
    )


def resolve_q0_smoothing(args: argparse.Namespace) -> float:
    if args.m1_q0_smoothing is not None:
        return float(args.m1_q0_smoothing)
    if args.method in {
        "m1_stratified_direct_llm_sir",
        "m1_stratified_direct_llm_oversample_sir",
        "m1_llm_seed_analog_oversample_sir",
    }:
        return 0.5
    return 0.0


def resolve_output_dir(args: argparse.Namespace) -> Path:
    resume_dir = resolve_resume_dir(args)
    if resume_dir is not None:
        return resume_dir
    raw = args.trajectory_dir or args.output_dir
    if raw:
        path = Path(raw)
    else:
        suffix = "mock" if args.mock else "real"
        path = Path("TTS") / "runs" / "tilted_case2" / f"{args.method}_{suffix}_seed={args.seed}"
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_resume_dir(args: argparse.Namespace) -> Path | None:
    if not args.resume_from:
        return None
    path = Path(args.resume_from)
    path = path if path.is_absolute() else REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"--resume-from path does not exist: {path}")
    if path.is_file():
        path = path.parent
    explicit = args.trajectory_dir or args.output_dir
    if explicit:
        explicit_path = Path(explicit)
        explicit_path = explicit_path if explicit_path.is_absolute() else REPO_ROOT / explicit_path
        if explicit_path.resolve() != path.resolve():
            raise SystemExit(
                "--resume-from cannot be combined with a different --trajectory-dir or --output-dir"
            )
    args.resume = True
    args.trajectory_dir = str(path)
    args.output_dir = str(path)
    return path


def build_llm(args: argparse.Namespace):
    if args.mock:
        return ExpandingMockCase2LLM()
    from strbo_v1.llm_advisor.client import OpenAIChatClient
    from strbo_v1.llm_advisor.config import LLMClientConfig

    return OpenAIChatClient(
        LLMClientConfig(
            api_key=args.api_key,
            base_url=args.llm_url.rstrip("/"),
            model=args.llm_model_name,
        ),
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        top_p=resolve_llm_top_p(args),
        presence_penalty=resolve_llm_presence_penalty(args),
        extra_body=build_llm_extra_body(args),
    )


def is_qwen35_reasoning_model(model_name: str) -> bool:
    text = str(model_name).lower().replace("_", "-").replace("/", "-")
    return "qwen3.5" in text and "coder" not in text


def use_qwen35_sampling_defaults(args: argparse.Namespace) -> bool:
    return bool(args.qwen35_sampling_defaults) and is_qwen35_reasoning_model(args.llm_model_name)


def resolve_llm_top_p(args: argparse.Namespace) -> float | None:
    if args.llm_top_p is not None:
        return float(args.llm_top_p)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["top_p"])
    return None


def resolve_llm_top_k(args: argparse.Namespace) -> int | None:
    if args.llm_top_k is not None:
        return int(args.llm_top_k)
    if use_qwen35_sampling_defaults(args):
        return int(QWEN35_DEFAULT_SAMPLING["top_k"])
    return None


def resolve_llm_min_p(args: argparse.Namespace) -> float | None:
    if args.llm_min_p is not None:
        return float(args.llm_min_p)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["min_p"])
    return None


def resolve_llm_presence_penalty(args: argparse.Namespace) -> float | None:
    if args.llm_presence_penalty is not None:
        return float(args.llm_presence_penalty)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["presence_penalty"])
    return None


def resolve_llm_repetition_penalty(args: argparse.Namespace) -> float | None:
    if args.llm_repetition_penalty is not None:
        return float(args.llm_repetition_penalty)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["repetition_penalty"])
    return None


def build_llm_extra_body(args: argparse.Namespace) -> dict[str, Any] | None:
    extra_body = parse_extra_body_json(args.llm_extra_body_json)
    top_k = resolve_llm_top_k(args)
    min_p = resolve_llm_min_p(args)
    repetition_penalty = resolve_llm_repetition_penalty(args)
    if top_k is not None:
        extra_body["top_k"] = top_k
    if min_p is not None:
        extra_body["min_p"] = min_p
    if repetition_penalty is not None:
        extra_body["repetition_penalty"] = repetition_penalty
    if args.disable_thinking:
        chat_template_kwargs = extra_body.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        chat_template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    return extra_body or None


def parse_extra_body_json(raw: str) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--llm-extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--llm-extra-body-json must decode to a JSON object")
    return dict(payload)


def build_mock_scorers():
    def vina(smiles_list: Sequence[str]) -> list[float]:
        return [-0.1 * float(len(str(smiles))) for smiles in smiles_list]

    def activity(smiles_list: Sequence[str]) -> list[float]:
        return [
            5.0 + 0.45 * str(smiles).count("N") + 0.10 * str(smiles).count("C")
            for smiles in smiles_list
        ]

    return vina, activity


def build_mock_analog_fn():
    def analog_fn(seed_smiles: Sequence[str]) -> list[str]:
        out: list[str] = []
        for seed in seed_smiles:
            text = str(seed)
            out.extend([text + "C", text + "N", text + "O"])
        return out

    return analog_fn


def build_real_scorers(args: argparse.Namespace, output_dir: Path):
    from strbo_v1.objective_nn import NNScorer, NNScorerConfig
    from strbo_v1.objective_vina import VinaScorer, VinaScorerConfig

    vina_cfg = VinaScorerConfig(
        vina_bin=args.vina_bin or None,
        cache_dir=output_dir / "vina_cache",
    )
    nn_cfg = NNScorerConfig(
        model_path=args.nn_model_path,
        on_error="all_nan",
    )
    return VinaScorer(vina_cfg), NNScorer(nn_cfg)


def build_real_analog_fn(args: argparse.Namespace, output_dir: Path):
    from strbo_v1.analog import ReasynConfig, generate_analogs

    model_path = args.reasyn_model_path or (
        "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,"
        "data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
    )
    devices = [
        int(part)
        for part in str(args.reasyn_devices).split(",")
        if part.strip()
    ]
    config = ReasynConfig(
        model_path=model_path,
        reasyn_repo=args.reasyn_repo or None,
        devices=devices or [0],
        time_limit=args.reasyn_time_limit,
        temp_dir=output_dir / "reasyn_tmp",
    )

    def analog_fn(seed_smiles: Sequence[str]) -> list[str]:
        df = generate_analogs(list(seed_smiles), config)
        if df is None or len(df) == 0:
            return []
        return [str(smiles) for smiles in df["smiles"].tolist()]

    def generate_with_targets(seed_smiles: Sequence[str]):
        return generate_analogs(list(seed_smiles), config)

    analog_fn.generate_with_targets = generate_with_targets  # type: ignore[attr-defined]
    return analog_fn


def parse_seed_smiles(raw: str) -> list[str]:
    seeds = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seed-smiles produced an empty seed set")
    return seeds


def best_observed(history, minimize: Sequence[bool]) -> dict[str, object] | None:
    finite = [
        (smiles, scores)
        for smiles, scores in history
        if len(scores) == 2 and scores[0] is not None and scores[1] is not None
    ]
    if not finite:
        return None
    best_vina = min(finite, key=lambda item: float(item[1][0]))
    best_activity = max(finite, key=lambda item: float(item[1][1]))
    balanced = min(finite, key=lambda item: float(item[1][0]) - float(item[1][1]))
    return {
        "best_vina": {"smiles": best_vina[0], "scores": list(best_vina[1])},
        "best_activity": {"smiles": best_activity[0], "scores": list(best_activity[1])},
        "balanced_proxy": {"smiles": balanced[0], "scores": list(balanced[1])},
        "minimize": list(minimize),
    }


def config_to_json(cfg: TiltedLDMCase2Config) -> dict[str, object]:
    payload = dict(cfg.__dict__)
    payload["gp_config"] = dict(cfg.gp_config.__dict__)
    return payload


def planned_config_json(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    return {
        "method": args.method,
        "init_size": args.init_size,
        "init_strategy": args.init_strategy,
        "budget": args.budget,
        "batch_size": args.batch_size,
        "smiles_max_len": args.smiles_max_len,
        "max_candidates_per_round": args.max_candidates_per_round,
        "max_empty_reservoir_rounds": args.max_empty_reservoir_rounds,
        "allow_early_stop": bool(args.allow_early_stop),
        "minimize": [True, False],
        "ref_point": [0.0, 5.0],
        "gp_config": {
            "impl": "smiles-strkernel" if args.kernel == "sk" else "fingerprint+tanimoto",
            "device": args.gp_device,
            "fit_n_itersteps": args.gp_fit_itersteps,
            "fp_n_bits": args.gp_fp_n_bits,
            "smiles_maxlen": args.smiles_max_len,
        },
        "ehvi_n_samples": args.ehvi_n_samples,
        "alpha_base_measure": args.alpha,
        "eta_ehvi_tilt": args.eta,
        "m1_k_direct_llm": args.m1_k_direct_llm,
        "m1_q0_smoothing": resolve_q0_smoothing(args),
        "m1_analog_n_llm_seeds": args.m1_analog_n_llm_seeds,
        "m1_analog_k_total": args.m1_analog_k_total,
        "llm_model_name": args.llm_model_name,
        "llm_temperature": args.llm_temperature,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_top_p": resolve_llm_top_p(args),
        "llm_presence_penalty": resolve_llm_presence_penalty(args),
        "llm_extra_body": build_llm_extra_body(args),
        "llm_disable_thinking": bool(args.disable_thinking),
        "llm_qwen35_reasoning_model": is_qwen35_reasoning_model(args.llm_model_name),
        "llm_qwen35_sampling_defaults_applied": use_qwen35_sampling_defaults(args),
        "llm_max_retries": args.llm_max_retries,
        "llm_retry_wait_seconds": args.llm_retry_wait_seconds,
        "trajectory_dir": args.trajectory_dir or str(output_dir),
        "resume_from_trajectory": bool(args.resume),
        "seed": args.seed,
        "verbose": bool(args.verbose),
    }


if __name__ == "__main__":
    raise SystemExit(main())
