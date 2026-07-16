from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from strbo_v1.gp import GPConfig
from strbo_v1.experiment_defaults import DEFAULT_NN_MODEL_PATH
from strbo_v1.ldm_tilted_case2.config import VALID_METHODS, TiltedLDMCase2Config
from strbo_v1.ldm_tilted_case2.loop import run_tilted_case2_search
from strbo_v1.llm_advisor.client import build_default_client_from_env


class MockCase2LLM:
    model_name = "mock-case2-llm"

    def __init__(self) -> None:
        self.call_log = []
        self.idx = 0

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self.idx += 1
        base = "C" * (3 + self.idx % 12)
        if '"seeds"' in user:
            text = json.dumps({"seeds": [{"smiles": base, "budget": 8, "intent": "local"}]})
        else:
            text = json.dumps({"direct_smiles": [
                {"smiles": base, "rationale": "chain"},
                {"smiles": base + "N", "rationale": "amine"},
                {"smiles": base + "O", "rationale": "alcohol"},
                {"smiles": "N" + base + "N", "rationale": "diamine"},
            ]})
        self.call_log.append({"system": system, "user": user, "response": text})
        return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_output_defaults(args)
    seed_smiles = parse_seed_smiles(args.seed_smiles)
    cfg = TiltedLDMCase2Config(
        method=args.method,
        init_size=args.init_size,
        init_strategy=resolve_init_strategy(args.init_strategy),
        budget=args.budget,
        batch_size=args.batch_size,
        alpha_base_measure=args.alpha,
        eta_ehvi_tilt=args.eta,
        ref_point=tuple(args.ref_point),
        gp_config=GPConfig(
            impl="fingerprint+tanimoto" if args.kernel == "fp" else "smiles-strkernel",
            device=args.gp_device,
            fit_n_itersteps=args.gp_fit_itersteps,
        ),
        trajectory_dir=args.trajectory_dir,
        seed=args.seed,
        verbose=args.verbose,
        ehvi_n_samples=args.ehvi_n_samples,
        m1_k_direct_llm=args.m1_k_direct_llm,
        m1_q0_smoothing=_m1_q0_smoothing(args),
        m1_analog_n_llm_seeds=args.m1_analog_n_llm_seeds,
        m1_analog_k_total=args.m1_analog_k_total,
        max_candidates_per_round=args.max_candidates_per_round,
        llm_max_retries=args.llm_max_retries,
        llm_retry_wait_seconds=args.llm_retry_wait_seconds,
        resume_from_trajectory=args.resume,
    )
    llm = MockCase2LLM() if args.mock else build_default_client_from_env(
        model=args.llm_model or None,
        timeout=args.llm_timeout,
    )
    analog_fn = mock_analog_fn if args.mock else build_real_analog_fn(args)
    scorer = (mock_vina, mock_activity) if args.mock else build_real_scorers(args)
    history, summary = run_tilted_case2_search(
        seed_smiles,
        scorer,
        analog_fn,
        config=cfg,
        llm=llm,
    )
    print(json.dumps({"history_size": len(history), "summary": summary}, indent=2))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(VALID_METHODS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--init-size", type=int, default=5)
    parser.add_argument(
        "--init-strategy",
        choices=["auto", "seed_smiles", "llm_cold_start"],
        default="auto",
        help=(
            "Initialization policy. auto uses LLM cold start for this case2 "
            "LLM script; seed_smiles preserves the previous fixed-seed history."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed-smiles", default="CCO,CCN,CCC,CCCN,CCCC")
    parser.add_argument("--kernel", choices=["fp", "sk"], default="fp")
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ref-point", type=float, nargs=2, default=(0.0, 5.0))
    parser.add_argument("--gp-device", default="cpu")
    parser.add_argument("--gp-fit-itersteps", type=int, default=20)
    parser.add_argument("--ehvi-n-samples", type=int, default=128)
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
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
    parser.add_argument("--m1-k-direct-llm", type=int, default=128)
    parser.add_argument("--m1-q0-smoothing", type=float, default=None)
    parser.add_argument("--m1-analog-n-llm-seeds", type=int, default=8)
    parser.add_argument("--m1-analog-k-total", type=int, default=1024)
    parser.add_argument("--max-candidates-per-round", type=int, default=256)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--trajectory-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vina-bin", default=None)
    parser.add_argument("--nn-model-path", default=DEFAULT_NN_MODEL_PATH)
    parser.add_argument("--reasyn-repo", default=None)
    parser.add_argument("--reasyn-model-path", default=None)
    parser.add_argument("--reasyn-devices", default="0")
    parser.add_argument("--reasyn-time-limit", type=int, default=120)
    return parser.parse_args(argv)


def resolve_init_strategy(raw: str) -> str:
    if raw == "auto":
        return "llm_cold_start"
    return raw


def apply_output_defaults(args: argparse.Namespace) -> None:
    if not args.output_dir:
        args.output_dir = str(default_output_dir(args))
    if not args.trajectory_dir:
        args.trajectory_dir = args.output_dir


def default_output_dir(args: argparse.Namespace) -> Path:
    return Path("output") / "case2" / f"{args.method}_seed={args.seed}"


def _m1_q0_smoothing(args: argparse.Namespace) -> float:
    if args.m1_q0_smoothing is not None:
        return float(args.m1_q0_smoothing)
    if args.method in {
        "m1_stratified_direct_llm_sir",
        "m1_stratified_direct_llm_oversample_sir",
        "m1_llm_seed_analog_oversample_sir",
    }:
        return 0.5
    return 0.0


def parse_seed_smiles(raw: str) -> list[str]:
    seeds = [part.strip() for part in raw.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seed-smiles produced an empty seed set")
    return seeds


def mock_analog_fn(seeds):
    out = []
    for seed in seeds:
        out.extend([seed + "C", seed + "N", seed + "O"])
    return out


def mock_vina(smiles_list):
    return [-float(len(smiles)) / 10.0 for smiles in smiles_list]


def mock_activity(smiles_list):
    return [5.0 + smiles.count("N") * 0.5 for smiles in smiles_list]


def build_real_scorers(args):
    from strbo_v1.objective_nn import NNScorer, NNScorerConfig
    from strbo_v1.objective_vina import VinaScorer, VinaScorerConfig

    cache_dir = Path(args.trajectory_dir) / "vina_cache"
    vina_cfg = VinaScorerConfig(vina_bin=args.vina_bin, cache_dir=cache_dir)
    nn_cfg = NNScorerConfig(model_path=args.nn_model_path, on_error="all_nan")
    return VinaScorer(vina_cfg), NNScorer(nn_cfg)


def build_real_analog_fn(args):
    from strbo_v1.analog import ReasynConfig, generate_analogs

    model_path = args.reasyn_model_path or (
        "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,"
        "data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
    )
    devices = [int(part) for part in str(args.reasyn_devices).split(",") if part.strip()]
    config = ReasynConfig(
        model_path=model_path,
        reasyn_repo=args.reasyn_repo,
        devices=devices or [0],
        time_limit=args.reasyn_time_limit,
        temp_dir=Path(args.trajectory_dir) / "reasyn_tmp",
    )

    def analog_fn(seed_smiles):
        df = generate_analogs(list(seed_smiles), config)
        if df is None or len(df) == 0:
            return []
        return [str(smiles) for smiles in df["smiles"].tolist()]

    def generate_with_targets(seed_smiles):
        return generate_analogs(list(seed_smiles), config)

    analog_fn.generate_with_targets = generate_with_targets
    return analog_fn


if __name__ == "__main__":
    raise SystemExit(main())
