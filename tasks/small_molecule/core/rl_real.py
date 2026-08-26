"""Real-mode RL adapter for the small-molecule task.

Runs in the task venv (torch + gpytorch + gauche + rdkit + sklearn). Builds the
real Vina + NN scorer pair, the real canonicalizing candidate domain, and the
EHVI-tilted GP acquisition selector + SMILES surrogate encoder.

Kept in a separate module so ``rl_adapter.py``'s mock path never imports the
heavy GP runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_NN_MODEL = (
    _REPO_ROOT / "tasks" / "small_molecule" / "resources" / "models" / "best_g12d_model.joblib"
)


def build_real_components(**kwargs: Any) -> Any:
    from ldm_rl.components import EnvComponents
    from ldm_tts.transport.parsing import (
        load_json_object,
        reject_keys,
        require_list,
        require_str,
    )

    from tasks.small_molecule.core import engine_adapters as _ea
    from tasks.small_molecule.core import workflow as _wf

    argv = [
        "--method", str(kwargs.get("method", "m1_stratified_direct_llm_sir")),
        "--kernel", str(kwargs.get("kernel", "sk")),
        "--acq", str(kwargs.get("acq", "ehvi")),
        "--acq-weights", str(kwargs.get("acq_weights", "0.5,0.5")),
        "--gp-device", str(kwargs.get("gp_device", "cpu")),
        "--gp-fit-itersteps", str(int(kwargs.get("gp_fit_itersteps", 20))),
        "--vina-exhaustiveness", str(int(kwargs.get("vina_exhaustiveness", 1))),
        "--vina-n-poses", str(int(kwargs.get("vina_n_poses", 1))),
        "--vina-max-workers", str(int(kwargs.get("vina_max_workers", 1))),
    ]
    args = _wf.parse_args(argv)

    for key in (
        "vina_bin",
        "nn_model_path",
        "vina_pdb_id",
        "vina_chain_id",
        "vina_cache_dir",
    ):
        if kwargs.get(key):
            setattr(args, key, str(kwargs[key]))
    if kwargs.get("seed") is not None:
        args.seed = int(kwargs["seed"])
    if kwargs.get("reservoir_size") is not None:
        args.max_candidates_per_round = int(kwargs["reservoir_size"])
    if not args.nn_model_path:
        args.nn_model_path = str(_DEFAULT_NN_MODEL)

    output_dir = Path(kwargs.get("output_dir", "/tmp/sm_rl_real"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _wf.build_config(args, output_dir)
    spec = _wf.describe_ldm_task(args)
    vina_fn, activity_fn = _wf.build_real_scorers(args, output_dir)
    evaluator = _ea.SmilesCandidateEvaluator(vina_fn, activity_fn)
    domain = _ea.SmilesCandidateDomain(cfg)
    encoder = _ea.SmilesSurrogateEncoder(cfg.gp_config)
    selector = _ea.TiltedAcquisitionSelector(cfg)

    # Optionally share the GP history across episodes (warm-up + all steps).
    gp_history_file = kwargs.get("gp_history_file")
    if gp_history_file:
        from tasks.small_molecule.core.rl_real_shared import (
            SharedEvaluator,
            SharedHistoryStore,
            SharedTiltedSelector,
        )

        store = SharedHistoryStore(str(gp_history_file))
        evaluator = SharedEvaluator(evaluator, store)
        selector = SharedTiltedSelector(selector, store)

    banned_keys = set(
        spec.response_spaces[0].metadata.get(
            "banned_score_keys",
            [
                "score",
                "objective_score",
                "constraint_score",
                "acquisition_score",
                "uncertainty",
                "proxy_value",
            ],
        )
    )

    def parse_action(text: str) -> list[Any]:
        data = load_json_object(text)
        reject_keys(data, banned_keys)
        rows = require_list(data, "direct_smiles")
        return [
            {
                "smiles": require_str(row, "smiles"),
                "rationale": str(row.get("rationale", "")),
            }
            for row in rows
        ]

    return EnvComponents(
        task_spec=spec,
        domain=domain,
        evaluator=evaluator,
        parse_action=parse_action,
        selector=selector,
        surrogate_encoder=encoder,
    )
