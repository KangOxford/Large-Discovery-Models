"""RL environment adapter for the mutation-effect-prediction task.

Builds the ``EnvComponents`` bundle the RL environment needs by reusing this
task's own ``describe_ldm_task`` and core adapters (candidate domain, mock/real
evaluators, surrogate encoder, GP selector). Real mode takes the same keyword
arguments the task CLI does (``upstream_root``, ``data_dir``, ``cv_dir``, ...).

The ``ldm_rl`` import is deferred to call time so this module stays importable
without the ``rl/`` directory on ``sys.path``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def build_rl_components(mode: str = "mock", **kwargs: Any) -> Any:
    from ldm_rl.components import EnvComponents
    from ldm_tts.optimization.gp import RBFGPUCBSelector

    from tasks.ai4bio_mutation_effect_prediction.core import workflow as _wf
    from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
        MutationPredictorCandidateDomain,
    )
    from tasks.ai4bio_mutation_effect_prediction.core.evaluator import (
        MLSBenchMutationEvaluator,
        MockMutationEvaluator,
    )
    from tasks.ai4bio_mutation_effect_prediction.core.surrogate import (
        FEATURE_VERSION,
        PredictorSpecEncoder,
    )

    reservoir_size = int(kwargs.get("reservoir_size", 2))
    args = _wf.parse_args(["--mock"] if mode == "mock" else [])
    args.reservoir_size = reservoir_size
    if kwargs.get("seed") is not None:
        args.seed = int(kwargs["seed"])
    spec = _wf.describe_ldm_task(args)

    domain = MutationPredictorCandidateDomain()
    if mode == "mock":
        evaluator = MockMutationEvaluator()
    else:
        missing = [
            name
            for name in ("upstream_root", "data_dir", "cv_dir")
            if kwargs.get(name) is None
        ]
        if missing:
            raise ValueError(
                "real ai4bio mode requires kwargs: " + ", ".join(missing)
            )
        evaluator = MLSBenchMutationEvaluator(
            upstream_root=kwargs["upstream_root"],
            data_dir=kwargs["data_dir"],
            cv_dir=kwargs["cv_dir"],
            run_dir=kwargs.get("run_dir") or Path("rl_runs/ai4bio"),
            timeout_seconds=float(kwargs.get("evaluation_timeout", 3540.0)),
            evaluator_python=str(kwargs.get("evaluator_python") or os.sys.executable),
        )
    context = {"assays": list(_wf.OFFICIAL_ASSAYS)} if mode == "real" else None
    encoder = PredictorSpecEncoder()
    selector = RBFGPUCBSelector(
        objective_name=spec.objectives[0].name,
        beta=float(kwargs.get("acquisition_beta", 1.0)),
        feature_version=FEATURE_VERSION,
    )
    return EnvComponents(
        task_spec=spec,
        domain=domain,
        evaluator=evaluator,
        context=context,
        selector=selector,
        surrogate_encoder=encoder,
    )
