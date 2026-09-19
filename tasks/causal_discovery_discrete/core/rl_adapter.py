"""RL environment adapter for the discrete causal-discovery task.

Builds the ``EnvComponents`` bundle by reusing this task's own
``describe_ldm_task`` and core adapters. The declared response-space parser
normalizes a single payload (not raw text), so the adapter supplies the
text-shaped ``parse_action`` wrapper the RL environment expects.

The ``ldm_rl`` import is deferred to call time so this module stays importable
without the ``rl/`` directory on ``sys.path``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def build_rl_components(mode: str = "mock", **kwargs: Any) -> Any:
    from ldm_rl.components import EnvComponents
    from ldm_rl.parsing import parse_candidate_list
    from ldm_tts.optimization.gp import RBFGPUCBSelector

    from tasks.causal_discovery_discrete.core import workflow as _wf
    from tasks.causal_discovery_discrete.core.candidate import (
        CausalAlgorithmCandidateDomain,
        normalize_algorithm_spec,
    )
    from tasks.causal_discovery_discrete.core.evaluator import (
        MLSBenchCausalEvaluator,
        MockCausalEvaluator,
    )
    from tasks.causal_discovery_discrete.core.surrogate import (
        FEATURE_VERSION,
        CausalSpecEncoder,
    )

    reservoir_size = int(kwargs.get("reservoir_size", 2))
    args = _wf.parse_args(["--mock"] if mode == "mock" else [])
    args.reservoir_size = reservoir_size
    if kwargs.get("seed") is not None:
        args.seed = int(kwargs["seed"])
    spec = _wf.describe_ldm_task(args)

    domain = CausalAlgorithmCandidateDomain()

    def parse_action(text: str) -> list[Any]:
        # The declared parser normalizes one payload, not raw text, so wrap it.
        payloads = parse_candidate_list(text, expected_count=reservoir_size)
        return [normalize_algorithm_spec(item) for item in payloads]

    if mode == "mock":
        evaluator = MockCausalEvaluator()
    else:
        if kwargs.get("upstream_root") is None:
            raise ValueError("real causal_discovery mode requires upstream_root")
        evaluator = MLSBenchCausalEvaluator(
            upstream_root=kwargs["upstream_root"],
            run_dir=kwargs.get("run_dir") or Path("rl_runs/causal_discovery"),
            timeout_seconds=float(kwargs.get("evaluation_timeout", 3540.0)),
            evaluator_python=str(kwargs.get("evaluator_python") or os.sys.executable),
        )
    context = {"cases": list(_wf.OFFICIAL_CASES)} if mode == "real" else None
    encoder = CausalSpecEncoder()
    selector = RBFGPUCBSelector(
        objective_name=spec.objectives[0].name,
        beta=float(kwargs.get("acquisition_beta", 1.0)),
        feature_version=FEATURE_VERSION,
    )
    return EnvComponents(
        task_spec=spec,
        domain=domain,
        evaluator=evaluator,
        parse_action=parse_action,
        context=context,
        selector=selector,
        surrogate_encoder=encoder,
    )
