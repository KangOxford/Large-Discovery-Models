"""RL environment adapter for the adaptive KV-cache quantization task.

Builds the ``EnvComponents`` bundle from this task's own adapters. The declared
response-space parser turns policy text into quantizer parameter specs; those
specs must be materialized into a complete ``AdaptiveKVQuantizer`` class before
admission, so the adapter supplies the materializing ``parse_action`` wrapper
(the same step the task's ``DeterministicQuantizerExpander`` performs).

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

    from tasks.llm_kv_adaptive_quantization.core import workflow as _wf
    from tasks.llm_kv_adaptive_quantization.core.candidate import (
        QuantizerCandidateDomain,
    )
    from tasks.llm_kv_adaptive_quantization.core.evaluator import (
        ContractThenMLSBenchEvaluator,
        MLSBenchEvaluator,
        MockQuantizerEvaluator,
        TensorContractEvaluator,
    )
    from tasks.llm_kv_adaptive_quantization.core.proposals import (
        materialize_quantizer_source,
        parse_quantizer_specs,
    )
    from tasks.llm_kv_adaptive_quantization.core.surrogate import (
        FEATURE_VERSION,
        QuantizerSourceEncoder,
    )

    reservoir_size = int(kwargs.get("reservoir_size", 2))
    args = _wf.parse_args(["--mock"] if mode == "mock" else [])
    args.reservoir_size = reservoir_size
    if kwargs.get("seed") is not None:
        args.seed = int(kwargs["seed"])
    spec = _wf.describe_ldm_task(args)

    domain = QuantizerCandidateDomain()
    seed_source = (args.seed_file if args.seed_file else _wf.DEFAULT_SEED).read_text(
        encoding="utf-8"
    )

    def parse_action(text: str) -> list[Any]:
        specs = parse_quantizer_specs(text, expected_count=reservoir_size)
        return [
            {"code": materialize_quantizer_source(seed_source, spec)}
            for spec in specs
        ]

    if mode == "mock":
        evaluator = MockQuantizerEvaluator()
    else:
        evaluator_python = (
            kwargs.get("evaluator_python")
            or os.environ.get("PYTHON", "")
            or os.sys.executable
        )
        upstream_root = kwargs.get("upstream_root")
        package_dir = kwargs.get("package_dir")
        if upstream_root is None or package_dir is None:
            raise ValueError(
                "real llm_kv_adaptive_quantization mode requires "
                "upstream_root and package_dir"
            )
        workloads = tuple(_wf._workloads(args))  # noqa: SLF001 - task-owned helper
        tensor_contract = TensorContractEvaluator(
            timeout_seconds=float(kwargs.get("contract_timeout", 60.0)),
            device=str(kwargs.get("contract_device", "cpu")),
            python_executable=evaluator_python,
        )
        evaluator = ContractThenMLSBenchEvaluator(
            tensor_contract,
            MLSBenchEvaluator(
                package_dir=Path(package_dir),
                upstream_root=Path(upstream_root),
                run_dir=Path(kwargs.get("run_dir") or "rl_runs/llm_kv"),
                workloads=workloads,
                devices=tuple(str(kwargs.get("devices", "0,1,2,3,4")).split(",")),
                model_id=str(kwargs.get("model_id", "Qwen/Qwen2.5-3B-Instruct")),
                max_examples=int(kwargs.get("max_examples", 0)),
                timeout_seconds=float(kwargs.get("evaluation_timeout", 34800.0)),
                cpu=bool(kwargs.get("cpu", False)),
                evaluator_python=evaluator_python,
            ),
        )
    context = {"workloads": list(_wf._workloads(args))} if mode == "real" else None  # noqa: SLF001

    encoder = QuantizerSourceEncoder()
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
