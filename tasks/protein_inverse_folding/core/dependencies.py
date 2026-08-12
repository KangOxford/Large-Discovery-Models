"""Dependency-check adapter for protein inverse folding."""

from __future__ import annotations

import importlib.util
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    check_cuda_visibility,
    check_llm_settings,
    fail,
    ok,
    plan_check_context,
    resolve_task_path,
    skip,
)


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    task, args, env, cwd, mode = plan_check_context(plan)
    checks: list[DependencyCheck] = []
    mock = mode == "mock" or bool(args.get("mock"))
    generator = arg_value(args, "generator", default="mock")
    evaluator = arg_value(args, "evaluator", default="mock")
    skip_eval = bool(args.get("skip-eval"))

    seed = resolve_task_path(arg_value(args, "seed-file", default="resources/seed_design.py"), cwd)
    if seed is not None and seed.is_file():
        checks.append(ok(task, "seed design", "Seed design exists.", str(seed)))
    else:
        checks.append(fail(task, "seed design", "Seed design is missing.", str(seed or "")))

    checks.extend(
        check_llm_settings(
            task,
            args,
            env,
            url_arg="llm-url",
            model_arg="llm-model-name",
            api_arg="api-key",
            url_env=("TTS_LLM_URL",),
            model_env=("TTS_LLM_MODEL",),
            api_env=("TTS_LLM_API_KEY", "OPENAI_API_KEY"),
            required=generator == "openai",
        )
    )

    if mock or evaluator == "mock" or skip_eval:
        checks.append(
            skip(task, "PyTorch", "Mock or evaluation-free runs do not import PyTorch.")
        )
        checks.append(
            skip(task, "CUDA", "Mock or evaluation-free runs do not require a GPU.")
        )
        checks.append(
            skip(task, "PInvBench data", "Mock or evaluation-free runs do not need datasets.")
        )
        return checks

    if importlib.util.find_spec("torch") is None:
        checks.append(
            fail(
                task,
                "PyTorch",
                "PyTorch is required for GPU smoke and benchmark evaluation.",
                "Install the task's real dependency group or use an existing GPU environment.",
            )
        )
    else:
        checks.append(ok(task, "PyTorch", "PyTorch is importable."))

    requested_devices = _int_values(args.get("gpu-device"))
    checks.append(
        check_cuda_visibility(
            task,
            "CUDA",
            requested_device="cuda",
            requested_devices=requested_devices,
            env=env,
        )
    )

    if evaluator == "gpu_smoke":
        checks.append(
            skip(task, "PInvBench data", "GPU smoke validates the tensor contract only.")
        )
        return checks

    scaffold = resolve_task_path(arg_value(args, "scaffold-path"), cwd)
    if scaffold is not None and scaffold.is_file():
        checks.append(ok(task, "MLS-Bench scaffold", "Scaffold exists.", str(scaffold)))
    else:
        checks.append(
            fail(
                task,
                "MLS-Bench scaffold",
                "Benchmark mode requires --scaffold-path to custom_invfold.py.",
                str(scaffold or ""),
            )
        )
    data_root = resolve_task_path(arg_value(args, "data-root"), cwd)
    expected = ("cath4.2", "cath4.3", "ts")
    missing = [name for name in expected if data_root is None or not (data_root / name).exists()]
    if not missing:
        checks.append(ok(task, "PInvBench data", "CATH4.2, CATH4.3, and TS data exist.", str(data_root)))
    else:
        checks.append(
            fail(
                task,
                "PInvBench data",
                f"Data root is missing benchmark directories: {', '.join(missing)}.",
                str(data_root or ""),
            )
        )
    return checks


def _int_values(value: Any) -> list[int]:
    values = value if isinstance(value, list) else [value]
    parsed: list[int] = []
    for item in values:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed
