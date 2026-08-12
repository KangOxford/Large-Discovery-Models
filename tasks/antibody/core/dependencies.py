"""Antibody dependency checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    bool_arg,
    check_cuda_visibility,
    check_llm_settings,
    configured_value,
    fail,
    load_yaml_object,
    ok,
    resolve_task_path,
    skip,
)


def check_antibody(
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
    *,
    mode: str = "",
) -> list[DependencyCheck]:
    task = "antibody"
    mock = bool_arg(args, "mock") or mode == "mock"
    checks: list[DependencyCheck] = []
    checks.extend(check_llm_settings(
        task,
        args,
        env,
        url_arg="llm-url",
        model_arg="llm-model-name",
        api_arg="api-key",
        url_env=("LLM_BASE_URL",),
        model_env=("LLM_MODEL", "LLM_MODEL_NAME"),
        api_env=("LLM_API_KEY",),
        required=not mock,
    ))

    antigen = arg_value(args, "antigen")
    antigens_file = resolve_task_path(arg_value(args, "antigens-file"), cwd)
    if antigen:
        checks.append(ok(task, "antigen input", "Single antigen is configured.", antigen))
    elif antigens_file is not None and antigens_file.exists():
        checks.append(ok(task, "antigen input", "Antigens file exists.", str(antigens_file)))
    elif antigens_file is not None:
        checks.append(fail(task, "antigen input", "Antigens file does not exist.", str(antigens_file)))
    else:
        checks.append(fail(task, "antigen input", "Provide --antigen or --antigens-file."))

    bo_config = resolve_task_path(
        arg_value(args, "config", default="resources/default_config.yaml"), cwd
    )
    config_data: dict[str, Any] = {}
    if bo_config is not None and bo_config.exists():
        checks.append(ok(task, "AntBO config", "AntBO config exists.", str(bo_config)))
        config_data = load_yaml_object(bo_config)
    else:
        checks.append(fail(task, "AntBO config", "AntBO config does not exist.", str(bo_config or "")))

    device = "cpu" if mock else str(arg_value(args, "device") or config_data.get("device") or "cuda")
    checks.append(check_cuda_visibility(task, "AntBO device", requested_device=device, env=env))

    if mock:
        checks.append(skip(task, "Absolut", "Mock antibody runs do not need Absolut."))
    else:
        bbox = config_data.get("bbox") if isinstance(config_data.get("bbox"), dict) else {}
        absolut_value = (
            arg_value(args, "absolut-path")
            or configured_value(env.get("ABSOLUT_PATH", ""))
            or bbox.get("path")
            or ""
        )
        absolut = resolve_task_path(str(absolut_value), cwd)
        executable = absolut / "src" / "bin" / "Absolut" if absolut is not None else None
        if executable is not None and executable.is_file() and os.access(executable, os.X_OK):
            checks.append(ok(task, "Absolut", "Absolut executable is available.", str(executable)))
        else:
            checks.append(fail(
                task,
                "Absolut",
                "Absolut executable is missing or not executable. Set --absolut-path, ABSOLUT_PATH, or bbox.path to the installation root.",
                str(executable or ""),
            ))
    return checks

