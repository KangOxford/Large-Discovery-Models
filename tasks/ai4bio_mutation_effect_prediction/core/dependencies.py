"""Dependency checks for mock and official mutation-effect evaluation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    check_cuda_visibility,
    fail,
    ok,
    resolve_task_path,
)
from tasks.ai4bio_mutation_effect_prediction.core.evaluator import (
    TASK_PATH,
    validate_official_data,
    validate_upstream_contract,
)


def check_task_dependencies(
    task: str,
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
    *,
    mode: str,
    include_optional: bool,
) -> list[DependencyCheck]:
    del include_optional
    if mode == "mock" or bool(args.get("mock")):
        return [ok(task, "mock path", "Mock campaign has no external dependencies.")]

    checks: list[DependencyCheck] = []
    executable = _python_executable(str(arg_value(args, "evaluator-python", default="")))
    if not executable:
        checks.append(fail(task, "evaluator Python", "Evaluator Python is not executable."))
    else:
        checks.append(
            ok(task, "evaluator Python", "Evaluator Python executable exists.", executable)
        )
        checks.append(_check_modules(task, executable))

    upstream_root = resolve_task_path(arg_value(args, "upstream-root"), cwd)
    data_dir = resolve_task_path(arg_value(args, "data-dir"), cwd)
    cv_dir = resolve_task_path(arg_value(args, "cv-dir"), cwd)
    if upstream_root is None:
        checks.append(fail(task, "upstream source", "Set --upstream-root to pinned MLS-Bench."))
    else:
        try:
            validate_upstream_contract(upstream_root / TASK_PATH)
        except Exception as exc:
            checks.append(fail(task, "upstream source", str(exc), str(upstream_root)))
        else:
            checks.append(
                ok(
                    task,
                    "upstream source",
                    "Pinned MLS-Bench task files match all recorded SHA-256 digests.",
                    str(upstream_root / TASK_PATH),
                )
            )
    if data_dir is None or cv_dir is None:
        checks.append(
            fail(task, "official data", "Set --data-dir and --cv-dir to official ProteinGym assets.")
        )
    elif executable:
        try:
            summary = validate_official_data(data_dir, cv_dir)
        except Exception as exc:
            checks.append(fail(task, "official data", str(exc)))
        else:
            checks.append(
                ok(
                    task,
                    "official data",
                    "All embeddings and predefined random five-fold assignments match.",
                    ", ".join(f"{name}={item['samples']}" for name, item in summary.items()),
                )
            )
    checks.append(
        check_cuda_visibility(
            task,
            "CUDA device",
            requested_device="cuda",
            env=env,
            requested_devices=[0],
        )
    )
    return checks


def _python_executable(raw: str) -> str:
    value = raw.strip() or sys.executable
    path = Path(value)
    if path.is_absolute() or "/" in value:
        return str(path.resolve()) if path.is_file() else ""
    return shutil.which(value) or ""


def _check_modules(task: str, executable: str) -> DependencyCheck:
    modules = ("numpy", "pandas", "scipy", "torch")
    try:
        completed = subprocess.run(
            [executable, "-c", "; ".join(f"import {name}" for name in modules)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return fail(task, "evaluator modules", f"Could not probe modules: {exc}")
    if completed.returncode:
        return fail(
            task,
            "evaluator modules",
            "Evaluator Python is missing required modules.",
            completed.stderr[-500:],
        )
    return ok(
        task,
        "evaluator modules",
        "Evaluator Python imports all required scientific modules.",
        ", ".join(modules),
    )
