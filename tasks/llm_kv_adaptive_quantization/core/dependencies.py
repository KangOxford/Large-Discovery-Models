"""Dependency checks for mock, tensor preflight, and real evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    check_cuda_visibility,
    check_llm_settings,
    fail,
    ok,
    resolve_task_path,
    skip,
)
from tasks.llm_kv_adaptive_quantization.core.evaluator import (
    OFFICIAL_COMMIT,
    OFFICIAL_FIXED_HARNESS_SHA256,
    TASK_PATH,
    fixed_harness_sha256,
)


CRITICAL_HASHES = {
    "instruction.md": "ba7e38f730a209e7fefa436be3e3e48c4c5a623f83af8a829f86541c44593fa2",
    "task.toml": "eaa758a600eaaa035409dd353371379ff7ba1dc446395d23ffb27684966b6b6e",
    "tests/meta/config.json": "e609c85dc81327a284bd9ac443c5d8de195a858fbd02876804a17e9ce2566dbe",
    "tests/meta/parser.py": "e800f82b38211c6447cd948d0514452a90b21a5838fa3cf4b49ca139b3d199f8",
    "tests/meta/score_spec.py": "d04e9bb0904714f593162edce204b4e11b7e0b998c649e9870b95c80cd6c3018",
}


def check_task_dependencies(
    task: str,
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
    *,
    mode: str,
    include_optional: bool,
) -> list[DependencyCheck]:
    is_mock = mode == "mock" or bool(args.get("mock"))
    is_preflight = mode == "preflight" or bool(args.get("preflight"))
    if is_mock:
        return [ok(task, "mock path", "Mock campaign has no external dependencies.")]

    checks: list[DependencyCheck] = []
    evaluator_python = _python_executable(arg_value(args, "evaluator-python"))
    checks.append(
        ok(task, "evaluator Python", "Evaluator Python executable exists.", evaluator_python)
        if evaluator_python
        else fail(task, "evaluator Python", "Configured evaluator Python is not executable.")
    )
    if evaluator_python:
        modules = ("torch",) if is_preflight else (
            "torch",
            "transformers",
            "datasets",
            "huggingface_hub",
        )
        checks.append(_check_modules(task, evaluator_python, modules))
    if is_preflight:
        checks.append(skip(task, "MLS-Bench", "Tensor preflight does not load a model or dataset."))
        return checks

    upstream_root = resolve_task_path(arg_value(args, "upstream-root"), cwd)
    package_dir = resolve_task_path(arg_value(args, "package-dir"), cwd)
    task_dir = None if upstream_root is None else upstream_root / TASK_PATH
    checks.extend(_check_upstream(task, upstream_root, task_dir))
    checks.append(_check_package_harness(task, package_dir))

    required_llm = arg_value(args, "proposal-mode", default="deterministic") == "openai"
    checks.extend(
        check_llm_settings(
            task,
            args,
            env,
            url_arg="llm-url",
            model_arg="llm-model-name",
            api_arg="api-key",
            url_env=("LDM_LLM_URL",),
            model_env=("LDM_LLM_MODEL",),
            api_env=("LDM_LLM_API_KEY",),
            required=required_llm,
        )
    )
    if not bool(args.get("cpu")):
        requested = [
            int(value)
            for value in arg_value(args, "devices", default="0,1,2,3,4").split(",")
            if value.strip().isdigit()
        ]
        checks.append(
            check_cuda_visibility(
                task,
                "CUDA devices",
                requested_device="cuda",
                env=env,
                requested_devices=requested,
            )
        )
    return checks


def _python_executable(raw: str) -> str:
    value = raw.strip() or sys.executable
    path = Path(value)
    if path.is_absolute() or "/" in value:
        return str(path.resolve()) if path.is_file() else ""
    return shutil.which(value) or ""


def _check_modules(
    task: str, executable: str, modules: tuple[str, ...]
) -> DependencyCheck:
    source = "; ".join(f"import {name}" for name in modules)
    try:
        result = subprocess.run(
            [executable, "-c", source],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return fail(task, "evaluator modules", f"Could not probe evaluator modules: {exc}")
    if result.returncode:
        return fail(
            task,
            "evaluator modules",
            "Evaluator Python is missing required modules: " + ", ".join(modules),
            result.stderr[-500:],
        )
    return ok(
        task,
        "evaluator modules",
        "Evaluator Python imports required modules.",
        ", ".join(modules),
    )


def _check_upstream(
    task: str, upstream_root: Path | None, task_dir: Path | None
) -> list[DependencyCheck]:
    if upstream_root is None or task_dir is None:
        return [fail(task, "upstream checkout", "Set --upstream-root to pinned MLS-Bench.")]
    checks: list[DependencyCheck] = []
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        head = result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        head = ""
    checks.append(
        ok(task, "upstream commit", "MLS-Bench checkout matches the immutable pin.", head)
        if head == OFFICIAL_COMMIT
        else fail(
            task,
            "upstream commit",
            f"MLS-Bench must be checked out at {OFFICIAL_COMMIT}.",
            head or str(upstream_root),
        )
    )
    mismatches = []
    for relative, expected in CRITICAL_HASHES.items():
        path = task_dir / relative
        actual = _sha256(path) if path.is_file() else "missing"
        if actual != expected:
            mismatches.append(f"{relative}={actual}")
    scoring = task_dir / "tests" / "mlsbench_src"
    if not scoring.is_dir():
        mismatches.append("tests/mlsbench_src=missing")
    checks.append(
        ok(task, "upstream task files", "Pinned task metadata and scoring source match.", str(task_dir))
        if not mismatches
        else fail(task, "upstream task files", "Pinned task files do not match.", "; ".join(mismatches))
    )
    return checks


def _check_package_harness(task: str, package_dir: Path | None) -> DependencyCheck:
    harness = None if package_dir is None else package_dir / "custom_quant_eval.py"
    if harness is None or not harness.is_file():
        return fail(task, "package harness", "Set --package-dir to transformers-kv-lab.")
    try:
        digest = fixed_harness_sha256(harness.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(task, "package harness", f"Could not validate custom_quant_eval.py: {exc}")
    if digest != OFFICIAL_FIXED_HARNESS_SHA256:
        return fail(task, "package harness", "Fixed harness region differs from the pinned source.", digest)
    return ok(task, "package harness", "Fixed harness region matches the pinned source.", str(harness))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
