"""Lightweight dependency preflight checks for LDM-TTS tasks."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ldm_tts.registration.registry import (
    REPOSITORY_RELATIVE_PREFIXES,
    TaskRegistrationError,
    get_task_definition,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DependencyCheck:
    """One dependency preflight result."""

    task: str
    name: str
    status: str
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def ok(task: str, name: str, message: str, detail: str = "") -> DependencyCheck:
    return DependencyCheck(task=task, name=name, status="ok", message=message, detail=detail)


def warn(task: str, name: str, message: str, detail: str = "") -> DependencyCheck:
    return DependencyCheck(task=task, name=name, status="warn", message=message, detail=detail)


def fail(task: str, name: str, message: str, detail: str = "") -> DependencyCheck:
    return DependencyCheck(task=task, name=name, status="fail", message=message, detail=detail)


def skip(task: str, name: str, message: str, detail: str = "") -> DependencyCheck:
    return DependencyCheck(task=task, name=name, status="skip", message=message, detail=detail)


def check_plan(plan: dict[str, Any], *, include_optional: bool = True) -> list[DependencyCheck]:
    """Run the dependency hook declared by a registered task manifest."""

    task = str(plan.get("task", "")).strip()
    try:
        definition = get_task_definition(task)
    except KeyError:
        return [warn(task or "unknown", "task", f"No registered task {task!r}.")]
    except TaskRegistrationError as exc:
        return [fail(task or "unknown", "task registration", str(exc))]
    if not definition.dependency_checker:
        return [
            warn(
                task,
                "task",
                "No dependency checker is declared in the task manifest.",
            )
        ]
    try:
        module_name, function_name = definition.dependency_checker.split(":", 1)
        module = importlib.import_module(module_name)
        checker = getattr(module, function_name)
    except (ImportError, AttributeError, ValueError) as exc:
        return [
            fail(
                task,
                "dependency checker",
                f"Could not load {definition.dependency_checker!r}: {exc}",
            )
        ]
    checks = checker(plan, include_optional=include_optional)
    if not isinstance(checks, list) or not all(
        isinstance(check, DependencyCheck) for check in checks
    ):
        return [
            fail(
                task,
                "dependency checker",
                "Dependency checker must return list[DependencyCheck].",
                definition.dependency_checker,
            )
        ]
    return checks


def plan_check_context(
    plan: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, str], Path, str]:
    """Normalize the runner-plan fields consumed by dependency hooks."""

    task = str(plan.get("task", ""))
    argv = [str(item) for item in plan.get("argv", [])]
    args = cli_args_to_map(argv)
    env = {**os.environ, **{str(k): str(v) for k, v in (plan.get("env_overrides") or {}).items()}}
    cwd = Path(str(plan.get("cwd") or REPO_ROOT)).resolve()
    mode = str(plan.get("mode") or "").lower()
    return task, args, env, cwd, mode


def cli_args_to_map(argv: list[str]) -> dict[str, Any]:
    """Convert runner argv into a best-effort flag map."""

    out: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        if not item.startswith("--"):
            index += 1
            continue
        key = item[2:]
        if key.startswith("no-"):
            set_arg_value(out, key[3:], False)
            index += 1
            continue
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            set_arg_value(out, key, argv[index + 1])
            index += 2
            continue
        set_arg_value(out, key, True)
        index += 1
    return out


def set_arg_value(out: dict[str, Any], key: str, value: Any) -> None:
    if key not in out:
        out[key] = value
        return
    current = out[key]
    if isinstance(current, list):
        current.append(value)
    else:
        out[key] = [current, value]


def arg_value(args: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = args.get(name)
        if isinstance(value, list):
            value = value[-1] if value else ""
        if value is not None and value is not False and str(value).strip():
            return str(value)
    return default


def bool_arg(args: dict[str, Any], name: str) -> bool:
    return bool(args.get(name))


def resolve_task_path(raw: str | None, cwd: Path) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    normalized = str(path).replace("\\", "/")
    if any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in REPOSITORY_RELATIVE_PREFIXES
    ):
        return REPO_ROOT / path
    return cwd / path


def check_llm_settings(
    task: str,
    args: dict[str, Any],
    env: dict[str, str],
    *,
    url_arg: str,
    model_arg: str,
    api_arg: str,
    url_env: tuple[str, ...],
    model_env: tuple[str, ...],
    api_env: tuple[str, ...],
    required: bool,
) -> list[DependencyCheck]:
    checks: list[DependencyCheck] = []
    url = configured_value(arg_value(args, url_arg) or first_env(env, url_env))
    model = configured_value(arg_value(args, model_arg) or first_env(env, model_env))
    api_key = configured_value(arg_value(args, api_arg) or first_env(env, api_env))

    if url:
        checks.append(ok(task, "LLM URL", "LLM base URL is configured.", url))
    elif required:
        checks.append(fail(task, "LLM URL", f"Missing LLM URL. Set --{url_arg} or one of {', '.join(url_env)}."))
    else:
        checks.append(skip(task, "LLM URL", "Mock/local path does not require an LLM URL."))

    if model:
        checks.append(ok(task, "LLM model", "LLM model name is configured.", model))
    elif required:
        checks.append(warn(task, "LLM model", f"No model configured; the task default may be used. Set --{model_arg} if needed."))
    else:
        checks.append(skip(task, "LLM model", "Mock/local path does not require an LLM model."))

    if api_key:
        checks.append(ok(task, "LLM API key", "LLM API key value is configured.", mask_secret(api_key)))
    elif url and is_local_url(url):
        checks.append(warn(task, "LLM API key", "No API key configured. Local OpenAI-compatible servers may accept EMPTY."))
    elif required:
        checks.append(fail(task, "LLM API key", f"Missing API key. Set --{api_arg} or one of {', '.join(api_env)}."))
    else:
        checks.append(skip(task, "LLM API key", "Mock/local path does not require an API key."))
    return checks


def configured_value(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", text):
        return ""
    return text


def first_env(env: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = env.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def is_local_url(url: str) -> bool:
    text = url.lower()
    return "127.0.0.1" in text or "localhost" in text or "0.0.0.0" in text


def mask_secret(value: str) -> str:
    if value == "EMPTY":
        return "EMPTY"
    return "***"


def parse_device_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def check_cuda_visibility(
    task: str,
    name: str,
    *,
    requested_device: str,
    env: dict[str, str],
    requested_devices: list[int] | None = None,
) -> DependencyCheck:
    device_text = str(requested_device or "").lower()
    if "cuda" not in device_text and not requested_devices:
        return skip(task, name, f"CUDA is not requested ({requested_device or 'cpu'}).")
    cuda_visible_is_set = "CUDA_VISIBLE_DEVICES" in env
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_is_set and cuda_visible in {"", "-1"}:
        return fail(
            task,
            name,
            "CUDA was requested, but CUDA_VISIBLE_DEVICES hides all GPUs.",
            "CUDA_VISIBLE_DEVICES=<empty>" if not cuda_visible else f"CUDA_VISIBLE_DEVICES={cuda_visible}",
        )
    visible = query_visible_gpu_ids()
    if visible is None:
        return warn(task, name, "CUDA was requested, but nvidia-smi is not available for preflight.", f"CUDA_VISIBLE_DEVICES={cuda_visible or '<unset>'}")
    if not visible:
        return fail(task, name, "CUDA was requested, but no GPUs were reported by nvidia-smi.")
    if requested_devices:
        missing = [device for device in requested_devices if device not in visible]
        if missing:
            return fail(task, name, f"Requested GPU device(s) are not visible: {missing}.", f"visible={sorted(visible)}")
    detail = f"visible={sorted(visible)} CUDA_VISIBLE_DEVICES={cuda_visible or '<unset>'}"
    return ok(task, name, "CUDA device visibility check passed.", detail)


def query_visible_gpu_ids() -> set[int] | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return set()
    ids: set[int] = set()
    for line in proc.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            ids.add(int(text))
    return ids


def load_yaml_object(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def has_failures(checks: list[DependencyCheck]) -> bool:
    return any(check.status == "fail" for check in checks)


def format_checks(checks: list[DependencyCheck]) -> str:
    labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}
    lines: list[str] = []
    for check in checks:
        prefix = labels.get(check.status, check.status.upper())
        line = f"[{prefix}] {check.task}: {check.name}: {check.message}"
        if check.detail:
            line += f" ({check.detail})"
        lines.append(line)
    return "\n".join(lines)


def checks_to_json(checks: list[DependencyCheck]) -> str:
    return json.dumps([check.to_dict() for check in checks], indent=2, sort_keys=True)
