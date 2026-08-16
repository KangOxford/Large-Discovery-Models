"""Dependency checks for mock and official causal-discovery evaluation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

from ldm_tts.registration.dependencies import DependencyCheck, arg_value, fail, ok, resolve_task_path
def check_task_dependencies(task: str, args: dict[str, Any], env: dict[str, str], cwd: Path, *, mode: str, include_optional: bool) -> list[DependencyCheck]:
    del env, include_optional
    if mode == "mock" or bool(args.get("mock")):
        return [ok(task, "mock path", "Mock campaign has no external dependencies.")]
    checks: list[DependencyCheck] = []
    modules = ("numpy", "pandas", "pgmpy", "causallearn")
    completed = subprocess.run([sys.executable, "-c", "; ".join(f"import {name}" for name in modules)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
    checks.append(ok(task, "evaluator modules", "Evaluator Python imports required scientific modules.", ", ".join(modules)) if completed.returncode == 0 else fail(task, "evaluator modules", "Evaluator Python is missing required modules.", completed.stderr[-500:]))
    upstream_root = resolve_task_path(arg_value(args, "upstream-root"), cwd)
    if upstream_root is None:
        checks.append(fail(task, "upstream source", "Set --upstream-root to pinned MLS-Bench."))
    else:
        try:
            from tasks.causal_discovery_discrete.core.evaluator import validate_upstream_contract

            validate_upstream_contract(upstream_root)
        except Exception as exc:
            checks.append(fail(task, "upstream source", str(exc), str(upstream_root)))
        else:
            checks.append(ok(task, "upstream source", "Pinned MLS-Bench source and scoring files match.", str(upstream_root)))
    return checks
