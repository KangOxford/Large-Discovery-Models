"""Small-molecule dependency checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    bool_arg,
    check_cuda_visibility,
    check_llm_settings,
    fail,
    ok,
    parse_device_ids,
    resolve_task_path,
    skip,
    warn,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REASYN_ENTRYPOINT_REL = Path("reasyn") / "sampler" / "parallel.py"
DEFAULT_REASYN_MODEL_PATHS = (
    "data/trained_model/nv-reasyn-ar-166m-v2.ckpt",
    "data/trained_model/nv-reasyn-eb-174m-v2.ckpt",
)
REASYN_IMPORT_TIMEOUT_SECONDS = 30


def check_small_molecule(
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
    *,
    mode: str = "",
    include_optional: bool = True,
) -> list[DependencyCheck]:
    task = "small_molecule"
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
        model_env=("LLM_MODEL_NAME", "LLM_MODEL"),
        api_env=("LLM_API_KEY",),
        required=not mock,
    ))

    checks.append(check_cuda_visibility(
        task,
        "GP device",
        requested_device=arg_value(args, "gp-device", default="cpu"),
        env=env,
    ))

    if mock:
        checks.append(skip(task, "Vina", "Mock small-molecule runs do not need AutoDock Vina."))
        checks.append(skip(task, "G12D activity model", "Mock small-molecule runs do not need the activity model."))
        checks.append(skip(task, "ReaSyn", "Mock small-molecule runs do not need ReaSyn."))
        return checks

    checks.append(check_vina(task, arg_value(args, "vina-bin", default=env.get("VINA_BIN", "")), env))

    nn_model = resolve_task_path(
        arg_value(args, "nn-model-path", default="resources/models/best_g12d_model.joblib"),
        cwd,
    )
    if nn_model is not None and nn_model.exists():
        checks.append(ok(task, "G12D activity model", "Activity model artifact exists.", str(nn_model)))
    else:
        checks.append(fail(task, "G12D activity model", "Activity model artifact is missing.", str(nn_model or "")))

    method = arg_value(args, "method")
    reasyn_requested = "analog" in method or bool(arg_value(args, "reasyn-repo", default=env.get("REASYN_HOME") or env.get("REASYN_REPO", "")))
    if include_optional and reasyn_requested:
        checks.extend(check_reasyn(args, env, cwd))
    else:
        checks.append(skip(task, "ReaSyn", "This method/config does not request ReaSyn analog generation."))
    return checks


def check_vina(task: str, explicit: str, env: dict[str, str]) -> DependencyCheck:
    raw = explicit or env.get("VINA_BIN", "")
    path: Path | None
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
    else:
        found = shutil.which("vina")
        path = Path(found).resolve() if found else None
    if path is None:
        return fail(task, "Vina", "AutoDock Vina was not found. Set VINA_BIN, args.vina-bin, or add vina to PATH.")
    if not path.exists():
        return fail(task, "Vina", "AutoDock Vina path does not exist.", str(path))
    if not os.access(path, os.X_OK):
        return fail(task, "Vina", "AutoDock Vina path is not executable.", str(path))
    try:
        proc = subprocess.run(
            [str(path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return fail(
            task,
            "Vina",
            "AutoDock Vina --help timed out after 10 seconds.",
            str(path),
        )
    except OSError as exc:  # pragma: no cover - platform-specific subprocess errors
        return fail(
            task,
            "Vina",
            f"AutoDock Vina could not be executed: {exc}",
            str(path),
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return fail(
            task,
            "Vina",
            f"AutoDock Vina --help exited with status {proc.returncode}.",
            detail or str(path),
        )
    return ok(task, "Vina", "AutoDock Vina executable responds to --help.", str(path))


def check_reasyn(args: dict[str, Any], env: dict[str, str], cwd: Path) -> list[DependencyCheck]:
    task = "small_molecule"
    checks: list[DependencyCheck] = []
    repo = resolve_task_path(
        arg_value(args, "reasyn-repo", default=env.get("REASYN_HOME") or env.get("REASYN_REPO", "")),
        cwd,
    )
    if repo is None:
        checks.append(fail(task, "ReaSyn repo", "Missing ReaSyn repo. Set REASYN_HOME, REASYN_REPO, or args.reasyn-repo."))
        return checks
    entrypoint = repo / REASYN_ENTRYPOINT_REL
    repo_ready = entrypoint.is_file()
    if repo_ready:
        checks.append(ok(task, "ReaSyn repo", "ReaSyn checkout contains reasyn/sampler/parallel.py.", str(repo)))
    else:
        checks.append(fail(task, "ReaSyn repo", "ReaSyn checkout is missing reasyn/sampler/parallel.py.", str(entrypoint)))

    python_bin = resolve_reasyn_python(arg_value(args, "reasyn-python", default=env.get("REASYN_PYTHON") or env.get("REASYN_BIN", "")), repo)
    python_ready = False
    if python_bin is None:
        python_bin = Path(sys.executable)
        python_ready = python_bin.is_file() and os.access(python_bin, os.X_OK)
        checks.append(warn(
            task,
            "ReaSyn Python",
            "No dedicated ReaSyn interpreter found; probing the current interpreter.",
            str(python_bin),
        ))
    elif not python_bin.exists():
        checks.append(fail(
            task,
            "ReaSyn Python",
            "ReaSyn Python interpreter does not exist.",
            str(python_bin),
        ))
    elif not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        checks.append(fail(
            task,
            "ReaSyn Python",
            "ReaSyn Python path is not an executable file.",
            str(python_bin),
        ))
    else:
        python_ready = True
        checks.append(ok(task, "ReaSyn Python", "ReaSyn Python interpreter is executable.", str(python_bin)))

    if repo_ready and python_ready:
        checks.append(check_reasyn_import(task, repo, python_bin, env))
    else:
        checks.append(skip(
            task,
            "ReaSyn import",
            "Import probe requires a valid checkout and Python interpreter.",
        ))

    model_path = arg_value(args, "reasyn-model-path", default=env.get("REASYN_MODEL_PATH", ""))
    model_parts = [part.strip() for part in model_path.split(",") if part.strip()] if model_path else list(DEFAULT_REASYN_MODEL_PATHS)
    missing_models: list[str] = []
    for item in model_parts:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = repo / path
        if not path.is_file() or path.stat().st_size == 0:
            missing_models.append(str(path))
    if len(model_parts) != 2:
        checks.append(fail(task, "ReaSyn checkpoints", "ReaSyn model path must resolve to exactly two checkpoints.", ",".join(model_parts)))
    elif missing_models:
        checks.append(fail(
            task,
            "ReaSyn checkpoints",
            "Missing or empty ReaSyn AR/Edit Bridge checkpoint(s).",
            ", ".join(missing_models),
        ))
    else:
        checks.append(ok(task, "ReaSyn checkpoints", "ReaSyn AR/Edit Bridge checkpoints exist.", ", ".join(model_parts)))

    checks.append(check_cuda_visibility(
        task,
        "ReaSyn CUDA devices",
        requested_device="cuda",
        requested_devices=parse_device_ids(arg_value(args, "reasyn-devices", default="0")),
        env=env,
    ))
    return checks


def check_reasyn_import(
    task: str,
    repo: Path,
    python_bin: Path,
    env: dict[str, str],
) -> DependencyCheck:
    """Probe the imports used by the ReaSyn bridge without loading checkpoints."""

    probe = (
        "import sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from reasyn.chem.mol import Molecule; "
        "import reasyn.sampler.parallel; "
        "Molecule('CCO'); "
        "print('ReaSyn import OK')"
    )
    try:
        proc = subprocess.run(
            [str(python_bin), "-c", probe, str(repo)],
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=REASYN_IMPORT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return fail(
            task,
            "ReaSyn import",
            f"ReaSyn import probe timed out after {REASYN_IMPORT_TIMEOUT_SECONDS} seconds.",
            str(python_bin),
        )
    except OSError as exc:  # pragma: no cover - platform-specific subprocess errors
        return fail(
            task,
            "ReaSyn import",
            f"Could not start ReaSyn Python: {exc}",
            str(python_bin),
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        return fail(
            task,
            "ReaSyn import",
            "Configured interpreter cannot import the ReaSyn runtime.",
            detail or str(python_bin),
        )
    return ok(
        task,
        "ReaSyn import",
        "Configured interpreter imports ReaSyn chemistry and sampler modules.",
        str(python_bin),
    )


def resolve_reasyn_python(raw: str, repo: Path) -> Path | None:
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.append(repo / ".venv" / "bin")
    for candidate in candidates:
        if candidate.is_dir():
            for name in ("python", "python3"):
                executable = candidate / name
                if executable.exists():
                    return executable
            versioned = sorted(path for path in candidate.glob("python3.*") if path.exists() and not path.is_dir())
            if versioned:
                return versioned[0]
            continue
        if candidate.exists() or raw:
            return candidate
    return None

