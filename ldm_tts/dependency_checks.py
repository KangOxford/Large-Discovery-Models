"""Lightweight dependency preflight checks for LDM-TTS tasks."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ldm_tts.task_registry import (
    REPOSITORY_RELATIVE_PREFIXES,
    TaskRegistrationError,
    get_task_definition,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NANOGPT_CACHE_DIR = Path(
    os.environ.get("AUTORESEARCH_CACHE_DIR", "~/.cache/autoresearch")
).expanduser().resolve()
NANOGPT_DATA_DIR = NANOGPT_CACHE_DIR / "data"
NANOGPT_TOKENIZER_DIR = NANOGPT_CACHE_DIR / "tokenizer"
REASYN_ENTRYPOINT_REL = Path("reasyn") / "sampler" / "parallel.py"
DEFAULT_REASYN_MODEL_PATHS = (
    "data/trained_model/nv-reasyn-ar-166m-v2.ckpt",
    "data/trained_model/nv-reasyn-eb-174m-v2.ckpt",
)
REASYN_IMPORT_TIMEOUT_SECONDS = 30


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


def check_nanogpt(
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
    *,
    mode: str = "",
    include_optional: bool = True,
) -> list[DependencyCheck]:
    task = "nanogpt"
    real = mode == "real" or arg_value(args, "generator") not in {"mock", "operation_mock"}
    checks: list[DependencyCheck] = []

    checks.extend(check_llm_settings(
        task,
        args,
        env,
        url_arg="llm-url",
        model_arg="llm-model-name",
        api_arg="api-key",
        url_env=("TTS_LLM_URL",),
        model_env=("TTS_LLM_MODEL",),
        api_env=("TTS_LLM_API_KEY", "OPENAI_API_KEY"),
        required=real,
    ))

    for label, key in (("train file", "train-file"), ("operation schema", "operation-schema")):
        path = resolve_task_path(arg_value(args, key), cwd)
        if path is None:
            checks.append(fail(task, label, f"Missing --{key}."))
        elif path.exists():
            checks.append(ok(task, label, f"{label} exists.", str(path)))
        else:
            checks.append(fail(task, label, f"{label} does not exist.", str(path)))

    if real and not (bool_arg(args, "skip-eval") and not include_optional):
        checks.extend(check_nanogpt_data())
        checks.append(check_cuda_visibility(task, "CUDA", requested_device="cuda", env=env))
    elif real:
        checks.extend([
            skip(
                task,
                "prepare.py data",
                "Data check omitted for a --skip-eval run because --no-optional was requested.",
            ),
            skip(
                task,
                "prepare.py tokenizer",
                "Tokenizer check omitted for a --skip-eval run because --no-optional was requested.",
            ),
        ])
        checks.append(skip(
            task,
            "CUDA",
            "CUDA check omitted because the resolved plan uses --skip-eval.",
        ))
    else:
        checks.append(skip(task, "nanoGPT data", "Mock nanoGPT runs do not need prepare.py data."))
    return checks


def check_nanogpt_data() -> list[DependencyCheck]:
    task = "nanogpt"
    checks: list[DependencyCheck] = []
    parquet_files = sorted(NANOGPT_DATA_DIR.glob("*.parquet")) if NANOGPT_DATA_DIR.exists() else []
    if parquet_files:
        checks.append(ok(task, "prepare.py data", f"Found {len(parquet_files)} parquet shard(s).", str(NANOGPT_DATA_DIR)))
    else:
        checks.append(fail(
            task,
            "prepare.py data",
            "No parquet shards found. Run `uv run --locked --group train --project tasks/nanogpt "
            "python tasks/nanogpt/scripts/prepare.py` first.",
            str(NANOGPT_DATA_DIR),
        ))
    tokenizer = NANOGPT_TOKENIZER_DIR / "tokenizer.pkl"
    token_bytes = NANOGPT_TOKENIZER_DIR / "token_bytes.pt"
    missing = [str(path) for path in (tokenizer, token_bytes) if not path.exists()]
    if not missing:
        checks.append(ok(task, "prepare.py tokenizer", "Tokenizer artifacts exist.", str(NANOGPT_TOKENIZER_DIR)))
    else:
        checks.append(fail(
            task,
            "prepare.py tokenizer",
            "Tokenizer artifacts are missing. Run `uv run --locked --group train --project tasks/nanogpt "
            "python tasks/nanogpt/scripts/prepare.py` first.",
            ", ".join(missing),
        ))
    return checks


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
