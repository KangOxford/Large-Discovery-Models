"""nanoGPT dependency checks."""

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
    fail,
    ok,
    resolve_task_path,
    skip,
)


NANOGPT_CACHE_DIR = Path(
    os.environ.get("AUTORESEARCH_CACHE_DIR", "~/.cache/autoresearch")
).expanduser().resolve()
NANOGPT_DATA_DIR = NANOGPT_CACHE_DIR / "data"
NANOGPT_TOKENIZER_DIR = NANOGPT_CACHE_DIR / "tokenizer"


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

