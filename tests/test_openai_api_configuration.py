from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ldm_tts.runner import build_plan, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _set_generic_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TTS_LLM_URL",
        "TTS_LLM_API_KEY",
        "TTS_LLM_MODEL",
        "LLM_MODEL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL_NAME", "served-model")


def test_nanogpt_accepts_generic_openai_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.nanogpt.ldm_task.procedure import parse_args

    _set_generic_llm_environment(monkeypatch)

    args = parse_args([])

    assert args.llm_url == "https://llm.example.test/v1"
    assert args.api_key == "test-secret"
    assert args.llm_model_name == "served-model"


def test_antibody_accepts_generic_openai_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.antibody.ldm_task.procedure import configure_llm_environment, parse_args

    _set_generic_llm_environment(monkeypatch)
    previous_backend_model = os.environ.get("LLM_MODEL")

    args = parse_args(["--antigen", "SMOKE"])
    try:
        configure_llm_environment(args)

        assert args.llm_url == "https://llm.example.test/v1"
        assert args.api_key == "test-secret"
        assert args.llm_model_name == "served-model"
        assert os.environ["LLM_MODEL"] == "served-model"
    finally:
        if previous_backend_model is None:
            os.environ.pop("LLM_MODEL", None)
        else:
            os.environ["LLM_MODEL"] = previous_backend_model


def test_small_molecule_accepts_generic_openai_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.small_molecule.ldm_task.procedure import parse_args

    _set_generic_llm_environment(monkeypatch)

    args = parse_args(["--mock"])

    assert args.llm_url == "https://llm.example.test/v1"
    assert args.api_key == "test-secret"
    assert args.llm_model_name == "served-model"


def test_nanogpt_forwards_openai_url_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.nanogpt.ldm_task import api_generate

    captured: dict[str, object] = {}

    async def fake_create_chat_completion(*, llm_url, api_key, completion_params):
        captured.update(
            llm_url=llm_url,
            api_key=api_key,
            completion_params=completion_params,
        )
        return "OK", {"total_tokens": 1}

    monkeypatch.setattr(api_generate, "_create_chat_completion", fake_create_chat_completion)

    content, usage = asyncio.run(
        api_generate.openai_compatible_generate(
            [{"role": "user", "content": "test"}],
            llm_url="https://llm.example.test/v1",
            api_key="test-secret",
            llm_model_name="served-model",
        )
    )

    assert content == "OK"
    assert usage == {"total_tokens": 1}
    assert captured["llm_url"] == "https://llm.example.test/v1"
    assert captured["api_key"] == "test-secret"
    assert captured["completion_params"]["model"] == "served-model"  # type: ignore[index]


@pytest.mark.parametrize(
    "config_path",
    (
        "config/nanogpt/real_operation_tool_best_of_n.yaml",
        "config/nanogpt/real_operation_tool_fixed_best_of_n.yaml",
    ),
)
def test_nanogpt_real_configs_keep_provider_settings_environment_only(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
) -> None:
    from tasks.nanogpt.ldm_task import procedure

    _set_generic_llm_environment(monkeypatch)
    path = REPO_ROOT / config_path
    plan = build_plan(load_config(path), path)
    args = procedure.parse_args([])

    assert "--llm-url" not in plan["argv"]
    assert "--api-key" not in plan["argv"]
    assert "--llm-model-name" not in plan["argv"]
    assert "test-secret" not in plan["command_display"]
    assert "--group train" in plan["command_display"]
    assert args.llm_url == "https://llm.example.test/v1"
    assert args.llm_model_name == "served-model"
    assert args.api_key == "test-secret"


def test_antibody_real_config_keeps_provider_settings_environment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.antibody.ldm_task import procedure

    _set_generic_llm_environment(monkeypatch)
    monkeypatch.setenv("ABSOLUT_PATH", "/opt/Absolut")
    path = REPO_ROOT / "config/antibody/real_lcb.yaml"

    plan = build_plan(load_config(path), path)
    args = procedure.parse_args(["--antigen", "1ADQ_A"])

    assert "--llm-url" not in plan["argv"]
    assert "--llm-model-name" not in plan["argv"]
    assert "--api-key" not in plan["argv"]
    assert "--absolut-path" not in plan["argv"]
    assert "test-secret" not in plan["command_display"]
    assert args.llm_url == "https://llm.example.test/v1"
    assert args.llm_model_name == "served-model"
    assert args.api_key == "test-secret"
    assert args.absolut_path == "/opt/Absolut"


def test_small_molecule_real_config_keeps_api_key_environment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.small_molecule.ldm_task import procedure

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "served-model")
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    path = REPO_ROOT / "config/small_molecule/real_m1_seed_analog.yaml"

    plan = build_plan(load_config(path), path)
    args = procedure.parse_args([])

    assert "--llm-url" not in plan["argv"]
    assert "--llm-model-name" not in plan["argv"]
    assert "--api-key" not in plan["argv"]
    assert "test-secret" not in plan["command_display"]
    assert args.llm_url == "https://llm.example.test/v1"
    assert args.llm_model_name == "served-model"
    assert args.api_key == "test-secret"
