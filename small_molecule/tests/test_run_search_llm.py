"""End-to-end CLI tests for the ``bo-*-ldm`` methods in run_search.py.

These tests invoke :func:`run_search.main` with a monkey-patched
LLM client (``MockLLMClient`` injected in place of
``OpenAIChatClient``) so no real API call is made.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import pytest

import run_search
from strbo_v1.llm_advisor import client as llm_client_module
from strbo_v1.llm_advisor.blocks import NoopBlock, ReviewBOBlock
from strbo_v1.llm_advisor.client import MockLLMClient, _serialize_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dynamic_mock() -> MockLLMClient:
    """MockLLMClient that returns sensible blocks for both phases
    (matches the test helper in test_bayesian_ldm_search.py)."""
    from strbo_v1.llm_advisor.blocks import NoopBlock, ReviewBOBlock
    from strbo_v1.llm_advisor.client import _serialize_blocks

    class _Dyn(MockLLMClient):
        def chat(self, system, user, *, json_mode=True):
            self.call_log.append({"system": system[:30]})
            if "PHASE B" in system:
                decisions = {}
                for line in user.splitlines():
                    s = line.strip()
                    if s.startswith("- ") and "  mu=" in s:
                        smi = s[2:].split("  mu=", 1)[0].strip()
                        if smi:
                            decisions[smi] = "ok"
                return _serialize_blocks(
                    [ReviewBOBlock(rationale="ok", decisions=decisions)]
                )
            return _serialize_blocks([NoopBlock(rationale="ok")])

    return _Dyn()


@pytest.fixture
def patch_openai_with_mock(monkeypatch):
    """Replace OpenAIChatClient with MockLLMClient in run_search's namespace."""
    def _factory(cfg, **_kwargs):
        return _dynamic_mock()
    # Patch the symbol in the llm_advisor.client module so the local
    # import in run_search._build_llm_advisor picks it up.
    monkeypatch.setattr(llm_client_module, "OpenAIChatClient", _factory)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_search_bo_tanimoto_ldm_writes_json(
    tmp_path: Path, patch_openai_with_mock, monkeypatch,
) -> None:
    """End-to-end: ``--method bo-tanimoto-ldm`` writes a valid JSON."""
    # Avoid loading .env from the real repo root; point to a fake one.
    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "LLM_API_KEY=fake-key\nLLM_BASE_URL=https://fake.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake.example/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    out_dir = tmp_path / "out"
    rc = run_search.main([
        "--method", "bo-tanimoto-ldm",
        "--seed", "0",
        "--seed-smiles", "CCO,CCN,CCC,CCCC",
        "--num-evaluations", "4",
        "--batch-size", "1",
        "--init-size", "2",
        "--acquisition", "ei",
        "--objective", "mock",
        "--gp-device", "cpu",
        "--output", str(out_dir),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("bo-tanimoto-ldm_seed=*.json"))
    assert len(out_files) == 1
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert payload["config"]["method"] == "bo-tanimoto-ldm"
    assert payload["config"]["n_objectives"] == 1
    assert payload["config"]["llm"]["model"] == "DeepSeek-V4-Flash"
    assert len(payload["history"]) >= 2        # at least init phase
    # The trajectory is embedded under "llm_trajectory".
    assert "llm_trajectory" in payload
    assert payload["llm_trajectory"]["status"] in ("completed", "fatal_error")
    assert len(payload["llm_trajectory"]["rounds"]) >= 1


def test_run_search_bo_strkernel_ldm_writes_json(
    tmp_path: Path, patch_openai_with_mock, monkeypatch,
) -> None:
    """End-to-end: ``--method bo-strkernel-ldm`` writes a valid JSON."""
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake.example/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    out_dir = tmp_path / "out"
    rc = run_search.main([
        "--method", "bo-strkernel-ldm",
        "--seed", "0",
        "--seed-smiles", "CCO,CCN,CCC,CCCC",
        "--num-evaluations", "4",
        "--batch-size", "1",
        "--init-size", "2",
        "--acquisition", "ei",
        "--objective", "mock",
        "--gp-device", "cpu",
        "--output", str(out_dir),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("bo-strkernel-ldm_seed=*.json"))
    assert len(out_files) == 1
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert payload["config"]["method"] == "bo-strkernel-ldm"
    # GP impl is the strkernel one (from the method suffix).
    assert payload["config"]["gp"]["impl"] == "smiles-strkernel"


def test_run_search_ldm_unknown_method_rejected(
    tmp_path: Path, monkeypatch,
) -> None:
    """``--method bo-foo-ldm`` is not in VALID_METHODS and gets rejected."""
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    with pytest.raises(SystemExit):
        run_search.main([
            "--method", "bo-foo-ldm",
            "--seed", "0",
            "--seed-smiles", "CCO",
            "--num-evaluations", "2",
            "--output", str(tmp_path),
            "--log-level", "WARNING",
        ])


def test_run_search_bo_tanimoto_unchanged(
    tmp_path: Path, monkeypatch,
) -> None:
    """Sanity check: the non-LDM ``bo-tanimoto`` method still works
    and does NOT embed an ``llm_trajectory`` key."""
    out_dir = tmp_path / "out"
    rc = run_search.main([
        "--method", "bo-tanimoto",
        "--seed", "0",
        "--seed-smiles", "CCO,CCN,CCC,CCCC",
        "--num-evaluations", "4",
        "--batch-size", "1",
        "--init-size", "2",
        "--acquisition", "ei",
        "--objective", "mock",
        "--gp-device", "cpu",
        "--output", str(out_dir),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("bo-tanimoto_seed=*.json"))
    assert len(out_files) == 1
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert payload["config"]["method"] == "bo-tanimoto"
    assert "llm_trajectory" not in payload
    assert "llm" not in payload["config"]


# ---------------------------------------------------------------------------
# LDM system-prompt supplement (--ldm-sys-prompt)
# ---------------------------------------------------------------------------


def test_run_search_ldm_sys_prompt_default_is_empty(
    tmp_path: Path, patch_openai_with_mock, monkeypatch,
) -> None:
    """With no ``--ldm-sys-prompt`` flag, the embedded trajectory's
    config echo carries ``llm.ldm_sys_prompt == ""`` (the default)."""
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake.example/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    out_dir = tmp_path / "out"
    rc = run_search.main([
        "--method", "bo-tanimoto-ldm",
        "--seed", "0",
        "--seed-smiles", "CCO,CCN,CCC,CCCC",
        "--num-evaluations", "4",
        "--batch-size", "1",
        "--init-size", "2",
        "--acquisition", "ei",
        "--objective", "mock",
        "--gp-device", "cpu",
        "--output", str(out_dir),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("bo-tanimoto-ldm_seed=*.json"))
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert payload["config"]["llm"]["ldm_sys_prompt"] == ""


def test_run_search_ldm_sys_prompt_inline_text_reaches_llm(
    tmp_path: Path, monkeypatch,
) -> None:
    """``--ldm-sys-prompt "inline text"`` is threaded through to the
    LLM's system prompt for every stage (inline mode)."""
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake.example/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    captured: list = []

    def _factory(cfg, **_kwargs):
        class _Dyn(MockLLMClient):
            def chat(self, system, user, *, json_mode=True):
                captured.append(system)
                if "PHASE B" in system or "STAGE B" in system or "review_bo" in system:
                    decisions = {}
                    for line in user.splitlines():
                        s = line.strip()
                        if s.startswith("- ") and "  mu=" in s:
                            smi = s[2:].split("  mu=", 1)[0].strip()
                            if smi:
                                decisions[smi] = "ok"
                    return _serialize_blocks(
                        [ReviewBOBlock(rationale="ok", decisions=decisions)]
                    )
                return _serialize_blocks([NoopBlock(rationale="ok")])

        return _Dyn()

    monkeypatch.setattr(llm_client_module, "OpenAIChatClient", _factory)

    out_dir = tmp_path / "out"
    inline = "Use analog heavily. Pool is the BO acquisition space."
    rc = run_search.main([
        "--method", "bo-tanimoto-ldm",
        "--seed", "0",
        "--seed-smiles", "CCO,CCN,CCC,CCCC",
        "--num-evaluations", "4",
        "--batch-size", "1",
        "--init-size", "2",
        "--acquisition", "ei",
        "--objective", "mock",
        "--gp-device", "cpu",
        "--ldm-sys-prompt", inline,
        "--output", str(out_dir),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("bo-tanimoto-ldm_seed=*.json"))
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert payload["config"]["llm"]["ldm_sys_prompt"] == inline
    assert len(captured) >= 1
    for s in captured:
        assert "## EXTERNAL GUIDANCE" in s
        assert inline in s


def test_run_search_ldm_sys_prompt_reads_file(
    tmp_path: Path, monkeypatch,
) -> None:
    """``--ldm-sys-prompt <path>`` reads the file's contents when the
    path is an existing file (file mode)."""
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake.example/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    captured: list = []

    def _factory(cfg, **_kwargs):
        class _Dyn(MockLLMClient):
            def chat(self, system, user, *, json_mode=True):
                captured.append(system)
                if "PHASE B" in system or "STAGE B" in system or "review_bo" in system:
                    decisions = {}
                    for line in user.splitlines():
                        s = line.strip()
                        if s.startswith("- ") and "  mu=" in s:
                            smi = s[2:].split("  mu=", 1)[0].strip()
                            if smi:
                                decisions[smi] = "ok"
                    return _serialize_blocks(
                        [ReviewBOBlock(rationale="ok", decisions=decisions)]
                    )
                return _serialize_blocks([NoopBlock(rationale="ok")])

        return _Dyn()

    monkeypatch.setattr(llm_client_module, "OpenAIChatClient", _factory)

    prompt_file = tmp_path / "guidance.txt"
    file_text = (
        "Line 1 from file.\n"
        "Line 2 from file.\n"
        "Line 3 from file."
    )
    prompt_file.write_text(file_text, encoding="utf-8")

    out_dir = tmp_path / "out"
    rc = run_search.main([
        "--method", "bo-tanimoto-ldm",
        "--seed", "0",
        "--seed-smiles", "CCO,CCN,CCC,CCCC",
        "--num-evaluations", "4",
        "--batch-size", "1",
        "--init-size", "2",
        "--acquisition", "ei",
        "--objective", "mock",
        "--gp-device", "cpu",
        "--ldm-sys-prompt", str(prompt_file),
        "--output", str(out_dir),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("bo-tanimoto-ldm_seed=*.json"))
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    # The config echo carries the file's contents (not the path).
    assert payload["config"]["llm"]["ldm_sys_prompt"] == file_text
    # And the LLM received it in its system prompt.
    assert len(captured) >= 1
    for s in captured:
        assert "## EXTERNAL GUIDANCE" in s
        assert "Line 1 from file." in s
        assert "Line 3 from file." in s


def test_run_search_ldm_sys_prompt_missing_file_falls_back_to_inline(
    tmp_path: Path, patch_openai_with_mock, monkeypatch,
) -> None:
    """When ``--ldm-sys-prompt <path>`` points to a non-existent file,
    the literal string is used as inline text (not an error)."""
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake.example/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    out_dir = tmp_path / "out"
    bogus = str(tmp_path / "does-not-exist.txt")
    rc = run_search.main([
        "--method", "bo-tanimoto-ldm",
        "--seed", "0",
        "--seed-smiles", "CCO,CCN,CCC,CCCC",
        "--num-evaluations", "4",
        "--batch-size", "1",
        "--init-size", "2",
        "--acquisition", "ei",
        "--objective", "mock",
        "--gp-device", "cpu",
        "--ldm-sys-prompt", bogus,
        "--output", str(out_dir),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    out_files = list(out_dir.glob("bo-tanimoto-ldm_seed=*.json"))
    payload = json.loads(out_files[0].read_text(encoding="utf-8"))
    # Falls back to the literal path string.
    assert payload["config"]["llm"]["ldm_sys_prompt"] == bogus


def test_run_search_config_from_dict_parses_ldm_sys_prompt() -> None:
    """``config_from_dict`` accepts the ``ldm-sys-prompt`` key (CLI
    long-form) and the equivalent ``ldm_sys_prompt`` underscore form."""
    from run_search import config_from_dict

    ns = config_from_dict({
        "method": "bo-tanimoto-ldm",
        "seed": 0,
        "seed-smiles": "CCO,CCN",
        "num-evaluations": 4,
        "batch-size": 1,
        "init-size": 2,
        "objective": "mock",
        "gp-device": "cpu",
        "ldm-sys-prompt": "my prompt text",
    })
    assert ns.ldm_sys_prompt == "my prompt text"
    assert ns.method == "bo-tanimoto-ldm"

    ns2 = config_from_dict({
        "method": "bo-tanimoto-ldm",
        "seed": 0,
        "seed-smiles": "CCO,CCN",
        "num-evaluations": 4,
        "batch-size": 1,
        "init-size": 2,
        "objective": "mock",
        "gp-device": "cpu",
        "ldm_sys_prompt": "underscore form",
    })
    assert ns2.ldm_sys_prompt == "underscore form"


def test_run_search_config_from_dict_default_ldm_sys_prompt_empty() -> None:
    """When ``ldm-sys-prompt`` is omitted, ``args.ldm_sys_prompt``
    defaults to ""."""
    from run_search import config_from_dict

    ns = config_from_dict({
        "method": "bo-tanimoto-ldm",
        "seed": 0,
        "seed-smiles": "CCO,CCN",
        "num-evaluations": 4,
        "batch-size": 1,
        "init-size": 2,
        "objective": "mock",
        "gp-device": "cpu",
    })
    assert ns.ldm_sys_prompt == ""


def test_run_search_resolve_ldm_sys_prompt_helper() -> None:
    """Unit-test ``_resolve_ldm_sys_prompt`` directly."""
    from run_search import _resolve_ldm_sys_prompt

    # Empty → empty.
    assert _resolve_ldm_sys_prompt("") == ""

    # Inline text (no file) → returned verbatim.
    assert _resolve_ldm_sys_prompt("hello world") == "hello world"

    # Existing file → file contents.
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8",
    ) as f:
        f.write("file contents here")
        path = f.name
    try:
        assert _resolve_ldm_sys_prompt(path) == "file contents here"
    finally:
        os.unlink(path)

    # Non-existent path → returned verbatim (fallback).
    assert (
        _resolve_ldm_sys_prompt("/non/existent/path/foo.txt")
        == "/non/existent/path/foo.txt"
    )
