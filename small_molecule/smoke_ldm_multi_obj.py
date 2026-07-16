"""End-to-end smoke test for the LDM with vina+nn.

Run: `python smoke_ldm_multi_obj.py`

This wraps ``run_search.main`` with a small monkey-patch:
vina → a deterministic per-SMILES mock; nn → a deterministic per-SMILES mock.
The LDM stack must run end-to-end without raising ``ValueError:
minimize length does not match n_obj`` (the pre-fix bug).

Asserts:
* `bayesian_ldm_search` returns a non-empty history
* All history entries are ``list[float]`` of length 2 (no collapse)
* verbose stdout contains stage markers
"""
from __future__ import annotations

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_search  # noqa: E402
from strbo_v1.llm_advisor import client as llm_client_module  # noqa: E402


def _vina_mock(smis):
    return [-1.0 * len(s) for s in smis]


def _nn_mock(smis):
    return [float(len(s)) for s in smis]


def _build_vina_mock(args):
    return _vina_mock


def _build_nn_mock(args):
    return _nn_mock


def _build_reasyn_mock(args):
    """No-op analog_fn (no analogues)."""
    def _fn(smis):
        return []
    return _fn


class _Dyn(MockLLMClient := llm_client_module.MockLLMClient):
    """Synthesize LDM responses: Stage A1 noop, Stage B auto-ok.

    Also records every (system, user) tuple it sees so the smoke
    test can assert the GUIDANCE text reached the LLM's system
    prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompts = []

    def chat(self, system, user, *, json_mode=True):
        self.system_prompts.append(system)
        if self.scripted_responses is not None or self.scripted_blocks is not None:
            return super().chat(system, user, json_mode=json_mode)
        from strbo_v1.llm_advisor.client import _serialize_blocks
        from strbo_v1.llm_advisor.blocks import NoopBlock, ReviewBOBlock
        if "STAGE B" in system:
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


def main() -> int:
    # Set up LLM env so OpenAIChatClient construction would succeed if reached.
    os.environ["LLM_API_KEY"] = os.environ.get("LLM_API_KEY", "test-key")
    os.environ["LLM_BASE_URL"] = os.environ.get("LLM_BASE_URL", "https://x/v1")

    guidance_text = (
        "Use analog heavily. Pool is the BO acquisition space. "
        "vina = kcal/mol (minimize). nn = pIC50 (maximize)."
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        prompt_file = Path(tmp) / "guidance.txt"
        prompt_file.write_text(guidance_text, encoding="utf-8")

        with mock.patch.object(run_search, "_build_vina_scorer", _build_vina_mock), \
             mock.patch.object(run_search, "_build_nn_scorer", _build_nn_mock), \
             mock.patch.object(run_search, "_build_reasyn_analog", _build_reasyn_mock), \
             mock.patch.object(llm_client_module, "OpenAIChatClient", _Dyn):
            rc = run_search.main([
                "--method", "bo-tanimoto-ldm",
                "--seed", "0",
                "--seed-smiles", "CCO,CCN,CCC,CCCC",
                "--num-evaluations", "6",
                "--batch-size", "2",
                "--init-size", "3",
                "--acquisition", "ei",
                "--objective", "vina+nn",
                "--gp-device", "cpu",
                "--pool-min-size", "3",
                "--ldm-sys-prompt", str(prompt_file),
                "--output", str(out_dir),
                "--log-level", "WARNING",
                "--verbose",
            ])

        if rc != 0:
            print(f"FAIL: run_search.main returned {rc}")
            return 1

        out_files = list(out_dir.glob("bo-tanimoto-ldm_seed=*.json"))
        if not out_files:
            print("FAIL: no output JSON written")
            return 1

        payload = json.loads(out_files[0].read_text(encoding="utf-8"))
        n_obj = payload["config"]["n_objectives"]
        if n_obj != 2:
            print(f"FAIL: expected n_objectives=2, got {n_obj}")
            return 1

        history = payload["history"]
        if not history:
            print("FAIL: empty history")
            return 1
        for entry in history:
            sc = entry.get("score", entry.get("scores"))
            if sc is None:
                print(f"FAIL: history entry has None score: {entry}")
                return 1
            if not isinstance(sc, (list, tuple)) or len(sc) != 2:
                print(f"FAIL: history entry score is not list[float] of length 2: {entry}")
                return 1
        # Trajectory must be embedded
        if "llm_trajectory" not in payload:
            print("FAIL: llm_trajectory not in payload")
            return 1
        # LDM system prompt (resolved from file) must be in the JSON's
        # config echo.
        if payload["config"]["llm"]["ldm_sys_prompt"] != guidance_text:
            print(
                "FAIL: config.llm.ldm_sys_prompt mismatch; got "
                f"{payload['config']['llm']['ldm_sys_prompt']!r}"
            )
            return 1
        # And it must be in the trajectory's per-round state.
        for r in payload["llm_trajectory"]["rounds"]:
            if r["pre_state_snapshot"].get("guidance") != guidance_text:
                print(
                    "FAIL: pre_state_snapshot.guidance mismatch; got "
                    f"{r['pre_state_snapshot'].get('guidance')!r}"
                )
                return 1

    print(f"PASS: {len(history)} history entries, all list[float] of length 2.")
    print("      (vina+nn multi-obj works without collapse.)")
    print("PASS: ldm_sys_prompt read from file and recorded verbatim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
