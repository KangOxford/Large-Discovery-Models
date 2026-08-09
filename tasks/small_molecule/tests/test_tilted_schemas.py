import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config
from strbo_v1.ldm_tilted_case2.prompts import (
    build_m1_prompt,
    build_m1_analog_seed_prompt,
    summarize_history,
)
from strbo_v1.ldm_tilted_case2.schemas import (
    parse_m1_direct_smiles,
    parse_seed_plan,
)
from strbo_v1.ldm_tilted_case2.sources import _chat_openai_subprocess, call_llm_json
from strbo_v1.llm_advisor.client import MockLLMClient


def test_parse_m1_direct_smiles():
    plan = parse_m1_direct_smiles('```json\n{"direct_smiles":[{"smiles":"CCO","rationale":"x"}]}\n```')
    assert plan.direct_smiles[0].smiles == "CCO"
    assert plan.direct_smiles[0].rationale == "x"


def test_parse_seed_plan():
    plan = parse_seed_plan('{"seeds":[{"smiles":"CCO","budget":80,"intent":"local"}]}')
    assert plan.seeds[0].budget == 80


def test_schema_rejects_candidate_score_name():
    with pytest.raises(ValueError, match="score"):
        parse_seed_plan('{"seeds":[{"smiles":"CCN","budget":1,"objective_score":1}]}')


def test_prompts_forbid_llm_objective_scoring():
    cfg = TiltedLDMCase2Config(method="m1_llm_seed_analog_oversample_sir")
    summary = summarize_history([("CCO", (-1.0, 6.0))], minimize=cfg.minimize)
    prompts = [
        build_m1_prompt(summary, cfg),
        build_m1_analog_seed_prompt(summary, cfg),
    ]
    joined = "\n".join(system + "\n" + user for system, user in prompts).lower()
    assert "objective score" not in joined
    assert "acquisition score" not in joined
    assert "direct_smiles" in joined
    assert "seeds" in joined


def test_prompts_include_smiles_length_limit():
    cfg = TiltedLDMCase2Config(method="m1_llm_seed_analog_oversample_sir", smiles_max_len=42)
    summary = summarize_history([("CCO", (-1.0, 6.0))], minimize=cfg.minimize)
    prompts = [
        build_m1_prompt(summary, cfg),
        build_m1_analog_seed_prompt(summary, cfg),
    ]
    joined = "\n".join(user for _system, user in prompts)
    assert "smiles_max_len=42" in joined


def test_m1_prompt_includes_elites_avoid_list_and_smiles_hygiene():
    cfg = TiltedLDMCase2Config(method="m1_stratified_direct_llm_sir")
    summary = summarize_history(
        [
            ("CCO", (-1.0, 6.0)),
            ("CCN", (-0.5, 7.0)),
            ("CCC", (-1.5, 5.0)),
        ],
        minimize=cfg.minimize,
    )
    _system, user = build_m1_prompt(summary, cfg, sample_count=8, strategy="test strategy")

    assert "mutation, crossover, and scaffold-hop" in user
    assert "avoid_exact_smiles" in user
    assert "top_low_vina" in user
    assert "top_high_activity" in user
    assert "Target context:" in user
    assert "KRAS G12D small-molecule candidates" in user
    assert "switch-II pocket" in user
    assert "Molecule context table:" in user
    assert "qualitative descriptors, not scores" in user
    assert '"history_role": "pareto_front"' in user
    assert '"size_class":' in user
    assert '"ring_pattern":' in user
    assert '"proposal_lesson":' in user
    assert "Avoid salts, dot-disconnected mixtures" in user
    assert "Background:" in user
    assert "Generation principles:" in user
    assert "Generation focus:" in user
    assert "infer useful proposal patterns from the observed history" in user
    assert "Do not name or hard-code any preferred structural class" in user
    assert "seed size is the target molecular size" in user
    assert "simple monotonic size series" in user
    assert "Do not over-restrict valid history-derived organic substituents" in user
    assert "For M1" not in user
    assert "M1" not in user
    assert "Strategy:" not in user
    assert "Cys12" not in user
    assert "warhead" not in user.lower()
    assert "sotorasib" not in user.lower()
    assert "adagrasib" not in user.lower()
    assert "halogenated" not in user
    assert "compact cyclic" not in user
    assert "linear polyamine" not in user
    assert "test strategy" in user


def test_m1_molecule_context_is_qualitative_and_capped():
    cfg = TiltedLDMCase2Config(method="m1_stratified_direct_llm_sir")
    history = [(f"CCCCCCCCN{i % 10}", (-2.0 - i / 10.0, 5.1 + i / 100.0)) for i in range(40)]
    summary = summarize_history(history, minimize=cfg.minimize)
    _system, user = build_m1_prompt(summary, cfg, sample_count=8)

    marker = "Molecule context table:\n"
    start = user.index(marker) + len(marker)
    end = user.index("\nHow to use the molecule context:", start)
    context_table = json.loads(user[start:end])

    assert 1 <= len(context_table) <= 24
    assert all("scores" not in item for item in context_table)
    assert all("mw" not in json.dumps(item).lower() for item in context_table)
    assert all("logp" not in json.dumps(item).lower() for item in context_table)
    assert all("tpsa" not in json.dumps(item).lower() for item in context_table)


def test_history_summary_keeps_pareto_and_failures():
    summary = summarize_history(
        [("CCO", (-1.0, 6.0)), ("BAD", (None, None)), ("CCN", (-0.5, 5.0))],
        minimize=(True, False),
    )
    assert summary["pareto_front"]
    assert summary["failures"][0]["smiles"] == "BAD"


def test_m1_prompt_adds_recent_diversity_alert_without_structural_prior():
    cfg = TiltedLDMCase2Config(method="m1_stratified_direct_llm_sir")
    summary = summarize_history(
        [
            ("CCO", (-1.0, 6.0)),
            ("CCNCCN", (-3.5, 5.65)),
            ("CCNCCNCCCN", (-4.6, 5.67)),
            ("CCNCCNCCCNCCCNCCN", (-5.0, 5.68)),
        ],
        minimize=cfg.minimize,
    )
    _system, user = build_m1_prompt(summary, cfg, sample_count=8, strategy="test")

    assert summary["recent_diversity_alert"]["status"] == "recent_selected_are_too_similar"
    assert "recent_diversity_alert" in user
    assert "avoid simple extensions" in user
    assert "halogenated" not in user
    assert "compact cyclic" not in user
    assert "linear polyamine" not in user


def test_llm_json_call_retries_parse_failure():
    client = MockLLMClient(
        scripted_responses=[
            "not json",
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "x"}]}),
        ]
    )
    result = call_llm_json(client, "s", "u", parse_m1_direct_smiles, max_retries=1)
    assert result.parsed.direct_smiles[0].smiles == "CCO"
    assert len(result.attempts) == 2


def test_llm_json_call_waits_between_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        "strbo_v1.ldm_tilted_case2.sources.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    client = MockLLMClient(
        scripted_responses=[
            "not json",
            json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "x"}]}),
        ]
    )

    result = call_llm_json(
        client,
        "s",
        "u",
        parse_m1_direct_smiles,
        max_retries=1,
        retry_wait_seconds=7.5,
    )

    assert result.parsed.direct_smiles[0].smiles == "CCO"
    assert sleeps == [7.5]


def test_llm_json_call_records_raw_text():
    text = json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "x"}]})
    client = MockLLMClient(scripted_responses=[text])
    result = call_llm_json(client, "s", "u", parse_m1_direct_smiles, max_retries=0)
    assert result.raw_text == text
    assert result.attempts[0]["raw_text"] == text
    assert result.attempts[0]["system_prompt"] == "s"
    assert result.attempts[0]["user_prompt"] == "u"
    assert result.attempts[0]["raw_output"] == text
    assert result.attempts[0]["parsed_json"] == {
        "direct_smiles": [{"smiles": "CCO", "rationale": "x"}]
    }


def test_openai_subprocess_uses_old_style_request_timeout(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["timeout"] = kwargs["timeout"]
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout='{"direct_smiles":[]}', stderr="")

    monkeypatch.setattr("strbo_v1.ldm_tilted_case2.sources.subprocess.run", fake_run)

    llm = SimpleNamespace(
        config=SimpleNamespace(api_key="key", base_url="https://example.test/v1", model="model"),
        temperature=0.2,
        max_tokens=None,
    )
    text = _chat_openai_subprocess(llm, "system", "user", 300.0)

    assert text == '{"direct_smiles":[]}'
    assert captured["timeout"] == 305.0
    assert captured["env"]["LDM_SUBPROCESS_LLM_TIMEOUT"] == "300.0"
    assert captured["env"]["LDM_SUBPROCESS_LLM_REQUEST_TIMEOUT"] == "300.0"


def test_llm_json_call_retries_llm_exception():
    text = json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "x"}]})

    class FlakyLLM:
        model_name = "flaky"

        def __init__(self):
            self.calls = 0

        def chat(self, system, user, *, json_mode=True):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("slow backend")
            return text

    client = FlakyLLM()
    result = call_llm_json(client, "s", "u", parse_m1_direct_smiles, max_retries=1)
    assert result.parsed.direct_smiles[0].smiles == "CCO"
    assert result.attempts[0]["error"] == "TimeoutError: slow backend"
    assert client.calls == 2


@pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM hard timeout is POSIX-only")
def test_llm_json_call_hard_times_out_blocking_chat():
    class BlockingLLM:
        model_name = "blocking"
        timeout = 0.1

        def chat(self, system, user, *, json_mode=True):
            time.sleep(2.0)
            return json.dumps({"direct_smiles": [{"smiles": "CCO", "rationale": "late"}]})

    started = time.monotonic()
    with pytest.raises(ValueError, match="hard timeout"):
        call_llm_json(BlockingLLM(), "s", "u", parse_m1_direct_smiles, max_retries=0)
    assert time.monotonic() - started < 1.0
