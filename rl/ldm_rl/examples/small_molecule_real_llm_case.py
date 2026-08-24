"""Drive a small-molecule RL episode with a real Qwen3.5 model as the policy.

Unlike ``small_molecule_rl_case.py`` (deterministic mock proposer), this script
calls an OpenAI-compatible chat server (``scripts/llm_server.py`` serving a
Qwen3.5 checkpoint) as the policy. The environment still uses the task's mock
scorers, so no vina/torch-GP is required; only the model side is real.

The transcript follows ``bridge.generate``'s convention: the rendered
``env.reset()`` prompt starts the conversation, each policy turn is appended as
an assistant message, and each ``step.observation`` (feedback) is appended as a
user message.

Usage (on the GPU node, after starting llm_server.py):

    export LLM_URL=http://127.0.0.1:8020/v1/chat/completions
    python rl/ldm_rl/examples/small_molecule_real_llm_case.py 6 3
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make ``ldm_rl`` (rl/) and the repo root (ldm_tts, tasks) importable when run
# as a plain script from the checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RL_ROOT = Path(__file__).resolve().parents[2]
for _path in (_RL_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import requests  # noqa: E402

from ldm_rl import EnvConfig  # noqa: E402
from ldm_rl.factories import build_env  # noqa: E402

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8020/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-2B")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))


def call_llm(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "enable_thinking": False,
    }
    response = requests.post(LLM_URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"llm_server returned no choices: {data}")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def main() -> int:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    reservoir_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    env = build_env(
        "small_molecule",
        mode="mock",
        config=EnvConfig(
            iterations=iterations,
            reservoir_size=reservoir_size,
            evaluations_per_round=1,
            reward="improvement",
        ),
    )

    messages: list[dict[str, str]] = [{"role": "user", "content": env.reset()}]
    trajectory: list[dict] = []
    for round_idx in range(iterations):
        raw = call_llm(messages)
        messages.append({"role": "assistant", "content": raw})
        step = env.step(raw)
        row: dict = {
            "round": round_idx,
            "reward": round(step.reward, 6),
            "terminated": step.terminated,
            "truncated": step.truncated,
        }
        evaluated = step.info["evaluated"]
        if evaluated:
            metrics = evaluated[0]["evaluation"]["metrics"]
            row.update(
                {
                    "smiles": evaluated[0]["candidate"]["payload"]["smiles"],
                    "vina": metrics["vina"],
                    "activity": metrics["activity"],
                }
            )
        else:
            row["parse_error"] = step.info["parse_error"]
            row["rejections"] = [item["reason"] for item in step.info["rejections"]]
        row["raw_response"] = raw[:200]
        trajectory.append(row)
        printable = {k: v for k, v in row.items() if k != "raw_response"}
        print(f"round={round_idx} " + json.dumps(printable, sort_keys=True), flush=True)
        messages.append({"role": "user", "content": step.observation})
        if step.done:
            break

    scored = [row for row in trajectory if "vina" in row]
    summary = {
        "status": "ok",
        "task": "small_molecule",
        "mode": "mock",
        "policy": "real_llm",
        "llm_model": LLM_MODEL,
        "rounds": len(trajectory),
        "successful_rounds": len(scored),
        "total_reward": round(sum(row["reward"] for row in trajectory), 6),
        "best_vina": min((row["vina"] for row in scored), default=None),
        "best_activity": max((row["activity"] for row in scored), default=None),
        "final_smiles": scored[-1]["smiles"] if scored else None,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
