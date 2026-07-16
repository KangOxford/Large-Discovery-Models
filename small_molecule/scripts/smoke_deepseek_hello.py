"""Smoke-test DeepSeek chat completion with a minimal hello prompt."""

from __future__ import annotations

import argparse
import os
import time

from strbo_v1.llm_advisor.client import OpenAIChatClient
from strbo_v1.llm_advisor.config import LLMClientConfig, load_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--message", default="hello")
    parser.add_argument("--json-mode", action="store_true")
    return parser.parse_args()


def build_client(model: str, timeout: float) -> OpenAIChatClient:
    load_env()
    return OpenAIChatClient(
        LLMClientConfig(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "").rstrip("/"),
            model=model,
        ),
        timeout=timeout,
    )


def main() -> int:
    args = parse_args()
    successes = 0
    for idx in range(1, args.attempts + 1):
        client = build_client(args.model, args.timeout)
        started = time.monotonic()
        try:
            text = client.chat(
                "You are a concise assistant. Reply naturally.",
                args.message,
                json_mode=args.json_mode,
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(f"attempt={idx} status=fail elapsed={elapsed:.2f}s error={type(exc).__name__}: {exc}")
        else:
            elapsed = time.monotonic() - started
            successes += 1
            compact = " ".join(text.split())[:240]
            print(f"attempt={idx} status=ok elapsed={elapsed:.2f}s reply={compact!r}")
        if idx < args.attempts and args.sleep > 0:
            time.sleep(args.sleep)
    print(f"summary attempts={args.attempts} successes={successes} failures={args.attempts - successes}")
    return 0 if successes == args.attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
