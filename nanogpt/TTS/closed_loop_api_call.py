#!/usr/bin/env python3
"""
Smoke test for the closed-loop OpenAI-compatible API endpoint.

Example:
    TTS_LLM_API_KEY=... python3 TTS/closed_loop_api_call.py
"""

from __future__ import annotations

import argparse
import os


DEFAULT_BASE_URL = "https://litellm.yangtzeailab.com/v1"
DEFAULT_MODEL = "Qwen3-VL-235B-A22B-Instruct-FP8"


def parse_args():
    parser = argparse.ArgumentParser(description="Call an OpenAI-compatible closed-loop LLM endpoint.")
    parser.add_argument("--base-url", default=os.environ.get("TTS_LLM_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("TTS_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("TTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--prompt", default="Write a Python function to sort a list")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set TTS_LLM_API_KEY or pass --api-key.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The standalone smoke test requires the `openai` package. "
            "The search runner can still use the endpoint through its httpx fallback."
        ) from exc

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(resp.choices[0].message.content)
    usage = resp.usage
    prompt_tokens = 0 if usage is None else usage.prompt_tokens
    completion_tokens = 0 if usage is None else usage.completion_tokens
    print("tokens:", prompt_tokens, completion_tokens)


if __name__ == "__main__":
    main()

