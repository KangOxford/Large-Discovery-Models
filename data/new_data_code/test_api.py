#!/usr/bin/env python3
"""Deprecated connectivity probe using the repository's LLM environment."""

from __future__ import annotations

import os
import sys


def main() -> int:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL_NAME", "DeepSeek-V4-Flash")
    base_url = os.environ.get("LLM_BASE_URL")
    if not api_key:
        print("error: LLM_API_KEY must be set", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI
    except ImportError:
        print("error: install openai>=1.0 in the active environment", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - command reports endpoint failures
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print((response.choices[0].message.content or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
