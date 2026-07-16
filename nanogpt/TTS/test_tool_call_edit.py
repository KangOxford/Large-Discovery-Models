#!/usr/bin/env python3
"""
Smoke test the SGLang/Qwen3-Coder tool-call edit path against TTS/real_train.py.

By default this writes an edited copy and patch under TTS/tool_call_tests/ without
modifying TTS/real_train.py. Pass --apply to overwrite the source file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TTS.api_generate import tool_call_generate
from TTS.search_core import (
    apply_search_replace_blocks,
    extract_search_replace_blocks,
    extract_tool_call_edit_blocks,
    extract_unified_diff,
    make_unified_diff,
    apply_unified_diff_to_text,
)


DEFAULT_URL = "http://127.0.0.1:52307/v1"
DEFAULT_MODEL = "Qwen3-Coder-30B-A3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Test tool-call editing on TTS/real_train.py.")
    parser.add_argument("--train-file", type=Path, default=Path("TTS/real_train.py"))
    parser.add_argument("--out-dir", type=Path, default=Path("TTS/tool_call_tests"))
    parser.add_argument("--base-url", default=os.environ.get("TTS_LLM_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.environ.get("TTS_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("TTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--request",
        default=(
            "Make one tiny safe edit to TTS/real_train.py: add a short comment next to TIME_BUDGET "
            "explaining that it is shortened for smoke tests. Do not change behavior."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Overwrite --train-file with the edited result.")
    return parser.parse_args()


def build_prompt(train_path: Path, train_text: str, request: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a code editing agent. Use the edit_train_py tool exactly once. "
                "Choose a small exact search/replace edit. The search string must be copied "
                "verbatim from the current file and must be unique. Do not return overlapping "
                "or redundant edits; merge changes that touch the same lines. Keep edits neat: "
                "for one-line changes, search and replace one line, not a whole function or section."
            ),
        },
        {
            "role": "user",
            "content": f"""Edit `{train_path}`.

Request:
{request}

Return only an `edit_train_py` tool call. Do not answer in prose.

Current file:
```python
{train_text}
```
""",
        },
    ]


def apply_model_response(train_text: str, response: str) -> tuple[str, str, str]:
    tool_blocks = extract_tool_call_edit_blocks(response)
    if tool_blocks:
        edited = apply_search_replace_blocks(train_text, tool_blocks)
        patch = make_unified_diff(train_text, edited, fromfile="a/real_train.py", tofile="b/real_train.py")
        return edited, patch, "tool_call"

    search_replace_blocks = extract_search_replace_blocks(response)
    if search_replace_blocks:
        edited = apply_search_replace_blocks(train_text, search_replace_blocks)
        patch = make_unified_diff(train_text, edited, fromfile="a/real_train.py", tofile="b/real_train.py")
        return edited, patch, "search_replace_fallback"

    patch = extract_unified_diff(response)
    if patch:
        edited = apply_unified_diff_to_text(train_text, patch)
        return edited, patch, "unified_diff_fallback"

    raise ValueError("Model response did not contain a tool call, SEARCH/REPLACE block, or unified diff.")


async def async_main():
    args = parse_args()
    train_path = args.train_file.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_text = train_path.read_text(encoding="utf-8")
    messages = build_prompt(train_path, train_text, args.request)
    response, usage = await tool_call_generate(
        messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        llm_url=args.base_url,
        llm_model_name=args.model,
        api_key=args.api_key,
    )
    if not isinstance(response, str):
        response = "" if response is None else str(response)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    response_path = out_dir / f"response_{stamp}.md"
    patch_path = out_dir / f"patch_{stamp}.diff"
    edited_path = out_dir / f"real_train_edited_{stamp}.py"
    meta_path = out_dir / f"meta_{stamp}.json"

    response_path.write_text(response, encoding="utf-8")
    edited, patch, mode = apply_model_response(train_text, response)
    patch_path.write_text(patch, encoding="utf-8")
    edited_path.write_text(edited, encoding="utf-8")

    if args.apply:
        train_path.write_text(edited, encoding="utf-8")

    meta = {
        "mode": mode,
        "train_file": str(train_path),
        "edited_path": str(edited_path),
        "patch_path": str(patch_path),
        "response_path": str(response_path),
        "applied_to_source": bool(args.apply),
        "usage": usage,
        "base_url": args.base_url,
        "model": args.model,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(meta, indent=2, sort_keys=True))


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
