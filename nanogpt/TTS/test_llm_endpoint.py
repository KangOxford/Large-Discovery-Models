#!/usr/bin/env python3
"""
Dependency-free diagnostic for the OpenAI-compatible LLM endpoint used by
TTS/run_expanded_search.py.

Examples:
    python TTS/test_llm_endpoint.py \
        --base-url http://135.84.176.142:20200/v1 \
        --model checkpoint-30 \
        --operation-schema TTS/operation_schema_real_train.json \
        --train-file TTS/real_train.py

    python TTS/test_llm_endpoint.py \
        --base-url http://135.84.176.142:20200/v1 \
        --model checkpoint-30 \
        --prompt-file TTS/runs/.../states/state_0002/prompt.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = "http://127.0.0.1:52307/v1"
DEFAULT_MODEL = "Qwen3-Coder-30B-A3B-Instruct"
GENERATOR_CHOICES = ["api", "closed_loop", "tool_call", "harness", "operation_tool"]
TOOL_GENERATORS = {"tool_call", "harness", "operation_tool"}
CURL_LIKE_SYSTEM = (
    "You propose hyperparameter edit operations for an iterative model-based "
    "(Bayesian) optimization search over a single training script. Return ONLY "
    "a tool call proposing 1-2 valid operations."
)
CURL_LIKE_USER = "<schema + 历史观测>"


EDIT_TRAIN_PY_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_train_py",
        "description": (
            "Edit train.py by replacing exact code snippets. Use small, unique SEARCH strings "
            "copied exactly from the current file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short description of the intended code change.",
                },
                "edits": {
                    "type": "array",
                    "description": "One or more exact search/replace edits to apply in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search": {
                                "type": "string",
                                "description": "Exact code from the current train.py to replace.",
                            },
                            "replace": {
                                "type": "string",
                                "description": "Replacement code with the same narrow scope as search.",
                            },
                        },
                        "required": ["search", "replace"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["summary", "edits"],
        },
    },
}


HARNESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_train_py",
            "description": "Inspect the current candidate train.py text, optionally around a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional exact text or keyword to locate before returning context.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around the query. Default 12.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_train_py",
            "description": "Apply one neat exact search/replace edit to the in-memory train.py.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Exact current train.py code to replace.",
                    },
                    "replace": {
                        "type": "string",
                        "description": "Replacement code with the same narrow scope as search.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short summary of this edit.",
                    },
                },
                "required": ["search", "replace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_edit",
            "description": "Finish after the desired train.py edits have been applied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the completed candidate edit.",
                    },
                },
            },
        },
    },
]


def canonical_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_").upper()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def ordered_operation_parameters(schema_path: Path, initial_features: str) -> tuple[list[str], dict[str, Any], str]:
    data = read_json(schema_path)
    raw_parameters = data.get("parameters")
    if not isinstance(raw_parameters, dict) or not raw_parameters:
        raise ValueError(f"{schema_path} must contain a non-empty 'parameters' object.")

    raw_order = data.get("parameter_order")
    if isinstance(raw_order, list) and raw_order:
        ordered_raw_names = [str(name) for name in raw_order]
    else:
        ordered_raw_names = [str(name) for name in raw_parameters]

    by_canonical = {canonical_name(name): spec for name, spec in raw_parameters.items()}
    ordered_names = [canonical_name(name) for name in ordered_raw_names if canonical_name(name) in by_canonical]
    if not ordered_names:
        raise ValueError(f"{schema_path} did not yield any operation parameter names.")

    spec = str(initial_features or "").strip()
    if not spec or spec.lower() == "all":
        active_names = ordered_names
    elif re.fullmatch(r"\d+", spec):
        active_names = ordered_names[: max(1, min(len(ordered_names), int(spec)))]
    else:
        requested = [canonical_name(part) for part in spec.split(",") if canonical_name(part)]
        unknown = [name for name in requested if name not in by_canonical]
        if unknown:
            raise ValueError(f"Unknown --initial-operation-features names: {unknown}")
        active_names = requested
    version = str(data.get("version") or schema_path.name)
    return active_names, by_canonical, version


def make_operation_tool_schema(active_names: list[str], max_operations: int) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "propose_train_operations",
            "description": (
                "Propose active-feature edits to train.py. Only use the allowed parameter names "
                "and value ranges from the active schema. Do not propose arbitrary code patches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the proposed knob changes.",
                    },
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max(1, int(max_operations)),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": active_names,
                                    "description": "Schema parameter name to edit.",
                                },
                                "op": {
                                    "type": "string",
                                    "enum": ["set_numeric", "set_choice"],
                                    "description": "Use set_numeric for int/float parameters and set_choice for choice parameters.",
                                },
                                "value": {
                                    "description": "The new value. It must satisfy the schema for the chosen name.",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "One short reason for this operation.",
                                },
                            },
                            "required": ["name", "op", "value"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["operations"],
                "additionalProperties": False,
            },
        },
    }


def make_feature_expansion_tool(inactive_names: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "name": {
            "type": "string",
            "description": "Inactive schema parameter name to activate as a new GP/search feature.",
        },
        "rationale": {
            "type": "string",
            "description": "Short reason this additional feature should now be searchable.",
        },
    }
    if inactive_names:
        properties["name"]["enum"] = inactive_names
    return {
        "type": "function",
        "function": {
            "name": "propose_operation_feature",
            "description": (
                "Activate one additional operation feature dimension for later GP scoring. "
                "Prefer inactive schema parameters. This action does not edit train.py by itself."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    }


def build_operation_tools(args: argparse.Namespace) -> list[dict[str, Any]]:
    active_names, all_specs, _version = ordered_operation_parameters(
        args.operation_schema,
        args.initial_operation_features,
    )
    tools = [make_operation_tool_schema(active_names, args.max_operations_per_step)]
    if not args.disable_feature_expansion:
        inactive_names = [name for name in all_specs if name not in set(active_names)]
        if inactive_names:
            tools.append(make_feature_expansion_tool(inactive_names))
    return tools


def compact_train_text(path: Path, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n\n# ... train.py truncated by test_llm_endpoint.py ...\n\n" + text[-keep:]


def build_operation_prompt(args: argparse.Namespace, tools: list[dict[str, Any]]) -> str:
    if args.prompt_mode == "minimal":
        return (
            "Active features: ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN, TOTAL_BATCH_SIZE, EMBEDDING_LR.\n"
            "Return one valid operation proposal."
        )
    if args.prompt_mode == "curl":
        return CURL_LIKE_USER

    active_names = (
        tools[0]
        .get("function", {})
        .get("parameters", {})
        .get("properties", {})
        .get("operations", {})
        .get("items", {})
        .get("properties", {})
        .get("name", {})
        .get("enum", [])
    )
    train_text = compact_train_text(args.train_file, args.train_chars) if args.train_file else ""
    train_block = f"\nCurrent parent `train.py`:\n```python\n{train_text}\n```\n" if train_text else ""
    return f"""We are doing dynamically expanded model-based search over `train.py`.

Objective:
- Improve `val_bpb` after executing the script.
- The metric is lower-is-better.
- You may edit only active top-level assignments whose names appear in the active schema.

Active features: {", ".join(active_names)}

Return format:
- Choose exactly one action.
- To edit train.py, call `propose_train_operations` with 1 to {args.max_operations_per_step} operations.
- Use `set_numeric` for int/float schema parameters and `set_choice` for choice parameters.
- Do not output SEARCH/REPLACE blocks or unified diffs.
{train_block}"""


def build_search_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8", errors="replace")
    if args.prompt_mode == "minimal":
        return "Propose one tiny safe train.py edit. Return only a diff or SEARCH/REPLACE block."
    if args.prompt_mode == "curl":
        return CURL_LIKE_USER

    train_text = compact_train_text(args.train_file, args.train_chars) if args.train_file else ""
    train_block = f"\nCurrent parent `train.py`:\n```python\n{train_text}\n```\n" if train_text else ""
    return f"""We are doing test-time inference scaling for autoresearch on `train.py`.

Objective:
- Improve `val_bpb` after executing the script.
- The default metric is validation BPB, where lower is better.
- Edit only `train.py`; do not require changes to data files or dependencies.

Return format:
- If tool calling is available, call the `edit_train_py` tool with a JSON object.
- Otherwise return one small SEARCH/REPLACE edit block or a valid unified diff for train.py.
- Keep the edit tiny; this is only an endpoint diagnostic.
{train_block}"""


def selected_generators(args: argparse.Namespace) -> list[str]:
    requested = args.generator or ["all"]
    selected: list[str] = []
    for item in requested:
        for part in str(item).split(","):
            name = part.strip()
            if not name:
                continue
            if name == "all":
                for generator in GENERATOR_CHOICES:
                    if generator not in selected:
                        selected.append(generator)
                continue
            if name not in GENERATOR_CHOICES:
                raise ValueError(f"Unknown generator {name!r}. Choices: {GENERATOR_CHOICES + ['all']}")
            if name not in selected:
                selected.append(name)
    if args.skip_tools:
        selected = [name for name in selected if name not in TOOL_GENERATORS]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast diagnostic for /models, plain chat, and run_expanded_search operation-tool chat calls."
        )
    )
    parser.add_argument("--base-url", "--llm-url", default=os.environ.get("TTS_LLM_URL", DEFAULT_URL))
    parser.add_argument("--model", "--llm-model-name", default=os.environ.get("TTS_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("TTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    parser.add_argument("--no-auth", action="store_true", help="Do not send an Authorization header. This matches the curl smoke test.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--body-chars", type=int, default=4000, help="Characters of response/error body to print.")
    parser.add_argument("--dry-run", action="store_true", help="Build and optionally dump payloads without sending HTTP requests.")
    parser.add_argument(
        "--prompt-mode",
        choices=["runner", "minimal", "curl"],
        default="runner",
        help=(
            "Prompt body to use. runner matches the search runner, minimal sends a tiny repo-shaped prompt, "
            "and curl matches the short prompt from the working curl command."
        ),
    )
    parser.add_argument(
        "--factor-sweep",
        action="store_true",
        help="Run a closed_loop factor sweep that starts from the working curl shape and adds runner-like factors.",
    )
    parser.add_argument(
        "--length-sweep",
        action="store_true",
        help="Run closed_loop runner-prompt requests across several --train-chars values.",
    )
    parser.add_argument(
        "--length-sweep-chars",
        default="0,1000,2000,4000,6000,8000,10000,12000",
        help="Comma-separated train.py character limits for --length-sweep.",
    )
    parser.add_argument(
        "--generator",
        action="append",
        default=None,
        help=(
            "Generator payload(s) to test: api, closed_loop, tool_call, harness, operation_tool, or all. "
            "May be repeated or comma-separated. Default: all."
        ),
    )
    parser.add_argument("--skip-models", action="store_true", help="Skip GET /models.")
    parser.add_argument("--skip-plain", action="store_true", help="Skip extra plain-chat baseline completion.")
    parser.add_argument("--skip-tools", action="store_true", help="Skip tool-based generators: tool_call, harness, operation_tool.")
    parser.add_argument("--operation-schema", type=Path, default=Path("TTS/operation_schema_real_train.json"))
    parser.add_argument("--train-file", type=Path, default=Path("TTS/real_train.py"))
    parser.add_argument("--train-chars", type=int, default=12000)
    parser.add_argument("--initial-operation-features", default="5")
    parser.add_argument("--max-operations-per-step", type=int, default=2)
    parser.add_argument("--disable-feature-expansion", action="store_true")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Replay a saved expanded-search prompt.md as the user message for the tool test.",
    )
    parser.add_argument(
        "--tool-choice",
        default="auto",
        choices=["auto", "required", "none"],
        help="Tool choice for the operation-tool request. run_expanded_search.py uses auto.",
    )
    parser.add_argument(
        "--dump-payload-dir",
        type=Path,
        default=None,
        help="Optional directory where request payload JSON files are written before sending.",
    )
    return parser.parse_args()


def endpoint_url(base_url: str, suffix: str) -> str:
    return base_url.rstrip("/") + "/" + suffix.lstrip("/")


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars> ..."


def pretty_json_or_text(raw: bytes, limit: int) -> str:
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return truncate(text, limit)
    return truncate(json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False), limit)


def summarize_payload(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "no request body"
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    messages = payload.get("messages", [])
    tools = payload.get("tools", [])
    return (
        f"{len(raw)} bytes, messages={len(messages)}, tools={len(tools)}, "
        f"max_tokens={payload.get('max_tokens')}, temperature={payload.get('temperature')}"
    )


def request_json(
    *,
    name: str,
    method: str,
    url: str,
    api_key: str,
    send_auth: bool,
    timeout: float,
    body_chars: int,
    payload: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    headers = {}
    if send_auth:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    print(f"\n=== {name} ===")
    print(f"{method} {url}")
    print(f"request: {summarize_payload(payload)}")
    print(f"auth header: {'yes' if send_auth else 'no'}")
    started = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.time() - started
            print(f"status: {resp.status} {getattr(resp, 'reason', '')} in {elapsed:.2f}s")
            print(pretty_json_or_text(raw, body_chars))
            try:
                parsed = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                parsed = None
            return 200 <= int(resp.status) < 300, parsed
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed = time.time() - started
        print(f"status: {exc.code} {exc.reason} in {elapsed:.2f}s")
        print(pretty_json_or_text(raw, body_chars))
        return False, None
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        elapsed = time.time() - started
        print(f"network_error after {elapsed:.2f}s: {exc!r}")
        return False, None
    except Exception as exc:
        elapsed = time.time() - started
        print(f"unexpected_error after {elapsed:.2f}s: {exc!r}")
        traceback.print_exc()
        return False, None


def print_dry_run(name: str, method: str, url: str, payload: dict[str, Any] | None = None) -> None:
    print(f"\n=== {name} ===")
    print(f"{method} {url}")
    print(f"dry-run request: {summarize_payload(payload)}")


def write_payload(args: argparse.Namespace, name: str, payload: dict[str, Any]) -> None:
    if args.dump_payload_dir is None:
        return
    args.dump_payload_dir.mkdir(parents=True, exist_ok=True)
    path = args.dump_payload_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote payload: {path}")


def plain_chat_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a terse endpoint smoke-test assistant."},
            {"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }


def search_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt_mode == "curl":
        return [
            {"role": "system", "content": CURL_LIKE_SYSTEM},
            {"role": "user", "content": CURL_LIKE_USER},
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are a careful code-research agent. Return only a unified diff "
                "for train.py unless explicitly asked for a full file."
            ),
        },
        {"role": "user", "content": build_search_prompt(args)},
    ]


def base_completion_payload(args: argparse.Namespace, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }


def api_generator_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = base_completion_payload(args, search_messages(args))
    payload["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": not args.disable_thinking}
    }
    return payload


def closed_loop_generator_payload(args: argparse.Namespace) -> dict[str, Any]:
    return base_completion_payload(args, search_messages(args))


def tool_call_generator_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = base_completion_payload(args, search_messages(args))
    payload["tools"] = [EDIT_TRAIN_PY_TOOL]
    payload["tool_choice"] = {"type": "function", "function": {"name": "edit_train_py"}}
    return payload


def harness_generator_payload(args: argparse.Namespace) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a code-editing agent inside a small Hermes-style harness. "
                "Use tools to inspect and edit the current in-memory train.py. "
                "Do not disclose a direct solution or answer with a patch directly. "
                "Use inspect_train_py when you need context. "
                "Use replace_in_train_py for neat, minimal exact edits. "
                "After at least one successful edit, call finish_edit."
            ),
        },
        {"role": "user", "content": build_search_prompt(args)},
    ]
    payload = base_completion_payload(args, messages)
    payload["tools"] = HARNESS_TOOLS
    payload["tool_choice"] = "auto"
    return payload


def operation_tool_generator_payload(args: argparse.Namespace) -> dict[str, Any]:
    tools = build_operation_tools(args)
    prompt = args.prompt_file.read_text(encoding="utf-8", errors="replace") if args.prompt_file else build_operation_prompt(args, tools)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": CURL_LIKE_SYSTEM
                if args.prompt_mode == "curl"
                else (
                    "You propose train.py search actions for a dynamically expanding "
                    "operation-feature space. Call exactly one provided tool: either "
                    "propose_train_operations to edit active features, or "
                    "propose_operation_feature to activate a new feature."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "tools": tools,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.tool_choice != "none":
        payload["tool_choice"] = args.tool_choice
    return payload


def generator_payload(args: argparse.Namespace, generator: str) -> dict[str, Any]:
    if generator == "api":
        return api_generator_payload(args)
    if generator == "closed_loop":
        return closed_loop_generator_payload(args)
    if generator == "tool_call":
        return tool_call_generator_payload(args)
    if generator == "harness":
        return harness_generator_payload(args)
    if generator == "operation_tool":
        return operation_tool_generator_payload(args)
    raise ValueError(f"Unsupported generator {generator!r}.")


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def factor_sweep_cases(args: argparse.Namespace) -> list[tuple[str, argparse.Namespace]]:
    return [
        (
            "01 curl-equivalent: short prompt, temp=1.0, max_tokens=1024, no auth",
            clone_args(args, prompt_mode="curl", temperature=1.0, max_tokens=1024, no_auth=True),
        ),
        (
            "02 curl + Authorization header",
            clone_args(args, prompt_mode="curl", temperature=1.0, max_tokens=1024, no_auth=False),
        ),
        (
            "03 curl prompt + runner temp/max_tokens, no auth",
            clone_args(args, prompt_mode="curl", temperature=0.0, max_tokens=256, no_auth=True),
        ),
        (
            "04 minimal runner prompt, temp=1.0, max_tokens=1024, no auth",
            clone_args(args, prompt_mode="minimal", temperature=1.0, max_tokens=1024, no_auth=True),
        ),
        (
            "05 minimal runner prompt + runner temp/max_tokens, no auth",
            clone_args(args, prompt_mode="minimal", temperature=0.0, max_tokens=256, no_auth=True),
        ),
        (
            "06 runner prompt without train.py, temp=1.0, max_tokens=1024, no auth",
            clone_args(args, prompt_mode="runner", train_chars=0, temperature=1.0, max_tokens=1024, no_auth=True),
        ),
        (
            "07 runner prompt with 2k train.py, temp=1.0, max_tokens=1024, no auth",
            clone_args(args, prompt_mode="runner", train_chars=2000, temperature=1.0, max_tokens=1024, no_auth=True),
        ),
        (
            "08 runner prompt with 12k train.py, temp=1.0, max_tokens=1024, no auth",
            clone_args(args, prompt_mode="runner", train_chars=12000, temperature=1.0, max_tokens=1024, no_auth=True),
        ),
        (
            "09 runner prompt with 12k train.py + runner temp/max_tokens, no auth",
            clone_args(args, prompt_mode="runner", train_chars=12000, temperature=0.0, max_tokens=256, no_auth=True),
        ),
        (
            "10 runner prompt with 12k train.py + runner temp/max_tokens + auth",
            clone_args(args, prompt_mode="runner", train_chars=12000, temperature=0.0, max_tokens=256, no_auth=False),
        ),
    ]


def run_factor_sweep(args: argparse.Namespace) -> bool:
    ok = True
    print("\nRunning closed_loop factor sweep. Each case uses POST /chat/completions with no tools.")
    for case_name, case_args in factor_sweep_cases(args):
        payload = closed_loop_generator_payload(case_args)
        safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", case_name.lower()).strip("_")
        write_payload(case_args, f"factor_sweep_{safe_name}", payload)
        if case_args.dry_run:
            print_dry_run(case_name, "POST", endpoint_url(case_args.base_url, "chat/completions"), payload)
            continue
        test_ok, parsed = request_json(
            name=case_name,
            method="POST",
            url=endpoint_url(case_args.base_url, "chat/completions"),
            api_key=case_args.api_key,
            send_auth=not case_args.no_auth,
            timeout=case_args.timeout,
            body_chars=case_args.body_chars,
            payload=payload,
        )
        print_response_hint(parsed)
        ok = ok and test_ok
    return ok


def parse_int_list(text: str) -> list[int]:
    values = []
    for raw in str(text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(max(0, int(raw)))
    if not values:
        raise ValueError("Expected at least one integer in --length-sweep-chars.")
    return values


def run_length_sweep(args: argparse.Namespace) -> bool:
    ok = True
    sizes = parse_int_list(args.length_sweep_chars)
    print("\nRunning closed_loop length sweep. Each case uses the runner prompt shape with no tools.")
    for size in sizes:
        case_args = clone_args(args, prompt_mode="runner", train_chars=size)
        payload = closed_loop_generator_payload(case_args)
        name = f"runner prompt with train_chars={size}"
        write_payload(case_args, f"length_sweep_train_chars_{size}", payload)
        if case_args.dry_run:
            print_dry_run(name, "POST", endpoint_url(case_args.base_url, "chat/completions"), payload)
            continue
        test_ok, parsed = request_json(
            name=name,
            method="POST",
            url=endpoint_url(case_args.base_url, "chat/completions"),
            api_key=case_args.api_key,
            send_auth=not case_args.no_auth,
            timeout=case_args.timeout,
            body_chars=case_args.body_chars,
            payload=payload,
        )
        print_response_hint(parsed)
        ok = ok and test_ok
    return ok


def print_response_hint(parsed: dict[str, Any] | None) -> None:
    if not isinstance(parsed, dict):
        return
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    print("assistant content present:", bool(content))
    print("assistant tool_calls present:", bool(tool_calls))
    if tool_calls:
        names = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            names.append(function.get("name"))
        print("tool call names:", names)


def main() -> int:
    args = parse_args()
    ok = True
    generators = selected_generators(args)

    if not args.skip_models:
        if args.dry_run:
            print_dry_run("models", "GET", endpoint_url(args.base_url, "models"))
        else:
            test_ok, _parsed = request_json(
                name="models",
                method="GET",
                url=endpoint_url(args.base_url, "models"),
                api_key=args.api_key,
                send_auth=not args.no_auth,
                timeout=args.timeout,
                body_chars=args.body_chars,
            )
            ok = ok and test_ok

    if args.factor_sweep:
        ok = run_factor_sweep(args) and ok
        print("\nsummary:", "DRY RUN" if args.dry_run else ("PASS" if ok else "FAIL"))
        return 0 if ok else 1

    if args.length_sweep:
        ok = run_length_sweep(args) and ok
        print("\nsummary:", "DRY RUN" if args.dry_run else ("PASS" if ok else "FAIL"))
        return 0 if ok else 1

    if not args.skip_plain:
        payload = plain_chat_payload(args)
        write_payload(args, "plain_chat_payload", payload)
        if args.dry_run:
            print_dry_run("plain chat", "POST", endpoint_url(args.base_url, "chat/completions"), payload)
        else:
            test_ok, parsed = request_json(
                name="plain chat",
                method="POST",
                url=endpoint_url(args.base_url, "chat/completions"),
                api_key=args.api_key,
                send_auth=not args.no_auth,
                timeout=args.timeout,
                body_chars=args.body_chars,
                payload=payload,
            )
            print_response_hint(parsed)
            ok = ok and test_ok

    for generator in generators:
        payload = generator_payload(args, generator)
        write_payload(args, f"generator_{generator}_payload", payload)
        if args.dry_run:
            print_dry_run(f"generator {generator}", "POST", endpoint_url(args.base_url, "chat/completions"), payload)
        else:
            test_ok, parsed = request_json(
                name=f"generator {generator}",
                method="POST",
                url=endpoint_url(args.base_url, "chat/completions"),
                api_key=args.api_key,
                send_auth=not args.no_auth,
                timeout=args.timeout,
                body_chars=args.body_chars,
                payload=payload,
            )
            print_response_hint(parsed)
            ok = ok and test_ok

    print("\nsummary:", "DRY RUN" if args.dry_run else ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
