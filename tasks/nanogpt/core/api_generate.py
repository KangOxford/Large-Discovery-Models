import asyncio
import json
from typing import Any


EDIT_TRAIN_PY_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_train_py",
        "description": (
            "Edit train.py by replacing exact code snippets. Use small, unique SEARCH strings "
            "copied exactly from the current file. Keep edits neat and minimal: for a one-line "
            "hyperparameter change, search for only that line. Do not wrap large unchanged setup, "
            "optimizer, dataloader, or training-loop regions around a small change. Do not return "
            "overlapping or redundant edits; if two changes touch the same lines, merge them into one edit."
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
                    "description": (
                        "One or more exact search/replace edits to apply in order. Edits must be "
                        "non-overlapping, and each later search must match the file after earlier "
                        "edits have been applied. Each edit should be the smallest unique snippet "
                        "that covers the intended changed lines."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "search": {
                                "type": "string",
                                "description": (
                                    "Exact code from the current train.py to replace. Keep it short, "
                                    "unique, and copied verbatim. For one-line changes, use one line."
                                ),
                            },
                            "replace": {
                                "type": "string",
                                "description": (
                                    "Replacement code with the same narrow scope as search. Avoid "
                                    "rewriting unchanged surrounding code."
                                ),
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
                        "description": "Optional exact text or regex-like keyword to locate before returning context.",
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
            "description": (
                "Apply one neat exact search/replace edit to the in-memory train.py. "
                "Use the smallest unique search text that covers the intended changed lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Exact current train.py code to replace. Keep it short and unique.",
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
            "description": "Finish after the desired train.py edits have been applied through replace_in_train_py.",
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


def _require_provider_setting(value: Any, environment_name: str) -> str:
    setting = "" if value is None else str(value).strip()
    if not setting:
        raise ValueError(
            f"Missing model provider setting. Set {environment_name} or pass it explicitly."
        )
    return setting


async def openai_compatible_generate(
        input_message,
        max_tokens=2048,
        temperature=0.,
        stop=None,
        llm_url=None,
        llm_model_name=None,
        disable_thinking=False,
        api_key=None,
        chat_template_extra=False,
        tools=None,
        tool_choice=None,
        logprobs=False,
        top_logprobs=None,
    ):
    """
    Generate from an OpenAI-compatible chat-completions endpoint.

    Returns (content, usage_dict), where usage_dict includes prompt_tokens,
    completion_tokens, and total_tokens when the provider returns them.
    """
    llm_url = _require_provider_setting(llm_url, "LLM_BASE_URL")
    llm_model_name = _require_provider_setting(llm_model_name, "LLM_MODEL_NAME")
    completion_params = {
        "messages": input_message,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if llm_model_name is not None:
        completion_params["model"] = llm_model_name
    if stop is not None:
        completion_params["stop"] = stop
    if tools is not None:
        completion_params["tools"] = tools
    if tool_choice is not None:
        completion_params["tool_choice"] = tool_choice
    if logprobs:
        completion_params["logprobs"] = True
        if top_logprobs is not None and int(top_logprobs) > 0:
            completion_params["top_logprobs"] = int(top_logprobs)
    if chat_template_extra:
        completion_params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": not disable_thinking}
        }
        if stop is not None:
            completion_params["extra_body"]["include_stop_str_in_output"] = True

    max_retries = 5
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await _create_chat_completion(
                llm_url=llm_url,
                api_key="EMPTY" if api_key is None else api_key,
                completion_params=completion_params,
            )
        except Exception as e:
            print(f"API Generate failed...")
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            raise ConnectionError(
                f"Failed to get valid result after {max_retries} attempts. "
                f"Last error: {str(last_exception)}"
            ) from last_exception


async def openai_compatible_chat_turn(
        input_message,
        max_tokens=2048,
        temperature=0.,
        stop=None,
        llm_url=None,
        llm_model_name=None,
        disable_thinking=False,
        api_key=None,
        chat_template_extra=False,
        tools=None,
        tool_choice=None,
        logprobs=False,
        top_logprobs=None,
    ):
    llm_url = _require_provider_setting(llm_url, "LLM_BASE_URL")
    llm_model_name = _require_provider_setting(llm_model_name, "LLM_MODEL_NAME")
    completion_params = {
        "messages": input_message,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if llm_model_name is not None:
        completion_params["model"] = llm_model_name
    if stop is not None:
        completion_params["stop"] = stop
    if tools is not None:
        completion_params["tools"] = tools
    if tool_choice is not None:
        completion_params["tool_choice"] = tool_choice
    if logprobs:
        completion_params["logprobs"] = True
        if top_logprobs is not None and int(top_logprobs) > 0:
            completion_params["top_logprobs"] = int(top_logprobs)
    if chat_template_extra:
        completion_params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": not disable_thinking}
        }
        if stop is not None:
            completion_params["extra_body"]["include_stop_str_in_output"] = True

    max_retries = 5
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await _create_chat_completion_turn(
                llm_url=llm_url,
                api_key="EMPTY" if api_key is None else api_key,
                completion_params=completion_params,
            )
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            raise ConnectionError(
                f"Failed to get valid result after {max_retries} attempts. "
                f"Last error: {str(last_exception)}"
            ) from last_exception


async def tool_call_generate(
        input_message,
        max_tokens=2048,
        temperature=0.,
        stop=None,
        llm_url=None,
        llm_model_name=None,
        disable_thinking=False,
        api_key=None,
        logprobs=False,
        top_logprobs=None,
    ):
    return await openai_compatible_generate(
        input_message,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        llm_url=llm_url,
        llm_model_name=llm_model_name,
        disable_thinking=disable_thinking,
        api_key=api_key,
        tools=[EDIT_TRAIN_PY_TOOL],
        tool_choice={"type": "function", "function": {"name": "edit_train_py"}},
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )


async def harness_generate(
        prompt,
        current_train_text,
        max_tokens=2048,
        temperature=0.,
        stop=None,
        llm_url=None,
        llm_model_name=None,
        disable_thinking=False,
        api_key=None,
        max_turns=6,
        logprobs=False,
        top_logprobs=None,
    ):
    harness = TrainPyHarness(current_train_text)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a code-editing agent inside a small Hermes-style harness. "
                "Use tools to inspect and edit the current in-memory train.py. "
                "Do not disclose a direct solution or answer with a patch directly. "
                "When checking any claim, only verify it from tool-observed evidence, "
                "and cite that evidence in tool calls or concise summaries. "
                "Use inspect_train_py when you need context. "
                "Use replace_in_train_py for neat, minimal exact edits. "
                "For one-line hyperparameter changes, replace one line. "
                "After at least one successful edit, call finish_edit."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    transcript: list[dict[str, Any]] = []

    for _turn in range(1, max(1, int(max_turns)) + 1):
        assistant_message, usage = await openai_compatible_chat_turn(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            llm_url=llm_url,
            llm_model_name=llm_model_name,
            disable_thinking=disable_thinking,
            api_key=api_key,
            tools=HARNESS_TOOLS,
            tool_choice="auto",
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        total_usage = add_usage(total_usage, usage)
        messages.append(assistant_message)
        transcript.append(compact_message_for_log(assistant_message))

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            content = assistant_message.get("content")
            if isinstance(content, str) and "finish" in content.lower() and harness.edits:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the harness tools. Do not disclose a direct solution; "
                        "verify claims only from tool-observed evidence. "
                        "Call replace_in_train_py for edits, then finish_edit."
                    ),
                }
            )
            continue

        should_finish = False
        for tool_call in tool_calls:
            tool_call_id = get_tool_call_id(tool_call)
            name = get_tool_call_name(tool_call)
            args = get_tool_call_arguments(tool_call)
            result = harness.invoke(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }
            )
            transcript.append({"role": "tool", "name": name, "content": result})
            if name == "finish_edit" and harness.edits:
                should_finish = True
        if should_finish:
            break

    response = harness.to_search_replace_response(transcript)
    return response, total_usage


async def vllm_generate(
        input_message, 
        max_tokens=2048, 
        temperature=0., 
        stop=None, 
        llm_url=None,
        llm_model_name=None,
        disable_thinking=False,
        api_key=None,
        logprobs=False,
        top_logprobs=None,
    ):
    """
    Generate a chat completion from an OpenAI-compatible endpoint.
    """
    content, usage = await openai_compatible_generate(
        input_message,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        llm_url=llm_url,
        llm_model_name=llm_model_name,
        disable_thinking=disable_thinking,
        api_key=api_key,
        chat_template_extra=True,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )
    return content, usage


async def _create_chat_completion(llm_url, api_key, completion_params):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return await _create_chat_completion_httpx(llm_url, api_key, completion_params)

    client = AsyncOpenAI(base_url=llm_url, api_key=api_key)
    completion = await client.chat.completions.create(**completion_params)
    usage = normalize_usage(None if completion.usage is None else completion.usage)
    choice = completion.choices[0]
    usage = attach_logprobs(usage, getattr(choice, "logprobs", None))
    content = normalize_message_content(choice.message)
    return content, usage


async def _create_chat_completion_turn(llm_url, api_key, completion_params):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return await _create_chat_completion_turn_httpx(llm_url, api_key, completion_params)

    client = AsyncOpenAI(base_url=llm_url, api_key=api_key)
    completion = await client.chat.completions.create(**completion_params)
    usage = normalize_usage(None if completion.usage is None else completion.usage)
    choice = completion.choices[0]
    usage = attach_logprobs(usage, getattr(choice, "logprobs", None))
    message = message_to_dict(choice.message)
    return message, usage


async def _create_chat_completion_httpx(llm_url, api_key, completion_params):
    import httpx

    url = llm_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=completion_params)
        response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    content = normalize_message_content(choice.get("message", {}))
    usage = normalize_usage(payload.get("usage", {}))
    usage = attach_logprobs(usage, choice.get("logprobs"))
    return content, usage


async def _create_chat_completion_turn_httpx(llm_url, api_key, completion_params):
    import httpx

    url = llm_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=completion_params)
        response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    message = choice.get("message", {})
    usage = normalize_usage(payload.get("usage", {}))
    usage = attach_logprobs(usage, choice.get("logprobs"))
    return normalize_message_dict(message), usage


def message_to_dict(message):
    if isinstance(message, dict):
        return normalize_message_dict(message)
    result: dict[str, Any] = {"role": getattr(message, "role", "assistant") or "assistant"}
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = content_blocks_to_text(content)
    if content is not None:
        result["content"] = content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = normalize_tool_calls_for_api(tool_calls)
    return result


def normalize_message_dict(message):
    result = {
        "role": message.get("role", "assistant"),
    }
    if message.get("content") is not None:
        content = message.get("content")
        result["content"] = content_blocks_to_text(content) if isinstance(content, list) else content
    if message.get("tool_calls"):
        result["tool_calls"] = normalize_tool_calls_for_api(message.get("tool_calls"))
    return result


def normalize_message_content(message):
    if isinstance(message, dict):
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
    else:
        content = getattr(message, "content", None)
        reasoning_content = getattr(message, "reasoning_content", None)
        tool_calls = getattr(message, "tool_calls", None)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content_blocks_to_text(content)
    if content is not None:
        return str(content)
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content
    if tool_calls:
        return serialize_tool_calls(tool_calls)
    return ""


def serialize_tool_calls(tool_calls):
    normalized = normalize_tool_calls_for_log(tool_calls)
    return "<tool_calls>\n" + json.dumps(normalized, indent=2, sort_keys=True) + "\n</tool_calls>"


def normalize_tool_calls_for_api(tool_calls):
    normalized = []
    for tool_call in tool_calls:
        tool_id, name, arguments = unpack_tool_call(tool_call)
        normalized.append(
            {
                "id": tool_id or f"call_{len(normalized) + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return normalized


def normalize_tool_calls_for_log(tool_calls):
    normalized = []
    for tool_call in tool_calls:
        tool_id, name, arguments = unpack_tool_call(tool_call)
        normalized.append(
            {
                "id": tool_id or f"call_{len(normalized) + 1}",
                "name": name,
                "arguments": arguments,
            }
        )
    return normalized


def unpack_tool_call(tool_call):
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        name = function.get("name") or tool_call.get("name")
        arguments = function.get("arguments") or tool_call.get("arguments") or {}
        tool_id = tool_call.get("id")
    else:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None) if function is not None else getattr(tool_call, "name", None)
        arguments = getattr(function, "arguments", None) if function is not None else getattr(tool_call, "arguments", {})
        tool_id = getattr(tool_call, "id", None)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments}
    if not isinstance(arguments, dict):
        arguments = {}
    return tool_id, name, arguments


def content_blocks_to_text(blocks):
    parts = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif "content" in block:
                parts.append(str(block["content"]))
            else:
                parts.append(str(block))
        else:
            text = getattr(block, "text", None)
            parts.append(text if isinstance(text, str) else str(block))
    return "\n".join(part for part in parts if part)


class TrainPyHarness:
    def __init__(self, text):
        self.text = text
        self.edits: list[dict[str, str]] = []
        self.summary = ""

    def invoke(self, name, args):
        if name == "inspect_train_py":
            return self.inspect(args)
        if name == "replace_in_train_py":
            return self.replace(args)
        if name == "finish_edit":
            summary = str(args.get("summary") or "").strip()
            if summary:
                self.summary = summary
            if not self.edits:
                return "ERROR: no edits have been applied yet. Use replace_in_train_py first."
            return f"OK: finished with {len(self.edits)} applied edit(s)."
        return f"ERROR: unknown harness tool {name!r}."

    def inspect(self, args):
        query = str(args.get("query") or "").strip()
        context_lines = int(args.get("context_lines", 12) or 12)
        context_lines = max(1, min(context_lines, 80))
        lines = self.text.splitlines()
        if not query:
            return numbered_excerpt(lines, 1, min(len(lines), context_lines * 2))

        matches = []
        for index, line in enumerate(lines, start=1):
            if query in line:
                matches.append(index)
        if not matches:
            return f"NOT FOUND: {query!r}. Try a shorter exact substring."
        chunks = []
        for line_no in matches[:5]:
            start = max(1, line_no - context_lines)
            end = min(len(lines), line_no + context_lines)
            chunks.append(numbered_excerpt(lines, start, end))
        return "\n---\n".join(chunks)

    def replace(self, args):
        search = str(args.get("search") or "")
        replace = str(args.get("replace") or "")
        summary = str(args.get("summary") or "").strip()
        if not search:
            return "ERROR: search must be non-empty."
        count = self.text.count(search)
        if count == 0:
            return (
                "ERROR: search text was not found exactly in the current train.py. "
                "Call inspect_train_py and try a smaller verbatim snippet."
            )
        if count > 1:
            return (
                f"ERROR: search text matched {count} times. "
                "Use a slightly larger but still neat unique snippet."
            )
        self.text = self.text.replace(search, replace, 1)
        self.edits.append({"search": search, "replace": replace, "summary": summary})
        if summary:
            self.summary = summary
        return f"OK: applied edit {len(self.edits)}. Current train.py length: {len(self.text)} chars."

    def to_search_replace_response(self, transcript):
        if not self.edits:
            transcript_text = json.dumps(transcript, indent=2, ensure_ascii=False)
            return (
                "Summary: harness did not apply any edits\n\n"
                "<harness_transcript>\n"
                + transcript_text
                + "\n</harness_transcript>\n"
            )
        summary = self.summary or "Harness applied train.py edits."
        parts = [
            f"Summary: {summary}",
            "",
            "<harness_transcript>",
            json.dumps(transcript, indent=2, ensure_ascii=False),
            "</harness_transcript>",
            "",
        ]
        for edit in self.edits:
            parts.extend(
                [
                    "train.py",
                    "<<<<<<< SEARCH",
                    edit["search"],
                    "=======",
                    edit["replace"],
                    ">>>>>>> REPLACE",
                    "",
                ]
            )
        return "\n".join(parts)


def numbered_excerpt(lines, start, end):
    return "\n".join(f"{line_no:5d}: {lines[line_no - 1]}" for line_no in range(start, end + 1))


def get_tool_call_id(tool_call):
    if isinstance(tool_call, dict):
        return tool_call.get("id") or f"call_{id(tool_call)}"
    return getattr(tool_call, "id", None) or f"call_{id(tool_call)}"


def get_tool_call_name(tool_call):
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        return function.get("name") or tool_call.get("name") or ""
    function = getattr(tool_call, "function", None)
    return (getattr(function, "name", None) if function is not None else None) or getattr(tool_call, "name", "") or ""


def get_tool_call_arguments(tool_call):
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        args = function.get("arguments") if isinstance(function, dict) else None
        if args is None:
            args = tool_call.get("arguments", {})
    else:
        function = getattr(tool_call, "function", None)
        args = getattr(function, "arguments", None) if function is not None else getattr(tool_call, "arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw_arguments": args}
    if isinstance(args, dict):
        return args
    return {}


def compact_message_for_log(message):
    compact = {"role": message.get("role", "assistant")}
    if message.get("content"):
        compact["content"] = str(message["content"])[:2000]
    if message.get("tool_calls"):
        compact["tool_calls"] = [
            {
                "name": get_tool_call_name(tool_call),
                "arguments": get_tool_call_arguments(tool_call),
            }
            for tool_call in message["tool_calls"]
        ]
    return compact


def add_usage(left, right):
    merged = {
        "prompt_tokens": int(left.get("prompt_tokens", 0)) + int(right.get("prompt_tokens", 0)),
        "completion_tokens": int(left.get("completion_tokens", 0)) + int(right.get("completion_tokens", 0)),
        "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
    }
    if left.get("logprobs") is not None or right.get("logprobs") is not None:
        merged["logprobs"] = merge_logprobs(left.get("logprobs"), right.get("logprobs"))
    return merged


def attach_logprobs(usage, logprobs):
    serializable = jsonable_logprobs(logprobs)
    if serializable is not None:
        usage["logprobs"] = serializable
    return usage


def merge_logprobs(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, dict) and isinstance(right, dict):
        merged = dict(left)
        if isinstance(left.get("content"), list) or isinstance(right.get("content"), list):
            merged["content"] = list(left.get("content") or []) + list(right.get("content") or [])
        if isinstance(left.get("token_logprobs"), list) or isinstance(right.get("token_logprobs"), list):
            merged["token_logprobs"] = list(left.get("token_logprobs") or []) + list(right.get("token_logprobs") or [])
        if isinstance(left.get("tokens"), list) or isinstance(right.get("tokens"), list):
            merged["tokens"] = list(left.get("tokens") or []) + list(right.get("tokens") or [])
        return merged
    return {"parts": [left, right]}


def jsonable_logprobs(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return jsonable_logprobs(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable_logprobs(value.dict())
    if isinstance(value, dict):
        return {str(key): jsonable_logprobs(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_logprobs(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_usage(usage):
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }
