"""OpenAI-compatible endpoint preflight and circuit-breaking primitives."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar

from ldm_tts.transport import ProposalRequest, ProposalResponse


class EndpointRequestError(RuntimeError):
    """Raised when an endpoint request fails or returns an invalid response."""


class EndpointCircuitOpen(EndpointRequestError):
    """Raised when repeated failures pause further endpoint requests."""


@dataclass
class EndpointCircuitBreaker:
    """Small in-process breaker that opens after consecutive request failures."""

    failure_threshold: int = 3
    recovery_timeout_seconds: float = 300.0
    consecutive_failures: int = 0
    state: str = "closed"
    opened_at: float | None = None
    last_error: str = ""

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.recovery_timeout_seconds < 0:
            raise ValueError("recovery_timeout_seconds must be non-negative")

    def before_request(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        if self.state != "open":
            return
        opened_at = now if self.opened_at is None else self.opened_at
        elapsed = now - float(opened_at)
        if elapsed < self.recovery_timeout_seconds:
            raise EndpointCircuitOpen(
                f"Endpoint circuit is open after {self.consecutive_failures} failures: "
                f"{self.last_error}"
            )
        self.state = "half_open"

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = "closed"
        self.opened_at = None
        self.last_error = ""

    def record_failure(self, error: BaseException, now: float | None = None) -> None:
        self.consecutive_failures += 1
        self.last_error = str(error)
        if self.consecutive_failures >= self.failure_threshold:
            self.state = "open"
            self.opened_at = time.monotonic() if now is None else float(now)

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }


ResultT = TypeVar("ResultT")


def call_with_circuit_breaker(
    breaker: EndpointCircuitBreaker,
    operation: Callable[..., ResultT],
    /,
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    """Run one request and open the circuit when the threshold is reached."""

    breaker.before_request()
    try:
        result = operation(*args, **kwargs)
    except EndpointRequestError as exc:
        breaker.record_failure(exc)
        if breaker.state == "open":
            raise EndpointCircuitOpen(
                f"Endpoint circuit opened after {breaker.consecutive_failures} failures: {exc}"
            ) from exc
        raise
    breaker.record_success()
    return result


def request_openai_chat(
    *,
    url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    extra_body: Mapping[str, Any] | None = None,
) -> str:
    """Return text from one validated OpenAI-compatible chat response."""

    result = request_openai_chat_response(
        url=url,
        model=model,
        api_key=api_key,
        messages=messages,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EndpointRequestError(
            "Chat response has no choices[0].message.content"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise EndpointRequestError("Chat response content is empty or not text")
    return content


def request_openai_chat_response(
    *,
    url: str,
    model: str,
    api_key: str,
    messages: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    tools: Sequence[Mapping[str, Any]] = (),
    tool_choice: Any = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one validated raw OpenAI-compatible chat response."""

    endpoint = chat_completions_url(url)
    body: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = [dict(tool) for tool in tools]
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if extra_body:
        reserved = set(body) & set(extra_body)
        if reserved:
            raise EndpointRequestError(
                "extra_body cannot override reserved chat field(s): " + ", ".join(sorted(reserved))
            )
        body.update(dict(extra_body))
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise EndpointRequestError(f"HTTP {exc.code} from chat endpoint") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EndpointRequestError(f"Chat endpoint request failed: {exc}") from exc
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EndpointRequestError(
            "Chat response has no choices[0].message"
        ) from exc
    if not isinstance(message, dict):
        raise EndpointRequestError("Chat response message is not an object")
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    if not (isinstance(content, str) and content.strip()) and not (
        isinstance(tool_calls, list) and tool_calls
    ):
        raise EndpointRequestError("Chat response contains neither text nor tool calls")
    if not isinstance(result, dict):
        raise EndpointRequestError("Chat response root is not an object")
    return result


def preflight_openai_chat(
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 30.0,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe authentication, model availability, latency, and response shape."""

    started = time.monotonic()
    content = request_openai_chat(
        url=url,
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": "Reply with exactly OK."}],
        timeout_seconds=timeout_seconds,
        max_tokens=32,
        temperature=0.0,
        extra_body=extra_body,
    )
    return {
        "status": "ok",
        "model": model,
        "latency_seconds": round(time.monotonic() - started, 6),
        "response_nonempty": bool(content.strip()),
    }


def chat_completions_url(raw: str) -> str:
    base = str(raw).strip().rstrip("/")
    if not base:
        raise EndpointRequestError("Chat endpoint URL is empty")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


class OpenAICompatibleProposalClient:
    """OpenAI-compatible proposal adapter with retry and breaker policy."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 120.0,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        extra_body: Mapping[str, Any] | None = None,
        breaker: EndpointCircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("proposal timeout_seconds must be positive")
        if max_tokens < 1:
            raise ValueError("proposal max_tokens must be positive")
        if max_retries < 0:
            raise ValueError("proposal max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("proposal retry_backoff_seconds must be non-negative")
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.extra_body = dict(extra_body or {})
        self.breaker = breaker or EndpointCircuitBreaker()
        self.sleep = sleep

    def preflight(self) -> dict[str, Any]:
        preflight_body = {
            name: value
            for name, value in self.extra_body.items()
            if name != "response_format"
        }
        return preflight_openai_chat(
            url=self.url,
            model=self.model,
            api_key=self.api_key,
            timeout_seconds=min(self.timeout_seconds, 30.0),
            extra_body=preflight_body,
        )

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        started = time.monotonic()
        last_error: EndpointRequestError | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                raw = call_with_circuit_breaker(
                    self.breaker,
                    request_openai_chat_response,
                    url=self.url,
                    model=self.model,
                    api_key=self.api_key,
                    messages=request.messages,
                    timeout_seconds=self.timeout_seconds,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    extra_body=self.extra_body,
                )
                message = raw["choices"][0]["message"]
                return ProposalResponse(
                    text=str(message.get("content") or ""),
                    tool_calls=tuple(
                        dict(item)
                        for item in message.get("tool_calls", [])
                        if isinstance(item, dict)
                    ),
                    usage=_numeric_usage(raw.get("usage")),
                    latency_seconds=time.monotonic() - started,
                    metadata={
                        "model": raw.get("model", self.model),
                        "finish_reason": raw["choices"][0].get("finish_reason"),
                        "attempts": attempt,
                        **dict(request.metadata),
                    },
                )
            except EndpointRequestError as exc:
                last_error = exc
                if attempt > self.max_retries:
                    break
                if self.retry_backoff_seconds:
                    self.sleep(self.retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error


def _numeric_usage(value: Any) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): numeric
        for name, numeric in value.items()
        if not isinstance(numeric, bool) and isinstance(numeric, (int, float))
    }


__all__ = [
    "EndpointCircuitBreaker",
    "EndpointCircuitOpen",
    "EndpointRequestError",
    "OpenAICompatibleProposalClient",
    "call_with_circuit_breaker",
    "chat_completions_url",
    "preflight_openai_chat",
    "request_openai_chat",
    "request_openai_chat_response",
]
