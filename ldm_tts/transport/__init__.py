"""Task-neutral proposal transport interface."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProposalRequest:
    """Transport-level chat request with no task-specific response meaning."""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("proposal request must contain at least one message")
        for message in self.messages:
            if not str(message.get("role", "")).strip():
                raise ValueError("proposal request messages require a role")


@dataclass(frozen=True)
class ProposalResponse:
    """Normalized text/tool response plus transport usage and timing."""

    text: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    usage: dict[str, int | float] = field(default_factory=dict)
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip() and not self.tool_calls:
            raise ValueError("proposal response must contain text or tool calls")
        if self.latency_seconds < 0:
            raise ValueError("proposal response latency must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class ProposalClient(Protocol):
    """Transport seam used by reservoir expansion adapters."""

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        ...


class CallableProposalClient:
    """Local adapter around a deterministic or test proposal callable."""

    def __init__(self, operation: Callable[[ProposalRequest], ProposalResponse | str]) -> None:
        self.operation = operation

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        started = time.monotonic()
        result = self.operation(request)
        if isinstance(result, ProposalResponse):
            return result
        return ProposalResponse(text=str(result), latency_seconds=time.monotonic() - started)


__all__ = [
    "CallableProposalClient",
    "ProposalClient",
    "ProposalRequest",
    "ProposalResponse",
]
