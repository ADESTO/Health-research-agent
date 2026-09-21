"""Provider-neutral LLM interface (the "swap seam").

Every agent talks to an `LLMClient`. Messages use one internal format (Anthropic-style content
blocks); each provider adapter translates to/from its own wire format. So the orchestrator and all
agents run unchanged on Claude or Groq.

Internal message format:
    {"role": "user" | "assistant", "content": [block, ...]}
    block = {"type": "text", "text": str}
          | {"type": "tool_use", "id": str, "name": str, "input": dict}
          | {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}
Tool spec:
    {"name": str, "description": str, "input_schema": <JSON schema>}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    content: list[dict]            # assistant content blocks, internal format (append to history as-is)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0          # all input tokens, including cache reads
    cached_input_tokens: int = 0   # of which served from the prompt cache (billed at a fraction)
    output_tokens: int = 0
    calls: int = 0

    def add(self, u: dict[str, int]) -> None:
        self.input_tokens += u.get("input_tokens", 0)
        self.cached_input_tokens += u.get("cached_input_tokens", 0)
        self.output_tokens += u.get("output_tokens", 0)
        self.calls += 1


class LLMClient(Protocol):
    provider: str
    model: str
    usage: Usage

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        force_tool: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def with_retries(fn, *, attempts: int = 5, base_delay: float = 2.0, retry_on: tuple = (Exception,)):
    """Retry transient provider errors (rate limits, 5xx, timeouts) with exponential backoff."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            last = exc
            name = type(exc).__name__.lower()
            status = getattr(exc, "status_code", None)
            transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or any(
                k in name for k in ("ratelimit", "timeout", "connection", "overloaded", "internalserver")
            )
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay * (2**i))
    raise last  # pragma: no cover
