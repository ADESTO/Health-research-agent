from __future__ import annotations

import copy
import inspect
import os

from research_agent.llm.base import LLMResponse, ToolCall, Usage, with_retries

_EPHEMERAL = {"type": "ephemeral"}


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, prompt_cache: bool = True):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.usage = Usage()
        self.prompt_cache = prompt_cache
        # Recent SDK versions dropped `temperature`; only send it where it is accepted.
        self._accepts_temperature = "temperature" in inspect.signature(self._client.messages.create).parameters

    def _cached(self, system: str, messages: list[dict], tools: list[dict] | None):
        """Mark cache breakpoints: the system prompt, the tool list, and — only when there is a conversation
        to reuse — the end of it.

        The system prompt and the tool schema are the same on every call a step makes, so caching them is
        always worth it: extraction sends one large, identical prefix per paper, and an agent loop resends a
        growing one every turn.

        The third breakpoint is not always worth it. A cache WRITE costs more than an ordinary input token,
        so marking the end of the conversation only pays when a later call will read it back. In an agent
        loop it will: the next turn resends everything up to here. In a one-shot call it never will — a
        paper is read once, and the paper's own text is the bulk of the request — so marking it would charge
        a write premium on the biggest part of the run's token spend for a cache entry nothing ever reads.
        So the breakpoint goes in only once the conversation has a turn behind it."""
        if not self.prompt_cache:
            return system, messages, tools
        sys_blocks = [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]
        tools = copy.deepcopy(tools) if tools else tools
        if tools:
            tools[-1]["cache_control"] = _EPHEMERAL
        messages = copy.deepcopy(messages)
        if self._reusable(messages):
            messages[-1]["content"][-1]["cache_control"] = _EPHEMERAL
        return sys_blocks, messages, tools

    @staticmethod
    def _reusable(messages: list[dict]) -> bool:
        """Will a later call resend this conversation? Only if it already holds an assistant turn, which is
        what tells a one-shot request apart from a loop in progress."""
        if len(messages) < 2 or not any(m.get("role") == "assistant" for m in messages):
            return False
        return bool(messages[-1].get("content")) and isinstance(messages[-1]["content"], list)

    def chat(self, system, messages, tools=None, force_tool=None, max_tokens=2048, temperature=0.0):
        system_p, messages_p, tools_p = self._cached(system, messages, tools)
        kwargs = dict(model=self.model, system=system_p, messages=messages_p, max_tokens=max_tokens)
        if self._accepts_temperature:
            kwargs["temperature"] = temperature
        if tools_p:
            kwargs["tools"] = tools_p
            if force_tool:
                kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
        resp = with_retries(lambda: self._client.messages.create(**kwargs))

        content, text_parts, calls = [], [], []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                calls.append(ToolCall(block.id, block.name, dict(block.input)))
        u = resp.usage
        usage = {"input_tokens": (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                 + (getattr(u, "cache_creation_input_tokens", 0) or 0),
                 "cached_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "output_tokens": u.output_tokens or 0}
        self.usage.add(usage)
        return LLMResponse("\n".join(text_parts), calls, content, resp.stop_reason or "", usage)
