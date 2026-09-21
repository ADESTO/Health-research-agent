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
        """Mark cache breakpoints: system prompt, tool list, and the end of the conversation so far.
        Agent loops resend a growing prefix every turn; caching it cuts input cost substantially."""
        if not self.prompt_cache:
            return system, messages, tools
        sys_blocks = [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]
        tools = copy.deepcopy(tools) if tools else tools
        if tools:
            tools[-1]["cache_control"] = _EPHEMERAL
        messages = copy.deepcopy(messages)
        if messages and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
            messages[-1]["content"][-1]["cache_control"] = _EPHEMERAL
        return sys_blocks, messages, tools

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
