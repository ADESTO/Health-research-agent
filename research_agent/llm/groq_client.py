"""Groq adapter (OpenAI-compatible chat.completions with function calling)."""
from __future__ import annotations

import json
import os
import uuid

from research_agent.llm.base import LLMResponse, ToolCall, Usage, with_retries


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        blocks = msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
        if msg["role"] == "assistant":
            text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
            tcs = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                for b in blocks if b["type"] == "tool_use"
            ]
            m = {"role": "assistant", "content": text or None}
            if tcs:
                m["tool_calls"] = tcs
            out.append(m)
        else:
            texts = [b["text"] for b in blocks if b["type"] == "text"]
            for b in blocks:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


class GroqClient:
    provider = "groq"

    def __init__(self, model: str, api_key: str | None = None):
        from groq import Groq

        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = model
        self.usage = Usage()

    def chat(self, system, messages, tools=None, force_tool=None, max_tokens=2048, temperature=0.0):
        kwargs = dict(model=self.model, messages=_to_openai_messages(system, messages),
                      max_tokens=max_tokens, temperature=temperature)
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}}
                for t in tools
            ]
            kwargs["tool_choice"] = (
                {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
            )
        resp = with_retries(lambda: self._client.chat.completions.create(**kwargs))
        choice = resp.choices[0]
        msg = choice.message

        content, calls = [], []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            call_id = tc.id or f"call_{uuid.uuid4().hex[:12]}"
            content.append({"type": "tool_use", "id": call_id, "name": tc.function.name, "input": args})
            calls.append(ToolCall(call_id, tc.function.name, args))
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                 "output_tokens": getattr(resp.usage, "completion_tokens", 0)}
        self.usage.add(usage)
        return LLMResponse(msg.content or "", calls, content, choice.finish_reason or "", usage)
