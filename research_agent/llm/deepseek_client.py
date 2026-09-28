"""DeepSeek adapter (OpenAI-compatible /chat/completions over plain httpx: no extra SDK needed).

Thinking mode: DeepSeek turns it on by default. With tools, it requires the model's
`reasoning_content` to be sent back on every later request, so we keep it as a "reasoning" block in
the internal message format and return it. DEEPSEEK_THINKING=disabled (our default) is faster and
cheaper for this tool-heavy workload.
"""
from __future__ import annotations

import json
import os
import uuid

import httpx

from research_agent.llm.base import LLMResponse, ToolCall, Usage, with_retries
from research_agent.llm.groq_client import _to_openai_messages


class DeepSeekError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response  # lets with_retries read retry-after


def _messages(system: str, messages: list[dict]) -> list[dict]:
    out = _to_openai_messages(system, messages)
    # re-attach reasoning to the matching assistant turns (same order as in `messages`)
    reasonings = ["\n".join(b.get("text", "") for b in m["content"] if b.get("type") == "reasoning")
                  for m in messages if m["role"] == "assistant" and isinstance(m["content"], list)]
    assistants = [m for m in out if m["role"] == "assistant"]
    for m, r in zip(assistants, reasonings):
        if r:
            m["reasoning_content"] = r
    return out


class DeepSeekClient:
    provider = "deepseek"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 thinking: str | None = None, transport: httpx.BaseTransport | None = None):
        self.model = model
        self.usage = Usage()
        self._key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self._key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (add it to .env)")
        self._url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self._thinking = (thinking or os.getenv("DEEPSEEK_THINKING") or "disabled").lower()
        self._http = httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0), transport=transport)

    def _post(self, payload: dict) -> dict:
        r = self._http.post(f"{self._url}/chat/completions", json=payload,
                            headers={"Authorization": f"Bearer {self._key}"})
        if r.status_code >= 400:
            try:
                detail = r.json().get("error", {}).get("message") or r.text
            except ValueError:
                detail = r.text
            hint = {401: "check DEEPSEEK_API_KEY", 402: "account balance is empty: top up at platform.deepseek.com",
                    422: "request parameters were rejected"}.get(r.status_code, "")
            raise DeepSeekError(f"DeepSeek error {r.status_code}: {detail[:400]}" + (f" ({hint})" if hint else ""),
                                status_code=r.status_code, response=r)
        return r.json()

    def chat(self, system, messages, tools=None, force_tool=None, max_tokens=2048, temperature=0.0):
        payload = {"model": self.model, "messages": _messages(system, messages), "max_tokens": max_tokens,
                   "thinking": {"type": "enabled" if self._thinking == "enabled" else "disabled"}}
        if self._thinking != "enabled":
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools]
            payload["tool_choice"] = ({"type": "function", "function": {"name": force_tool}}
                                      if force_tool else "auto")
        data = with_retries(lambda: self._post(payload))

        msg = data["choices"][0]["message"]
        content, calls = [], []
        if msg.get("reasoning_content"):
            content.append({"type": "reasoning", "text": msg["reasoning_content"]})
        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": fn.get("arguments")}
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            content.append({"type": "tool_use", "id": call_id, "name": fn.get("name"), "input": args})
            calls.append(ToolCall(call_id, fn.get("name"), args))
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0),
                 "cached_input_tokens": u.get("prompt_cache_hit_tokens", 0),
                 "output_tokens": u.get("completion_tokens", 0)}
        self.usage.add(usage)
        return LLMResponse(msg.get("content") or "", calls, content,
                           data["choices"][0].get("finish_reason") or "", usage)
