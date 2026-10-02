"""Groq adapter (OpenAI-compatible chat.completions with function calling)."""
from __future__ import annotations

import json
import os
import re
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


def _recover_rejected_call(exc, tools, force_tool=None) -> LLMResponse | None:
    """Groq validates tool arguments against the JSON schema on its side and rejects the whole
    response on any mismatch (e.g. a string where the schema says array). When the rejected call is
    still valid JSON carrying a usable payload, return it instead: the agent loop repairs minor type
    mismatches itself. Only ever runs on a Groq `tool_use_failed` error.

    Open-weight models also mislabel the call itself, most often as "json" or "function" with the real
    arguments inside. When exactly one tool was asked for, the name carries no information anyway, so the
    payload is taken as a call to that tool."""
    body = getattr(exc, "body", None)
    err = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(err, dict) or err.get("code") != "tool_use_failed" or not tools:
        return None
    try:
        gen = json.loads(err.get("failed_generation") or "")
    except (json.JSONDecodeError, TypeError):
        return None  # e.g. output cut off mid-JSON: let the agent loop ask again
    if isinstance(gen, list) and gen:
        gen = gen[0]
    name = gen.get("name") if isinstance(gen, dict) else None
    args = gen.get("arguments", gen.get("parameters", {})) if isinstance(gen, dict) else None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    known = {t["name"] for t in tools}
    if name not in known:
        # The name is wrong, not the payload. Only safe when there is no choice of tool to get wrong, and
        # only with something to act on: an empty payload is worse than asking the model again.
        only = force_tool if force_tool in known else (tools[0]["name"] if len(tools) == 1 else None)
        if not isinstance(args, dict) or not args:
            args = {k: v for k, v in gen.items() if k not in ("name", "arguments", "parameters")} \
                if isinstance(gen, dict) else {}            # the payload sent as the object itself
        if not only or not args:
            return None
        name = only
    if not isinstance(args, dict):
        return None
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    content = [{"type": "tool_use", "id": call_id, "name": name, "input": args}]
    return LLMResponse("", [ToolCall(call_id, name, args)], content, "tool_calls", {})


MIN_REPLY_TOKENS = 600         # below this a structured reply is likely to be cut off; better to fail clearly
_TPM = re.compile(r"tokens per minute \(TPM\): Limit (\d+), Requested (\d+)", re.I)


def _fit_to_limit(exc, max_tokens: int) -> int | None:
    """For a 'request too large' rejection on the per-minute TOKEN limit: a smaller max_tokens that fits.
    None when it is another error, an output-token cap, or the prompt alone is too big."""
    text = str(exc)
    if getattr(exc, "status_code", None) not in (413, 429) and "request too large" not in text.lower():
        return None
    if "output tokens per minute" in text.lower():
        return None
    m = _TPM.search(text)
    if not m:
        return None
    limit, requested = int(m.group(1)), int(m.group(2))
    if requested <= limit:
        return None                    # an ordinary rate limit (the minute was busy): with_retries waits it out
    smaller = max_tokens - (requested - limit) - 64
    return smaller if smaller >= MIN_REPLY_TOKENS else None


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
        resp, exc = None, None
        for _ in range(3):
            try:
                resp = with_retries(lambda: self._client.chat.completions.create(**kwargs))
                break
            except Exception as e:  # noqa: BLE001
                exc = e
                fit = _fit_to_limit(e, kwargs["max_tokens"])
                if fit is None:
                    break
                # Groq counts prompt + max_tokens against the per-minute limit and says by how much a request is
                # over: give the reply that much less room and send it again (most replies are far shorter).
                kwargs["max_tokens"] = fit
        if resp is None:
            recovered = _recover_rejected_call(exc, tools, force_tool)
            if recovered is not None:
                return recovered
            if getattr(exc, "status_code", None) == 413 or "request too large" in str(exc).lower():
                output_limit = "output tokens per minute" in str(exc).lower()
                raise RuntimeError(
                    ("This Groq model limits OUTPUT tokens per minute below the max_tokens being requested. "
                     "Lower AGENT_MAX_TOKENS, EXTRACTION_MAX_TOKENS and REPORT_MAX_TOKENS below that limit, "
                     "or use a model without an output cap (e.g. openai/gpt-oss-20b). ")
                    if output_limit else
                    ("Groq rejected a request whose PROMPT alone is close to your tokens-per-minute limit, so even "
                     "a short reply does not fit. Use the 'Groq free tier' settings from .env.example (they shrink "
                     "prompts: CONTEXT_BUDGET_CHARS, TOOL_RESULT_CHARS, FULLTEXT_READ_CHARS, ABSTRACT_BATCH), or a "
                     "higher Groq tier. ")
                    + f"Original error: {exc}") from exc
            raise exc
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
