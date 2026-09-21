"""Generic tool-using agent loop, shared by every specialist agent and the orchestrator.

Each agent = its own system prompt + its own small tool set + a `finish` tool whose schema is that
agent's output contract. The agent loops (LLM -> tool calls -> results) until it calls `finish`;
its output is saved as a note in the shared run state, where later agents can read it.
"""
from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from research_agent.config import settings
from research_agent.tools.base import Tool

MAX_RESULT_CHARS = 9000


def _dump(obj) -> str:
    """Serialise a tool result, shrinking the longest list first so the JSON stays valid."""
    text = json.dumps(obj, default=str, ensure_ascii=False)
    if len(text) <= MAX_RESULT_CHARS or not isinstance(obj, dict):
        return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "…[truncated]"
    obj = dict(obj)
    dropped = 0
    while len(text) > MAX_RESULT_CHARS:
        lists = [(k, v) for k, v in obj.items() if isinstance(v, list) and len(v) > 1]
        if not lists:
            return text[:MAX_RESULT_CHARS] + "…[truncated]"
        key, longest = max(lists, key=lambda kv: len(json.dumps(kv[1], default=str)))
        keep = max(1, int(len(longest) * 0.75))
        dropped += len(longest) - keep
        obj[key] = longest[:keep]
        obj["_truncated"] = f"{dropped} list items omitted to fit — narrow the query or lower the limit"
        text = json.dumps(obj, default=str, ensure_ascii=False)
    return text


def _preview(obj, n: int = 400) -> str:
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


@dataclass
class Agent:
    name: str
    role: str                         # one line, shown to the orchestrator
    system: str
    tools: list[Tool]
    finish_schema: dict
    max_turns: int = field(default_factory=lambda: settings.agent_max_turns)
    parallel_tools: bool = False
    strong_model: bool = False

    # ------------------------------------------------------------------ helpers
    def _tool_specs(self) -> list[dict]:
        finish = {"name": "finish", "description": "Submit your final output. Call exactly once, at the end.",
                  "input_schema": self.finish_schema}
        return [t.spec() for t in self.tools] + [finish]

    def _call_tool(self, ctx, tool: Tool, args: dict):
        kwargs = dict(args)
        if "_agent" in inspect.signature(tool.fn).parameters:
            kwargs["_agent"] = self.name
        return tool.fn(ctx, **kwargs)

    def context_message(self, ctx, task: str) -> str:
        from research_agent.agents.briefing import brief

        return (f"Research question: {ctx.question}\n\n"
                f"Your task from the orchestrator: {task}\n\n"
                f"Current shared state:\n{brief(ctx, for_agent=self.name)}")

    # ------------------------------------------------------------------ main loop
    def run(self, ctx, task: str) -> dict:
        llm = ctx.llm_factory(strong=self.strong_model)
        tools_by_name = {t.name: t for t in self.tools}
        specs = self._tool_specs()
        messages = [{"role": "user", "content": [{"type": "text", "text": self.context_message(ctx, task)}]}]
        ctx.emit(self.name, "start", {"task": task})
        nudges = 0

        for turn in range(self.max_turns):
            force = "finish" if turn == self.max_turns - 1 else None
            resp = llm.chat(self.system, messages, tools=specs, force_tool=force, max_tokens=4096)
            messages.append({"role": "assistant", "content": resp.content or [{"type": "text", "text": "…"}]})
            if resp.text.strip():
                ctx.emit(self.name, "message", {"text": resp.text[:1500]})

            if not resp.tool_calls:
                nudges += 1
                if nudges > 2:
                    return self._save(ctx, {"summary": resp.text, "_note": "agent ended without finish"})
                messages.append({"role": "user", "content": [{"type": "text", "text":
                                 "Continue using your tools, or call `finish` with your output."}]})
                continue

            finish_call = next((c for c in resp.tool_calls if c.name == "finish"), None)
            others = [c for c in resp.tool_calls if c.name != "finish"]
            results = self._execute(ctx, others, tools_by_name)
            if finish_call is not None:
                return self._save(ctx, finish_call.input)
            messages.append({"role": "user", "content": results})

        # Ran out of turns without finishing: force a finish from what it has so far.
        messages.append({"role": "user", "content": [{"type": "text", "text": "Out of turns. Call finish now."}]})
        resp = llm.chat(self.system, messages, tools=specs, force_tool="finish", max_tokens=4096)
        payload = resp.tool_calls[0].input if resp.tool_calls else {"summary": resp.text}
        return self._save(ctx, payload)

    def _execute(self, ctx, calls, tools_by_name) -> list[dict]:
        def one(call):
            ctx.emit(self.name, "tool_call", {"tool": call.name, "input": call.input})
            tool = tools_by_name.get(call.name)
            try:
                if tool is None:
                    raise ValueError(f"unknown tool {call.name}")
                out = self._call_tool(ctx, tool, call.input)
                is_error = isinstance(out, dict) and "error" in out and len(out) == 1
            except Exception as exc:  # errors go back to the model so it can correct itself
                out, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
            ctx.emit(self.name, "tool_result", {"tool": call.name, "error": is_error, "preview": _preview(out)})
            return {"type": "tool_result", "tool_use_id": call.id, "content": _dump(out), "is_error": is_error}

        if self.parallel_tools and len(calls) > 1:
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                return list(pool.map(one, calls))
        return [one(c) for c in calls]

    def _save(self, ctx, payload: dict) -> dict:
        ctx.save_note(self.name, payload)
        ctx.emit(self.name, "finish", {"output": _preview(payload, 1200)})
        return payload
