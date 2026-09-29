"""Generic tool-using agent loop, shared by every specialist agent and the orchestrator.

Each agent = its own system prompt + its own small tool set + a `finish` tool whose schema is that
agent's output contract. The agent loops (LLM -> tool calls -> results) until it calls `finish`;
its output is saved as a note in the shared run state, where later agents can read it.
"""
from __future__ import annotations

import inspect
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from research_agent.config import settings
from research_agent.tools.base import Tool

KEEP_RECENT_MESSAGES = 4     # the latest tool round-trips are never shortened
COMPACT_TO_CHARS = 400


def _msg_chars(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        for b in m["content"]:
            total += len(b.get("text") or b.get("content") or json.dumps(b.get("input") or {}))
    return total


def _label(llm, step: str) -> None:
    """Remember which step a model client belongs to, so a run's usage can be broken down by step."""
    try:
        llm._step = step
    except Exception:
        pass


def compact(messages: list[dict], budget: int) -> list[dict]:
    """Shorten old tool results (oldest first) until the conversation fits the budget.

    Agents keep their durable state in the database (shortlist, claims, extractions), so an old
    search result can be cut to a stub without losing work; the recent turns stay intact."""
    if budget <= 0 or _msg_chars(messages) <= budget:
        return messages
    out = [dict(m, content=[dict(b) for b in m["content"]]) for m in messages]
    for m in out[1:-KEEP_RECENT_MESSAGES]:
        for b in m["content"]:
            if b.get("type") == "tool_result" and len(b["content"]) > COMPACT_TO_CHARS:
                b["content"] = b["content"][:COMPACT_TO_CHARS] + "…[older result shortened to save context]"
        if _msg_chars(out) <= budget:
            break
    return out


def _dump(obj, limit: int | None = None) -> str:
    """Serialise a tool result, shrinking the longest list first so the JSON stays valid."""
    MAX_RESULT_CHARS = limit or settings.tool_result_chars
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


def coerce_to_schema(args: dict, schema: dict) -> dict:
    """Repair common top-level type slips from smaller models (a string where a list is expected,
    "5" where an integer is expected). Values that already match the schema are left untouched, so
    this never changes what a well-behaved model sends."""
    if not isinstance(args, dict):
        return {}
    props = (schema or {}).get("properties", {})
    out = dict(args)
    for key, value in args.items():
        want = (props.get(key) or {}).get("type")
        if want == "array" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                out[key] = parsed if isinstance(parsed, list) else [value]
            except json.JSONDecodeError:
                out[key] = [value]
        elif want == "array" and isinstance(value, dict):
            out[key] = [value]
        elif want == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            out[key] = int(value)
        elif want == "number" and isinstance(value, str):
            try:
                out[key] = float(value)
            except ValueError:
                pass
        elif want == "object" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    out[key] = parsed
            except json.JSONDecodeError:
                pass
    return out


def _preview(obj, n: int = 400) -> str:
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def repair_truncated_json(raw: str) -> dict | None:
    """Close a JSON object that was cut off mid-way, keeping only the values that were complete.

    Walks the text tracking strings and open brackets, remembers every point where a value inside a
    container had just ended, and closes the brackets at the latest such point that parses."""
    if not raw.lstrip().startswith("{"):
        return None
    stack: list[str] = []
    cuts: list[tuple[int, str]] = []
    in_str = esc = False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if stack and stack[-1] == "[":         # a complete string item in a list
                    cuts.append((i + 1, "".join(reversed(stack))))
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if stack:
                cuts.append((i + 1, "".join(reversed(stack))))
            else:
                cuts.append((i + 1, ""))
        elif ch == "," and stack:
            cuts.append((i, "".join(reversed(stack))))
    closer = {"{": "}", "[": "]"}
    # prefer cutting where only the outer object is open, so a half-written item is dropped, not kept
    clean = [c for c in reversed(cuts) if c[1].count("{") <= 1]
    for pos, opened in clean + [c for c in reversed(cuts) if c not in clean]:
        text = raw[:pos].rstrip().rstrip(",")
        text = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", text)   # a key whose value never came
        try:
            val = json.loads(text + "".join(closer[c] for c in opened))
        except json.JSONDecodeError:
            continue
        if isinstance(val, dict):
            return val
    return None


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
    long_output: bool = False         # final answer is a long structured list: allow a bigger output budget

    # ------------------------------------------------------------------ helpers
    def _max_tokens(self) -> int:
        base = settings.report_max_tokens if self.name == "synthesis" else settings.agent_max_tokens
        return max(base, settings.long_output_max_tokens) if self.long_output else base

    def _tool_specs(self) -> list[dict]:
        finish = {"name": "finish", "description": "Submit your final output. Call exactly once, at the end.",
                  "input_schema": self.finish_schema}
        return [t.spec() for t in self.tools] + [finish]

    def _call_tool(self, ctx, tool: Tool, args: dict):
        # Smaller models sometimes invent arguments (e.g. {"": ""} for a tool that takes none).
        # Drop anything the tool does not accept instead of failing, and say so in the result.
        args = coerce_to_schema(args, tool.input_schema)
        params = inspect.signature(tool.fn).parameters
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {k: v for k, v in (args or {}).items()
                  if accepts_any or (k in params and k not in ("ctx", "_agent"))}
        ignored = sorted(set(args or {}) - set(kwargs))
        if "_agent" in params:
            kwargs["_agent"] = self.name
        out = tool.fn(ctx, **kwargs)
        if ignored and isinstance(out, dict):
            out = {**out, "_ignored_arguments": ignored}
        return out

    def context_message(self, ctx, task: str) -> str:
        from research_agent.agents.briefing import brief

        return (f"Research question: {ctx.question}\n\n"
                f"Your task from the orchestrator: {task}\n\n"
                f"Current shared state:\n{brief(ctx, for_agent=self.name)}")

    # ------------------------------------------------------------------ main loop
    def _chat(self, llm, messages, specs, force=None):
        # Trim in one large step and keep the result: providers cache the start of a conversation, and
        # re-trimming a little on every turn would change that start each time and throw the cache away.
        budget = settings.context_budget_chars
        if budget and _msg_chars(messages) > budget:
            messages[:] = compact(messages, int(budget * 0.6))
        return llm.chat(self.system, messages, tools=specs, force_tool=force, max_tokens=self._max_tokens())

    @staticmethod
    def _add_user_text(messages: list[dict], text: str) -> None:
        """Attach a note to the last user message (keeps user/assistant turns alternating)."""
        if messages[-1]["role"] == "user":
            messages[-1] = dict(messages[-1], content=list(messages[-1]["content"]) + [{"type": "text", "text": text}])
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def run(self, ctx, task: str) -> dict:
        llm = ctx.llm_factory(strong=self.strong_model, step=self.name)
        _label(llm, self.name)
        tools_by_name = {t.name: t for t in self.tools}
        specs = self._tool_specs()
        finish_only = [specs[-1]]  # when forcing a finish, offer ONLY the finish tool
        messages = [{"role": "user", "content": [{"type": "text", "text": self.context_message(ctx, task)}]}]
        ctx.emit(self.name, "start", {"task": task})
        nudges = bad_calls = cut_finish = 0
        self._seen: dict[str, str] = {}

        for turn in range(self.max_turns - 1):
            try:
                resp = self._chat(llm, messages, specs)
            except Exception as exc:
                # Some models occasionally emit a malformed or disallowed tool call and the provider
                # rejects the whole response. Tell the model and let it try again (twice at most).
                if "tool_use_failed" in str(exc) or "tool call validation" in str(exc).lower():
                    bad_calls += 1
                    ctx.emit(self.name, "error", {"error": f"invalid tool call rejected by provider: {str(exc)[:300]}"})
                    if bad_calls <= 2:
                        hint = ("It was cut off before the JSON was complete: keep the arguments much shorter."
                                if "parse tool call arguments" in str(exc) else
                                "Check each argument's type against the tool schema (lists must be JSON arrays).")
                        self._add_user_text(messages, "Your last tool call was invalid and was rejected. "
                                            "Use only the listed tools with valid arguments. " + hint)
                        continue
                    break
                raise
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
            if finish_call is not None and "_raw_arguments" in (finish_call.input or {}):
                # Provider returned unparseable (usually truncated) finish JSON. Ask once for a shorter one.
                cut_finish += 1
                ctx.emit(self.name, "error", {"error": "finish arguments were cut off before the JSON was complete"})
                if cut_finish > 1:
                    return self._save(ctx, self._salvage(finish_call.input["_raw_arguments"]))
                results = results + [
                    {"type": "tool_result", "tool_use_id": finish_call.id, "is_error": True,
                     "content": "Your finish call was cut off before its JSON was complete (output token limit, "
                                f"{len(finish_call.input['_raw_arguments'] or '')} characters written). Call finish "
                                "again at about half that length: keep every item, but write each text field in one "
                                "or two sentences and leave optional fields out."}]
                messages.append({"role": "user", "content": results})
                continue
            if finish_call is not None:
                return self._save(ctx, coerce_to_schema(finish_call.input, self.finish_schema))
            left = self.max_turns - 2 - turn
            if left <= 2:
                results = results + [{"type": "text", "text":
                                      f"{left} turn(s) left. Wrap up now and call `finish` next."}]
            messages.append({"role": "user", "content": results})

        # Out of turns: force a finish, offering only the finish tool so the model cannot call anything else.
        self._add_user_text(messages, "Out of turns. Call `finish` now with your output so far.")
        try:
            resp = self._chat(llm, messages, finish_only, force="finish")
            if resp.tool_calls:
                args = resp.tool_calls[0].input or {}
                if "_raw_arguments" in args:
                    return self._save(ctx, self._salvage(args["_raw_arguments"]))
                return self._save(ctx, coerce_to_schema(args, self.finish_schema))
            payload = {"summary": resp.text}
        except Exception as exc:
            ctx.emit(self.name, "error", {"error": f"forced finish failed: {str(exc)[:300]}"})
            payload = {"summary": "Agent ran out of turns before summarising; its work is saved in the shared state."}
        payload["_note"] = "auto-finished"
        return self._save(ctx, payload)

    def _execute(self, ctx, calls, tools_by_name) -> list[dict]:
        def one(call):
            ctx.emit(self.name, "tool_call", {"tool": call.name, "input": call.input})
            tool = tools_by_name.get(call.name)
            key = call.name + json.dumps(call.input, sort_keys=True, default=str)
            seen = getattr(self, "_seen", {})
            try:
                if tool is None:
                    raise ValueError(f"unknown tool {call.name}")
                if tool.read_only and key in seen:
                    out = {"note": "You already made this exact call in this task; its result is above. "
                                   "Try a different query, act on what you found, or finish."}
                else:
                    out = self._call_tool(ctx, tool, call.input)
                    if tool.read_only:
                        seen[key] = "1"
                is_error = isinstance(out, dict) and "error" in out and len(out) == 1
            except Exception as exc:  # errors go back to the model so it can correct itself
                out, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
            ctx.emit(self.name, "tool_result", {"tool": call.name, "error": is_error, "preview": _preview(out)})
            limit = max(tool.max_chars or 0, settings.tool_result_chars) if tool else None
            return {"type": "tool_result", "tool_use_id": call.id, "content": _dump(out, limit), "is_error": is_error}

        if self.parallel_tools and len(calls) > 1:
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                return list(pool.map(one, calls))
        return [one(c) for c in calls]

    def _salvage(self, raw: str) -> dict:
        """Recover what we can from finish arguments that were cut off mid-JSON (output token limit).

        First try to keep every complete item (a cut-off list of designs keeps the designs that were
        finished); fall back to pulling out top-level text fields."""
        note = "finish output was cut off (raise LONG_OUTPUT_MAX_TOKENS / AGENT_MAX_TOKENS)"
        repaired = repair_truncated_json(raw or "")
        out: dict = coerce_to_schema(repaired, self.finish_schema) if repaired else {}
        out["_note"] = note + ("; the last, unfinished item was dropped" if repaired else "")
        for key in self.finish_schema.get("properties", {}):
            if key in out:
                continue
            m = re.search(r'"%s"\s*:\s*"' % re.escape(key), raw or "")
            if not m:
                continue
            rest, end, i = raw[m.end():], None, 0
            while i < len(rest):
                if rest[i] == "\\":
                    i += 2
                    continue
                if rest[i] == '"':
                    end = i
                    break
                i += 1
            chunk = rest[:end] if end is not None else rest.rstrip("\\")
            try:
                out[key] = json.loads('"' + chunk + '"')
            except json.JSONDecodeError:
                out[key] = chunk.replace("\\n", "\n").replace('\\"', '"')
        return out

    def _save(self, ctx, payload: dict) -> dict:
        ctx.save_note(self.name, payload)
        ctx.emit(self.name, "finish", {"output": _preview(payload, 1200)})
        return payload
