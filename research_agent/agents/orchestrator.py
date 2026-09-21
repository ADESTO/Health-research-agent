"""Research Manager / Orchestrator.

An LLM agent whose tools are the other agents. It decides who works next, with what instructions,
and can run independent agents in parallel (e.g. Methods and Trends) by calling both in one turn.
Guardrails in code keep it honest: agents that need a shortlist or extractions refuse to run
without them, and the report can only be written after claims have been verified.

`mode="pipeline"` runs the same agents in a fixed order instead of an LLM planner — useful as a
baseline and fallback.
"""
from __future__ import annotations

import traceback

from research_agent.agents.base import Agent
from research_agent.agents.report import finalize_report
from research_agent.agents.specialists import SPECIALISTS
from research_agent.config import settings
from research_agent.runstate import RunContext
from research_agent.tools.base import STR, Tool, obj
from research_agent.tools.claims import verify_claims

ORDER = ["discovery", "literature", "methods", "trends", "gaps", "evidence", "synthesis"]


def _precondition(ctx: RunContext, agent: str) -> str | None:
    n_short = len(ctx.shortlist_ids())
    if agent != "discovery" and agent != "trends" and n_short == 0:
        return "The shortlist is empty. Run discovery first."
    if agent in ("methods", "gaps", "evidence", "synthesis"):
        n_ex = ctx.pg.execute(
            """SELECT count(*) n FROM run_papers rp JOIN extractions e
               ON e.paper_id = rp.paper_id AND e.schema_version = %s WHERE rp.run_id = %s""",
            (ctx.extraction_version, ctx.run_id)).fetchone()["n"]
        if n_ex == 0:
            return "No extractions yet. Run literature first."
    return None


def run_specialist(ctx: RunContext, agent_name: str, task: str) -> dict:
    """Run one specialist on its own DB connection (so several can run in parallel)."""
    agent = SPECIALISTS[agent_name]
    err = _precondition(ctx, agent_name)
    if err:
        return {"error": err}
    child = ctx.child()
    try:
        if agent_name == "synthesis":
            # Report may only be written from verified claims: verify anything still pending.
            verify_claims(child)
        out = agent.run(child, task)
        if agent_name == "synthesis":
            body = out.get("report_markdown") or out.get("summary") or ""
            report, audit = finalize_report(child, body)
            child.pg.execute("UPDATE runs SET report_md=%s WHERE run_id=%s", (report, ctx.run_id))
            child.emit("synthesis", "report", audit)
            return {"report_written": True, "chars": len(report), "citation_audit": audit}
        return out
    except Exception as exc:
        child.emit(agent_name, "error", {"error": str(exc), "trace": traceback.format_exc()[-2000:]})
        return {"error": f"{agent_name} failed: {exc}"}
    finally:
        child.close()


def _delegate_tool(name: str) -> Tool:
    agent = SPECIALISTS[name]
    return Tool(
        f"call_{name}",
        f"{agent.role} Give it specific instructions in `task`.",
        obj({"task": STR}, ["task"]),
        lambda ctx, task, _n=name: run_specialist(ctx, _n, task),
    )


def _status(ctx: RunContext) -> dict:
    from research_agent.agents.briefing import brief

    report = ctx.pg.execute("SELECT report_md IS NOT NULL AS has_report FROM runs WHERE run_id=%s",
                            (ctx.run_id,)).fetchone()
    return {"state": brief(ctx, for_agent="orchestrator", note_chars=800), "report_written": report["has_report"]}


ORCHESTRATOR = Agent(
    name="orchestrator",
    role="Plans the research and delegates to specialist agents.",
    system=f"""You are the Research Manager coordinating a team of specialist agents to answer a health research
question from an arXiv corpus. You do not analyse papers yourself; you delegate and check quality.

Team: {"; ".join(f"{n}: {a.role}" for n, a in SPECIALISTS.items())}

Typical plan: discovery -> literature -> methods and trends (call both in the SAME turn so they run in
parallel) -> gaps -> evidence -> synthesis. Adapt it:
- If discovery returns fewer than ~15 papers, send it back once with concrete ideas to broaden.
- If literature reports key fields rarely stated, ask it to read more papers in full text.
- If evidence rejects or fails many claims, you may ask methods/gaps to propose better-grounded claims,
  then run evidence again.
Give each agent a specific task tailored to the question. Use get_status to check progress.
When synthesis has written the report, call finish.""",
    tools=[_delegate_tool(n) for n in ORDER] + [Tool("get_status", "Shared state summary.", obj({}), _status)],
    finish_schema=obj({"summary": STR}, ["summary"]),
    max_turns=settings.orchestrator_max_turns,
    parallel_tools=True,
    strong_model=True,
)


def _ensure_report(ctx: RunContext) -> str | None:
    row = ctx.pg.execute("SELECT report_md FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()
    if row["report_md"]:
        return row["report_md"]
    res = run_specialist(ctx, "synthesis", "Write the final report.")
    if "error" in res:
        return None
    return ctx.pg.execute("SELECT report_md FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()["report_md"]


def run_research(question: str, mode: str = "orchestrated", provider: str | None = None,
                 llm_factory=None, on_event=None, run_id: str | None = None) -> dict:
    ctx = RunContext.create(question, provider=provider, llm_factory=llm_factory, on_event=on_event,
                            run_id=run_id)
    try:
        if mode == "pipeline":
            for name in ORDER:
                res = run_specialist(ctx, name, f"Do your part for this question: {question}")
                if "error" in res and name in ("discovery", "literature"):
                    raise RuntimeError(res["error"])
        else:
            ORCHESTRATOR.run(ctx, "Plan and run the research, then have synthesis write the report.")
        report = _ensure_report(ctx)
        ctx.finish(report, None if report else "no report produced")
        return {"run_id": ctx.run_id, "report": report, "usage": ctx.token_usage()}
    except Exception as exc:
        ctx.emit("system", "error", {"error": str(exc), "trace": traceback.format_exc()[-2000:]})
        ctx.finish(None, str(exc))
        raise
    finally:
        ctx.close()
