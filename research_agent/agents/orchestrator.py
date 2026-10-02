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
            # Then, before a word is written, send the claims whose state rests on reading depth back to
            # the papers. Doing it here rather than after means the report is written from the better
            # numbers instead of quoting thin ones and being corrected later.
            if settings.resolve_evidence:
                try:
                    from research_agent.tools.resolve import resolve

                    resolve(child)
                except Exception as exc:   # a failed attempt must never cost the run its report
                    child.emit("evidence_resolution", "error", {"error": str(exc)[:300]})
        if agent_name == "discovery":
            task += _probe_hint(child)
        out = agent.run(child, task)
        if agent_name == "literature" and "error" not in out:
            _extract_question_fields(child)
        if agent_name == "discovery" and settings.citations:
            from research_agent.tools.citations import ensure

            problem = ensure(child)   # cache citation links among the shortlisted papers (a few requests)
            if problem:
                child.emit("citations", "message", {"text": problem})
        if agent_name == "discovery" and not child.shortlist_ids():
            # Common model failure: it searches, sees results, then finishes without shortlisting.
            # Give it one explicit second chance before concluding the corpus has nothing.
            child.emit("discovery", "message", {"text": "Shortlist empty after discovery; retrying once."})
            out = agent.run(child, task + "\n\nYour previous attempt finished with an EMPTY shortlist. Search "
                             "again without year or category filters, and call add_to_shortlist for every "
                             "relevant paper. Only finish empty if nothing in the corpus is relevant.")
        if agent_name == "synthesis":
            body = out.get("report_markdown") or out.get("summary") or ""
            if not body.strip():
                body = ("## Summary\n\n_The synthesis agent returned no report text (usually its output was cut "
                        "off by the token limit). Raise REPORT_MAX_TOKENS and rerun with `--redo synthesis`. "
                        "The code-checked sections below are still valid._")
            number_check = None
            if settings.number_check:
                try:
                    from research_agent.agents.number_check import resolve_numbers

                    body, number_check = resolve_numbers(child, body)
                except Exception as exc:   # the check improves a report; it must never lose one
                    child.emit("number_check", "error", {"error": str(exc)[:300]})
            report, audit = finalize_report(child, body, number_check)
            child.pg.execute("UPDATE runs SET report_md=%s WHERE run_id=%s", (report, ctx.run_id))
            child.emit("synthesis", "report", audit)
            return {"report_written": True, "chars": len(report), "citation_audit": audit}
        return out
    except Exception as exc:
        child.emit(agent_name, "error", {"error": str(exc), "trace": traceback.format_exc()[-2000:]})
        return {"error": f"{agent_name} failed: {exc}"}
    finally:
        child.close()


def _probe_hint(ctx: RunContext) -> str:
    """Point discovery at the protocol's concepts, so coverage_probe checks what the question cares about."""
    from research_agent.opportunity.protocol import probe_concepts
    from research_agent.tools.extraction import protocol_of

    protocol = protocol_of(ctx)
    probes = probe_concepts(protocol)
    if not probes or not protocol.get("topic_query"):
        return ""
    return (f"\n\nBefore finishing, run coverage_probe with topic '{protocol['topic_query']}' and these "
            f"concepts: {probes}. Add the relevant papers it shows are missing.")


def _extract_question_fields(ctx: RunContext) -> None:
    """After the literature agent, extract the protocol's question-specific fields for every paper."""
    from research_agent.opportunity.protocol import extract_protocol_fields
    from research_agent.tools.extraction import protocol_of

    protocol = protocol_of(ctx)
    if not protocol or not protocol.get("fields"):
        return
    ctx.emit("protocol_extraction", "start", {"task": "extract question-specific fields"})
    stats = extract_protocol_fields(ctx, protocol)
    ctx.save_note("protocol_extraction", stats)
    ctx.emit("protocol_extraction", "finish", {"output": stats})


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


def _no_evidence_report(ctx: RunContext) -> str:
    note = ctx.notes().get("discovery", {})
    n_corpus = ctx.pg.execute("SELECT count(*) n FROM papers").fetchone()["n"]
    searches = note.get("search_log") or []
    lines = [f"# No relevant papers found", "",
             f"**Question:** {ctx.question}", "",
             f"The Discovery agent searched a corpus of {n_corpus:,} health-research papers and did not find "
             "papers that address this question, so no analysis was run and no claims were made.", ""]
    if note.get("coverage_notes"):
        lines += ["**Discovery notes:** " + str(note["coverage_notes"]), ""]
    if searches:
        lines += ["**Searches tried:**", ""] + [f"- {s}" for s in searches[:15]] + [""]
    lines += ["Possible reasons: the corpus is only partly loaded, the topic is rare on arXiv, or the search "
              "terms were too narrow. Try rephrasing, or load more of the corpus with `ingest`."]
    return "\n".join(lines)


def _ensure_report(ctx: RunContext) -> str | None:
    row = ctx.pg.execute("SELECT report_md FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()
    if row["report_md"]:
        return row["report_md"]
    if not ctx.shortlist_ids():
        return _no_evidence_report(ctx)
    res = run_specialist(ctx, "synthesis", "Write the final report.")
    if "error" in res:
        return None
    return ctx.pg.execute("SELECT report_md FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()["report_md"]


def _already_done(ctx: RunContext, name: str) -> bool:
    """On resume, skip agents that finished properly last time (auto-finished ones run again)."""
    note = ctx.notes().get(name)
    if not note or note.get("_note"):
        return False
    return name != "discovery" or bool(ctx.shortlist_ids())


def run_research(question: str | None = None, mode: str = "orchestrated", provider: str | None = None,
                 llm_factory=None, on_event=None, run_id: str | None = None,
                 resume: str | None = None) -> dict:
    from research_agent.db import init_schema

    init_schema()   # older databases lack newer tables and indexes; this only adds what is missing
    if resume:
        ctx = RunContext.resume(resume, provider=provider, llm_factory=llm_factory, on_event=on_event)
    else:
        ctx = RunContext.create(question, provider=provider, llm_factory=llm_factory, on_event=on_event,
                                run_id=run_id)
    question = ctx.question
    if on_event:
        on_event("system", "run", {"run_id": ctx.run_id, "resumed": bool(resume)})
    try:
        ctx.save_note("mode", {"mode": mode})
        if settings.ask_protocol:
            # Question-specific fields: the general extraction form cannot anticipate every concept a
            # question turns on (vector data, attention models...). A failed protocol never stops the run.
            try:
                from research_agent.opportunity.protocol import ensure_protocol

                ensure_protocol(ctx, question)
            except Exception as exc:
                ctx.emit("protocol", "error", {"error": str(exc)[:300]})
        if mode == "pipeline":
            for name in ORDER:
                if name != "discovery" and not ctx.shortlist_ids():
                    break  # nothing relevant found: _ensure_report explains that instead of failing
                if resume and _already_done(ctx, name):
                    ctx.emit(name, "skipped", {"reason": "finished in the earlier attempt"})
                    continue
                res = run_specialist(ctx, name, f"Do your part for this question: {question}")
                if "error" in res and (name in ("discovery", "literature") or "DAILY token limit" in res["error"]):
                    raise RuntimeError(res["error"])
        else:
            task = "Plan and run the research, then have synthesis write the report."
            if resume:
                task += (" This run is being RESUMED: check get_status and only delegate the work that is "
                         "still missing.")
            ORCHESTRATOR.run(ctx, task)
        report = _ensure_report(ctx)
        ctx.finish(report, None if report else "no report produced")
        return {"run_id": ctx.run_id, "report": report, "usage": ctx.token_usage(),
                "usage_by_step": ctx.usage_by_step()}
    except Exception as exc:
        ctx.emit("system", "error", {"error": str(exc), "trace": traceback.format_exc()[-2000:]})
        ctx.finish(None, str(exc))
        exc.run_id = ctx.run_id  # so the CLI can say how to resume
        raise
    finally:
        ctx.close()
