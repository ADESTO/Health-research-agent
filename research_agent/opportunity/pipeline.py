"""Research Opportunity Map pipeline.

    protocol -> discovery -> literature -> protocol extraction (code) -> map computation (code)
             -> gap reasoning (LLM, hypotheses tested by code) -> research designs (LLM, references
             checked by code) -> rendered map

Resumable like `ask`: finished steps are skipped when a run is resumed.
"""
from __future__ import annotations

import traceback

from research_agent.agents.orchestrator import _already_done, run_specialist
from research_agent.opportunity.agents import DESIGN, GAP_REASONING, check_designs, check_gap_reasoning
from research_agent.opportunity.compute import compute_map
from research_agent.opportunity.protocol import ensure_protocol, extract_protocol_fields, probe_concepts
from research_agent.opportunity.render import references, render_map
from research_agent.runstate import RunContext


def _step_done(ctx, name: str) -> bool:
    note = ctx.notes().get(name)
    return bool(note) and not note.get("_note")


def run_map(question: str | None = None, provider: str | None = None, llm_factory=None, on_event=None,
            run_id: str | None = None, resume: str | None = None) -> dict:
    from research_agent.db import init_schema

    init_schema()  # older databases lack the map's tables; this only adds what is missing
    if resume:
        ctx = RunContext.resume(resume, provider=provider, llm_factory=llm_factory, on_event=on_event)
    else:
        ctx = RunContext.create(question, provider=provider, llm_factory=llm_factory, on_event=on_event,
                                run_id=run_id)
    question = ctx.question
    ctx.save_note("mode", {"mode": "map"})       # lets the web page label the run before the map exists
    if on_event:
        on_event("system", "run", {"run_id": ctx.run_id, "resumed": bool(resume)})
    dropped: list[str] = []
    try:
        # 1. protocol ---------------------------------------------------------------------------
        protocol = ensure_protocol(ctx, question)
        if not protocol["fields"]:
            raise RuntimeError("the protocol defined no usable question-specific fields")
        scope = ""
        if protocol.get("inclusion"):
            scope += " Include papers that: " + "; ".join(protocol["inclusion"]) + "."
        if protocol.get("exclusion"):
            scope += " Exclude papers that: " + "; ".join(protocol["exclusion"]) + "."

        # 2-3. discovery and literature (the same agents as `ask`) -------------------------------
        probes = probe_concepts(protocol)
        probe_note = (f" Run coverage_probe with topic '{protocol.get('topic_query')}' and these concepts: "
                      f"{probes}." if probes and protocol.get("topic_query") else "")
        for name, task in (("discovery", f"Build the shortlist for: {question}.{scope}{probe_note}"),
                           ("literature", f"Extract the shortlisted papers for: {question}. Read the most "
                                          "central ones in full.")):
            if resume and _already_done(ctx, name):
                ctx.emit(name, "skipped", {"reason": "finished in the earlier attempt"})
                continue
            res = run_specialist(ctx, name, task)
            if "error" in res:
                raise RuntimeError(res["error"])
            if name == "discovery" and not ctx.shortlist_ids():
                raise RuntimeError("discovery found no relevant papers; the map needs a shortlist")

        # 4. question-specific fields for every paper (code + extraction calls) ------------------
        # (usually already done right after the literature step; this catches papers added since)
        earlier = ctx.notes().get("protocol_extraction")
        stats = extract_protocol_fields(ctx, protocol)
        if stats.get("newly_extracted") or not earlier:
            ctx.save_note("protocol_extraction", stats)
            ctx.emit("protocol_extraction", "finish", {"output": stats})

        # 5. the map itself (code) ---------------------------------------------------------------
        m = compute_map(ctx, protocol)
        ctx.save_note("map", {"map": m})
        ctx.emit("map", "finish", {"output": {k: len(m[k]) for k in ("established", "emerging", "gaps",
                                                                       "novelty")}})

        # 6. gap reasoning (LLM proposes, code tests) ---------------------------------------------
        if not (resume and _step_done(ctx, "gap_reasoning")):
            raw = GAP_REASONING.run(ctx, "Explain the gaps on the map, testing every explanation.")
            checked, gone = check_gap_reasoning(ctx, raw)
            dropped += gone
            ctx.save_note("gap_reasoning", {"checked": checked, "dropped": gone})
        reasoning = (ctx.notes().get("gap_reasoning") or {}).get("checked") or {}

        # 7. candidate designs (LLM proposes, code checks references) -----------------------------
        if not (resume and _step_done(ctx, "design")):
            raw = DESIGN.run(ctx, "Propose candidate research designs from the map.")
            designs, gone = check_designs(ctx, raw)
            dropped += gone
            ctx.save_note("design", {"designs": designs, "dropped": gone})
        designs = (ctx.notes().get("design") or {}).get("designs") or []

        # 8. render ------------------------------------------------------------------------------
        md = render_map(question, protocol, m, reasoning, designs, {"dropped": dropped})
        from research_agent.tools.results import results_markdown

        from research_agent.tools.review import prisma_markdown

        from research_agent.tools.citations import citation_markdown

        from research_agent.tools.burden import burden_markdown

        from research_agent.tools import opportunities

        try:
            opportunities.record(ctx)
            opp = opportunities.markdown(ctx)
        except Exception:
            opp = []
        extra = _computed_sections(ctx) + opp
        extra += results_markdown(ctx) + citation_markdown(ctx) + burden_markdown(ctx) + prisma_markdown(ctx)
        if extra:
            md += "\n" + "\n".join(extra)
        cited = []
        for sec in ("established", "emerging", "gaps"):
            for it in m[sec]:
                cited += it["paper_ids"][:4]
        for d in designs:
            cited += d["builds_on"] + [p["paper_id"] for p in d["supporting"] + d["challenging"]]
        for g in reasoning.get("gaps", []):
            cited += [nm["paper_id"] for nm in g["near_misses"]]
        cited = list(dict.fromkeys(cited))
        md += f"\n## References ({len(cited)} cited of {m['N']} analysed)\n\n" + "\n".join(references(ctx.pg, cited))
        ctx.finish(md)
        return {"run_id": ctx.run_id, "report": md, "map": m, "usage": ctx.token_usage(),
                "usage_by_step": ctx.usage_by_step()}
    except Exception as exc:
        ctx.emit("system", "error", {"error": str(exc), "trace": traceback.format_exc()[-2000:]})
        ctx.finish(None, str(exc))
        exc.run_id = ctx.run_id
        raise
    finally:
        ctx.close()


def _computed_sections(ctx) -> list[str]:
    """What every map gains from the run-level checks: the scope its numbers count over, whether each
    opportunity has been done already, and whether the papers back each other up. Each is optional, so one
    that fails costs its own section and never the map."""
    out: list[str] = []
    try:
        from research_agent.tools import cohort

        out += cohort.markdown(ctx)
    except Exception as exc:
        ctx.emit("cohort", "error", {"error": str(exc)[:300]})
    try:
        from research_agent.tools import precedent

        precedent.check(ctx)
        out += precedent.markdown(ctx)
    except Exception as exc:
        ctx.emit("precedent", "error", {"error": str(exc)[:300]})
    try:
        from research_agent.tools import corroboration

        corroboration.corroborate(ctx)
        out += corroboration.markdown(ctx)
    except Exception as exc:
        ctx.emit("corroboration", "error", {"error": str(exc)[:300]})
    return out
