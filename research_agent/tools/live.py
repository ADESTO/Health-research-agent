"""The AI decides, code limits: searching PubMed during a run when the loaded corpus looks too thin.

Two places can decide:
  - the discovery agent of an agent-chosen run, through the search_pubmed_live tool, as one more search;
  - in a systematic run, a sufficiency check after screening: the model sees what came out eligible (how many,
    from where, which years and designs, judged from titles) against the question, and answers whether that is
    enough. If not, it writes a PubMed query, and the papers found are screened like the rest.

Every live search is recorded on the run with who asked, why, the exact query and what it brought in, and
PRISMA shows it as its own source. Code enforces the limits whatever the model asks for: PubMed only, at most
LIVE_SEARCH_MAX_CALLS searches and LIVE_SEARCH_MAX_PAPERS papers per run, never past the run's own caps, and
nothing at all with LIVE_SEARCH=0. See ingestion/pubmed.py for what a search loads.
"""
from __future__ import annotations

import json

from research_agent.config import settings
from research_agent.tools.base import INT, STR, Tool, obj

NOTE = "live_search"

SUFFICIENCY_SYSTEM = """You check whether a systematic literature search found enough studies to answer a research
question, before the studies are read. You see the question, the review's eligibility criteria and the titles
and years of the studies that passed screening.

Judge whether the eligible set can answer the question as asked: enough studies, from the places, populations,
periods and kinds of study the question is about. A thin set is a reason to search further; so is a set that
misses a country or population the question names. Searching costs time and money, so do not search when the
set is adequate.

If more studies are needed, write a PubMed query that would find the missing ones: the question's concepts
joined with AND, synonyms with OR, MeSH terms where useful (e.g. "arthritis"[MeSH Terms] AND (Kenya OR
Uganda)). It must stay within the question; it is not a way to change the question."""


def _state(ctx) -> dict:
    return ctx.notes().get(NOTE) or {"calls": 0, "papers": 0, "searches": []}


def room(ctx) -> tuple[int, int]:
    """(searches left, papers left) for this run."""
    s = _state(ctx)
    return (max(0, settings.live_search_max_calls - s["calls"]),
            max(0, settings.live_search_max_papers - s["papers"]))


def run_search(ctx, query: str, why: str, by: str, limit: int | None = None, client=None) -> dict:
    """One live search, within the run's limits, recorded on the run. Returns the result or {"error"}."""
    from research_agent.ingestion import pubmed
    from research_agent.tools.review import log_identified

    if not settings.live_search:
        return {"error": "live search is switched off on this server (LIVE_SEARCH=0)"}
    calls_left, papers_left = room(ctx)
    if calls_left <= 0 or papers_left <= 0:
        return {"error": f"this run has used its live searches ({settings.live_search_max_calls} searches, "
                         f"{settings.live_search_max_papers} papers); work with what is loaded"}
    query = " ".join((query or "").split())[:500]
    if len(query) < 3:
        return {"error": "give a PubMed query"}
    n = max(1, min(int(limit or papers_left), papers_left, 200))
    ctx.emit("live_search", "start", {"task": f"searching PubMed: {query[:160]}"})
    try:
        res = pubmed.live_search(ctx.pg, query, limit=n, client=client, label=query)
    except Exception as exc:
        ctx.emit("live_search", "error", {"error": str(exc)[:300]})
        return {"error": f"PubMed could not be reached: {str(exc)[:200]}"}
    log_identified(ctx, res["paper_ids"], f"live PubMed search: {query}"[:200])
    s = _state(ctx)
    s["calls"] += 1
    s["papers"] += len(res["paper_ids"])
    s["searches"].append({"by": by, "why": " ".join((why or "").split())[:500], "query": query,
                          "matched": res["matched"], "brought_in": len(res["paper_ids"]),
                          "open_access": res["loaded_open_access"], "abstract_only": res["loaded_abstract_only"]})
    ctx.save_note(NOTE, s)
    ctx.emit("live_search", "finish", {"output": {"papers": len(res["paper_ids"])}})
    return res


# ---------------------------------------------------------------- the discovery agent's tool
def _tool(ctx, query: str, why: str, limit: int | None = None) -> dict:
    res = run_search(ctx, query, why, by="discovery agent", limit=limit)
    if "error" in res:
        return res
    rows = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, source, title, year, left(abstract, 240) AS abstract_start FROM papers "
        "WHERE paper_id = ANY(%s)", (res["paper_ids"] or [""],)).fetchall()}
    return {"query": query, "found_in_pubmed": res["matched"],
            "results": [rows[p] for p in res["paper_ids"] if p in rows][:60],
            "note": "These papers are now in the corpus. Add the ones that answer the question with "
                    "add_to_shortlist, as with any search. Papers with source 'pubmed' have an abstract only."}


LIVE_TOOL = Tool(
    "search_pubmed_live",
    "Search PubMed itself when the loaded corpus clearly lacks studies the question needs (few results, a "
    "country, population or period the question names is missing). Fetches the matching papers into the corpus "
    "and returns them; then add the relevant ones to the shortlist. Limited per run, so use it only when the "
    "corpus searches have come up short, and say why.",
    obj({"query": {**STR, "description": 'PubMed syntax, e.g. "arthritis"[MeSH Terms] AND (Kenya OR Uganda)'},
         "why": {**STR, "description": "What the loaded corpus is missing"},
         "limit": {**INT, "description": "Papers to fetch (at most 200)"}}, ["query", "why"]),
    _tool)


# ---------------------------------------------------------------- the systematic run's check
def sufficiency(ctx, protocol: dict, eligible: list[str]) -> dict:
    """Ask the model whether the eligible set is enough. Returns {enough, reason, query}."""
    rows = ctx.pg.execute("SELECT paper_id, title, year, source FROM papers WHERE paper_id = ANY(%s) "
                          "ORDER BY year DESC", (eligible or [""],)).fetchall()
    years = [r["year"] for r in rows if r["year"]]
    listing = "\n".join(f"- {r['title'][:160]} ({r['year']})" for r in rows[:80])
    text = (f"Research question: {ctx.question}\n\n"
            f"Include: {'; '.join(protocol.get('inclusion') or []) or 'not stated'}\n"
            f"Exclude: {'; '.join(protocol.get('exclusion') or []) or 'not stated'}\n"
            + (f"Countries: {', '.join(protocol.get('countries') or [])}\n" if protocol.get("countries") else "")
            + f"Search query used: {protocol.get('search_query') or protocol.get('topic_query') or ''}\n\n"
            f"Eligible after screening: {len(rows)} studies"
            + (f", published {min(years)} to {max(years)}" if years else "") + ".\n"
            + (listing or "(none)") + ("\n(and more)" if len(rows) > 80 else ""))
    tool = {"name": "record_sufficiency", "description": "Record whether the eligible studies are enough.",
            "input_schema": obj({"enough": {"type": "boolean"},
                                 "reason": {**STR, "description": "One or two sentences"},
                                 "query": {**STR, "description": "PubMed query for the missing studies, or ''"}},
                                ["enough", "reason"])}
    llm = ctx.llm_factory(step="live_search")
    setattr(llm, "_step", "live_search")
    resp = llm.chat(SUFFICIENCY_SYSTEM, [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    tools=[tool], force_tool="record_sufficiency", max_tokens=600)
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    return {"enough": bool(args.get("enough", True)), "reason": str(args.get("reason") or "")[:500],
            "query": " ".join(str(args.get("query") or "").split())[:500]}


# ---------------------------------------------------------------- evidence check after reading (every run)
EVIDENCE_SYSTEM = """You check whether the studies collected for a literature review contain the KINDS of evidence its
question needs, before the review is written. A review that concludes "no study has shown X" when the studies
were never searched for X is not a finding; it is a gap in the search.

First decide what evidence would answer the question. For example: effects of an intervention or a tool on
outcomes need randomised, quasi-experimental or before-after evaluations reporting those outcomes; prevalence
needs population-based or facility-based surveys; diagnostic accuracy needs accuracy studies against a
reference standard; management practice needs audits, cohorts or guideline-adherence studies; trends need
studies spread across the period. Then look at the studies you are given (titles, years and stated designs)
and name each kind of evidence the question needs that is missing or thin.

For each, write a PubMed query that would find it: the question's concepts joined with AND, synonyms with OR,
and the design as publication types or design words (e.g. ("artificial intelligence" OR "machine learning")
AND (Kenya OR Uganda OR Africa) AND ("Randomized Controlled Trial"[pt] OR "Clinical Trial"[pt] OR
"randomised" OR "stepped-wedge" OR "before-after")). Stay inside the question. List the most important first.
If the studies already hold what the question needs, say so and list nothing."""

EVIDENCE_TOOL = {"name": "record_evidence_check", "description": "Record which kinds of evidence are missing.",
                 "input_schema": obj({
                     "needed": {"type": "array", "items": {"type": "object", "properties": {
                         "evidence": {**STR, "description": "The kind of evidence, e.g. 'trials of AI tools with patient outcomes'"},
                         "held": {**STR, "description": "What the collected studies hold of it"},
                         "missing": {"type": "boolean"},
                         "query": {**STR, "description": "PubMed query that would find it, or '' when not missing"}},
                         "required": ["evidence", "missing"]}},
                     "reason": STR}, ["needed"])}

EVIDENCE_NOTE = "evidence_check"
EVIDENCE_ADD_MAX = 25      # papers one search may add to the run, whatever the shortlist cap


def _designs(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        for d in {x.lower() for x in (r["data"].get("study_designs") or [])}:
            out[d] = out.get(d, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:25])


def evidence_check(ctx) -> dict:
    """After the papers are read: does the run hold the kinds of evidence its question needs? For each kind that
    is missing, search PubMed for it (within the run's live-search limits), screen what comes back against the
    protocol when there is one, add the relevant papers and read them. Returns what was done (also saved as a
    note, and shown in the report). Never raises."""
    from research_agent.tools.extraction import _rows, extract_papers, protocol_of

    done = ctx.notes().get(EVIDENCE_NOTE)
    if done:
        return done
    out = {"checked": False, "needed": [], "searches": [], "added": []}
    if not settings.live_search:
        out["note"] = "live search is switched off on this server"
        ctx.save_note(EVIDENCE_NOTE, out)
        return out
    try:
        rows = _rows(ctx)
        protocol = protocol_of(ctx) or {}
        listing = "\n".join(f"- {r['title'][:150]} ({r['year']}; {', '.join((r['data'].get('study_designs') or [])[:2]) or 'design not stated'})"
                            for r in sorted(rows, key=lambda r: -(r['year'] or 0))[:120])
        text = (f"Research question: {ctx.question}\n\n"
                + (f"Include: {'; '.join(protocol.get('inclusion') or [])}\n" if protocol.get("inclusion") else "")
                + (f"Countries: {', '.join(protocol.get('countries') or [])}\n" if protocol.get("countries") else "")
                + f"\nStudies collected: {len(rows)}. Stated designs (studies per design): "
                + json.dumps(_designs(rows)) + f"\n\n{listing}")
        ctx.emit("evidence_check", "start", {"task": "does the run hold the kinds of evidence the question needs?"})
        llm = ctx.llm_factory(step="live_search")
        setattr(llm, "_step", "live_search")
        resp = llm.chat(EVIDENCE_SYSTEM, [{"role": "user", "content": [{"type": "text", "text": text}]}],
                        tools=[EVIDENCE_TOOL], force_tool="record_evidence_check", max_tokens=1500)
        args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
        out["checked"] = True
        out["reason"] = " ".join(str(args.get("reason") or "").split())[:500]
        out["needed"] = [{"evidence": " ".join(str(n.get("evidence") or "").split())[:200],
                          "held": " ".join(str(n.get("held") or "").split())[:300],
                          "missing": bool(n.get("missing")), "query": " ".join(str(n.get("query") or "").split())[:500]}
                         for n in args.get("needed") or [] if isinstance(n, dict) and n.get("evidence")][:6]
        new_ids: list[str] = []
        have = set(ctx.shortlist_ids())
        for n in [x for x in out["needed"] if x["missing"] and x["query"]]:
            if room(ctx)[0] <= 0:
                n["searched"] = "not searched: the run's live searches were used up"
                continue
            res = run_search(ctx, n["query"], f"the studies lack {n['evidence']}", by="evidence check after reading",
                             limit=60)
            if "error" in res:
                n["searched"] = res["error"]
                continue
            found = [p for p in res["paper_ids"] if p not in have and p not in new_ids]
            keep = found
            if protocol.get("inclusion") or protocol.get("exclusion"):
                from research_agent.tools.review import log_decision
                from research_agent.tools.systematic import screen

                decisions = screen(ctx, found, protocol)
                keep = [p for p in found if (decisions.get(p) or {}).get("decision") in ("include", "unclear")]
                gone = [p for p in found if p not in keep]
                if gone:
                    log_decision(ctx, gone, "excluded", "evidence check: not eligible at screening")
            keep = keep[:EVIDENCE_ADD_MAX]
            n["searched"] = f"PubMed matched {res['matched']}; {len(found)} new to the run; {len(keep)} relevant added"
            n["added"] = keep
            new_ids += keep
        if new_ids:
            from research_agent.tools.review import log_decision

            for pid in new_ids:      # past the shortlist cap on purpose: this is the evidence the question needs
                ctx.pg.execute("INSERT INTO run_papers (run_id, paper_id, added_by, reason, score) VALUES "
                               "(%s,%s,'evidence_check',%s,%s) ON CONFLICT (run_id, paper_id) DO NOTHING",
                               (ctx.run_id, pid, "found by the evidence check: a kind of evidence the question needs", 0.9))
            log_decision(ctx, new_ids, "included", "evidence check: the kind of evidence the question needs")
            extract_papers(ctx, new_ids, depth="abstract")
            open_access = [p for p in new_ids if p.startswith("PMC")]
            if open_access:
                extract_papers(ctx, open_access, depth="fulltext", limit=len(open_access))
        out["added"] = new_ids
        ctx.emit("evidence_check", "finish", {"output": {"missing": sum(1 for n in out["needed"] if n["missing"]),
                                                         "papers_added": len(new_ids)}})
    except Exception as exc:          # the check improves a run; it must never cost one
        out["error"] = str(exc)[:300]
        ctx.emit("evidence_check", "error", {"error": str(exc)[:300]})
    ctx.save_note(EVIDENCE_NOTE, out)
    return out


def evidence_markdown(ctx) -> list[str]:
    e = ctx.notes().get(EVIDENCE_NOTE) or {}
    if not e.get("checked"):
        return []
    L = ["## Was the right kind of evidence looked for? (computed)", "",
         "After the papers were read, the AI listed the kinds of evidence the question needs and checked whether "
         "the collected studies hold them. Each kind that was missing was searched for in PubMed, and relevant "
         "papers were added and read. A conclusion that evidence is absent rests on this search, not only on "
         "the papers first collected.", ""]
    for n in e.get("needed") or []:
        state = "missing from the studies first collected" if n["missing"] else "present"
        line = f"- **{n['evidence']}**: {state}."
        if n.get("held"):
            line += f" {n['held']}"
        if n.get("query"):
            line += f" Searched: `{n['query']}`."
        if n.get("searched"):
            line += f" {n['searched']}."
        L.append(line)
    if e.get("reason"):
        L += ["", e["reason"]]
    return L + [""]


def markdown(ctx) -> list[str]:
    s = ctx.notes().get(NOTE) or {}
    checks = s.get("checks") or []
    if not s.get("searches") and not checks:
        return []
    L = ["## Live PubMed searches (computed)", "",
         "The loaded corpus was searched first. These searches of PubMed were made during the run because the AI "
         "judged the studies found so far insufficient; code limited them to PubMed and to "
         f"{settings.live_search_max_papers} papers. Papers open access in PMC were read with their full text; "
         "the others from their PubMed abstract.", ""]
    for c in checks:
        if c.get("enough"):
            L.append(f"- Check after screening: the eligible studies were judged sufficient. {c.get('reason', '')}")
    for x in s.get("searches") or []:
        L.append(f"- Searched by the {x['by']}: `{x['query']}`. Why: {x['why'] or 'not stated'} "
                 f"PubMed matched {x['matched']}; {x['brought_in']} brought into the run "
                 f"({x['open_access']} new open-access, {x['abstract_only']} new abstract-only).")
    return L + [""]
