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
