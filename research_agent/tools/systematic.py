"""A systematic search: every paper a fixed query finds, screened against the protocol, all of them analysed.

In the default (purposive) mode an agent searches, reads results, and chooses up to MAX_SHORTLIST papers.
Every number downstream is then a share of what the agent picked, and the gaps it reports can be overturned
by papers it never added. That is fine for exploring a question and wrong for stating anything about a body
of literature.

Systematic mode replaces that choice with a procedure anyone can repeat:

  1. identify   the protocol's search_query, written before any result was seen, is run over the whole
                corpus, and every paper it matches is logged as identified with the query that found it.
  2. screen     each identified paper's title and abstract is judged against the protocol's inclusion and
                exclusion criteria, twenty to a call, and every decision is logged with its reason. When in
                doubt a paper is kept for the full read: title-and-abstract screening excludes only what is
                clearly out.
  3. include    every eligible paper is analysed. When more are eligible than SYSTEMATIC_MAX_ANALYSED, a
                RANDOM sample is analysed, drawn with a recorded seed, so the shares it produces estimate the
                eligible set without bias; the top of a relevance ranking would not. The same applies when the
                query identifies more papers than SYSTEMATIC_MAX_SCREENED can be screened.
  4. recall     a fixed query misses synonyms an agent would have caught. The question's nearest papers by
                meaning are checked against what the query found, and those it missed are listed in the
                report. They are not added: adding them would make the set depend on judgement again.

The run records all of it (query, counts, seed, misses) in its `search` note, and the screening log and
PRISMA counts that already existed now describe a real search.

The ceiling is the corpus. A systematic search of what is loaded (open-access PMC topic slices and arXiv) is
systematic within that, and the report says "of the corpus", never "of the literature".
"""
from __future__ import annotations

import hashlib
import random

from research_agent.config import settings

SCREEN_BATCH = 20
RECALL_PROBE = 40          # nearest papers by meaning checked against the query's yield

SCREEN_SYSTEM = """You screen papers for a systematic review, from their title and abstract only.

For each paper decide include, exclude or unclear against the criteria below. Exclude only when the title or
abstract shows the paper clearly fails a criterion; name that criterion. When the abstract does not say
enough to decide, answer unclear: the paper goes on to be read in full, which is the safe error at this
stage. Judge each paper on its own text; never carry a fact from one paper to another.

Research question: {question}

Include papers that:
{inclusion}

Exclude papers that:
{exclusion}"""


def mode_of(ctx) -> str:
    return ((ctx.notes().get("search") or {}).get("mode")) or "purposive"


def is_systematic(ctx) -> bool:
    return mode_of(ctx) == "systematic"


def set_mode(ctx, mode: str) -> None:
    note = ctx.notes().get("search") or {}
    if note.get("mode") != mode:
        ctx.save_note("search", {**note, "mode": mode})


def _seed(ctx) -> int:
    return int(hashlib.sha256(ctx.run_id.encode()).hexdigest(), 16) % 1_000_000


def identify(ctx, query: str) -> list[str]:
    from research_agent.tools.query import tsquery_sql

    sql, params = tsquery_sql(query, prefix="q")
    return [r["paper_id"] for r in ctx.pg.execute(
        f"SELECT paper_id FROM papers WHERE tsv @@ ({sql}) ORDER BY paper_id", params).fetchall()]


def _screen_batch(llm, ctx, papers: list[dict], protocol: dict) -> dict[str, dict]:
    ids = [p["paper_id"] for p in papers]
    tool = {"name": "record_screening", "description": "Record one screening decision per paper.",
            "input_schema": {"type": "object", "properties": {"records": {"type": "array", "items": {
                "type": "object", "properties": {
                    "paper_id": {"type": "string", "enum": ids},
                    "decision": {"type": "string", "enum": ["include", "exclude", "unclear"]},
                    "criterion": {"type": "string", "description": "The criterion the decision turns on"},
                    "reason": {"type": "string", "description": "One short sentence"}},
                "required": ["paper_id", "decision", "reason"]}}}, "required": ["records"]}}
    system = SCREEN_SYSTEM.format(
        question=ctx.question,
        inclusion="\n".join(f"- {x}" for x in protocol.get("inclusion") or ["(none stated)"]),
        exclusion="\n".join(f"- {x}" for x in protocol.get("exclusion") or ["(none stated)"]))
    text = "\n\n".join(f"=== Paper {p['paper_id']} ===\nTitle: {p['title']}\nAbstract: {(p['abstract'] or '')[:2500]}"
                       for p in papers)
    resp = llm.chat(system, [{"role": "user", "content": [{"type": "text", "text": text}]}], tools=[tool],
                    force_tool="record_screening", max_tokens=min(settings.long_output_max_tokens, 120 * len(papers) + 400))
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    if "_raw_arguments" in args:
        from research_agent.agents.base import repair_truncated_json

        args = repair_truncated_json(args["_raw_arguments"] or "") or {}
    out = {}
    for rec in args.get("records") or []:
        if isinstance(rec, dict) and rec.get("paper_id") in ids and rec.get("decision") in ("include", "exclude", "unclear"):
            out[rec["paper_id"]] = rec
    return out


def screen(ctx, ids: list[str], protocol: dict) -> dict[str, dict]:
    """{paper_id: {decision, reason, criterion}} for every paper that came back from screening."""
    from concurrent.futures import ThreadPoolExecutor

    papers = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, title, abstract FROM papers WHERE paper_id = ANY(%s)", (ids,)).fetchall()}
    ordered = [papers[p] for p in ids if p in papers]
    batches = [ordered[i:i + SCREEN_BATCH] for i in range(0, len(ordered), SCREEN_BATCH)]
    llm = ctx.llm_factory(step="screening")
    setattr(llm, "_step", "screening")
    decisions: dict[str, dict] = {}

    def one(batch):
        try:
            return _screen_batch(llm, ctx, batch, protocol)
        except Exception:
            return {}
    with ThreadPoolExecutor(max_workers=max(1, settings.extraction_workers)) as pool:
        for got in pool.map(one, batches):
            decisions.update(got)
    # a paper a batch dropped gets one more chance on its own, so a cut-off reply does not exclude anyone
    missing = [p for p in ordered if p["paper_id"] not in decisions]
    for p in missing:
        try:
            decisions.update(_screen_batch(llm, ctx, [p], protocol))
        except Exception:
            pass
    return decisions


def systematic_discovery(ctx, protocol: dict, max_analysed: int | None = None,
                         max_screened: int | None = None) -> dict:
    """Build the shortlist by systematic search. Returns, and stores as the `search` note, what was done."""
    from research_agent.tools.review import log_decision, log_identified

    max_analysed = int(max_analysed if max_analysed is not None else settings.systematic_max_analysed)
    max_screened = int(max_screened if max_screened is not None else settings.systematic_max_screened)
    query = (protocol.get("search_query") or "").strip()
    fallback = not query
    if fallback:
        query = (protocol.get("topic_query") or ctx.question).strip()
    seed = _seed(ctx)
    rng = random.Random(seed)

    found = identify(ctx, query)
    log_identified(ctx, found, f"systematic: {query}")
    to_screen = found
    if len(found) > max_screened:
        to_screen = sorted(rng.sample(found, max_screened))
        rest = sorted(set(found) - set(to_screen))
        log_decision(ctx, rest, "over_limit", f"identified; not in the random sample of {max_screened} screened "
                                              f"(seed {seed})")
    ctx.emit("discovery", "start", {"task": f"systematic search: {len(found)} identified by `{query[:120]}`; "
                                            f"screening {len(to_screen)}"})
    decisions = screen(ctx, to_screen, protocol)
    eligible, excluded = [], []
    for pid in to_screen:
        d = decisions.get(pid)
        if d is None:
            continue
        if d["decision"] == "exclude":
            excluded.append(pid)
            log_decision(ctx, [pid], "excluded", (d.get("criterion") + ": " if d.get("criterion") else "")
                         + (d.get("reason") or ""))
        else:
            eligible.append(pid)
    unscreened = [p for p in to_screen if p not in decisions]
    if unscreened:
        log_decision(ctx, unscreened, "excluded", "could not be screened (no decision came back)")

    analysed = eligible
    if len(eligible) > max_analysed:
        analysed = sorted(rng.sample(eligible, max_analysed))
        log_decision(ctx, sorted(set(eligible) - set(analysed)), "over_limit",
                     f"eligible; not in the random sample of {max_analysed} analysed (seed {seed})")
    _shortlist(ctx, analysed, decisions)
    misses = recall_check(ctx, set(found))
    note = {"mode": "systematic", "query": query, "query_was_fallback": fallback, "seed": seed,
            "identified": len(found), "screened": len(to_screen), "screened_is_sample": len(found) > max_screened,
            "eligible": len(eligible), "excluded": len(excluded), "unscreened": len(unscreened),
            "analysed": len(analysed), "analysed_is_sample": len(eligible) > max_analysed,
            "missed_by_query": misses}
    ctx.save_note("search", note)
    ctx.emit("discovery", "finish", {"output": {k: note[k] for k in ("identified", "screened", "eligible",
                                                                        "analysed")}})
    return note


def _shortlist(ctx, ids: list[str], decisions: dict) -> None:
    from research_agent.tools.review import log_decision

    for pid in ids:
        d = decisions.get(pid) or {}
        ctx.pg.execute(
            "INSERT INTO run_papers (run_id, paper_id, added_by, reason, score) VALUES (%s,%s,'systematic',%s,%s) "
            "ON CONFLICT (run_id, paper_id) DO NOTHING",
            (ctx.run_id, pid, ("eligible at screening; " + (d.get("reason") or ""))[:300], 1.0))
    log_decision(ctx, ids, "included", "eligible at title and abstract screening")


def recall_check(ctx, found: set[str], probe: int = RECALL_PROBE) -> list[dict]:
    """The question's nearest papers by meaning that the fixed query did not find: what the query may be
    missing. Listed, never added."""
    from research_agent.embeddings import get_embedder

    try:
        rows = ctx.pg.execute(
            "SELECT paper_id, title, year FROM papers WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> %s LIMIT %s", (get_embedder().embed_query(ctx.question), int(probe))).fetchall()
    except Exception:
        return []
    return [{"paper_id": r["paper_id"], "title": (r["title"] or "")[:180], "year": r["year"]}
            for r in rows if r["paper_id"] not in found][:15]


def markdown(ctx) -> list[str]:
    """How the papers were found, for the report."""
    n = ctx.notes().get("search") or {}
    if n.get("mode") != "systematic" or "identified" not in n:
        return []
    L = ["## How the papers were found (computed)", "",
         f"**Systematic search of this corpus**, not of the literature: open-access PMC topic slices and arXiv, "
         f"as loaded. Query, fixed before any result was read: `{n['query']}`"
         + (" (the protocol wrote no specific search query, so its broad topic query was used)."
            if n.get("query_was_fallback") else "."), "",
         f"- Identified: {n['identified']} papers matched the query.",
         f"- Screened on title and abstract: {n['screened']}"
         + (f" (a random sample, seed {n['seed']}, because more were identified than can be screened)."
            if n.get("screened_is_sample") else "."),
         f"- Eligible: {n['eligible']}; excluded with a recorded reason: {n['excluded']}"
         + (f"; could not be screened: {n['unscreened']}" if n.get("unscreened") else "") + ".",
         f"- Analysed: {n['analysed']}"
         + (f" (a random sample of the {n['eligible']} eligible, seed {n['seed']}, so shares estimate the "
            "eligible set)." if n.get("analysed_is_sample") else ", every eligible paper."), ""]
    if n.get("missed_by_query"):
        ids = ", ".join(f"[{m['paper_id']}]" for m in n["missed_by_query"][:8])
        L += [f"**What the query may have missed.** {len(n['missed_by_query'])} of the papers closest to the "
              f"question in meaning were not matched by the query, for example {ids}. They were not added, "
              "because adding them would make the set depend on judgement again; if they are relevant, widen the "
              "query and run again.", ""]
    return L
