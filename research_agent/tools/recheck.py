"""Focused re-checks for things the general extraction under-captures.

The general extraction form cannot anticipate every method or data type ("attention layers",
"entomological surveillance"), so an absence count on it can be an artifact. When papers mention the terms
but their extracted field does not record them, each such paper gets one narrow question:

    "Does this paper itself use <concept>? Copy the sentence that shows it."

The answer counts only with a quote that is verified against the text the model was shown. Confirmed
values are merged into the paper's extracted field for this run, so every count (claims, value_counts,
the map) sees them. "Mentioned in the background" is a 'no'; an unverifiable answer is 'unclear'.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings
from research_agent.db import connect
from research_agent.tools.base import STR, STRS, Tool, obj
from research_agent.tools.query import tsquery_sql

MAX_RECHECK = 40          # papers per concept per run
PASSAGES = 4              # text windows per paper
WINDOW = 450              # characters either side of a mention

RECHECK_SYSTEM = """You check one fact in one research paper.

Question: does THIS paper itself use {concept} (as a model, method, or data it analyses)?
- yes: the paper's own work uses it. Copy one sentence that shows this, exactly as written.
- no: it is only mentioned in the background, related work or future work, or not at all.
- unclear: the text shown is not enough to tell.
Answer only from the text shown."""


def _terms(any_of) -> list[str]:
    return [t for t in dict.fromkeys(" ".join(str(x).split()).strip() for x in any_of or []) if t]


def concept_key(terms: list[str]) -> str:
    return "|".join(sorted({t.lower() for t in terms}))


def _match_sql(terms: list[str], prefix: str) -> tuple[str, dict]:
    return tsquery_sql(" OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms), prefix=prefix)


def mentioning(pg, paper_ids: list[str], terms: list[str]) -> set[str]:
    """Papers among paper_ids whose title, abstract or indexed full text mention any of the terms."""
    if not paper_ids or not terms:
        return set()
    sql, params = _match_sql(terms, "mt")
    rows = pg.execute(
        f"""SELECT p.paper_id FROM papers p
            LEFT JOIN paper_fulltext f ON f.paper_id = p.paper_id AND f.status = 'ok'
            WHERE p.paper_id = ANY(%(ids)s) AND (p.tsv @@ ({sql}) OR coalesce(f.tsv @@ ({sql}), false))""",
        {**params, "ids": list(paper_ids)}).fetchall()
    return {r["paper_id"] for r in rows}


def verdicts(pg, run_id: str, field: str, terms: list[str]) -> dict[str, str]:
    rows = pg.execute("SELECT paper_id, verdict FROM rechecks WHERE run_id=%s AND field=%s AND concept=%s",
                      (run_id, field, concept_key(terms))).fetchall()
    return {r["paper_id"]: r["verdict"] for r in rows}


def confirmed_values(pg, run_id: str) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """{paper_id: {field: [(value, quote), ...]}} for verified 'yes' answers in this run."""
    out: dict[str, dict[str, list]] = {}
    for r in pg.execute("SELECT paper_id, field, value, quote FROM rechecks WHERE run_id=%s AND verdict='yes'",
                        (run_id,)).fetchall():
        out.setdefault(r["paper_id"], {}).setdefault(r["field"], []).append((r["value"], r["quote"]))
    return out


def _passages(text: str, terms: list[str]) -> list[str]:
    spans = []
    for t in terms:
        rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"s?(?![A-Za-z0-9])", re.I)
        for m in rx.finditer(text):
            spans.append((max(0, m.start() - WINDOW), min(len(text), m.end() + WINDOW)))
            if len(spans) >= 3 * PASSAGES:
                break
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [text[a:b].strip() for a, b in merged[:PASSAGES]]


def _check_tool(terms: list[str]) -> dict:
    return {"name": "record_check", "description": "Record the answer.",
            "input_schema": obj({"uses": {"type": "string", "enum": ["yes", "no", "unclear"]},
                                 "which": {"type": "string", "enum": terms,
                                           "description": "Which of the terms the paper uses (if yes)"},
                                 "quote": {**STR, "description": "The sentence that shows it, copied exactly"}},
                                ["uses"])}


def _ask(llm, concept: str, terms: list[str], paper: dict, text: str) -> dict:
    from research_agent.tools.extraction import _words, quote_found

    resp = llm.chat(RECHECK_SYSTEM.format(concept=concept),
                    [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    tools=[_check_tool(terms)], force_tool="record_check", max_tokens=400)
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    verdict = args.get("uses") if args.get("uses") in ("yes", "no", "unclear") else "unclear"
    quote = " ".join(str(args.get("quote") or "").split())[:400]
    value = args.get("which") if args.get("which") in terms else terms[0]
    if verdict == "yes":
        words = _words(text)
        shingles = {tuple(words[i:i + k]) for k in (3, 4, 5) for i in range(len(words) - k + 1)}
        if not quote_found(quote, words, shingles):
            verdict = "unclear"      # a 'yes' without a quote that is really in the paper does not count
    return {"verdict": verdict, "value": value, "quote": quote if verdict == "yes" else ""}


def recheck(ctx, field: str, any_of: list[str], concept: str | None = None,
            paper_ids: list[str] | None = None) -> dict:
    """Re-check papers that mention the terms but whose extracted `field` does not record them."""
    from research_agent.tools import claims
    from research_agent.tools.claims import _hits, _norm
    from research_agent.tools.extraction import known_fields

    lists, _ = known_fields(ctx)
    if field not in lists:
        return {"error": f"re-checks apply to list fields ({', '.join(lists)}); enum fields have fixed values"}
    terms = _terms(any_of)
    if not terms:
        return {"error": "give the terms to check (any_of)"}
    concept = concept or " or ".join(terms[:4])
    rows = {r["paper_id"]: r for r in claims._rows(ctx) if not paper_ids or r["paper_id"] in set(paper_ids)}
    ensure_fulltext_index(ctx)
    needles = [_norm(t) for t in terms]
    missed = [pid for pid in mentioning(ctx.pg, list(rows), terms)
              if not _hits(rows[pid], field, needles, False)]
    done = verdicts(ctx.pg, ctx.run_id, field, terms)
    todo = [pid for pid in missed if pid not in done][:MAX_RECHECK]
    papers = {r["paper_id"]: r for r in ctx.pg.execute(
        """SELECT p.paper_id, p.title, p.abstract, f.clean_text FROM papers p
           LEFT JOIN paper_fulltext f ON f.paper_id = p.paper_id AND f.status = 'ok'
           WHERE p.paper_id = ANY(%s)""", (todo,)).fetchall()}
    llm = ctx.llm_factory(step="recheck")
    setattr(llm, "_step", "recheck")
    key = concept_key(terms)
    results: dict[str, dict] = {}

    def work(pid):
        p = papers[pid]
        text = f"Title: {p['title']}\n\nAbstract: {p['abstract']}"
        found = _passages(p.get("clean_text") or "", terms)
        if found:
            text += "\n\nPassages from the full text that mention it:\n\n" + "\n\n[...]\n\n".join(found)
        return pid, _ask(llm, concept, terms, p, text)

    with ThreadPoolExecutor(max_workers=settings.extraction_workers) as pool:
        futures = [pool.submit(work, pid) for pid in todo if pid in papers]
        for fut in as_completed(futures):
            try:
                pid, res = fut.result()
                results[pid] = res
            except Exception as exc:   # one failed call leaves that paper unchecked, not the run broken
                ctx.emit("recheck", "error", {"error": str(exc)[:200]})
    conn = connect()
    try:
        for pid, res in results.items():
            conn.execute(
                "INSERT INTO rechecks (run_id, paper_id, field, concept, verdict, value, quote) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id, paper_id, field, concept) DO UPDATE "
                "SET verdict=EXCLUDED.verdict, value=EXCLUDED.value, quote=EXCLUDED.quote, ts=now()",
                (ctx.run_id, pid, field, key, res["verdict"], res["value"], res["quote"]))
    finally:
        conn.close()
    all_v = verdicts(ctx.pg, ctx.run_id, field, terms)
    out = {"field": field, "concept": concept, "papers_mentioning_but_not_recorded": len(missed),
           "checked_now": len(results), "confirmed": sum(1 for v in all_v.values() if v == "yes"),
           "only_mentioned": sum(1 for v in all_v.values() if v == "no"),
           "unclear": sum(1 for v in all_v.values() if v == "unclear"),
           "not_checked": max(0, len(missed) - len(all_v))}
    ctx.emit("recheck", "finish", {"output": out})
    return out


def ensure_fulltext_index(ctx) -> None:
    """Fetch full text for every analysed paper once per run, so mention checks see methods sections.
    Fetching is not reading: no model call is made, and the full-text reading budget is untouched."""
    if (ctx.notes().get("fulltext_index") or {}).get("done"):
        return
    from research_agent.ingestion.fulltext import fetch_fulltext

    try:
        status = fetch_fulltext(ctx.pg, ctx.shortlist_ids())
    except Exception as exc:     # no full text available (offline, missing files): abstracts still work
        ctx.emit("recheck", "error", {"error": f"full-text fetch failed: {str(exc)[:200]}"})
        status = {}
    ctx.save_note("fulltext_index", {"done": True,
                                     "status": {s: sum(1 for v in status.values() if v == s)
                                                for s in set(status.values())}})


RECHECK_TOOL = Tool(
    "recheck_field",
    "Re-check papers that MENTION a method or data type in their text but whose extracted field does not "
    "record it (the general extraction form under-captures things like attention layers or entomological "
    "data). Each such paper gets one quoted yes/no question; confirmed uses are added to the field, so "
    "counts afterwards include them. Use before claiming something is rare or absent.",
    obj({"field": STR, "any_of": {**STRS, "description": "The terms and synonyms, e.g. ['transformer', "
                                   "'attention', 'temporal fusion transformer']"},
         "concept": {**STR, "description": "Plain name, e.g. 'transformer or attention-based models'"}},
        ["field", "any_of"]),
    recheck,
)
