"""Search the full texts of the papers on a question's topic, not only their titles and abstracts.

A gap is computed from the shortlist, and the corpus check widens it to every paper on the topic. But that
check reads titles and abstracts only, and the things gaps are about (a validation design, an interaction
term, a safety endpoint) are exactly the things abstracts leave out. So "0 of 347 topic papers mention it"
says little, and "10 of 347 mention it" names ten papers nobody read.

This fetches the full text of the topic's papers (the arXiv ones from the local files, PMC ones from NCBI,
twenty to a request), keeps them so later runs on the same topic reuse them, and runs the same search over
them. Two things it is careful about:

  - A match is a mention, not a finding. "We did not model the interaction between rainfall and
    temperature" matches a search for an interaction and says the opposite. So every match comes back with
    the passage around it, for the gap reasoning to read before anything is called absent.
  - A text that could not be fetched is not a text that was searched. The counts say how many full texts
    were actually read, and absence is only ever stated over those.

The topic set is capped (FULLTEXT_CHECK_MAX, 300 by default, ranked by how well each paper matches the topic)
so a broad topic cannot turn one map into an hour of downloads. 0 turns the check off.
"""
from __future__ import annotations

import re

from research_agent.config import settings

MAX_CANDIDATES = 12
_OPS = {"or", "and", "not"}


def search_terms(search: str) -> list[str]:
    """The words and phrases a search query looks for, for finding the passage a match came from."""
    q = search or ""
    # excluded terms ("-review", "NOT review", '-"case report"') are what a match must NOT contain, so they
    # are never what a passage is chosen for
    q = re.sub(r'(?:(?<=\s)|^)-"[^"]*"|(?:(?<=\s)|^)-\S+|\bNOT\s+("[^"]*"|\S+)', " ", q)
    phrases = re.findall(r'"([^"]+)"', q)
    rest = re.sub(r'"[^"]*"', " ", q)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", rest) if w.lower() not in _OPS]
    return list(dict.fromkeys([p.strip() for p in phrases if p.strip()] + words))


def topic_paper_ids(ctx, topic_query: str, limit: int) -> list[str]:
    """The corpus's papers on the topic, best match first."""
    from research_agent.tools.query import tsquery_sql

    sql, params = tsquery_sql(topic_query, prefix="t")
    return [r["paper_id"] for r in ctx.pg.execute(
        f"SELECT paper_id FROM papers WHERE source <> 'upload' AND tsv @@ ({sql}) ORDER BY ts_rank_cd(tsv, ({sql})) DESC, paper_id "
        f"LIMIT %(lim)s", {**params, "lim": int(limit)}).fetchall()]


def _ensure_topic_texts(ctx, topic_query: str, limit: int) -> dict:
    """Fetch the topic's full texts once per run and topic; later calls reuse what was fetched."""
    cache = getattr(ctx, "_topic_texts", None)
    if cache is None:
        cache = {}
        try:
            ctx._topic_texts = cache
        except Exception:
            pass
    if topic_query in cache:
        return cache[topic_query]
    from research_agent.ingestion.fulltext import fetch_fulltext

    ids = topic_paper_ids(ctx, topic_query, limit)
    problem = None
    try:
        fetch_fulltext(ctx.pg, ids)
    except Exception as exc:          # a network problem costs the full-text check, never the map
        problem = str(exc)[:200]
    readable = [r["paper_id"] for r in ctx.pg.execute(
        "SELECT paper_id FROM paper_fulltext WHERE paper_id = ANY(%s) AND status = 'ok'", (ids,)).fetchall()]
    info = {"ids": ids, "readable": readable, "problem": problem}
    cache[topic_query] = info
    return info


def search_fulltexts(ctx, topic_query: str, search: str, limit: int | None = None) -> dict | None:
    """Among the topic's papers whose full text could be read, which ones mention the search, with passages."""
    limit = settings.fulltext_check_max if limit is None else int(limit)
    if not topic_query or not search or limit <= 0:
        return None
    from research_agent.tools.query import tsquery_sql
    from research_agent.tools.recheck import _passages

    info = _ensure_topic_texts(ctx, topic_query, limit)
    if not info["readable"]:
        return {"topic_papers_checked": len(info["ids"]), "full_texts_read": 0, "matching": 0, "candidates": [],
                **({"fetch_problem": info["problem"]} if info["problem"] else {}),
                "note": "No full text of a topic paper could be read, so nothing is said about full texts."}
    sql, params = tsquery_sql(search, prefix="s")
    rows = ctx.pg.execute(
        f"""SELECT f.paper_id, p.title, p.year, f.clean_text FROM paper_fulltext f JOIN papers p USING (paper_id)
            WHERE f.paper_id = ANY(%(ids)s) AND f.status = 'ok'
              AND to_tsvector('english', f.clean_text) @@ ({sql})""",
        {**params, "ids": info["readable"]}).fetchall()
    shortlist = set(ctx.shortlist_ids())
    terms = search_terms(search)
    candidates = []
    for r in rows:
        passages = _passages(r["clean_text"] or "", terms) if terms else []
        # the passage that names the most search terms is the one worth reading first
        best = max(passages, key=lambda t: sum(x.lower() in t.lower() for x in terms), default="")
        candidates.append({"paper_id": r["paper_id"], "title": (r["title"] or "")[:200], "year": r["year"],
                           "analysed_in_this_run": r["paper_id"] in shortlist,
                           "passage": " ".join(best.split())[:600]})
    candidates.sort(key=lambda c: (c["analysed_in_this_run"], -(c["year"] or 0)))
    out = {"topic_papers_checked": len(info["ids"]), "full_texts_read": len(info["readable"]),
           "matching": len(rows), "not_analysed": sum(1 for c in candidates if not c["analysed_in_this_run"]),
           "candidates": candidates[:MAX_CANDIDATES],
           "note": "A match is a mention, not a finding: read each passage before calling the practice present "
                   "or absent."}
    if info["problem"]:
        out["fetch_problem"] = info["problem"]
    return out
