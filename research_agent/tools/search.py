"""Discovery tools: hybrid (semantic + keyword) search, similarity, shortlist management."""
from __future__ import annotations

import re

from research_agent.config import settings
from research_agent.embeddings import get_embedder
from research_agent.tools.base import INT, STR, STRS, Tool, obj
from research_agent.tools.query import tsquery_sql

RRF_K = 60
SOURCE = {"type": "string", "enum": ["arxiv", "pmc", "all"],
          "description": "Which corpus to search: arxiv preprints, pmc published open-access articles, "
                         "or all (default)."}


def _filters(year_from, year_to, categories, source=None) -> tuple[str, dict]:
    # a user's own documents are never part of the searchable corpus: they enter only the runs they are
    # attached to, and other users never see them
    clauses, params = ["source <> 'upload'"], {}
    if source and source != "all":
        clauses.append("source = %(src)s"); params["src"] = str(source)
    if year_from:
        clauses.append("year >= %(yf)s"); params["yf"] = int(year_from)
    if year_to:
        clauses.append("year <= %(yt)s"); params["yt"] = int(year_to)
    if categories:
        clauses.append("categories && %(cats)s::text[]"); params["cats"] = list(categories)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def hybrid_search(ctx, query: str, keywords: str | None = None, year_from: int | None = None,
                  year_to: int | None = None, categories: list[str] | None = None, limit: int = 25,
                  source: str | None = None) -> dict:
    """Reciprocal-rank fusion of pgvector cosine search and Postgres full-text search."""
    limit = max(1, min(int(limit or 25), 60))
    where, params = _filters(year_from, year_to, categories, source)
    tq_sql, tq_params = tsquery_sql(keywords or query)
    params.update(tq_params, qvec=get_embedder().embed_query(query), lim=limit, run=ctx.run_id)
    sql = f"""
    WITH q AS (SELECT ({tq_sql}) AS tq),
    sem AS (
        SELECT paper_id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS r
        FROM (SELECT paper_id, embedding FROM papers WHERE embedding IS NOT NULL {where}
              ORDER BY embedding <=> %(qvec)s LIMIT 150) s
    ),
    kw AS (
        SELECT paper_id, row_number() OVER (ORDER BY rank DESC) AS r
        FROM (SELECT paper_id, ts_rank_cd(tsv, q.tq) AS rank FROM papers, q
              WHERE tsv @@ q.tq {where} ORDER BY rank DESC LIMIT 150) k
    ),
    fused AS (
        SELECT paper_id, coalesce(1.0/({RRF_K}+sem.r), 0) + coalesce(1.0/({RRF_K}+kw.r), 0) AS score,
               sem.r AS sem_rank, kw.r AS kw_rank
        FROM sem FULL OUTER JOIN kw USING (paper_id)
    )
    SELECT p.paper_id, p.source, p.title, p.year, p.primary_category, left(p.abstract, 280) AS abstract_start,
           round(f.score::numeric, 5) AS score, f.sem_rank, f.kw_rank,
           EXISTS (SELECT 1 FROM run_papers rp WHERE rp.run_id = %(run)s AND rp.paper_id = p.paper_id)
               AS in_shortlist
    FROM fused f JOIN papers p USING (paper_id)
    ORDER BY f.score DESC LIMIT %(lim)s
    """
    rows = ctx.pg.execute(sql, params).fetchall()
    from research_agent.tools.review import log_identified

    log_identified(ctx, [r["paper_id"] for r in rows], f"search: {query}")
    return {"query": query, "keywords": keywords or query, "n": len(rows), "results": rows}


def corpus_count(ctx, keywords: str, year_from: int | None = None, year_to: int | None = None,
                 source: str | None = None) -> dict:
    """How many health-corpus papers match a keyword query (to gauge how big a topic is)."""
    where, params = _filters(year_from, year_to, None, source)
    tq_sql, tq_params = tsquery_sql(keywords)
    params.update(tq_params)
    n = ctx.pg.execute(f"SELECT count(*) AS n FROM papers WHERE tsv @@ ({tq_sql}) {where}", params).fetchone()["n"]
    total = ctx.pg.execute(f"SELECT count(*) AS n FROM papers WHERE TRUE {where}", params).fetchone()["n"]
    # Record the count so the report's number audit can accept "n of N" figures that came from this tool.
    ctx.emit("corpus_count", "count", {"keywords": keywords, "n": n, "total": total})
    by_source = {r["source"]: r["n"] for r in ctx.pg.execute(
        f"SELECT source, count(*) AS n FROM papers WHERE tsv @@ ({tq_sql}) {where} GROUP BY source",
        params).fetchall()}
    return {"by_source": by_source, "keywords": keywords, "matching_papers": n, "health_corpus_size": total}


def coverage_probe(ctx, topic: str, concepts: list[str], limit: int = 6, source: str | None = None) -> dict:
    """For each concept (a method or data type the question cares about), how many papers on the topic in
    the whole corpus mention it, how many of those are on the shortlist, and the best ones that are not.
    A concept that is common in the corpus but thin on the shortlist is a sign discovery missed a strand."""
    where, fparams = _filters(None, None, None, source)
    shortlisted = set(ctx.shortlist_ids())
    t_sql, t_params = tsquery_sql(topic, prefix="tp")
    topic_total = ctx.pg.execute(f"SELECT count(*) n FROM papers WHERE tsv @@ ({t_sql}) {where}",
                                 {**t_params, **fparams}).fetchone()["n"]
    out = []
    for concept in (concepts or [])[:8]:
        c_sql, c_params = tsquery_sql(concept, prefix="cp")
        params = {**t_params, **c_params, **fparams, "ids": list(shortlisted) or [""], "lim": max(1, min(int(limit), 15))}
        match = f"tsv @@ ({t_sql}) AND tsv @@ ({c_sql}) {where}"
        n = ctx.pg.execute(f"SELECT count(*) n FROM papers WHERE {match}", params).fetchone()["n"]
        on = ctx.pg.execute(f"SELECT count(*) n FROM papers WHERE {match} AND paper_id = ANY(%(ids)s)",
                            params).fetchone()["n"]
        missing = ctx.pg.execute(
            f"""SELECT paper_id, source, title, year FROM papers WHERE {match} AND NOT (paper_id = ANY(%(ids)s))
                ORDER BY ts_rank_cd(tsv, ({t_sql}) && ({c_sql})) DESC, year DESC LIMIT %(lim)s""",
            params).fetchall()
        ctx.emit("coverage_probe", "count", {"n": n, "total": topic_total, "keywords": f"{topic} + {concept}"})
        from research_agent.tools.review import log_identified

        log_identified(ctx, [m["paper_id"] for m in missing], f"coverage probe: {concept}")
        under = n >= 5 and on < 0.3 * n
        out.append({"concept": concept, "corpus_papers": n, "on_shortlist": on,
                    "under_covered": under, "not_shortlisted_examples": missing})
    return {"topic": topic, "topic_papers_in_corpus": topic_total, "probes": out,
            "note": "Counts are title/abstract keyword matches. For every under_covered concept, read the "
                    "examples and add the ones that answer the question; skip off-topic ones."}


def find_similar(ctx, paper_id: str, limit: int = 10) -> dict:
    rows = ctx.pg.execute(
        """SELECT p.paper_id, p.title, p.year, round((1 - (p.embedding <=> s.embedding))::numeric, 3) AS similarity
           FROM papers s, papers p
           WHERE s.paper_id = %s AND p.paper_id <> s.paper_id AND p.source <> 'upload'
           ORDER BY p.embedding <=> s.embedding LIMIT %s""",
        (paper_id, max(1, min(int(limit), 30)))).fetchall()
    from research_agent.tools.review import log_identified

    log_identified(ctx, [r["paper_id"] for r in rows], f"similar to {paper_id}")
    return {"paper_id": paper_id, "similar": rows}


def get_papers(ctx, paper_ids: list[str]) -> dict:
    rows = ctx.pg.execute(
        """SELECT paper_id, source, title, year, authors, categories, primary_category, abstract, doi,
                  journal_ref FROM papers WHERE paper_id = ANY(%s)""", (list(paper_ids)[:20],)).fetchall()
    return {"papers": rows}


# ---------------------------------------------------------------- shortlist (shared run state)
def _norm_title(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).split())


def _surname(authors: str) -> str:
    first = re.split(r",|\band\b", authors or "")[0].strip()
    return first.split()[-1].lower() if first.split() else ""


def _same_work(a: dict, b: dict) -> bool:
    """A preprint and its published version: same DOI, or same title (one may extend the other) and
    the same first author."""
    if a["doi"] and b["doi"] and a["doi"].lower().strip() == b["doi"].lower().strip():
        return True
    ta, tb = _norm_title(a["title"]), _norm_title(b["title"])
    short, long_ = sorted((ta, tb), key=len)
    return len(short) >= 40 and long_.startswith(short) and _surname(a["authors"]) == _surname(b["authors"])


def add_to_shortlist(ctx, paper_ids: list[str], reason: str, _agent: str = "discovery") -> dict:
    """Add papers to the run's shortlist. The same study can exist twice, as an arXiv preprint and as
    its published PMC version; only one copy is kept (the published one), so nothing is counted twice."""
    cols = "paper_id, source, title, doi, authors"
    # a user's document enters a run only when its owner attaches it, never through an agent
    rows = {r["paper_id"]: r for r in ctx.pg.execute(
        f"SELECT {cols} FROM papers WHERE paper_id = ANY(%s) AND source <> 'upload'", (list(paper_ids),)).fetchall()}
    unknown = sorted(set(paper_ids) - set(rows))
    shortlist = {r["paper_id"]: r for r in ctx.pg.execute(
        f"SELECT {cols} FROM run_papers JOIN papers USING (paper_id) WHERE run_id=%s", (ctx.run_id,)).fetchall()}
    added, duplicates, replaced, full = 0, [], [], 0
    for pid in [p for p in dict.fromkeys(paper_ids) if p in rows]:
        new = rows[pid]
        if pid in shortlist:
            continue
        twin = next((o for o in shortlist.values() if o["source"] != new["source"] and _same_work(o, new)), None)
        if twin and new["source"] != "pmc":
            duplicates.append({"paper_id": pid, "same_work_as": twin["paper_id"]})
            continue
        if twin:   # a published PMC version replaces the arXiv preprint already on the list
            ctx.pg.execute("DELETE FROM run_papers WHERE run_id=%s AND paper_id=%s", (ctx.run_id, twin["paper_id"]))
            shortlist.pop(twin["paper_id"])
            replaced.append({"removed_preprint": twin["paper_id"], "kept_published": pid})
        if len(shortlist) >= settings.max_shortlist:
            full += 1
            continue
        cur = ctx.pg.execute(
            "INSERT INTO run_papers (run_id, paper_id, added_by, reason) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT DO NOTHING", (ctx.run_id, pid, _agent, reason))
        added += cur.rowcount
        shortlist[pid] = new
    out = {"added": added, "shortlist_size": len(shortlist), "max_shortlist": settings.max_shortlist}
    from research_agent.tools.review import log_decision, log_identified

    log_identified(ctx, list(rows), "added directly")
    log_decision(ctx, [p for p in shortlist if p in rows], "included", reason)
    for d in duplicates:
        log_decision(ctx, [d["paper_id"]], "duplicate", f"same work as {d['same_work_as']}")
    for r in replaced:
        log_decision(ctx, [r["removed_preprint"]], "duplicate", f"preprint of {r['kept_published']}")
    if full:
        log_decision(ctx, [p for p in rows if p not in shortlist and p not in {d["paper_id"] for d in duplicates}],
                     "over_limit", "shortlist full")
    if unknown:
        out["unknown_ids"] = unknown
    if duplicates:
        out["skipped_duplicates"] = duplicates
    if replaced:
        out["replaced_preprints"] = replaced
    if full:
        out["warning"] = f"shortlist full; {full} ids not added"
    return out


def remove_from_shortlist(ctx, paper_ids: list[str], reason: str = "") -> dict:
    mine = {r["paper_id"] for r in ctx.pg.execute(
        "SELECT paper_id FROM run_papers WHERE run_id=%s AND added_by='user'", (ctx.run_id,)).fetchall()}
    kept = [p for p in paper_ids if p in mine]
    paper_ids = [p for p in paper_ids if p not in mine]
    if kept and not paper_ids:
        return {"removed": 0, "shortlist_size": len(ctx.shortlist_ids()),
                "note": f"{', '.join(kept)} were added by the user and stay in the run"}
    cur = ctx.pg.execute("DELETE FROM run_papers WHERE run_id=%s AND paper_id = ANY(%s)",
                         (ctx.run_id, list(paper_ids)))
    from research_agent.tools.review import log_decision

    log_decision(ctx, list(paper_ids), "excluded", reason or "judged off-topic")
    return {"removed": cur.rowcount, "shortlist_size": len(ctx.shortlist_ids())}


def view_shortlist(ctx) -> dict:
    rows = ctx.pg.execute(
        """SELECT p.paper_id, p.title, p.year FROM run_papers rp JOIN papers p USING (paper_id)
           WHERE rp.run_id=%s ORDER BY p.year, p.paper_id""", (ctx.run_id,)).fetchall()
    years: dict[int, int] = {}
    for r in rows:
        years[r["year"]] = years.get(r["year"], 0) + 1
    return {"size": len(rows), "by_year": years, "papers": rows}


YEAR_PROPS = {"year_from": INT, "year_to": INT}

SEARCH_TOOLS = [
    Tool("hybrid_search",
         "Search the health-research corpus. Combines semantic similarity on `query` (natural language) "
         "with keyword matching on `keywords` (web-search syntax: quoted phrases, OR, -exclude; defaults "
         "to `query`). Returns ranked papers with a short abstract snippet.",
         obj({"query": STR, "keywords": STR, **YEAR_PROPS,
              "categories": {**STRS, "description": "arXiv categories to require, e.g. ['eess.IV']"},
              "source": SOURCE, "limit": INT}, ["query"]),
         hybrid_search, read_only=True),
    Tool("corpus_count",
         "Count how many papers in the whole health corpus match a keyword query. Use to size a topic.",
         obj({"keywords": STR, **YEAR_PROPS, "source": SOURCE}, ["keywords"]), corpus_count, read_only=True),
    Tool("coverage_probe",
         "Check the shortlist is not missing a strand of the literature: for each concept (method family or "
         "data type, with synonyms in web-search syntax), compare how many topic papers in the whole corpus "
         "mention it with how many are shortlisted, and list the best ones not yet shortlisted.",
         obj({"topic": {**STR, "description": "The question's topic query, e.g. malaria OR plasmodium"},
              "concepts": {**STRS, "description": "One query per concept, e.g. ['transformer OR attention', "
                                                  "'entomological OR mosquito OR vector OR larval']"},
              "source": SOURCE, "limit": INT}, ["topic", "concepts"]),
         coverage_probe, read_only=True),
    Tool("find_similar", "Nearest neighbours of a paper by embedding similarity.",
         obj({"paper_id": STR, "limit": INT}, ["paper_id"]), find_similar, read_only=True),
    Tool("get_papers", "Full metadata and abstract for up to 20 paper ids.",
         obj({"paper_ids": STRS}, ["paper_ids"]), get_papers, read_only=True),
]

SHORTLIST_TOOLS = [
    Tool("add_to_shortlist",
         "Add papers to this run's shortlist (the set every other agent will analyse). Give a reason.",
         obj({"paper_ids": STRS, "reason": STR}, ["paper_ids", "reason"]), add_to_shortlist),
    Tool("remove_from_shortlist", "Remove off-topic papers from the shortlist.",
         obj({"paper_ids": STRS, "reason": STR}, ["paper_ids"]), remove_from_shortlist),
    Tool("view_shortlist", "Show the current shortlist with its year distribution.", obj({}), view_shortlist),
]
