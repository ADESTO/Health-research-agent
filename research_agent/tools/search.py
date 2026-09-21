"""Discovery tools: hybrid (semantic + keyword) search, similarity, shortlist management."""
from __future__ import annotations

from research_agent.config import settings
from research_agent.embeddings import get_embedder
from research_agent.tools.base import INT, STR, STRS, Tool, obj

RRF_K = 60


def _filters(year_from, year_to, categories) -> tuple[str, dict]:
    clauses, params = [], {}
    if year_from:
        clauses.append("year >= %(yf)s"); params["yf"] = int(year_from)
    if year_to:
        clauses.append("year <= %(yt)s"); params["yt"] = int(year_to)
    if categories:
        clauses.append("categories && %(cats)s::text[]"); params["cats"] = list(categories)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def hybrid_search(ctx, query: str, keywords: str | None = None, year_from: int | None = None,
                  year_to: int | None = None, categories: list[str] | None = None, limit: int = 25) -> dict:
    """Reciprocal-rank fusion of pgvector cosine search and Postgres full-text search."""
    limit = max(1, min(int(limit or 25), 60))
    where, params = _filters(year_from, year_to, categories)
    params.update(
        qvec=get_embedder().embed_query(query),
        tq=keywords or query,
        lim=limit,
        run=ctx.run_id,
    )
    sql = f"""
    WITH q AS (SELECT websearch_to_tsquery('english', %(tq)s) AS tq),
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
    SELECT p.paper_id, p.title, p.year, p.primary_category, left(p.abstract, 280) AS abstract_start,
           round(f.score::numeric, 5) AS score, f.sem_rank, f.kw_rank,
           EXISTS (SELECT 1 FROM run_papers rp WHERE rp.run_id = %(run)s AND rp.paper_id = p.paper_id)
               AS in_shortlist
    FROM fused f JOIN papers p USING (paper_id)
    ORDER BY f.score DESC LIMIT %(lim)s
    """
    rows = ctx.pg.execute(sql, params).fetchall()
    return {"query": query, "keywords": keywords or query, "n": len(rows), "results": rows}


def corpus_count(ctx, keywords: str, year_from: int | None = None, year_to: int | None = None) -> dict:
    """How many health-corpus papers match a keyword query (to gauge how big a topic is)."""
    where, params = _filters(year_from, year_to, None)
    params["tq"] = keywords
    n = ctx.pg.execute(
        f"SELECT count(*) AS n FROM papers WHERE tsv @@ websearch_to_tsquery('english', %(tq)s) {where}",
        params).fetchone()["n"]
    total = ctx.pg.execute(f"SELECT count(*) AS n FROM papers WHERE TRUE {where}", params).fetchone()["n"]
    return {"keywords": keywords, "matching_papers": n, "health_corpus_size": total}


def find_similar(ctx, paper_id: str, limit: int = 10) -> dict:
    rows = ctx.pg.execute(
        """SELECT p.paper_id, p.title, p.year, round((1 - (p.embedding <=> s.embedding))::numeric, 3) AS similarity
           FROM papers s, papers p
           WHERE s.paper_id = %s AND p.paper_id <> s.paper_id
           ORDER BY p.embedding <=> s.embedding LIMIT %s""",
        (paper_id, max(1, min(int(limit), 30)))).fetchall()
    return {"paper_id": paper_id, "similar": rows}


def get_papers(ctx, paper_ids: list[str]) -> dict:
    rows = ctx.pg.execute(
        """SELECT paper_id, title, year, authors, categories, primary_category, abstract, doi, journal_ref
           FROM papers WHERE paper_id = ANY(%s)""", (list(paper_ids)[:20],)).fetchall()
    return {"papers": rows}


# ---------------------------------------------------------------- shortlist (shared run state)
def add_to_shortlist(ctx, paper_ids: list[str], reason: str, _agent: str = "discovery") -> dict:
    current = len(ctx.shortlist_ids())
    room = settings.max_shortlist - current
    valid = [r["paper_id"] for r in ctx.pg.execute(
        "SELECT paper_id FROM papers WHERE paper_id = ANY(%s)", (list(paper_ids),)).fetchall()]
    unknown = sorted(set(paper_ids) - set(valid))
    added = 0
    for pid in valid[: max(room, 0)]:
        cur = ctx.pg.execute(
            "INSERT INTO run_papers (run_id, paper_id, added_by, reason) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT DO NOTHING", (ctx.run_id, pid, _agent, reason))
        added += cur.rowcount
    out = {"added": added, "shortlist_size": current + added, "max_shortlist": settings.max_shortlist}
    if unknown:
        out["unknown_ids"] = unknown
    if len(valid) > room:
        out["warning"] = f"shortlist full; {len(valid) - max(room, 0)} ids not added"
    return out


def remove_from_shortlist(ctx, paper_ids: list[str], reason: str = "") -> dict:
    cur = ctx.pg.execute("DELETE FROM run_papers WHERE run_id=%s AND paper_id = ANY(%s)",
                         (ctx.run_id, list(paper_ids)))
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
              "limit": INT}, ["query"]),
         hybrid_search),
    Tool("corpus_count",
         "Count how many papers in the whole health corpus match a keyword query. Use to size a topic.",
         obj({"keywords": STR, **YEAR_PROPS}, ["keywords"]), corpus_count),
    Tool("find_similar", "Nearest neighbours of a paper by embedding similarity.",
         obj({"paper_id": STR, "limit": INT}, ["paper_id"]), find_similar),
    Tool("get_papers", "Full metadata and abstract for up to 20 paper ids.",
         obj({"paper_ids": STRS}, ["paper_ids"]), get_papers),
]

SHORTLIST_TOOLS = [
    Tool("add_to_shortlist",
         "Add papers to this run's shortlist (the set every other agent will analyse). Give a reason.",
         obj({"paper_ids": STRS, "reason": STR}, ["paper_ids", "reason"]), add_to_shortlist),
    Tool("remove_from_shortlist", "Remove off-topic papers from the shortlist.",
         obj({"paper_ids": STRS, "reason": STR}, ["paper_ids"]), remove_from_shortlist),
    Tool("view_shortlist", "Show the current shortlist with its year distribution.", obj({}), view_shortlist),
]
