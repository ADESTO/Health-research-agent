"""Ingest the arXiv health subset into Postgres + pgvector.

    python -m research_agent.cli ingest            # full health subset
    python -m research_agent.cli ingest --limit 2000   # quick dev load

DuckDB scans the parquet (local folder or straight from Hugging Face), applies the health filter,
and streams record batches; we embed each batch and COPY it into Postgres. Re-running is safe:
papers already loaded are skipped, so an interrupted load resumes where it stopped.
"""
from __future__ import annotations

import time

import duckdb

from research_agent.config import settings
from research_agent.db import connect, init_schema
from research_agent.embeddings import get_embedder, paper_text_for_embedding
from research_agent.ingestion.health_filter import health_subset_sql, year_stats_sql


def duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    if "hf://" in settings.metadata_glob or "hf://" in settings.paper_text_glob:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


def scope(limit_samples: int = 20, csv_path: str | None = None) -> dict:
    """Measure the health subset before loading anything (counts by year, reason, category)."""
    con = duck()
    sql = health_subset_sql(settings.metadata_glob, settings.min_year)
    con.execute(f"CREATE TEMP TABLE subset AS {sql}")
    total = con.execute("SELECT count(*) FROM subset").fetchone()[0]
    by_year = con.execute("SELECT year, count(*) n FROM subset GROUP BY 1 ORDER BY 1").fetchall()
    by_reason = con.execute(
        "SELECT split_part(health_reason, ':', 1) kind, count(*) FROM subset GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    top_terms = con.execute(
        "SELECT regexp_replace(health_reason, ' \\(.*', '') r, count(*) n FROM subset GROUP BY 1 "
        "ORDER BY 2 DESC LIMIT 25"
    ).fetchall()
    by_cat = con.execute(
        "SELECT primary_category, count(*) n FROM subset GROUP BY 1 ORDER BY 2 DESC LIMIT 20"
    ).fetchall()
    samples = con.execute(
        f"SELECT paper_id, health_reason, title FROM subset USING SAMPLE {int(limit_samples)} ROWS"
    ).fetchall()
    if csv_path:  # a bigger random sample to label by hand: is each paper really health research?
        con.execute(f"COPY (SELECT paper_id, health_reason, primary_category, title, '' AS is_health "
                    f"FROM subset USING SAMPLE 300 ROWS) TO '{csv_path}' (HEADER)")
    return {"total": total, "by_year": by_year, "by_reason": by_reason, "top_reasons": top_terms,
            "top_primary_categories": by_cat, "samples": samples}


def load_year_stats(pg) -> int:
    con = duck()
    rows = con.execute(year_stats_sql(settings.metadata_glob, settings.min_year)).fetchall()
    with pg.cursor() as cur:
        cur.execute("DELETE FROM corpus_year_stats WHERE source = 'arxiv'")
        with cur.copy("COPY corpus_year_stats (source, year, primary_category, n_papers) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(("arxiv", *r))
    return len(rows)


def load_papers(pg, limit: int | None = None, batch_size: int = 512, log=print,
                from_year: int | None = None) -> int:
    embedder = get_embedder()
    existing = {r["paper_id"] for r in pg.execute("SELECT paper_id FROM papers").fetchall()}
    con = duck()
    cur_duck = con.execute(health_subset_sql(settings.metadata_glob, max(settings.min_year, from_year or 0), limit))
    names = [d[0] for d in cur_duck.description]
    cols = ["paper_id", "title", "abstract", "authors", "categories", "primary_category", "year",
            "first_version_date", "doi", "journal_ref", "license", "health_reason", "embedding"]
    inserted, t0, embed_secs, read_secs = 0, time.time(), 0.0, 0.0
    while True:
        t_read = time.time()
        batch = cur_duck.fetchmany(batch_size)  # plain tuples: no pyarrow needed at runtime
        read_secs += time.time() - t_read
        if not batch:
            break
        rows = [r for r in (dict(zip(names, t)) for t in batch) if r["paper_id"] not in existing]
        if not rows:
            continue
        t_emb = time.time()
        vecs = embedder.embed_documents(
            [paper_text_for_embedding(r["title"], r["abstract"]) for r in rows]
        )
        embed_secs += time.time() - t_emb
        with pg.cursor() as cur, cur.copy(f"COPY papers ({', '.join(cols)}) FROM STDIN") as cp:
            for r, v in zip(rows, vecs):
                cp.write_row([
                    r["paper_id"], " ".join(r["title"].split()), " ".join(r["abstract"].split()),
                    r["authors"], list(r["categories"]), r["primary_category"], r["year"],
                    r["first_version_date"], r["doi"], r["journal_ref"], r["license"],
                    r["health_reason"], v,
                ])
        inserted += len(rows)
        rate = inserted / max(time.time() - t0, 1e-6)
        log(f"  loaded {inserted:,} papers ({rate:,.1f}/s overall | embedding {inserted / max(embed_secs, 1e-6):,.1f}/s"
            f" | reading parquet {read_secs:,.0f}s so far)")
    return inserted


def build_indexes(pg) -> None:
    pg.execute("SET maintenance_work_mem = '1GB'")
    pg.execute(
        "CREATE INDEX IF NOT EXISTS papers_embedding_hnsw ON papers "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    pg.execute("ANALYZE papers")


def run_ingest(limit: int | None = None, log=print, from_year: int | None = None) -> dict:
    init_schema()
    pg = connect()
    try:
        log("Counting all-arXiv papers per year/category (trend denominators)…")
        n_stats = load_year_stats(pg)
        log(f"  {n_stats:,} year×category rows")
        log(f"Loading health subset (embedder: {get_embedder().name})…")
        n = load_papers(pg, limit=limit, log=log, from_year=from_year)
        log("Building HNSW vector index…")
        build_indexes(pg)
        total = pg.execute("SELECT count(*) AS n FROM papers").fetchone()["n"]
        return {"inserted": n, "total_papers": total, "year_stat_rows": n_stats}
    finally:
        pg.close()
