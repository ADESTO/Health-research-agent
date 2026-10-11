"""Live arXiv search: preprints the loaded arXiv slice does not hold (posted after the dataset snapshot, or left
out by the health filter), fetched during a run when the AI judges the corpus too thin.

Uses arXiv's public API (no key; arXiv asks for at most one request every three seconds, which one search per
call respects). Papers come in as their arXiv record, title and abstract, under their arXiv id like any other
arXiv paper; when one is chosen to be read in full, its PDF is fetched from arxiv.org (ingestion/openaccess.py).
Searching and loading cost no tokens.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

from research_agent.config import settings

API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_last = [0.0]
MIN_ABSTRACT = 80


def _http():
    import httpx

    return httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True,
                        headers={"User-Agent": "health-research-agent/1.0 (literature review)"})


def arxiv_query(q: str) -> str:
    """arXiv's own syntax (ti:, abs:, all:, AND, OR, ANDNOT) is kept; bare words and phrases get all:."""
    q = " ".join((q or "").split())
    if re.search(r"\b(ti|abs|all|au|cat|co|jr):", q):
        return q

    def field(m):
        w = m.group(0)
        return w if w in ("AND", "OR", "ANDNOT") else f"all:{w}"
    return re.sub(r'"[^"]+"|[A-Za-z][\w\-]*', field, q)


def _t(node, path) -> str:
    el = node.find(path, _NS)
    return " ".join((el.text or "").split()) if el is not None and el.text else ""


def parse(xml: str) -> list[dict]:
    out = []
    for e in ET.fromstring(xml).findall("a:entry", _NS):
        m = re.search(r"arxiv\.org/abs/(.+?)(v\d+)?$", _t(e, "a:id"))
        abstract = _t(e, "a:summary")
        if not m or len(abstract) < MIN_ABSTRACT:
            continue
        cats = [c.get("term") for c in e.findall("a:category", _NS) if c.get("term")]
        prim = e.find("x:primary_category", _NS)
        out.append({"paper_id": m.group(1), "title": _t(e, "a:title"), "abstract": abstract,
                    "authors": ", ".join(_t(a, "a:name") for a in e.findall("a:author", _NS))[:2000],
                    "year": int(_t(e, "a:published")[:4] or 0), "doi": _t(e, "x:doi") or None,
                    "journal_ref": _t(e, "x:journal_ref") or None, "categories": cats or ["arxiv"],
                    "primary_category": prim.get("term") if prim is not None else (cats[0] if cats else "arxiv")})
    return out


def search(client, query: str, limit: int = 50) -> list[dict]:
    wait = 3.1 - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    r = client.get(API, params={"search_query": arxiv_query(query), "start": 0, "max_results": int(limit),
                                "sortBy": "relevance"})
    _last[0] = time.monotonic()
    r.raise_for_status()
    return parse(r.text)


def insert(pg, rows: list[dict], reason: str) -> list[str]:
    """Add papers the corpus does not hold yet (with embeddings). Returns the ids added."""
    from research_agent.embeddings import get_embedder, paper_text_for_embedding

    have = {r["paper_id"] for r in pg.execute("SELECT paper_id FROM papers WHERE paper_id = ANY(%s)",
                                              ([r["paper_id"] for r in rows] or [""],)).fetchall()}
    new = [r for r in rows if r["paper_id"] not in have]
    if not new:
        return []
    vecs = get_embedder().embed_documents([paper_text_for_embedding(r["title"], r["abstract"]) for r in new])
    with pg.cursor() as cur:
        for r, v in zip(new, vecs):
            cur.execute(
                "INSERT INTO papers (paper_id, source, title, abstract, authors, categories, primary_category, year, "
                "doi, journal_ref, license, health_reason, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (paper_id) DO NOTHING",
                (r["paper_id"], r.get("source", "arxiv"), r["title"], r["abstract"], r["authors"],
                 list(r.get("categories") or ["arxiv"]), r.get("primary_category") or "arxiv", int(r.get("year") or 0),
                 r.get("doi"), r.get("journal_ref"), r.get("license") or "arXiv preprint", reason[:200], v))
    return [r["paper_id"] for r in new]


def live_search(pg, query: str, limit: int = 50, from_year: int | None = None, client=None, label: str = "") -> dict:
    """Search arXiv and load what is found. Same return shape as ingestion/pubmed.live_search."""
    client = client or _http()
    from_year = from_year or settings.min_year
    rows = [r for r in search(client, query, limit) if not r["year"] or r["year"] >= from_year]
    added = insert(pg, rows, f"arxiv live: {(label or query)[:80]}")
    return {"query": query, "matched": len(rows), "with_abstract": len(rows), "loaded_open_access": len(added),
            "loaded_abstract_only": 0, "paper_ids": list(dict.fromkeys(r["paper_id"] for r in rows))}
