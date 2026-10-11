"""Live Semantic Scholar search: papers across all disciplines and publishers, including journals neither PMC
nor arXiv holds (many African and engineering journals among them).

Each paper found is stored under the most useful id it has, so nothing is held twice:
    open access in PMC   loaded through the PMC loader (licence checked, full text available)
    a PubMed record      PMID<n>, read from its abstract (or a legal copy found later)
    an arXiv preprint    its arXiv id
    anything else        S2<corpus id>, cited as [S2123456789]
Papers without an abstract are skipped. Works without a key (a shared, slower allowance); a free key in
SEMANTIC_SCHOLAR_API_KEY raises it. Searching and loading cost no tokens.
"""
from __future__ import annotations

import os
import time

from research_agent.config import settings

API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,externalIds,authors,venue,journal,publicationTypes,openAccessPdf"
MIN_ABSTRACT = 80


def headers() -> dict:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    return {"x-api-key": key} if key else {}


def _http():
    import httpx

    return httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True,
                        headers={"User-Agent": "health-research-agent/1.0 (literature review)"})


def search(client, query: str, limit: int = 50, from_year: int | None = None) -> list[dict]:
    """Relevance search; retries when the shared allowance is busy (HTTP 429)."""
    params = {"query": " ".join((query or "").split())[:300], "limit": min(100, int(limit)), "fields": FIELDS}
    if from_year:
        params["year"] = f"{from_year}-"
    for wait in (0, 2, 5, 10):
        if wait:
            time.sleep(wait)
        r = client.get(f"{API}/paper/search", params=params, headers=headers())
        if r.status_code == 429:
            continue
        r.raise_for_status()
        return (r.json() or {}).get("data") or []
    raise RuntimeError("Semantic Scholar is busy (rate limited); set SEMANTIC_SCHOLAR_API_KEY for a larger allowance")


def _row(p: dict, pid: str, source: str) -> dict:
    j = p.get("journal") or {}
    return {"paper_id": pid, "source": source, "title": " ".join((p.get("title") or "").split()),
            "abstract": " ".join((p.get("abstract") or "").split()),
            "authors": ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:40]),
            "year": int(p.get("year") or 0), "doi": (p.get("externalIds") or {}).get("DOI"),
            "journal_ref": j.get("name") or p.get("venue") or None, "categories": [source],
            "primary_category": source,
            "license": {"pubmed": "PubMed record: title and abstract only", "arxiv": "arXiv preprint",
                        "s2": "Semantic Scholar record: title and abstract only"}[source]}


def live_search(pg, query: str, limit: int = 50, from_year: int | None = None, client=None, label: str = "") -> dict:
    """Search Semantic Scholar and load what is found. Same return shape as ingestion/pubmed.live_search."""
    from research_agent.ingestion import arxiv_live, pmc

    client = client or _http()
    from_year = from_year or settings.min_year
    found = [p for p in search(client, query, limit, from_year) if len((p.get("abstract") or "")) >= MIN_ABSTRACT
             and p.get("title")]
    ids, rows, want_pmc = [], [], {}
    for p in found:
        ext = p.get("externalIds") or {}
        if ext.get("PubMedCentral"):
            pid = "PMC" + str(ext["PubMedCentral"]).removeprefix("PMC")
            want_pmc[pid] = p
        elif ext.get("PubMed"):
            pid = f"PMID{ext['PubMed']}"
            rows.append(_row(p, pid, "pubmed"))
        elif ext.get("ArXiv"):
            pid = str(ext["ArXiv"])
            rows.append(_row(p, pid, "arxiv"))
        elif ext.get("CorpusId"):
            pid = f"S2{ext['CorpusId']}"
            rows.append(_row(p, pid, "s2"))
        else:
            continue
        ids.append(pid)
    have = {r["paper_id"] for r in pg.execute("SELECT paper_id FROM papers WHERE paper_id = ANY(%s)",
                                              (ids or [""],)).fetchall()}
    # open-access PMC versions come in through the PMC loader, with their licence checked; the others as records
    todo_pmc = [p for p in want_pmc if p not in have]
    loaded_pmc = 0
    if todo_pmc:
        try:
            for row in pmc.fetch_ids(pmc.PMCClient(), todo_pmc):
                if pmc.licence_allows_reuse(row["license"]) or pmc.licence_is_noncommercial(row["license"]):
                    rows.append({"paper_id": row["paper_id"], "source": "pmc", "title": row["title"],
                                 "abstract": row["abstract"], "authors": row["authors"],
                                 "categories": row["categories"], "primary_category": "pmc",
                                 "year": row["year"], "doi": row["doi"], "journal_ref": row["journal_ref"],
                                 "license": row["license"]})
                    loaded_pmc += 1
        except Exception:
            pass
        for pid in todo_pmc:     # not reusable from PMC: keep it as a record, by PMID if it has one
            if not any(r["paper_id"] == pid for r in rows):
                p = want_pmc[pid]
                ext = p.get("externalIds") or {}
                alt = f"PMID{ext['PubMed']}" if ext.get("PubMed") else f"S2{ext.get('CorpusId')}"
                rows.append(_row(p, alt, "pubmed" if ext.get("PubMed") else "s2"))
                ids[ids.index(pid)] = alt
    added = arxiv_live.insert(pg, rows, f"semantic scholar live: {(label or query)[:80]}")
    return {"query": query, "matched": len(found), "with_abstract": len(found),
            "loaded_open_access": sum(1 for r in rows if r["source"] in ("pmc", "arxiv") and r["paper_id"] in added),
            "loaded_abstract_only": sum(1 for r in rows if r["source"] in ("pubmed", "s2") and r["paper_id"] in added),
            "paper_ids": list(dict.fromkeys(i for i in ids if i in have or i in added))}
