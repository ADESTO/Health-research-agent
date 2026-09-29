"""Citation graph from OpenAlex (https://openalex.org). Works without a key ($0.10/day of free usage); a free
key in OPENALEX_API_KEY raises that to $1/day, far more than runs use.

Papers are matched to OpenAlex works by DOI (arXiv papers through their arXiv DOI, 10.48550/arXiv.<id>).
One request covers 50 papers, so the citation links AMONG the analysed papers cost a handful of requests: a
paper A cites B when B's OpenAlex id is in A's reference list. Following references outwards (snowballing)
costs more requests and only happens when an agent asks for it.

Everything is cached in `openalex_works`; nothing here is needed for a run to succeed, and a network problem
only means the citation parts are left out.
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx

from research_agent.tools.base import INT, STR, STRS, Tool, obj

API = "https://api.openalex.org"
BATCH = 50
_transport: httpx.BaseTransport | None = None     # tests plug a mock transport in here


class OpenAlexError(RuntimeError):
    pass


class OpenAlexClient:
    def __init__(self, transport: httpx.BaseTransport | None = None, retries: int = 3):
        self._http = httpx.Client(timeout=httpx.Timeout(30.0, connect=15.0), transport=transport or _transport)
        self._retries = retries
        self._last = 0.0

    def get(self, path: str, **params) -> dict:
        key = (os.getenv("OPENALEX_API_KEY") or "").strip()
        if key:
            params["api_key"] = key          # free key: $1/day of usage instead of $0.10 without one
        mail = os.getenv("OPENALEX_EMAIL") or os.getenv("PMC_EMAIL")
        if mail:
            params["mailto"] = mail          # harmless; older OpenAlex setups used it for the polite pool
        for attempt in range(self._retries + 1):
            gap = 0.12 - (time.time() - self._last)
            if gap > 0:
                time.sleep(gap)
            try:
                r = self._http.get(f"{API}/{path}", params=params)
            except httpx.TransportError as exc:
                self._last = time.time()
                if attempt == self._retries:
                    raise OpenAlexError(f"OpenAlex unreachable: {exc}") from exc
                time.sleep(2 ** attempt)
                continue
            self._last = time.time()
            if r.status_code in (429, 500, 502, 503) and attempt < self._retries:
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                raise OpenAlexError(f"OpenAlex error {r.status_code}: {r.text[:200]}")
            return r.json()
        raise OpenAlexError("OpenAlex: gave up after retries")


def _short(oa_id: str | None) -> str | None:
    return oa_id.rsplit("/", 1)[-1] if oa_id else None


def _norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I).lower()
    return d or None


def paper_doi(row: dict) -> str | None:
    if row.get("doi"):
        return _norm_doi(row["doi"])
    if row.get("source") == "arxiv":
        base_id = re.sub(r"v\d+$", "", row["paper_id"]).lower()   # arXiv DOIs carry no version
        return f"10.48550/arxiv.{base_id}"
    return None


def resolve(pg, paper_ids: list[str], client: OpenAlexClient | None = None) -> dict:
    """Look up papers not yet cached. Returns {'looked_up', 'found'}; raises OpenAlexError on network failure."""
    have = {r["paper_id"] for r in pg.execute("SELECT paper_id FROM openalex_works WHERE paper_id = ANY(%s)",
                                              (list(paper_ids),)).fetchall()}
    todo = [p for p in dict.fromkeys(paper_ids) if p not in have]
    if not todo:
        return {"looked_up": 0, "found": 0}
    rows = pg.execute("SELECT paper_id, source, doi FROM papers WHERE paper_id = ANY(%s)", (todo,)).fetchall()
    by_doi = {}
    for r in rows:
        d = paper_doi(r)
        if d:
            by_doi[d] = r["paper_id"]
    client = client or OpenAlexClient()
    found = 0
    dois = list(by_doi)
    for i in range(0, len(dois), BATCH):
        chunk = dois[i:i + BATCH]
        data = client.get("works", filter="doi:" + "|".join(chunk), select="id,doi,cited_by_count,referenced_works",
                          **{"per-page": BATCH})
        for w in data.get("results", []):
            pid = by_doi.get(_norm_doi(w.get("doi")))
            if not pid:
                continue
            pg.execute("INSERT INTO openalex_works (paper_id, openalex_id, cited_by_count, referenced_works) "
                       "VALUES (%s,%s,%s,%s::jsonb) ON CONFLICT (paper_id) DO UPDATE SET openalex_id=EXCLUDED.openalex_id, "
                       "cited_by_count=EXCLUDED.cited_by_count, referenced_works=EXCLUDED.referenced_works, fetched_at=now()",
                       (pid, _short(w["id"]), w.get("cited_by_count") or 0,
                        json.dumps([_short(x) for x in w.get("referenced_works") or []])))
            found += 1
    for pid in todo:   # remember misses too, so they are not looked up again every time
        pg.execute("INSERT INTO openalex_works (paper_id) VALUES (%s) ON CONFLICT DO NOTHING", (pid,))
    return {"looked_up": len(todo), "found": found}


def ensure(ctx) -> str | None:
    """Resolve the run's papers once. Returns an error message instead of raising."""
    try:
        resolve(ctx.pg, ctx.shortlist_ids())
        return None
    except Exception as exc:
        return f"citation data unavailable: {str(exc)[:160]}"


def _works(pg, paper_ids: list[str]) -> dict[str, dict]:
    return {r["paper_id"]: r for r in pg.execute(
        "SELECT paper_id, openalex_id, cited_by_count, referenced_works FROM openalex_works "
        "WHERE paper_id = ANY(%s) AND openalex_id IS NOT NULL", (list(paper_ids),)).fetchall()}


def citation_graph(ctx, fetch: bool = True) -> dict:
    """Citations among the analysed papers, and each paper's citation count across all literature."""
    problem = ensure(ctx) if fetch else None
    ids = ctx.shortlist_ids()
    works = _works(ctx.pg, ids)
    oa_to_pid = {w["openalex_id"]: pid for pid, w in works.items()}
    edges = [(pid, oa_to_pid[ref]) for pid, w in works.items() for ref in (w["referenced_works"] or [])
             if ref in oa_to_pid and oa_to_pid[ref] != pid]
    cited_in_set: dict[str, int] = {}
    cites_in_set: dict[str, int] = {}
    for a, b in edges:
        cited_in_set[b] = cited_in_set.get(b, 0) + 1
        cites_in_set[a] = cites_in_set.get(a, 0) + 1
    titles = {r["paper_id"]: (r["title"], r["year"]) for r in ctx.pg.execute(
        "SELECT paper_id, title, year FROM papers WHERE paper_id = ANY(%s)", (ids,)).fetchall()}
    central = sorted(cited_in_set, key=lambda p: (-cited_in_set[p], -(works[p]["cited_by_count"] or 0)))
    isolated = [p for p in works if not cited_in_set.get(p) and not cites_in_set.get(p)]
    return {"papers": len(ids), "matched_in_openalex": len(works), "links_between_papers": len(edges),
            "most_cited_within_set": [{"paper_id": p, "title": titles.get(p, ("", ""))[0], "year": titles.get(p, ("", ""))[1],
                                       "cited_by_papers_in_set": cited_in_set[p],
                                       "cited_by_all_literature": works[p]["cited_by_count"]} for p in central[:10]],
            "most_cited_overall": [{"paper_id": p, "title": titles.get(p, ("", ""))[0], "cited_by_all_literature": w["cited_by_count"]}
                                   for p, w in sorted(works.items(), key=lambda kv: -(kv[1]["cited_by_count"] or 0))[:10]],
            "isolated": len(isolated), "edges": edges[:500],
            **({"problem": problem} if problem else {})}


def snowball(ctx, paper_ids: list[str], direction: str = "both", limit: int = 20) -> dict:
    """Papers in THIS corpus that the given papers cite (backward) or that cite them (forward), and are not yet
    shortlisted. Found papers are logged as identified for the review record."""
    from research_agent.tools.review import log_identified

    problem = ensure(ctx)
    if problem:
        return {"error": problem}
    try:
        resolve(ctx.pg, paper_ids)
    except Exception as exc:
        return {"error": f"citation data unavailable: {str(exc)[:160]}"}
    works = _works(ctx.pg, paper_ids)
    client = OpenAlexClient()
    candidates: dict[str, set] = {}
    try:
        if direction in ("backward", "both"):
            refs = sorted({r for w in works.values() for r in (w["referenced_works"] or [])})
            for i in range(0, min(len(refs), 400), BATCH):
                data = client.get("works", filter="openalex:" + "|".join(refs[i:i + BATCH]), select="id,doi",
                                  **{"per-page": BATCH})
                for w in data.get("results", []):
                    if w.get("doi"):
                        candidates.setdefault(_norm_doi(w["doi"]), set()).add("cited by a shortlisted paper")
        if direction in ("forward", "both"):
            for pid, w in list(works.items())[:25]:
                data = client.get("works", filter=f"cites:{w['openalex_id']}", select="id,doi", **{"per-page": 100})
                for x in data.get("results", []):
                    if x.get("doi"):
                        candidates.setdefault(_norm_doi(x["doi"]), set()).add(f"cites {pid}")
    except OpenAlexError as exc:
        return {"error": str(exc)}
    shortlisted = set(ctx.shortlist_ids())
    dois = [d for d in candidates if d]
    local = []
    if dois:
        arxiv = {d.split("arxiv.", 1)[1]: d for d in dois if d.startswith("10.48550/arxiv.")}
        rows = ctx.pg.execute(
            """SELECT paper_id, source, title, year, doi FROM papers
               WHERE lower(doi) = ANY(%s) OR (source = 'arxiv' AND lower(paper_id) = ANY(%s))""",
            (dois, list(arxiv))).fetchall()
        for r in rows:
            if r["paper_id"] in shortlisted:
                continue
            d = paper_doi(r)
            local.append({"paper_id": r["paper_id"], "source": r["source"], "title": r["title"], "year": r["year"],
                          "how_found": sorted(candidates.get(d, set()) or candidates.get(_norm_doi(r["doi"]), set()))})
    local = local[:max(1, min(int(limit), 60))]
    log_identified(ctx, [x["paper_id"] for x in local], "citation snowballing")
    return {"found_in_corpus": local, "linked_works_outside_corpus": max(0, len(dois) - len(local)),
            "note": "Read titles; add only papers that answer the question (add_to_shortlist)."}


def citation_info(ctx, paper_id: str) -> dict:
    g = citation_graph(ctx)
    cites = [b for a, b in g["edges"] if a == paper_id]
    cited_by = [a for a, b in g["edges"] if b == paper_id]
    w = _works(ctx.pg, [paper_id]).get(paper_id)
    return {"paper_id": paper_id, "in_openalex": bool(w), "cited_by_all_literature": (w or {}).get("cited_by_count"),
            "cites_papers_in_set": cites, "cited_by_papers_in_set": cited_by}


def citation_markdown(ctx) -> list[str]:
    """Code-built report section from cached data only (no network at report time)."""
    g = citation_graph(ctx, fetch=False)
    if not g["matched_in_openalex"]:
        return []
    from research_agent.agents.report import _cite

    lines = ["## Citation structure (computed)", "",
             f"{g['matched_in_openalex']} of the {g['papers']} analysed papers were matched in OpenAlex; "
             f"they cite each other {g['links_between_papers']} times, and {g['isolated']} neither cite nor are "
             "cited by another analysed paper.", ""]
    if g["most_cited_within_set"]:
        lines += ["Most cited within this literature (a sign of the studies it builds on):", ""]
        for p in g["most_cited_within_set"][:6]:
            lines.append(f"- {_cite(p['paper_id'])} {p['title']} ({p['year']}): cited by {p['cited_by_papers_in_set']} "
                         f"analysed papers, {p['cited_by_all_literature']} citations overall")
        lines.append("")
    return lines


CITATION_TOOLS = [
    Tool("snowball", "Follow citations from given papers: papers in this corpus that they cite (backward) or "
         "that cite them (forward), not yet shortlisted. Catches studies keyword searches miss.",
         obj({"paper_ids": STRS, "direction": {"type": "string", "enum": ["backward", "forward", "both"]},
              "limit": INT}, ["paper_ids"]), snowball, read_only=True, max_chars=12000),
    Tool("citation_graph", "Citations among the analysed papers: the most cited within this literature and "
         "overall, and how connected the literature is.", obj({}), citation_graph, read_only=True, max_chars=12000),
]
CITATION_INFO_TOOL = Tool("citation_info", "One paper's citations: which analysed papers it cites and which cite it, "
                          "and its citation count overall.", obj({"paper_id": STR}, ["paper_id"]), citation_info,
                          read_only=True)
