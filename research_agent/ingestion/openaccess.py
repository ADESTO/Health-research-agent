"""Legal open-access copies of papers whose full text the corpus does not hold.

A paper chosen to be read in full may have no text in PMC (a PubMed record, a paywalled journal article, a
paper found through Semantic Scholar) or in the arXiv dataset (an arXiv paper posted after it). Before it is
read from its abstract alone, this module looks for a free, legal copy:

    arXiv papers        the PDF on arxiv.org
    papers with a DOI   Unpaywall, which knows where a legal copy is: the publisher's site, a university
                        repository (often the authors' accepted manuscript), or a preprint server
    Semantic Scholar    its open-access PDF link, for papers without a DOI

The copy is downloaded once, split into sections and cached in `paper_fulltext` like any other full text,
with where it came from recorded in `resolution`. A copy whose text does not contain most of the paper's
title is rejected (a wrong document, a cover page). A paper with no copy is marked as looked for, so it is
not looked up again. No API key is needed; Unpaywall asks for an email address (UNPAYWALL_EMAIL, or
PMC_EMAIL), and without one only arXiv and Semantic Scholar are tried. OPEN_ACCESS_LOOKUP=0 turns it off.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from research_agent.config import settings

MIN_CHARS = 3000          # less than this is a cover page, an abstract page or a failed extraction
LOOKED = "open access: none found"

_HEADING = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?(abstract|summary|background|introduction|methods?|materials and methods|"
    r"patients and methods|study design|data(?: sources| collection)?|statistical analysis|results|findings|"
    r"discussion|limitations|strengths and limitations|conclusions?|interpretation|references|bibliography|"
    r"acknowledg(?:e)?ments?)\s*:?\s*$", re.I)


def _http():
    import httpx

    return httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True,
                        headers={"User-Agent": "health-research-agent/1.0 (literature review; open-access lookup)"})


def email() -> str:
    return os.getenv("UNPAYWALL_EMAIL") or os.getenv("PMC_EMAIL") or ""


def to_sections(pages: list[dict]) -> list[dict]:
    """Page-by-page PDF text into the paper's own sections, so methods and results are found by heading.
    The reference list is dropped."""
    lines = "\n".join(p["text"] for p in pages).splitlines()
    out, head, buf = [], "Front matter", []
    for line in lines:
        m = _HEADING.match(line)
        if m and len(line.strip()) < 60:
            if buf:
                out.append({"heading": head, "text": "\n".join(buf).strip()})
            head, buf = m.group(1).strip().title(), []
        else:
            buf.append(line)
    if buf:
        out.append({"heading": head, "text": "\n".join(buf).strip()})
    out = [s for s in out if s["text"] and not re.match(r"references|bibliography|acknowledg", s["heading"], re.I)]
    return out if len(out) > 1 else [{"heading": p["heading"], "text": p["text"]} for p in pages]


def _title_found(title: str, text: str) -> bool:
    tokens = {t for t in re.findall(r"[a-z]{4,}", (title or "").lower())}
    if not tokens:
        return True
    head = text[:15000].lower()
    return sum(1 for t in tokens if t in head) / len(tokens) >= 0.5


def unpaywall(client, doi: str) -> dict | None:
    """The best legal open-access location Unpaywall knows for a DOI, or None."""
    if not doi or not email():
        return None
    r = client.get(f"https://api.unpaywall.org/v2/{doi.strip()}", params={"email": email()})
    if r.status_code != 200:
        return None
    loc = (r.json() or {}).get("best_oa_location") or {}
    url = loc.get("url_for_pdf") or loc.get("url")
    if not url:
        return None
    return {"url": url, "licence": loc.get("license") or "no licence stated",
            "where": f"{loc.get('host_type') or 'unknown host'}, {loc.get('version') or 'version not stated'}"}


def semantic_scholar_pdf(client, paper_id: str, doi: str | None) -> dict | None:
    from research_agent.ingestion import s2

    key = f"CorpusId:{paper_id[2:]}" if paper_id.startswith("S2") else (f"DOI:{doi}" if doi else None)
    if not key:
        return None
    r = client.get(f"{s2.API}/paper/{key}", params={"fields": "openAccessPdf"}, headers=s2.headers())
    if r.status_code != 200:
        return None
    pdf = (r.json() or {}).get("openAccessPdf") or {}
    return {"url": pdf["url"], "licence": pdf.get("license") or "no licence stated",
            "where": "Semantic Scholar open-access link"} if pdf.get("url") else None


def locate(client, paper: dict) -> dict | None:
    """Where a legal copy of this paper is, trying the sources in order."""
    pid, source, doi = paper["paper_id"], paper["source"], paper.get("doi")
    if source == "arxiv":
        return {"url": f"https://arxiv.org/pdf/{pid}", "licence": "arXiv", "where": "arXiv preprint"}
    try:
        found = unpaywall(client, doi)
    except Exception:
        found = None
    if found:
        return found
    try:
        return semantic_scholar_pdf(client, pid, doi)
    except Exception:
        return None


def fetch_copy(client, paper: dict) -> tuple[list[dict], str] | None:
    """(sections, where it came from) for a paper, or None when no usable legal copy exists."""
    from research_agent.uploads import MAX_BYTES, sections_of

    loc = locate(client, paper)
    if not loc:
        return None
    r = client.get(loc["url"])
    if r.status_code != 200 or len(r.content) > MAX_BYTES:
        return None
    ctype = (r.headers.get("content-type") or "").lower()
    name = "copy.pdf" if ("pdf" in ctype or r.content[:5] == b"%PDF-") else "copy.html"
    pages = sections_of(name, r.content)
    sections = to_sections(pages) if name.endswith(".pdf") else pages
    text = "\n".join(s["text"] for s in sections)
    raw = "\n".join(s["heading"] + "\n" + s["text"] for s in pages)
    if name.endswith(".html"):
        raw = re.sub(r"<[^>]+>", " ", r.content[:300000].decode("utf-8", errors="ignore")) + raw
    if len(text) < MIN_CHARS or not _title_found(paper.get("title") or "", raw):
        return None
    return sections, f"open access: {loc['where']}; {loc['licence']}; {loc['url']}"[:500]


def fill_missing(pg, paper_ids: list[str], client=None) -> dict[str, str]:
    """For papers whose full text is marked missing and not yet looked for, find and cache a legal copy.
    Returns {paper_id: new status} for the papers it handled."""
    if not settings.open_access_lookup or not paper_ids:
        return {}
    rows = pg.execute(
        "SELECT p.paper_id, p.source, p.doi, p.title FROM papers p JOIN paper_fulltext f USING (paper_id) "
        "WHERE p.paper_id = ANY(%s) AND f.status = 'missing' AND f.resolution IS NULL AND p.source <> 'upload'",
        (paper_ids,)).fetchall()
    if not rows:
        return {}
    client = client or _http()
    out: dict[str, str] = {}

    def one(p):
        try:
            return p, fetch_copy(client, p)
        except Exception:
            return p, None
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p, got in pool.map(one, rows):
            if got:
                sections, where = got
                clean = "\n\n".join(f"## {s['heading']}\n{s['text']}" for s in sections)
                pg.execute("UPDATE paper_fulltext SET status='ok', clean_text=%s, sections=%s::jsonb, resolution=%s, "
                           "n_chars=%s WHERE paper_id=%s",
                           (clean, json.dumps(sections, ensure_ascii=False), where, len(clean), p["paper_id"]))
                out[p["paper_id"]] = "ok"
            else:
                pg.execute("UPDATE paper_fulltext SET resolution=%s WHERE paper_id=%s", (LOOKED, p["paper_id"]))
    return out
