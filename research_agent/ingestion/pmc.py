"""PubMed Central ingestion: topic slices of the PMC Open Access Subset.

arXiv gives us preprints from the computational side of health research. PMC gives the published
clinical and epidemiological literature that arXiv structurally lacks, which is exactly what every
report so far had to caveat away.

We pull slices rather than the whole subset: an E-utilities search for a topic, then metadata and
abstracts for the hits. Full text is fetched lazily for shortlisted papers, as on the arXiv side.
Only licences that allow commercial reuse are kept (CC0 / CC BY / CC BY-SA), and the licence is
stored per paper.

    python -m research_agent.cli pmc-ingest "malaria forecasting" --from-year 2010

E-utilities allows 3 requests/second, or 10 with an API key (NCBI_API_KEY in .env); set
PMC_EMAIL/PMC_TOOL so NCBI can contact you about heavy use.
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import httpx

from research_agent.config import settings

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_FILTER = '"open access"[filter]'
# Licences that permit commercial reuse. CC BY-NC*, "no-cc" and unspecified licences are skipped.
# Licence classification. We look for explicit Creative Commons signals only: a CC URL, a CC name,
# or PMC's licence-type attribute. A bare "by" (as in "(c) 2020 by the authors") proves nothing.
_NC = re.compile(r"creativecommons\.org/licenses/by-nc|\bcc[-\s]?by[-\s]?nc|\bby-nc\b|"
                 r"attribution[-\s]?non[-\s]?commercial|\bnon[-\s]?commercial\b", re.I)
_CC_REUSE = re.compile(r"creativecommons\.org/(licenses/by(-sa|-nd)?/|publicdomain/(zero|mark))|\bcc0\b|"
                       r"\bcc[-\s]?by\b(?![-\s]?nc)|creative\s+commons\s+attribution\b(?![-\s]*non)|"
                       r"public\s+domain\s+(dedication|mark)", re.I)
# PMC's license-type attribute comes first in the stored string, e.g. "cc-by https://…" or "by-sa …"
_TYPE_ATTR = re.compile(r"(cc[-\s]?)?by(-sa|-nd)?(\s|$)", re.I)


class PMCError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


def _key() -> str | None:
    return os.getenv("NCBI_API_KEY") or os.getenv("PMC_API_KEY")


class PMCClient:
    """Thin E-utilities client with the rate limit NCBI asks for, and retries for transient errors."""

    def __init__(self, transport: httpx.BaseTransport | None = None, retries: int = 4):
        self._http = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0), transport=transport)
        self._min_gap = 0.11 if _key() else 0.34   # 10/s with a key, 3/s without
        self._last = 0.0
        self._retries = retries

    def _request(self, method: str, endpoint: str, params: dict) -> str:
        params = {"db": "pmc", "tool": os.getenv("PMC_TOOL", "health-research-agent"),
                  "email": os.getenv("PMC_EMAIL", ""), **params}
        if _key():
            params["api_key"] = _key()
        params = {k: v for k, v in params.items() if v != ""}
        for attempt in range(self._retries + 1):
            gap = self._min_gap - (time.time() - self._last)
            if gap > 0:
                time.sleep(gap)
            try:
                if method == "POST":   # long id lists go in the body, as NCBI recommends
                    r = self._http.post(f"{EUTILS}/{endpoint}.fcgi", data=params)
                else:
                    r = self._http.get(f"{EUTILS}/{endpoint}.fcgi", params=params)
            except httpx.TransportError as exc:
                self._last = time.time()
                if attempt == self._retries:
                    raise PMCError(f"PMC {endpoint}: network error {exc}") from exc
                time.sleep(2 ** attempt)
                continue
            self._last = time.time()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self._retries:
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                hint = " (NCBI throttles unauthenticated clients; set NCBI_API_KEY)" if r.status_code == 429 else ""
                raise PMCError(f"PMC {endpoint} error {r.status_code}: {r.text[:300]}{hint}",
                               status_code=r.status_code, response=r)
            return r.text
        raise PMCError(f"PMC {endpoint}: gave up after {self._retries} retries")

    def get(self, endpoint: str, **params) -> str:
        return self._request("GET", endpoint, params)

    def post(self, endpoint: str, **params) -> str:
        return self._request("POST", endpoint, params)


MAX_IDS = 9999   # esearch returns at most this many ids per request


def _date_term(query: str, start: date, end: date, field: str = "PDAT", extra: str = "") -> str:
    return (f'({query}) AND {OA_FILTER} AND ("{start:%Y/%m/%d}"[{field}] : "{end:%Y/%m/%d}"[{field}])'
            + extra)


def _ids_in_range(client: PMCClient, query: str, start: date, end: date, log=print,
                  field: str = "PDAT", extra: str = "") -> list[str]:
    """Every id matching the query in a date range, by halving the range until each search fits under the
    cap esearch will return."""
    root = ET.fromstring(client.get("esearch", term=_date_term(query, start, end, field, extra), retmax=MAX_IDS))
    count = int(root.findtext("Count") or 0)
    ids = [i.text for i in root.findall(".//IdList/Id") if i.text]
    if count <= MAX_IDS:
        return ids
    if start < end:
        mid = start + (end - start) // 2
        return (_ids_in_range(client, query, start, mid, log, field, extra)
                + _ids_in_range(client, query, mid + timedelta(days=1), end, log, field, extra))
    if field == "PDAT":
        # One publication date over the cap. PubMed dates an article with no stated day to 1 January, so a
        # single "day" can hold a whole year of them. Split that day again by the date each record entered
        # PubMed, which is spread out, instead of losing everything past the cap.
        day = f' AND ("{start:%Y/%m/%d}"[PDAT] : "{start:%Y/%m/%d}"[PDAT])'
        return _ids_in_range(client, query, date(1900, 1, 1), date.today(), log, field="CRDT", extra=day)
    log(f"  warning: {count:,} matches on {start} ({field}); only the first {MAX_IDS:,} are reachable. "
        "Narrow the search (add terms, or ingest a few years at a time).")
    return ids


def search_ids(client: PMCClient, query: str, from_year: int | None = None, to_year: int | None = None,
               log=print) -> list[str]:
    """Every PMCID matching the query, as a plain list.

    We hold the ids ourselves instead of paging through NCBI's history server: a history session
    expires when a slow ingest (embedding on a CPU) takes too long, and history paging stops at about
    10,000 results. Ids never expire, and date ranges are split until each search fits under the cap."""
    start = date(from_year or 1900, 1, 1)
    end = date(to_year or date.today().year, 12, 31)
    ids = _ids_in_range(client, query, start, end, log)
    return list(dict.fromkeys(i.upper() if i.upper().startswith("PMC") else f"PMC{i}" for i in ids))


def year_counts(client: PMCClient, from_year: int, to_year: int) -> list[tuple[int, int]]:
    """How many open-access PMC articles exist per year: the denominator for PMC trend figures."""
    out = []
    for year in range(from_year, to_year + 1):
        xml = client.get("esearch", term=f'{OA_FILTER} AND ("{year}"[PDAT] : "{year}"[PDAT])', retmax=0)
        out.append((year, int(ET.fromstring(xml).findtext("Count") or 0)))
    return out


# ---------------------------------------------------------------- JATS parsing
def _text(node) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _licence(article) -> str:
    lic = article.find(".//permissions/license")
    if lic is None:
        return ""
    bits = [lic.get("license-type") or "", lic.get("{http://www.w3.org/1999/xlink}href") or ""]
    link = lic.find(".//ext-link")
    if link is not None:
        bits.append(link.get("{http://www.w3.org/1999/xlink}href") or "")
    bits.append(_text(lic)[:800])   # the CC sentence often follows a long copyright line
    return " ".join(b for b in bits if b).strip()


def licence_kind(licence: str) -> str:
    """'reusable' (CC0 / CC BY / BY-SA / BY-ND), 'noncommercial' (CC BY-NC*), or 'unknown'."""
    if not licence:
        return "unknown"
    if _NC.search(licence):
        return "noncommercial"
    if _CC_REUSE.search(licence) or _TYPE_ATTR.match(licence.strip()):
        return "reusable"
    return "unknown"


def licence_allows_reuse(licence: str) -> bool:
    """True for CC0 / CC BY / CC BY-SA / CC BY-ND. Non-commercial and unrecognised licences are left out."""
    return licence_kind(licence) == "reusable"


def licence_is_noncommercial(licence: str) -> bool:
    """A stated Creative Commons non-commercial licence (CC BY-NC, BY-NC-SA, BY-NC-ND)."""
    return licence_kind(licence) == "noncommercial"


def audit_licences(pg, remove: bool = False, allow_noncommercial: bool = False) -> dict:
    """Re-check the licence of every PMC paper already loaded, with the current rules.

    Papers loaded before a rule changed may no longer qualify. With remove=True they are deleted,
    except any a run has already analysed (those are kept and listed, so no report loses its source)."""
    rows = pg.execute("SELECT paper_id, license FROM papers WHERE source='pmc'").fetchall()
    kinds: dict[str, list[str]] = {"reusable": [], "noncommercial": [], "unknown": []}
    for r in rows:
        kinds[licence_kind(r["license"] or "")].append(r["paper_id"])
    failing = kinds["unknown"] + ([] if allow_noncommercial else kinds["noncommercial"])
    used = {r["paper_id"] for r in pg.execute(
        "SELECT DISTINCT paper_id FROM run_papers WHERE paper_id = ANY(%s)", (failing,)).fetchall()}
    removed = 0
    if remove and failing:
        removable = [p for p in failing if p not in used]
        removed = pg.execute("DELETE FROM papers WHERE paper_id = ANY(%s)", (removable,)).rowcount
    return {"checked": len(rows), **{k: len(v) for k, v in kinds.items()}, "failing": len(failing),
            "removed": removed, "kept_because_used_in_runs": sorted(used)}


def parse_article(article) -> dict | None:
    """One <article> element to a paper row, or None when it has no abstract or no usable id."""
    front = article.find("front")
    if front is None:
        return None
    ids = {i.get("pub-id-type"): _text(i) for i in front.findall(".//article-id")}
    pmcid = ids.get("pmcid") or ids.get("pmc") or ""
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = "PMC" + pmcid
    title = _text(front.find(".//title-group/article-title"))
    abstract_nodes = [a for a in front.findall(".//abstract") if a.get("abstract-type") != "graphical"]
    abstract = " ".join(_text(a) for a in abstract_nodes).strip()
    if not pmcid or not title or len(abstract) < 80:
        return None
    authors = []
    for c in front.findall(".//contrib[@contrib-type='author']"):
        surname, given = _text(c.find(".//surname")), _text(c.find(".//given-names"))
        if surname:
            authors.append(f"{given} {surname}".strip())
    # Earliest publication date. A "collection" date is the issue the article is filed under and can be
    # a year later than publication, so it is only used when nothing else is given.
    published, other = [], []
    for d in front.findall(".//pub-date"):
        y = _text(d.find("year"))
        if y.isdigit():
            kind = d.get("pub-type") or d.get("date-type") or ""
            (other if kind == "collection" else published).append(int(y))
    year = min(published or other or [0])
    journal = _text(front.find(".//journal-title"))
    subjects = [_text(s) for s in front.findall(".//subj-group/subject") if _text(s)]
    return {
        "paper_id": pmcid.upper(),
        "title": " ".join(title.split()),
        "abstract": " ".join(abstract.split()),
        "authors": ", ".join(authors[:40]),
        "categories": subjects[:10] or ["pmc"],
        "primary_category": "pmc",
        "year": year,
        "doi": ids.get("doi"),
        "journal_ref": journal,
        "license": _licence(article),
    }


def fetch_ids(client: PMCClient, pmcids: list[str]) -> list[dict]:
    """Metadata and abstracts for these PMCIDs (up to a few hundred per call)."""
    xml = client.post("efetch", id=",".join(p.removeprefix("PMC") for p in pmcids), retmode="xml")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise PMCError(f"could not parse PMC response for {pmcids[0]}…: {exc}") from exc
    return [row for row in (parse_article(a) for a in root.iter("article")) if row]


# ---------------------------------------------------------------- ingest
def ingest(query: str, limit: int | None = None, from_year: int | None = None, to_year: int | None = None,
           batch_size: int = 200, log=print, client: PMCClient | None = None,
           include_noncommercial: bool = False) -> dict:
    """Load one topic slice of the PMC open-access subset into `papers` with source='pmc'.

    By default only licences that allow commercial reuse are kept. include_noncommercial=True also
    keeps CC BY-NC and similar: fine for private research, and the licence is stored per paper so
    reports can mark them. Articles with no stated licence are skipped either way."""
    from research_agent.db import connect, init_schema
    from research_agent.embeddings import get_embedder, paper_text_for_embedding

    init_schema()
    client = client or PMCClient()
    from_year = from_year or settings.min_year
    ids = search_ids(client, query, from_year, to_year, log=log)
    pg = connect()
    try:
        existing = {r["paper_id"] for r in pg.execute(
            "SELECT paper_id FROM papers WHERE source='pmc'").fetchall()}
        todo = [i for i in ids if i not in existing]
        already = len(ids) - len(todo)
        if limit:
            todo = todo[:limit]
        log(f"PMC: {len(ids):,} open-access articles match; {already:,} already loaded; fetching {len(todo):,}"
            + (" (including non-commercial licences)" if include_noncommercial else ""))
        # denominators first, so trends work even if a long ingest is interrupted
        load_year_stats(pg, client, from_year, to_year, log=log)
        if not todo:
            return {"query": query, "matched": len(ids), "inserted": 0, "skipped_licence": 0}

        embedder = get_embedder()
        inserted = skipped_lic = 0
        t0 = time.time()
        cols = ["paper_id", "source", "title", "abstract", "authors", "categories", "primary_category",
                "year", "doi", "journal_ref", "license", "health_reason", "embedding"]
        for start in range(0, len(todo), batch_size):
            rows = fetch_ids(client, todo[start:start + batch_size])
            keep = []
            for r in rows:
                if r["paper_id"] in existing:
                    continue
                if not (licence_allows_reuse(r["license"])
                        or (include_noncommercial and licence_is_noncommercial(r["license"]))):
                    skipped_lic += 1
                elif r["year"] >= from_year:
                    keep.append(r)
                    existing.add(r["paper_id"])
            if keep:
                vecs = embedder.embed_documents(
                    [paper_text_for_embedding(r["title"], r["abstract"]) for r in keep])
                with pg.cursor() as cur, cur.copy(f"COPY papers ({', '.join(cols)}) FROM STDIN") as cp:
                    for r, v in zip(keep, vecs):
                        cp.write_row([r["paper_id"], "pmc", r["title"], r["abstract"], r["authors"],
                                      r["categories"], r["primary_category"], r["year"], r["doi"],
                                      r["journal_ref"], r["license"], f"pmc:{query[:80]}", v])
                inserted += len(keep)
            done = min(start + batch_size, len(todo))
            log(f"  {inserted:,} loaded / {done:,} of {len(todo):,} read "
                f"({inserted / max(time.time() - t0, 1e-6):,.1f}/s, {skipped_lic:,} skipped on licence)")
        if inserted:
            # The vector and text indexes are updated row by row as papers are inserted, so no rebuild
            # is needed; refreshing planner statistics keeps searches fast after a large load.
            pg.execute("ANALYZE papers")
    finally:
        pg.close()
    return {"query": query, "matched": len(ids), "inserted": inserted, "skipped_licence": skipped_lic}


def load_year_stats(pg, client: PMCClient, from_year: int, to_year: int | None = None, log=print) -> int:
    """PMC's own per-year totals, so trend figures normalise against PMC rather than arXiv."""
    to_year = to_year or time.gmtime().tm_year
    have = {r["year"] for r in pg.execute(
        "SELECT year FROM corpus_year_stats WHERE source='pmc'").fetchall()}
    missing = [y for y in range(from_year, to_year + 1) if y not in have]
    if not missing:
        return 0
    log(f"  fetching PMC per-year totals for {len(missing)} year(s)…")
    for year, n in year_counts(client, min(missing), max(missing)):
        pg.execute("INSERT INTO corpus_year_stats (source, year, primary_category, n_papers) "
                   "VALUES ('pmc', %s, 'pmc', %s) ON CONFLICT (source, year, primary_category) "
                   "DO UPDATE SET n_papers = EXCLUDED.n_papers", (year, n))
    return len(missing)


# ---------------------------------------------------------------- full text
_JATS_DROP = {"table-wrap", "fig", "disp-formula", "inline-formula", "ref-list", "back", "graphic"}


def jats_sections(article) -> tuple[str, list[dict]]:
    """Body text of a JATS article as (clean_text, [{heading, text}])."""
    body = article.find("body")
    if body is None:
        return "", []
    for parent in article.iter():
        for child in list(parent):
            if child.tag in _JATS_DROP:
                parent.remove(child)
    sections = []
    for sec in body.findall("sec"):
        heading = _text(sec.find("title")) or "Section"
        paras = " ".join(_text(p) for p in sec.findall(".//p"))
        if paras.strip():
            sections.append({"heading": heading, "text": paras.strip()})
    if not sections:
        text = " ".join(_text(p) for p in body.findall(".//p")).strip()
        return text, ([{"heading": "Body", "text": text}] if text else [])
    return "\n\n".join(f"{s['heading']}\n{s['text']}" for s in sections), sections


def fetch_fulltext(pg, paper_ids: list[str], client: PMCClient | None = None) -> dict[str, str]:
    """Fetch and cache full text for PMC papers. Returns {paper_id: status}."""
    from research_agent.ingestion.fulltext import _json, assess

    client = client or PMCClient()
    out: dict[str, str] = {}
    for start in range(0, len(paper_ids), 20):
        chunk = paper_ids[start:start + 20]
        xml = client.get("efetch", id=",".join(chunk), retmode="xml")
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        for article in root.iter("article"):
            row = parse_article(article)
            pid = row["paper_id"] if row else ""
            if not pid:
                continue
            clean, sections = jats_sections(article)
            status = assess(clean, clean, row["title"]) if clean else "missing"
            pg.execute(
                "INSERT INTO paper_fulltext (paper_id, status, clean_text, sections, resolution, n_chars) "
                "VALUES (%s,%s,%s,%s::jsonb,'pmc-oa',%s) ON CONFLICT (paper_id) DO NOTHING",
                (pid, status, clean, _json(sections), len(clean)))
            out[pid] = status
    for pid in set(paper_ids) - set(out):
        pg.execute("INSERT INTO paper_fulltext (paper_id, status) VALUES (%s,'missing') "
                   "ON CONFLICT DO NOTHING", (pid,))
        out[pid] = "missing"
    return out
