"""Live PubMed search: papers fetched during a run, when the AI judges the loaded corpus too thin.

The corpus is what was loaded beforehand (PMC open-access slices, arXiv). A question about arthritis in
Uganda can find almost nothing there and still have a literature. So a run may search PubMed itself:

    who decides   the AI (the discovery agent in an agent-chosen run; a sufficiency check after screening in a
                  systematic run), and it must say why and give the exact query
    what code     only PubMed through NCBI's official API, never the open web; at most LIVE_SEARCH_MAX_CALLS
      allows      searches and LIVE_SEARCH_MAX_PAPERS papers per run; off with LIVE_SEARCH=0
    what comes    a paper that is open access in PMC comes in through the PMC loader, with its licence checked
      in          and its full text available; any other paper comes in as its PubMed record, title and
                  abstract only, cited as [PMID12345]. Records with no abstract are skipped.

Fetched papers join the corpus for good (they are public literature), marked with the query that found
them, so later runs find them without fetching again. Within the run they are logged as identified by
"live PubMed search: <query>", so PRISMA shows the live search as its own source.

No tokens are spent fetching: searching, loading and embedding are code. Tokens are spent only when the new
papers are screened and read, at the same rate as any other paper.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from research_agent.config import settings

MIN_ABSTRACT = 80


def _text(node) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def search(client, query: str, from_year: int | None = None, limit: int = 200) -> list[str]:
    """PMIDs for a PubMed query, best match first."""
    term = f"({query})" + (f' AND ("{from_year}"[PDAT] : "3000"[PDAT])' if from_year else "")
    root = ET.fromstring(client.get("esearch", db="pubmed", term=term, retmax=int(limit), sort="relevance"))
    return [i.text for i in root.findall(".//IdList/Id") if i.text]


def parse(article) -> dict | None:
    """One <PubmedArticle> to a record, or None without a usable abstract."""
    cit = article.find("MedlineCitation")
    if cit is None:
        return None
    pmid = _text(cit.find("PMID"))
    art = cit.find("Article")
    if not pmid or art is None:
        return None
    title = _text(art.find("ArticleTitle"))
    parts = []
    for a in art.findall(".//Abstract/AbstractText"):
        label, text = a.get("Label"), _text(a)
        if text:
            parts.append(f"{label}: {text}" if label else text)
    abstract = " ".join(parts)
    if not title or len(abstract) < MIN_ABSTRACT:
        return None
    authors = []
    for au in art.findall(".//AuthorList/Author"):
        last, fore = _text(au.find("LastName")), _text(au.find("ForeName"))
        if last:
            authors.append(f"{fore} {last}".strip())
    year_txt = _text(art.find(".//Journal/JournalIssue/PubDate/Year")) or \
        _text(art.find(".//Journal/JournalIssue/PubDate/MedlineDate"))[:4]
    year = int(year_txt) if year_txt[:4].isdigit() else 0
    ids = {i.get("IdType"): _text(i) for i in article.findall(".//PubmedData/ArticleIdList/ArticleId")}
    pmcid = ids.get("pmc") or ""
    return {"pmid": pmid, "pmcid": pmcid.upper() if pmcid.upper().startswith("PMC") else "",
            "title": title, "abstract": abstract, "authors": ", ".join(authors[:40]), "year": year,
            "journal": _text(art.find(".//Journal/Title")), "doi": ids.get("doi"),
            "types": [_text(t) for t in art.findall(".//PublicationTypeList/PublicationType")]}


def fetch(client, pmids: list[str]) -> list[dict]:
    out = []
    for start in range(0, len(pmids), 200):
        xml = client.post("efetch", db="pubmed", id=",".join(pmids[start:start + 200]), retmode="xml")
        out += [r for r in (parse(a) for a in ET.fromstring(xml).iter("PubmedArticle")) if r]
    return out


def live_search(pg, query: str, limit: int = 200, from_year: int | None = None, client=None,
                label: str = "") -> dict:
    """Search PubMed and load what is found. Returns the paper ids in search order (PMC ids for open-access
    papers, PMID<n> for the rest) and what was loaded."""
    from research_agent.embeddings import get_embedder, paper_text_for_embedding
    from research_agent.ingestion import pmc

    client = client or pmc.PMCClient()
    from_year = from_year or settings.min_year
    pmids = search(client, query, from_year, limit)
    records = fetch(client, pmids) if pmids else []
    have = {r["paper_id"] for r in pg.execute(
        "SELECT paper_id FROM papers WHERE paper_id = ANY(%s)",
        ([f"PMID{r['pmid']}" for r in records] + [r["pmcid"] for r in records if r["pmcid"]] or [""],)).fetchall()}
    # open-access PMC versions first: full text and a checked licence
    want_pmc = [r["pmcid"] for r in records if r["pmcid"] and r["pmcid"] not in have]
    pmc_rows = {}
    if want_pmc:
        try:
            for row in pmc.fetch_ids(client, want_pmc):
                if pmc.licence_allows_reuse(row["license"]) or pmc.licence_is_noncommercial(row["license"]):
                    pmc_rows[row["paper_id"]] = row
        except Exception:
            pmc_rows = {}
    ids, new_rows = [], []
    for r in records:
        if r["pmcid"] and (r["pmcid"] in have or r["pmcid"] in pmc_rows):
            pid = r["pmcid"]
            if pid in pmc_rows and pid not in have:
                row = pmc_rows[pid]
                new_rows.append((pid, "pmc", row["title"], row["abstract"], row["authors"], row["categories"],
                                 row["year"] or r["year"], row["doi"] or r["doi"], row["journal_ref"] or r["journal"],
                                 row["license"]))
                have.add(pid)
        else:
            pid = f"PMID{r['pmid']}"
            if pid not in have:
                new_rows.append((pid, "pubmed", r["title"], r["abstract"], r["authors"], ["pubmed"], r["year"],
                                 r["doi"], r["journal"], "PubMed record: title and abstract only"))
                have.add(pid)
        if r["year"] and r["year"] < from_year:
            continue
        ids.append(pid)
    if new_rows:
        vecs = get_embedder().embed_documents([paper_text_for_embedding(t, a) for _p, _s, t, a, *_r in new_rows])
        reason = f"pubmed live: {(label or query)[:80]}"
        with pg.cursor() as cur:
            for (pid, src, title, abstract, authors, cats, year, doi, journal, lic), v in zip(new_rows, vecs):
                cur.execute(
                    "INSERT INTO papers (paper_id, source, title, abstract, authors, categories, primary_category, "
                    "year, doi, journal_ref, license, health_reason, embedding) VALUES "
                    "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (paper_id) DO NOTHING",
                    (pid, src, title, abstract, authors, list(cats or [src]), src, int(year or 0), doi, journal,
                     lic, reason, v))
    return {"query": query, "matched": len(pmids), "with_abstract": len(records),
            "loaded_open_access": sum(1 for r in new_rows if r[1] == "pmc"),
            "loaded_abstract_only": sum(1 for r in new_rows if r[1] == "pubmed"),
            "paper_ids": list(dict.fromkeys(ids))}


def pubmed_url(paper_id: str) -> str:
    m = re.match(r"^PMID(\d+)$", paper_id or "")
    return f"https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/" if m else ""
