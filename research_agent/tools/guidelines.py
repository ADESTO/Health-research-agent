"""A reference library of clinical guidelines: what practice SHOULD be, kept apart from what studies report.

Guidelines are not evidence in the run's sense. A EULAR recommendation is a judgement made from studies, often
the same ones a run analyses, so counting it beside them would count those studies twice and let authority
stand in for data. So guidelines live in their own tables and never enter a count, a claim state, a gap or a
pooled figure. They are a benchmark: reports set what the run's studies report next to what the guidelines
recommend, and say where the two meet and where they part.

    add       a guideline from PubMed Central (by PMCID: EULAR and ACR recommendations are mostly open access)
              or from a file (PDF or text: national guidelines). Its text is split into chunks and an LLM
              records each recommendation in that chunk WORD FOR WORD, with its strength and evidence grade as
              written. Code checks every recommendation against the chunk it came from; one that is not found
              is dropped. Strength is kept only if its wording is in the chunk.
    search    recommendations matching words, for a condition, by code (full-text search).
    compare   for a finished run: each recommendation's interventions against the interventions the run's
              papers report (the quote-checked `interventions` field), counted over the papers that state any.
              A study reporting a drug is not a study following the guideline: the comparison says how often
              the recommended options appear in the literature, nothing more.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings
from research_agent.tools.base import STR, Tool, obj

CHUNK_CHARS = 12000
MAX_CHUNKS = 40                      # about 480,000 characters: a long national guideline, not a textbook
ACTIONS = ["recommend", "recommend_against", "consider", "not_specified"]
STRENGTHS = ["strong", "conditional", "weak", "expert_opinion", "good_practice", "not_stated"]

SYSTEM = """You record the recommendations in one part of a clinical guideline. Nothing else.

A recommendation is a statement the guideline makes about what should or should not be done: who it applies
to, what to do, and often how strongly. Background, rationale, research agenda and descriptions of studies are
not recommendations. Tables often hold the recommendations with their grades: read them.

For each recommendation in the text you are given:
- text: the recommendation copied WORD FOR WORD from the text, complete. Code checks it against the text and
  discards one that is not found. Do not paraphrase, merge or shorten.
- number: its number or label as written (e.g. 3, 2.1, R5), or ''.
- population: who it applies to, in a few words.
- interventions: what it is about, as short canonical names, 1 to 4 words each (e.g. methotrexate,
  glucocorticoids, NSAIDs, exercise therapy, weight loss, total knee replacement). Only what the text names.
- action: recommend, recommend_against, consider (may be considered, can be used), or not_specified.
- strength_as_written and evidence_as_written: the grade exactly as the guideline writes it (e.g. "strong",
  "conditional", "SoR A", "LoE 1b", "Grade B"), or '' when this part of the text does not give it.
- strength: your reading of strength_as_written as strong, conditional, weak, expert_opinion, good_practice
  or not_stated.

If this part of the guideline has no recommendations, return an empty list: that is a correct answer."""


def _tool() -> dict:
    rec = {"type": "object", "properties": {
        "text": STR, "number": STR, "population": STR,
        "interventions": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": ACTIONS},
        "strength_as_written": STR, "evidence_as_written": STR,
        "strength": {"type": "string", "enum": STRENGTHS}}, "required": ["text"]}
    return {"name": "record_recommendations", "description": "Record the recommendations in this text.",
            "input_schema": obj({"recommendations": {"type": "array", "items": rec}}, ["recommendations"])}


# ---------------------------------------------------------------- getting the text
def chunks_of(sections: list[dict], size: int = CHUNK_CHARS) -> list[str]:
    """The guideline as chunks of about `size` characters, cut at section boundaries where possible, so a
    recommendation and its grade usually arrive together."""
    out, cur = [], ""
    for s in sections:
        block = f"## {s['heading']}\n{s['text']}".strip()
        while len(block) > size:                       # one very long section: cut it at paragraph or sentence
            cut = max(block.rfind("\n", 0, size), block.rfind(". ", 0, size))
            cut = cut if cut > size // 2 else size
            if cur:
                out.append(cur); cur = ""
            out.append(block[:cut + 1]); block = block[cut + 1:]
        if len(cur) + len(block) + 2 > size and cur:
            out.append(cur); cur = ""
        cur = (cur + "\n\n" + block).strip()
    if cur:
        out.append(cur)
    return out[:MAX_CHUNKS]


def _jats(article) -> tuple[dict, list[dict]]:
    """Metadata and sections of a PMC article, TABLES INCLUDED: guideline recommendations usually sit in a
    table with their grades, which the paper reader drops."""
    from research_agent.ingestion.pmc import _licence, _text

    front = article.find("front")
    title = _text(front.find(".//title-group/article-title")) if front is not None else ""
    years = [int(_text(d.find("year"))) for d in (front.findall(".//pub-date") if front is not None else [])
             if _text(d.find("year")).isdigit()]
    meta = {"title": title, "year": min(years) if years else None, "licence": _licence(article),
            "journal": _text(front.find(".//journal-title")) if front is not None else ""}
    sections = []
    for tw in article.iter("table-wrap"):
        caption = _text(tw.find("caption")) or _text(tw.find("label")) or "Table"
        rows = []
        for tr in tw.iter("tr"):
            cells = [" ".join("".join(c.itertext()).split()) for c in tr if c.tag in ("td", "th")]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            sections.append({"heading": caption[:200], "text": "\n".join(rows)})
    body = article.find("body")
    for sec in (body.findall("sec") if body is not None else []):
        heading = _text(sec.find("title")) or "Section"
        paras = "\n".join(_text(p) for p in sec.findall(".//p") if _text(p))
        if paras:
            sections.append({"heading": heading, "text": paras})
    return meta, sections


def text_from_pmc(pmcid: str, client=None) -> tuple[dict, list[dict]]:
    import xml.etree.ElementTree as ET

    from research_agent.ingestion.pmc import PMCClient

    client = client or PMCClient()
    pid = pmcid.upper().removeprefix("PMC")
    root = ET.fromstring(client.get("efetch", id=pid, retmode="xml"))
    article = next(iter(root.iter("article")), None)
    if article is None:
        raise ValueError(f"PMC returned no article for {pmcid}")
    return _jats(article)


def text_from_file(path: str) -> list[dict]:
    """Sections of a PDF (one per page) or a text/Markdown file (split at headings)."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        pages = [(" ".join((p.extract_text() or "").split(" "))).strip() for p in PdfReader(path).pages]
        return [{"heading": f"Page {i}", "text": t} for i, t in enumerate(pages, 1) if t]
    raw = open(path, encoding="utf-8", errors="replace").read()
    parts = re.split(r"^#{1,3} +(.+)$", raw, flags=re.M)
    if len(parts) == 1:
        return [{"heading": "Text", "text": raw.strip()}]
    out = [{"heading": "Introduction", "text": parts[0].strip()}] if parts[0].strip() else []
    out += [{"heading": h.strip(), "text": t.strip()} for h, t in zip(parts[1::2], parts[2::2]) if t.strip()]
    return out


# ---------------------------------------------------------------- reading (LLM, checked by code)
def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def check_recommendation(rec: dict, chunk: str, words: list[str], shingles: set) -> dict | None:
    from research_agent.tools.extraction import quote_found

    text = " ".join(str(rec.get("text") or "").split())[:1500]
    if len(text) < 15 or not quote_found(text, words, shingles):
        return None
    flat = _norm(chunk)
    sw = " ".join(str(rec.get("strength_as_written") or "").split())[:60]
    ev = " ".join(str(rec.get("evidence_as_written") or "").split())[:60]
    sw_ok = bool(sw) and _norm(sw) in flat
    ev_ok = bool(ev) and _norm(ev) in flat
    rec_words = set(_norm(text).split())
    interventions = []
    for term in rec.get("interventions") or []:
        t = " ".join(str(term).split())[:60]
        tw = [w for w in _norm(t).split() if len(w) > 2]
        # the term must be about this recommendation: one of its words is in the recommendation's own text
        # (prefix match, so "glucocorticoids" counts for "glucocorticoid" and "csDMARDs" for "csdmard")
        if t and any(any(r.startswith(w[:6]) or w.startswith(r[:6]) for r in rec_words if len(r) > 2) for w in tw):
            interventions.append(t.lower())
    return {"text": text, "number": str(rec.get("number") or "")[:20],
            "population": str(rec.get("population") or "")[:160],
            "interventions": list(dict.fromkeys(interventions))[:8],
            "action": rec.get("action") if rec.get("action") in ACTIONS else "not_specified",
            "strength_as_written": sw if sw_ok else "", "evidence_as_written": ev if ev_ok else "",
            "strength": (rec.get("strength") if rec.get("strength") in STRENGTHS else "not_stated")
            if sw_ok else "not_stated"}


def _read_chunk(llm, chunk: str) -> tuple[list[dict], int]:
    from research_agent.tools.extraction import _words

    resp = llm.chat(SYSTEM, [{"role": "user", "content": [{"type": "text", "text": chunk}]}], tools=[_tool()],
                    force_tool="record_recommendations", max_tokens=max(settings.long_output_max_tokens, 4000))
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    if "_raw_arguments" in args:
        from research_agent.agents.base import repair_truncated_json

        args = repair_truncated_json(args["_raw_arguments"] or "") or {}
    words = _words(chunk)
    shingles = {tuple(words[i:i + k]) for k in (3, 4, 5) for i in range(len(words) - k + 1)}
    raw = [r for r in args.get("recommendations") or [] if isinstance(r, dict)]
    kept = [c for c in (check_recommendation(r, chunk, words, shingles) for r in raw) if c]
    return kept, len(raw) - len(kept)


def add_guideline(pg, llm_factory, title: str, issuer: str, year: int | None, conditions: list[str],
                  region: str, sections: list[dict], source: str, licence: str = "", url: str = "") -> dict:
    """Store a guideline and its checked recommendations. Returns what was kept and dropped."""
    chunks = chunks_of(sections)
    if not chunks:
        return {"error": "no text could be read from this guideline"}
    llm = llm_factory(step="guidelines")
    setattr(llm, "_step", "guidelines")
    results: dict[int, list[dict]] = {}
    dropped = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, settings.extraction_workers)) as pool:
        futures = {pool.submit(_read_chunk, llm, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            try:
                kept, gone = fut.result()
                results[futures[fut]] = kept
                dropped += gone
            except Exception:
                failed += 1
    recs, seen = [], set()
    for i in sorted(results):
        for r in results[i]:
            key = _norm(r["text"])[:300]
            if key not in seen:               # a recommendation repeated in a summary table is stored once
                seen.add(key)
                recs.append(r)
    gid = pg.execute(
        "INSERT INTO guidelines (title, issuer, year, conditions, region, source, licence, url, n_chars, "
        "chunks_read, chunks_failed) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (title[:300], issuer[:120], year, [c.lower().strip() for c in conditions if c.strip()], region[:80],
         source[:200], licence[:800], url[:400], sum(len(c) for c in chunks), len(chunks) - failed, failed)
    ).fetchone()["id"]
    with pg.cursor() as cur:
        for n, r in enumerate(recs, 1):
            cur.execute(
                "INSERT INTO guideline_recommendations (guideline_id, seq, number, text, population, interventions, "
                "action, strength, strength_as_written, evidence_as_written) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (gid, n, r["number"], r["text"], r["population"], r["interventions"], r["action"], r["strength"],
                 r["strength_as_written"], r["evidence_as_written"]))
    return {"guideline_id": gid, "recommendations": len(recs), "dropped_not_found_in_text": dropped,
            "chunks_read": len(chunks) - failed, "chunks_failed": failed}


def list_guidelines(pg, condition: str | None = None) -> list[dict]:
    rows = pg.execute(
        "SELECT g.id, g.title, g.issuer, g.year, g.conditions, g.region, g.source, g.licence, g.chunks_failed, "
        "(SELECT count(*) FROM guideline_recommendations r WHERE r.guideline_id = g.id) AS n_recs "
        "FROM guidelines g ORDER BY g.id").fetchall()
    if condition:
        c = condition.lower()
        rows = [r for r in rows if any(c in x or x in c for x in r["conditions"] or [])]
    return rows


def label(g: dict) -> str:
    """How a guideline is named in a report: issuer, year and title."""
    return f"{g['issuer']} {g['year'] or ''}".strip() + f", {g['title']}"


def recommendations(pg, guideline_ids: list[int] | None = None, condition: str | None = None) -> list[dict]:
    ids = guideline_ids or [g["id"] for g in list_guidelines(pg, condition)]
    if not ids:
        return []
    return pg.execute(
        "SELECT r.*, g.title, g.issuer, g.year, g.region FROM guideline_recommendations r JOIN guidelines g "
        "ON g.id = r.guideline_id WHERE r.guideline_id = ANY(%s) ORDER BY r.guideline_id, r.seq", (ids,)).fetchall()


def search(pg, query: str, condition: str | None = None, limit: int = 12) -> list[dict]:
    """Recommendations matching the words of a query (code: Postgres full-text search, any word)."""
    words = [w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())]
    if not words:
        return []
    ids = [g["id"] for g in list_guidelines(pg, condition)]
    if not ids:
        return []
    tsq = " | ".join(words[:12])
    return pg.execute(
        "SELECT r.guideline_id, r.seq, r.number, r.text, r.population, r.interventions, r.action, r.strength, "
        "r.strength_as_written, r.evidence_as_written, g.title, g.issuer, g.year, g.region, "
        "ts_rank(r.tsv, to_tsquery('english', %s)) AS rank FROM guideline_recommendations r "
        "JOIN guidelines g ON g.id = r.guideline_id WHERE r.guideline_id = ANY(%s) "
        "AND r.tsv @@ to_tsquery('english', %s) ORDER BY rank DESC LIMIT %s",
        (tsq, ids, tsq, int(limit))).fetchall()


# ---------------------------------------------------------------- comparing with a run (code only)
# Drug classes, so a recommendation about "csDMARDs" finds studies of methotrexate. A study naming a member
# counts for the class; a study naming the class counts for the class only, never for each member.
CLASSES = {
    "csdmard": ["methotrexate", "sulfasalazine", "sulphasalazine", "leflunomide", "hydroxychloroquine",
                "chloroquine", "conventional synthetic dmard"],
    "bdmard": ["adalimumab", "etanercept", "infliximab", "golimumab", "certolizumab", "tocilizumab",
               "sarilumab", "rituximab", "abatacept", "anakinra", "biologic", "tnf inhibitor", "anti-tnf"],
    "tsdmard": ["tofacitinib", "baricitinib", "upadacitinib", "filgotinib", "jak inhibitor"],
    "glucocorticoid": ["prednisone", "prednisolone", "methylprednisolone", "dexamethasone", "corticosteroid",
                       "steroid", "triamcinolone", "glucocorticoid"],
    "nsaid": ["ibuprofen", "diclofenac", "naproxen", "celecoxib", "etoricoxib", "meloxicam", "indomethacin",
              "aspirin", "nsaid", "non-steroidal anti-inflammatory", "cox-2 inhibitor"],
    "urate lowering": ["allopurinol", "febuxostat", "probenecid", "benzbromarone", "pegloticase",
                       "urate lowering", "urate-lowering"],
    "analgesic": ["paracetamol", "acetaminophen", "tramadol", "opioid", "duloxetine"],
    "exercise": ["exercise", "physiotherapy", "physical therapy", "strengthening", "aerobic"],
    "joint replacement": ["arthroplasty", "joint replacement", "knee replacement", "hip replacement"],
}
_ALIASES = {"dmard": "csdmard", "csdmards": "csdmard", "dmards": "csdmard", "bdmards": "bdmard",
            "biologics": "bdmard", "tsdmards": "tsdmard", "jak inhibitors": "tsdmard", "nsaids": "nsaid",
            "glucocorticoids": "glucocorticoid", "corticosteroids": "glucocorticoid", "steroids": "glucocorticoid",
            "ult": "urate lowering", "urate-lowering therapy": "urate lowering", "urate lowering therapy":
            "urate lowering", "mtx": "methotrexate", "hcq": "hydroxychloroquine", "ssz": "sulfasalazine",
            "physical activity": "exercise", "exercise therapy": "exercise"}


def _class_of(term: str) -> str | None:
    t = _norm(term)
    t = _ALIASES.get(t, t)
    if t in CLASSES:
        return t
    return None


def _has(phrase: str, text: str) -> bool:
    """`phrase` as whole words in `text` (a plural s allowed): "steroid" is not in "non steroidal"."""
    p = _norm(phrase)
    return bool(p) and re.search(r"(?<![a-z0-9])" + re.escape(p) + r"s?(?![a-z0-9])", text) is not None


def term_matches(rec_term: str, study_value: str) -> bool:
    """Does a study's intervention count for a recommendation's intervention?"""
    r, s = _norm(_ALIASES.get(_norm(rec_term), rec_term)), _norm(_ALIASES.get(_norm(study_value), study_value))
    if not r or not s:
        return False
    if r == s or (len(r) >= 5 and _has(r, s)) or (len(s) >= 5 and _has(s, r)):
        return True
    cls = _class_of(r)
    if cls:                                   # the recommendation names a class: any member counts
        return any(_has(m, s) for m in CLASSES[cls]) or _class_of(s) == cls
    return False


def compare(ctx, condition: str | None = None, guideline_ids: list[int] | None = None) -> dict:
    """Each recommendation against the run's papers: how many of the papers that state any intervention report
    one the recommendation names. Code only."""
    from research_agent.tools.extraction import _rows, _values

    recs = recommendations(ctx.pg, guideline_ids, condition)
    rows = _rows(ctx)
    stating = [r for r in rows if _values(r["data"], "interventions")]
    out = []
    for rec in recs:
        hits = {}
        for r in stating:
            vals = _values(r["data"], "interventions")
            matched = [v for v in vals for t in rec["interventions"] if term_matches(t, v)]
            if matched:
                hits[r["paper_id"]] = sorted(set(matched))[:3]
        if rec["interventions"] and stating:
            try:      # the same "count" event every counting tool emits: lets the text quote this n of N
                ctx.emit("guidelines", "count", {"n": len(hits), "total": len(stating)})
            except Exception:
                pass
        out.append({"guideline_id": rec["guideline_id"], "seq": rec["seq"], "number": rec["number"],
                    "guideline": label(rec), "text": rec["text"], "action": rec["action"],
                    "strength": rec["strength"], "strength_as_written": rec["strength_as_written"],
                    "interventions": rec["interventions"], "papers": len(hits), "paper_ids": list(hits)[:12],
                    "matched_values": sorted({v for vs in hits.values() for v in vs})[:8]})
    result = {"condition": condition, "papers_counted": len(rows), "papers_stating_interventions": len(stating),
              "recommendations": out,
              "note": "A paper reporting a drug or therapy is not a paper following the guideline. These counts say "
                      "how often the options a guideline names appear in the analysed studies, nothing more. "
                      "Guidelines are a reference here and never count as evidence."}
    ctx.save_note("guidelines:compare", {k: v for k, v in result.items()} | {"guideline_ids": guideline_ids})
    return result


def markdown(ctx) -> list[str]:
    """The comparison as a report section, when one has been made for this run."""
    note = ctx.notes().get("guidelines:compare")
    if not note:
        return []
    from research_agent.agents.report import _cite

    res = compare(ctx, note.get("condition"), note.get("guideline_ids"))
    if not res["recommendations"]:
        return []
    n = res["papers_stating_interventions"]
    L = ["## What guidelines recommend and what the studies report (computed)", "",
         f"Recommendations from the guideline library, set against the {n} analysed papers that state which "
         "treatments or interventions they used or studied. Each recommendation is quoted exactly as the "
         "guideline writes it.", "",
         "| Guideline | Recommendation | Strength | Papers reporting what it names |", "|---|---|---|---|"]
    for r in res["recommendations"]:
        text = r["text"] if len(r["text"]) <= 260 else r["text"][:257].rsplit(" ", 1)[0] + "…"
        strength = r["strength_as_written"] or "not stated"
        num = f" ({r['number']})" if r["number"] else ""
        papers = (f"{r['papers']} of {n}" + (": " + ", ".join(_cite(p) for p in r["paper_ids"][:4])
                                             + (" …" if r["papers"] > 4 else "") if r["papers"] else ""))
        if not r["interventions"]:
            papers = "not compared (names no treatment)"
        L.append(f"| {r['guideline']}{num} | {text.replace('|', '/')} | {strength} | {papers} |")
    L += ["", res["note"], ""]
    zero = [r for r in res["recommendations"] if r["interventions"] and r["action"] == "recommend"
            and r["strength"] == "strong" and r["papers"] == 0]
    if zero and n:
        L += [f"{len(zero)} strong recommendations name options that none of the {n} papers report. That can mean "
              "the option is not used in these settings, or only that it is not studied: the run's papers cannot "
              "tell the two apart.", ""]
    return L


# ---------------------------------------------------------------- tools for agents
def _search_tool(ctx, query: str, condition: str | None = None) -> dict:
    hits = search(ctx.pg, query, condition)
    return {"recommendations": [{"cite": f"[GL{h['guideline_id']}.{h['seq']}]",
                                 "cite_as": label(h) + (f", recommendation {h['number']}" if h["number"] else ""),
                                 "text": h["text"], "strength": h["strength_as_written"] or "not stated",
                                 "evidence": h["evidence_as_written"] or "not stated",
                                 "population": h["population"], "region": h["region"]} for h in hits],
            "note": "Guidelines are a reference, not evidence: quote them as what is recommended, never as a finding. "
                    "Name the guideline in your sentence ('According to the 2022 EULAR recommendations, ...') and "
                    "cite it with its marker exactly as given, e.g. [GL1.3]. An empty list means the library holds "
                    "no matching recommendation, not that none exists."}


GUIDELINE_TOOL = Tool(
    "guideline_recommendations",
    "Search the guideline library (EULAR, ACR, national guidelines the user added) for recommendations, quoted "
    "word for word with their strength. Use it to say what guidelines recommend, beside what the studies report.",
    obj({"query": {**STR, "description": "Words to look for, e.g. methotrexate first line"},
         "condition": {**STR, "description": "Optional, e.g. rheumatoid arthritis"}}, ["query"]),
    _search_tool, read_only=True, max_chars=12000)


def to_json(rows) -> str:
    return json.dumps(rows, default=str, indent=2)


# ---------------------------------------------------------------- guidelines inside reports, maps and drafts
# Writers cite a recommendation as [GL<guideline id>.<number in the library>], e.g. [GL1.3], and say in their
# own words whose it is ("According to the 2022 EULAR recommendations, ..."). Code then checks that the
# recommendation exists, that anything quoted from it is really in it, renders the marker as a readable
# citation, and lists every recommendation cited, word for word, at the end. A guideline never backs an
# "n of N": those still come from the run's own counts.
GL_CITE = re.compile(r"\[GL(\d+)\.(\d+)\]")
MAX_BRIEF_RECS = 12


def _words_of(text: str) -> set[str]:
    return {w.rstrip("s") for w in _norm(text).split() if len(w) > 3}


def relevant_guidelines(pg, text: str) -> list[dict]:
    """Guidelines whose conditions the run is about: every word of a condition appears in the question or
    protocol (so "rheumatoid arthritis" is relevant to a question on arthritis in Kenya only if it names it,
    while a guideline filed under "arthritis" is relevant to both)."""
    have = _words_of(text)
    out = []
    for g in list_guidelines(pg):
        for c in g["conditions"] or []:
            cw = _words_of(c)
            if cw and cw <= have:
                out.append(g)
                break
    return out


def _run_text(ctx) -> str:
    from research_agent.tools.extraction import protocol_of

    p = protocol_of(ctx) or {}
    return " ".join([ctx.question, p.get("setting") or "", " ".join(p.get("inclusion") or [])])


def for_brief(ctx) -> dict | None:
    """What the writer of a report, map or draft gets: the relevant recommendations with their cite markers,
    each with how many of the run's papers report what it names (counted by code, and allowed in the text)."""
    gls = relevant_guidelines(ctx.pg, _run_text(ctx))
    if not gls:
        return None
    res = compare(ctx, None, [g["id"] for g in gls])
    n = res["papers_stating_interventions"]
    ranked = sorted(res["recommendations"], key=lambda r: (-r["papers"], r["strength"] != "strong", r["seq"]))
    recs = []
    for r in ranked[:MAX_BRIEF_RECS]:
        recs.append({"cite": f"[GL{r['guideline_id']}.{r['seq']}]", "guideline": r["guideline"],
                     "number_in_guideline": r["number"], "text": r["text"][:300],
                     "strength": r["strength_as_written"] or "not stated", "action": r["action"],
                     "names": r["interventions"],
                     "studies_reporting_it": f"{r['papers']} of {n}" if r["interventions"] and n else "not compared"})
    return {"recommendations": recs, "papers_stating_treatments": n,
            "how_to_use": "Set what the studies report against what these guidelines recommend, in your own prose, "
                          "naming the guideline: 'According to the 2022 EULAR recommendations, ... [GL1.3]'. "
                          "Cite each with its marker exactly as given. Quote a recommendation only word for word. "
                          "A guideline is what is recommended, never a finding or support for a claim: do not "
                          "count it as a study. 'studies_reporting_it' counts the run's papers that report what "
                          "the recommendation names; you may quote it as an n of N."}


def render_citations(pg, text: str) -> tuple[str, list[dict], int]:
    """Check and render [GLx.y] markers. Returns (text, recommendations cited, markers removed)."""
    keys = sorted({(int(a), int(b)) for a, b in GL_CITE.findall(text or "")})
    if not keys:
        return text, [], 0
    rows = {(r["guideline_id"], r["seq"]): r for r in pg.execute(
        "SELECT r.guideline_id, r.seq, r.number, r.text, r.strength_as_written, r.evidence_as_written, g.title, "
        "g.issuer, g.year, g.source, g.url, g.licence FROM guideline_recommendations r JOIN guidelines g ON "
        "g.id = r.guideline_id WHERE (r.guideline_id, r.seq) IN (SELECT * FROM unnest(%s::bigint[], %s::int[]))",
        ([k[0] for k in keys], [k[1] for k in keys])).fetchall()}
    removed = 0
    from research_agent.tools.extraction import _words, quote_found

    # a sentence that cites a recommendation and quotes it must quote it exactly
    def check_sentence(m):
        sent = m.group(0)
        for a, b in GL_CITE.findall(sent):
            r = rows.get((int(a), int(b)))
            if not r:
                continue
            w = _words(r["text"])
            sh = {tuple(w[i:i + k]) for k in (3, 4, 5) for i in range(len(w) - k + 1)}
            for q in re.findall(r"[\"“]([^\"”]{12,})[\"”]", sent):
                if not quote_found(q, w, sh):
                    sent = sent.replace(f"[GL{a}.{b}]", f"[GL{a}.{b}] [quote not found in the recommendation]", 1)
        return sent
    text = re.sub(r"[^.!?\n]*\[GL\d+\.\d+\][^.!?\n]*[.!?]?", check_sentence, text)

    def render(m):
        nonlocal removed
        r = rows.get((int(m.group(1)), int(m.group(2))))
        if not r:
            removed += 1
            return "[guideline citation removed: not in the library]"
        num = f", rec. {r['number']}" if r["number"] else ""
        return f"[{r['issuer']} {r['year'] or ''}".rstrip() + f"{num}]"
    text = GL_CITE.sub(render, text)
    return text, [rows[k] for k in keys if k in rows], removed


def references_markdown(cited: list[dict]) -> list[str]:
    """Every recommendation a text cited, word for word, with its grade and source."""
    if not cited:
        return []
    L = ["## Guidelines referred to", "",
         "Recommendations cited above, quoted exactly as the guidelines write them. Guidelines are a reference "
         "for what is recommended; they are not counted as evidence anywhere in this document.", ""]
    for r in cited:
        num = f" recommendation {r['number']}" if r["number"] else ""
        grade = " / ".join(x for x in (r["strength_as_written"], r["evidence_as_written"]) if x) or "grade not stated"
        where = (f"https://pmc.ncbi.nlm.nih.gov/articles/{r['source']}/" if str(r["source"]).startswith("PMC")
                 else (r["url"] or r["source"]))
        L.append(f"- **{r['issuer']} {r['year'] or ''}**{num} ({grade}): \"{r['text']}\" "
                 f"From: {r['title']}. {where}")
    return L + [""]
