"""Literature-analysis tools: structured extraction per paper + deterministic queries over the results.

Extraction is the only place an LLM reads papers. Everything downstream (method counts, dataset
prevalence, geographic coverage, evidence checks) is SQL over the `extractions` table.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings
from research_agent.db import connect
from research_agent.ingestion.fulltext import fetch_fulltext, select_for_reading
from research_agent.tools.base import INT, STR, STRS, Tool, obj

LIST_FIELDS = ["task_types", "health_domains", "data_modalities", "datasets", "geography",
               "methods", "evaluation_metrics", "limitations"]
ENUM_FIELDS = {
    "validation_level": ["none", "internal", "external", "prospective", "clinical_trial", "not_stated"],
    "code_or_data_available": ["yes", "no", "not_stated"],
}
TEXT_FIELDS = ["problem", "sample_size", "key_findings"]
ALL_FIELDS = LIST_FIELDS + list(ENUM_FIELDS) + TEXT_FIELDS

_list = {"type": "array", "items": {"type": "string"}}

EXTRACTION_TOOL = {
    "name": "record_extraction",
    "description": "Record the structured extraction for this paper.",
    "input_schema": {
        "type": "object",
        "properties": {
            "problem": {"type": "string", "description": "The health problem addressed, one sentence."},
            "task_types": {**_list, "description": "e.g. diagnosis, detection, segmentation, prognosis, "
                           "risk prediction, forecasting, treatment recommendation, drug discovery, "
                           "generation, question answering, epidemic modelling, causal inference, review"},
            "health_domains": {**_list, "description": "Disease areas / specialties, e.g. malaria, breast cancer, "
                               "cardiology, mental health, maternal health"},
            "data_modalities": {**_list, "description": "e.g. chest X-ray, CT, MRI, histopathology, "
                                "dermoscopy, retinal fundus, ultrasound, EHR structured, clinical notes, "
                                "genomics, ECG, EEG, wearable sensors, surveillance counts, survey, "
                                "climate/environmental, mobility, social media"},
            "datasets": {**_list, "description": "Named datasets exactly as written (e.g. MIMIC-IV, CheXpert, "
                         "TCGA, UK Biobank). Only names, no descriptions."},
            "geography": {**_list, "description": "Countries/regions where the DATA comes from "
                          "(e.g. Kenya, sub-Saharan Africa, USA). Not author affiliations."},
            "methods": {**_list, "description": "Model families / techniques, e.g. CNN, U-Net, vision "
                        "transformer, LLM, foundation model, gradient boosting, logistic regression, "
                        "graph neural network, federated learning, SEIR model, Bayesian model"},
            "evaluation_metrics": {**_list, "description": "e.g. AUROC, accuracy, F1, Dice, RMSE"},
            "sample_size": {"type": "string", "description": "Patients/images/records as stated, or ''"},
            "validation_level": {"type": "string", "enum": ENUM_FIELDS["validation_level"],
                                 "description": "Strongest validation reported: internal split/CV, "
                                 "external dataset/site, prospective, clinical trial"},
            "code_or_data_available": {"type": "string", "enum": ENUM_FIELDS["code_or_data_available"]},
            "key_findings": {"type": "string", "description": "Main result in <= 2 sentences, with numbers if given"},
            "limitations": {**_list, "description": "Limitations the AUTHORS state. Do not invent any."},
        },
        "required": ALL_FIELDS,
    },
}

EXTRACTION_SYSTEM = """You extract structured facts from health research papers for a literature database.
Rules:
- Record only what the text states. If something is not stated, use an empty list, '' or 'not_stated'.
  Never guess — an empty value is correct and useful.
- Keep list items short (1-5 words), canonical names (e.g. 'vision transformer', not 'our ViT-based model').
- geography = where the data was collected, not where authors work.
- You are reading {source}. {source_note}"""


def _needs(existing: dict | None, depth: str) -> bool:
    if existing is None:
        return True
    return depth == "fulltext" and existing["source"] == "abstract"


def _extract_one(llm, paper: dict, fulltext: str | None) -> tuple[str, dict]:
    source = "fulltext" if fulltext else "abstract"
    note = ("Sections from the full paper follow the abstract." if fulltext else
            "Only the title and abstract are available, so many fields may be not stated.")
    user = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    if fulltext:
        user += f"\n\nSelected full-text sections:\n{fulltext}"
    resp = llm.chat(EXTRACTION_SYSTEM.format(source=source, source_note=note),
                    [{"role": "user", "content": [{"type": "text", "text": user}]}],
                    tools=[EXTRACTION_TOOL], force_tool="record_extraction", max_tokens=1200)
    if not resp.tool_calls:
        raise ValueError("model returned no extraction")
    return source, normalise(resp.tool_calls[0].input)


def normalise(data: dict) -> dict:
    out = {}
    for f in LIST_FIELDS:
        vals = data.get(f) or []
        if isinstance(vals, str):
            vals = [v for v in vals.split(",")]
        seen, clean = set(), []
        for v in vals:
            v = " ".join(str(v).split()).strip(" .;")
            if v and v.lower() not in seen and v.lower() not in {"not stated", "none", "n/a", "unknown"}:
                seen.add(v.lower()); clean.append(v)
        out[f] = clean[:15]
    for f, allowed in ENUM_FIELDS.items():
        v = str(data.get(f) or "not_stated").strip().lower().replace(" ", "_")
        out[f] = v if v in allowed else "not_stated"
    for f in TEXT_FIELDS:
        out[f] = " ".join(str(data.get(f) or "").split())[:600]
    return out


def extract_papers(ctx, paper_ids: list[str] | None = None, depth: str = "abstract") -> dict:
    """Extract (or reuse cached extractions for) shortlisted papers.

    depth='abstract' for everything; depth='fulltext' reads selected full-text sections
    (capped at MAX_FULLTEXT papers per run)."""
    shortlist = ctx.shortlist_ids()
    ids = [p for p in (paper_ids or shortlist) if p in set(shortlist)]
    if depth == "fulltext":
        ids = ids[: settings.max_fulltext]
    papers = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, title, abstract FROM papers WHERE paper_id = ANY(%s)", (ids,)).fetchall()}
    cached = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, source FROM extractions WHERE schema_version=%s AND paper_id = ANY(%s)",
        (ctx.extraction_version, ids)).fetchall()}
    todo = [p for p in ids if _needs(cached.get(p), depth)]

    fulltexts: dict[str, str | None] = {}
    ft_status: dict[str, str] = {}
    if depth == "fulltext" and todo:
        ft_status = fetch_fulltext(ctx.pg, todo)
        rows = ctx.pg.execute("SELECT paper_id, sections FROM paper_fulltext WHERE paper_id = ANY(%s) "
                              "AND status='ok'", (todo,)).fetchall()
        fulltexts = {r["paper_id"]: select_for_reading(r["sections"] or []) for r in rows}

    done, failed = 0, []
    llm = ctx.llm_factory()  # one client shared by the worker threads (SDK clients are thread-safe)

    def work(pid: str):
        source, data = _extract_one(llm, papers[pid], fulltexts.get(pid))
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO extractions (paper_id, schema_version, source, data, model) "
                "VALUES (%s,%s,%s,%s::jsonb,%s) ON CONFLICT (paper_id, schema_version) DO UPDATE "
                "SET source=EXCLUDED.source, data=EXCLUDED.data, model=EXCLUDED.model, created_at=now()",
                (pid, ctx.extraction_version, source, json.dumps(data), llm.model))
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=settings.extraction_workers) as pool:
        futures = {pool.submit(work, pid): pid for pid in todo if pid in papers}
        for fut in as_completed(futures):
            try:
                fut.result(); done += 1
            except Exception as exc:  # one bad paper must not sink the run
                failed.append({"paper_id": futures[fut], "error": str(exc)[:200]})

    return {"requested": len(ids), "reused_from_cache": len(ids) - len(todo), "newly_extracted": done,
            "failed": failed, "depth": depth,
            "fulltext_status": {s: sum(1 for v in ft_status.values() if v == s) for s in set(ft_status.values())}}


# ---------------------------------------------------------------- deterministic queries
def _rows(ctx) -> list[dict]:
    return ctx.pg.execute(
        """SELECT e.paper_id, e.source, e.data, p.year, p.title
           FROM run_papers rp JOIN extractions e ON e.paper_id = rp.paper_id AND e.schema_version = %s
           JOIN papers p ON p.paper_id = rp.paper_id WHERE rp.run_id = %s""",
        (ctx.extraction_version, ctx.run_id)).fetchall()


def _values(data: dict, field: str) -> list[str]:
    v = data.get(field)
    if field in ENUM_FIELDS:
        return [] if v in (None, "not_stated") else [v]
    return list(v or [])


def extraction_coverage(ctx) -> dict:
    rows = _rows(ctx)
    n = len(rows)
    stated = {f: sum(1 for r in rows if _values(r["data"], f) or (f in TEXT_FIELDS and r["data"].get(f)))
              for f in ALL_FIELDS}
    return {"shortlist_size": len(ctx.shortlist_ids()), "extracted": n,
            "by_source": {s: sum(1 for r in rows if r["source"] == s) for s in ("abstract", "fulltext")},
            "field_stated_rate": {f: round(stated[f] / n, 2) if n else 0 for f in ALL_FIELDS}}


def value_counts(ctx, field: str, top: int = 30) -> dict:
    if field not in LIST_FIELDS and field not in ENUM_FIELDS:
        return {"error": f"field must be one of {LIST_FIELDS + list(ENUM_FIELDS)}"}
    rows = _rows(ctx)
    counts: dict[str, dict] = {}
    for r in rows:
        for v in _values(r["data"], field):
            k = v.lower()
            c = counts.setdefault(k, {"value": v, "n": 0, "paper_ids": []})
            c["n"] += 1
            c["paper_ids"].append(r["paper_id"])
    stated = sum(1 for r in rows if _values(r["data"], field))
    ranked = sorted(counts.values(), key=lambda c: -c["n"])[: max(1, min(int(top), 80))]
    for c in ranked:
        c["paper_ids"] = c["paper_ids"][:8]
    return {"field": field, "n_papers": len(rows), "n_stated": stated, "n_not_stated": len(rows) - stated,
            "values": ranked}


def cross_tab(ctx, field_a: str, field_b: str, top: int = 12) -> dict:
    rows = _rows(ctx)
    table: dict[tuple, int] = {}
    for r in rows:
        for a in {v.lower() for v in _values(r["data"], field_a)}:
            for b in {v.lower() for v in _values(r["data"], field_b)}:
                table[(a, b)] = table.get((a, b), 0) + 1
    ranked = sorted(table.items(), key=lambda kv: -kv[1])[: max(1, min(int(top) * 3, 100))]
    return {"field_a": field_a, "field_b": field_b, "n_papers": len(rows),
            "pairs": [{"a": a, "b": b, "n": n} for (a, b), n in ranked]}


def field_by_year(ctx, field: str, any_of: list[str]) -> dict:
    rows = _rows(ctx)
    needles = [s.lower() for s in any_of]
    years: dict[int, dict] = {}
    for r in rows:
        y = years.setdefault(r["year"], {"year": r["year"], "papers": 0, "matching": 0})
        y["papers"] += 1
        if any(any(n in v.lower() for n in needles) for v in _values(r["data"], field)):
            y["matching"] += 1
    return {"field": field, "any_of": any_of, "note": "within the shortlist only",
            "by_year": [years[k] for k in sorted(years)]}


def list_extractions(ctx, fields: list[str] | None = None, paper_ids: list[str] | None = None,
                     limit: int = 25) -> dict:
    fields = [f for f in (fields or ["problem", "methods", "datasets", "geography"]) if f in ALL_FIELDS]
    rows = _rows(ctx)
    if paper_ids:
        rows = [r for r in rows if r["paper_id"] in set(paper_ids)]
    rows = rows[: max(1, min(int(limit), 60))]
    return {"papers": [{"paper_id": r["paper_id"], "year": r["year"], "title": r["title"],
                        "source": r["source"], **{f: r["data"].get(f) for f in fields}} for r in rows]}


FIELD_ENUM = {"type": "string", "enum": LIST_FIELDS + list(ENUM_FIELDS)}

EXTRACTION_TOOLS = [
    Tool("extract_papers",
         "Run structured extraction on shortlisted papers (cached across runs). depth='abstract' is cheap "
         "and covers everything; depth='fulltext' reads methods/data/results/limitations sections for "
         f"up to {settings.max_fulltext} papers — use it for the most central papers.",
         obj({"paper_ids": {**STRS, "description": "defaults to the whole shortlist"},
              "depth": {"type": "string", "enum": ["abstract", "fulltext"]}}),
         extract_papers),
    Tool("extraction_coverage", "How many shortlisted papers are extracted, and how often each field is stated.",
         obj({}), extraction_coverage),
]

ANALYSIS_TOOLS = [
    Tool("value_counts", "Count values of an extracted field across the shortlist (with example paper ids). "
         "n_not_stated tells you how often papers are silent on it.",
         obj({"field": FIELD_ENUM, "top": INT}, ["field"]), value_counts),
    Tool("cross_tab", "Co-occurrence counts between two extracted fields (e.g. methods × data_modalities).",
         obj({"field_a": FIELD_ENUM, "field_b": FIELD_ENUM, "top": INT}, ["field_a", "field_b"]), cross_tab),
    Tool("field_by_year", "Per-year count of shortlisted papers whose field contains any of the given terms.",
         obj({"field": FIELD_ENUM, "any_of": STRS}, ["field", "any_of"]), field_by_year),
    Tool("list_extractions", "Show extracted records (chosen fields) for shortlisted papers.",
         obj({"fields": STRS, "paper_ids": STRS, "limit": INT}), list_extractions),
]
