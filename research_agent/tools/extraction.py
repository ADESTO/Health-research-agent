"""Literature-analysis tools: structured extraction per paper + deterministic queries over the results.

Extraction is the only place an LLM reads papers. Everything downstream (method counts, dataset
prevalence, geographic coverage, evidence checks) is SQL over the `extractions` table.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings
from research_agent.db import connect
from research_agent.ingestion.fulltext import fetch_fulltext, select_for_reading
from research_agent.tools.base import INT, STR, STRS, Tool, obj
from research_agent.tools.results import STRUCT_FIELDS, STRUCT_SCHEMA

# health-v4 widened the form beyond AI and modelling papers: clinical, laboratory, pharmacological and
# epidemiological studies are described by what was studied, in whom, with what, how it works and what was
# measured. Runs read under an earlier version keep the earlier form (see fields_for).
GENERAL_FIELDS = ["study_designs", "populations", "organisms", "interventions", "mechanisms", "targets",
                  "outcomes"]
LIST_FIELDS = ["task_types", "health_domains", "data_modalities", "datasets", "geography",
               "methods", "evaluation_metrics", "limitations"] + GENERAL_FIELDS
# health-v5 splits `validation_level`. Through v4 a single "external" value covered two different things:
# a model tested on a LATER PERIOD of the same data, and a model tested in a DIFFERENT PLACE. Reports then
# read "externally validated" for both, and a comparison of external against internal mixed them. A run that
# cared about geographic transfer could not tell them apart, and the miscoding was only ever caught by an
# agent reading the paper. They are now separate rungs, ordered weakest to strongest, each answering one
# question. Runs read under v4 or earlier keep the old enum (see enums_for) so their numbers stay comparable.
ENUM_FIELDS = {
    "validation_level": ["none", "internal", "temporal_holdout", "external_site", "prospective",
                        "clinical_trial", "not_stated"],
    "code_or_data_available": ["yes", "no", "not_stated"],
}
ENUMS_V4 = {
    "validation_level": ["none", "internal", "external", "prospective", "clinical_trial", "not_stated"],
    "code_or_data_available": ["yes", "no", "not_stated"],
}
TEXT_FIELDS = ["problem", "sample_size", "key_findings"]
# Fields the report leans on. Each needs a quote from the text, checked by code; values without one are dropped.
EVIDENCE_FIELDS = ["methods", "datasets", "data_modalities", "geography", "validation_level",
                   "code_or_data_available", "study_designs", "populations", "organisms", "interventions",
                   "mechanisms", "targets"]
ALL_FIELDS = LIST_FIELDS + list(ENUM_FIELDS) + TEXT_FIELDS
LEGACY_VERSIONS = {"health-v1", "health-v2", "health-v3"}
OLD_ENUM_VERSIONS = LEGACY_VERSIONS | {"health-v4"}


def fields_for(version: str | None) -> tuple[list[str], list[str], list[str]]:
    """(list fields, evidence fields, all fields) of the form a given extraction version was read with."""
    if version in LEGACY_VERSIONS:
        drop = set(GENERAL_FIELDS)
        return ([f for f in LIST_FIELDS if f not in drop], [f for f in EVIDENCE_FIELDS if f not in drop],
                [f for f in ALL_FIELDS if f not in drop])
    return list(LIST_FIELDS), list(EVIDENCE_FIELDS), list(ALL_FIELDS)


def enums_for(version: str | None) -> dict[str, list[str]]:
    """The enum values a given extraction version was read with. A run read before validation_level was
    split keeps the single `external` value, so its stored records and its counts still agree."""
    return {k: list(v) for k, v in (ENUMS_V4 if version in OLD_ENUM_VERSIONS else ENUM_FIELDS).items()}


_list = {"type": "array", "items": {"type": "string"}}

EXTRACTION_TOOL = {
    "name": "record_extraction",
    "description": "Record the structured extraction for this paper.",
    "input_schema": {
        "type": "object",
        "properties": {
            "problem": {"type": "string", "description": "The health problem addressed, one sentence."},
            "task_types": {**_list, "description": "What the study sets out to do, e.g. diagnosis, prognosis, "
                           "risk prediction, forecasting, treatment efficacy, safety, mechanism of action, "
                           "pharmacokinetics, drug interaction, drug discovery, resistance surveillance, "
                           "prevalence estimation, risk factor analysis, screening, epidemic modelling, "
                           "causal inference, review"},
            "health_domains": {**_list, "description": "Disease areas / specialties, e.g. malaria, "
                               "antimicrobial resistance, depression, anxiety, schizophrenia, diabetes, "
                               "thyroid disorders, reproductive endocrinology, cardiology, maternal health"},
            "data_modalities": {**_list, "description": "Kinds of data or material analysed, e.g. EHR "
                                "structured, clinical notes, chest X-ray, MRI, histopathology, ECG, EEG, "
                                "genomics, whole-genome sequences, bacterial isolates, blood or serum samples, "
                                "urine, tissue, cell lines, hormone assays, drug concentrations, questionnaires "
                                "or rating scales, interviews, wearable sensors, surveillance counts, survey, "
                                "claims or prescription records, climate/environmental, social media"},
            "datasets": {**_list, "description": "Named datasets, registries or cohorts exactly as written (e.g. "
                         "MIMIC-IV, UK Biobank, NHANES, GLASS, TCGA). Only names, no descriptions."},
            "geography": {**_list, "description": "Countries/regions where the DATA or samples come from "
                          "(e.g. Kenya, sub-Saharan Africa, USA). Not author affiliations."},
            "methods": {**_list, "description": "Analytical, laboratory and modelling techniques, e.g. "
                        "logistic regression, Cox regression, meta-analysis, mixed-effects model, CNN, "
                        "gradient boosting, LLM, SEIR model, disk diffusion, broth microdilution, PCR, "
                        "whole-genome sequencing, ELISA, LC-MS/MS, molecular docking, "
                        "population pharmacokinetic modelling"},
            "evaluation_metrics": {**_list, "description": "Measures the results are reported in, e.g. AUROC, "
                                   "accuracy, RMSE, odds ratio, hazard ratio, relative risk, mean difference, "
                                   "MIC, IC50, EC50, Cmax, AUC (plasma), half-life, prevalence"},
            "study_designs": {**_list, "description": "Study design(s), e.g. randomised controlled trial, "
                              "cohort, case-control, cross-sectional, case series, systematic review, "
                              "meta-analysis, qualitative, in vitro, animal study, in silico, "
                              "pharmacokinetic study, modelling study, diagnostic accuracy study"},
            "populations": {**_list, "description": "Who or what was studied, e.g. adults with major "
                            "depressive disorder, children under 5, pregnant women, ICU patients, healthy "
                            "volunteers, postmenopausal women, Wistar rats, HepG2 cells"},
            "organisms": {**_list, "description": "Pathogens and model organisms named as studied, e.g. "
                          "Klebsiella pneumoniae, MRSA, Escherichia coli, Plasmodium falciparum, mice. "
                          "Empty for studies of humans only."},
            "interventions": {**_list, "description": "Drugs, compounds, therapies, exposures or programmes "
                              "studied, e.g. ceftriaxone, carbapenems, sertraline, cognitive behavioural "
                              "therapy, levothyroxine, oral contraceptives, antibiotic stewardship"},
            "mechanisms": {**_list, "description": "Mechanisms of action, pathways or resistance mechanisms "
                           "the paper studies or reports as its own finding, e.g. serotonin reuptake "
                           "inhibition, beta-lactamase production, efflux pump, biofilm formation, HPA axis "
                           "dysregulation, CYP3A4 inhibition, receptor agonism, negative feedback"},
            "targets": {**_list, "description": "Molecular targets, receptors, enzymes, hormones or genes "
                        "central to the study, e.g. 5-HT1A receptor, dopamine D2 receptor, CYP2D6, mecA, "
                        "blaNDM-1, blaCTX-M, oestrogen receptor alpha, TSH, cortisol, insulin"},
            "outcomes": {**_list, "description": "Outcomes or endpoints measured, e.g. mortality, remission, "
                         "PHQ-9 score, HbA1c, treatment failure, resistance prevalence, MIC, plasma "
                         "concentration, adverse events, hospital admission"},
            "sample_size": {"type": "string", "description": "Participants/samples/isolates/records as stated, or ''"},
            "validation_level": {"type": "string", "enum": ENUM_FIELDS["validation_level"],
                                 "description": "For predictive models and diagnostic tests, the strongest "
                                 "validation reported. Each value answers ONE question, so pick by what was "
                                 "held out, not by the word the paper uses: internal = a random split or "
                                 "cross-validation inside one dataset; temporal_holdout = tested on a LATER "
                                 "PERIOD of the same setting (a later year, out-of-fit forecasting, rolling "
                                 "origin), however the paper labels it; external_site = tested in a DIFFERENT "
                                 "PLACE, site, cohort or population than it was fitted on (including spatial "
                                 "cross-validation across places); prospective = tested on data collected "
                                 "after the model was fixed; clinical_trial for a trial of a treatment. "
                                 "A paper that calls a later-period test 'external validation' is "
                                 "temporal_holdout. Otherwise not_stated."},
            "code_or_data_available": {"type": "string", "enum": ENUM_FIELDS["code_or_data_available"]},
            "key_findings": {"type": "string", "description": "Main result in <= 2 sentences, with numbers if given"},
            "limitations": {**_list, "description": "Limitations the AUTHORS state. Do not invent any."},
            **STRUCT_SCHEMA,
            "evidence": {
                "type": "object",
                "description": "For each of these fields that has a value, 1-2 short passages (up to 25 words "
                               "each) COPIED WORD FOR WORD from the text above that show it. Code checks every "
                               "passage against the text; a field whose passages are not found is discarded.",
                "properties": {f: _list for f in EVIDENCE_FIELDS},
            },
        },
        "required": ALL_FIELDS + STRUCT_FIELDS + ["evidence"],
    },
}

# ---------------------------------------------------------------- run protocol (question-specific fields)
SPECIAL_FIELDS = {"corpus", "read", "year"}   # usable in `where` filters: paper source, read depth, year


def protocol_of(ctx) -> dict | None:
    """The validated protocol for this run, if the Protocol agent has written one."""
    note = ctx.notes().get("protocol") if hasattr(ctx, "notes") else None
    return (note or {}).get("protocol") if isinstance(note, dict) else None


def known_fields(ctx) -> tuple[list[str], dict[str, list[str]]]:
    """Base fields plus this run's protocol fields: (list fields, {enum field: allowed values})."""
    version = getattr(ctx, "extraction_version", None)
    lists = fields_for(version)[0]
    enums = enums_for(version)
    for f in (protocol_of(ctx) or {}).get("fields", []):
        if f["type"] == "enum":
            enums[f["name"]] = f["values"]
        else:
            lists.append(f["name"])
    return lists, enums


def _field_error(ctx, *fields: str) -> str | None:
    lists, enums = known_fields(ctx)
    for f in fields:
        if f not in lists and f not in enums:
            return f"field must be one of {lists + list(enums)}"
    return None


def _needs(existing: dict | None, depth: str) -> bool:
    if existing is None:
        return True
    return depth == "fulltext" and existing["source"] == "abstract"


_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"}))
    return _WORD.findall(text)


def quote_found(quote: str, source_words: list[str], source_shingles: set[tuple]) -> bool:
    """True when the quote appears in the text, allowing for punctuation and small edits: its words must
    appear as a run of at least 5 consecutive words (or all of them, for shorter quotes)."""
    q = _words(quote)
    if len(q) < 3:
        return False
    n = min(5, len(q))
    return any(tuple(q[i:i + n]) in source_shingles for i in range(len(q) - n + 1))


def check_evidence(data: dict, evidence, source_text: str, fields: list[str] | None = None,
                   list_fields: list[str] | None = None) -> dict:
    """Keep a field's values only when at least one of its quotes is really in the text read."""
    fields = EVIDENCE_FIELDS if fields is None else fields
    list_fields = LIST_FIELDS if list_fields is None else list_fields
    words = _words(source_text)
    shingles = {tuple(words[i:i + k]) for k in (3, 4, 5) for i in range(len(words) - k + 1)}
    evidence = evidence if isinstance(evidence, dict) else {}
    kept, unverified = {}, {}
    for f in fields:
        quotes = evidence.get(f) or []
        if isinstance(quotes, str):
            quotes = [quotes]
        good = [q for q in quotes if isinstance(q, str) and quote_found(q, words, shingles)][:2]
        has_value = bool(data[f]) if f in list_fields else data[f] != "not_stated"
        if not has_value:
            continue
        if good:
            kept[f] = [" ".join(q.split())[:300] for q in good]
        else:
            unverified[f] = data[f]
            data[f] = [] if f in list_fields else "not_stated"
    data["evidence"] = kept
    if unverified:
        data["_unverified"] = unverified
    return data


def normalise(data: dict, version: str | None = None) -> dict:
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
    for f, allowed in enums_for(version).items():
        v = str(data.get(f) or "not_stated").strip().lower().replace(" ", "_")
        out[f] = v if v in allowed else "not_stated"
    for f in TEXT_FIELDS:
        out[f] = " ".join(str(data.get(f) or "").split())[:600]
    return out


def extract_papers(ctx, paper_ids: list[str] | None = None, depth: str = "abstract",
                   limit: int | None = None, force: bool = False) -> dict:
    """Extract (or reuse cached extractions for) shortlisted papers.

    depth='abstract' for everything; depth='fulltext' reads selected full-text sections
    (capped at MAX_FULLTEXT papers per run, or `limit` when given).
    force=True reads papers again even when they were already read at this depth: for a re-read after the
    reading instructions or the run's protocol fields changed."""
    shortlist = ctx.shortlist_ids()
    ids = [p for p in (paper_ids or shortlist) if p in set(shortlist)]
    if depth == "fulltext":
        ids = ids[: max(1, int(limit)) if limit else settings.max_fulltext]
    papers = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, title, abstract FROM papers WHERE paper_id = ANY(%s)", (ids,)).fetchall()}
    cached = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, source FROM extractions WHERE schema_version=%s AND paper_id = ANY(%s)",
        (ctx.extraction_version, ids)).fetchall()}
    todo = [p for p in ids if force or _needs(cached.get(p), depth)]

    fulltexts: dict[str, str | None] = {}
    ft_status: dict[str, str] = {}
    if depth == "fulltext" and todo:
        ft_status = fetch_fulltext(ctx.pg, todo)
        rows = ctx.pg.execute("SELECT paper_id, sections FROM paper_fulltext WHERE paper_id = ANY(%s) "
                              "AND status='ok'", (todo,)).fetchall()
        fulltexts = {r["paper_id"]: select_for_reading(r["sections"] or [], budget=settings.fulltext_read_chars) for r in rows}

    # One read per paper covers the general fields and, when the run has a protocol, its question-specific
    # fields too. Abstract-only papers are read in batches.
    from research_agent.tools.reading import read_papers

    protocol = protocol_of(ctx)
    items = [{**papers[pid], "fulltext": fulltexts.get(pid)} for pid in todo if pid in papers]
    results, failed, model = read_papers(ctx, items, protocol, include_base=True, step="extraction")
    conn = connect()
    try:
        for pid, res in results.items():
            conn.execute(
                "INSERT INTO extractions (paper_id, schema_version, source, data, model) "
                "VALUES (%s,%s,%s,%s::jsonb,%s) ON CONFLICT (paper_id, schema_version) DO UPDATE "
                "SET source=EXCLUDED.source, data=EXCLUDED.data, model=EXCLUDED.model, created_at=now()",
                (pid, ctx.extraction_version, res["source"], json.dumps(res["base"]), model))
            if res.get("protocol") is not None:
                conn.execute(
                    "INSERT INTO protocol_extractions (run_id, paper_id, source, data) VALUES (%s,%s,%s,%s::jsonb) "
                    "ON CONFLICT (run_id, paper_id) DO UPDATE SET source=EXCLUDED.source, data=EXCLUDED.data",
                    (ctx.run_id, pid, res["source"], json.dumps(res["protocol"])))
    finally:
        conn.close()
    done = len(results)

    dropped: dict[str, int] = {}
    for r in ctx.pg.execute("SELECT data FROM extractions WHERE schema_version=%s AND paper_id = ANY(%s)",
                            (ctx.extraction_version, ids)).fetchall():
        for f in (r["data"].get("_unverified") or {}):
            dropped[f] = dropped.get(f, 0) + 1
    return {"requested": len(ids), "reused_from_cache": len(ids) - len(todo), "newly_extracted": done,
            "failed": failed, "depth": depth,
            "fields_dropped_without_evidence": dropped,
            "fulltext_status": {s: sum(1 for v in ft_status.values() if v == s) for s in set(ft_status.values())}}


# ---------------------------------------------------------------- deterministic queries
def _rows(ctx) -> list[dict]:
    """One record per extracted shortlisted paper. `data` merges the cached base extraction with this
    run's protocol fields; `corpus` is where the paper came from (arxiv | pmc); `source` is read depth."""
    rows = ctx.pg.execute(
        """SELECT e.paper_id, e.source, e.data, pe.data AS pdata, p.year, p.title, p.source AS corpus
           FROM run_papers rp JOIN extractions e ON e.paper_id = rp.paper_id AND e.schema_version = %s
           JOIN papers p ON p.paper_id = rp.paper_id
           LEFT JOIN protocol_extractions pe ON pe.run_id = rp.run_id AND pe.paper_id = rp.paper_id
           WHERE rp.run_id = %s""",
        (ctx.extraction_version, ctx.run_id)).fetchall()
    out = []
    for r in rows:
        data = dict(r["data"])
        extra = r.pop("pdata") or {}
        if extra:
            data.update({k: v for k, v in extra.items() if k not in ("evidence", "_unverified")})
            data["evidence"] = {**(data.get("evidence") or {}), **(extra.get("evidence") or {})}
            if extra.get("_unverified"):
                data["_unverified"] = {**(data.get("_unverified") or {}), **extra["_unverified"]}
        out.append({**r, "data": data})
    _merge_rechecks(ctx, out)
    # The run's cohort, if it has one, is applied here and nowhere else: every count downstream then
    # counts the same papers without each caller having to restate the scope (see tools/cohort.py).
    from research_agent.tools import cohort

    return cohort.apply(ctx, out)


def _merge_rechecks(ctx, rows: list[dict]) -> None:
    """Add values confirmed by a focused re-check (each with a verified quote) to the extracted fields."""
    from research_agent.tools.recheck import confirmed_values

    try:
        confirmed = confirmed_values(ctx.pg, ctx.run_id)
    except Exception:   # a database from before re-checks existed: nothing to merge
        return
    for r in rows:
        for field, found in confirmed.get(r["paper_id"], {}).items():
            vals = list(r["data"].get(field) or []) if not isinstance(r["data"].get(field), str) else []
            have = {v.lower() for v in vals}
            added = [v for v, _ in found if v and v.lower() not in have]
            if not added:
                continue
            r["data"][field] = vals + added
            ev = dict(r["data"].get("evidence") or {})
            ev[field] = list(ev.get(field) or []) + [q for _, q in found if q][:2]
            r["data"]["evidence"] = ev
            r["data"].setdefault("_rechecked", {})[field] = added


def _values(data: dict, field: str) -> list[str]:
    """Values of a field as a list. Enum-style fields (a single string, including protocol enums) give
    one value, or none when 'not_stated'."""
    v = data.get(field)
    if v is None:
        return []
    if isinstance(v, str):
        return [] if v in ("", "not_stated") else [v]
    return [x for x in v if x]


def extraction_coverage(ctx) -> dict:
    rows = _rows(ctx)
    n = len(rows)
    _, ev_fields, all_fields = fields_for(ctx.extraction_version)
    stated = {f: sum(1 for r in rows if _values(r["data"], f) or (f in TEXT_FIELDS and r["data"].get(f)))
              for f in all_fields}
    return {"shortlist_size": len(ctx.shortlist_ids()), "extracted": n,
            "by_source": {s: sum(1 for r in rows if r["source"] == s) for s in ("abstract", "fulltext")},
            "field_stated_rate": {f: round(stated[f] / n, 2) if n else 0 for f in all_fields},
            "dropped_without_evidence": {f: sum(1 for r in rows if f in (r["data"].get("_unverified") or {}))
                                         for f in ev_fields}}


def value_counts(ctx, field: str, top: int = 30) -> dict:
    err = _field_error(ctx, field)
    if err:
        return {"error": err}
    rows = _rows(ctx)
    counts: dict[str, dict] = {}
    for r in rows:
        for k, v in {v.lower(): v for v in _values(r["data"], field)}.items():   # a paper counts once per value
            c = counts.setdefault(k, {"value": v, "n": 0, "paper_ids": []})
            c["n"] += 1
            c["paper_ids"].append(r["paper_id"])
    stated = sum(1 for r in rows if _values(r["data"], field))
    ranked = sorted(counts.values(), key=lambda c: -c["n"])[: max(1, min(int(top), 80))]
    for c in ranked:
        c["paper_ids"] = c["paper_ids"][:8]
    # Not recorded as evidence for the report audit: with dozens of values, almost every small "n of N"
    # would match one of them by coincidence. Numbers a report quotes are measured with test_claim instead.
    return {"field": field, "n_papers": len(rows), "n_stated": stated, "n_not_stated": len(rows) - stated,
            "values": ranked}


def cross_tab(ctx, field_a: str, field_b: str, top: int = 12) -> dict:
    err = _field_error(ctx, field_a, field_b)
    if err:
        return {"error": err}
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
    err = _field_error(ctx, field)
    if err:
        return {"error": err}
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
    lists, enums = known_fields(ctx)
    allowed = set(ALL_FIELDS) | set(lists) | set(enums) | {"evidence"}
    fields = [f for f in (fields or ["problem", "methods", "datasets", "geography"]) if f in allowed]
    rows = _rows(ctx)
    if paper_ids:
        rows = [r for r in rows if r["paper_id"] in set(paper_ids)]
    rows = rows[: max(1, min(int(limit), 60))]
    return {"papers": [{"paper_id": r["paper_id"], "year": r["year"], "title": r["title"],
                        "source": r["source"], "corpus": r.get("corpus"),
                        **{f: r["data"].get(f) for f in fields}} for r in rows]}


FIELD_ENUM = {"type": "string",
              "description": "An extracted field: " + ", ".join(LIST_FIELDS + list(ENUM_FIELDS)) +
                             ", or a question-specific protocol field (names start with q_)."}

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
         obj({"field": FIELD_ENUM, "top": INT}, ["field"]), value_counts, read_only=True),
    Tool("cross_tab", "Co-occurrence counts between two extracted fields (e.g. methods × data_modalities).",
         obj({"field_a": FIELD_ENUM, "field_b": FIELD_ENUM, "top": INT}, ["field_a", "field_b"]), cross_tab, read_only=True),
    Tool("field_by_year", "Per-year count of shortlisted papers whose field contains any of the given terms.",
         obj({"field": FIELD_ENUM, "any_of": STRS}, ["field", "any_of"]), field_by_year, read_only=True),
    Tool("list_extractions", "Show extracted records (chosen fields) for shortlisted papers.",
         obj({"fields": STRS, "paper_ids": STRS, "limit": INT}), list_extractions),
]
