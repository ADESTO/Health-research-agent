"""Protocol: the question-specific fields a Research Opportunity Map needs.

The base extraction covers what every health paper has (methods, datasets, validation level...). A
question like "Can ML improve 1-6 month malaria forecasting?" hinges on things the base schema does not
record: forecast horizon, whether forecasts are probabilistic, how validation was split, whether
interventions or reporting delays were modelled. The Protocol agent defines those fields for the
question; every shortlisted paper is then extracted against them, with evidence quotes checked by code,
exactly like the base fields.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.agents.base import Agent
from research_agent.config import settings
from research_agent.db import connect
from research_agent.ingestion.fulltext import select_for_reading
from research_agent.tools.base import STR, STRS, obj
from research_agent.tools.extraction import ENUM_FIELDS, LIST_FIELDS, _list, check_evidence
from research_agent.tools.search import SEARCH_TOOLS

MAX_FIELDS = 10

_FIELD_SPEC = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "snake_case, e.g. forecast_horizon"},
        "type": {"type": "string", "enum": ["enum", "list"],
                 "description": "enum = exactly one category per paper; list = several values allowed"},
        "values": {**STRS, "description": "enum: the mutually exclusive categories (not_stated is added "
                                          "automatically). list: suggested canonical values."},
        "definition": {"type": "string", "description": "How to decide the value from a paper, one or two "
                                                         "sentences, precise enough for another reader."},
        "desirable": {**STRS, "description": "Values that represent the practice or opportunity the question "
                                             "cares about, e.g. spatial_holdout, yes. Their absence becomes a "
                                             "candidate gap."},
        "search": {"type": "object", "description": "Optional: for each desirable value, a keyword query "
                                                    "(web-search syntax; parentheses group) to count it across "
                                                    "the whole corpus. Name only the practice, not the topic."},
        "role": {"type": "string", "enum": ["method", "data", "outcome", "evaluation", "setting", "deployment",
                                            "reporting"],
                 "description": "Which dimension of a study this field describes."},
        "groups": {"type": "array", "description": "Desirable values that are the SAME practice, so their absence "
                                                  "is one gap, e.g. spatial_holdout + spatiotemporal_holdout = "
                                                  "'testing on unseen places'.",
                   "items": {"type": "object", "properties": {"label": STR, "values": STRS},
                             "required": ["label", "values"]}},
    },
    "required": ["name", "type", "definition"],
}

PROTOCOL = Agent(
    name="protocol",
    role="Turns the research question into a review protocol: scope, inclusion criteria and the "
         "question-specific fields to extract from every paper.",
    system=f"""You are the Protocol agent for a Research Opportunity Map. Before any paper is read, define what
must be measured in every paper to answer the question, the way a systematic review protocol does.

Every paper is ALREADY extracted for these base fields, so do not redefine them:
{", ".join(LIST_FIELDS + list(ENUM_FIELDS))}.

Define 4-{MAX_FIELDS} ADDITIONAL fields that the question hinges on. For a forecasting question these are
typically: forecast horizon, spatial unit, target variable, whether forecasts are probabilistic, how
validation was split (random, temporal holdout, spatial holdout), whether interventions or reporting delays
are modelled, whether a simple baseline was compared. For other questions, choose what decides the answer.
Rules:
- Prefer enum fields with 3-7 mutually exclusive categories: they can be counted exactly.
- Every definition must let a reader decide the value from the paper text alone.
- Mark as `desirable` the values the question treats as good practice or opportunity: their rarity is
  what the map reports as a gap.
- If several desirable values are the same underlying practice, put them in one `group` with a plain label
  (e.g. spatial_holdout and spatiotemporal_holdout both mean testing on places the model never saw). Values
  that are genuinely different practices stay separate.
- Give each field a `role` (method, data, outcome, evaluation, setting, deployment, reporting).
- Give `search` keyword queries for desirable values so their rarity can be checked across the corpus. The
  query names the practice only (e.g. "spatial cross-validation" OR "leave-one-district-out"); the topic is
  added separately from topic_query.
You may use hybrid_search and corpus_count to see how papers in this area describe things.
Also write the setting (population, place, time frame), 3-6 inclusion and exclusion criteria, and a
topic_query that finds this topic's papers across the corpus (used to check whether a gap is real).""",
    tools=[t for t in SEARCH_TOOLS if t.name in ("hybrid_search", "corpus_count")],
    finish_schema=obj({
        "setting": STR,
        "topic_query": {"type": "string", "description": "Keyword query (web-search syntax) that finds papers "
                                                          "on this question's topic across the corpus, e.g. "
                                                          "malaria (forecast OR prediction OR \"early warning\")"},
        "inclusion": STRS,
        "exclusion": STRS,
        "fields": {"type": "array", "items": _FIELD_SPEC},
    }, ["fields"]),
    max_turns=6,
    long_output=True,
)


def _snake(v) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_")


def _clip(text, limit: int) -> str:
    """Shorten at a word boundary and say so, instead of cutting a word in half."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + " …"


_ROLES = {"method", "data", "outcome", "evaluation", "setting", "deployment", "reporting"}


def validate_protocol(raw: dict) -> tuple[dict, list[str]]:
    """Clean the agent's protocol into something code can rely on; report what was fixed or dropped."""
    problems: list[str] = []
    base = set(LIST_FIELDS) | set(ENUM_FIELDS)
    fields, seen = [], set()
    for f in (raw.get("fields") or []):
        if not isinstance(f, dict):
            continue
        stem = _snake(f.get("name", "")).removeprefix("q_")
        name = f"q_{stem}"
        if not stem or name in seen or stem in base:
            problems.append(f"dropped field {f.get('name')!r}: empty, duplicate or already a base field")
            continue
        ftype = "enum" if f.get("type") == "enum" else "list"
        definition = _clip(f.get("definition"), 600)
        if ftype == "enum":
            values = list(dict.fromkeys(_snake(v) for v in (f.get("values") or []) if _snake(v)))
            values = [v for v in values if v != "not_stated"]
            if len(values) < 2:
                problems.append(f"dropped enum {name}: needs at least two categories")
                continue
            values.append("not_stated")
            desirable = [d for d in (_snake(v) for v in f.get("desirable") or []) if d in values
                         and d != "not_stated"]
        else:
            values = [" ".join(str(v).split()) for v in (f.get("values") or []) if str(v).strip()][:20]
            desirable = [" ".join(str(v).split()).lower() for v in (f.get("desirable") or []) if str(v).strip()]
        search = f.get("search") if isinstance(f.get("search"), dict) else {}
        search = {(_snake(k) if ftype == "enum" else str(k).lower()): str(q)[:300]
                  for k, q in search.items() if q}
        groups, taken = [], set()
        for g in f.get("groups") or []:
            if not isinstance(g, dict):
                continue
            members = [(_snake(v) if ftype == "enum" else " ".join(str(v).split()).lower()) for v in g.get("values") or []]
            members = [v for v in dict.fromkeys(members) if v in desirable and v not in taken]
            if len(members) < 2:
                problems.append(f"{name}: dropped group {g.get('label')!r}; it needs two or more desirable values")
                continue
            taken.update(members)
            groups.append({"label": _clip(g.get("label") or " or ".join(members), 80), "values": members})
        role = f.get("role") if f.get("role") in _ROLES else ""
        fields.append({"name": name, "type": ftype, "values": values, "definition": definition,
                       "desirable": desirable, "search": search, "groups": groups, "role": role})
        seen.add(name)
        if len(fields) == MAX_FIELDS:
            break
    if not fields:
        problems.append("no usable question-specific fields")
    clean = {"setting": _clip(raw.get("setting"), 900),
             "topic_query": " ".join(str(raw.get("topic_query") or "").split())[:300],
             "inclusion": [_clip(x, 400) for x in (raw.get("inclusion") or [])][:8],
             "exclusion": [_clip(x, 400) for x in (raw.get("exclusion") or [])][:8],
             "fields": fields}
    return clean, problems


# ---------------------------------------------------------------- extraction against the protocol
def protocol_tool(protocol: dict) -> dict:
    props, names = {}, []
    for f in protocol["fields"]:
        names.append(f["name"])
        if f["type"] == "enum":
            props[f["name"]] = {"type": "string", "enum": f["values"], "description": f["definition"]}
        else:
            hint = f" Suggested values: {', '.join(f['values'])}." if f["values"] else ""
            props[f["name"]] = {**_list, "description": f["definition"] + hint}
    props["evidence"] = {
        "type": "object",
        "description": "For each field with a value (not not_stated / empty), 1-2 short passages (up to 25 "
                       "words) COPIED WORD FOR WORD from the text that show it. Checked by code.",
        "properties": {n: _list for n in names},
    }
    return {"name": "record_protocol_fields", "description": "Record the question-specific fields.",
            "input_schema": {"type": "object", "properties": props, "required": names + ["evidence"]}}


PROTOCOL_EXTRACTION_SYSTEM = """You extract question-specific facts from a health research paper.
Research question: {question}
Rules:
- Record only what the text states. If it is not stated, use not_stated or an empty list. Never guess.
- Follow each field's definition exactly; pick the single best category for enum fields.
- evidence: copy the exact words that show each value. If you cannot point to words in the text, the value is
  a guess: leave it out."""


def _normalise_protocol(data: dict, protocol: dict) -> dict:
    out = {}
    for f in protocol["fields"]:
        v = data.get(f["name"])
        if f["type"] == "enum":
            v = _snake(v or "not_stated")
            out[f["name"]] = v if v in f["values"] else "not_stated"
        else:
            vals = v if isinstance(v, list) else ([x for x in str(v).split(",")] if v else [])
            seen, clean = set(), []
            for x in vals:
                x = " ".join(str(x).split()).strip(" .;")
                if x and x.lower() not in seen and x.lower() not in {"not stated", "none", "n/a"}:
                    seen.add(x.lower()); clean.append(x)
            out[f["name"]] = clean[:10]
    return out


def _paper_text(pg, paper: dict, read_full: bool) -> tuple[str, str]:
    text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    if read_full:
        row = pg.execute("SELECT sections FROM paper_fulltext WHERE paper_id=%s AND status='ok'",
                         (paper["paper_id"],)).fetchone()
        if row and row["sections"]:
            return "fulltext", text + "\n\nSelected full-text sections:\n" + select_for_reading(row["sections"])
    return "abstract", text


def extract_protocol_fields(ctx, protocol: dict) -> dict:
    """Extract this run's protocol fields for every extracted shortlisted paper (skips ones already done).
    Papers read in full for the base extraction are read in full here too."""
    papers = {r["paper_id"]: r for r in ctx.pg.execute(
        """SELECT p.paper_id, p.title, p.abstract, e.source FROM run_papers rp
           JOIN papers p USING (paper_id)
           JOIN extractions e ON e.paper_id = rp.paper_id AND e.schema_version = %s
           WHERE rp.run_id = %s""", (ctx.extraction_version, ctx.run_id)).fetchall()}
    done_ids = {r["paper_id"] for r in ctx.pg.execute(
        "SELECT paper_id FROM protocol_extractions WHERE run_id=%s", (ctx.run_id,)).fetchall()}
    todo = [pid for pid in papers if pid not in done_ids]
    tool = protocol_tool(protocol)
    names = [f["name"] for f in protocol["fields"]]
    list_names = [f["name"] for f in protocol["fields"] if f["type"] == "list"]
    system = PROTOCOL_EXTRACTION_SYSTEM.format(question=ctx.question) + "\n\nField definitions:\n" + "\n".join(
        f"- {f['name']}: {f['definition']}" for f in protocol["fields"])
    llm = ctx.llm_factory()
    failed, dropped = [], {}

    def work(pid: str):
        conn = connect()
        try:
            depth, text = _paper_text(conn, papers[pid], papers[pid]["source"] == "fulltext")
            resp = llm.chat(system, [{"role": "user", "content": [{"type": "text", "text": text}]}],
                            tools=[tool], force_tool="record_protocol_fields",
                            max_tokens=settings.extraction_max_tokens)
            if not resp.tool_calls:
                raise ValueError("model returned no protocol extraction")
            args = resp.tool_calls[0].input or {}
            if "_raw_arguments" in args:
                raise ValueError("protocol extraction was cut off; raise EXTRACTION_MAX_TOKENS")
            data = check_evidence(_normalise_protocol(args, protocol), args.get("evidence"), text,
                                  fields=names, list_fields=list_names)
            conn.execute(
                "INSERT INTO protocol_extractions (run_id, paper_id, source, data) VALUES (%s,%s,%s,%s::jsonb) "
                "ON CONFLICT (run_id, paper_id) DO UPDATE SET source=EXCLUDED.source, data=EXCLUDED.data",
                (ctx.run_id, pid, depth, json.dumps(data)))
            return data
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=settings.extraction_workers) as pool:
        futures = {pool.submit(work, pid): pid for pid in todo}
        for fut in as_completed(futures):
            try:
                data = fut.result()
                for f in (data.get("_unverified") or {}):
                    dropped[f] = dropped.get(f, 0) + 1
            except Exception as exc:
                failed.append({"paper_id": futures[fut], "error": str(exc)[:200]})
    return {"papers": len(papers), "newly_extracted": len(todo) - len(failed), "reused": len(done_ids),
            "failed": failed, "fields_dropped_without_evidence": dropped}
