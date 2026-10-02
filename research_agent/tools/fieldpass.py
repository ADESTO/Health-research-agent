"""One field, read for on purpose, in the papers that left it blank.

A paper's record is written in a single reading that has to fill fifteen general fields and the run's
question-specific ones at once. A detail the question turns on, stated once in a methods section, loses to
that competition and the field comes back "not stated". Counting it then measures the reading, not the
literature: a malaria run found 33 of 45 studies with no validation design recorded, read four of them in
full, and every one described a held-out design in its text.

This is the repair. It takes one field, finds the papers whose record leaves it empty, shows each paper's
full text, and asks about that field alone:

    "What validation design does THIS paper use? Copy the sentence that says so."

Nothing is taken on the model's word. A value counts only with a passage checked against the text the model
was shown, exactly as in the first reading, and a paper that genuinely does not say stays "not stated". Only
the one field is written, so the rest of the record is left as it was and the pass can be repeated for
another field without undoing this one.

The other way a field fails is worse, because it looks full. A category that merges two things the question
has to tell apart cannot be un-merged by reading harder: a run asking how often forecasts are benchmarked
against a SEASONAL baseline found its field offered only "naive_or_seasonal_naive_baseline", which counts
plain persistence and seasonal naive as one thing, and the question died there. So a pass can also define a
finer field and code every paper into it from full text, leaving the old field untouched beside it.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings
from research_agent.db import connect
from research_agent.tools.base import INT, STR, Tool, obj

MAX_PAPERS = 60               # papers per pass, so one call cannot spend a run's budget

SYSTEM = """You read one research paper to answer ONE question about it.

{question}

{values}
Rules:
- Answer only from the text shown, about what THIS paper did. A method named in the background, in related
  work or as future work is not what the paper did.
- quote: copy the sentence that shows your answer, word for word from the text above. Code checks it
  against the paper; an answer whose quote is not found is discarded.
- If the text does not say, answer {blank} and leave the quote empty. That is a useful answer, not a
  failure: this field being genuinely unstated is itself what the run is measuring."""


def _base_definition(field: str) -> str:
    """The wording the extraction form itself uses for a field, so a field pass reads papers by the same
    rule the first read was meant to follow. Without it a pass would judge `validation_level` by the name
    alone, which is exactly how a later-period test gets filed as a different-place one."""
    from research_agent.tools.extraction import EXTRACTION_TOOL

    spec = EXTRACTION_TOOL["input_schema"]["properties"].get(field) or {}
    return spec.get("description") or f"The paper's {field.replace('_', ' ')}."


def _field_def(ctx, field: str) -> dict | None:
    """The field as the run defines it: its type, its allowed values and the definition papers are read by."""
    from research_agent.tools.extraction import LIST_FIELDS, enums_for, protocol_of

    for f in (protocol_of(ctx) or {}).get("fields", []):
        if f["name"] == field:
            return {"name": field, "type": f["type"], "values": f.get("values") or [],
                    "definition": f.get("definition") or "", "protocol": True}
    enums = enums_for(getattr(ctx, "extraction_version", None))
    if field in enums:
        return {"name": field, "type": "enum", "values": enums[field],
                "definition": _base_definition(field), "protocol": False}
    if field in LIST_FIELDS:
        return {"name": field, "type": "list", "values": [],
                "definition": f"The paper's {field.replace('_', ' ')}.", "protocol": False}
    return None


def _tool(fd: dict) -> dict:
    if fd["type"] == "enum":
        value = {"type": "string", "enum": fd["values"], "description": fd["definition"][:500]}
    else:
        value = {"type": "array", "items": {"type": "string"},
                 "description": fd["definition"][:500] + " Empty list if the text does not say."}
    return {"name": "record_field", "description": f"Record {fd['name']} for this paper.",
            "input_schema": obj({"value": value,
                                 "quote": {**STR, "description": "The sentence that shows it, copied exactly "
                                                                 "from the text. Empty if not stated."}},
                                ["value", "quote"])}


def _blank(fd: dict) -> str:
    return "'not_stated'" if fd["type"] == "enum" else "an empty list"


def _ask(llm, fd: dict, text: str) -> dict:
    """One paper, one field. Returns {value, quote} with the quote already checked against the text."""
    from research_agent.tools.extraction import _words, quote_found

    values = (f"Allowed answers: {', '.join(fd['values'])}.\n" if fd["type"] == "enum"
              else "Answer with short canonical names, 1 to 5 words each.\n")
    system = SYSTEM.format(question=f"{fd['name'].removeprefix('q_').replace('_', ' ')}: {fd['definition']}",
                           values=values, blank=_blank(fd))
    resp = llm.chat(system, [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    tools=[_tool(fd)], force_tool="record_field", max_tokens=600)
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    quote = " ".join(str(args.get("quote") or "").split())[:400]
    if fd["type"] == "enum":
        value = args.get("value") if args.get("value") in fd["values"] else "not_stated"
        empty = value in ("", "not_stated", "none", "not_reported")
    else:
        raw = args.get("value") or []
        value = [" ".join(str(v).split())[:80] for v in (raw if isinstance(raw, list) else [raw]) if str(v).strip()]
        empty = not value
    if empty:
        return {"value": "not_stated" if fd["type"] == "enum" else [], "quote": "", "filled": False}
    words = _words(text)
    shingles = {tuple(words[i:i + k]) for k in (3, 4, 5) for i in range(len(words) - k + 1)}
    if not quote_found(quote, words, shingles):
        # an answer whose quote is not in the paper is the thing this pass exists to avoid
        return {"value": "not_stated" if fd["type"] == "enum" else [], "quote": "", "filled": False,
                "dropped_without_quote": True}
    return {"value": value, "quote": quote, "filled": True}


def _blank_now(data: dict, fd: dict) -> bool:
    v = data.get(fd["name"])
    return not v or (isinstance(v, str) and v in ("not_stated", "not_reported", "none", ""))


def add_field(ctx, name: str, values: list[str], definition: str, kind: str = "enum") -> dict:
    """Add a finer field to this run's protocol, to be coded from full text by a pass.

    For when the first field cannot answer the question because two things the question separates were
    written as one category. The old field is left exactly as it is, so nothing already counted changes."""
    from research_agent.opportunity.protocol import _clip, _snake, _unbundle

    stem = _snake(name).removeprefix("q_")
    if not stem:
        return {"error": "the field needs a name"}
    field = f"q_{stem}"
    fd = _field_def(ctx, field)
    if fd:
        return {"error": f"{field} is already a field of this run; fill it instead, or choose another name"}
    kind = "enum" if kind != "list" else "list"
    vals = [v for v in dict.fromkeys(_snake(v) for v in values or []) if v and v != "not_stated"]
    if kind == "enum" and len(vals) < 2:
        return {"error": "an enum field needs at least two categories"}
    notes = ctx.notes().get("protocol") or {}
    protocol = notes.get("protocol")
    if not isinstance(protocol, dict) or not isinstance(protocol.get("fields"), list):
        return {"error": "this run has no protocol to add a field to"}
    problems: list[str] = []
    if kind == "enum":
        vals, unbundled = _unbundle(vals)
        problems += unbundled
        vals.append("not_stated")
    protocol = {**protocol, "fields": list(protocol["fields"]) + [
        {"name": field, "type": kind, "values": vals, "definition": _clip(definition, 600),
         "desirable": [], "search": {}, "groups": [], "role": "", "added_by": "fieldpass"}]}
    ctx.save_note("protocol", {**notes, "protocol": protocol})
    return {"field": field, "type": kind, "values": vals, "notes": problems,
            "next": f"fill it with a pass: it is empty for every paper until then"}


def fill_field(ctx, field: str, limit: int | None = None, paper_ids: list[str] | None = None) -> dict:
    """Read the full text of the run's papers that leave `field` blank, asking about that field alone.

    Only that field is written, with its quote. A paper that genuinely does not say stays blank, and a
    paper with no full text available is reported rather than guessed at."""
    from research_agent.ingestion.fulltext import fetch_fulltext, select_for_reading
    from research_agent.tools.extraction import _rows

    fd = _field_def(ctx, field)
    if not fd:
        return {"error": f"{field!r} is not a field of this run; use one of the extracted or q_ fields"}
    rows = {r["paper_id"]: r for r in _rows(ctx)}
    blank = [pid for pid, r in rows.items() if _blank_now(r["data"], fd)]
    if paper_ids:
        blank = [p for p in blank if p in set(paper_ids)]
    todo = blank[: max(1, min(int(limit), MAX_PAPERS)) if limit else MAX_PAPERS]
    if not todo:
        return {"field": field, "blank_before": 0, "note": "every paper already records this field"}

    status = fetch_fulltext(ctx.pg, todo)
    texts = {r["paper_id"]: select_for_reading(r["sections"] or [], budget=settings.fulltext_read_chars)
             for r in ctx.pg.execute("SELECT paper_id, sections FROM paper_fulltext WHERE paper_id = ANY(%s) "
                                     "AND status='ok'", (todo,)).fetchall()}
    meta = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, title, abstract FROM papers WHERE paper_id = ANY(%s)", (todo,)).fetchall()}
    llm = ctx.llm_factory(step="fieldpass")
    setattr(llm, "_step", "fieldpass")
    results: dict[str, dict] = {}

    def work(pid):
        p = meta[pid]
        text = f"Title: {p['title']}\n\nAbstract: {p['abstract']}"
        if texts.get(pid):
            text += f"\n\nFull-text sections:\n{texts[pid]}"
        return pid, _ask(llm, fd, text)

    with ThreadPoolExecutor(max_workers=settings.extraction_workers) as pool:
        futures = [pool.submit(work, pid) for pid in todo if pid in meta and texts.get(pid)]
        for fut in as_completed(futures):
            try:
                pid, res = fut.result()
                results[pid] = res
            except Exception as exc:    # one failed call leaves that paper blank, not the pass broken
                ctx.emit("fieldpass", "error", {"error": str(exc)[:200]})

    filled = {pid: r for pid, r in results.items() if r["filled"]}
    _write(ctx, fd, filled)
    no_text = [pid for pid in todo if not texts.get(pid)]
    out = {"field": field, "blank_before": len(blank), "read_now": len(results), "filled": len(filled),
           "still_not_stated": len(results) - len(filled),
           "dropped_without_a_quote": sum(1 for r in results.values() if r.get("dropped_without_quote")),
           "no_full_text_available": no_text,
           "fetch": {s: sum(1 for v in status.values() if v == s) for s in set(status.values())},
           "values_found": _counts(filled),
           "examples": [{"paper_id": p, "value": r["value"], "quote": r["quote"][:200]}
                        for p, r in list(filled.items())[:5]]}
    ctx.save_note(f"fieldpass:{field}", out)
    return out


def _counts(filled: dict) -> dict:
    out: dict[str, int] = {}
    for r in filled.values():
        for v in (r["value"] if isinstance(r["value"], list) else [r["value"]]):
            out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _write(ctx, fd: dict, filled: dict) -> None:
    """Write the one field into the paper's record, keeping everything else and the quote beside it."""
    if not filled:
        return
    conn = connect()
    try:
        for pid, r in filled.items():
            if fd["protocol"]:
                row = conn.execute("SELECT data FROM protocol_extractions WHERE run_id=%s AND paper_id=%s",
                                   (ctx.run_id, pid)).fetchone()
            else:
                row = conn.execute("SELECT data FROM extractions WHERE schema_version=%s AND paper_id=%s",
                                   (ctx.extraction_version, pid)).fetchone()
            data = dict((row or {}).get("data") or {})
            data[fd["name"]] = r["value"]
            ev = dict(data.get("evidence") or {})
            ev[fd["name"]] = [r["quote"]]
            data["evidence"] = ev
            # provenance: this value came from a pass that read for this field alone, not the first reading
            data.setdefault("_filled_by_pass", {})[fd["name"]] = {"quote": r["quote"][:300]}
            if isinstance(data.get("_unverified"), dict):      # it is verified now
                data["_unverified"].pop(fd["name"], None)
            if fd["protocol"]:
                conn.execute(
                    "INSERT INTO protocol_extractions (run_id, paper_id, source, data) VALUES (%s,%s,%s,%s::jsonb) "
                    "ON CONFLICT (run_id, paper_id) DO UPDATE SET data=EXCLUDED.data",
                    (ctx.run_id, pid, "fulltext", json.dumps(data, default=str)))
            else:
                conn.execute(
                    "UPDATE extractions SET data=%s::jsonb WHERE schema_version=%s AND paper_id=%s",
                    (json.dumps(data, default=str), ctx.extraction_version, pid))
    finally:
        conn.close()


def field_pass(ctx, field: str, limit: int | None = None, values: list[str] | None = None,
               definition: str = "", kind: str = "enum") -> dict:
    """Fill a field from full text. With `values` and `definition`, first add it as a new, finer field."""
    added = None
    if values:
        added = add_field(ctx, field, values, definition, kind)
        if "error" in added:
            return added
        field = added["field"]
    res = fill_field(ctx, field, limit=limit)
    return {**res, **({"added_field": added} if added else {})}


FIELDPASS_TOOL = Tool(
    "fill_field_from_full_text",
    "Read the full text of the papers that leave one field blank, asking about that field alone, and record "
    "what they say with a quote checked against the text. Use it when a count on a field looks like it "
    "measures how the papers were read rather than what they did ('not stated' in most of them). Only that "
    "field is written; a paper that genuinely does not say stays blank.",
    obj({"field": {**STR, "description": "The field to fill, e.g. q_validation_split or validation_level. "
                                         "With `values`, the name of a NEW field to define and then fill."},
         "limit": {**INT, "description": f"How many papers (default and maximum {MAX_PAPERS})"},
         "values": {"type": "array", "items": {"type": "string"},
                    "description": "Only to define a NEW, finer field: its categories. Use this when the "
                                   "existing field merges two things the question has to tell apart (a "
                                   "category joining them with 'or'), which no amount of re-reading can "
                                   "undo. The old field is left as it is."},
         "definition": {**STR, "description": "With `values`: what the new field means, written so a reader "
                                              "can decide the value from the paper's text alone."}},
        ["field"]),
    field_pass)
