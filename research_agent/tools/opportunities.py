"""A research opportunity as a thing in its own right, not a paragraph in a report.

Gaps, untried combinations, candidate designs and the researcher's own findings all answer the same
question: here is something worth doing, and here is why. Until now each lived inside one run's notes as
JSON, so it could be read once in a report and never again: not revised, not challenged, not compared with
what a later run found, not exported on its own.

This records each as a row carrying what a researcher has to see before spending a year on it:

    what it is            kind, label, the research question it implies
    what supports it      the claims and papers behind it
    what weakens it       claims on the same field that the measurement contradicts or cannot settle
    what is unresolved    hypotheses tested and not settled, alternative explanations still standing
    what it would take    candidate methods, data and validation, when a design agent has proposed them
    where it came from    the agent, the step and the numbers it was computed from

Nothing is invented here. Every field is carried over from something the run already computed and checked,
so an opportunity can always be traced back to the papers underneath it.
"""
from __future__ import annotations

KINDS = ["gap", "combination", "design", "finding", "direction"]
STATES = ["open", "addressed", "dismissed", "superseded"]
# A claim whose state is one of these cannot support anything; it is listed as what weakens the opportunity.
_WEAK = ("contradicted", "not_searched_enough", "uncertain")


def _claims_by_field(ctx) -> dict[str, list[dict]]:
    """The run's verified claims, grouped by the extracted field they counted."""
    out: dict[str, list[dict]] = {}
    for c in ctx.pg.execute(
            "SELECT id, text, predicate, status, state FROM claims WHERE run_id=%s AND status <> 'rejected' "
            "ORDER BY id", (ctx.run_id,)).fetchall():
        field = (c["predicate"] or {}).get("field")
        if field:
            out.setdefault(field, []).append(c)
    return out


def _split(claims: list[dict]) -> tuple[list[str], list[str]]:
    """(what supports, what weakens) as claim ids, judged by the state code, not by the claim's wording."""
    good = [f"C{c['id']}" for c in claims if c["state"] in ("supported", "partially_supported")]
    weak = [f"C{c['id']}" for c in claims if c["state"] in _WEAK or c["status"] == "unsupported"]
    return good, sorted(set(weak))


def record(ctx) -> dict:
    """Write this run's opportunities as rows. Safe to re-run: an item keeps its id and is updated."""
    notes = ctx.notes()
    m = (notes.get("map") or {}).get("map") or {}
    reasoning = {g["gap_id"]: g for g in ((notes.get("gap_reasoning") or {}).get("checked") or {}).get("gaps", [])}
    hyps = {h["id"]: h for h in ((notes.get("hypotheses") or {}).get("items") or [])}
    designs = (notes.get("design") or {}).get("designs") or []
    by_field = _claims_by_field(ctx)
    rows: list[dict] = []

    def unresolved(item_id: str) -> tuple[list[str], list[str]]:
        """Hypotheses tested for this item that did not settle, and the explanations still standing."""
        mine = [h for h in hyps.values() if h.get("gap_id") == item_id]
        open_q = [h["text"] for h in mine if h.get("verdict") not in ("supported", "not_supported")]
        alt = [h["text"] for h in mine if h.get("verdict") == "supported" and not h.get("restates_item")]
        r = reasoning.get(item_id) or {}
        open_q += [str(x) for x in (r.get("open_questions") or r.get("what_would_settle_it") or [])]
        return list(dict.fromkeys(open_q))[:8], list(dict.fromkeys(alt))[:8]

    for it in m.get("gaps", []):
        good, weak = _split(by_field.get(it.get("field"), []))
        open_q, alt = unresolved(it["id"])
        rows.append({
            "item_id": it["id"], "kind": "gap", "label": it.get("label", "")[:300],
            "question": (reasoning.get(it["id"], {}) or {}).get("research_question"),
            "field": it.get("field"), "value": str(it.get("value") or "")[:300],
            "confidence": it.get("confidence"),
            "evidence": {"supporting_claims": good, "weakening_claims": weak,
                         "supporting_papers": (it.get("paper_ids") or [])[:20],
                         "measured": {k: it[k] for k in ("n", "N", "share", "ci95", "stated_rate") if k in it},
                         "unresolved_questions": open_q, "alternative_explanations": alt},
            "provenance": {"agent": "map", "step": "compute_map", "item": it["id"]}})

    for it in m.get("novelty", []):
        a, b = it.get("a") or {}, it.get("b") or {}
        good_a, weak_a = _split(by_field.get(a.get("field"), []))
        good_b, weak_b = _split(by_field.get(b.get("field"), []))
        rows.append({
            "item_id": it["id"], "kind": "combination",
            "label": f"{a.get('label', '')} with {b.get('label', '')}"[:300],
            "question": f"What would {a.get('label')} together with {b.get('label')} show?",
            "field": a.get("field"), "value": str(a.get("value") or "")[:300],
            "confidence": it.get("confidence"),
            "evidence": {"supporting_claims": sorted(set(good_a + good_b)),
                         "weakening_claims": sorted(set(weak_a + weak_b)),
                         # the other side, kept as a field and a value rather than only inside the label, so
                         # a precedent check can ask for both sides without parsing prose
                         "second_field": b.get("field"), "second_value": [str(b.get("value") or "")],
                         # each side separately IS the prior art: these are the studies nearest to it
                         "prior_studies": (it.get("a_paper_ids") or [])[:10] + (it.get("b_paper_ids") or [])[:10],
                         "measured": {k: it[k] for k in ("observed", "expected", "p", "n_a", "n_b") if k in it},
                         "unresolved_questions": [], "alternative_explanations": []},
            "provenance": {"agent": "map", "step": "novelty", "item": it["id"]}})

    for d in designs:
        addressed = d.get("addresses") or []
        good = sorted({c for a in addressed for c in (next((r["evidence"]["supporting_claims"]
                                                            for r in rows if r["item_id"] == a), []))})
        weak = sorted({c for a in addressed for c in (next((r["evidence"]["weakening_claims"]
                                                            for r in rows if r["item_id"] == a), []))})
        rows.append({
            "item_id": d["id"], "kind": "design", "label": (d.get("title") or d["id"])[:300],
            "question": d.get("research_question"), "field": None, "value": None,
            "confidence": None,
            "evidence": {"supporting_claims": good, "weakening_claims": weak,
                         "supporting_papers": (d.get("builds_on") or [])[:20],
                         "prior_studies": [p["paper_id"] for p in (d.get("supporting") or [])][:10],
                         "challenging_papers": [p["paper_id"] for p in (d.get("challenging") or [])][:10],
                         "addresses": addressed,
                         "candidate_methods": [x for x in [d.get("target"), d.get("predictors"),
                                                           d.get("baseline")] if x],
                         "required_data": [x for x in [d.get("data_sources"), d.get("spatial_unit"),
                                                       d.get("horizon")] if x],
                         "validation_requirements": [d["validation_strategy"]] if d.get("validation_strategy") else [],
                         "unresolved_questions": [h["text"] for h in (d.get("rests_on") or [])
                                                  if h.get("verdict") != "supported"],
                         "alternative_explanations": list(d.get("risks") or [])[:8]},
            "provenance": {"agent": "design", "step": "check_designs", "item": d["id"]}})

    # an ordinary (non-map) run: the Gap agent's gaps and the directions it suggested
    for i, g in enumerate((notes.get("gaps") or {}).get("gaps") or [], 1):
        if not isinstance(g, dict) or not g.get("gap"):
            continue
        cids = [str(c).upper() for c in (g.get("claim_ids") or [])]
        cids = [c if c.startswith("C") else f"C{c}" for c in cids]
        held = {f"C{c['id']}": c for cs in by_field.values() for c in cs}
        rows.append({
            "item_id": f"X{i}", "kind": "gap", "label": str(g["gap"])[:300],
            "question": g.get("why_it_matters"), "field": None, "value": None,
            "confidence": g.get("confidence"),
            "evidence": {"supporting_claims": [c for c in cids if held.get(c, {}).get("state")
                                               in ("supported", "partially_supported")],
                         "weakening_claims": [c for c in cids if held.get(c, {}).get("state") in _WEAK],
                         "supporting_papers": [], "unresolved_questions": [], "alternative_explanations": []},
            "provenance": {"agent": "gaps", "step": "gap_agent", "item": f"X{i}"}})
    for i, text in enumerate((notes.get("gaps") or {}).get("research_directions") or [], 1):
        rows.append({"item_id": f"R{i}", "kind": "direction", "label": str(text)[:300], "question": str(text),
                     "field": None, "value": None, "confidence": None,
                     "evidence": {"supporting_claims": [], "weakening_claims": []},
                     "provenance": {"agent": "gaps", "step": "research_directions", "item": f"R{i}"}})

    rows += _from_researchers(ctx)
    _write(ctx, rows)
    return {"recorded": len(rows), "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in KINDS
                                               if any(r["kind"] == k for r in rows)}}


def _from_researchers(ctx) -> list[dict]:
    """Findings the open-ended researcher confirmed on this run, with the grade code gave them."""
    try:
        found = ctx.pg.execute(
            """SELECT f.id, f.statement, f.grade, f.status, f.reasons, f.evidence, r.id rid
               FROM research_findings f JOIN researchers r ON r.id = f.researcher_id
               WHERE r.run_id = %s AND f.status <> 'rejected' ORDER BY f.id""", (ctx.run_id,)).fetchall()
    except Exception:                       # a database from before the researcher existed
        return []
    out = []
    for f in found:
        ev = f["evidence"] or {}
        out.append({
            "item_id": f"F{f['id']}", "kind": "finding", "label": str(f["statement"])[:300],
            "question": None, "field": None, "value": None,
            "confidence": {"grade": f["grade"], "reasons": f["reasons"] or []},
            "evidence": {"supporting_claims": [], "weakening_claims": [],
                         "supporting_papers": (ev.get("examples") or [])[:20],
                         "measured": ev.get("discovery") or {},
                         "held_out": ev.get("holdout") or {},
                         # a rival the gauntlet could not rule out is exactly what is still unresolved
                         "unresolved_questions": [str(a.get("explanation")) for a in (ev.get("rivals") or [])
                                                  if a.get("verdict") == "inconclusive"][:8],
                         "alternative_explanations": [str(a.get("explanation")) for a in
                                                      (ev.get("rivals") or [])][:8]},
            "provenance": {"agent": "researcher", "step": "gauntlet", "researcher_id": f["rid"],
                           "finding_id": f["id"]}})
    return out


def _write(ctx, rows: list[dict]) -> None:
    import json

    for r in rows:
        ctx.pg.execute(
            """INSERT INTO research_opportunities
                   (run_id, item_id, kind, label, question, field, value, confidence, evidence, provenance)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
               ON CONFLICT (run_id, item_id) DO UPDATE SET
                   kind=EXCLUDED.kind, label=EXCLUDED.label, question=EXCLUDED.question,
                   field=EXCLUDED.field, value=EXCLUDED.value, confidence=EXCLUDED.confidence,
                   evidence=EXCLUDED.evidence, provenance=EXCLUDED.provenance""",
            (ctx.run_id, r["item_id"], r["kind"], r["label"], r.get("question"), r.get("field"), r.get("value"),
             json.dumps(r.get("confidence"), default=str), json.dumps(r.get("evidence") or {}, default=str),
             json.dumps(r.get("provenance") or {}, default=str)))


def listing(ctx, kind: str | None = None, state: str | None = None) -> list[dict]:
    sql = ("SELECT id, item_id, kind, label, question, field, value, state, confidence, evidence, provenance, "
           "created_at FROM research_opportunities WHERE run_id=%s")
    params: list = [ctx.run_id]
    if kind:
        sql += " AND kind=%s"; params.append(kind)
    if state:
        sql += " AND state=%s"; params.append(state)
    return ctx.pg.execute(sql + " ORDER BY array_position(%s, kind), item_id",
                          params + [KINDS]).fetchall()


def set_state(ctx, item_id: str, state: str, note: str = "") -> dict:
    if state not in STATES:
        return {"error": f"state must be one of {', '.join(STATES)}"}
    row = ctx.pg.execute("UPDATE research_opportunities SET state=%s, state_note=%s "
                         "WHERE run_id=%s AND item_id=%s RETURNING item_id, state",
                         (state, note or None, ctx.run_id, item_id)).fetchone()
    return dict(row) if row else {"error": f"{item_id} is not an opportunity of this run"}


def markdown(ctx) -> list[str]:
    """The opportunities table for the report: what is worth doing, and what each one rests on."""
    rows = listing(ctx)
    if not rows:
        return []
    L = ["## Research opportunities (computed)", "",
         "Each is recorded with the claims that support it, the claims that weaken it, and what is still "
         "unresolved, so it can be checked rather than taken on trust.",
         "", "| id | What | Supported by | Weakened by | Still unresolved |", "|---|---|---|---|---|"]
    for r in rows:
        e = r["evidence"] or {}
        unres = len(e.get("unresolved_questions") or []) + len(e.get("alternative_explanations") or [])
        L.append(f"| {r['item_id']} | {str(r['label'])[:80]} | "
                 f"{', '.join(e.get('supporting_claims') or []) or '–'} | "
                 f"{', '.join(e.get('weakening_claims') or []) or '–'} | {unres or '–'} |")
    return L + [""]
