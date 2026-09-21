"""Claims and deterministic evidence checks.

Analysis agents do not write free-floating statements into the report. They *propose claims*, each
with a machine-checkable predicate. The Evidence layer evaluates predicates with plain code against
the extraction table (prevalence claims) or the corpus (trend claims). The Synthesis agent may only
state claims whose status is `supported`, and must quote the computed numbers.

prevalence predicate:
    {"field": "datasets", "any_of": ["MIMIC-III", "MIMIC-IV", "MIMIC"], "over": "all"|"stated",
     "min_share": 0.5, "max_share": null, "min_count": null, "max_count": null}
trend predicate:
    {"keywords": "\"foundation model\"", "early": [2018, 2020], "late": [2023, 2025],
     "direction": "increase"|"decrease", "min_ratio": 1.5}
"""
from __future__ import annotations

import json

from research_agent.tools.base import INT, NUM, STR, STRS, Tool, obj
from research_agent.tools.extraction import ENUM_FIELDS, LIST_FIELDS, _rows, _values
from research_agent.tools.trends import topic_series, window_ratio


def _validate(claim_type: str, p: dict) -> str | None:
    if claim_type == "prevalence":
        if p.get("field") not in LIST_FIELDS + list(ENUM_FIELDS):
            return f"field must be one of {LIST_FIELDS + list(ENUM_FIELDS)}"
        if not p.get("any_of"):
            return "any_of must list at least one value (include synonyms/spellings)"
        if p.get("over", "all") not in ("all", "stated"):
            return "over must be 'all' or 'stated'"
        if all(p.get(k) is None for k in ("min_share", "max_share", "min_count", "max_count")):
            return "give at least one bound (min_share/max_share/min_count/max_count) so the claim is testable"
        return None
    if claim_type == "trend":
        for k in ("keywords", "early", "late", "direction"):
            if not p.get(k):
                return f"trend predicate needs {k}"
        if p["direction"] not in ("increase", "decrease"):
            return "direction must be increase or decrease"
        return None
    return "claim_type must be 'prevalence' or 'trend'"


def propose_claim(ctx, text: str, claim_type: str, predicate: dict, _agent: str = "unknown") -> dict:
    err = _validate(claim_type, predicate or {})
    if err:
        return {"error": err}
    r = ctx.pg.execute(
        "INSERT INTO claims (run_id, agent, text, claim_type, predicate) VALUES (%s,%s,%s,%s,%s::jsonb) RETURNING id",
        (ctx.run_id, _agent, text, claim_type, json.dumps(predicate))).fetchone()
    return {"claim_id": r["id"], "status": "pending", "note": "Evidence agent will verify it."}


def evaluate_prevalence(ctx, p: dict) -> dict:
    rows = _rows(ctx)
    field, needles = p["field"], [s.lower().strip() for s in p["any_of"] if s.strip()]
    exact = field in ENUM_FIELDS
    matched, matched_values = [], set()
    stated = 0
    for r in rows:
        vals = _values(r["data"], field)
        if vals:
            stated += 1
        hits = [v for v in vals if any((n == v.lower()) if exact else (n in v.lower()) for n in needles)]
        if hits:
            matched.append(r["paper_id"])
            matched_values.update(h.lower() for h in hits)
    over = p.get("over", "all")
    denom = stated if over == "stated" else len(rows)
    n = len(matched)
    share = (n / denom) if denom else 0.0
    checks = []
    if p.get("min_share") is not None: checks.append(share >= float(p["min_share"]))
    if p.get("max_share") is not None: checks.append(share <= float(p["max_share"]))
    if p.get("min_count") is not None: checks.append(n >= int(p["min_count"]))
    if p.get("max_count") is not None: checks.append(n <= int(p["max_count"]))
    abstract_only = sum(1 for r in rows if r["source"] == "abstract")
    ok = bool(denom) and all(checks)
    return {
        "supported": ok, "n_matching": n, "denominator": denom, "over": over,
        "share": round(share, 3), "n_papers_extracted": len(rows), "n_field_stated": stated,
        "matched_paper_ids": matched[:60], "matched_values": sorted(matched_values)[:25],
        "caveat": (f"{abstract_only}/{len(rows)} papers were read from abstracts only; absence of a value "
                   f"often means 'not reported in the abstract'.") if abstract_only else "",
    }


def evaluate_trend(ctx, p: dict) -> dict:
    series = topic_series(ctx, p["keywords"])["series"]
    w = window_ratio(series, tuple(p["early"]), tuple(p["late"]))
    ratio = w["ratio_late_over_early"]
    min_ratio = float(p.get("min_ratio") or 1.5)
    if ratio is None:
        ok = False
    elif ratio == "inf":
        ok = p["direction"] == "increase"
    else:
        ok = ratio >= min_ratio if p["direction"] == "increase" else ratio <= 1 / min_ratio
    total = sum(s["matching"] for s in series)
    return {"supported": ok and total >= 10, "window": w, "min_ratio": min_ratio, "total_matching_papers": total,
            "caveat": "" if total >= 10 else "fewer than 10 matching papers — too few to call a trend"}


def evaluate(ctx, claim: dict) -> dict:
    p = claim["predicate"]
    return evaluate_prevalence(ctx, p) if claim["claim_type"] == "prevalence" else evaluate_trend(ctx, p)


def verify_claims(ctx, claim_ids: list[int] | None = None) -> dict:
    sql = "SELECT * FROM claims WHERE run_id=%s AND status <> 'rejected'"
    params: list = [ctx.run_id]
    if claim_ids:
        sql += " AND id = ANY(%s)"; params.append(list(claim_ids))
    else:
        sql += " AND status = 'pending'"
    out = []
    for c in ctx.pg.execute(sql, params).fetchall():
        res = evaluate(ctx, c)
        status = "supported" if res["supported"] else "unsupported"
        ctx.pg.execute("UPDATE claims SET status=%s, result=%s::jsonb WHERE id=%s",
                       (status, json.dumps(res, default=str), c["id"]))
        brief = {k: res[k] for k in ("n_matching", "denominator", "share", "matched_values", "caveat")
                 if k in res} if c["claim_type"] == "prevalence" else \
                {"window": res["window"], "total_matching_papers": res["total_matching_papers"]}
        out.append({"claim_id": c["id"], "agent": c["agent"], "text": c["text"], "status": status, **brief})
    return {"verified": len(out), "claims": out}


def list_claims(ctx, status: str | None = None) -> dict:
    sql = "SELECT id, agent, text, claim_type, predicate, status, review_note FROM claims WHERE run_id=%s"
    params: list = [ctx.run_id]
    if status:
        sql += " AND status=%s"; params.append(status)
    return {"claims": ctx.pg.execute(sql + " ORDER BY id", params).fetchall()}


def reject_claim(ctx, claim_id: int, reason: str) -> dict:
    ctx.pg.execute("UPDATE claims SET status='rejected', review_note=%s WHERE id=%s AND run_id=%s",
                   (reason, claim_id, ctx.run_id))
    return {"claim_id": claim_id, "status": "rejected"}


def revise_claim(ctx, claim_id: int, predicate: dict, note: str, text: str | None = None) -> dict:
    """Fix a predicate that does not faithfully test its claim text (e.g. missing synonyms).
    Allowed once per claim, and logged — revising until a claim passes is not allowed."""
    c = ctx.pg.execute("SELECT * FROM claims WHERE id=%s AND run_id=%s", (claim_id, ctx.run_id)).fetchone()
    if not c:
        return {"error": "no such claim"}
    if (c["review_note"] or "").startswith("revised:"):
        return {"error": "claim already revised once; reject it or keep the verdict"}
    err = _validate(c["claim_type"], predicate)
    if err:
        return {"error": err}
    ctx.pg.execute("UPDATE claims SET predicate=%s::jsonb, text=coalesce(%s, text), status='pending', "
                   "review_note=%s WHERE id=%s",
                   (json.dumps(predicate), text, f"revised: {note} | original predicate: {json.dumps(c['predicate'])}",
                    claim_id))
    return verify_claims(ctx, [claim_id])


PREDICATE_DOC = (
    "prevalence: {field, any_of:[synonyms], over:'all'|'stated', min_share|max_share|min_count|max_count}; "
    "trend: {keywords, early:[y1,y2], late:[y3,y4], direction:'increase'|'decrease', min_ratio}"
)

PROPOSE_TOOL = Tool(
    "propose_claim",
    "Propose a checkable claim for the report. It will be verified by code before it can be used. "
    "Predicate formats — " + PREDICATE_DOC + ". Include all synonyms/spellings in any_of.",
    obj({"text": STR, "claim_type": {"type": "string", "enum": ["prevalence", "trend"]},
         "predicate": {"type": "object"}}, ["text", "claim_type", "predicate"]),
    propose_claim,
)

EVIDENCE_TOOLS = [
    Tool("list_claims", "List this run's claims, optionally by status.",
         obj({"status": {"type": "string", "enum": ["pending", "supported", "unsupported", "rejected"]}}),
         list_claims),
    Tool("verify_claims", "Evaluate claims by code (all pending ones if no ids given). Returns n/N, shares, "
         "matched values and caveats.", obj({"claim_ids": {"type": "array", "items": INT}}), verify_claims),
    Tool("revise_claim", "Once per claim: replace a predicate that does not faithfully test the claim text "
         "(e.g. missing synonyms, wrong field). Re-verifies immediately. Explain why in note.",
         obj({"claim_id": INT, "predicate": {"type": "object"}, "note": STR, "text": STR},
             ["claim_id", "predicate", "note"]), revise_claim),
    Tool("reject_claim", "Reject a claim whose predicate cannot faithfully test it, or that overreaches.",
         obj({"claim_id": INT, "reason": STR}, ["claim_id", "reason"]), reject_claim),
]
_ = (NUM, STRS)
