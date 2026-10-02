"""Try to settle thin evidence before the report leans on it.

A claim can pass its bound and still be worth very little: measured over eight papers, or over a set where
most records came from abstracts, so the count says as much about how the papers were read as about the
papers. `tools/epistemics` already names that: such a claim comes out `uncertain`, `not_searched_enough` or
`partially_supported` rather than `supported`, with the reasons attached.

Naming it is not the same as doing something about it. Until now the run wrote the report anyway, and a
researcher had to notice the state code and ask for a deeper read themselves — which only works if someone
is watching. This closes that: before synthesis writes anything, every claim the report could lean on whose
state rests on READING DEPTH is sent back to the papers. Its abstract-only papers are read in full, the
papers the count missed first (that is where under-counting hides), then it is measured again.

Three things make this safe to run unattended:

  - it only touches claims whose weakness is depth. A claim that is thin because the literature is thin
    (too few papers exist) cannot be fixed by reading, so it is left alone and said to be left alone.
  - it never edits a claim's text or predicate. The same predicate is re-measured against better records,
    so a number can only move because a paper turned out to say something its abstract did not.
  - a claim can get worse. `partially_supported` becoming `contradicted` is the system working, and is
    recorded exactly like an improvement.

What comes out is a resolution record: for every attempt, the state before, the state after, how many papers
were read, and which papers changed. That record is the honest answer to "how much of this rests on
abstracts?", and it is kept whether or not the attempt helped.
"""
from __future__ import annotations

from research_agent.config import settings

# States worth an attempt, and why each one might be a reading problem rather than a literature problem.
WORTH_READING = {
    "not_searched_enough": "the measurement cannot carry the claim: papers mention it without it reaching "
                           "the field, or the zero rests on abstracts",
    "partially_supported": "the bound holds, but on records that are mostly abstracts",
    "not_reported": "nothing reports it, and a full-text read is what would turn that into an absence",
}
# A state no amount of reading can mend: too few papers exist to decide, whatever depth they are read at.
NOT_A_READING_PROBLEM = {
    "uncertain": "measured over too few papers to decide; reading them more closely cannot add papers",
}
ABSTRACT_SHARE = 0.34      # below this share of abstract-only papers, depth is not the limiting factor


def _depth_limited(facts: dict) -> bool:
    """Is this claim's weakness about how its papers were read?"""
    extracted = facts.get("papers_extracted") or facts.get("denominator") or 0
    abstract = facts.get("abstract_only") or 0
    if facts.get("mentioned_but_not_recorded"):
        return True
    return bool(extracted) and abstract / extracted >= ABSTRACT_SHARE


def candidates(ctx) -> list[dict]:
    """Claims the report could lean on whose state is a reading problem, worst first."""
    rows = ctx.pg.execute(
        "SELECT id, text, claim_type, predicate, status, state, state_facts FROM claims "
        "WHERE run_id=%s AND status <> 'rejected' AND claim_type='prevalence' AND state IS NOT NULL "
        "ORDER BY id", (ctx.run_id,)).fetchall()
    out = []
    for c in rows:
        facts = c["state_facts"] or {}
        if c["state"] in NOT_A_READING_PROBLEM:
            out.append({**c, "skip": NOT_A_READING_PROBLEM[c["state"]]})
        elif c["state"] in WORTH_READING and _depth_limited(facts):
            out.append({**c, "skip": None, "why": WORTH_READING[c["state"]]})
        elif c["state"] in WORTH_READING:
            out.append({**c, "skip": "its papers were mostly read in full already, so depth is not what "
                                     "limits it"})
    order = list(WORTH_READING) + list(NOT_A_READING_PROBLEM)
    return sorted(out, key=lambda c: (c["skip"] is not None, order.index(c["state"])))


def resolve(ctx, max_claims: int | None = None, per_claim: int | None = None,
            max_papers: int | None = None) -> dict:
    """Read deeper for the claims whose state is a reading problem, re-measure them, and record what moved.

    Every read costs a model call, so the pass is bounded twice: at most `max_claims` claims and
    `per_claim` papers each, and at most `max_papers` papers over the whole pass. The second bound is what
    keeps a run's cost predictable when many claims come out thin at once; a claim left unattempted for want
    of budget is recorded as such rather than quietly dropped."""
    from research_agent.agents.followup import read_in_full

    max_claims = max(0, int(max_claims if max_claims is not None else settings.resolve_max_claims))
    per_claim = max(1, int(per_claim if per_claim is not None else settings.resolve_papers_per_claim))
    budget = max(0, int(max_papers if max_papers is not None else settings.resolve_max_papers))
    found = candidates(ctx)
    attempts, skipped = [], [{"claim": f"C{c['id']}", "state": c["state"], "left_alone_because": c["skip"]}
                             for c in found if c["skip"]]
    todo = [c for c in found if not c["skip"]][:max_claims]
    for c in todo:
        before_state, before = c["state"], _counted(c["state_facts"])
        if budget <= 0:
            skipped.append({"claim": f"C{c['id']}", "state": c["state"],
                            "left_alone_because": "the pass had no reading budget left; raise "
                                                  "RESOLVE_MAX_PAPERS to go further"})
            continue
        try:
            res = read_in_full(ctx, f"C{c['id']}", limit=min(per_claim, budget))
        except Exception as exc:          # a failed attempt is recorded, never fatal: the report still goes out
            attempts.append({"claim": f"C{c['id']}", "state_before": before_state, "state_after": before_state,
                             "failed": str(exc)[:200]})
            continue
        # A re-count id is the signal that papers were actually read again. `note` is not: read_in_full
        # returns one on success too, so testing for it would skip every attempt that worked.
        if res.get("error") or not res.get("recount_claim"):
            attempts.append({"claim": f"C{c['id']}", "state_before": before_state, "state_after": before_state,
                             "papers_read": 0, "note": res.get("error") or res.get("note")})
            continue
        # the re-count is its own claim (the original is never edited); the original's state is recomputed
        # against the now-fuller records so the report reads the improved one
        from research_agent.tools.claims import reject_claim, verify_claims

        budget -= res.get("read_in_full_now", 0) or 0
        verify_claims(ctx, [c["id"]])
        if res.get("recount_claim"):
            # The re-count counts exactly what the original now counts, so keeping both would put the same
            # measurement in the report twice. It stays in the database, rejected, as the audit trail.
            reject_claim(ctx, int(str(res["recount_claim"]).lstrip("C")),
                         f"superseded: C{c['id']} was re-measured against the fuller records instead")
        after = ctx.pg.execute("SELECT state, state_facts, status FROM claims WHERE id=%s", (c["id"],)).fetchone()
        attempts.append({
            "claim": f"C{c['id']}", "text": c["text"][:160], "attempted_because": c.get("why"),
            "state_before": before_state, "state_after": after["state"],
            "counted_before": before, "counted_after": _counted(after["state_facts"]),
            "papers_read": res.get("read_in_full_now", 0),
            "papers_that_changed": [p["paper_id"] for p in res.get("papers_that_changed") or []],
            "recount_claim": res.get("recount_claim"),
            "moved": _movement(before_state, after["state"]),
        })
    record = {"attempted": len(attempts), "skipped": skipped,
              "attempts": attempts, **_trend(attempts)}
    ctx.save_note("evidence_resolution", record)
    if attempts:
        ctx.emit("evidence_resolution", "finish", {"output": {k: record[k] for k in
                                                              ("attempted", "resolved", "unchanged", "worsened")}})
    return record


def _counted(facts: dict | None) -> str | None:
    f = facts or {}
    return f"{f.get('n_matching')} of {f['denominator']}" if f.get("denominator") is not None else None


# How much a state is worth, so a move can be called an improvement or not without anyone's judgement.
_WORTH = {"not_searched_enough": 0, "uncertain": 1, "not_reported": 2, "partially_supported": 3,
          "contradicted": 4, "supported": 5}


def _movement(before: str | None, after: str | None) -> str:
    b, a = _WORTH.get(before or "", 0), _WORTH.get(after or "", 0)
    if (before or "") == (after or ""):
        return "unchanged"
    # contradicted is not "worse" than partially_supported: it is a settled answer, just not the one hoped for
    if before == "partially_supported" and after == "contradicted":
        return "settled against the claim"
    return "better" if a > b else "worse"


def _trend(attempts: list[dict]) -> dict:
    """How much of the run's thin evidence a deeper read settled. The number that matters is not how many
    claims improved but how many stayed thin: those are what the report must hedge."""
    resolved = [a for a in attempts if a.get("state_after") in ("supported", "contradicted")]
    unchanged = [a for a in attempts if a.get("moved") == "unchanged"]
    worsened = [a for a in attempts if a.get("moved") == "worse"]
    read = sum(a.get("papers_read") or 0 for a in attempts)
    return {"resolved": len(resolved), "unchanged": len(unchanged), "worsened": len(worsened),
            "papers_read": read,
            "resolution_rate": round(len(resolved) / len(attempts), 2) if attempts else None,
            "still_thin": [a["claim"] for a in attempts
                           if a.get("state_after") not in ("supported", "contradicted")]}


def markdown(ctx) -> list[str]:
    """What a deeper read settled and what it could not, for the report."""
    record = (ctx.notes().get("evidence_resolution") or {})
    attempts = record.get("attempts") or []
    if not attempts and not record.get("skipped"):
        return []
    L = ["## What a deeper read settled (computed)", "",
         "Before this report was written, every claim whose state rested on reading depth had its "
         "abstract-only papers read in full and was measured again. The same predicate was re-run against "
         "fuller records, so a number moved only where a paper said more than its abstract did.", ""]
    if attempts:
        L += ["| Claim | State before | State after | Counted before | Counted after | Papers read |",
              "|---|---|---|---|---|---|"]
        for a in attempts:
            L.append(f"| {a['claim']} | {a.get('state_before', '')} | {a.get('state_after', '')} | "
                     f"{a.get('counted_before') or '–'} | {a.get('counted_after') or '–'} | "
                     f"{a.get('papers_read', 0)} |")
        L += ["", f"{record.get('resolved', 0)} of {record.get('attempted', 0)} were settled either way; "
                  f"{record.get('unchanged', 0)} did not move. {record.get('papers_read', 0)} papers were read "
                  "in full for this."]
        if record.get("still_thin"):
            L.append(f"Still resting on thin evidence after reading: {', '.join(record['still_thin'])}. "
                     "Treat anything the report says about those as provisional.")
    if record.get("skipped"):
        L += ["", "Left alone, because reading more closely could not change them: "
                  + "; ".join(f"{s['claim']} ({s['left_alone_because']})" for s in record["skipped"][:6]) + "."]
    return L + [""]
