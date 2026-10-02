"""What is known, what is merely not reported, and what was never looked for hard enough.

A claim's `status` says whether its measurement passed the bound the agent asserted. That is not the same
question a researcher asks, which is how much the number is worth. "0 of 32 papers report allele X" passes a
rarity bound and reads in a report as "the allele is absent from East Africa", when all it says is that no
paper in this shortlist recorded it.

So every verified claim also gets a STATE, computed here from numbers that already exist (the measurement, the
absence check, and how the papers were read). The state travels with the facts behind it, so a report, an
export and the map all say the same thing and a reader can see why:

    supported              measured, the bound holds, on an evidence base big enough to mean something
    partially_supported    the bound holds, but on a thin base, a small subgroup or mostly abstracts
    contradicted           measured on a base big enough to mean something, and the bound fails
    uncertain              too few papers to decide either way
    not_reported           nothing in the corpus reports it; that is an observation, not an absence
    not_searched_enough    papers mention it in their text without it reaching the extracted field, or the
                           zero rests on abstracts: the measurement cannot carry the claim at all

Nothing here calls an LLM. The states are a function of the run's own numbers, so they are reproducible and
they are the same whoever reads them.
"""
from __future__ import annotations

from research_agent.config import settings

STATES = ["supported", "partially_supported", "contradicted", "uncertain", "not_reported", "not_searched_enough"]

# How each state should be read in a sentence, for reports and exports.
PLAIN = {
    "supported": "supported by the papers analysed",
    "partially_supported": "supported, but on evidence thin enough to qualify it",
    "contradicted": "the measurement goes against it",
    "uncertain": "too few papers to decide",
    "not_reported": "not reported in the papers analysed, which is not the same as absent",
    "not_searched_enough": "the papers were not read closely enough for this to be measured",
}
ABSENCE_NOTE = ("No paper analysed reports it. Whether it is genuinely absent from the field is NOT "
                "established by this run: a corpus of {denominator} papers, {read_in_full} read in full and "
                "{abstract_only} from the abstract alone, can only show what these papers say.")


def _facts(ctx, res: dict, claim_type: str) -> dict:
    """The numbers a reader needs to judge the state, gathered once so every output quotes the same ones."""
    from research_agent.tools.extraction import extraction_coverage

    out: dict = {}
    try:
        cov = extraction_coverage(ctx)
        out["read_in_full"] = cov["by_source"].get("fulltext", 0)
        out["abstract_only"] = cov["by_source"].get("abstract", 0)
        out["papers_extracted"] = cov["extracted"]
    except Exception:                                  # a run whose records are gone: state it without them
        pass
    if claim_type == "trend":
        for k in ("total_matching_papers", "source", "within"):
            if res.get(k) is not None:
                out[k] = res[k]
        return out
    for k in ("n_matching", "denominator", "share", "over", "n_field_stated"):
        if res.get(k) is not None:
            out[k] = res[k]
    if res.get("subgroup"):
        out["subgroup_size"] = res["subgroup"].get("size")
    if res.get("absence_problem"):
        out["mentioned_but_not_recorded"] = res["absence_problem"]
    return out


def state_of(ctx, claim_type: str, result: dict | None) -> dict:
    """{state, reasons, facts} for one verified claim. `result` is what tools.claims.evaluate returned."""
    res = result or {}
    facts = _facts(ctx, res, claim_type)
    base = settings.min_evidence_base
    reasons: list[str] = []

    if claim_type == "trend":
        total = res.get("total_matching_papers") or 0
        if total < 10:
            return {"state": "uncertain", "facts": facts,
                    "reasons": [f"only {total} matching papers across the window, too few to call a trend"]}
        if res.get("caveat"):
            reasons.append(res["caveat"])
        state = "supported" if res.get("supported") else "contradicted"
        if state == "supported" and reasons:
            state = "partially_supported"
        return {"state": state, "facts": facts, "reasons": reasons}

    denom = res.get("denominator") or 0
    n = res.get("n_matching")
    abstract_only = facts.get("abstract_only", 0)
    extracted = facts.get("papers_extracted") or denom

    # 1. the measurement itself cannot be trusted
    if res.get("absence_problem"):
        return {"state": "not_searched_enough", "facts": facts,
                "reasons": [res["absence_problem"],
                            "papers mention it in their text without it reaching the extracted field"]}
    if n == 0 and extracted and abstract_only * 2 > extracted:
        return {"state": "not_searched_enough", "facts": facts,
                "reasons": [f"{abstract_only} of {extracted} papers were read from the abstract alone, so a "
                            "count of zero says more about how they were read than about the literature"]}
    # 2. too few papers to decide either way
    if denom < base:
        return {"state": "uncertain", "facts": facts,
                "reasons": [f"measured over {denom} papers, fewer than the {base} this run treats as enough "
                            "to decide"]}
    # 3. nothing reports it: an observation about the corpus, never an absence in the world
    if n == 0:
        return {"state": "not_reported", "facts": facts,
                "reasons": [ABSENCE_NOTE.format(denominator=denom, read_in_full=facts.get("read_in_full", 0),
                                                abstract_only=abstract_only)]}
    # 4. measured, on a base worth believing
    if not res.get("supported"):
        return {"state": "contradicted", "facts": facts,
                "reasons": [f"measured {n} of {denom}, which fails the bound the claim asserts"]}
    if res.get("caveat"):
        reasons.append(res["caveat"])
    sub = (res.get("subgroup") or {}).get("size")
    if sub is not None and sub < base:
        reasons.append(f"measured inside a subgroup of {sub} papers")
    if abstract_only * 2 > (extracted or 1):
        reasons.append(f"{abstract_only} of {extracted} papers were read from the abstract alone")
    return {"state": "partially_supported" if reasons else "supported", "facts": facts, "reasons": reasons}


def summary(ctx) -> dict:
    """How many claims of the run sit in each state, for the report and the web page."""
    rows = ctx.pg.execute("SELECT state, count(*) n FROM claims WHERE run_id=%s AND state IS NOT NULL "
                          "GROUP BY state", (ctx.run_id,)).fetchall()
    return {r["state"]: r["n"] for r in rows}


def markdown(ctx) -> list[str]:
    """A short table of what the run established and what it only failed to find."""
    rows = ctx.pg.execute(
        "SELECT id, text, state, state_facts FROM claims WHERE run_id=%s AND state IS NOT NULL "
        "AND status <> 'rejected' ORDER BY array_position(%s, state), id",
        (ctx.run_id, STATES)).fetchall()
    if not rows:
        return []
    L = ["## What this run establishes (computed)", "",
         "Every claim carries the state code measured for it, not the state its wording claims. "
         "`not_reported` and `not_searched_enough` are observations about this corpus, not about the field.",
         "", "| Claim | State | Measured |", "|---|---|---|"]
    for r in rows:
        f = r["state_facts"] or {}
        measured = (f"{f['n_matching']} of {f['denominator']}" if f.get("denominator") is not None
                    else (f"{f.get('total_matching_papers', 0)} papers in the window"))
        L.append(f"| C{r['id']} {r['text'][:90]} | {r['state']} | {measured} |")
    counts = summary(ctx)
    weak = sum(counts.get(s, 0) for s in ("not_reported", "not_searched_enough", "uncertain"))
    L += ["", f"{counts.get('supported', 0)} supported, {counts.get('partially_supported', 0)} supported with "
              f"qualifications, {counts.get('contradicted', 0)} contradicted by the measurement, and {weak} that "
              "this corpus cannot settle.", ""]
    return L
