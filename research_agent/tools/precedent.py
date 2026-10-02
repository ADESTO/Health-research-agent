"""Has this been done already?

An opportunity is only worth anything if nobody has done it. Until now the run could not say: a gap was
computed from the SHORTLIST, sixty-odd papers out of a corpus of hundreds of thousands, so "no paper does
this" meant "no paper among the ones this run happened to read". A researcher reading that has to go and
search the literature themselves, which is the work the system was supposed to do.

So each opportunity gets a precedent check: the whole corpus is searched for the studies nearest to it, and
each one is graded by WHAT WAS MATCHED, not by how a model felt about it:

    direct      every term the opportunity turns on appears in the same paper. Someone has done this.
    partial     the main thing appears, the qualifier does not: done, but not in the way that matters here
                (the method without the setting, one half of a combination plus the other's field)
    adjacent    close by semantic distance only, with none of the required terms: the neighbourhood is
                occupied, which is worth reading, and is not precedent
    none        nothing in the corpus reaches even that

Nothing here calls a model. The grade is a function of the terms an opportunity is written on and the text
of the candidate papers, so it is reproducible and a reader can check it: every verdict carries the terms it
was judged on and the papers at each level.

The honesty rule that matters: an opportunity whose terms cannot be read off a structured field — a prose
label from a design agent, a research direction written as a sentence — is capped at `adjacent`, however
good the search hits look. A sentence cannot establish that a paper did the same thing; only a field and a
value can. The verdict says so rather than guessing.
"""
from __future__ import annotations

import json
import re

from research_agent.embeddings import get_embedder

LEVELS = ["direct", "partial", "adjacent", "none"]
CANDIDATES = 40             # papers pulled from the corpus per opportunity
PER_LEVEL = 8               # papers reported at each level
MIN_TERM_LEN = 4            # shorter words are matched as whole words only

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "could", "do", "does", "for", "from", "has",
    "have", "how", "in", "into", "is", "it", "its", "more", "most", "no", "not", "of", "on", "only", "or",
    "other", "same", "show", "shows", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "together", "up", "use", "used", "uses", "using", "was", "were", "what", "when",
    "whether", "which", "while", "who", "why", "with", "within", "without", "would", "study", "studies",
    "paper", "papers", "research", "paper's", "work", "paired", "versus", "vs", "none", "stated",
    "not_stated", "any", "all", "both", "each", "new", "rare", "rarely", "never", "always", "often",
}
_WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*")
# Values that say which way a field points without saying what the field is about. A gap on
# `code_or_data_available = yes` must not be graded by looking for the word "yes": every paper that
# happens to contain it would come back as precedent for sharing code. The field's own words carry the
# meaning, so these contribute nothing and the field name is used instead.
_POLARITY = {"yes", "no", "true", "false", "present", "absent", "available", "shared", "unavailable",
             "some", "any", "other", "unknown", "reported", "unreported", "high", "low"}


def _terms(text: str) -> list[str]:
    """The distinctive words of a label or value, in order, without the filler."""
    out, seen = [], set()
    for w in _WORD.findall(str(text or "").lower().replace("_", " ")):
        if w in _STOP or len(w) < 2 or w.isdigit() or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _mentions(text: str, term: str) -> bool:
    """Short terms (mic, auc, ml) match as whole words so 'mic' does not hit 'microscopy'."""
    if len(term) < MIN_TERM_LEN:
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"s?(?![a-z0-9])", text))
    return term in text


def requirements(ctx, row: dict) -> dict:
    """What a paper would have to contain to count as precedent for this opportunity.

    `groups` is a list of term groups: a paper is a DIRECT precedent when it matches every group, and a
    PARTIAL one when it matches some but not all. Groups come from structured fields wherever they exist
    (a gap's field and value, both sides of a combination) and never from prose alone."""
    ev = row.get("evidence") or {}
    kind, groups, structured, reason = row["kind"], [], True, ""
    if kind == "combination":
        # the two sides of an untried pair, each as its own group: a direct precedent has to have both
        sides = [(row.get("field"), row.get("value"))]
        second = [str(x) for x in (ev.get("second_value") or []) if x]
        if second:
            sides.append((ev.get("second_field"), second[0]))
        else:   # an older row kept the second side only inside the label ("<a> with <b>")
            parts = re.split(r"\s+with\s+", str(row.get("label") or ""), maxsplit=1)
            if len(parts) == 2:
                sides = [(row.get("field"), parts[0]), (ev.get("second_field"), parts[1])]
        groups = [_field_and_value(f, v) for f, v in sides]
    elif kind == "gap":
        groups = [_field_and_value(row.get("field"), row.get("value"))]
    else:                                   # design, finding, direction: prose only
        structured = False
        reason = PROSE_CAP
        groups = [_terms(row.get("label"))[:6]] if _terms(row.get("label")) else []
    groups = [g for g in groups if g]
    # A group of only short or polarity-like words cannot identify the same study: grading on it would
    # count any paper containing the word. Say so and cap the verdict rather than reporting a false one.
    if structured and not any(any(len(t) >= MIN_TERM_LEN for t in g) for g in groups):
        structured, reason = False, THIN_TERMS
    return {"groups": groups, "topic": _topic_terms(ctx), "structured": structured, "cap_reason": reason}


def _field_and_value(field, value) -> list[str]:
    """The words a paper would have to contain for this field and value. A value that only says which way
    the field points (yes, no, available) contributes nothing, so the field's own words carry it."""
    vals = [t for t in _terms(value) if t not in _POLARITY]
    return vals or _terms(field)


def _topic_terms(ctx) -> list[str]:
    """The run's own subject, so a candidate has to be about the same thing: from the protocol's topic query
    when there is one, otherwise from the research question."""
    from research_agent.tools.extraction import protocol_of

    protocol = protocol_of(ctx) or {}
    return _terms(protocol.get("topic_query") or ctx.question)[:6]


def _query(row: dict, req: dict) -> str:
    bits = [str(row.get("question") or ""), str(row.get("label") or "")]
    return " ".join(b for b in bits if b.strip())[:500] or " ".join(req["topic"])


def _candidates(ctx, query: str, limit: int = CANDIDATES) -> list[dict]:
    """The corpus's nearest papers to a query: vector neighbours fused with full-text hits.

    Deliberately not `hybrid_search`: that one logs every paper it sees into the run's screening log, and a
    precedent check is not screening. It must not change what the run reports it looked at."""
    from research_agent.tools.query import tsquery_sql

    tq_sql, tq_params = tsquery_sql(query)
    params = {**tq_params, "qvec": get_embedder().embed_query(query), "lim": int(limit)}
    sql = f"""
    WITH q AS (SELECT ({tq_sql}) AS tq),
    sem AS (SELECT paper_id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) r
            FROM (SELECT paper_id, embedding FROM papers WHERE embedding IS NOT NULL
                  ORDER BY embedding <=> %(qvec)s LIMIT 120) s),
    kw AS (SELECT paper_id, row_number() OVER (ORDER BY rank DESC) r
           FROM (SELECT paper_id, ts_rank_cd(tsv, q.tq) rank FROM papers, q WHERE tsv @@ q.tq
                 ORDER BY rank DESC LIMIT 120) k)
    SELECT p.paper_id, p.title, p.year, p.source, p.abstract,
           coalesce(1.0/(60+sem.r), 0) + coalesce(1.0/(60+kw.r), 0) AS score
    FROM sem FULL OUTER JOIN kw USING (paper_id) JOIN papers p USING (paper_id)
    ORDER BY score DESC LIMIT %(lim)s
    """
    return ctx.pg.execute(sql, params).fetchall()


def _grade(cand: dict, req: dict, extracted: dict | None) -> tuple[str, list[str]]:
    """(level, which groups matched) for one candidate paper.

    A paper the run itself analysed is judged on its EXTRACTED fields as well as its text, because a checked
    extraction with a verified quote is better evidence than a word appearing in an abstract."""
    haystack = f"{cand.get('title') or ''} {cand.get('abstract') or ''}".lower()
    if extracted:
        values = " ".join(str(v).lower() for v in extracted.values()
                          if isinstance(v, (str, int, float)))
        lists = " ".join(str(x).lower() for v in extracted.values() if isinstance(v, list) for x in v
                         if isinstance(x, (str, int, float)))
        haystack = f"{haystack} {values} {lists}"
    on_topic = not req["topic"] or any(_mentions(haystack, t) for t in req["topic"])
    matched = []
    for i, group in enumerate(req["groups"]):
        # a group matches when most of its distinctive words are there: a value like
        # "temporal_holdout_or_rolling_origin" should not need every token to count
        hits = sum(1 for t in group if _mentions(haystack, t))
        if hits and hits >= max(1, (len(group) + 1) // 2):
            matched.append(str(i))
    if not req["groups"] or not on_topic:
        return ("adjacent" if on_topic else "none"), matched
    if len(matched) == len(req["groups"]):
        return ("direct" if req["structured"] else "adjacent"), matched
    if matched:
        return "partial", matched
    return "adjacent", matched


def check(ctx, item_id: str | None = None, limit: int = CANDIDATES) -> dict:
    """Grade every opportunity of this run (or one of them) against the whole corpus, and store the result
    on the opportunity row so a report, an export and the web page all read the same verdict."""
    from research_agent.tools.extraction import _rows
    from research_agent.tools.opportunities import listing

    rows = [r for r in listing(ctx) if item_id is None or r["item_id"] == str(item_id).upper()]
    if not rows:
        return {"error": f"no opportunity {item_id} in this run" if item_id else
                "this run has no opportunities recorded yet; write the report or run `export opportunities`"}
    analysed = {r["paper_id"]: r for r in _rows(ctx)}
    out = []
    for row in rows:
        req = requirements(ctx, row)
        query = _query(row, req)
        found: dict[str, list[dict]] = {lv: [] for lv in LEVELS}
        for cand in _candidates(ctx, query, limit):
            level, matched = _grade(cand, req, (analysed.get(cand["paper_id"]) or {}).get("data"))
            if level == "none":
                continue
            found[level].append({"paper_id": cand["paper_id"], "title": (cand["title"] or "")[:200],
                                 "year": cand["year"], "analysed_in_this_run": cand["paper_id"] in analysed,
                                 "matched_groups": len(matched)})
        verdict = next((lv for lv in LEVELS if found[lv]), "none")
        result = {
            "verdict": verdict,
            "means": MEANS[verdict] + ((" " + req["cap_reason"]) if req["cap_reason"] else ""),
            "judged_on": {"term_groups": req["groups"], "topic_terms": req["topic"],
                          "from_structured_field": req["structured"],
                          **({"capped_at_adjacent_because": req["cap_reason"]} if req["cap_reason"] else {})},
            "searched": {"query": query[:200], "candidates": limit, "corpus": "whole corpus, not the shortlist"},
            **{lv: found[lv][:PER_LEVEL] for lv in LEVELS if lv != "none"},
            "counts": {lv: len(found[lv]) for lv in LEVELS if lv != "none"},
        }
        ctx.pg.execute(
            "UPDATE research_opportunities SET evidence = evidence || %s::jsonb "
            "WHERE run_id=%s AND item_id=%s", (json.dumps({"precedent": result}, default=str),
                                               ctx.run_id, row["item_id"]))
        out.append({"item_id": row["item_id"], "kind": row["kind"], "label": row["label"][:120],
                    "verdict": verdict, "counts": result["counts"]})
    return {"checked": len(out), "by_verdict": {lv: sum(1 for o in out if o["verdict"] == lv) for lv in LEVELS
                                               if any(o["verdict"] == lv for o in out)}, "opportunities": out}


MEANS = {
    "direct": "a paper in the corpus already does this: read it before going further",
    "partial": "part of this has been done, but not the part the opportunity turns on",
    "adjacent": "nothing in the corpus does this, but related work sits next to it",
    "none": "nothing in the corpus comes close, which may mean the question is new or badly worded",
}
PROSE_CAP = ("This opportunity is written as prose rather than on a field and a value, so it can never be "
             "graded above adjacent: a sentence cannot establish that a paper did the same thing.")
THIN_TERMS = ("The field and value this opportunity rests on give no word distinctive enough to identify "
              "the same study, so it is capped at adjacent: matching on it would count any paper that "
              "happens to contain the word.")


def markdown(ctx) -> list[str]:
    """A column for the report: for each opportunity, whether anyone has done it, with the papers."""
    from research_agent.tools.opportunities import listing

    rows = [r for r in listing(ctx) if (r["evidence"] or {}).get("precedent")]
    if not rows:
        return []
    L = ["## Has it been done already? (computed)", "",
         "Each opportunity was searched for across the whole corpus, not just the papers this run read. "
         "A verdict rests on matching the terms the opportunity is written on, listed with it, so it can be "
         "checked.",
         "", "| id | What | Already done? | Nearest papers |", "|---|---|---|---|"]
    for r in rows:
        p = r["evidence"]["precedent"]
        level = p.get("verdict", "none")
        near = p.get(level) or p.get("partial") or p.get("adjacent") or []
        ids = ", ".join(f"[{x['paper_id']}]" for x in near[:3]) or "–"
        L.append(f"| {r['item_id']} | {str(r['label'])[:70]} | {level} | {ids} |")
    return L + ["", "`direct` means a paper already does it; `partial` that a part of it is done; "
                    "`adjacent` that the neighbourhood is occupied but the thing itself is not.", ""]
