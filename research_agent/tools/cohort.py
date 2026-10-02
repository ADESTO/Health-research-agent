"""Which papers a run's numbers are counted over.

A run's question usually carries a scope: African malaria forecasting studies, East African isolates,
paediatric trials. Until now that scope lived only in prose, in the research question and in a
researcher's charter, where nothing could enforce it. A charter's `scope` scores how close a proposed
QUESTION is to the goal, and `out_of_bounds` matches phrases in a question's text; neither touches which
papers a test counts. So a study from the wrong continent sat in every denominator until an agent happened
to read it, and the scope had to be re-written into each test predicate by hand, from memory, every time.

A cohort is that scope, stored once on the run and applied in the one place paper records are built
(`extraction._rows`), so every count downstream — claims, the researcher's statistical tests, the map,
value counts, the report, the exports — counts the same papers without anyone remembering to ask.

    include   conditions a paper must meet, in the same form as a claim's `where`
    exclude   conditions that put a paper out of scope
    unstated  what to do with a paper that is silent on an `include` field: keep (default) or exclude

`unstated` matters more than it looks. The fields a cohort is written on (geography above all) are
extracted fields, so a paper read only from its abstract is often silent on them. Excluding silence would
drop papers for how they were read rather than for what they are, which is the same read-depth trap that
makes a count of zero look like an absence. So silence is KEPT by default and counted separately, and the
number kept that way is reported wherever the cohort is: a cohort resting on 30 unstated papers is telling
you to run a field pass, not to trust the filter.

A cohort can only be applied after extraction, because it is written on extracted fields: reading is never
filtered, and `extract_papers` still reads the whole shortlist. Excluding a paper here removes it from the
counting, not from the run.
"""
from __future__ import annotations

import json

UNSTATED = ("keep", "exclude")


def _conds(x) -> list[dict]:
    if x is None:
        return []
    return [x] if isinstance(x, dict) else [c for c in x if isinstance(c, dict)]


def validate(ctx, include, exclude) -> str | None:
    """None when every condition can be counted; otherwise what to fix."""
    from research_agent.tools.extraction import SPECIAL_FIELDS, known_fields

    lists, enums = known_fields(ctx)
    known = set(lists) | set(enums) | set(SPECIAL_FIELDS) | {"source"}
    all_conds = _conds(include) + _conds(exclude)
    if not all_conds:
        return "a cohort needs at least one include or exclude condition"
    for c in all_conds:
        f = c.get("field")
        if f not in known:
            return f"field must be one of {sorted(known)}"
        if f == "year":
            if c.get("min") is None and c.get("max") is None:
                return "a year condition needs min and/or max"
        elif not (c.get("any_of") or c.get("none_of")):
            return f"condition on {f} needs any_of or none_of"
    return None


def of(ctx) -> dict | None:
    """The cohort stored on this run, or None when it counts every shortlisted paper."""
    cached = getattr(ctx, "_cohort", "missing")
    if cached != "missing":
        return cached
    try:
        row = ctx.pg.execute("SELECT cohort FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()
        value = (row or {}).get("cohort") or None
    except Exception:                       # a database from before the column existed
        value = None
    try:
        ctx._cohort = value
    except Exception:
        pass
    return value


def set_cohort(ctx, include=None, exclude=None, unstated: str = "keep", note: str = "") -> dict:
    """Store the scope this run's numbers are counted over. Re-settable: the next count uses the new one."""
    if unstated not in UNSTATED:
        return {"error": f"unstated must be one of {', '.join(UNSTATED)}"}
    err = validate(ctx, include, exclude)
    if err:
        return {"error": err}
    cohort = {"include": _conds(include), "exclude": _conds(exclude), "unstated": unstated,
              "note": (note or "").strip()[:500]}
    ctx.pg.execute("UPDATE runs SET cohort=%s::jsonb WHERE run_id=%s", (json.dumps(cohort), ctx.run_id))
    try:
        ctx._cohort = cohort
    except Exception:
        pass
    return {"cohort": cohort, **summary(ctx)}


def clear(ctx) -> dict:
    ctx.pg.execute("UPDATE runs SET cohort=NULL WHERE run_id=%s", (ctx.run_id,))
    try:
        ctx._cohort = None
    except Exception:
        pass
    return {"cohort": None, "note": "every shortlisted paper is counted again"}


def _states(row: dict, field: str) -> bool:
    """Does this paper say anything at all about the field a cohort condition is written on?"""
    from research_agent.tools.extraction import _values

    if field in ("corpus", "read", "source"):
        return True
    if field == "year":
        return bool(row.get("year"))
    return bool(_values(row["data"], field))


def apply(ctx, rows: list[dict]) -> list[dict]:
    """The rows a cohort counts. Also records, on ctx, why each excluded paper was excluded."""
    cohort = of(ctx)
    if not cohort:
        return rows
    from research_agent.tools.claims import row_matches
    from research_agent.tools.extraction import known_fields

    _, enums = known_fields(ctx)
    include, exclude = _conds(cohort.get("include")), _conds(cohort.get("exclude"))
    keep_unstated = cohort.get("unstated", "keep") == "keep"
    kept, out_of_scope, unstated = [], [], []
    for r in rows:
        if any(row_matches(r, c, enums) for c in exclude):
            out_of_scope.append(r["paper_id"])
            continue
        if not include:
            kept.append(r)
            continue
        if all(row_matches(r, c, enums) for c in include):
            kept.append(r)
            continue
        # It does not meet the scope. Out of scope, or merely silent? Only silence on EVERY condition it
        # fails counts as silence: a paper that states a value outside the scope is out of scope, whatever
        # else it leaves blank.
        failed = [c for c in include if not row_matches(r, c, enums)]
        silent = [c["field"] for c in failed if not _states(r, c["field"])]
        if len(silent) == len(failed) and keep_unstated:
            unstated.append(r["paper_id"])
            kept.append(r)
        elif len(silent) == len(failed):
            unstated.append(r["paper_id"])
        else:
            out_of_scope.append(r["paper_id"])
    try:
        ctx._cohort_excluded = {"out_of_scope": out_of_scope, "unstated": unstated,
                               "unstated_kept": keep_unstated}
    except Exception:
        pass
    return kept


def summary(ctx) -> dict:
    """How many papers the cohort counts, how many it dropped, and how many it kept despite silence."""
    from research_agent.tools.extraction import _rows

    rows = _rows(ctx)                               # already filtered; the counts come from the pass above
    cohort = of(ctx)
    if not cohort:
        return {"counted": len(rows), "cohort": None}
    ex = getattr(ctx, "_cohort_excluded", {}) or {}
    unstated_ids = ex.get("unstated") or []
    out = {"counted": len(rows), "excluded_out_of_scope": len(ex.get("out_of_scope") or []),
           "unstated_on_a_cohort_field": len(unstated_ids),
           "unstated_are": "counted" if cohort.get("unstated", "keep") == "keep" else "excluded",
           "conditions": describe(cohort)}
    if unstated_ids and cohort.get("unstated", "keep") == "keep":
        out["warning"] = (f"{len(unstated_ids)} of the {len(rows)} papers counted are silent on a field the "
                          "cohort is written on, so they are in scope only by default. Run a field pass on "
                          "that field before relying on the denominator.")
    return out


def describe(cohort: dict | None) -> str:
    """The cohort in one line, for a report, an export or a log."""
    if not cohort:
        return "every shortlisted paper"

    def one(c: dict) -> str:
        if c.get("field") == "year":
            return f"year {c.get('min', '')}-{c.get('max', '')}"
        if c.get("any_of"):
            return f"{c['field']} in {', '.join(map(str, c['any_of']))}"
        return f"{c['field']} not in {', '.join(map(str, c.get('none_of') or []))}"
    parts = []
    if _conds(cohort.get("include")):
        parts.append(" and ".join(one(c) for c in _conds(cohort["include"])))
    if _conds(cohort.get("exclude")):
        parts.append("excluding " + " or ".join(one(c) for c in _conds(cohort["exclude"])))
    line = "; ".join(parts) or "every shortlisted paper"
    if cohort.get("unstated") == "exclude":
        line += " (papers silent on these fields are excluded)"
    return line


def markdown(ctx) -> list[str]:
    """A line for the report, so a reader can see what the denominators are counted over."""
    cohort = of(ctx)
    if not cohort:
        return []
    s = summary(ctx)
    L = ["## What the numbers are counted over (computed)", "",
         f"**Cohort:** {s['conditions']}", ""]
    if cohort.get("note"):
        L += [cohort["note"], ""]
    L += [f"Every count in this report is over the {s['counted']} papers that meet it. "
          f"{s['excluded_out_of_scope']} analysed papers were excluded as out of scope."]
    if s.get("unstated_on_a_cohort_field"):
        L.append(f"{s['unstated_on_a_cohort_field']} are silent on a field the cohort is written on and are "
                 f"{s['unstated_are']}, which is a decision about how they were read, not about what they are.")
    return L + [""]
