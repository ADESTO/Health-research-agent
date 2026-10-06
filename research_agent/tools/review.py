"""Systematic-review record: screening log, PRISMA 2020 flow, protocol document.

Discovery logs every paper it sees: 'identified' when a search, similarity or citation step surfaces it, then
'included', 'excluded' (with the reason given), 'duplicate' (the same work in another corpus) or 'over_limit'.
Papers identified but never added are counted as excluded at screening ('not selected'). From that log, code
(not the model) builds the PRISMA 2020 flow counts and diagram.
"""
from __future__ import annotations

from research_agent.db import connect


def _exec(ctx, sql: str, rows: list[tuple]) -> None:
    if not rows:
        return
    try:
        with ctx.pg.cursor() as cur:
            cur.executemany(sql, rows)
    except Exception:   # the log is a record, never a reason for a search to fail
        pass


def log_identified(ctx, paper_ids, found_by: str) -> None:
    _exec(ctx, "INSERT INTO screening (run_id, paper_id, stage, found_by) VALUES (%s,%s,'identified',%s) "
               "ON CONFLICT (run_id, paper_id) DO NOTHING",
          [(ctx.run_id, pid, found_by[:200]) for pid in dict.fromkeys(paper_ids)])


def log_decision(ctx, paper_ids, stage: str, reason: str = "", criterion: str | None = None) -> None:
    """Record a decision. `criterion` is the short label PRISMA counts exclusions under (a protocol criterion,
    "could not be screened"...); `reason` is the sentence for that one paper."""
    _exec(ctx, "INSERT INTO screening (run_id, paper_id, stage, reason, criterion, found_by) "
               "VALUES (%s,%s,%s,%s,%s,'decision') ON CONFLICT (run_id, paper_id) DO UPDATE SET "
               "stage=EXCLUDED.stage, reason=EXCLUDED.reason, criterion=EXCLUDED.criterion, ts=now()",
          [(ctx.run_id, pid, stage, (reason or "")[:300], (criterion or None) and criterion[:120])
           for pid in dict.fromkeys(paper_ids)])


def screening_rows(ctx) -> list[dict]:
    return ctx.pg.execute(
        """SELECT s.paper_id, p.source AS corpus, p.year, p.title, s.stage, s.criterion, s.reason, s.found_by, s.ts
           FROM screening s JOIN papers p USING (paper_id) WHERE s.run_id=%s ORDER BY s.stage, p.year DESC""",
        (ctx.run_id,)).fetchall()


def prisma_counts(ctx) -> dict:
    rows = screening_rows(ctx)
    included = set(ctx.shortlist_ids())
    by_corpus: dict[str, int] = {}
    for r in rows:
        by_corpus[r["corpus"]] = by_corpus.get(r["corpus"], 0) + 1
    # papers added without passing through a logged search (older runs) still count as identified
    for pid in included - {r["paper_id"] for r in rows}:
        by_corpus["unlogged"] = by_corpus.get("unlogged", 0) + 1
    identified = sum(by_corpus.values())
    duplicates = sum(1 for r in rows if r["stage"] == "duplicate")
    over_limit = sum(1 for r in rows if r["stage"] == "over_limit")
    excluded = [r for r in rows if r["stage"] == "excluded"]
    reasons: dict[str, int] = {}
    for r in excluded:
        key = (r["reason"] or "no reason given").strip()[:80]
        reasons[key] = reasons.get(key, 0) + 1
    screened = identified - duplicates
    ft = {r["paper_id"]: r["status"] for r in ctx.pg.execute(
        "SELECT paper_id, status FROM paper_fulltext WHERE paper_id = ANY(%s)", (list(included) or [""],)).fetchall()}
    read = {r["paper_id"]: r["source"] for r in ctx.pg.execute(
        "SELECT paper_id, source FROM extractions WHERE schema_version=%s AND paper_id = ANY(%s)",
        (ctx.extraction_version, list(included) or [""])).fetchall()}
    read_full = sum(1 for p in included if read.get(p) == "fulltext")
    excluded_at_screening = max(0, screened - len(included))
    not_selected = max(0, excluded_at_screening - len(excluded) - over_limit)
    out = {"identified": identified, "identified_by_corpus": by_corpus, "duplicates_removed": duplicates,
           "screened": screened, "excluded_at_screening": excluded_at_screening,
           "excluded_with_reason": len(excluded), "exclusion_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
           "exclusion_breakdown": _breakdown(excluded, rows, not_selected, excluded_at_screening),
           "not_selected": not_selected,
           "over_shortlist_limit": over_limit, "included": len(included),
           "full_text_sought": sum(1 for p in included if p in ft),
           "full_text_not_retrieved": sum(1 for p in included if ft.get(p) not in (None, "ok")),
           "assessed_in_full": read_full, "abstract_only": len(included) - read_full}
    out.update(_after_reading(ctx, len(included)))
    return out


def _breakdown(excluded: list[dict], rows: list[dict], not_selected: int, total: int) -> dict:
    """Every paper excluded at screening, under exactly one reason, so the reasons add up to the total.

    Exclusions are grouped by the criterion they turn on when one was recorded (systematic screening names
    one), otherwise by their stated reason. Papers the cap left out and papers a search surfaced but no one
    chose are reasons too, and are listed as such rather than silently making up the difference."""
    out: dict[str, int] = {}
    for r in excluded:
        key = (r.get("criterion") or r["reason"] or "no reason given").strip()[:80]
        out[key] = out.get(key, 0) + 1
    out = dict(sorted(out.items(), key=lambda kv: -kv[1]))
    for r in rows:
        if r["stage"] != "over_limit":
            continue
        why = r["reason"] or ""
        if why.startswith("identified") and "screened" in why:
            key = "not in the random sample screened (over the screening cap)"
        elif why.startswith("eligible") and "analysed" in why:
            key = "eligible, not in the random sample analysed (over the analysis cap)"
        else:
            key = "shortlist full (over the shortlist cap)"
        out[key] = out.get(key, 0) + 1
    if not_selected:
        out["surfaced by a search, not selected"] = not_selected
    gap = total - sum(out.values())
    if gap > 0:                       # never let the reasons fall short of the total without saying so
        out["no reason recorded"] = gap
    return out


def _after_reading(ctx, included: int) -> dict:
    """Included papers the run's cohort leaves out of the counts once they have been read (a setting outside
    the protocol's countries, for example). Reported as a separate PRISMA step: these papers were read, so
    they are not screening exclusions."""
    try:
        from research_agent.tools import cohort

        if not cohort.of(ctx):
            return {}
        s = cohort.summary(ctx)
    except Exception:
        return {}
    return {"excluded_after_reading": s.get("excluded_out_of_scope", 0), "cohort": s.get("conditions"),
            "counted_in_synthesis": s.get("counted", included),
            "kept_though_silent_on_cohort_field": s.get("unstated_on_a_cohort_field", 0)}


def prisma_markdown(ctx) -> list[str]:
    c = prisma_counts(ctx)
    if not c["identified"]:
        return []
    corp = ", ".join(f"{src}: {n}" for src, n in c["identified_by_corpus"].items())
    reasons = "; ".join(f"{k} ({v})" for k, v in c["exclusion_breakdown"].items())
    lines = ["## Review record (PRISMA 2020)", "",
             "Counts come from the screening log kept during discovery. Papers a search surfaced but discovery "
             "did not add are counted as excluded at screening.", "",
             "| Stage | Papers |", "|---|---|",
             f"| Records identified ({corp}) | {c['identified']} |",
             f"| Duplicates removed (same work in both corpora) | {c['duplicates_removed']} |",
             f"| Records screened (title and abstract) | {c['screened']} |",
             f"| Excluded at screening | {c['excluded_at_screening']} |",
             f"| Included in the review | {c['included']} |",
             f"| Full text sought | {c['full_text_sought']} |",
             f"| Full text not retrieved | {c['full_text_not_retrieved']} |",
             f"| Read in full | {c['assessed_in_full']} |",
             f"| Read from the abstract only | {c['abstract_only']} |"]
    if "excluded_after_reading" in c:
        lines += [f"| Excluded after reading (outside the cohort: {c['cohort']}) | {c['excluded_after_reading']} |",
                  f"| Counted in the synthesis | {c['counted_in_synthesis']} |"]
    lines.append("")
    if reasons:
        lines += [f"Reasons for exclusion at screening (all {c['excluded_at_screening']}): {reasons}.", ""]
    if c.get("kept_though_silent_on_cohort_field"):
        lines += [f"{c['kept_though_silent_on_cohort_field']} counted papers do not state the field the cohort is "
                  "written on (often the study location, in papers read from the abstract only) and are kept by "
                  "default.", ""]
    lines += ["```mermaid", "flowchart TD",
              f'  A["Records identified: {c["identified"]}"] --> B["Duplicates removed: {c["duplicates_removed"]}"]',
              f'  B --> C["Records screened: {c["screened"]}"]',
              f'  C --> D["Excluded at screening: {c["excluded_at_screening"]}"]',
              f'  C --> E["Included: {c["included"]}"]',
              f'  E --> F["Read in full: {c["assessed_in_full"]}"]',
              f'  E --> G["Abstract only: {c["abstract_only"]}"]']
    if "excluded_after_reading" in c:
        lines += [f'  E --> H["Excluded after reading: {c["excluded_after_reading"]}"]',
                  f'  E --> I["Counted in synthesis: {c["counted_in_synthesis"]}"]']
    lines += ["```", ""]
    return lines


def protocol_markdown(ctx) -> str:
    """The review protocol as a document: question, scope, fields, search strategy."""
    from research_agent.tools.extraction import protocol_of

    notes = ctx.notes()
    protocol = protocol_of(ctx) or {}
    disc = notes.get("discovery") or {}
    row = ctx.pg.execute("SELECT created_at FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()
    corpus = {r["source"]: (r["n"], r["lo"], r["hi"]) for r in ctx.pg.execute(
        "SELECT source, count(*) n, min(year) lo, max(year) hi FROM papers WHERE source <> 'upload' "
        "GROUP BY source").fetchall()}
    L = ["# Review protocol", "", f"**Question:** {ctx.question}", "",
         f"**Registered:** {row['created_at']:%Y-%m-%d %H:%M} (run {ctx.run_id})" if row else "", "",
         "## Sources", ""]
    L += [f"- {src}: {n:,} papers, {lo}–{hi}" for src, (n, lo, hi) in corpus.items()] + [""]
    if protocol.get("setting"):
        L += ["## Setting", "", protocol["setting"], ""]
    if protocol.get("countries"):
        L += ["## Countries", "", "Studies count only when their data come from: " + ", ".join(protocol["countries"])
              + ". This is checked by code against each paper's quoted study location after it is read.", ""]
    if protocol.get("inclusion") or protocol.get("exclusion"):
        L += ["## Eligibility", ""] + [f"- Include: {x}" for x in protocol.get("inclusion", [])] \
             + [f"- Exclude: {x}" for x in protocol.get("exclusion", [])] + [""]
    L += ["## Data items", "",
          "General fields for every paper: task, health domain, data types, datasets, geography, methods, "
          "evaluation metrics, validation level, code or data availability, limitations, reported results and "
          "reported associations. Every value needs a quote from the paper, checked by code.", ""]
    for f in protocol.get("fields", []):
        vals = ", ".join(f["values"]) if f.get("values") else "free text"
        L.append(f"- **{f['name'].removeprefix('q_')}** ({vals}): {f.get('definition', '')}")
    if protocol.get("fields"):
        L.append("")
    searches = disc.get("search_log") or []
    if searches or protocol.get("topic_query"):
        L += ["## Search strategy", ""]
        if protocol.get("topic_query"):
            L.append(f"- Topic query: `{protocol['topic_query']}`")
        for s in searches[:30]:
            L.append(f"- {s.get('query') if isinstance(s, dict) else s}"
                     + (f" (keywords: `{s.get('keywords')}`)" if isinstance(s, dict) and s.get("keywords") else ""))
        L.append("")
    L += ["## Screening and synthesis", "",
          "Titles and abstracts were screened by the discovery agent against the question; up to the run's limit "
          "of papers were read in full. Counts and trends are computed by code from quote-verified records; "
          "claims are checked against those counts before they enter the report.", ""]
    return "\n".join(x for x in L if x is not None)
