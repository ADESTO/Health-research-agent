"""Render the Research Opportunity Map as Markdown. Pure formatting: every number comes from the map."""
from __future__ import annotations

import re


def _pct(x) -> str:
    return f"{round(100 * (x or 0))}%"


def _cite(pid: str) -> str:
    return f"[{pid}]" if pid.startswith("PMC") else f"[arXiv:{pid}]"


def _ids(pids, k=4) -> str:
    return ", ".join(_cite(p) for p in pids[:k]) + (" …" if len(pids) > k else "")


def _ci(it: dict) -> str:
    ci = it.get("ci95")
    return f", 95% CI {_pct(ci[0])}–{_pct(ci[1])}" if ci else ""


def _p(p) -> str:
    if p is None:
        return ""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _members(it: dict) -> str:
    return f" (counts any of: {', '.join(it['members'])})" if it.get("members") else ""


def _grade(g: dict) -> str:
    return f"**{g['grade']}**: " + "; ".join(g["reasons"])


def _corpus_lines(c: dict) -> list[str]:
    """The corpus check in the report: titles and abstracts across the topic, then full texts, then the papers
    a reader should open before believing the gap."""
    unread = c.get("not_analysed") or 0
    L = [f"- **Corpus check:** {c['matching']} of {c['topic_total']} topic papers in the whole corpus match "
         f"`{c['search']}` in their title or abstract"
         + (f"; {unread} of them were not analysed in this run." if unread else ".")]
    ft = c.get("fulltext")
    if ft and ft.get("full_texts_read"):
        L.append(f"- **Full texts:** {ft['matching']} of the {ft['full_texts_read']} topic papers whose full text "
                 f"could be searched mention it (of the {ft['topic_papers_checked']} best matches to the topic). "
                 "A mention is not a finding: the passages are below.")
        for cand in ft.get("candidates", [])[:5]:
            tag = "" if cand["analysed_in_this_run"] else " (not analysed in this run)"
            L.append(f"  - [{cand['paper_id']}]{tag}: \"{cand['passage'][:320]}\"")
    elif ft is not None:
        L.append("- **Full texts:** none of the topic papers' full texts could be searched, so this rests on "
                 "titles and abstracts alone.")
    elif unread:
        ex = ", ".join(f"[{x['paper_id']}]" for x in (c.get("not_analysed_examples") or [])[:5])
        L.append(f"- **Not analysed but mentioning it:** {ex}. Read these before treating the gap as real.")
    return L


def render_map(question: str, protocol: dict, m: dict, reasoning: dict | None, designs: list[dict],
               notes: dict) -> str:
    L: list[str] = [f"# Research Opportunity Map", "", f"**Question:** {question}", ""]
    corp = ", ".join(f"{n} {c}" for c, n in (m.get("corpora") or {}).items())
    L += [f"Based on **{m['N']} papers** ({corp}), **{m.get('N_fulltext', 0)} read in full**, published "
          f"{m['years'][0]}–{m['years'][1]}." if m.get("years") else f"Based on {m['N']} papers.", ""]
    if protocol.get("setting"):
        L += [f"**Setting:** {protocol['setting']}", ""]

    L += ["## How to read this map", "",
          "Every section is computed by code from evidence-checked extractions of the papers above. Grades "
          "list the reasons that produced them. Explanations of gaps are hypotheses that were each tested "
          "against the papers, and their verdicts come from the counts, not from the model. Paper ids after "
          "an item are examples, not the full list.", ""]

    L += ["## What is established", ""]
    if not m["established"]:
        L += ["_Nothing reaches the threshold (30% of papers and at least 5 papers)._", ""]
    for it in m["established"]:
        L += [f"- **{it['id']} · {it['label']}**: {it['n']} of {it['N']} papers ({_pct(it['share'])}{_ci(it)}){_members(it)}. "
              f"Evidence {_grade(it['strength'])}. {_ids(it['paper_ids'])}"]
    L += [""]

    L += ["## What is emerging", ""]
    if not m["emerging"]:
        L += ["_Nothing rises by 1.5x or more with a one-sided p of 0.2 or less._", ""]
    for it in m["emerging"]:
        t = it["trend"]
        change = (f"×{t['ratio']}" if t.get("ratio") else "new")
        L += [f"- **{it['id']} · {it['label']}**: {it['n']} papers; {_pct(it['share_early'])} of papers before "
              f"{t['since']} vs {_pct(it['share_recent'])} since ({change}"
              + (f", one-sided {_p(t['p'])}" if t.get("p") is not None else "") + f"). Evidence {_grade(it['strength'])}. "
              f"{_ids(it['paper_ids'])}"]
    L += [""]

    if m.get("watch"):
        L += ["**Watch list:** rising in recent papers, but the rise could be chance at this sample size "
              "(one-sided p above 0.2), so these are not claimed as emerging: " + "; ".join(
                  f"{w['id']} {w['label']} ({w['n']} papers, {_pct(w['share_early'])} → {_pct(w['share_recent'])}, "
                  f"{_p(w['trend']['p'])})" for w in m["watch"]) + ".", ""]

    L += ["## What is missing", ""]
    by_gap = {g["gap_id"]: g for g in (reasoning or {}).get("gaps", [])}
    if not m["gaps"]:
        L += ["_No desirable practice defined by the protocol is rare in these papers._", ""]
    for g in m["gaps"]:
        L += [f"### {g['id']} · {g['label']}", "",
              f"- **Observed:** {g['n']} of {g['N']} papers ({_pct(g['share'])}{_ci(g)}); {g['n_fulltext']} of "
              f"{g['N_fulltext']} papers read in full.{_members(g)}",
              f"- **Gap confidence:** {_grade(g['confidence'])}"]
        if g.get("corpus_check"):
            L += _corpus_lines(g["corpus_check"])
        r = by_gap.get(g["id"])
        if r:
            L += _reasoning_lines(r)
        L += [""]
    extra = [r for gid, r in by_gap.items() if not gid.startswith("G")]
    for r in extra:
        L += [f"### Reasoning on {r['gap_id']}", ""] + _reasoning_lines(r) + [""]
    if reasoning and reasoning.get("cross_cutting"):
        L += ["### Patterns across gaps", ""] + [f"- {x}" for x in reasoning["cross_cutting"]] + [""]

    L += ["## Evidence strength at a glance", "", "| Item | Papers | Full-text | Corpora | Grade |",
          "|---|---|---|---|---|"]
    for it in m["established"] + m["emerging"]:
        L += [f"| {it['id']} {it['label']} | {it['n']}/{it['N']} | {it['n_fulltext']} | "
              f"{', '.join(it['corpora'])} | {it['strength']['grade']} |"]
    for g in m["gaps"]:
        L += [f"| {g['id']} {g['label']} (gap) | {g['n']}/{g['N']} | {g['n_fulltext']}/{g['N_fulltext']} | "
              f"{', '.join(g['corpora']) or 'none'} | {g['confidence']['grade']} confidence |"]
    L += [""]

    L += ["## Potential novelty", "",
          "Pairs of components from different dimensions (for example a method and a data source) that are each "
          "common in these papers but rarely combined. *Expected* is how often they would co-occur if chosen "
          "independently; the p-value is the hypergeometric chance of seeing this few or fewer. Many pairs are "
          "compared, so treat single p-values near 0.05 as leads, not findings.", ""]
    if not m["novelty"]:
        L += ["_No underexplored combinations stand out._", ""]
    for n in m["novelty"]:
        L += [f"- **{n['id']} · {n['a']['label']} + {n['b']['label']}**: together in {n['observed']} "
              f"paper(s), {n['expected']} expected ({n['a']['n']} and {n['b']['n']} papers use each; "
              f"chance of so few by independence {_p(n.get('p'))}). "
              + (f"Together in {_ids(n['together_in'])}." if n["together_in"] else "")]
    L += [""]

    L += ["## Candidate research designs", ""]
    if not designs:
        L += ["_No designs passed the checks._", ""]
    for d in designs:
        L += [f"### {d['id']} · {d.get('title') or 'Design'}", ""]
        if d.get("research_question"):
            L += [f"*{d['research_question']}*", ""]
        L += ["| Slot | Choice |", "|---|---|"]
        for label, key in (("Target", "target"), ("Predictors", "predictors"), ("Data sources", "data_sources"),
                           ("Horizon", "horizon"), ("Spatial unit", "spatial_unit"),
                           ("Validation strategy", "validation_strategy"), ("Baseline", "baseline")):
            v = d.get(key)
            if v:
                L += [f"| {label} | {', '.join(v) if isinstance(v, list) else v} |"]
        L += [f"| Addresses | {', '.join(d['addresses'])} |", ""]
        if d.get("rests_on"):
            L += ["**Rests on:** " + "; ".join(
                f"{h['id']} ({h['verdict']}{', restates the gap' if h.get('restates_item') else ''}): {h['text']}"
                for h in d["rests_on"]), ""]
        if d.get("builds_on"):
            L += [f"**Builds on:** {_ids(d['builds_on'], 8)}", ""]
        for title, key in (("Evidence supporting this direction", "supporting"),
                           ("Evidence challenging this direction", "challenging")):
            L += [f"**{title}:**", ""]
            if not d.get(key):
                L += ["- _none identified_"]
            for p in d.get(key) or []:
                line = f"- {_cite(p['paper_id'])} {p.get('title', '')} ({p.get('year', '')}): {p.get('why', '')}"
                if p.get("quotes"):
                    line += f' From the paper: "{p["quotes"][0]}"'
                L += [line]
            L += [""]
        if d.get("risks"):
            L += ["**Risks:** " + "; ".join(d["risks"]), ""]

    L += ["## Protocol", ""]
    for f in protocol.get("fields", []):
        vals = ", ".join(f["values"]) if f["values"] else "free text"
        des = f" Desirable: {', '.join(f['desirable'])}." if f.get("desirable") else ""
        L += [f"- **{f['name'].removeprefix('q_')}** ({f['type']}: {vals}). {f['definition']}{des}"]
    if protocol.get("inclusion"):
        L += ["", "**Inclusion:** " + "; ".join(protocol["inclusion"])]
    if protocol.get("exclusion"):
        L += ["", "**Exclusion:** " + "; ".join(protocol["exclusion"])]
    cov = m.get("field_coverage") or {}
    q_cov = {k: v for k, v in cov.items() if k.startswith("q_")}
    if q_cov:
        L += ["", "**How often each question-specific field is reported:** " +
              ", ".join(f"{k.removeprefix('q_')} {_pct(v)}" for k, v in q_cov.items())]
    L += [""]

    t = m.get("thresholds") or {}
    L += ["## Method", "",
          f"- Established: at least {_pct(t.get('established_min_share'))} of papers and "
          f"{t.get('established_min_n')} papers.",
          f"- Emerging: at least {t.get('emerging_min_n')} papers, a share since {m.get('recent_since')} at least "
          f"{t.get('emerging_min_ratio')}x its earlier share, and a one-sided Fisher p of at most "
          f"{t.get('emerging_max_p')}; weaker rises go to the watch list. A rise with p of 0.05 or more is "
          "never graded high.",
          f"- Missing: a desirable value in at most {t.get('missing_max_n')} papers or "
          f"{_pct(t.get('missing_max_share'))} of papers.",
          "- Evidence strength and gap confidence are point scores; the reasons listed are the points awarded.",
          "- Shares carry 95% Wilson intervals. Emerging rises use a one-sided Fisher exact test (recent vs "
          "earlier papers); novelty uses the hypergeometric probability of so little overlap.",
          "- Values the protocol groups as one practice (for example spatial and spatiotemporal holdout) form "
          "one gap, so near-duplicates are not reported twice.",
          "- Keyword searches honour quotes, OR, AND, NOT, -exclusions and parentheses.",
          "- Extracted values without a verified quote from the paper were discarded before counting.", ""]
    if notes.get("dropped"):
        L += ["**Removed during checks:** " + "; ".join(notes["dropped"][:12]), ""]
    return "\n".join(L)


def _reasoning_lines(r: dict) -> list[str]:
    out = []
    if r.get("explanation"):
        out += [f"- **Why (reasoning):** {r['explanation']}"]
    for h in r.get("hypotheses", []):
        if h.get("restates_item"):
            out += [f"  - {h['id']} (re-measures the gap itself, not an explanation; {h['measured']}): {h['text']}"]
            continue
        mark = {"supported": "supported", "not supported": "not supported", "untestable": "untestable"}[h["verdict"]]
        out += [f"  - {h['id']} ({h.get('role', 'cause')}, **{mark}**, {h['measured']}): {h['text']}"]
    if r.get("artifact_risk"):
        out += [f"- **Could it be an artifact?** {r['artifact_risk']} risk. {r.get('artifact_reason', '')}"]
    for ln in r.get("linked_gaps", []):
        out += [f"- **Linked to {ln['gap_id']}:** {ln.get('relation', '')} ({ln['hypothesis_id']}, "
                f"{ln['verdict']})"]
    for nm in r.get("near_misses", []):
        out += [f"- **Near miss** {_cite(nm['paper_id'])}: {nm.get('what_it_does', '')}; missing: "
                f"{nm.get('what_is_missing', '')}"]
    if r.get("would_change_if"):
        out += [f"- **Would change if:** {r['would_change_if']}"]
    if r.get("speculation"):
        out += [f"- **Untested speculation:** {r['speculation']}"]
    return out


def references(pg, pids: list[str]) -> list[str]:
    rows = {r["paper_id"]: r for r in pg.execute(
        "SELECT paper_id, source, title, year, authors, license FROM papers WHERE paper_id = ANY(%s)",
        (pids,)).fetchall()}
    out = []
    for pid in pids:
        r = rows.get(pid)
        if not r:
            continue
        first = re.split(r",|\band\b", r["authors"] or "")[0].strip() or "Unknown"
        url = (f"https://pmc.ncbi.nlm.nih.gov/articles/{pid}/" if r["source"] == "pmc"
               else f"https://arxiv.org/abs/{pid}")
        out.append(f"- **{pid if r['source'] == 'pmc' else 'arXiv:' + pid}**: {r['title']} ({first} et al., "
                   f"{r['year']}). {url}")
    return out
