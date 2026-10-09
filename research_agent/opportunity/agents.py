"""The two reasoning agents of the Research Opportunity Map.

Gap Reasoning goes beneath the computed gaps: why does each gap exist, is it real or an artifact of
reporting, which gaps depend on each other, and which papers nearly fill it. Its reasoning is free, but
every explanation it wants on the map must be stated as a hypothesis with a test (a subgroup count),
and code, not the model, decides whether the evidence supports it.

Research Design turns the map into candidate studies: dataset, target, predictors, horizon, spatial unit,
validation strategy and baseline, each tied to map items and to papers that support or challenge it.
"""
from __future__ import annotations

import re

from research_agent.agents.base import Agent
from research_agent.tools.base import STR, STRS, Tool, obj
from research_agent.tools.claims import PREDICATE_DOC, TEST_TOOL, _measure, _norm, _validate
from research_agent.tools.extraction import ANALYSIS_TOOLS, _rows, _values, known_fields
from research_agent.tools.recheck import RECHECK_TOOL
from research_agent.tools.results import RESULT_TOOLS
from research_agent.tools.search import SEARCH_TOOLS

MIN_TESTABLE = 5   # a subgroup smaller than this cannot support or refute anything


def _map(ctx) -> dict:
    return (ctx.notes().get("map") or {}).get("map") or {}


def get_map(ctx) -> dict:
    from research_agent.opportunity.compute import compact_map

    m = _map(ctx)
    if not m:
        return {"error": "the map has not been computed yet"}
    out = compact_map(m)
    hyps = (ctx.notes().get("hypotheses") or {}).get("items") or []
    if hyps:
        out["hypotheses_tested"] = [{k: h[k] for k in ("id", "gap_id", "text", "verdict", "measured")}
                                    for h in hyps]
    gr = (ctx.notes().get("gap_reasoning") or {}).get("checked")
    if gr:
        out["gap_reasoning"] = gr
    return out


MIN_TREND_PAPERS = 10   # the same floor evaluate_trend uses before calling a trend


def _restates(item: dict, claim_type: str, p: dict) -> bool:
    """A hypothesis that just counts the map item itself (same field, same values, no subgroup) re-measures
    the gap; it cannot explain it."""
    if claim_type != "prevalence" or p.get("where") or p.get("field") != item.get("field"):
        return False
    item_vals = {_norm(v) for v in str(item.get("value", "")).split("|") if v}
    return bool(item_vals) and {_norm(v) for v in p.get("any_of") or []} <= item_vals


def test_hypothesis(ctx, gap_id: str, text: str, claim_type: str, predicate: dict, role: str = "cause") -> dict:
    """Measure a hypothesis about a gap and record the verdict. Code decides supported / not supported /
    untestable from the count; the model only proposes."""
    m = _map(ctx)
    items_by_id = {x["id"]: x for sec in ("gaps", "novelty", "emerging", "watch", "established") for x in m.get(sec, [])}
    if gap_id not in items_by_id:
        return {"error": f"unknown map item {gap_id}; use ids from get_map"}
    p = dict(predicate or {})
    err = _validate(claim_type, p, ctx)
    if err:
        return {"error": err}
    note = ctx.notes().get("hypotheses") or {}
    items = note.get("items") or []
    for h in items:   # the same test twice is one hypothesis, not two pieces of evidence
        if h["claim_type"] == claim_type and h["predicate"] == p:
            return {"hypothesis_id": h["id"], "verdict": h["verdict"], "measured": h["measured"],
                    "note": f"This exact test was already run as {h['id']}; cite that id instead."}
    measured = _measure(ctx, claim_type, p, text)
    if not measured:
        return {"error": "no extracted papers to test against"}
    if measured.get("text_problem"):
        return {"error": "The hypothesis text does not match the data: " + measured["text_problem"],
                "measured": measured["summary"], "fix": "Rewrite the text from the measured numbers."}
    r = measured["result"]
    if claim_type == "prevalence" and r.get("denominator", 0) < MIN_TESTABLE:
        verdict = "untestable"
    elif claim_type == "trend" and r.get("total_matching_papers", 0) < MIN_TREND_PAPERS:
        verdict = "untestable"
    else:
        verdict = "supported" if r["supported"] else "not supported"
    restates = _restates(items_by_id[gap_id], claim_type, p)
    hid = f"H{len(items) + 1}"
    items.append({"id": hid, "gap_id": gap_id, "role": role, "text": text, "claim_type": claim_type,
                  "predicate": p, "verdict": verdict, "measured": measured["summary"], "restates_item": restates,
                  "matched_paper_ids": r.get("matched_paper_ids", [])[:20],
                  "subgroup": r.get("subgroup")})
    ctx.save_note("hypotheses", {"items": items})
    out = {"hypothesis_id": hid, "verdict": verdict, "measured": measured["summary"],
           **({"subgroup": r["subgroup"]} if r.get("subgroup") else {}),
           **({"caveat": r["caveat"]} if r.get("caveat") else {})}
    if restates:
        out["note"] = ("This re-measures the map item itself, so it cannot explain it. Test a cause instead: "
                       "a difference between subgroups (use where), or a co-occurring factor.")
    return out


def find_passages(ctx, terms: list[str], fields: list[str] | None = None, paper_ids: list[str] | None = None,
                  limit: int = 15) -> dict:
    """Papers whose extracted values, stated limitations or findings mention any of `terms`, with the
    verified evidence quotes behind their fields. Use it to find near misses and the reasons authors give."""
    lists, enums = known_fields(ctx)
    fields = [f for f in (fields or lists + list(enums)) if f in lists or f in enums]
    terms_l = [t.lower() for t in terms if t.strip()]
    out = []
    for r in _rows(ctx):
        if paper_ids and r["paper_id"] not in paper_ids:
            continue
        d = r["data"]
        hits = {f: v for f in fields for v in [_values(d, f)] if v and any(t in " ".join(v).lower() for t in terms_l)}
        text_hit = any(t in (d.get("key_findings") or "").lower() or
                       any(t in x.lower() for x in d.get("limitations") or []) for t in terms_l)
        if not hits and not text_hit and not paper_ids:
            continue
        ev = d.get("evidence") or {}
        out.append({"paper_id": r["paper_id"], "title": r["title"], "year": r["year"], "corpus": r.get("corpus"),
                    "read": r["source"], "matched_fields": hits,
                    "evidence_quotes": {f: ev[f] for f in hits if f in ev},
                    "limitations_stated_by_authors": d.get("limitations") or [],
                    "key_findings": d.get("key_findings") or ""})
        if len(out) >= limit:
            break
    return {"n": len(out), "papers": out,
            "note": "evidence_quotes were checked against the paper text; limitations and findings are the "
                    "extractor's summary of what the authors state."}


MAP_TOOLS = [Tool("get_map", "The computed opportunity map: established, emerging, gaps, novelty, with numbers, "
                  "grades and reasons (plus hypotheses already tested).", obj({}), get_map, max_chars=20000)]
PASSAGE_TOOL = Tool(
    "find_passages", "Find papers whose extracted fields, stated limitations or findings mention any of the "
    "terms; returns verified evidence quotes. Pass paper_ids to read specific papers.",
    obj({"terms": STRS, "fields": STRS, "paper_ids": STRS, "limit": {"type": "integer"}}, ["terms"]),
    find_passages, max_chars=15000)
HYPOTHESIS_TOOL = Tool(
    "test_hypothesis",
    "State a hypothesis about a map item (gap_id is G#, N#, R# or E#) and the count that would test it; code "
    "measures it and records supported / not supported / untestable. role: cause | artifact | link | "
    "exception. Predicate formats: " + PREDICATE_DOC + ". Write the text from numbers you have measured "
    "(test several predicates with test_claim first if unsure).",
    obj({"gap_id": STR, "text": STR, "claim_type": {"type": "string", "enum": ["prevalence", "trend"]},
         "predicate": {"type": "object"},
         "role": {"type": "string", "enum": ["cause", "artifact", "link", "exception"]}},
        ["gap_id", "text", "claim_type", "predicate"]),
    test_hypothesis)


def _analysis(*names):
    return [t for t in ANALYSIS_TOOLS if t.name in names]


def _content_tool():
    from research_agent.tools.content import CONTENT_TOOL

    return CONTENT_TOOL


GAP_REASONING = Agent(
    name="gap_reasoning",
    role="Explains the computed research gaps: causes, artifacts, dependencies and near misses, each tested.",
    system=f"""You are the Gap Reasoning agent of a Research Opportunity Map. Code has already found the gaps
(values the question cares about that almost no paper has) and graded how confident the absence is. Your job
is to go deeper than "X is rare" and find what the evidence says about WHY, using these lenses:

1. Real or artifact? Is the absence concentrated in papers read only from abstracts (where={{field:'read',
   any_of:['abstract']}})? In one corpus? Is the field rarely reported at all (see field_coverage)?
2. Causes. What do papers that come close have in common that others lack (data type, region, method,
   corpus, period)? What limitations do authors state (find_passages)?
3. Links between gaps. Does one gap make another likely? Test co-occurrence, e.g. among papers without
   shared code, how many validate externally?
4. Exceptions and near misses. Which papers partly fill the gap, and what do they do differently?
5. What would change the conclusion.

Rules:
- Every explanation you want on the map must go through test_hypothesis with a count that could prove it
  wrong. Use `where` to test within subgroups. Code decides the verdict; "untestable" is an honest outcome.
- Measure before you write numbers into a hypothesis (test_claim or value_counts first).
- Re-counting the gap itself ("X appears in 2 of 70 papers") is not an explanation and is flagged as such.
  Explain it: compare subgroups with `where`, or test a factor that would make the practice hard or unneeded.
- Each hypothesis is tested once; to use a result again, cite its H id rather than re-running it.
- Reasoning you cannot test may go in `speculation`, clearly labelled.
- Cite papers only by ids that tools returned.
- Say what your counts are counted over. A count over the analysed papers is about the analysed papers:
  write "none of the 91 analysed papers", never "no paper in the corpus" or "the field". Each gap's
  corpus_check says how many topic papers OUTSIDE the analysed set mention it (not_analysed) and, where full
  texts were searched, gives the passages. Those papers were not read: name how many, and read the passages
  before calling the practice absent. A passage that says a study did NOT do it is evidence for the gap; one
  that says it did is evidence against.
- get_content_analysis(with_quotes=true) holds what the studies read in full say: their stated limitations,
  the future work they call for, conflicts between their findings and how their methods changed. Use it for
  candidate causes and near misses: an explanation the authors themselves give is a strong hypothesis to test.
Start with get_map, then get_content_analysis. Aim for 2-4 tested hypotheses per high or moderate confidence gap.""",
    tools=MAP_TOOLS + [TEST_TOOL, HYPOTHESIS_TOOL, PASSAGE_TOOL, RECHECK_TOOL, _content_tool()]
    + [t for t in RESULT_TOOLS if t.name == "contradictions"]
    + _analysis("value_counts", "cross_tab", "field_by_year")
          + [t for t in SEARCH_TOOLS if t.name == "corpus_count"],
    finish_schema=obj({
        "gaps": {"type": "array", "items": {"type": "object", "properties": {
            "gap_id": STR,
            "explanation": {**STR, "description": "2-4 sentences: the best-supported account of why this gap "
                                                   "exists, referring to hypothesis ids like H3"},
            "hypothesis_ids": STRS,
            "artifact_risk": {"type": "string", "enum": ["low", "medium", "high"]},
            "artifact_reason": STR,
            "linked_gaps": {"type": "array", "items": {"type": "object", "properties": {
                "gap_id": STR, "relation": STR, "hypothesis_id": STR}}},
            "near_misses": {"type": "array", "items": {"type": "object", "properties": {
                "paper_id": STR, "what_it_does": STR, "what_is_missing": STR}}},
            "would_change_if": STR,
            "speculation": STR}}},
        "cross_cutting": {**STRS, "description": "Deeper patterns across gaps, each referring to hypothesis ids"},
    }, ["gaps"]),
    max_turns=18,
    strong_model=True,
    long_output=True,
)

_DESIGN_ITEM = {"type": "object", "properties": {
    "title": STR,
    "research_question": STR,
    "target": {**STR, "description": "What is predicted or estimated, with units"},
    "predictors": STRS,
    "data_sources": {**STRS, "description": "Named datasets or data types, ideally ones papers on the map used"},
    "horizon": STR,
    "spatial_unit": STR,
    "validation_strategy": {**STR, "description": "How the evaluation is split, e.g. leave-one-district-out"},
    "baseline": STR,
    "addresses": {**STRS, "description": "Ids this design targets: map items (G#, N#, R#, E#) and gaps or "
                                         "directions from the studies' content (CG#, CD#)"},
    "rests_on": {**STRS, "description": "Tested hypothesis ids (H#) that justify the design"},
    "builds_on": {**STRS, "description": "Paper ids whose data or methods it reuses"},
    "supporting": {"type": "array", "items": {"type": "object", "properties": {"paper_id": STR, "why": STR}}},
    "challenging": {"type": "array", "items": {"type": "object", "properties": {"paper_id": STR, "why": STR}}},
    "risks": STRS,
    "guidance": {**STR, "description": "One or two sentences on how the design relates to current guideline "
                                       "recommendations, naming the guideline and citing its marker, e.g. "
                                       "'According to the 2022 EULAR recommendations ... [GL1.3]'. '' when the "
                                       "guideline library has nothing relevant."},
}, "required": ["title", "target", "validation_strategy", "addresses"]}

_DESIGN_PLAN_ITEM = {"type": "object", "properties": {
    "title": STR,
    "idea": {**STR, "description": "The study in one or two sentences: what it estimates and how it is tested"},
    "addresses": {**STRS, "description": "Ids this design targets: map items (G#, N#, R#, E#) and gaps or "
                                         "directions from the studies' content (CG#, CD#)"},
    "rests_on": {**STRS, "description": "Tested hypothesis ids (H#) that justify the design"},
    "papers": {**STRS, "description": "Paper ids you found that bear on it, for or against"},
}, "required": ["title", "idea", "addresses"]}

def _guideline_tool():
    from research_agent.tools.guidelines import GUIDELINE_TOOL

    return GUIDELINE_TOOL


_DESIGN_BRIEF = """Each design is a study a researcher could actually run. Base it first on what the studies read in
full say (get_content_analysis): the gaps they point to (CG#) and the directions drawn from them (CD#), with
the limitations and future work their authors state; then on the counted gaps (G#) and novel combinations
(N#) from the map, which show how widespread a problem is. Ground choices in the map's numbers and the tested hypotheses from gap
reasoning (get_map includes them). Prefer designs that fix a high-confidence gap with data the field already
uses, and designs resting on supported hypotheses; one resting on a hypothesis that was not supported must
say why it is still worth doing. Use only paper ids that tools returned."""

# Designs are written one per call. Asking for all of them in one finish produced a long JSON that the output
# limit cut off ("finish arguments were cut off"), losing the last design or all of them. The planner returns
# short outlines; a writer turns each into a full design in its own call, so one long design can never take
# the others down with it.
DESIGN = Agent(
    name="design_plan",
    role="Turns the opportunity map into concrete candidate research designs with supporting and challenging "
         "papers.",
    system="""You are the Research Design agent of a Research Opportunity Map. Choose 2-4 candidate studies and
return a short outline of each: a title, the idea in one or two sentences, the map items it addresses, the
hypotheses it rests on and the paper ids that bear on it. Each design is written out in full later, one at a
time, so keep the outline short.

""" + _DESIGN_BRIEF,
    tools=MAP_TOOLS + [PASSAGE_TOOL, _content_tool()] + _analysis("value_counts", "cross_tab", "list_extractions"),
    finish_schema=obj({"designs": {"type": "array", "items": _DESIGN_PLAN_ITEM}}, ["designs"]),
    max_turns=12,
    strong_model=True,
    long_output=True,
)

DESIGN_WRITER = Agent(
    name="design_writer",
    role="Writes one candidate research design in full.",
    system="""You are the Research Design Writer of a Research Opportunity Map. Write ONE candidate study in
full, from the outline in your task.

Give: research question, target (with units), predictors, data sources, forecast horizon or time frame,
spatial unit, validation strategy (how data is split: this is where most gaps are), baseline to beat, and the
map items it addresses. Then:
- supporting: papers whose data, methods or findings make the design feasible or promising, and why;
- rests_on: the tested hypotheses (H#) that justify it;
- challenging: papers whose findings or stated limitations argue it may fail or be hard, and why. Look for
  these honestly (find_passages on limitations): a design with no challenging evidence was not looked at hard
  enough.
Keep each text field to one or two sentences.

If the design concerns treatment, management or care, look up what guidelines recommend with
guideline_recommendations and fill `guidance`: name the guideline in the sentence and cite the recommendation
with its marker exactly as the tool gives it. A guideline is what is recommended, not evidence for the design.

""" + _DESIGN_BRIEF,
    tools=MAP_TOOLS + [PASSAGE_TOOL, _content_tool()] + _analysis("list_extractions") + [_guideline_tool()],
    finish_schema=_DESIGN_ITEM,
    max_turns=6,
    strong_model=True,
    long_output=True,
)


def write_designs(ctx) -> dict:
    """Plan the designs, then write each in its own call. Returns {"designs": [...]} for check_designs. A
    design whose writer fails keeps its outline, so it is reported thinly rather than lost."""
    import json as _json

    ctx.emit("design", "start", {"task": "plan the designs, then write each one in its own call"})
    plan = DESIGN.run(ctx, "Choose candidate research designs from the map and outline them.")
    outlines = [o for o in plan.get("designs") or [] if isinstance(o, dict)][:4]
    designs = []
    for k, o in enumerate(outlines, 1):
        task = (f"Write design {k} of {len(outlines)} in full. Outline:\n"
                + _json.dumps({x: o.get(x) for x in ("title", "idea", "addresses", "rests_on", "papers")}))
        try:
            full = DESIGN_WRITER.run(ctx, task)
        except Exception as exc:
            ctx.emit("design_writer", "error", {"error": str(exc)[:300]})
            full = {}
        if not isinstance(full, dict) or not full.get("title"):
            full = {"title": o.get("title"), "research_question": o.get("idea"),
                    "_note": "written from the outline only"}
        # the outline's references are kept when the writer leaves them out
        full["addresses"] = full.get("addresses") or o.get("addresses") or []
        full["rests_on"] = full.get("rests_on") or o.get("rests_on") or []
        designs.append(full)
    ctx.emit("design", "finish", {"output": {"designs": len(designs)}})
    return {"designs": designs}


# ---------------------------------------------------------------- post-checks on the agents' output
# A sentence about absence ("no paper", "none of", "0/91") that names the corpus, the literature or the field
# when what was counted is the analysed papers. The claims layer already refuses that wording; gap reasoning
# never went through it, which is how "No paper in the corpus reports a rainfall x temperature interaction"
# reached a report beside a corpus check that had found ten candidate papers.
_ABSENCE = re.compile(r"\b(no|none|never|zero|nothing|absent|absence)\b|\b0\s*(/|of)\s*\d", re.I)
_WIDE = [
    (re.compile(r"\bno (paper|papers|study|studies) in the (whole |wider |entire |full )?(corpus|literature|field)\b", re.I),
     "no analysed paper"),
    (re.compile(r"\bnone of the (papers|studies) in the (whole |wider |entire |full )?(corpus|literature|field)\b", re.I),
     "none of the analysed papers"),
    (re.compile(r"\b(in|across|throughout) the (whole |wider |entire |full )?(corpus|literature|field)\b", re.I),
     "in the analysed papers"),
]


def _gap_corpus(m: dict, gid: str) -> dict:
    return next((g.get("corpus_check") or {} for g in m.get("gaps", []) if g.get("id") == gid), {})


def scope_wording(text: str, corpus: dict | None = None) -> tuple[str, str | None]:
    """Rewrite absence sentences that claim more than was counted, and add what the corpus check found
    outside the analysed papers. Returns (text, a note saying what was changed, or None)."""
    if not text:
        return text, None
    changed = False
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _ABSENCE.search(sentence):
            for rx, rep in _WIDE:
                # keep a capital where the sentence had one ("No paper in the corpus" -> "No analysed paper")
                sentence, n = rx.subn(lambda m, r=rep: r[0].upper() + r[1:] if m.group(0)[0].isupper() else r,
                                      sentence)
                changed = changed or bool(n)
        out.append(sentence)
    text = " ".join(out)
    unread = (corpus or {}).get("not_analysed") or 0
    ft = (corpus or {}).get("fulltext") or {}
    extra = []
    if unread and "not analysed" not in text and "not read" not in text:
        extra.append(f"{unread} further topic papers mention it in their title or abstract and were not analysed.")
    if ft.get("matching") and "full text" not in text.lower():
        extra.append(f"{ft['matching']} of {ft['full_texts_read']} topic papers mention it in their full text; "
                     "a mention is not a finding, and their passages are listed with the corpus check.")
    if extra:
        text = (text.rstrip() + " " + " ".join(extra)).strip()
    note = None
    if changed or extra:
        note = ("absence was stated for the corpus but counted over the analysed papers; the wording was "
                "narrowed" if changed else "") + ("; " if changed and extra else "") + \
               ("what the corpus check found outside the analysed papers was added" if extra else "")
    return text, note


def check_gap_reasoning(ctx, raw: dict) -> tuple[dict, list[str]]:
    """Keep only references that exist: map ids, tested hypotheses, shortlisted papers."""
    m = _map(ctx)
    gap_ids = {g["id"] for g in m.get("gaps", [])}
    all_ids = gap_ids | {x["id"] for s in ("novelty", "emerging", "watch", "established") for x in m.get(s, [])}
    hyps = {h["id"]: h for h in ((ctx.notes().get("hypotheses") or {}).get("items") or [])}
    shortlist = set(ctx.shortlist_ids())
    dropped: list[str] = []
    out = []
    for g in raw.get("gaps") or []:
        gid = g.get("gap_id")
        if gid not in all_ids:
            dropped.append(f"explanation for unknown item {gid}")
            continue
        hids = [h for h in (g.get("hypothesis_ids") or []) if h in hyps]
        dropped += [f"{gid}: unknown hypothesis {h}" for h in (g.get("hypothesis_ids") or []) if h not in hyps]
        hids = list(dict.fromkeys(hids + [h for h, v in hyps.items() if v["gap_id"] == gid]))
        links = []
        for ln in g.get("linked_gaps") or []:
            if ln.get("gap_id") in all_ids and ln.get("hypothesis_id") in hyps:
                links.append({**ln, "verdict": hyps[ln["hypothesis_id"]]["verdict"]})
            else:
                dropped.append(f"{gid}: link to {ln.get('gap_id')} without a tested hypothesis")
        near = []
        for nm in g.get("near_misses") or []:
            if nm.get("paper_id") in shortlist:
                near.append(nm)
            else:
                dropped.append(f"{gid}: near miss {nm.get('paper_id')} is not in the analysed papers")
        scoped, note = scope_wording(g.get("explanation", ""), _gap_corpus(m, gid))
        if note:
            dropped.append(f"{gid}: {note}")
        out.append({"gap_id": gid, "explanation": scoped,
                    "hypotheses": [hyps[h] for h in hids], "artifact_risk": g.get("artifact_risk"),
                    "artifact_reason": scope_wording(g.get("artifact_reason", ""))[0], "linked_gaps": links,
                    "near_misses": near, "would_change_if": g.get("would_change_if", ""),
                    "speculation": g.get("speculation", "")})
    return {"gaps": out, "cross_cutting": [scope_wording(str(x))[0] for x in raw.get("cross_cutting") or []]}, dropped


def _paper_card(ctx, pid: str, rows_by_id: dict) -> dict:
    r = rows_by_id.get(pid)
    if not r:
        return {"paper_id": pid}
    d = r["data"]
    ev = d.get("evidence") or {}
    quotes = [q for f in ev for q in ev[f]][:3]
    return {"paper_id": pid, "title": r["title"], "year": r["year"], "corpus": r.get("corpus"),
            "read": r["source"], "key_findings": d.get("key_findings") or "",
            "limitations": (d.get("limitations") or [])[:3], "quotes": quotes}


def check_designs(ctx, raw: dict) -> tuple[list[dict], list[str]]:
    m = _map(ctx)
    ids = {x["id"] for s in ("gaps", "novelty", "emerging", "watch", "established") for x in m.get(s, [])}
    from research_agent.tools import content

    a = content.of(ctx)
    ids |= {g["id"] for g in a.get("gaps") or []} | {d["id"] for d in a.get("directions") or []}
    hyps = {h["id"]: h for h in ((ctx.notes().get("hypotheses") or {}).get("items") or [])}
    rows_by_id = {r["paper_id"]: r for r in _rows(ctx)}
    dropped: list[str] = []
    designs = []
    for k, d in enumerate(raw.get("designs") or [], 1):
        did = f"D{k}"
        refs = list(dict.fromkeys((d.get("addresses") or []) + (d.get("rests_on") or [])))
        addresses = [a for a in refs if a in ids]
        rests_on = [a for a in refs if a in hyps]      # hypotheses given as 'addresses' are moved here
        dropped += [f"{did}: unknown reference {a}" for a in refs if a not in ids and a not in hyps]
        if not addresses:
            dropped.append(f"{did} ({d.get('title')}): dropped, it addresses no item on the map")
            continue

        def papers(key):
            keep = []
            for x in d.get(key) or []:
                pid = x.get("paper_id") if isinstance(x, dict) else x
                if pid in rows_by_id:
                    keep.append({**(x if isinstance(x, dict) else {}), **_paper_card(ctx, pid, rows_by_id)})
                else:
                    dropped.append(f"{did}: {key} paper {pid} is not in the analysed papers")
            return keep

        designs.append({**{k2: d.get(k2) for k2 in ("title", "research_question", "target", "predictors",
                                                      "data_sources", "horizon", "spatial_unit",
                                                      "validation_strategy", "baseline", "risks", "guidance")},
                        "id": did, "addresses": addresses,
                        "rests_on": [{"id": h, "verdict": hyps[h]["verdict"], "text": hyps[h]["text"],
                                      "restates_item": hyps[h].get("restates_item", False)} for h in rests_on],
                        "builds_on": [p["paper_id"] for p in papers("builds_on")],
                        "supporting": papers("supporting"), "challenging": papers("challenging")})
    return designs, dropped
