"""What the papers read in full actually say: findings, methods and how they changed, gaps and directions.

Counting tells a reader how often something appears. It cannot say what the studies found, how a field's
methods moved from one approach to another and why, or which questions the authors themselves say are still
open. This module reads for that, in two stages:

    per paper   one deep read of each paper read in full (cached across runs): aim, design, data and
                analysis in the authors' terms, why they chose their approach, findings, how they set their
                work against earlier studies, limitations and future work. Every finding, comparison,
                limitation, recommendation and methods description carries a passage copied from the paper,
                and code drops whatever it cannot find in the text read (the same check as extraction).
    across      three calls of a stronger model over all the paper records, ordered by year:
                  1. findings: themes and content claims, each tied to the verified passages behind it and
                     to any that contradict it
                  2. method evolution: periods, the shifts between them with the reasons authors give, and
                     what stayed the same
                  3. gaps and directions: open questions grounded in the papers' own limitations, future work
                     and unresolved conflicts, and directions that address them

Code checks the synthesis as it checks everything else: a passage id must exist (so every statement rests on
text that was found in the paper), a paper cited in prose must be one that was analysed, a period may only
hold papers published in its years, a shift must run forwards in time (its "after" papers later than its
"before" papers), and every direction must address a gap that survived. The strength of a claim (how many
studies support it, how many contradict it) is counted by code from the passages, never taken from the model.

The result is saved on the run (note `content_analysis`) and used by the report, the opportunity map, the
trend and gap agents, gap reasoning and the research designs.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings
from research_agent.tools.base import Tool, obj

VERSION = "content-v1"
NOTE = "content_analysis"
_STR = {"type": "string"}
_ITEM = lambda key, desc: {"type": "array", "items": {"type": "object", "properties": {  # noqa: E731
    key: {"type": "string", "description": desc},
    "quote": {"type": "string", "description": "Up to 40 words copied WORD FOR WORD from the text"}},
    "required": [key, "quote"]}}

# ---------------------------------------------------------------- stage 1: one deep read per paper
PAPER_SYSTEM = """You read a health research paper closely so its content can be compared with other studies on
the same question. Read the whole text you are given before recording anything.

Describe the study in your own precise words, as a reviewer would for a methods table:
- aim: what the study set out to answer.
- design: the study design and how it was carried out (e.g. a community-based cross-sectional survey with
  multistage cluster sampling; a retrospective hospital cohort; a randomised trial with two arms).
- setting: where, when and in whom (country, site, period, population).
- data: what data or samples were collected and how (instruments, case definitions, diagnostic criteria,
  assays, registries), with the sample size.
- analysis: the analytical methods in detail (statistical tests, models, adjustment, validation), and how the
  main outcome was defined or measured.

Then copy evidence from the text. Every item needs a passage of up to 40 words copied WORD FOR WORD from the
text above; code checks each passage and drops any item whose passage it cannot find, so never paraphrase:
- methods_quotes: 1-3 passages that show the design, data or analysis described above.
- rationale: why the authors chose their approach, or what they say is new about it.
- findings: the main results, each with its numbers as written (up to 6).
- prior_work: how the authors set their results or methods against earlier studies ("similar to", "in
  contrast to", "unlike previous studies", "for the first time").
- limitations: limitations the authors state. Do not invent any.
- future_work: further research or practice the authors recommend.
- interpretation: in one or two sentences, what the authors conclude their results mean.
Record only what the text states. An empty list is correct when the paper says nothing of that kind."""

PAPER_TOOL = {
    "name": "record_paper_analysis",
    "description": "Record the close reading of this paper.",
    "input_schema": {"type": "object", "properties": {
        "aim": _STR, "design": _STR, "setting": _STR, "data": _STR, "analysis": _STR,
        "methods_quotes": {"type": "array", "items": _STR},
        "rationale": _ITEM("statement", "Why this approach, or what is new about it"),
        "findings": _ITEM("finding", "One result, with its numbers"),
        "prior_work": _ITEM("statement", "How the result or method compares with earlier studies"),
        "limitations": _ITEM("limitation", "A limitation the authors state"),
        "future_work": _ITEM("recommendation", "Research or practice the authors recommend"),
        "interpretation": _STR,
    }, "required": ["aim", "design", "setting", "data", "analysis", "methods_quotes", "findings",
                    "limitations", "future_work", "interpretation"]},
}

# item kind -> (key holding the statement, letter in its passage id)
KINDS = {"rationale": ("statement", "R"), "findings": ("finding", "F"), "prior_work": ("statement", "P"),
         "limitations": ("limitation", "L"), "future_work": ("recommendation", "W")}
KIND_NAMES = {"R": "rationale", "F": "finding", "P": "comparison with earlier work", "L": "limitation",
              "W": "future work", "M": "methods"}


def _clean(s, n: int = 700) -> str:
    return " ".join(str(s or "").split())[:n]


def verify_card(args: dict, text: str) -> dict:
    """Keep each item only when its passage is really in the text read; number the survivors."""
    from research_agent.tools.extraction import _words, quote_found

    words = _words(text)
    shingles = {tuple(words[i:i + k]) for k in (3, 4, 5) for i in range(len(words) - k + 1)}
    card = {k: _clean(args.get(k), 900) for k in ("aim", "design", "setting", "data", "analysis", "interpretation")}
    dropped = 0
    mq = [q for q in (args.get("methods_quotes") or []) if isinstance(q, str)]
    card["methods_quotes"] = [_clean(q, 400).lstrip("# ") for q in mq if quote_found(q, words, shingles)][:3]
    dropped += len(mq) - len(card["methods_quotes"])
    card["methods_verified"] = bool(card["methods_quotes"])
    for kind, (key, _letter) in KINDS.items():
        kept = []
        for it in args.get(kind) or []:
            if not isinstance(it, dict) or not it.get(key):
                continue
            if isinstance(it.get("quote"), str) and quote_found(it["quote"], words, shingles):
                kept.append({"text": _clean(it[key], 500), "quote": _clean(it["quote"], 400).lstrip("# ")})
            else:
                dropped += 1
        card[kind] = kept[:8]
    card["dropped_without_passage"] = dropped
    return card


def _paper_text(row: dict, sections: list) -> str:
    from research_agent.ingestion.fulltext import select_for_reading

    body = select_for_reading(sections or [], budget=settings.analysis_read_chars)
    return f"Title: {row['title']}\n\nAbstract: {row.get('abstract') or ''}\n\nFull text:\n{body}"


def _read_one(llm, row: dict, sections: list) -> dict:
    from research_agent.agents.base import repair_truncated_json

    text = _paper_text(row, sections)
    resp = llm.chat(PAPER_SYSTEM, [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    tools=[PAPER_TOOL], force_tool="record_paper_analysis",
                    max_tokens=max(4000, settings.extraction_max_tokens))
    if not resp.tool_calls:
        raise ValueError("model returned no reading")
    args = resp.tool_calls[0].input or {}
    if "_raw_arguments" in args:          # cut off: keep what was complete
        args = repair_truncated_json(args["_raw_arguments"] or "") or {}
    return verify_card(args, text)


def fulltext_papers(ctx) -> list[dict]:
    """The run's papers read in full (within its cohort), oldest first."""
    from research_agent.tools.extraction import _rows

    rows = [r for r in _rows(ctx) if r["source"] == "fulltext"]
    rows.sort(key=lambda r: (r["year"] or 0, r["paper_id"]))
    return rows[: settings.analysis_max_papers]


def read_cards(ctx, paper_ids: list[str]) -> tuple[dict[str, dict], list[dict]]:
    """Cards for these papers: cached ones reused, the rest read now. Returns ({paper_id: card}, failures)."""
    if not paper_ids:
        return {}, []
    cached = {r["paper_id"]: r["data"] for r in ctx.pg.execute(
        "SELECT paper_id, data FROM paper_analyses WHERE version=%s AND paper_id = ANY(%s)",
        (VERSION, paper_ids)).fetchall()}
    todo = [p for p in paper_ids if p not in cached]
    failed: list[dict] = []
    if todo:
        rows = {r["paper_id"]: r for r in ctx.pg.execute(
            "SELECT paper_id, title, abstract FROM papers WHERE paper_id = ANY(%s)", (todo,)).fetchall()}
        secs = {r["paper_id"]: r["sections"] for r in ctx.pg.execute(
            "SELECT paper_id, sections FROM paper_fulltext WHERE status='ok' AND paper_id = ANY(%s)",
            (todo,)).fetchall()}
        todo = [p for p in todo if p in rows and secs.get(p)]
        ctx.emit("content_analysis", "start", {"task": f"reading {len(todo)} papers in depth"})
        llm = ctx.llm_factory(step="paper_analysis")
        setattr(llm, "_step", "paper_analysis")
        with ThreadPoolExecutor(max_workers=max(1, settings.extraction_workers)) as pool:
            futs = {pool.submit(_read_one, llm, rows[p], secs[p]): p for p in todo}
            for fut in as_completed(futs):
                pid = futs[fut]
                try:
                    card = fut.result()
                except Exception as exc:          # one paper must not sink the analysis
                    failed.append({"paper_id": pid, "error": str(exc)[:200]})
                    continue
                ctx.pg.execute(
                    "INSERT INTO paper_analyses (paper_id, version, data, model) VALUES (%s,%s,%s::jsonb,%s) "
                    "ON CONFLICT (paper_id, version) DO UPDATE SET data=EXCLUDED.data, model=EXCLUDED.model, "
                    "created_at=now()", (pid, VERSION, json.dumps(card), getattr(llm, "model", "")))
                cached[pid] = card
    return {p: cached[p] for p in paper_ids if p in cached}, failed


# ---------------------------------------------------------------- passages and the digest the model reads
def passages(cards: dict[str, dict]) -> dict[str, dict]:
    """Every verified passage by id: '<paper>#F2' is the paper's second finding, '#M1' its first methods passage."""
    out = {}
    for pid, c in cards.items():
        for i, q in enumerate(c.get("methods_quotes") or [], 1):
            out[f"{pid}#M{i}"] = {"paper_id": pid, "kind": "M", "text": "", "quote": q}
        for kind, (_key, letter) in KINDS.items():
            for i, it in enumerate(c.get(kind) or [], 1):
                out[f"{pid}#{letter}{i}"] = {"paper_id": pid, "kind": letter, "text": it["text"], "quote": it["quote"]}
    return out


def cite(pid: str) -> str:
    return f"[{pid}]" if pid.startswith(("PMC", "UP", "PMID")) else f"[arXiv:{pid}]"


def digest(papers: list[dict], cards: dict[str, dict], per_paper: int = 2600) -> str:
    """All paper records, oldest first, in the form the synthesis calls read. Passages are named by id only."""
    blocks = []
    for r in papers:
        c = cards.get(r["paper_id"])
        if not c:
            continue
        pid = r["paper_id"]
        lines = [f"=== {pid} · {r['year']} · {r['title'][:160]} ===",
                 f"Aim: {c['aim']}", f"Design: {c['design']}", f"Setting: {c['setting']}", f"Data: {c['data']}",
                 f"Analysis: {c['analysis']}" + ("" if c.get("methods_verified") else " (methods not confirmed by a passage)")]
        if c.get("methods_quotes"):
            lines.append("Methods passages: " + ", ".join(f"{pid}#M{i}" for i in range(1, len(c['methods_quotes']) + 1)))
        for kind, (_key, letter) in KINDS.items():
            for i, it in enumerate(c.get(kind) or [], 1):
                lines.append(f"{pid}#{letter}{i} ({KIND_NAMES[letter]}): {it['text']}")
        if c.get("interpretation"):
            lines.append(f"Authors conclude: {c['interpretation']}")
        blocks.append("\n".join(lines)[:per_paper])
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- stage 2: across the papers
_SYNTH_BASE = """You are writing the analytical core of a literature review for a health researcher, from close
readings of every study that was read in full. The studies are listed oldest first. Each passage has an id
like PMC123#F2 (F finding, M methods, R rationale, P comparison with earlier work, L limitation, W future
work); code has confirmed that every listed passage is in its paper.

Research question: {question}

Rules:
- Work from what the studies say, not from how many there are. Compare, contrast and explain.
- Every statement you make must rest on passage ids from the list; give the ids in the fields provided.
- In prose, cite papers as [PMC1234567], [arXiv:2401.00001], [PMID123] or [UP12] exactly as they appear in
  the list. Never cite a paper that is not listed, and never invent ids, numbers or details.
- Do not write counts of studies ("5 of 12 studies"): code counts the passages you cite and adds the numbers.
- Write in measured scientific prose, with no em dashes."""

FINDINGS_SYSTEM = _SYNTH_BASE + """

Your task: what do the studies find? Group the results into themes (e.g. prevalence and its range, risk
factors, outcomes of treatment) and, for each theme, write a synthesis that sets the studies' results side by
side, notes where they agree, where they differ and what about the studies (setting, population, design, case
definition, period) may explain the differences. Then state the content claims a reader can take away: each
claim names the passages that support it and any that contradict it."""

FINDINGS_TOOL = {"name": "record_findings", "description": "Record the synthesis of findings.",
                 "input_schema": obj({
                     "themes": {"type": "array", "items": obj({
                         "theme": _STR,
                         "synthesis": {**_STR, "description": "One or two paragraphs comparing the studies"},
                         "evidence": {"type": "array", "items": _STR}}, ["theme", "synthesis", "evidence"])},
                     "claims": {"type": "array", "items": obj({
                         "statement": _STR,
                         "supporting": {"type": "array", "items": _STR, "description": "Passage ids"},
                         "contradicting": {"type": "array", "items": _STR, "description": "Passage ids"},
                         "why_they_differ": {**_STR, "description": "When there are contradicting passages"}},
                         ["statement", "supporting"])}}, ["themes", "claims"])}

EVOLUTION_SYSTEM = _SYNTH_BASE + """

Your task: how have the research methods changed over time? This is the most important part of the review,
so be detailed and concrete. Using the designs, settings, data, analysis and methods passages of each study,
and what authors say about earlier work and about why they chose their approach:
- divide the period covered into two to five phases whose methods genuinely differ (choose the boundaries
  from the studies, not from round numbers), and for each phase describe in detail the typical study
  designs, populations and settings, case definitions and diagnostic criteria, data collection, sample
  sizes, analytical and statistical methods, and validation or quality practices, naming the studies;
- describe each shift between phases: what changed, when it appears, which studies show the older and the
  newer practice, and the reasons authors give (or that their rationale and comparisons with earlier work
  suggest);
- say what has not changed, and which methodological problems recur across the whole period.
List under each phase only studies published within its years."""

EVOLUTION_TOOL = {"name": "record_method_evolution", "description": "Record how the methods changed over time.",
                  "input_schema": obj({
                      "overview": {**_STR, "description": "Two or three paragraphs: the arc of the whole period"},
                      "phases": {"type": "array", "items": obj({
                          "label": _STR, "from_year": {"type": "integer"}, "to_year": {"type": "integer"},
                          "description": {**_STR, "description": "A detailed paragraph or two on the methods of the phase"},
                          "papers": {"type": "array", "items": _STR},
                          "evidence": {"type": "array", "items": _STR}},
                          ["label", "from_year", "to_year", "description", "papers", "evidence"])},
                      "shifts": {"type": "array", "items": obj({
                          "shift": {**_STR, "description": "From what to what, in a short phrase"},
                          "description": {**_STR, "description": "What changed, when, and why"},
                          "before": {"type": "array", "items": _STR, "description": "Paper ids showing the older practice"},
                          "after": {"type": "array", "items": _STR, "description": "Paper ids showing the newer practice"},
                          "evidence": {"type": "array", "items": _STR}},
                          ["shift", "description", "before", "after", "evidence"])},
                      "continuities": {"type": "array", "items": obj({
                          "statement": _STR, "evidence": {"type": "array", "items": _STR}}, ["statement", "evidence"])},
                  }, ["overview", "phases", "shifts"])}

GAPS_SYSTEM = _SYNTH_BASE + """

Your task: what remains unknown or poorly done, and what should be studied next? Base every gap on the
studies' content: limitations their authors state, future work they call for, conflicts between their
findings that no study resolves, methods weaknesses that recur, and populations, settings or questions the
studies say were left out. A gap is not "few studies exist"; it is a question the evidence leaves open, and
why it matters for the research question. Then propose research directions, each addressing one or more gaps
by their number (CG1, CG2 ...), with a sketch of a study design that would close it. You are given the
findings and method evolution already written, so build on them rather than repeating them."""

GAPS_TOOL = {"name": "record_gaps_directions", "description": "Record the gaps and research directions.",
             "input_schema": obj({
                 "gaps": {"type": "array", "items": obj({
                     "gap": _STR,
                     "kind": {"type": "string", "enum": ["unanswered_question", "conflicting_evidence",
                                                         "methodological_weakness", "population_or_setting",
                                                         "translation_to_practice"]},
                     "explanation": {**_STR, "description": "Why it is open and why it matters, citing papers"},
                     "evidence": {"type": "array", "items": _STR}}, ["gap", "kind", "explanation", "evidence"])},
                 "directions": {"type": "array", "items": obj({
                     "direction": _STR,
                     "rationale": _STR,
                     "design_sketch": {**_STR, "description": "Design, population, data, analysis, in 2-3 sentences"},
                     "addresses": {"type": "array", "items": _STR, "description": "Gap numbers, e.g. CG1"},
                     "evidence": {"type": "array", "items": _STR}}, ["direction", "rationale", "addresses"])},
             }, ["gaps", "directions"])}

_ID = re.compile(r"(?:arXiv:)?(?:(?:PMC|UP|PMID)\d+|\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:#[A-Z]\d+)?")
_PAPER_CITE = re.compile(r"\[(?:arXiv:)?((?:PMC|UP|PMID)\d+|\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})\]")


def _call(ctx, system: str, tool: dict, user: str) -> dict:
    from research_agent.agents.base import repair_truncated_json

    llm = ctx.llm_factory(strong=True, step="content_synthesis")
    setattr(llm, "_step", "content_synthesis")
    resp = llm.chat(system, [{"role": "user", "content": [{"type": "text", "text": user}]}],
                    tools=[tool], force_tool=tool["name"], max_tokens=settings.long_output_max_tokens)
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    if "_raw_arguments" in args:
        args = repair_truncated_json(args["_raw_arguments"] or "") or {}
    return args


class _Checker:
    """Validates ids in the synthesis against the paper records, collecting what was removed and why."""

    def __init__(self, papers: list[dict], cards: dict[str, dict]):
        self.years = {r["paper_id"]: r["year"] for r in papers if r["paper_id"] in cards}
        self.passages = passages(cards)
        self.dropped: list[str] = []

    def ids(self, ids, where: str) -> list[str]:
        good = []
        for x in ids or []:
            x = str(x).strip().strip("[]")
            if x in self.passages:
                good.append(x)
            else:
                self.dropped.append(f"{where}: unknown passage {x}")
        return list(dict.fromkeys(good))

    def papers(self, ids, where: str) -> list[str]:
        good = []
        for x in ids or []:
            x = str(x).strip().strip("[]").removeprefix("arXiv:").split("#")[0]
            if x in self.years:
                good.append(x)
            else:
                self.dropped.append(f"{where}: paper {x} was not read in full")
        return list(dict.fromkeys(good))

    def prose(self, text: str, where: str) -> str:
        """Paper citations in prose must be analysed papers; a passage id written in prose becomes its paper."""
        text = _clean(text, 6000)
        # "[PMC1#F2, PMC3#L1]" -> "[PMC1] [PMC3]"; a passage id written in prose becomes its paper's citation
        def split(m):
            parts = [x.strip() for x in re.split(r"[,;]", m.group(1)) if x.strip()]
            if all(_ID.fullmatch(x) for x in parts):
                return " ".join(f"[{x}]" for x in parts)
            return m.group(0)
        text = re.sub(r"\[([^\]\[]+[,;][^\]\[]+)\]", split, text)
        text = re.sub(r"\[((?:arXiv:)?[^\]\s#]+)#[A-Z]\d+\]", r"[\1]", text)

        def fix(m):
            pid = m.group(1)
            if pid in self.years:
                return cite(pid)
            self.dropped.append(f"{where}: citation of {pid} removed (not read in full)")
            return ""
        return re.sub(r"\s+([.,;:])", r"\1", _PAPER_CITE.sub(fix, text)).strip()

    def papers_of(self, pids) -> list[str]:
        return list(dict.fromkeys(self.passages[p]["paper_id"] for p in pids))


def _strength(n_for: int, n_against: int) -> str:
    if n_for == 0:
        return "no supporting study"
    s = "1 study" if n_for == 1 else f"{n_for} studies"
    if n_against:
        a = "1 study" if n_against == 1 else f"{n_against} studies"
        return f"contested: supported by {s}, contradicted by {a}"
    return f"reported by a single study" if n_for == 1 else f"consistent across {s}"


def check_findings(raw: dict, ck: _Checker) -> dict:
    themes = []
    for k, t in enumerate(raw.get("themes") or [], 1):
        ev = ck.ids(t.get("evidence"), f"theme {k}")
        if not ev:
            ck.dropped.append(f"theme '{_clean(t.get('theme'), 80)}' dropped: no verified passage")
            continue
        themes.append({"theme": _clean(t.get("theme"), 160), "synthesis": ck.prose(t.get("synthesis"), f"theme {k}"),
                       "evidence": ev, "papers": ck.papers_of(ev)})
    claims = []
    for c in raw.get("claims") or []:
        sup = ck.ids(c.get("supporting"), "claim")
        con = [x for x in ck.ids(c.get("contradicting"), "claim") if x not in sup]
        if not sup:
            ck.dropped.append(f"claim '{_clean(c.get('statement'), 80)}' dropped: no verified supporting passage")
            continue
        p_for, p_against = ck.papers_of(sup), [p for p in ck.papers_of(con) if p not in ck.papers_of(sup)]
        claims.append({"id": f"K{len(claims) + 1}", "statement": ck.prose(c.get("statement"), "claim"),
                       "supporting": sup, "contradicting": con, "papers_for": p_for, "papers_against": p_against,
                       "strength": _strength(len(p_for), len(p_against)),
                       "why_they_differ": ck.prose(c.get("why_they_differ"), "claim") if con else ""})
    return {"themes": themes, "claims": claims}


def _median(xs: list[int]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if xs else 0


def check_evolution(raw: dict, ck: _Checker) -> dict:
    phases = []
    for ph in raw.get("phases") or []:
        try:
            lo, hi = int(ph.get("from_year")), int(ph.get("to_year"))
        except (TypeError, ValueError):
            ck.dropped.append(f"phase '{_clean(ph.get('label'), 60)}' dropped: no years")
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        inside = []
        for p in ck.papers(ph.get("papers"), f"phase {lo}-{hi}"):
            if lo <= (ck.years[p] or 0) <= hi:
                inside.append(p)
            else:
                ck.dropped.append(f"phase {lo}-{hi}: {p} ({ck.years[p]}) moved out, published outside the phase")
        ev = [x for x in ck.ids(ph.get("evidence"), f"phase {lo}-{hi}")
              if lo <= (ck.years[ck.passages[x]["paper_id"]] or 0) <= hi]
        inside = list(dict.fromkeys(inside + ck.papers_of(ev)))
        if not inside:
            ck.dropped.append(f"phase {lo}-{hi} dropped: none of its studies was published in those years")
            continue
        phases.append({"label": _clean(ph.get("label"), 120), "from_year": lo, "to_year": hi,
                       "description": ck.prose(ph.get("description"), f"phase {lo}-{hi}"),
                       "papers": inside, "evidence": ev,
                       "studies_read_in_full": sum(1 for y in ck.years.values() if lo <= (y or 0) <= hi)})
    phases.sort(key=lambda p: p["from_year"])
    shifts = []
    for s in raw.get("shifts") or []:
        name = _clean(s.get("shift"), 160)
        before, after = ck.papers(s.get("before"), f"shift '{name}'"), ck.papers(s.get("after"), f"shift '{name}'")
        if not before or not after:
            ck.dropped.append(f"shift '{name}' dropped: it needs studies on both sides")
            continue
        mb, ma = _median([ck.years[p] or 0 for p in before]), _median([ck.years[p] or 0 for p in after])
        if ma <= mb:
            ck.dropped.append(f"shift '{name}' dropped: its 'after' studies (median {ma:g}) are not later than "
                              f"its 'before' studies (median {mb:g})")
            continue
        shifts.append({"shift": name, "description": ck.prose(s.get("description"), f"shift '{name}'"),
                       "before": before, "after": after, "evidence": ck.ids(s.get("evidence"), f"shift '{name}'"),
                       "years": [int(mb), int(ma)]})
    cont = [{"statement": ck.prose(c.get("statement"), "continuity"), "evidence": ev}
            for c in raw.get("continuities") or [] for ev in [ck.ids(c.get("evidence"), "continuity")] if ev]
    return {"overview": ck.prose(raw.get("overview"), "overview"), "phases": phases, "shifts": shifts,
            "continuities": cont}


def check_gaps(raw: dict, ck: _Checker) -> dict:
    gaps, renumber = [], {}
    for k, g in enumerate(raw.get("gaps") or [], 1):
        ev = ck.ids(g.get("evidence"), f"gap {k}")
        if not ev:
            ck.dropped.append(f"gap '{_clean(g.get('gap'), 80)}' dropped: no verified passage from the studies")
            continue
        gid = f"CG{len(gaps) + 1}"
        renumber[f"CG{k}"] = gid
        papers = ck.papers_of(ev)
        kinds = sorted({KIND_NAMES[ck.passages[x]["kind"]] for x in ev})
        gaps.append({"id": gid, "gap": ck.prose(g.get("gap"), gid), "kind": g.get("kind") or "unanswered_question",
                     "explanation": ck.prose(g.get("explanation"), gid), "evidence": ev, "papers": papers,
                     "raised_in": len(papers), "grounded_in": kinds})
    dirs = []
    for d in raw.get("directions") or []:
        addr = [renumber[a] for a in (str(x).strip() for x in d.get("addresses") or []) if a in renumber]
        if not addr:
            ck.dropped.append(f"direction '{_clean(d.get('direction'), 80)}' dropped: it addresses no gap that "
                              "survived the checks")
            continue
        dirs.append({"id": f"CD{len(dirs) + 1}", "direction": ck.prose(d.get("direction"), "direction"),
                     "rationale": ck.prose(d.get("rationale"), "direction"),
                     "design_sketch": ck.prose(d.get("design_sketch"), "direction"), "addresses": addr,
                     "evidence": ck.ids(d.get("evidence"), "direction")})
    return {"gaps": gaps, "directions": dirs}


def _brief_of(part: dict) -> str:
    return json.dumps(part, ensure_ascii=False)[:12000]


def analyse(ctx) -> dict | None:
    """Run (or reuse) the content analysis for this run. Returns the saved note, or None without full texts."""
    papers = fulltext_papers(ctx)
    ids = [r["paper_id"] for r in papers]
    note = ctx.notes().get(NOTE) or {}
    if note.get("papers") == ids and note.get("version") == VERSION:
        return note
    if not ids:
        return None
    cards, failed = read_cards(ctx, ids)
    papers = [r for r in papers if r["paper_id"] in cards]
    if not papers:
        return None
    ctx.emit("content_synthesis", "start", {"task": f"comparing {len(papers)} studies read in full"})
    text = digest(papers, cards)
    ck = _Checker(papers, cards)
    user = f"Studies read in full ({len(papers)}), oldest first:\n\n{text}"
    findings = check_findings(_call(ctx, FINDINGS_SYSTEM.format(question=ctx.question), FINDINGS_TOOL, user), ck)
    evolution = check_evolution(_call(ctx, EVOLUTION_SYSTEM.format(question=ctx.question), EVOLUTION_TOOL, user), ck)
    already = (f"\n\nFindings already written:\n{_brief_of(findings)}\n\nMethod evolution already written:\n"
               f"{_brief_of({k: evolution[k] for k in ('overview', 'shifts', 'continuities')})}")
    gd = check_gaps(_call(ctx, GAPS_SYSTEM.format(question=ctx.question), GAPS_TOOL, user + already), ck)
    years = [r["year"] for r in papers if r["year"]]
    out = {"version": VERSION, "papers": ids, "analysed": [r["paper_id"] for r in papers],
           "years": [min(years), max(years)] if years else None, "failed": failed,
           "passages_dropped": sum(c.get("dropped_without_passage", 0) for c in cards.values()),
           **findings, "evolution": evolution, **gd, "dropped": ck.dropped[:60]}
    ctx.save_note(NOTE, out)
    ctx.emit("content_synthesis", "finish", {"output": {"studies": len(papers), "claims": len(findings["claims"]),
                                                         "phases": len(evolution["phases"]),
                                                         "gaps": len(gd["gaps"]), "directions": len(gd["directions"])}})
    return out


def ensure(ctx) -> dict | None:
    """analyse(), but never at the cost of the run."""
    if not settings.content_analysis:
        return None
    try:
        return analyse(ctx)
    except Exception as exc:
        ctx.emit("content_analysis", "error", {"error": str(exc)[:300]})
        return None


def of(ctx) -> dict:
    return ctx.notes().get(NOTE) or {}


def counts(ctx) -> set[tuple[int, int]]:
    """'n of N studies read in full' pairs that code counted here, for the report's number audit."""
    a = of(ctx)
    n = len(a.get("analysed") or [])
    if not n:
        return set()
    pairs = {(n, n)}
    for c in a.get("claims") or []:
        pairs |= {(len(c["papers_for"]), n), (len(c["papers_against"]), n)}
    for g in a.get("gaps") or []:
        pairs.add((g["raised_in"], n))
    for ph in (a.get("evolution") or {}).get("phases") or []:
        pairs.add((ph["studies_read_in_full"], n))
    return pairs


# ---------------------------------------------------------------- what agents and the report see
def compact(ctx, quotes: bool = False) -> dict:
    a = of(ctx)
    if not a.get("analysed"):
        return {"note": "No content analysis: no paper was read in full in this run."}
    out = {"studies_read_in_full": len(a["analysed"]), "years": a.get("years"),
           "themes": [{"theme": t["theme"], "synthesis": t["synthesis"], "papers": t["papers"]} for t in a["themes"]],
           "content_claims": [{k: c[k] for k in ("id", "statement", "strength", "papers_for", "papers_against",
                                                   "why_they_differ")} for c in a["claims"]],
           "method_evolution": {"overview": a["evolution"]["overview"],
                                "phases": [{k: p[k] for k in ("label", "from_year", "to_year", "description", "papers",
                                                              "studies_read_in_full")} for p in a["evolution"]["phases"]],
                                "shifts": [{k: s[k] for k in ("shift", "description", "before", "after", "years")}
                                           for s in a["evolution"]["shifts"]],
                                "continuities": [c["statement"] for c in a["evolution"]["continuities"]]},
           "content_gaps": [{k: g[k] for k in ("id", "gap", "kind", "explanation", "papers", "raised_in")} for g in a["gaps"]],
           "content_directions": [{k: d[k] for k in ("id", "direction", "rationale", "design_sketch", "addresses")}
                                  for d in a["directions"]],
           "note": "Read from the full texts and checked by code: every item rests on passages found in the papers. "
                   "Counts here (strength, raised_in, studies_read_in_full) are code's counts over the studies "
                   "read in full."}
    if quotes:
        pas = passages_for(ctx)
        wanted = [i for c in a["claims"] for i in c["supporting"][:2]] + [i for g in a["gaps"] for i in g["evidence"][:2]]
        out["passages"] = {k: pas[k]["quote"] for k in dict.fromkeys(wanted) if k in pas}
    return out


def passages_for(ctx) -> dict[str, dict]:
    a = of(ctx)
    ids = a.get("analysed") or []
    if not ids:
        return {}
    cards = {r["paper_id"]: r["data"] for r in ctx.pg.execute(
        "SELECT paper_id, data FROM paper_analyses WHERE version=%s AND paper_id = ANY(%s)", (VERSION, ids)).fetchall()}
    return passages(cards)


def get_content_analysis(ctx, with_quotes: bool = False) -> dict:
    return compact(ctx, quotes=with_quotes)


CONTENT_TOOL = Tool(
    "get_content_analysis",
    "What the studies read in full say, from close reading checked by code: themes of their findings, content "
    "claims with how many studies support or contradict each, how research methods changed over time (phases "
    "and shifts, with the reasons authors give), and the gaps and directions the papers themselves point to. "
    "Use it as the substance of trends, gaps and directions; counts size it.",
    obj({"with_quotes": {"type": "boolean", "description": "Also return the passages behind claims and gaps"}}),
    get_content_analysis, max_chars=24000, read_only=True)


# ---------------------------------------------------------------- rendering
def _q(pas: dict, ids: list[str], k: int = 2) -> list[str]:
    out = []
    for x in ids[:k]:
        p = pas.get(x)
        if p:
            out.append(f'  - {cite(p["paper_id"])} ({KIND_NAMES[p["kind"]]}): "{p["quote"]}"')
    return out


def _cites(pids: list[str], k: int = 8) -> str:
    return ", ".join(cite(p) for p in pids[:k]) + (" and others" if len(pids) > k else "")


def markdown(ctx, heading_level: int = 2) -> list[str]:
    a = of(ctx)
    if not a.get("analysed"):
        return []
    h, hh = "#" * heading_level, "#" * (heading_level + 1)
    pas = passages_for(ctx)
    n = len(a["analysed"])
    y = a.get("years") or ["", ""]
    L = [f"{h} In-depth analysis of the {n} studies read in full", "",
         f"Each study published {y[0]} to {y[1]} that was read in full was read closely for its design, data, "
         "analysis, findings, limitations and the further work its authors call for. Everything below rests on "
         "passages that code found word for word in the papers (a few are quoted under each item); how many "
         "studies support or contradict a statement is counted by code from those passages.", ""]
    ev = a.get("evolution") or {}
    L += [f"{hh} How the research methods have changed", ""]
    if ev.get("overview"):
        L += [ev["overview"], ""]
    for ph in ev.get("phases") or []:
        L += [f"**{ph['label']} ({ph['from_year']} to {ph['to_year']}; {ph['studies_read_in_full']} of the {n} "
              f"studies).** {ph['description']}", "", f"Studies: {_cites(ph['papers'])}."]
        L += _q(pas, ph["evidence"]) + [""]
    if ev.get("shifts"):
        L += ["**Shifts between phases**", ""]
        for s in ev["shifts"]:
            L += [f"- **{s['shift']}** (older practice around {s['years'][0]}: {_cites(s['before'], 4)}; newer "
                  f"around {s['years'][1]}: {_cites(s['after'], 4)}). {s['description']}"]
            L += _q(pas, s["evidence"])
        L += [""]
    if ev.get("continuities"):
        L += ["**What has not changed:** " + " ".join(c["statement"] for c in ev["continuities"]), ""]

    L += [f"{hh} What the studies found", ""]
    for t in a.get("themes") or []:
        L += [f"**{t['theme']}.** {t['synthesis']}", ""]
    if a.get("claims"):
        L += ["**Claims from the studies' content**", ""]
        for c in a["claims"]:
            line = f"- **{c['id']}** {c['statement']} *({c['strength']}: {_cites(c['papers_for'], 6)}"
            if c["papers_against"]:
                line += f"; against: {_cites(c['papers_against'], 6)}"
            line += ")*"
            if c.get("why_they_differ"):
                line += f" Why they differ: {c['why_they_differ']}"
            L += [line] + _q(pas, c["supporting"]) + _q(pas, c["contradicting"], 1)
        L += [""]

    if a.get("gaps"):
        L += [f"{hh} Gaps the studies themselves point to", ""]
        for g in a["gaps"]:
            L += [f"- **{g['id']} · {g['gap']}** ({g['kind'].replace('_', ' ')}; raised in {g['raised_in']} of the "
                  f"{n} studies, from their {', '.join(g['grounded_in'])}). {g['explanation']}"]
            L += _q(pas, g["evidence"])
        L += [""]
    if a.get("directions"):
        L += [f"{hh} Research directions", ""]
        for d in a["directions"]:
            L += [f"- **{d['id']} · {d['direction']}** (addresses {', '.join(d['addresses'])}). {d['rationale']}"
                  + (f" *Possible design:* {d['design_sketch']}" if d.get("design_sketch") else "")]
            L += _q(pas, d["evidence"], 1)
        L += [""]
    notes = []
    if a.get("passages_dropped"):
        notes.append(f"{a['passages_dropped']} passages the reader gave could not be found in the papers and were "
                     "discarded with the statements they carried")
    if a.get("failed"):
        notes.append(f"{len(a['failed'])} studies could not be read closely")
    if a.get("dropped"):
        notes.append(f"{len(a['dropped'])} references in the synthesis were removed or corrected by the checks "
                     "(passages that do not exist, papers not read in full, studies outside a phase's years, "
                     "shifts that ran backwards in time)")
    if notes:
        L += ["*Checks:* " + "; ".join(notes) + ".", ""]
    return L
