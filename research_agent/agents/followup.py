"""Follow-up questions about a finished run.

A researcher asks about a claim, a gap, a paper or a number, and the Follow-up agent answers from the run's
own material: its papers, extractions, verified quotes, claims and map. The rules of the report still apply:

- any new number is counted by code first (test_claim / claims that are verified at once);
- the answer is audited like a report: invented paper ids are removed, unsupported claim citations are
  marked, and "n of N" figures no code produced are tagged [unverified];
- claims added here are stored with agent 'followup', so the original report stays as it was.

The conversation is kept per run (table `followups`), so a researcher can come back to it later.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass

from research_agent.agents.base import Agent
from research_agent.agents.report import (_CITE_CLAIM, _cite, _cited_ids, align_claim_citations, allowed_counts,
                                          audit_numbers, normalise_citations, run_facts)
from research_agent.db import connect
from research_agent.runstate import RunContext
from research_agent.tools.base import INT, STR, STRS, Tool, obj
from research_agent.tools.claims import PREDICATE_DOC, TEST_TOOL, propose_claim, verify_claims
from research_agent.tools.extraction import ANALYSIS_TOOLS, _rows, protocol_of
from research_agent.tools.recheck import RECHECK_TOOL, _passages

HISTORY_TURNS = 6          # earlier exchanges given to the agent
_schema_ready = False


def _ensure_schema() -> None:
    """Databases created before follow-ups existed lack the table; add it (once per process)."""
    global _schema_ready
    if not _schema_ready:
        from research_agent.db import init_schema

        init_schema()
        _schema_ready = True
MAP_PREFIXES = ("E", "R", "W", "G", "N", "H", "D")


# ---------------------------------------------------------------- tools over the finished run
def run_overview(ctx) -> dict:
    """The run at a glance: question, facts, the claims with their counts, map items and protocol fields."""
    facts = run_facts(ctx)
    claims = [{"id": f"C{c['id']}", "status": c["status"], "agent": c["agent"], "text": c["text"][:220],
               "counted": _counted(c["result"])}
              for c in ctx.pg.execute("SELECT id, agent, text, status, result FROM claims WHERE run_id=%s ORDER BY id",
                                      (ctx.run_id,)).fetchall()]
    out = {"question": ctx.question, "run_facts": facts, "claims": claims}
    notes = ctx.notes()
    m = (notes.get("map") or {}).get("map")
    if m:
        out["map_items"] = {sec: [f"{it['id']} {it['label']}" for it in m.get(sec, [])]
                            for sec in ("established", "emerging", "watch", "gaps") if m.get(sec)}
        out["map_items"]["novelty"] = [f"{n['id']} {n['a']['label']} + {n['b']['label']}" for n in m.get("novelty", [])]
        out["hypotheses"] = [f"{h['id']} ({h['verdict']}) {h['text'][:160]}"
                             for h in (notes.get("hypotheses") or {}).get("items", [])]
        out["designs"] = [f"{d['id']} {d.get('title')}" for d in (notes.get("design") or {}).get("designs", [])]
    protocol = protocol_of(ctx)
    if protocol:
        out["question_specific_fields"] = {f["name"]: f.get("values") or "free text" for f in protocol["fields"]}
    report = ctx.pg.execute("SELECT report_md FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()["report_md"] or ""
    out["report_opening"] = report[:1500]
    return out


def _counted(result) -> str | None:
    r = result or {}
    if r.get("denominator") is not None:
        return f"{r.get('n_matching')} of {r['denominator']}"
    w = r.get("window") or {}
    if w.get("ratio_late_over_early") is not None:
        return f"x{w['ratio_late_over_early']} ({w.get('early_mean')} -> {w.get('late_mean')})"
    return None


def get_item(ctx, item_id: str) -> dict:
    """Everything behind one item: a claim (C12) with the papers it counted and their quotes, or a map item
    (G2, N1, E3, R1, W1), a tested hypothesis (H4) or a design (D1)."""
    item_id = item_id.strip().upper()
    if item_id.startswith("C") and item_id[1:].isdigit():
        c = ctx.pg.execute("SELECT id, agent, text, claim_type, predicate, status, result, review_note FROM claims "
                           "WHERE run_id=%s AND id=%s", (ctx.run_id, int(item_id[1:]))).fetchone()
        if not c:
            return {"error": f"{item_id} is not a claim of this run"}
        out = {"id": item_id, "text": c["text"], "status": c["status"], "made_by": c["agent"],
               "type": c["claim_type"], "predicate": c["predicate"], "counted": _counted(c["result"]),
               "review_note": c["review_note"]}
        r = c["result"] or {}
        if c["claim_type"] == "prevalence":
            field = (c["predicate"] or {}).get("field")
            rows = {x["paper_id"]: x for x in _rows(ctx)}
            out["papers_counted"] = [
                {"paper_id": pid, "title": rows[pid]["title"], "year": rows[pid]["year"], "read": rows[pid]["source"],
                 "values": rows[pid]["data"].get(field), "quotes": (rows[pid]["data"].get("evidence") or {}).get(field, [])[:2]}
                for pid in (r.get("matched_paper_ids") or [])[:30] if pid in rows]
            out["caveat"] = r.get("caveat")
        return out
    notes = ctx.notes()
    m = (notes.get("map") or {}).get("map") or {}
    for sec in ("established", "emerging", "watch", "gaps", "novelty"):
        for it in m.get(sec, []):
            if it.get("id") == item_id:
                return {"section": sec, **{k: v for k, v in it.items() if k != "_ids"}}
    for h in (notes.get("hypotheses") or {}).get("items", []):
        if h["id"] == item_id:
            return h
    for d in (notes.get("design") or {}).get("designs", []):
        if d["id"] == item_id:
            return d
    return {"error": f"no item {item_id} in this run (claims are C<number>; map items G, N, E, R, W; H; D)"}


def read_paper(ctx, paper_id: str, terms: list[str] | None = None, max_chars: int = 12000) -> dict:
    """One analysed paper: its extracted record with quotes, and its text. With `terms`, only the passages
    that mention them; otherwise the most informative sections (methods, data, results, limitations)."""
    from research_agent.ingestion.fulltext import select_for_reading

    if paper_id not in set(ctx.shortlist_ids()):
        return {"error": f"{paper_id} is not one of the papers analysed in this run"}
    fetch_error = _fetch(ctx, [paper_id])
    p = ctx.pg.execute(
        """SELECT p.paper_id, p.title, p.year, p.abstract, p.source, f.clean_text, f.sections FROM papers p
           LEFT JOIN paper_fulltext f ON f.paper_id = p.paper_id AND f.status='ok' WHERE p.paper_id=%s""",
        (paper_id,)).fetchone()
    record = next((r for r in _rows(ctx) if r["paper_id"] == paper_id), None)
    out = {"paper_id": paper_id, "title": p["title"], "year": p["year"], "corpus": p["source"], "abstract": p["abstract"],
           "extracted": {k: v for k, v in (record or {}).get("data", {}).items() if k != "_unverified"}}
    if p["clean_text"]:
        if terms:
            found = _passages(p["clean_text"], [t for t in terms if t.strip()])
            out["passages"] = found or "none of the terms appear in the full text"
        else:
            out["full_text_sections"] = select_for_reading(p["sections"] or [], budget=max(2000, min(int(max_chars), 20000)))
    else:
        out["full_text"] = ("could not be retrieved right now (" + fetch_error + "); answer from the abstract"
                            if fetch_error else "not available for this paper; answer from the abstract and "
                            "extracted record")
    return out


def _fetch(ctx, paper_ids: list[str]) -> str | None:
    """Download full texts not yet stored (arXiv from the local files, PMC from NCBI). A network problem
    must not break a follow-up: it is reported, and the papers that did arrive are used."""
    from research_agent.ingestion.fulltext import fetch_fulltext

    try:
        fetch_fulltext(ctx.pg, paper_ids)
        return None
    except Exception as exc:
        return f"full-text download failed: {str(exc)[:160]}"


MAX_READ_IN_FULL = 20      # papers read in full per request (each is one careful model read)


def read_in_full(ctx, claim_id: str, limit: int = 12) -> dict:
    """Check a count against full texts: read the claim's abstract-only papers in full (papers the count
    missed first, since that is where under-counting hides), update their records, and count again.
    The original claim is left as it was; the new count is recorded as a follow-up claim."""
    from research_agent.config import settings
    from research_agent.ingestion.fulltext import select_for_reading
    from research_agent.tools import claims as C
    from research_agent.tools.extraction import known_fields
    from research_agent.tools.reading import read_papers

    cid = int(str(claim_id).upper().lstrip("C"))
    c = ctx.pg.execute("SELECT id, text, claim_type, predicate, status, result FROM claims WHERE run_id=%s AND id=%s",
                       (ctx.run_id, cid)).fetchone()
    if not c:
        return {"error": f"C{cid} is not a claim of this run"}
    if c["claim_type"] != "prevalence":
        return {"error": "only counts over the analysed papers can be re-checked with full texts (trends are "
                         "measured over the whole corpus)"}
    p = c["predicate"]
    _, enums = known_fields(ctx)
    where = p.get("where") or []
    where = [where] if isinstance(where, dict) else where
    population = [r for r in C._rows(ctx) if all(C.row_matches(r, w, enums) for w in where)]
    needles = [C._norm(x) for x in p["any_of"]]
    exact = p["field"] in enums
    abstract_only = [r for r in population if r["source"] == "abstract"]
    missed = [r for r in abstract_only if not C._hits(r, p["field"], needles, exact)]
    counted = [r for r in abstract_only if r not in missed]
    todo = (missed + counted)[:max(1, min(int(limit), MAX_READ_IN_FULL))]
    if not todo:
        return {"claim": f"C{cid}", "note": "every paper this claim counts over was already read in full",
                "counted": _counted(c["result"])}

    ids = [r["paper_id"] for r in todo]
    fetch_error = _fetch(ctx, ids)
    texts = {r["paper_id"]: select_for_reading(r["sections"] or [], budget=settings.fulltext_read_chars)
             for r in ctx.pg.execute("SELECT paper_id, sections FROM paper_fulltext WHERE paper_id = ANY(%s) "
                                     "AND status='ok'", (ids,)).fetchall()}
    papers = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, title, abstract FROM papers WHERE paper_id = ANY(%s)", (ids,)).fetchall()}
    items = [{**papers[pid], "fulltext": texts[pid]} for pid in ids if texts.get(pid)]
    before = {r["paper_id"]: bool(C._hits(r, p["field"], needles, exact)) for r in todo}
    results, failed, model = read_papers(ctx, items, protocol_of(ctx), include_base=True, step="followup_reading")
    for pid, res in results.items():
        ctx.pg.execute(
            "INSERT INTO extractions (paper_id, schema_version, source, data, model) VALUES (%s,%s,%s,%s::jsonb,%s) "
            "ON CONFLICT (paper_id, schema_version) DO UPDATE SET source=EXCLUDED.source, data=EXCLUDED.data, "
            "model=EXCLUDED.model, created_at=now()",
            (pid, ctx.extraction_version, res["source"], json.dumps(res["base"]), model))
        if res.get("protocol") is not None:
            ctx.pg.execute(
                "INSERT INTO protocol_extractions (run_id, paper_id, source, data) VALUES (%s,%s,%s,%s::jsonb) "
                "ON CONFLICT (run_id, paper_id) DO UPDATE SET source=EXCLUDED.source, data=EXCLUDED.data",
                (ctx.run_id, pid, res["source"], json.dumps(res["protocol"])))
    after_rows = {r["paper_id"]: r for r in C._rows(ctx)}
    changed = [{"paper_id": pid, "title": papers[pid]["title"],
                "now_counted": bool(C._hits(after_rows[pid], p["field"], needles, exact)),
                "quotes": (after_rows[pid]["data"].get("evidence") or {}).get(p["field"], [])[:2]}
               for pid in results if pid in after_rows
               and bool(C._hits(after_rows[pid], p["field"], needles, exact)) != before[pid]]
    # the re-count is a new claim, so the report's evidence table keeps the original result
    new = ctx.pg.execute(
        "INSERT INTO claims (run_id, agent, text, claim_type, predicate) VALUES (%s,'followup',%s,'prevalence',%s::jsonb) "
        "RETURNING id", (ctx.run_id, f"C{cid} re-counted after {len(results)} more of its papers were read in full.",
                         json.dumps(p))).fetchone()["id"]
    verify_claims(ctx, [new])
    row = ctx.pg.execute("SELECT status, result FROM claims WHERE id=%s", (new,)).fetchone()
    return {"claim": f"C{cid}", "before": _counted(c["result"]), "after": _counted(row["result"]),
            "recount_claim": f"C{new}", "recount_status": row["status"],
            "read_in_full_now": len(results), "no_full_text_available": [pid for pid in ids if not texts.get(pid)],
            "failed": failed, "papers_that_changed": changed,
            **({"fetch_problem": fetch_error} if fetch_error else {}),
            "still_abstract_only": max(0, len(abstract_only) - len(results)),
            "note": "Records of papers read in full are updated for this run and later runs."}


def add_claim(ctx, text: str, claim_type: str, predicate: dict) -> dict:
    """Propose a claim and verify it at once, so the answer can cite it straight away."""
    res = propose_claim(ctx, text, claim_type, predicate, _agent="followup")
    if res.get("claim_id") and res.get("status") == "pending":
        verify_claims(ctx, [res["claim_id"]])
        row = ctx.pg.execute("SELECT status, result FROM claims WHERE id=%s", (res["claim_id"],)).fetchone()
        res.update(status=row["status"], counted=_counted(row["result"]))
        res.pop("note", None)
    return res


OVERVIEW_TOOL = Tool("run_overview", "The run at a glance: its question, facts, claims (with counts), map items, "
                     "hypotheses, designs and question-specific fields.", obj({}), run_overview, read_only=True,
                     max_chars=16000)
ITEM_TOOL = Tool("get_item", "Everything behind one item: a claim (C12) with the papers it counted and their "
                 "quotes, a map item (G2, N1, E3, R1, W1), a hypothesis (H4) or a design (D1).",
                 obj({"item_id": STR}, ["item_id"]), get_item, read_only=True, max_chars=16000)
READ_TOOL = Tool("read_paper", "Read one analysed paper: its extracted record with quotes and its text. Give "
                 "terms to get only the passages that mention them.",
                 obj({"paper_id": STR, "terms": STRS, "max_chars": INT}, ["paper_id"]), read_paper, read_only=True,
                 max_chars=16000)
READ_IN_FULL_TOOL = Tool(
    "read_in_full", "Check a claim against full texts: read its abstract-only papers in full (papers the count "
    "missed first), update their records, and count again. Returns the count before and after, the papers whose "
    "status changed with quotes, and a new claim id for the re-count. Costs one careful read per paper; "
    f"at most {MAX_READ_IN_FULL} per call.",
    obj({"claim_id": STR, "limit": INT}, ["claim_id"]), read_in_full)
ADD_CLAIM_TOOL = Tool("add_claim", "Count something new and record it as a claim, verified immediately. Use "
                      "test_claim first, then write the text from the measured numbers. Predicate formats: "
                      + PREDICATE_DOC + ".",
                      obj({"text": STR, "claim_type": {"type": "string", "enum": ["prevalence", "trend"]},
                           "predicate": {"type": "object"}}, ["text", "claim_type", "predicate"]), add_claim)


@dataclass
class FollowupAgent(Agent):
    """Its context is the conversation so far, not the full shared state of a run in progress."""
    history: str = ""

    def context_message(self, ctx, task: str) -> str:
        return (f"The run's research question: {ctx.question}\n\n"
                + (f"Conversation so far:\n{self.history}\n\n" if self.history else "")
                + f"New question from the researcher:\n{task}")


FOLLOWUP_SYSTEM = """You are the Follow-up agent. A research run has finished; the researcher is asking about
its results: a claim, a gap, a paper, a number or a what-if. Answer from the run's own material, using tools.

How to work:
- Start with get_item for any id the question names (C12, G2, N1, H4, D1), or run_overview to orient.
- To show which papers are behind a count, use get_item on the claim; to check what a paper says, read_paper
  (with terms, to jump to the passages). read_paper also opens the full text of papers the run only read from
  their abstract.
- When a researcher doubts a count, or many of its papers were read from abstracts only ('read': 'abstract'
  in get_item), use read_in_full on the claim: it reads those papers in full and counts again. Report the
  count before and after, cite the re-count claim, and name papers whose status changed with their quotes.
- For a new count or a what-if ("does it hold in PMC only?", "what if LSTMs count as deep learning?"), use
  test_claim, then add_claim with text written from the measured numbers, and cite it. Use `where` for
  subgroups. Before calling something rare, run recheck_field.
- Every "n of N" you write must come from a claim or a tool result. Cite claims as [C12] and papers as
  [PMC123] or [arXiv:2401.00001], only ids the tools gave you. Quote papers word for word when you quote.
- Say plainly when something cannot be answered from this run's papers, and what would answer it.

Write the answer in clear, measured prose: a direct answer first, then the evidence. Keep it as long as the
question needs and no longer. Do not use em dashes."""

FOLLOWUP = FollowupAgent(
    name="followup",
    role="Answers a researcher's follow-up questions about a finished run.",
    system=FOLLOWUP_SYSTEM,
    tools=[OVERVIEW_TOOL, ITEM_TOOL, READ_TOOL, READ_IN_FULL_TOOL, TEST_TOOL, ADD_CLAIM_TOOL, RECHECK_TOOL]
    + [t for t in ANALYSIS_TOOLS if t.name in ("value_counts", "cross_tab", "list_extractions")],
    finish_schema=obj({"answer": {**STR, "description": "The answer in Markdown, with citations"}}, ["answer"]),
    max_turns=12,
)


# ---------------------------------------------------------------- answering, checking, storing
def audit_answer(ctx, text: str) -> tuple[str, dict]:
    """The report's checks, applied to an answer."""
    text = normalise_citations(text or "")
    claims = {c["id"]: c for c in ctx.pg.execute(
        "SELECT id, text, status, claim_type, result FROM claims WHERE run_id=%s ORDER BY id", (ctx.run_id,)).fetchall()}
    text, fixes = align_claim_citations(text, claims)
    shortlist = set(ctx.shortlist_ids())
    bad_papers = sorted({pid for pid in _cited_ids(text) if pid not in shortlist})
    for pid in bad_papers:
        text = text.replace(_cite(pid), "[citation removed: not in analysed set]")
    bad_claims = sorted({int(c) for c in _CITE_CLAIM.findall(text) if int(c) not in claims
                         or claims[int(c)]["status"] != "supported"})
    for cid in bad_claims:
        text = text.replace(f"[C{cid}]", f"[C{cid}, not verified]")
    text, unverified = audit_numbers(text, allowed_counts(ctx, claims))
    return text, {"removed_paper_citations": bad_papers, "flagged_claim_citations": bad_claims,
                  "unverified_numbers": unverified, "claim_citation_fixes": fixes}


def history(run_id: str, limit: int = 50) -> list[dict]:
    _ensure_schema()
    conn = connect()
    try:
        return conn.execute("SELECT id, role, content, status, meta, ts FROM followups WHERE run_id=%s "
                            "ORDER BY id DESC LIMIT %s", (run_id, limit)).fetchall()[::-1]
    finally:
        conn.close()


def _history_text(rows: list[dict]) -> str:
    done = [r for r in rows if r["status"] == "done" and r["content"]]
    lines = []
    for r in done[-2 * HISTORY_TURNS:]:
        who = "Researcher" if r["role"] == "user" else "You"
        lines.append(f"{who}: {r['content'][:1500]}")
    return "\n\n".join(lines)


def _question_text(question: str, focus: str | None) -> str:
    return f"(About {focus}) {question}" if focus else question


def start(run_id: str, question: str, focus: str | None = None) -> tuple[int, int]:
    """Store the question and a pending answer; returns (question id, answer id)."""
    _ensure_schema()
    conn = connect()
    try:
        if not conn.execute("SELECT 1 FROM runs WHERE run_id=%s", (run_id,)).fetchone():
            raise ValueError(f"No run with id {run_id}")
        q = conn.execute("INSERT INTO followups (run_id, role, content, meta) VALUES (%s,'user',%s,%s::jsonb) "
                         "RETURNING id", (run_id, question, json.dumps({"focus": focus}))).fetchone()["id"]
        a = conn.execute("INSERT INTO followups (run_id, role, status) VALUES (%s,'assistant','pending') RETURNING id",
                         (run_id,)).fetchone()["id"]
        return q, a
    finally:
        conn.close()


def answer(run_id: str, question: str, focus: str | None = None, provider: str | None = None, llm_factory=None,
           answer_id: int | None = None) -> dict:
    """Answer one follow-up question and store the exchange. Never changes the run's report."""
    _ensure_schema()
    if answer_id is None:
        _, answer_id = start(run_id, question, focus)
    ctx = RunContext.attach(run_id, provider=provider, llm_factory=llm_factory)
    try:
        before = {r["id"] for r in ctx.pg.execute("SELECT id FROM claims WHERE run_id=%s", (run_id,)).fetchall()}
        q_id = ctx.pg.execute("SELECT max(id) AS q FROM followups WHERE run_id=%s AND role='user' AND id < %s",
                              (run_id, answer_id)).fetchone()["q"] or answer_id
        rows = [r for r in history(run_id) if r["id"] < q_id]      # the conversation before this question
        agent = FollowupAgent(**{k: getattr(FOLLOWUP, k) for k in
                                 ("name", "role", "system", "tools", "finish_schema", "max_turns")},
                              history=_history_text(rows))
        out = agent.run(ctx, _question_text(question, focus))
        text, audit = audit_answer(ctx, out.get("answer") or out.get("summary") or "")
        if not text.strip():
            text = "_No answer was produced. Try asking again, or narrow the question._"
        new_claims = sorted(r["id"] for r in ctx.pg.execute("SELECT id FROM claims WHERE run_id=%s",
                                                                (run_id,)).fetchall() if r["id"] not in before)
        meta = {"audit": audit, "new_claims": [f"C{c}" for c in new_claims], "focus": focus}
        ctx.pg.execute("UPDATE followups SET content=%s, status='done', meta=%s::jsonb WHERE id=%s",
                       (text, json.dumps(meta), answer_id))
        return {"id": answer_id, "answer": text, **meta}
    except Exception as exc:
        ctx.pg.execute("UPDATE followups SET content=%s, status='failed' WHERE id=%s",
                       (f"The follow-up failed: {str(exc)[:300]}", answer_id))
        raise
    finally:
        ctx.close()


def answer_in_background(run_id: str, question: str, focus: str | None = None, provider: str | None = None) -> dict:
    q, a = start(run_id, question, focus)

    def work():
        try:
            answer(run_id, question, focus, provider=provider, answer_id=a)
        except Exception:
            pass   # the failure is stored on the answer row
    threading.Thread(target=work, daemon=True).start()
    return {"question_id": q, "answer_id": a}
