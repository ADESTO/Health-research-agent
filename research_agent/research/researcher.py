"""The open-ended researcher.

You give it a charter (a goal, a scope, what is out of bounds) and a finished run or map to work on. It keeps an
agenda of questions and a notebook, and works in cycles, one question per cycle, until its budget or agenda is
spent. It chooses its own questions and patterns; code decides what counts as a finding.

Safeguards against weak judgment (see gauntlet.py): patterns are specs code counts; the researcher sees only the
discovery half of the studies; every proposed finding runs subgroup tests, the trap library, a critic's rival
explanations and a held-out confirmation at a threshold that tightens with every proposal; the grade is computed.

Safeguards against drift:
- the charter is re-read at the start of every cycle, and every cycle works on one agenda question;
- new questions must hang under an existing one (after the first cycle), duplicates are refused, and a question
  far from the charter (by embedding similarity) or touching what is out of bounds waits for your approval;
- the notebook remembers every pattern tested: testing it again returns the earlier result instead of a new
  test (and does not add to the multiple-testing count);
- each question gets a few cycles; one that stalls twice or uses its cycles is parked with the reason;
- a supervisor reviews the agenda against the charter every few cycles: prunes, merges, re-prioritises and
  says whether the work is still worth its cost;
- token budgets per day and in total, a cycle limit, and pause / stop at any time.
"""
from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import dataclass

from research_agent.agents.base import Agent
from research_agent.config import settings
from research_agent.db import connect
from research_agent.research import gauntlet as GT
from research_agent.research import patterns as P
from research_agent.runstate import RunContext
from research_agent.tools.base import INT, STR, STRS, Tool, obj

MAX_OPEN_QUESTIONS = 12


# ---------------------------------------------------------------- creating and managing
def _schema() -> None:
    from research_agent.agents.followup import _ensure_schema

    _ensure_schema()


def create(run_id: str, goal: str, scope: str = "", out_of_bounds: list[str] | None = None, success: str = "",
           max_cycles: int = 20, daily_tokens: int = 2_000_000, total_tokens: int = 10_000_000,
           split: str = "random", split_year: int | None = None, provider: str | None = None) -> int:
    _schema()
    if not (goal or "").strip():
        raise ValueError("give the researcher a goal")
    conn = connect()
    try:
        run = conn.execute("SELECT status FROM runs WHERE run_id=%s", (run_id,)).fetchone()
        if not run:
            raise ValueError(f"No run with id {run_id}")
        charter = {"goal": goal.strip(), "scope": (scope or "").strip(),
                   "out_of_bounds": [x.strip() for x in out_of_bounds or [] if x.strip()],
                   "success": (success or "").strip() or
                   "Patterns that survive the held-out test and matter for the goal, with their evidence trail."}
        spl = {"method": split if split in ("random", "year") else "random", "seed": uuid.uuid4().hex[:12],
               "year": split_year}
        return conn.execute(
            "INSERT INTO researchers (run_id, charter, split, max_cycles, daily_tokens, total_tokens, provider) "
            "VALUES (%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s) RETURNING id",
            (run_id, json.dumps(charter), json.dumps(spl), max_cycles, daily_tokens, total_tokens, provider)
        ).fetchone()["id"]
    finally:
        conn.close()


def schedule(rid: int, delay: int = 0) -> int | None:
    """Queue the next cycle, unless one is already waiting. (A running one is usually the cycle asking: it
    finishes right after, and one worker claims a researcher's jobs one at a time.)"""
    from research_agent.jobs import enqueue

    conn = connect()
    try:
        busy = conn.execute("SELECT id FROM jobs WHERE kind='research_cycle' AND payload->>'researcher_id' = %s "
                            "AND status = 'queued'", (str(rid),)).fetchone()
    finally:
        conn.close()
    return None if busy else enqueue("research_cycle", {"researcher_id": rid}, max_attempts=2, delay_seconds=delay)


def start_in_background(run_id: str, goal: str, **kw) -> dict:
    rid = create(run_id, goal, **kw)
    return {"researcher_id": rid, "job_id": schedule(rid)}


def set_status(rid: int, status: str, note: str | None = None) -> None:
    conn = connect()
    try:
        conn.execute("UPDATE researchers SET status=%s, status_note=%s, updated_at=now() WHERE id=%s",
                     (status, note, rid))
        conn.execute("INSERT INTO research_notebook (researcher_id, kind, content) VALUES (%s,'status',%s::jsonb)",
                     (rid, json.dumps({"status": status, "note": note})))
    finally:
        conn.close()
    if status == "active":
        schedule(rid)


def decide_question(rid: int, agenda_id: int, approve: bool) -> None:
    conn = connect()
    try:
        conn.execute("UPDATE research_agenda SET status=%s, status_note=%s, updated_at=now() WHERE id=%s AND "
                     "researcher_id=%s AND status='needs_approval'",
                     ("open" if approve else "pruned", "approved by you" if approve else "declined by you", agenda_id, rid))
        waiting = conn.execute("SELECT status, status_note FROM researchers WHERE id=%s", (rid,)).fetchone()
    finally:
        conn.close()
    if approve and waiting and waiting["status"] == "paused" and "approval" in (waiting["status_note"] or ""):
        set_status(rid, "active", "resumed after your approval")


def get(rid: int) -> dict | None:
    _schema()
    conn = connect()
    try:
        r = conn.execute("SELECT * FROM researchers WHERE id=%s", (rid,)).fetchone()
        if not r:
            return None
        agenda = conn.execute("SELECT id, parent_id, question, why, status, status_note, priority, cycles_used, "
                              "tokens_used, scope_score FROM research_agenda WHERE researcher_id=%s ORDER BY id",
                              (rid,)).fetchall()
        findings = conn.execute("SELECT id, agenda_id, test_id, statement, grade, status, reasons, evidence, created_at "
                                "FROM research_findings WHERE researcher_id=%s ORDER BY id DESC", (rid,)).fetchall()
        notebook = conn.execute("SELECT id, agenda_id, cycle, kind, content, ts FROM research_notebook WHERE "
                                "researcher_id=%s AND kind <> 'usage' ORDER BY id DESC LIMIT 80", (rid,)).fetchall()
        tests = conn.execute("SELECT count(*) n FROM research_tests WHERE researcher_id=%s", (rid,)).fetchone()["n"]
        today = _tokens_today(conn, rid)
    finally:
        conn.close()
    return {"researcher": r, "agenda": agenda, "findings": findings, "notebook": notebook, "tests": tests,
            "tokens_today": today}


def list_for_run(run_id: str) -> list[dict]:
    _schema()
    conn = connect()
    try:
        return conn.execute(
            """SELECT r.id, r.charter, r.status, r.status_note, r.cycles_done, r.max_cycles, r.tokens_used,
                      r.total_tokens, r.created_at,
                      (SELECT count(*) FROM research_findings f WHERE f.researcher_id = r.id AND f.status='confirmed') confirmed
               FROM researchers r WHERE r.run_id=%s ORDER BY r.id DESC""", (run_id,)).fetchall()
    finally:
        conn.close()


def _tokens_today(conn, rid: int) -> int:
    return int(conn.execute("SELECT coalesce(sum((content->>'tokens')::bigint), 0) n FROM research_notebook "
                            "WHERE researcher_id=%s AND kind='usage' AND ts > now() - interval '1 day'",
                            (rid,)).fetchone()["n"])


# ---------------------------------------------------------------- the working context
@dataclass
class Workspace:
    id: int
    charter: dict
    disc: list[dict]
    hold: list[dict]
    lists: list[str]
    enums: dict
    cycle: int
    agenda_id: int | None = None
    seeding: bool = False

    @property
    def disc_ids(self) -> set[str]:
        return {r["paper_id"] for r in self.disc}


def _open(rid: int, llm_factory=None, provider=None) -> tuple[RunContext, dict]:
    from research_agent.tools.extraction import _rows, known_fields

    conn = connect()
    try:
        r = conn.execute("SELECT * FROM researchers WHERE id=%s", (rid,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise ValueError(f"No researcher {rid}")
    ctx = RunContext.attach(str(r["run_id"]), provider=provider or r["provider"], llm_factory=llm_factory)
    rows = _rows(ctx)
    disc, hold = P.split_rows(rows, r["split"]["seed"], r["split"].get("method", "random"), r["split"].get("year"))
    lists, enums = known_fields(ctx)
    ctx._research = Workspace(id=rid, charter=r["charter"], disc=disc, hold=hold, lists=lists, enums=enums,
                              cycle=r["cycles_done"] + 1)
    return ctx, r


def _W(ctx) -> Workspace:
    return ctx._research


def _note(ctx, kind: str, content: dict, agenda_id: int | None = None) -> None:
    w = _W(ctx)
    ctx.pg.execute("INSERT INTO research_notebook (researcher_id, agenda_id, cycle, kind, content) VALUES "
                   "(%s,%s,%s,%s,%s::jsonb)", (w.id, agenda_id if agenda_id is not None else w.agenda_id, w.cycle,
                                              kind, json.dumps(content, default=str)))


# ---------------------------------------------------------------- tools (discovery half only)
def research_overview(ctx) -> dict:
    """Charter, the current question, the agenda, what has been tried, findings, budget and the data available."""
    from collections import Counter

    w = _W(ctx)
    agenda = ctx.pg.execute("SELECT id, parent_id, question, status, priority, cycles_used FROM research_agenda "
                            "WHERE researcher_id=%s ORDER BY id", (w.id,)).fetchall()
    tests = ctx.pg.execute("SELECT id, description, result FROM research_tests WHERE researcher_id=%s ORDER BY id DESC "
                           "LIMIT 25", (w.id,)).fetchall()
    findings = ctx.pg.execute("SELECT id, statement, grade FROM research_findings WHERE researcher_id=%s ORDER BY id",
                              (w.id,)).fetchall()
    notes = ctx.pg.execute("SELECT cycle, kind, content FROM research_notebook WHERE researcher_id=%s AND kind IN "
                           "('cycle','note','park','checkpoint') ORDER BY id DESC LIMIT 12", (w.id,)).fetchall()
    r = ctx.pg.execute("SELECT cycles_done, max_cycles, tokens_used, total_tokens FROM researchers WHERE id=%s",
                       (w.id,)).fetchone()
    fields = {}
    for f in w.lists + list(w.enums):
        c = Counter(str(v).lower() for row in w.disc for v in (row["data"].get(f) if isinstance(row["data"].get(f), list)
                                                                 else [row["data"].get(f)]) if v not in (None, "", "not_stated"))
        if c:
            fields[f] = [f"{v} ({n})" for v, n in c.most_common(8)]
    notes_map = ctx.notes()
    m = (notes_map.get("map") or {}).get("map") or {}
    leads = {"gaps": [f"{g['id']} {g['label']}" for g in m.get("gaps", [])][:8],
             "untried_combinations": [f"{n['id']} {n['a']['label']} + {n['b']['label']}" for n in m.get("novelty", [])][:6],
             "untested_hypotheses": [h["text"][:160] for h in (notes_map.get("hypotheses") or {}).get("items", [])
                                     if h.get("verdict") in ("untestable", "not_tested")][:6],
             "reported_gaps": [g.get("gap") for g in (notes_map.get("gaps") or {}).get("gaps") or [] if isinstance(g, dict)][:6]}
    return {"charter": w.charter, "current_question": w.agenda_id,
            "agenda": [{**a, "current": a["id"] == w.agenda_id} for a in agenda],
            "tested_patterns": [{"test_id": t["id"], "pattern": t["description"], "p": t["result"].get("p"),
                                 "direction": t["result"].get("direction"), "testable": t["result"].get("testable")}
                                for t in tests],
            "findings": findings, "recent_notebook": [{"cycle": n["cycle"], "kind": n["kind"], **n["content"]} for n in notes],
            "budget": {"cycles_done": r["cycles_done"], "max_cycles": r["max_cycles"],
                       "tokens_used": r["tokens_used"], "total_tokens": r["total_tokens"]},
            "discovery_half": {"papers": len(w.disc), "note": "You see only this half; the other half is kept for "
                               "confirming findings and is never shown to you."},
            "fields_with_common_values": fields, "leads_from_the_run": {k: v for k, v in leads.items() if v}}


def explore_counts(ctx, field: str, top: int = 20) -> dict:
    from collections import Counter

    w = _W(ctx)
    if field not in w.lists + list(w.enums) + ["corpus", "read", "year"]:
        return {"error": f"field must be one of {w.lists + list(w.enums) + ['corpus', 'read', 'year']}"}
    key = {"read": "source"}.get(field, field)
    c = Counter()
    for r in w.disc:
        v = r.get(key) if field in ("corpus", "read", "year") else r["data"].get(field)
        for x in (v if isinstance(v, list) else [v]):
            if x not in (None, "", "not_stated"):
                c[str(x).lower()] += 1
    return {"field": field, "papers": len(w.disc), "counts": dict(c.most_common(max(1, min(int(top), 50))))}


def explore_cross(ctx, field_a: str, field_b: str, top: int = 8) -> dict:
    from collections import Counter

    w = _W(ctx)
    for f in (field_a, field_b):
        if f not in w.lists + list(w.enums):
            return {"error": f"fields must be among {w.lists + list(w.enums)}"}

    def vals(r, f):
        v = r["data"].get(f)
        return {str(x).lower() for x in (v if isinstance(v, list) else [v]) if x not in (None, "", "not_stated")}
    top_a = [v for v, _ in Counter(x for r in w.disc for x in vals(r, field_a)).most_common(top)]
    top_b = [v for v, _ in Counter(x for r in w.disc for x in vals(r, field_b)).most_common(top)]
    table = {a: {b: sum(1 for r in w.disc if a in vals(r, field_a) and b in vals(r, field_b)) for b in top_b} for a in top_a}
    return {"rows": field_a, "columns": field_b, "papers": len(w.disc), "table": table}


def list_studies(ctx, conditions: list | None = None, fields: list[str] | None = None, limit: int = 25) -> dict:
    w = _W(ctx)
    conds = P._conds(conditions)
    rows = [r for r in w.disc if P._match(r, conds, w.enums)]
    fields = [f for f in (fields or ["methods", "geography", "data_modalities", "validation_level"])
              if f in w.lists + list(w.enums)]
    return {"matching": len(rows), "studies": [
        {"paper_id": r["paper_id"], "title": r["title"], "year": r["year"], "corpus": r.get("corpus"),
         "read": r["source"], **{f: r["data"].get(f) for f in fields}} for r in rows[:max(1, min(int(limit), 60))]]}


def read_study(ctx, paper_id: str, terms: list[str] | None = None) -> dict:
    from research_agent.agents.followup import read_paper

    if paper_id not in _W(ctx).disc_ids:
        return {"error": "that study is in the held-out half or not in this run; you can read discovery-half studies only"}
    return read_paper(ctx, paper_id, terms)


def test_pattern(ctx, spec: dict, description: str = "") -> dict:
    w = _W(ctx)
    spec = dict(spec or {})
    err = P.validate(spec, w.lists, w.enums)
    if err:
        return {"error": err}
    sig = P.signature(spec)
    old = ctx.pg.execute("SELECT id, description, result, ts FROM research_tests WHERE researcher_id=%s AND signature=%s",
                         (w.id, sig)).fetchone()
    if old:
        return {"already_tested": True, "test_id": old["id"], "result": old["result"],
                "note": "Your notebook already has this test; it was not run again and does not count as a new test."}
    result = P.evaluate(spec, w.disc, w.enums, ctx)
    desc = (description or "").strip()[:300] or P.describe(spec)
    tid = ctx.pg.execute("INSERT INTO research_tests (researcher_id, agenda_id, signature, spec, description, result) "
                         "VALUES (%s,%s,%s,%s::jsonb,%s,%s::jsonb) RETURNING id",
                         (w.id, w.agenda_id, sig, json.dumps(spec), desc, json.dumps(result, default=str))).fetchone()["id"]
    n_tests = ctx.pg.execute("SELECT count(*) n FROM research_tests WHERE researcher_id=%s", (w.id,)).fetchone()["n"]
    _note(ctx, "test", {"test_id": tid, "pattern": desc, "p": result.get("p"), "direction": result.get("direction")})
    return {"test_id": tid, "pattern": P.describe(spec), "result": result, "significant_in_discovery": P.significant(result),
            "tests_so_far": n_tests,
            "note": "Discovery half only. Propose a finding only for a pattern that matters for the charter; each "
                    "proposal is confirmed on the held-out half at a stricter threshold than the last."}


def propose_finding(ctx, test_id: int, statement: str) -> dict:
    w = _W(ctx)
    problem = GT.plain_statement_problem(statement)
    if problem:
        return {"error": problem}
    t = ctx.pg.execute("SELECT id, spec, result, description FROM research_tests WHERE id=%s AND researcher_id=%s",
                       (int(test_id), w.id)).fetchone()
    if not t:
        return {"error": "no such test in your notebook; run test_pattern first"}
    done = ctx.pg.execute("SELECT id, grade FROM research_findings WHERE researcher_id=%s AND test_id=%s",
                          (w.id, t["id"])).fetchone()
    if done:
        return {"error": f"this pattern was already judged (finding {done['id']}, {done['grade']}); it cannot be "
                         "proposed twice"}
    spec, disc = t["spec"], t["result"]
    subgroups = GT.subgroup_tests(spec, w.disc, w.enums, disc["direction"], ctx)
    trap_list = GT.traps(spec, w.disc, disc, w.enums, ctx)
    rivals = critic(ctx, spec, disc, subgroups, trap_list, statement) if P.significant(disc) else []
    tested = [GT.test_alternative(spec, a, w.disc, w.enums, disc, ctx) for a in rivals[:4]]
    k = 1 + ctx.pg.execute("SELECT count(*) n FROM research_findings WHERE researcher_id=%s", (w.id,)).fetchone()["n"]
    alpha = 0.05 / k                               # the bar rises with every finding proposed
    holdout = GT.holdout_test(spec, w.hold, w.enums, disc["direction"], alpha, ctx) if P.significant(disc) else \
        {"verdict": "not run", "alpha": alpha}
    g = GT.grade(disc, subgroups, trap_list, tested, holdout if holdout.get("result") else
                 {"verdict": "too few papers"}) if P.significant(disc) else GT.grade(disc, {}, [], [], {})
    examples = []
    for res in (disc, (holdout.get("result") or {})):
        papers = res.get("papers")
        examples += (papers.get("a_with", []) if isinstance(papers, dict) else papers or [])[:4]
    evidence = {"pattern": P.describe(spec), "spec": spec, "discovery": disc, "subgroups": subgroups, "traps": trap_list,
                "rivals": tested, "holdout": holdout, "examples": list(dict.fromkeys(examples))[:8],
                "papers": {"discovery": len(w.disc), "holdout": len(w.hold)}}
    fid = ctx.pg.execute(
        "INSERT INTO research_findings (researcher_id, agenda_id, test_id, statement, grade, status, reasons, evidence) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb) RETURNING id",
        (w.id, w.agenda_id, t["id"], statement.strip(), g["grade"], g["status"], json.dumps(g["reasons"]),
         json.dumps(evidence, default=str))).fetchone()["id"]
    _note(ctx, "finding", {"finding_id": fid, "statement": statement.strip(), "grade": g["grade"]})
    # the researcher learns the verdict and why, but never the held-out counts, so it cannot tune to them
    return {"finding_id": fid, "grade": g["grade"], "status": g["status"], "reasons": g["reasons"],
            "rival_explanations": [{"explanation": a["explanation"], "verdict": a["verdict"]} for a in tested]}


def _similar(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"[a-z]{3,}", a.lower())), set(re.findall(r"[a-z]{3,}", b.lower()))
    jac = len(wa & wb) / len(wa | wb) if wa | wb else 0
    return max(jac, difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio())


def scope_score(charter: dict, question: str) -> float:
    """Cosine similarity between the question and the charter's goal and scope."""
    import numpy as np

    from research_agent.embeddings import get_embedder

    emb = get_embedder()
    a = np.asarray(emb.embed_query(f"{charter.get('goal', '')} {charter.get('scope', '')}"), dtype=float)
    b = np.asarray(emb.embed_query(question), dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def add_question(ctx, question: str, why: str = "", parent_id: int | None = None, priority: int = 50) -> dict:
    w = _W(ctx)
    question = " ".join((question or "").split())
    if len(question.split()) < 5:
        return {"error": "write the question as a full sentence"}
    existing = ctx.pg.execute("SELECT id, question, status FROM research_agenda WHERE researcher_id=%s",
                              (w.id,)).fetchall()
    for e in existing:
        if _similar(e["question"], question) >= 0.7:
            return {"error": f"too close to question {e['id']} ({e['status']}): \"{e['question'][:120]}\". Work on "
                             "that one, or ask something different."}
    if not w.seeding:
        if parent_id is None:
            parent_id = w.agenda_id
        if parent_id not in {e["id"] for e in existing}:
            return {"error": "new questions must hang under an existing agenda question (parent_id)"}
    if sum(e["status"] == "open" for e in existing) >= MAX_OPEN_QUESTIONS:
        return {"error": f"the agenda already has {MAX_OPEN_QUESTIONS} open questions; park or answer some first"}
    score = scope_score(w.charter, question)
    bounds = [b for b in w.charter.get("out_of_bounds") or [] if b.lower() in question.lower()]
    status = "needs_approval" if bounds or score < settings.research_scope_min else "open"
    note = (f"touches what the charter puts out of bounds: {', '.join(bounds)}" if bounds else
            f"far from the charter (similarity {score:.2f}); needs your approval" if status != "open" else None)
    aid = ctx.pg.execute("INSERT INTO research_agenda (researcher_id, parent_id, question, why, status, status_note, "
                         "priority, scope_score) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                         (w.id, parent_id, question, (why or "")[:500], status, note,
                          max(0, min(int(priority), 100)), score)).fetchone()["id"]
    return {"agenda_id": aid, "status": status, **({"note": note} if note else {})}


def park_question(ctx, agenda_id: int, reason: str) -> dict:
    w = _W(ctx)
    n = ctx.pg.execute("UPDATE research_agenda SET status='parked', status_note=%s, updated_at=now() WHERE id=%s AND "
                       "researcher_id=%s AND status='open' RETURNING id", (reason[:400], int(agenda_id), w.id)).fetchone()
    if n:
        _note(ctx, "park", {"agenda_id": int(agenda_id), "reason": reason[:400]}, agenda_id=int(agenda_id))
    return {"parked": bool(n)}


def write_note(ctx, text: str) -> dict:
    _note(ctx, "note", {"text": (text or "")[:1500]})
    return {"saved": True}


_COND = {"type": "array", "items": {"type": "object"}, "description": "conditions: {field, any_of} or {field, none_of}; "
         "fields include extracted and question-specific ones, corpus, read, year (min/max)"}
PATTERN_SPEC = {"type": "object", "description": (
    "One of: {kind:'difference', outcome:[conds], group_a:[conds], group_b:[conds] (optional: the rest)}; "
    "{kind:'prevalence', outcome:[conds], within:[conds], max_share|min_share}; "
    "{kind:'comparison', method_a:[terms], method_b:[terms], metric}; "
    "{kind:'trend', outcome:[conds], split_year:int, within:[conds]}")}

OVERVIEW = Tool("research_overview", "Your charter, the current question, the agenda, patterns already tested, findings, "
                "budget, the fields you can count with their common values, and leads from the run.", obj({}),
                research_overview, read_only=True, max_chars=16000)
COUNTS = Tool("explore_counts", "How often each value of a field occurs among the discovery-half studies.",
              obj({"field": STR, "top": INT}, ["field"]), explore_counts, read_only=True)
CROSS = Tool("explore_cross", "Two fields against each other among the discovery-half studies.",
             obj({"field_a": STR, "field_b": STR, "top": INT}, ["field_a", "field_b"]), explore_cross, read_only=True)
LIST = Tool("list_studies", "Discovery-half studies matching conditions, with the fields you ask for.",
            obj({"conditions": _COND, "fields": STRS, "limit": INT}), list_studies, read_only=True, max_chars=14000)
READ = Tool("read_study", "Read one discovery-half study: its record and text, or only the passages mentioning terms.",
            obj({"paper_id": STR, "terms": STRS}, ["paper_id"]), read_study, read_only=True, max_chars=14000)
TEST = Tool("test_pattern", "Count a pattern on the discovery half, with an exact test. Patterns already in your "
            "notebook return their earlier result.", obj({"spec": PATTERN_SPEC, "description": STR}, ["spec"]), test_pattern)
PROPOSE = Tool("propose_finding", "Put a tested pattern through the gauntlet: subgroup tests, the trap library, a critic's "
               "rival explanations and confirmation on the held-out half. Returns the computed grade. The statement is in "
               "words, without numbers.", obj({"test_id": INT, "statement": STR}, ["test_id", "statement"]), propose_finding)
ADD_Q = Tool("add_question", "Add a question to the agenda, under the current one (or another parent).",
             obj({"question": STR, "why": STR, "parent_id": INT, "priority": INT}, ["question", "why"]), add_question)
PARK = Tool("park_question", "Park a question you cannot make progress on, with the reason.",
            obj({"agenda_id": INT, "reason": STR}, ["agenda_id", "reason"]), park_question)
NOTE = Tool("write_note", "Write to your notebook: what you noticed, what you ruled out, what to try next.",
            obj({"text": STR}, ["text"]), write_note)


# ---------------------------------------------------------------- the agents
RESEARCHER_SYSTEM = """You are the Researcher agent. You investigate a body of studies on your own initiative,
within a charter, and report patterns only when they survive testing.

How you work:
- Read research_overview first: your charter, your current question, what you already tested and found, and
  your notebook. Work only on the current question.
- Explore the discovery half (explore_counts, explore_cross, list_studies, read_study), then state a pattern as
  a spec and count it with test_pattern. Look for patterns that would matter for the charter's goal: differences
  between kinds of studies, what goes with what, what is rare and why, which methods beat others in the same
  papers, what changed over time.
- Be sceptical of your own patterns. Before proposing a finding, look for obvious confounders yourself (place,
  corpus, period, read depth). Before calling something rare, read a few studies to see whether it is really
  absent or only unrecorded.
- propose_finding sends a tested pattern through code checks and a critic, then tests it on the held-out half you
  never see. Every proposal raises the bar for the next, so propose only patterns that matter. A rejected
  finding is information: note why, and move on.
- Write finding statements in plain words without numbers; the counts come from code.
- Keep the notebook useful with write_note: what you noticed, what you ruled out, what to try next.
- Add a sub-question only when the current one needs it. Park the current question if you cannot make progress.
- Finish with the progress you made on the current question: none, some, or answered."""

SEED_TASK = ("The agenda is empty. Read research_overview (the charter and the leads from the run), explore the data "
             "a little, then add three to six top-level questions that serve the charter's goal, most promising first "
             "(priority 0-100). Questions must be answerable by counting and reading these studies.")


@dataclass
class ResearcherAgent(Agent):
    def context_message(self, ctx, task: str) -> str:
        w = _W(ctx)
        return (f"Charter:\n{json.dumps(w.charter, ensure_ascii=False)}\n\nCycle {w.cycle}. "
                f"Discovery half: {len(w.disc)} studies (the held-out half is hidden from you).\n\n{task}")


RESEARCHER = ResearcherAgent(
    name="research", role="Investigates on its own initiative within a charter.", system=RESEARCHER_SYSTEM,
    tools=[OVERVIEW, COUNTS, CROSS, LIST, READ, TEST, PROPOSE, ADD_Q, PARK, NOTE],
    finish_schema=obj({"progress": {"type": "string", "enum": ["none", "some", "answered"]}, "summary": STR,
                       "next_step": STR}, ["progress", "summary"]),
    max_turns=14, strong_model=True)

CRITIC_SYSTEM = """You are the Critic agent. A researcher believes it found a pattern. Your job is to find what else
could explain it, so that code can test each rival explanation.

Give up to four rivals, each testable:
- stratify: a possible confounder, as a condition {field, any_of}. Code checks whether the pattern holds both
  among studies that meet the condition and among those that do not. Good confounders: where the studies were
  done, which corpus, read in full or from the abstract, period, study design, data source.
- redefine: a different definition of one side of the pattern (side: outcome, group_a, group_b or within, and a
  new any_of list), for when the result may hinge on how a term was defined.
Use the fields and values you are shown. Prefer the rivals most likely to be true. Finish with your list."""

CRITIC = Agent(
    name="critic", role="Proposes rival explanations for a pattern.", system=CRITIC_SYSTEM, tools=[COUNTS, CROSS],
    finish_schema=obj({"alternatives": {"type": "array", "items": obj({
        "explanation": STR, "kind": {"type": "string", "enum": ["stratify", "redefine"]}, "field": STR,
        "any_of": STRS, "none_of": STRS, "side": STR}, ["explanation", "kind"])}}, ["alternatives"]),
    max_turns=6, strong_model=True)


def critic(ctx, spec: dict, disc: dict, subgroups: dict, trap_list: list, statement: str) -> list[dict]:
    fields = research_overview(ctx)["fields_with_common_values"]
    task = json.dumps({"statement": statement, "pattern": P.describe(spec), "spec": spec,
                       "discovery_result": {k: disc.get(k) for k in ("n", "group_a", "group_b", "share", "p", "a_better",
                                                                      "b_better", "difference")},
                       "subgroup_tests": subgroups, "traps": trap_list, "fields_with_common_values": fields},
                      ensure_ascii=False, default=str)
    try:
        out = CRITIC.run(ctx, "Propose rival explanations for this pattern:\n" + task)
    except Exception as exc:              # a failed critic leaves the finding untested by rivals, not approved
        _note(ctx, "note", {"text": f"critic failed: {str(exc)[:200]}"})
        return []
    return [a for a in out.get("alternatives") or [] if isinstance(a, dict)][:4]


SUPERVISOR_SYSTEM = """You are the Research Supervisor agent. Every few cycles you review an autonomous researcher's
agenda against its charter, so it does not drift or waste its budget.

Read research_overview and the cost and yield of each question you are given. Then:
- prune questions that left the charter or are no longer worth pursuing (prune_question, with the reason);
- merge questions that ask the same thing (merge_questions);
- re-prioritise so the most promising open questions come first (set_priority);
- finish with a short assessment and whether the work is still worth continuing."""


def prune_question(ctx, agenda_id: int, reason: str) -> dict:
    n = ctx.pg.execute("UPDATE research_agenda SET status='pruned', status_note=%s, updated_at=now() WHERE id=%s AND "
                       "researcher_id=%s AND status IN ('open','parked') RETURNING id",
                       (f"supervisor: {reason[:300]}", int(agenda_id), _W(ctx).id)).fetchone()
    return {"pruned": bool(n)}


def merge_questions(ctx, keep_id: int, drop_id: int) -> dict:
    n = ctx.pg.execute("UPDATE research_agenda SET status='pruned', status_note=%s, updated_at=now() WHERE id=%s AND "
                       "researcher_id=%s AND status IN ('open','parked') RETURNING id",
                       (f"merged into question {int(keep_id)}", int(drop_id), _W(ctx).id)).fetchone()
    return {"merged": bool(n)}


def set_priority(ctx, agenda_id: int, priority: int) -> dict:
    n = ctx.pg.execute("UPDATE research_agenda SET priority=%s, updated_at=now() WHERE id=%s AND researcher_id=%s "
                       "RETURNING id", (max(0, min(int(priority), 100)), int(agenda_id), _W(ctx).id)).fetchone()
    return {"updated": bool(n)}


SUPERVISOR = Agent(
    name="research_supervisor", role="Keeps the researcher on its charter.", system=SUPERVISOR_SYSTEM,
    tools=[OVERVIEW,
           Tool("prune_question", "Prune a question, with the reason.", obj({"agenda_id": INT, "reason": STR},
                ["agenda_id", "reason"]), prune_question),
           Tool("merge_questions", "Merge a duplicate question into another.", obj({"keep_id": INT, "drop_id": INT},
                ["keep_id", "drop_id"]), merge_questions),
           Tool("set_priority", "Set a question's priority (0-100).", obj({"agenda_id": INT, "priority": INT},
                ["agenda_id", "priority"]), set_priority)],
    finish_schema=obj({"assessment": STR, "keep_going": {"type": "boolean"}}, ["assessment", "keep_going"]),
    max_turns=8, strong_model=True)


def _yield(ctx) -> list[dict]:
    """Cost and yield per question, computed by code for the supervisor."""
    w = _W(ctx)
    rows = ctx.pg.execute(
        """SELECT a.id, a.question, a.status, a.cycles_used, a.tokens_used,
                  (SELECT count(*) FROM research_tests t WHERE t.agenda_id = a.id) tests,
                  (SELECT count(*) FROM research_findings f WHERE f.agenda_id = a.id AND f.status='confirmed') confirmed,
                  (SELECT count(*) FROM research_findings f WHERE f.agenda_id = a.id AND f.status='provisional') provisional,
                  a.scope_score
           FROM research_agenda a WHERE a.researcher_id=%s ORDER BY a.id""", (w.id,)).fetchall()
    return rows


# ---------------------------------------------------------------- one cycle
_LOCK_BASE = 91_000_000           # advisory lock key space for researchers: one cycle at a time per researcher


def run_cycle(rid: int, llm_factory=None, provider: str | None = None, schedule_next: bool = False) -> dict:
    """One cycle at a time per researcher, even with several workers: a second one steps aside."""
    _schema()
    lock = connect()
    try:
        if not lock.execute("SELECT pg_try_advisory_lock(%s) AS ok", (_LOCK_BASE + int(rid),)).fetchone()["ok"]:
            return {"skipped": "another cycle of this researcher is running"}
        return _run_cycle(rid, llm_factory, provider, schedule_next)
    finally:
        lock.close()                    # closing the session releases the lock


def _run_cycle(rid: int, llm_factory=None, provider: str | None = None, schedule_next: bool = False) -> dict:
    """One cycle: budget checks, a checkpoint when due, then one question worked on. Never runs when paused."""
    conn = connect()
    try:
        r = conn.execute("SELECT * FROM researchers WHERE id=%s", (rid,)).fetchone()
        if not r:
            raise ValueError(f"No researcher {rid}")
        if r["status"] != "active":
            return {"skipped": r["status"]}
        if r["cycles_done"] >= r["max_cycles"]:
            conn.close()
            set_status(rid, "finished", "cycle limit reached")
            return {"finished": "cycle limit reached"}
        if r["tokens_used"] >= r["total_tokens"]:
            conn.close()
            set_status(rid, "finished", "total budget used")
            return {"finished": "total budget used"}
        if _tokens_today(conn, rid) >= r["daily_tokens"]:
            if schedule_next:
                schedule(rid, delay=3600)
            return {"waiting": "daily budget reached; the next cycle waits"}
    finally:
        if not conn.closed:
            conn.close()

    ctx, r = _open(rid, llm_factory, provider)
    w = _W(ctx)
    out: dict = {"cycle": w.cycle}
    try:
        n_agenda = ctx.pg.execute("SELECT count(*) n FROM research_agenda WHERE researcher_id=%s", (rid,)).fetchone()["n"]
        if n_agenda == 0:
            w.seeding = True
            res = RESEARCHER.run(ctx, SEED_TASK)
            w.seeding = False
            _note(ctx, "cycle", {"seed": True, "summary": res.get("summary") or res.get("progress") or ""})
            out["seeded"] = ctx.pg.execute("SELECT count(*) n FROM research_agenda WHERE researcher_id=%s",
                                           (rid,)).fetchone()["n"]
            return out
        if w.cycle % max(1, settings.research_checkpoint_every) == 0:
            task = ("Review the agenda. Cost and yield per question (computed by code):\n"
                    + json.dumps(_yield(ctx), default=str))
            res = SUPERVISOR.run(ctx, task)
            _note(ctx, "checkpoint", {"assessment": res.get("assessment") or res.get("summary") or "",
                                      "keep_going": res.get("keep_going", True)})
            if res.get("keep_going") is False:
                ctx.pg.execute("UPDATE researchers SET status='paused', status_note=%s, updated_at=now() WHERE id=%s",
                               ("the supervisor recommends stopping: " + str(res.get("assessment") or "")[:300], rid))
                out["paused"] = "supervisor recommends stopping"
                return out
        # questions that used their cycles or stalled twice are parked, with the reason
        for q in ctx.pg.execute("SELECT id, cycles_used, stalled FROM research_agenda WHERE researcher_id=%s AND "
                                "status='open' AND (cycles_used >= %s OR stalled >= 2)",
                                (rid, settings.research_question_cycles)).fetchall():
            reason = ("no progress in two cycles in a row" if q["stalled"] >= 2 else
                      f"used its {settings.research_question_cycles} cycles without an answer")
            ctx.pg.execute("UPDATE research_agenda SET status='parked', status_note=%s, updated_at=now() WHERE id=%s",
                           (reason, q["id"]))
            _note(ctx, "park", {"agenda_id": q["id"], "reason": reason}, agenda_id=q["id"])
        item = ctx.pg.execute("SELECT id, question, why FROM research_agenda WHERE researcher_id=%s AND status='open' "
                              "ORDER BY priority DESC, cycles_used, id LIMIT 1", (rid,)).fetchone()
        if not item:
            waiting = ctx.pg.execute("SELECT count(*) n FROM research_agenda WHERE researcher_id=%s AND "
                                     "status='needs_approval'", (rid,)).fetchone()["n"]
            status, note = (("paused", f"waiting for your approval of {waiting} new question(s)") if waiting else
                            ("finished", "the agenda has no open questions left"))
            ctx.pg.execute("UPDATE researchers SET status=%s, status_note=%s, updated_at=now() WHERE id=%s",
                           (status, note, rid))
            out[status] = note
            return out
        w.agenda_id = item["id"]
        before = ctx.pg.execute("SELECT count(*) n FROM research_findings WHERE researcher_id=%s", (rid,)).fetchone()["n"]
        res = RESEARCHER.run(ctx, f"Current question (agenda {item['id']}): {item['question']}\n"
                                  f"Why it matters: {item['why'] or 'see the charter'}")
        progress = res.get("progress") if res.get("progress") in ("none", "some", "answered") else "none"
        after = ctx.pg.execute("SELECT count(*) n FROM research_findings WHERE researcher_id=%s", (rid,)).fetchone()["n"]
        if progress == "none" and after > before:
            progress = "some"
        ctx.pg.execute("UPDATE research_agenda SET cycles_used = cycles_used + 1, stalled = CASE WHEN %s THEN stalled + 1 "
                       "ELSE 0 END, status = CASE WHEN %s THEN 'answered' ELSE status END, updated_at=now() WHERE id=%s",
                       (progress == "none", progress == "answered", item["id"]))
        _note(ctx, "cycle", {"question": item["question"], "progress": progress,
                             "summary": (res.get("summary") or "")[:1500], "next_step": (res.get("next_step") or "")[:500]})
        out.update(agenda_id=item["id"], progress=progress, new_findings=after - before)
        return out
    finally:
        tokens = sum(int(getattr(c, "usage", None).input_tokens + c.usage.output_tokens)
                     for c in ctx.clients if getattr(c, "usage", None) is not None)
        ctx.pg.execute("UPDATE researchers SET cycles_done = cycles_done + 1, tokens_used = tokens_used + %s, "
                       "updated_at=now() WHERE id=%s", (tokens, rid))
        if w.agenda_id:
            ctx.pg.execute("UPDATE research_agenda SET tokens_used = tokens_used + %s WHERE id=%s", (tokens, w.agenda_id))
        _note(ctx, "usage", {"tokens": tokens})
        still = ctx.pg.execute("SELECT status FROM researchers WHERE id=%s", (rid,)).fetchone()["status"]
        ctx.close()
        out["tokens"] = tokens
        if schedule_next and still == "active":
            schedule(rid, delay=settings.research_cycle_gap_seconds)
