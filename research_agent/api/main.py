"""HTTP API. Runs are long (minutes), so POST /runs starts one in a background thread and the client
polls GET /runs/{id} for the agent timeline and the report. mode "map" builds a Research Opportunity
Map instead of a written report; it is shown and downloaded the same way.

    uvicorn research_agent.api.main:app --reload     # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.db import get_conn

STATIC = Path(__file__).parent / "static"

@asynccontextmanager
async def _lifespan(_app):
    """Bring an existing database up to date (new tables and indexes only) before serving requests."""
    from research_agent.db import init_schema

    try:
        init_schema()
        from research_agent.agents.followup import schema_is_ready

        schema_is_ready()
    except Exception as exc:   # the server should still start; endpoints report their own errors
        print(f"database migration failed: {exc}")
    workers = []
    if settings.embedded_workers > 0:   # single-machine setup: the server works its own queue
        from research_agent.jobs import start_embedded

        workers = start_embedded(settings.embedded_workers)
    yield
    for w in workers:
        w.stop()


app = FastAPI(title="Health Research Intelligence API", version="0.1.0", lifespan=_lifespan)

# Simple per-IP rate limit: runs cost money. A map costs about two ordinary runs, so it counts twice.
_RUNS_PER_HOUR = 5
_COST = {"map": 2}
# a run is a map if its protocol step saved a note (true from the map's first step onwards)
_IS_MAP = ("coalesce((SELECT n.content->>'mode' = 'map' FROM run_notes n WHERE n.run_id = runs.run_id AND n.agent = 'mode'), "
           # runs from before the mode note: a map has a map note, or a protocol and none of the report agents
           "EXISTS (SELECT 1 FROM run_notes n WHERE n.run_id = runs.run_id AND n.agent = 'map') OR "
           "(EXISTS (SELECT 1 FROM run_notes n WHERE n.run_id = runs.run_id AND n.agent = 'protocol') AND NOT EXISTS "
           "(SELECT 1 FROM run_notes n WHERE n.run_id = runs.run_id AND n.agent IN ('methods','trends','gaps','evidence','synthesis'))))")
_hits: dict[str, deque] = defaultdict(deque)


class RunRequest(BaseModel):
    question: str = Field(min_length=10, max_length=600)
    mode: str = Field(default="orchestrated", pattern="^(orchestrated|pipeline|map)$")
    provider: str | None = Field(default=None, pattern="^(anthropic|groq|deepseek)$")


@app.get("/", include_in_schema=False)
def index():
    """The browser UI (static/index.html); the JSON API is unchanged."""
    return FileResponse(STATIC / "index.html")


@app.get("/runs")
def list_runs(limit: int = 10):
    with get_conn() as pg:
        rows = pg.execute(
            f"SELECT run_id, question, status, provider, created_at, {_IS_MAP} AS is_map FROM runs "
            "ORDER BY created_at DESC LIMIT %s", (max(1, min(int(limit), 50)),)).fetchall()
    return {"runs": rows}


@app.get("/health")
def health():
    with get_conn() as pg:
        rows = pg.execute("SELECT source, count(*) n FROM papers GROUP BY source").fetchall()
    return {"ok": True, "papers": sum(r["n"] for r in rows), "by_source": {r["source"]: r["n"] for r in rows}}


@app.post("/runs", status_code=202)
def start_run(req: RunRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    q, now = _hits[ip], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    cost = _COST.get(req.mode, 1)
    if len(q) + cost > _RUNS_PER_HOUR:
        raise HTTPException(429, "Run limit reached; try again later.")
    q.extend([now] * cost)

    import uuid

    from research_agent.jobs import enqueue

    run_id = str(uuid.uuid4())
    with get_conn() as pg:
        pg.execute("INSERT INTO runs (run_id, question, status) VALUES (%s,%s,'queued')", (run_id, req.question))
    # a durable job: it survives a server restart, and a worker that dies mid-run is replaced
    job = enqueue("map" if req.mode == "map" else "run",
                  {"run_id": run_id, "question": req.question, "mode": req.mode, "provider": req.provider})
    return {"run_id": run_id, "mode": req.mode, "job_id": job, "poll": f"/runs/{run_id}"}


@app.get("/runs/{run_id}")
def get_run(run_id: str, after_event: int = 0):
    with get_conn() as pg:
        run = pg.execute("SELECT run_id, question, status, provider, model, error, input_tokens, output_tokens, "
                         "llm_calls, created_at, finished_at, report_md IS NOT NULL AS has_report, "
                         f"{_IS_MAP} AS is_map FROM runs WHERE run_id=%s", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "no such run")
        events = pg.execute("SELECT id, agent, kind, payload, ts FROM run_events WHERE run_id=%s AND id > %s "
                            "ORDER BY id LIMIT 500", (run_id, after_event)).fetchall()
        n_short = pg.execute("SELECT count(*) n FROM run_papers WHERE run_id=%s", (run_id,)).fetchone()["n"]
    from research_agent.jobs import job_for_run

    return {"run": run, "shortlist_size": n_short, "events": events, "job": job_for_run(run_id)}


@app.get("/runs/{run_id}/report")
def get_report(run_id: str):
    with get_conn() as pg:
        row = pg.execute("SELECT report_md FROM runs WHERE run_id=%s", (run_id,)).fetchone()
    if not row or not row["report_md"]:
        raise HTTPException(404, "report not ready")
    return {"report_markdown": row["report_md"]}


class FollowupRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    about: str | None = Field(default=None, pattern="^[CGNERWHDcgnerwhd][0-9]{1,6}$")
    provider: str | None = Field(default=None, pattern="^(anthropic|groq|deepseek)$")


_FOLLOWUPS_PER_HOUR = 40
_fhits: dict[str, deque] = defaultdict(deque)


@app.post("/runs/{run_id}/followups", status_code=202)
def ask_followup(run_id: str, req: FollowupRequest, request: Request):
    """Ask a follow-up question about a finished run. Answered in the background; poll the GET below."""
    from research_agent.agents import followup

    ip = request.client.host if request.client else "unknown"
    q, now = _fhits[ip], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= _FOLLOWUPS_PER_HOUR:
        raise HTTPException(429, "Follow-up limit reached; try again later.")
    with get_conn() as pg:
        run = pg.execute("SELECT status FROM runs WHERE run_id=%s", (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, "no such run")
    if run["status"] not in ("done", "failed"):
        raise HTTPException(409, "The run is still in progress; ask follow-ups once it has finished.")
    q.append(now)
    ids = followup.answer_in_background(run_id, req.question, req.about.upper() if req.about else None, req.provider)
    return {**ids, "poll": f"/runs/{run_id}/followups"}


@app.get("/runs/{run_id}/followups")
def list_followups(run_id: str):
    from research_agent.agents.followup import history

    return {"messages": [{"id": r["id"], "role": r["role"], "content": r["content"], "status": r["status"],
                          "about": (r["meta"] or {}).get("focus"), "new_claims": (r["meta"] or {}).get("new_claims", []),
                          "ts": r["ts"]} for r in history(run_id, limit=200)]}


class DraftRequest(BaseModel):
    kind: str = Field(pattern="^(proposal|review)$")
    direction: str = Field(default="", max_length=1500)
    about: list[str] = Field(default_factory=list, max_length=6)
    provider: str | None = Field(default=None, pattern="^(anthropic|groq|deepseek)$")
    citation_style: str = Field(default="author-year", pattern="^(author-year|numbered)$")


_DRAFTS_PER_HOUR = 6
_dhits: dict[str, deque] = defaultdict(deque)
_ITEM_ID = re.compile(r"^[CGNERWHDcgnerwhd][0-9]{1,6}$")


@app.get("/runs/{run_id}/graph")
def get_graph(run_id: str):
    """The run as a graph (nodes and links) and as a mind map (tree), built from the database."""
    from research_agent.runstate import RunContext
    from research_agent.tools.graph import build_graph

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        return build_graph(ctx)
    finally:
        ctx.close()


@app.get("/runs/{run_id}/directions")
def get_directions(run_id: str):
    """Gaps and directions a draft can argue for (map gaps, untried combinations, designs, reported gaps)."""
    from research_agent.agents.draft import directions
    from research_agent.runstate import RunContext

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        return {"directions": directions(ctx)}
    finally:
        ctx.close()


@app.post("/runs/{run_id}/drafts", status_code=202)
def start_draft(run_id: str, req: DraftRequest, request: Request):
    """Draft a research proposal or a review manuscript from a finished run. Written in the background."""
    from research_agent.agents import draft

    if any(not _ITEM_ID.match(a) for a in req.about):
        raise HTTPException(422, "about: item ids like G2, N1 or D1")
    ip = request.client.host if request.client else "unknown"
    q, now = _dhits[ip], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= _DRAFTS_PER_HOUR:
        raise HTTPException(429, "Draft limit reached; try again later.")
    with get_conn() as pg:
        run = pg.execute("SELECT status, report_md IS NOT NULL AS has_report FROM runs WHERE run_id=%s",
                         (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, "no such run")
    if run["status"] != "done" or not run["has_report"]:
        raise HTTPException(409, "Drafts are written from a finished run with a report or map.")
    q.append(now)
    ids = draft.draft_in_background(run_id, req.kind, req.direction, [a.upper() for a in req.about], req.provider,
                                    req.citation_style)
    return {**ids, "poll": f"/drafts/{ids['draft_id']}"}


@app.get("/runs/{run_id}/drafts")
def run_drafts(run_id: str):
    from research_agent.agents.draft import list_drafts

    return {"drafts": list_drafts(run_id)}


@app.get("/drafts/{draft_id}")
def get_draft(draft_id: int):
    from research_agent.agents.draft import get_draft as fetch

    d = fetch(draft_id)
    if not d:
        raise HTTPException(404, "no such draft")
    return {"draft": d}


@app.get("/drafts/{draft_id}/export/{fmt}")
def export_draft(draft_id: int, fmt: str):
    """Download a draft as md, docx, html, or its cited references as bib or ris."""
    from fastapi.responses import Response

    from research_agent.exports import DRAFT_FORMATS, export_draft as build

    if fmt not in DRAFT_FORMATS:
        raise HTTPException(404, f"unknown format; choose from {', '.join(DRAFT_FORMATS)}")
    try:
        data, name, media = build(draft_id, fmt)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


class ResearcherRequest(BaseModel):
    goal: str = Field(min_length=10, max_length=600)
    scope: str = Field(default="", max_length=600)
    out_of_bounds: list[str] = Field(default_factory=list, max_length=10)
    max_cycles: int = Field(default=20, ge=1, le=200)
    daily_tokens: int = Field(default=2_000_000, ge=10_000, le=100_000_000)
    total_tokens: int = Field(default=10_000_000, ge=10_000, le=1_000_000_000)
    split: str = Field(default="random", pattern="^(random|year)$")
    split_year: int | None = Field(default=None, ge=1990, le=2100)
    provider: str | None = Field(default=None, pattern="^(anthropic|groq|deepseek)$")


@app.post("/runs/{run_id}/researchers", status_code=202)
def start_researcher(run_id: str, req: ResearcherRequest):
    """Start an open-ended researcher on a finished run. It works in background cycles within its budget."""
    from research_agent.research import researcher as R

    with get_conn() as pg:
        run = pg.execute("SELECT status FROM runs WHERE run_id=%s", (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, "no such run")
    if run["status"] != "done":
        raise HTTPException(409, "A researcher works on a finished run or map.")
    if req.split == "year" and not req.split_year:
        raise HTTPException(422, "a year split needs split_year")
    try:
        out = R.start_in_background(run_id, req.goal, scope=req.scope, out_of_bounds=req.out_of_bounds,
                                    max_cycles=req.max_cycles, daily_tokens=req.daily_tokens,
                                    total_tokens=req.total_tokens, split=req.split, split_year=req.split_year,
                                    provider=req.provider)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {**out, "poll": f"/researchers/{out['researcher_id']}"}


@app.get("/runs/{run_id}/researchers")
def run_researchers(run_id: str):
    from research_agent.research.researcher import list_for_run

    return {"researchers": list_for_run(run_id)}


@app.get("/researchers/{rid}")
def get_researcher(rid: int):
    from research_agent.research.researcher import get

    r = get(rid)
    if not r:
        raise HTTPException(404, "no such researcher")
    return r


class ResearcherAction(BaseModel):
    action: str = Field(pattern="^(pause|resume|stop)$")


@app.post("/researchers/{rid}/status")
def researcher_status(rid: int, req: ResearcherAction):
    from research_agent.research.researcher import get, set_status

    r = get(rid)
    if not r:
        raise HTTPException(404, "no such researcher")
    if r["researcher"]["status"] in ("stopped", "finished") and req.action == "resume":
        raise HTTPException(409, "This researcher has ended; start a new one.")
    set_status(rid, {"pause": "paused", "resume": "active", "stop": "stopped"}[req.action], f"{req.action}d by you")
    return {"ok": True}


class QuestionDecision(BaseModel):
    approve: bool


@app.post("/researchers/{rid}/agenda/{agenda_id}")
def decide_agenda_question(rid: int, agenda_id: int, req: QuestionDecision):
    from research_agent.research.researcher import decide_question

    decide_question(rid, agenda_id, req.approve)
    return {"ok": True}


@app.get("/runs/{run_id}/export/{fmt}")
def export_run(run_id: str, fmt: str):
    """Download a run as md, docx, html, bib, ris, csv, xlsx, protocol, screening, burden or burden_chart."""
    from fastapi.responses import Response

    from research_agent.exports import FORMATS, export

    if fmt not in FORMATS:
        raise HTTPException(404, f"unknown format; choose from {', '.join(FORMATS)}")
    try:
        data, name, media = export(run_id, fmt)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/runs/{run_id}/claims")
def get_claims(run_id: str):
    with get_conn() as pg:
        rows = pg.execute("SELECT id, agent, text, claim_type, predicate, status, result, review_note "
                          "FROM claims WHERE run_id=%s ORDER BY id", (run_id,)).fetchall()
    return {"claims": rows}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
