"""HTTP API. Runs are long (minutes), so POST /runs starts one in a background thread and the client
polls GET /runs/{id} for the agent timeline and the report. mode "map" builds a Research Opportunity
Map instead of a written report; it is shown and downloaded the same way.

    uvicorn research_agent.api.main:app --reload     # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from research_agent.db import get_conn

STATIC = Path(__file__).parent / "static"

@asynccontextmanager
async def _lifespan(_app):
    """Bring an existing database up to date (new tables and indexes only) before serving requests."""
    from research_agent.db import init_schema

    try:
        init_schema()
    except Exception as exc:   # the server should still start; endpoints report their own errors
        print(f"database migration failed: {exc}")
    yield


app = FastAPI(title="Health Research Intelligence API", version="0.1.0", lifespan=_lifespan)

# Simple per-IP rate limit: runs cost money. A map costs about two ordinary runs, so it counts twice.
_RUNS_PER_HOUR = 5
_COST = {"map": 2}
# a run is a map if its protocol step saved a note (true from the map's first step onwards)
_IS_MAP = "EXISTS (SELECT 1 FROM run_notes n WHERE n.run_id = runs.run_id AND n.agent = 'protocol')"
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
        n = pg.execute("SELECT count(*) n FROM papers").fetchone()["n"]
    return {"ok": True, "papers": n}


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

    from research_agent.agents.orchestrator import run_research
    from research_agent.opportunity.pipeline import run_map

    run_id = str(uuid.uuid4())
    with get_conn() as pg:
        pg.execute("INSERT INTO runs (run_id, question, status) VALUES (%s,%s,'queued')", (run_id, req.question))

    def worker():
        try:
            if req.mode == "map":
                run_map(req.question, provider=req.provider, run_id=run_id)
            else:
                run_research(req.question, mode=req.mode, provider=req.provider, run_id=run_id)
        except Exception as exc:  # recorded on the run row; make sure it never stays 'queued'
            with get_conn() as pg:
                pg.execute("UPDATE runs SET status='failed', error=coalesce(error, %s) WHERE run_id=%s",
                           (str(exc)[:500], run_id))

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id, "mode": req.mode, "poll": f"/runs/{run_id}"}


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
    return {"run": run, "shortlist_size": n_short, "events": events}


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


@app.get("/runs/{run_id}/claims")
def get_claims(run_id: str):
    with get_conn() as pg:
        rows = pg.execute("SELECT id, agent, text, claim_type, predicate, status, result, review_note "
                          "FROM claims WHERE run_id=%s ORDER BY id", (run_id,)).fetchall()
    return {"claims": rows}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
