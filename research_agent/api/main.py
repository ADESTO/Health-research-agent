"""HTTP API. Runs are long (minutes), so POST /runs starts one in a background thread and the client
polls GET /runs/{id} for the agent timeline and the report.

    uvicorn research_agent.api.main:app --reload
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from research_agent.db import get_conn

app = FastAPI(title="Health Research Intelligence API", version="0.1.0")

# Simple per-IP rate limit: runs cost money.
_RUNS_PER_HOUR = 5
_hits: dict[str, deque] = defaultdict(deque)


class RunRequest(BaseModel):
    question: str = Field(min_length=10, max_length=600)
    mode: str = Field(default="orchestrated", pattern="^(orchestrated|pipeline)$")
    provider: str | None = Field(default=None, pattern="^(anthropic|groq)$")


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
    if len(q) >= _RUNS_PER_HOUR:
        raise HTTPException(429, "Run limit reached; try again later.")
    q.append(now)

    import uuid

    from research_agent.agents.orchestrator import run_research

    run_id = str(uuid.uuid4())
    with get_conn() as pg:
        pg.execute("INSERT INTO runs (run_id, question, status) VALUES (%s,%s,'queued')", (run_id, req.question))

    def worker():
        try:
            run_research(req.question, mode=req.mode, provider=req.provider, run_id=run_id)
        except Exception as exc:  # recorded on the run row; make sure it never stays 'queued'
            with get_conn() as pg:
                pg.execute("UPDATE runs SET status='failed', error=coalesce(error, %s) WHERE run_id=%s",
                           (str(exc)[:500], run_id))

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id, "poll": f"/runs/{run_id}"}


@app.get("/runs/{run_id}")
def get_run(run_id: str, after_event: int = 0):
    with get_conn() as pg:
        run = pg.execute("SELECT run_id, question, status, provider, model, error, input_tokens, output_tokens, "
                         "llm_calls, created_at, finished_at, report_md IS NOT NULL AS has_report "
                         "FROM runs WHERE run_id=%s", (run_id,)).fetchone()
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


@app.get("/runs/{run_id}/claims")
def get_claims(run_id: str):
    with get_conn() as pg:
        rows = pg.execute("SELECT id, agent, text, claim_type, predicate, status, result, review_note "
                          "FROM claims WHERE run_id=%s ORDER BY id", (run_id,)).fetchall()
    return {"claims": rows}
