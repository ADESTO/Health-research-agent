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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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

@app.middleware("http")
async def _access(request: Request, call_next):
    """With AUTH_REQUIRED=1: who is asking, and may they touch this run? (see research_agent/auth.py)
    Without it, everyone is a local admin and nothing changes."""
    request.state.user = None
    if not settings.auth_required:
        return await call_next(request)
    from research_agent import auth

    path = request.url.path
    with get_conn() as pg:
        user = auth.user_for_token(pg, auth.token_from(request))
        if user is None and path not in auth.PUBLIC:
            return JSONResponse({"detail": "Sign in with your invite link to use this."}, status_code=401)
        if user is not None:
            run_id = auth.run_of_path(pg, path)
            # a run you may not see is reported missing, so ids of other people's runs reveal nothing
            verdict = "hidden" if run_id is False else (auth.access(pg, user, run_id, request.method) if run_id else "ok")
            if verdict == "hidden":
                return JSONResponse({"detail": "no such run"}, status_code=404)
            if verdict == "read_only":
                return JSONResponse({"detail": "This is a shared sample run, so it is read-only. Start your own "
                                               "run to ask follow-ups or draft from it."}, status_code=403)
    request.state.user = user
    return await call_next(request)


def _user(request: Request) -> dict | None:
    return getattr(request.state, "user", None)


@app.get("/login", include_in_schema=False)
def login(token: str = ""):
    """The invite link: store its token as a cookie, then open the app."""
    from research_agent import auth

    with get_conn() as pg:
        user = auth.user_for_token(pg, token)
    if not user:
        return JSONResponse({"detail": "This invite link is not valid. Ask for a new one."}, status_code=401)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(auth.COOKIE, token, max_age=auth.COOKIE_DAYS * 86400, httponly=True, samesite="lax",
                    secure=settings.public_url.startswith("https://"))
    return resp


@app.get("/logout", include_in_schema=False)
def logout():
    from research_agent import auth

    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(auth.COOKIE)
    return resp


@app.get("/me")
def me(request: Request):
    """Who is signed in, and how many runs they have left today. auth=false on a local install."""
    user = _user(request)
    if not settings.auth_required:
        return {"auth": False}
    if user is None:
        return {"auth": True, "signed_in": False}
    from research_agent import auth

    with get_conn() as pg:
        used = auth.runs_today(pg, user["id"])
    return {"auth": True, "signed_in": True, "name": user["name"], "is_admin": user["is_admin"],
            "runs_per_day": settings.runs_per_day, "runs_left_today": max(0, settings.runs_per_day - used)}


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
    # purposive: an agent chooses the papers; systematic: a fixed query, every hit screened, all eligible analysed
    search: str = Field(default="purposive", pattern="^(purposive|systematic)$")
    # the user's own documents to include (ids from /documents), and corpus papers a link pointed to
    documents: list[int] = Field(default_factory=list, max_length=30)
    include_papers: list[str] = Field(default_factory=list, max_length=30)


@app.get("/", include_in_schema=False)
def index():
    """The browser UI (static/index.html); the JSON API is unchanged."""
    return FileResponse(STATIC / "index.html")


@app.get("/runs")
def list_runs(request: Request, limit: int = 10):
    user = _user(request)
    where, params = "", []
    if user is not None and not user["is_admin"]:
        where, params = "WHERE owner_id = %s OR shared ", [user["id"]]
    with get_conn() as pg:
        rows = pg.execute(
            f"SELECT run_id, question, status, provider, created_at, shared, {_IS_MAP} AS is_map, "
            f"(owner_id IS NOT DISTINCT FROM %s) AS mine FROM runs {where}"
            "ORDER BY created_at DESC LIMIT %s",
            [user["id"] if user else None] + params + [max(1, min(int(limit), 50))]).fetchall()
    if user is None:
        for r in rows:
            r["mine"] = True
    return {"runs": rows}


@app.get("/health")
def health():
    with get_conn() as pg:
        rows = pg.execute("SELECT source, count(*) n FROM papers WHERE source <> 'upload' GROUP BY source").fetchall()
    return {"ok": True, "papers": sum(r["n"] for r in rows), "by_source": {r["source"]: r["n"] for r in rows}}


@app.post("/runs", status_code=202)
def start_run(req: RunRequest, request: Request):
    user = _user(request)
    cost = _COST.get(req.mode, 1)
    if user is not None:
        # signed-in users: a daily allowance each, counted from the runs table
        from research_agent import auth

        if not user["is_admin"]:
            with get_conn() as pg:
                used = auth.runs_today(pg, user["id"])
            if used + cost > settings.runs_per_day:
                raise HTTPException(429, f"You have used today's {settings.runs_per_day} runs (a map counts as "
                                         "two). More become available 24 hours after each one started.")
    else:
        ip = request.client.host if request.client else "unknown"
        q, now = _hits[ip], time.time()
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) + cost > _RUNS_PER_HOUR:
            raise HTTPException(429, "Run limit reached; try again later.")
        q.extend([now] * cost)

    import uuid

    from research_agent.jobs import enqueue

    run_id = str(uuid.uuid4())
    from research_agent import uploads

    with get_conn() as pg:
        docs = uploads.owned(pg, user["id"] if user else None, req.documents)
        if len(docs) < len(set(req.documents)):
            raise HTTPException(404, "one of the documents is not yours or no longer exists")
        corpus = [r["paper_id"] for r in pg.execute(
            "SELECT paper_id FROM papers WHERE paper_id = ANY(%s) AND source <> 'upload'",
            (req.include_papers or [""],)).fetchall()]
        pg.execute("INSERT INTO runs (run_id, question, status, owner_id, requested_mode) "
                   "VALUES (%s,%s,'queued',%s,%s)", (run_id, req.question, user["id"] if user else None, req.mode))
        if docs or corpus:
            uploads.set_for_run(pg, run_id, [f"UP{d}" for d in docs] + corpus)
    # a durable job: it survives a server restart, and a worker that dies mid-run is replaced
    job = enqueue("map" if req.mode == "map" else "run",
                  {"run_id": run_id, "question": req.question, "mode": req.mode, "provider": req.provider,
                   "search": req.search})
    return {"run_id": run_id, "mode": req.mode, "search": req.search, "job_id": job, "poll": f"/runs/{run_id}"}


# Failures that resuming cannot fix: the question itself found nothing to analyse. Anything else (credits ran out,
# a rate limit, a timeout, the server restarting) is worth another try from the last finished step.
_NOT_RESUMABLE = ("found no eligible papers", "found no relevant papers", "no usable question-specific fields",
                  "needs the protocol", "the map needs a shortlist")


def resumable(error: str | None) -> bool:
    return not any(x in (error or "") for x in _NOT_RESUMABLE)


@app.post("/runs/{run_id}/resume")
def resume_run(run_id: str):
    """Carry on a failed run from its last finished step. The run's owner and admins may (see _access); a resume
    does not count against the daily allowance, since the run was already counted when it started."""
    from research_agent.jobs import enqueue

    with get_conn() as pg:
        run = pg.execute("SELECT status, error, requested_mode FROM runs WHERE run_id=%s", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "no such run")
        if run["status"] != "failed":
            raise HTTPException(409, "Only a run that stopped can be resumed.")
        if not resumable(run["error"]):
            raise HTTPException(409, "This run found nothing to analyse, so resuming would stop in the same place. "
                                     "Start a new run with a wider question.")
        job = pg.execute("SELECT kind, payload, status FROM jobs WHERE payload->>'run_id' = %s AND kind IN ('run','map') "
                         "ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
        if job and job["status"] in ("queued", "running"):
            raise HTTPException(409, "This run is already being resumed.")
        pg.execute("UPDATE runs SET status='queued', error=NULL, finished_at=NULL WHERE run_id=%s", (run_id,))
    kind = job["kind"] if job else ("map" if run["requested_mode"] == "map" else "run")
    payload = dict(job["payload"]) if job else {"run_id": run_id, "mode": run["requested_mode"] or "orchestrated"}
    jid = enqueue(kind, {**payload, "resume": True})
    return {"run_id": run_id, "job_id": jid, "resumed": True}


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

    return {"run": {**run, "resumable": run["status"] == "failed" and resumable(run["error"])},
            "shortlist_size": n_short, "events": events, "job": job_for_run(run_id)}


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


@app.get("/runs/{run_id}/opportunities")
def get_opportunities(run_id: str, kind: str | None = None, state: str | None = None):
    """What this run says is worth doing, each with what supports it, what weakens it and where it came from."""
    from research_agent.runstate import RunContext
    from research_agent.tools import epistemics, opportunities

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        return {"opportunities": opportunities.listing(ctx, kind, state),
                "claims_by_state": epistemics.summary(ctx), "state_means": epistemics.PLAIN}
    finally:
        ctx.close()


@app.post("/runs/{run_id}/opportunities/{item_id}")
def set_opportunity_state(run_id: str, item_id: str, body: dict):
    """Mark an opportunity as addressed, dismissed or superseded, with a note saying why."""
    from research_agent.runstate import RunContext
    from research_agent.tools import opportunities

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        res = opportunities.set_state(ctx, item_id, str(body.get("state") or ""), str(body.get("note") or ""))
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res
    finally:
        ctx.close()


@app.get("/runs/{run_id}/corroboration")
def get_corroboration(run_id: str, driver: str | None = None):
    """Which papers back each other up, finding by finding, with quotes from each side."""
    from research_agent.runstate import RunContext
    from research_agent.tools.corroboration import corroborate_tool

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        return corroborate_tool(ctx, driver=driver, limit=80, per_finding=40)
    finally:
        ctx.close()


@app.post("/runs/{run_id}/precedent")
def post_precedent(run_id: str, body: dict | None = None):
    """Search the whole corpus for studies that already do each of this run's opportunities."""
    from research_agent.runstate import RunContext
    from research_agent.tools import precedent

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        res = precedent.check(ctx, item_id=(body or {}).get("item_id"))
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res
    finally:
        ctx.close()


class DocumentRequest(BaseModel):
    kind: str = Field(default="paper", pattern="^(paper|report)$")
    title: str = Field(default="", max_length=300)
    year: int | None = Field(default=None, ge=1900, le=2100)
    authors: str = Field(default="", max_length=500)
    filename: str | None = Field(default=None, max_length=200)
    content_base64: str | None = Field(default=None, max_length=30_000_000)
    url: str | None = Field(default=None, max_length=500)


@app.post("/documents", status_code=201)
def add_document(req: DocumentRequest, request: Request):
    """Add one of your own documents (a file, as base64, or a link). Private to you; attach it to a run when
    you start one."""
    from research_agent import uploads

    user = _user(request)
    if not req.content_base64 and not req.url:
        raise HTTPException(422, "add a file or a link")
    try:
        data = uploads.b64(req.content_base64) if req.content_base64 else None
    except Exception:
        raise HTTPException(422, "the file could not be decoded")
    with get_conn() as pg:
        res = uploads.add_document(pg, user["id"] if user else None, kind=req.kind, title=req.title,
                                   year=req.year, authors=req.authors, filename=req.filename, data=data,
                                   url=req.url)
    if "error" in res:
        raise HTTPException(422, res["error"])
    return res


@app.get("/documents")
def list_documents(request: Request):
    from research_agent import uploads

    user = _user(request)
    with get_conn() as pg:
        return {"documents": uploads.list_documents(pg, user["id"] if user else None)}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, request: Request):
    from research_agent import uploads

    user = _user(request)
    with get_conn() as pg:
        if not uploads.delete_document(pg, user["id"] if user else None, doc_id):
            raise HTTPException(404, "no such document")
    return {"deleted": doc_id}


@app.get("/documents/{doc_id}/file")
def document_file(doc_id: int, request: Request):
    from fastapi.responses import Response

    from research_agent import uploads

    user = _user(request)
    with get_conn() as pg:
        f = uploads.file_of(pg, user["id"] if user else None, doc_id)
    if not f or f["content"] is None:
        raise HTTPException(404, "no such document")
    name = (f["filename"] or f"document-{doc_id}").replace('"', "")
    return Response(bytes(f["content"]), media_type=f["content_type"] or "application/octet-stream",
                    headers={"Content-Disposition": f'inline; filename="{name}"'})


@app.get("/guidelines")
def get_guidelines(condition: str | None = None):
    """The guideline library (a reference, never counted as evidence)."""
    from research_agent.tools import guidelines as GL

    with get_conn() as pg:
        return {"guidelines": GL.list_guidelines(pg, condition)}


@app.get("/guidelines/{guideline_id}")
def get_guideline(guideline_id: int):
    from research_agent.tools import guidelines as GL

    with get_conn() as pg:
        recs = GL.recommendations(pg, [guideline_id])
    if not recs:
        raise HTTPException(404, "no such guideline, or it has no recommendations")
    return {"recommendations": recs}


@app.post("/runs/{run_id}/guidelines/compare")
def compare_guidelines(run_id: str, body: dict):
    """A finished run's studies against the library's recommendations for a condition (code only)."""
    from research_agent.runstate import RunContext
    from research_agent.tools import guidelines as GL

    ctx = RunContext.attach(run_id)
    try:
        res = GL.compare(ctx, (body or {}).get("condition"), (body or {}).get("guideline_ids"))
        return {**res, "markdown": "\n".join(GL.markdown(ctx))}
    finally:
        ctx.close()


class MetaRequest(BaseModel):
    condition: str = Field(min_length=3, max_length=120)
    limit: int | None = Field(default=None, ge=1, le=150)
    provider: str | None = Field(default=None, pattern="^(anthropic|groq|deepseek)$")


@app.post("/runs/{run_id}/meta", status_code=202)
def start_meta(run_id: str, req: MetaRequest):
    """Read a finished run's papers for the prevalence of one condition and pool them (a background job)."""
    from research_agent.jobs import enqueue

    with get_conn() as pg:
        run = pg.execute("SELECT status FROM runs WHERE run_id=%s", (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, "no such run")
    if run["status"] != "done":
        raise HTTPException(409, "A meta-analysis works on a finished run.")
    job = enqueue("meta", {"run_id": run_id, "condition": req.condition.strip(), "limit": req.limit,
                           "provider": req.provider})
    return {"job_id": job, "poll": f"/runs/{run_id}/meta?condition={req.condition.strip()}"}


@app.get("/runs/{run_id}/meta")
def get_meta(run_id: str, condition: str):
    """The pooled result for one condition, with every study row and its quotes (code only, no model call)."""
    from research_agent.runstate import RunContext
    from research_agent.tools import meta

    ctx = RunContext.attach(run_id)
    try:
        if not ctx.notes().get(meta._note_key(condition)):
            raise HTTPException(404, "no meta-analysis for this condition yet")
        res = meta.pool_run(ctx, condition)
        return {**res, "markdown": "\n".join(meta.markdown(ctx, condition))}
    finally:
        ctx.close()


@app.get("/runs/{run_id}/meta/forest.png")
def get_forest(run_id: str, condition: str):
    import tempfile

    from research_agent.runstate import RunContext
    from research_agent.tools import meta

    ctx = RunContext.attach(run_id)
    try:
        if not ctx.notes().get(meta._note_key(condition)):
            raise HTTPException(404, "no meta-analysis for this condition yet")
        path = meta.forest_plot(ctx, condition, tempfile.mktemp(suffix=".png"))
    finally:
        ctx.close()
    if not path:
        raise HTTPException(404, "no condition has two or more studies to pool")
    from fastapi.responses import Response

    data = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return Response(data, media_type="image/png")


@app.get("/runs/{run_id}/cohort")
def get_cohort(run_id: str):
    """The scope this run's numbers are counted over, and how many papers it drops."""
    from research_agent.runstate import RunContext
    from research_agent.tools import cohort

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        from research_agent.tools.extraction import _rows, _values, known_fields

        lists, enums = known_fields(ctx)
        # the fields a scope can be written on, each with the values these papers actually state, so the
        # page can offer choices instead of asking for spellings
        cohort.clear_cache(ctx)
        rows = _rows_unscoped(ctx)
        fields = {}
        for f in ["geography", "study_designs", "populations", "organisms"] + [n for n in enums
                                                                                  if n.startswith("q_")]:
            if f not in lists and f not in enums:
                continue
            seen: dict[str, int] = {}
            for r in rows:
                for v in _values(r["data"], f):
                    seen[v] = seen.get(v, 0) + 1
            if seen:
                fields[f] = [v for v, _ in sorted(seen.items(), key=lambda kv: -kv[1])][:40]
        return {"cohort": cohort.of(ctx), **cohort.summary(ctx), "fields": fields}
    finally:
        ctx.close()


def _rows_unscoped(ctx):
    """Every analysed paper, ignoring the run's cohort: the choices offered must include what it excludes."""
    from research_agent.tools import cohort
    from research_agent.tools.extraction import _rows

    saved = cohort.of(ctx)
    ctx._cohort = None
    try:
        return _rows(ctx)
    finally:
        ctx._cohort = saved


@app.post("/runs/{run_id}/cohort")
def post_cohort(run_id: str, body: dict):
    """Set the scope (or clear it with {"clear": true}). Every later count uses it."""
    from research_agent.runstate import RunContext
    from research_agent.tools import cohort

    try:
        ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    except ValueError:
        raise HTTPException(404, "no such run")
    try:
        if body.get("clear"):
            return cohort.clear(ctx)
        res = cohort.set_cohort(ctx, include=body.get("include"), exclude=body.get("exclude"),
                                unstated=str(body.get("unstated") or "keep"), note=str(body.get("note") or ""))
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res
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


def _admin_only(request: Request, what: str) -> None:
    user = _user(request)
    if user is not None and not user["is_admin"] and settings.researchers_admin_only:
        raise HTTPException(403, f"{what} is not available during the beta.")


@app.post("/runs/{run_id}/researchers", status_code=202)
def start_researcher(run_id: str, req: ResearcherRequest, request: Request):
    """Start an open-ended researcher on a finished run. It works in background cycles within its budget."""
    from research_agent.research import researcher as R

    _admin_only(request, "Starting an open-ended researcher")

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


@app.post("/researchers/{rid}/budget")
def researcher_budget(rid: int, body: dict, request: Request):
    """Give a researcher more cycles or tokens. One that stopped for want of room starts working again."""
    from research_agent.research import researcher as R

    _admin_only(request, "Changing a researcher's budget")

    res = R.set_budget(rid, max_cycles=body.get("max_cycles"), add_cycles=body.get("add_cycles"),
                       total_tokens=body.get("total_tokens"), add_tokens=body.get("add_tokens"),
                       daily_tokens=body.get("daily_tokens"))
    if "error" in res:
        raise HTTPException(404 if "no researcher" in res["error"] else 400, res["error"])
    return res


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
