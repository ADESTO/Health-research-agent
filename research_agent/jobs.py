"""Durable job queue in Postgres: runs, maps, follow-ups and drafts survive server restarts and worker crashes.

    enqueue(kind, payload)  adds a job
    Worker.run_forever()    claims jobs (FOR UPDATE SKIP LOCKED, so workers never share one), sends a
                            heartbeat while working, retries failures after a delay, and gives up after
                            max_attempts, marking the run or follow-up failed
    requeue_stale()         jobs whose worker stopped sending heartbeats go back in the queue; a run picked
                            up again resumes from its last finished step instead of starting over

Start workers with `python -m research_agent.cli worker`, or let the web server run one inside itself
(EMBEDDED_WORKER=1, the default) for a single-machine setup.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
import uuid

from research_agent.db import connect

HEARTBEAT_SECONDS = 20
STALE_AFTER_SECONDS = 180
RETRY_DELAY_SECONDS = 30
_llm_factory = None        # tests plug a fake model in here


def enqueue(kind: str, payload: dict, max_attempts: int = 2, delay_seconds: int = 0) -> int:
    conn = connect()
    try:
        return conn.execute(
            "INSERT INTO jobs (kind, payload, max_attempts, run_at) VALUES (%s,%s::jsonb,%s, now() + %s * interval '1 second') "
            "RETURNING id", (kind, json.dumps(payload), max_attempts, int(delay_seconds))).fetchone()["id"]
    finally:
        conn.close()


def job_for_run(run_id: str) -> dict | None:
    conn = connect()
    try:
        return conn.execute("""SELECT id, kind, status, attempts, max_attempts, error, created_at, finished_at FROM jobs
                               WHERE payload->>'run_id' = %s AND kind IN ('run','map') ORDER BY id DESC LIMIT 1""",
                            (run_id,)).fetchone()
    finally:
        conn.close()


def requeue_stale(conn) -> int:
    """Jobs whose worker went silent: back to the queue, or failed when out of attempts."""
    rows = conn.execute(
        f"""UPDATE jobs SET status = CASE WHEN attempts < max_attempts THEN 'queued' ELSE 'failed' END,
                   error = coalesce(error, '') || 'worker stopped responding; ', locked_by = NULL,
                   finished_at = CASE WHEN attempts < max_attempts THEN NULL ELSE now() END
            WHERE status = 'running' AND heartbeat_at < now() - interval '{STALE_AFTER_SECONDS} seconds'
            RETURNING id, status, kind, payload""").fetchall()
    for r in rows:
        if r["status"] == "failed":
            _mark_failed(conn, r["kind"], r["payload"], "the worker stopped responding")
    return len(rows)


def _mark_failed(conn, kind: str, payload: dict, error: str) -> None:
    if kind in ("run", "map"):
        conn.execute("UPDATE runs SET status='failed', error=coalesce(error, %s), finished_at=now() WHERE run_id=%s",
                     (error[:500], payload["run_id"]))
    elif kind == "followup":
        conn.execute("UPDATE followups SET status='failed', content=%s WHERE id=%s",
                     (f"The follow-up failed: {error[:300]}", payload["answer_id"]))
    elif kind == "research_cycle":
        from research_agent.research import researcher

        researcher.run_cycle(payload["researcher_id"], llm_factory=factory, schedule_next=True)
    elif kind == "draft":
        conn.execute("UPDATE drafts SET status='failed', content_md=%s, finished_at=now() WHERE id=%s",
                     (f"The draft failed: {error[:300]}", payload["draft_id"]))
    elif kind == "research_cycle":
        conn.execute("UPDATE researchers SET status='paused', status_note=%s, updated_at=now() WHERE id=%s",
                     (f"a cycle failed: {error[:300]}", payload["researcher_id"]))


def execute(kind: str, payload: dict, attempt: int) -> None:
    """Do one job. A retried run or map resumes from its last finished step."""
    factory = _llm_factory
    if kind == "run":
        from research_agent.agents.orchestrator import run_research

        if attempt > 1:
            run_research(resume=payload["run_id"], mode=payload.get("mode", "orchestrated"),
                         provider=payload.get("provider"), llm_factory=factory)
        else:
            run_research(payload["question"], mode=payload.get("mode", "orchestrated"), provider=payload.get("provider"),
                         run_id=payload["run_id"], llm_factory=factory)
    elif kind == "map":
        from research_agent.opportunity.pipeline import run_map

        if attempt > 1:
            run_map(resume=payload["run_id"], provider=payload.get("provider"), llm_factory=factory)
        else:
            run_map(payload["question"], provider=payload.get("provider"), run_id=payload["run_id"], llm_factory=factory)
    elif kind == "followup":
        from research_agent.agents import followup

        followup.answer(payload["run_id"], payload["question"], payload.get("focus"), provider=payload.get("provider"),
                        llm_factory=factory, answer_id=payload["answer_id"])
    elif kind == "research_cycle":
        from research_agent.research import researcher

        researcher.run_cycle(payload["researcher_id"], llm_factory=factory, schedule_next=True)
    elif kind == "draft":
        from research_agent.agents import draft

        draft.run_draft(payload["draft_id"], provider=payload.get("provider"), llm_factory=factory)
    else:
        raise ValueError(f"unknown job kind {kind!r}")


class Worker:
    def __init__(self, name: str | None = None, poll_seconds: float = 2.0):
        self.name = name or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self.poll = poll_seconds
        self._stop = threading.Event()

    def claim(self, conn) -> dict | None:
        return conn.execute(
            """UPDATE jobs SET status='running', locked_by=%s, heartbeat_at=now(), attempts=attempts+1
               WHERE id = (SELECT id FROM jobs WHERE status='queued' AND run_at <= now()
                           ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1)
               RETURNING id, kind, payload, attempts, max_attempts""", (self.name,)).fetchone()

    def run_once(self) -> bool:
        """Claim and run one job. Returns False when the queue was empty."""
        conn = connect()
        try:
            requeue_stale(conn)
            job = self.claim(conn)
        finally:
            conn.close()
        if not job:
            return False
        beat = threading.Event()

        def heartbeat():
            while not beat.wait(HEARTBEAT_SECONDS):
                c = connect()
                try:
                    c.execute("UPDATE jobs SET heartbeat_at=now() WHERE id=%s", (job["id"],))
                finally:
                    c.close()
        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            execute(job["kind"], job["payload"], job["attempts"])
            status, error = "done", None
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            beat.set()
        conn = connect()
        try:
            if status == "done":
                conn.execute("UPDATE jobs SET status='done', finished_at=now(), locked_by=NULL WHERE id=%s", (job["id"],))
            elif job["attempts"] < job["max_attempts"]:
                conn.execute(f"""UPDATE jobs SET status='queued', locked_by=NULL, error=coalesce(error,'') || %s,
                                 run_at = now() + interval '{RETRY_DELAY_SECONDS} seconds' WHERE id=%s""",
                             (error[:400] + "; ", job["id"]))
            else:
                conn.execute("UPDATE jobs SET status='failed', finished_at=now(), locked_by=NULL, "
                             "error=coalesce(error,'') || %s WHERE id=%s", (error[:400], job["id"]))
                _mark_failed(conn, job["kind"], job["payload"], error)
        finally:
            conn.close()
        return True

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.run_once():
                    self._stop.wait(self.poll)
            except Exception:
                traceback.print_exc()
                self._stop.wait(self.poll * 5)

    def stop(self) -> None:
        self._stop.set()


def start_embedded(n: int = 1) -> list[Worker]:
    workers = [Worker(name=f"embedded-{i}") for i in range(max(1, n))]
    for w in workers:
        threading.Thread(target=w.run_forever, daemon=True).start()
    return workers
