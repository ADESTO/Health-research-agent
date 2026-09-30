"""Shared run workspace. Agents never pass big payloads to each other directly; they read and write
this shared state (Postgres), and hand each other short summaries."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from research_agent.config import settings
from research_agent.db import connect
from research_agent.llm import LLMClient, get_llm


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _routing_factory(base, clients: list):
    """Client factory for a run: passes the step name through for model routing (test and custom factories
    that take only `strong` still work), and keeps every client so usage can be totalled per step."""
    import inspect

    try:
        takes_step = "step" in inspect.signature(base).parameters
    except (TypeError, ValueError):
        takes_step = False

    def factory(strong: bool = False, step: str | None = None):
        client = base(strong=strong, step=step) if takes_step else base(strong=strong)
        if step:
            try:
                client._step = step
            except Exception:
                pass
        clients.append(client)
        return client
    factory._routed = True
    return factory


def usage_cost(u: dict) -> float | None:
    """USD cost from token counts, using PRICE_* settings (per million tokens). None when prices are unset."""
    pi, pc, po = settings.price_input_per_m, settings.price_cached_input_per_m, settings.price_output_per_m
    if not (pi or po):
        return None
    fresh = u["input_tokens"] - u["cached_input_tokens"]
    return round((fresh * pi + u["cached_input_tokens"] * (pc if pc else pi) + u["output_tokens"] * po) / 1e6, 4)


@dataclass
class RunContext:
    run_id: str
    question: str
    llm_factory: Callable[..., LLMClient]
    pg: Any = None
    on_event: Callable[[str, str, dict], None] | None = None
    clients: list = field(default_factory=list)   # every LLM client created in this run (for token totals)
    _llm: LLMClient | None = field(default=None, repr=False)
    _ev: str | None = field(default=None, repr=False)   # extraction schema version this run was read with

    def __post_init__(self):
        if self.llm_factory is not None and not getattr(self.llm_factory, "_routed", False):
            object.__setattr__(self, "llm_factory", _routing_factory(self.llm_factory, self.clients))

    def __setattr__(self, name, value):
        # a factory assigned later (tests, custom code) gets the same step routing and usage accounting
        if (name == "llm_factory" and value is not None and not getattr(value, "_routed", False)
                and "clients" in self.__dict__):
            value = _routing_factory(value, self.clients)
        object.__setattr__(self, name, value)

    # --- lifecycle -------------------------------------------------------------------------
    @classmethod
    def create(cls, question: str, provider: str | None = None, llm_factory=None, on_event=None,
               run_id: str | None = None) -> "RunContext":
        base = llm_factory or (lambda strong=False, step=None: get_llm(provider, strong=strong, step=step))
        clients: list = []
        factory = _routing_factory(base, clients)

        ctx = cls(run_id=run_id or str(uuid.uuid4()), question=question, llm_factory=factory,
                  pg=connect(), on_event=on_event, clients=clients)
        llm = ctx.llm
        ctx.pg.execute(
            "INSERT INTO runs (run_id, question, status, provider, model) VALUES (%s,%s,'running',%s,%s) "
            "ON CONFLICT (run_id) DO UPDATE SET status='running', provider=EXCLUDED.provider, model=EXCLUDED.model",
            (ctx.run_id, question, llm.provider, llm.model),
        )
        ctx._prepare_session()
        ctx._pin_version()
        return ctx

    @classmethod
    def attach(cls, run_id: str, provider: str | None = None, llm_factory=None, on_event=None) -> "RunContext":
        """Open a finished run for follow-up questions WITHOUT changing it: its status, report and finish time
        stay as they are (resume, by contrast, reopens the run and clears its report)."""
        pg = connect()
        row = pg.execute("SELECT question FROM runs WHERE run_id=%s", (run_id,)).fetchone()
        if not row:
            pg.close()
            raise ValueError(f"No run with id {run_id}")
        base = llm_factory or (lambda strong=False, step=None: get_llm(provider, strong=strong, step=step))
        clients: list = []
        factory = _routing_factory(base, clients)

        ctx = cls(run_id=run_id, question=row["question"], llm_factory=factory, pg=pg, on_event=on_event,
                  clients=clients)
        ctx._prepare_session()
        ctx._pin_version()
        return ctx

    @classmethod
    def resume(cls, run_id: str, provider: str | None = None, llm_factory=None, on_event=None) -> "RunContext":
        """Re-attach to an earlier run: its shortlist, extractions, claims and agent notes are reused."""
        pg = connect()
        row = pg.execute("SELECT question FROM runs WHERE run_id=%s", (run_id,)).fetchone()
        pg.close()
        if not row:
            raise ValueError(f"No run with id {run_id}")
        ctx = cls.create(row["question"], provider=provider, llm_factory=llm_factory, on_event=on_event,
                         run_id=run_id)
        ctx.pg.execute("UPDATE runs SET error=NULL, finished_at=NULL, report_md=NULL WHERE run_id=%s", (run_id,))
        return ctx

    def child(self) -> "RunContext":
        """Same run, separate DB connection + LLM client — lets sub-agents run in parallel threads."""
        c = RunContext(self.run_id, self.question, self.llm_factory, connect(), self.on_event, self.clients)
        c._ev = self._ev
        c._prepare_session()
        return c

    def _pin_version(self) -> None:
        """Every run keeps the extraction schema version it was read with. A run opened later (follow-ups, map,
        drafts, exports, resume) must read the same records, not look for them under a newer version that a
        schema change introduced. Runs from before this was recorded get the version that covers most of their
        papers (ties go to the version with more full-text reads, then to the current one)."""
        try:
            row = self.pg.execute("SELECT extraction_version FROM runs WHERE run_id=%s", (self.run_id,)).fetchone()
        except Exception:        # a database from before the column: the current version, as before
            self._ev = settings.extraction_schema_version
            return
        version = row["extraction_version"] if row else None
        if not version:
            current = settings.extraction_schema_version
            rows = self.pg.execute(
                """SELECT e.schema_version v, count(*) n, sum((e.source = 'fulltext')::int) ft
                   FROM run_papers rp JOIN extractions e USING (paper_id) WHERE rp.run_id=%s GROUP BY 1""",
                (self.run_id,)).fetchall()
            version = max(rows, key=lambda r: (r["n"], r["ft"], r["v"] == current))["v"] if rows else current
            self.pg.execute("UPDATE runs SET extraction_version=%s WHERE run_id=%s", (version, self.run_id))
        self._ev = version

    def _prepare_session(self) -> None:
        # HNSW + WHERE filters: widen the candidate pool, and on pgvector >= 0.8 keep scanning
        # the index until enough rows pass the filter (iterative scans). Older versions ignore it.
        for stmt in ("SET hnsw.ef_search = 200", "SET hnsw.iterative_scan = relaxed_order"):
            try:
                self.pg.execute(stmt)
            except Exception:
                pass

    def close(self) -> None:
        if self.pg is not None:
            self.pg.close()

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = self.llm_factory()
        return self._llm

    # --- events / notes --------------------------------------------------------------------
    def emit(self, agent: str, kind: str, payload: dict | None = None) -> None:
        payload = payload or {}
        self.pg.execute("INSERT INTO run_events (run_id, agent, kind, payload) VALUES (%s,%s,%s,%s::jsonb)",
                        (self.run_id, agent, kind, _dumps(payload)))
        if self.on_event:
            self.on_event(agent, kind, payload)

    def save_note(self, agent: str, content: dict) -> None:
        self.pg.execute(
            "INSERT INTO run_notes (run_id, agent, content) VALUES (%s,%s,%s::jsonb) "
            "ON CONFLICT (run_id, agent) DO UPDATE SET content = EXCLUDED.content, ts = now()",
            (self.run_id, agent, _dumps(content)),
        )

    def notes(self) -> dict[str, dict]:
        rows = self.pg.execute("SELECT agent, content FROM run_notes WHERE run_id=%s ORDER BY ts",
                               (self.run_id,)).fetchall()
        return {r["agent"]: r["content"] for r in rows}

    # --- shortlist -------------------------------------------------------------------------
    def shortlist_ids(self) -> list[str]:
        return [r["paper_id"] for r in self.pg.execute(
            "SELECT paper_id FROM run_papers WHERE run_id=%s ORDER BY score DESC NULLS LAST, paper_id",
            (self.run_id,)).fetchall()]

    def token_usage(self) -> dict:
        out = {"input_tokens": sum(c.usage.input_tokens for c in self.clients),
               "cached_input_tokens": sum(c.usage.cached_input_tokens for c in self.clients),
               "output_tokens": sum(c.usage.output_tokens for c in self.clients),
               "llm_calls": sum(c.usage.calls for c in self.clients)}
        cost = usage_cost(out)
        if cost is not None:
            out["cost_usd"] = cost
        return out

    def usage_by_step(self) -> dict:
        """Tokens (and cost, when prices are set) per step: each agent, extraction, protocol fields, re-checks."""
        steps: dict[str, dict] = {}
        for c in self.clients:
            s = steps.setdefault(getattr(c, "_step", "other"),
                                 {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
            s["input_tokens"] += c.usage.input_tokens
            s["cached_input_tokens"] += c.usage.cached_input_tokens
            s["output_tokens"] += c.usage.output_tokens
            s["llm_calls"] += c.usage.calls
        for s in steps.values():
            cost = usage_cost(s)
            if cost is not None:
                s["cost_usd"] = cost
        return dict(sorted(steps.items(), key=lambda kv: -(kv[1].get("cost_usd") or kv[1]["input_tokens"])))

    def finish(self, report_md: str | None, error: str | None = None) -> None:
        u = self.token_usage()
        try:
            self.save_note("usage", {"total": u, "by_step": self.usage_by_step()})
        except Exception:
            pass
        self.pg.execute(
            "UPDATE runs SET status=%s, report_md=%s, error=%s, finished_at=now(), "
            "input_tokens=%s, output_tokens=%s, llm_calls=%s WHERE run_id=%s",
            ("failed" if error else "done", report_md, error,
             u["input_tokens"], u["output_tokens"], u["llm_calls"], self.run_id),
        )

    @property
    def extraction_version(self) -> str:
        return self._ev or settings.extraction_schema_version
