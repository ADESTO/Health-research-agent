"""Command line entry point.

    python -m research_agent.cli init-db
    python -m research_agent.cli scope                       # size the health subset (no DB writes)
    python -m research_agent.cli ingest [--limit N]
    python -m research_agent.cli ask "your question" [--mode orchestrated|pipeline] [--provider anthropic|groq]
    python -m research_agent.cli report RUN_ID
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_event(agent: str, kind: str, payload: dict) -> None:
    if kind == "start":
        print(f"\n▶ {agent}: {payload.get('task', '')[:160]}")
    elif kind == "tool_call":
        args = json.dumps(payload.get("input", {}), ensure_ascii=False)
        print(f"   · {agent} → {payload['tool']}({args[:140]})")
    elif kind == "tool_result" and payload.get("error"):
        print(f"   ! {agent} ← {payload['tool']} error: {payload.get('preview', '')[:200]}")
    elif kind == "finish":
        print(f"✔ {agent} finished")
    elif kind == "error":
        print(f"✖ {agent}: {payload.get('error')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("scope")
    p_ing = sub.add_parser("ingest"); p_ing.add_argument("--limit", type=int)
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=["orchestrated", "pipeline"], default="orchestrated")
    p_ask.add_argument("--provider", choices=["anthropic", "groq"])
    p_ask.add_argument("--out", help="write the report to this .md file")
    p_rep = sub.add_parser("report"); p_rep.add_argument("run_id")
    args = ap.parse_args(argv)

    if args.cmd == "init-db":
        from research_agent.db import init_schema

        init_schema(); print("schema ready")
    elif args.cmd == "scope":
        from research_agent.ingestion.load import scope

        res = scope()
        print(f"Health subset: {res['total']:,} papers")
        print("By year:", ", ".join(f"{y}: {n:,}" for y, n in res["by_year"]))
        print("By rule:", res["by_reason"])
        print("Top reasons:"); [print(f"  {r}: {n:,}") for r, n in res["top_reasons"]]
        print("Top primary categories:"); [print(f"  {c}: {n:,}") for c, n in res["top_primary_categories"]]
        print("Random samples:"); [print(f"  {pid} [{why}] {t[:100]}") for pid, why, t in res["samples"]]
    elif args.cmd == "ingest":
        from research_agent.ingestion.load import run_ingest

        print(json.dumps(run_ingest(limit=args.limit), indent=2))
    elif args.cmd == "ask":
        from research_agent.agents.orchestrator import run_research

        res = run_research(args.question, mode=args.mode, provider=args.provider, on_event=_print_event)
        print("\n" + "=" * 80 + "\n")
        print(res["report"] or "(no report)")
        print(f"\nrun_id={res['run_id']}  tokens={res['usage']}")
        if args.out and res["report"]:
            Path(args.out).write_text(res["report"]); print(f"saved to {args.out}")
    elif args.cmd == "report":
        from research_agent.db import get_conn

        with get_conn() as pg:
            row = pg.execute("SELECT status, report_md, error FROM runs WHERE run_id=%s", (args.run_id,)).fetchone()
        print(row["report_md"] if row and row["report_md"] else row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
