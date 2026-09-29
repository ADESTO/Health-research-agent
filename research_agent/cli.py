"""Command line entry point.

    python -m research_agent.cli init-db
    python -m research_agent.cli scope                       # size the health subset (no DB writes)
    python -m research_agent.cli ingest [--limit N]                 # arXiv health subset
    python -m research_agent.cli pmc-ingest "malaria forecasting"   # a PubMed Central topic slice
    python -m research_agent.cli ask "your question" [--mode orchestrated|pipeline] [--provider anthropic|groq|deepseek]
    python -m research_agent.cli map "your question" [--provider deepseek]   # Research Opportunity Map
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
    elif kind == "run":
        print(f"run_id={payload['run_id']}" + ("  (resumed)" if payload.get("resumed") else ""))
    elif kind == "skipped":
        print(f"↷ {agent} skipped: {payload.get('reason')}")




def chat(run_id: str, question: str | None = None, about: str | None = None, provider: str | None = None) -> int:
    """Follow-up questions about a finished run, in the terminal. The report itself is never changed."""
    import re as _re

    from research_agent.agents import followup
    from research_agent.db import get_conn, init_schema

    init_schema()
    with get_conn() as pg:
        run = pg.execute("SELECT question, status FROM runs WHERE run_id=%s", (run_id,)).fetchone()
    if not run:
        print(f"No run with id {run_id}. `python -m research_agent.cli runs` lists recent runs."); return 1

    def ask(q: str, focus: str | None) -> None:
        print("\n…thinking (it may look up claims, count, or read papers)\n")
        try:
            res = followup.answer(run_id, q, focus, provider=provider)
        except Exception as exc:
            print(f"The follow-up failed: {exc}"); return
        print(res["answer"])
        if res["new_claims"]:
            print(f"\n(new claims recorded for this run: {', '.join(res['new_claims'])})")

    if question:
        ask(question, about)
        return 0
    print(f"Follow-up on: {run['question']}")
    past = followup.history(run_id)
    if past:
        print(f"\n{len([p for p in past if p['role'] == 'user'])} earlier question(s). Last exchange:")
        for r in past[-2:]:
            print(f"\n{'You' if r['role'] == 'user' else 'Answer'}: {r['content'][:800]}")
    print("\nAsk about a claim (C12), a map item (G2, N1), a paper or a number. Type 'exit' to leave.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q:
            continue
        if q.lower() in ("exit", "quit", ":q"):
            break
        m = _re.search(r"\b([CGNERWHD]\d+)\b", q)
        ask(q, m.group(1).upper() if m else None)
    return 0

def format_usage(by_step: dict, total: dict | None = None) -> str:
    """A small table: where a run's tokens (and money, when PRICE_* is set) went."""
    if not by_step:
        return ""
    rows = [(k, v) for k, v in by_step.items() if v["llm_calls"]]
    has_cost = any("cost_usd" in v for _, v in rows)
    tot_in = sum(v["input_tokens"] for _, v in rows) or 1
    lines = ["", f"{'step':<20}{'calls':>7}{'input':>11}{'cached':>9}{'output':>9}{'share':>8}"
             + (f"{'cost $':>9}" if has_cost else "")]
    for k, v in rows:
        cached = f"{round(100 * v['cached_input_tokens'] / v['input_tokens'])}%" if v["input_tokens"] else "-"
        lines.append(f"{k:<20}{v['llm_calls']:>7}{v['input_tokens']:>11,}{cached:>9}{v['output_tokens']:>9,}"
                     f"{round(100 * v['input_tokens'] / tot_in):>7}%"
                     + (f"{v.get('cost_usd', 0):>9.4f}" if has_cost else ""))
    if total and "cost_usd" in total:
        lines.append(f"{'total':<20}{'':>7}{'':>11}{'':>9}{'':>9}{'':>8}{total['cost_usd']:>9.4f}")
    if not has_cost:
        lines.append("(set PRICE_INPUT_PER_M, PRICE_CACHED_INPUT_PER_M and PRICE_OUTPUT_PER_M in .env to see cost)")
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    p_scope = sub.add_parser("scope")
    p_scope.add_argument("--samples", type=int, default=20)
    p_scope.add_argument("--csv", help="also write 300 random papers to this CSV for manual review")
    p_ing = sub.add_parser("ingest"); p_ing.add_argument("--limit", type=int)
    p_ing.add_argument("--from-year", type=int,
                       help="only load papers from this year on (e.g. recent years first); rerun without it later")
    p_pmc = sub.add_parser("pmc-ingest", help="load a PubMed Central open-access topic slice")
    p_pmc.add_argument("query", help='PMC search, e.g. "malaria forecasting" or "sepsis AND machine learning"')
    p_pmc.add_argument("--limit", type=int, help="stop after this many articles")
    p_pmc.add_argument("--from-year", type=int)
    p_pmc.add_argument("--to-year", type=int)
    p_pmc.add_argument("--include-noncommercial", action="store_true",
                       help="also keep CC BY-NC articles (private research use; marked in reports)")
    p_lic = sub.add_parser("pmc-licences", help="re-check licences of PMC papers already loaded")
    p_lic.add_argument("--remove", action="store_true", help="delete papers that no longer qualify")
    p_lic.add_argument("--include-noncommercial", action="store_true",
                       help="treat CC BY-NC papers as allowed (if you loaded them on purpose)")
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question", nargs="?", help="omit when using --resume")
    p_ask.add_argument("--resume", metavar="RUN_ID", help="continue a failed run, skipping finished agents")
    p_ask.add_argument("--redo", metavar="AGENTS",
                       help="with --resume: comma-separated agents to run again, e.g. synthesis or evidence,synthesis")
    p_ask.add_argument("--mode", choices=["orchestrated", "pipeline"], default="orchestrated")
    p_ask.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_ask.add_argument("--out", help="write the report to this .md file")
    p_map = sub.add_parser("map", help="build a Research Opportunity Map for a question")
    p_map.add_argument("question", nargs="?", help="omit when using --resume")
    p_map.add_argument("--resume", metavar="RUN_ID", help="continue a failed map run")
    p_map.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_map.add_argument("--out", help="write the map to this .md file")
    p_rep = sub.add_parser("report"); p_rep.add_argument("run_id")
    p_chat = sub.add_parser("chat", help="ask follow-up questions about a finished run")
    p_chat.add_argument("run_id")
    p_chat.add_argument("--question", "-q", help="ask one question and exit")
    p_chat.add_argument("--about", help="the item the question is about, e.g. C512 or G2")
    p_chat.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_exp = sub.add_parser("export", help="export a finished run (report, references, data, review record)")
    p_exp.add_argument("run_id")
    p_exp.add_argument("--format", "-f", default="docx",
                       help="md, docx, html, bib, ris, csv, xlsx, protocol, screening, burden, burden_chart")
    p_exp.add_argument("--out", "-o", help="file to write (default: a name based on the run and format)")
    p_dr = sub.add_parser("draft", help="draft a research proposal or review manuscript from a finished run")
    p_dr.add_argument("run_id", nargs="?", help="the run to build on (not needed with --export)")
    p_dr.add_argument("--type", "-t", choices=["proposal", "review"], default="proposal", dest="kind")
    p_dr.add_argument("--direction", "-d", default="", help="the gap or direction to argue for, in your words")
    p_dr.add_argument("--about", default="", help="map items it builds on, e.g. G2 or G2,D1")
    p_dr.add_argument("--directions", action="store_true", help="list the run's gaps and directions, then exit")
    p_dr.add_argument("--list", action="store_true", help="list the run's drafts, then exit")
    p_dr.add_argument("--export", metavar="DRAFT_ID", type=int, help="export an existing draft instead")
    p_dr.add_argument("--format", "-f", default="docx", choices=["md", "docx", "html", "bib", "ris"])
    p_dr.add_argument("--out", "-o", help="file to write")
    p_dr.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_w = sub.add_parser("worker", help="work the job queue (runs, maps, follow-ups started from the web page)")
    p_w.add_argument("--concurrency", "-c", type=int, default=1)
    sub.add_parser("burden-fetch", help="download WHO estimates of malaria cases and deaths by country")
    p_bi = sub.add_parser("burden-import", help="import burden estimates from a CSV (iso3 or country, year, cases)")
    p_bi.add_argument("path"); p_bi.add_argument("--source", default="CSV")
    p_use = sub.add_parser("usage", help="tokens (and cost) per step for a finished run")
    p_use.add_argument("run_id")
    sub.add_parser("runs", help="list recent runs (to find a run_id to resume)")
    args = ap.parse_args(argv)

    if args.cmd == "init-db":
        from research_agent.db import init_schema

        init_schema(); print("schema ready")
    elif args.cmd == "scope":
        from research_agent.ingestion.load import scope

        res = scope(args.samples, args.csv)
        print(f"Health subset: {res['total']:,} papers")
        print("By year:", ", ".join(f"{y}: {n:,}" for y, n in res["by_year"]))
        print("By rule:", res["by_reason"])
        print("Top reasons:"); [print(f"  {r}: {n:,}") for r, n in res["top_reasons"]]
        print("Top primary categories:"); [print(f"  {c}: {n:,}") for c, n in res["top_primary_categories"]]
        print("Random samples:"); [print(f"  {pid} [{why}] {t[:100]}") for pid, why, t in res["samples"]]
        if args.csv:
            print(f"Wrote 300 random papers to {args.csv}: mark is_health y/n to measure precision")
    elif args.cmd == "ingest":
        from research_agent.ingestion.load import run_ingest

        print(json.dumps(run_ingest(limit=args.limit, from_year=args.from_year), indent=2))
    elif args.cmd == "pmc-ingest":
        from research_agent.ingestion.pmc import ingest as pmc_ingest

        res = pmc_ingest(args.query, limit=args.limit, from_year=args.from_year, to_year=args.to_year,
                         include_noncommercial=args.include_noncommercial)
        print(json.dumps(res, indent=2))
        print("Done. New papers are tagged source='pmc' and are already in the search indexes.")
    elif args.cmd == "pmc-licences":
        from research_agent.db import connect, init_schema
        from research_agent.ingestion.pmc import audit_licences

        init_schema()
        pg = connect()
        try:
            res = audit_licences(pg, remove=args.remove, allow_noncommercial=args.include_noncommercial)
        finally:
            pg.close()
        print(json.dumps(res, indent=2))
        if res["failing"] and not args.remove:
            print(f"{res['failing']:,} papers do not meet the licence rules; rerun with --remove to delete them.")
    elif args.cmd == "ask":
        from research_agent.agents.orchestrator import run_research

        if not args.question and not args.resume:
            ap.error("give a question, or --resume RUN_ID")
        if args.redo:
            if not args.resume:
                ap.error("--redo needs --resume RUN_ID")
            from research_agent.db import get_conn

            redo = [a.strip() for a in args.redo.split(",") if a.strip()]
            with get_conn() as pg:
                pg.execute("DELETE FROM run_notes WHERE run_id=%s AND agent = ANY(%s)", (args.resume, redo))
                pg.execute("DELETE FROM claims WHERE run_id=%s AND agent = ANY(%s)", (args.resume, redo))
                if "evidence" in redo:  # re-check every remaining claim from scratch
                    pg.execute("UPDATE claims SET status='pending', result=NULL, review_note=NULL "
                               "WHERE run_id=%s", (args.resume,))
            print(f"will re-run: {', '.join(redo)}")
        try:
            res = run_research(args.question, mode=args.mode, provider=args.provider, on_event=_print_event,
                               resume=args.resume)
        except Exception as exc:
            print(f"\n✖ Run stopped: {exc}")
            if getattr(exc, "run_id", None):
                print(f"\nNothing is lost. Continue later with:\n  python -m research_agent.cli ask "
                      f"--resume {exc.run_id} --mode {args.mode}" + (f" --provider {args.provider}" if args.provider else ""))
            return 1
        print("\n" + "=" * 80 + "\n")
        print(res["report"] or "(no report)")
        print(f"\nrun_id={res['run_id']}  (token use: python -m research_agent.cli usage {res['run_id']})")
        if args.out and res["report"]:
            Path(args.out).write_text(res["report"]); print(f"saved to {args.out}")
    elif args.cmd == "map":
        from research_agent.opportunity.pipeline import run_map

        if not args.question and not args.resume:
            ap.error("give a question, or --resume RUN_ID")
        try:
            res = run_map(args.question, provider=args.provider, on_event=_print_event, resume=args.resume)
        except Exception as exc:
            print(f"\n✖ Map stopped: {exc}")
            if getattr(exc, "run_id", None):
                print(f"\nNothing is lost. Continue later with:\n  python -m research_agent.cli map "
                      f"--resume {exc.run_id}" + (f" --provider {args.provider}" if args.provider else ""))
            return 1
        print("\n" + "=" * 80 + "\n")
        print(res["report"])
        print(f"\nrun_id={res['run_id']}  (token use: python -m research_agent.cli usage {res['run_id']})")
        if args.out:
            Path(args.out).write_text(res["report"]); print(f"saved to {args.out}")
    elif args.cmd == "chat":
        return chat(args.run_id, args.question, args.about, args.provider)
    elif args.cmd == "draft":
        return draft_cmd(args, ap)
    elif args.cmd == "worker":
        import threading as _t

        from research_agent.db import init_schema
        from research_agent.jobs import Worker

        init_schema()
        workers = [Worker() for _ in range(max(1, args.concurrency))]
        print(f"{len(workers)} worker(s) waiting for jobs. Ctrl+C to stop.")
        threads = [_t.Thread(target=w.run_forever, daemon=True) for w in workers]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=1)
        except KeyboardInterrupt:
            for w in workers:
                w.stop()
            print("stopping (a job in progress resumes from its last finished step when a worker next starts)")
    elif args.cmd == "export":
        from research_agent.exports import export

        try:
            data, name, _ = export(args.run_id, args.format)
        except Exception as exc:
            print(f"Export failed: {exc}"); return 1
        Path(args.out or name).write_bytes(data)
        print(f"saved {args.out or name} ({len(data):,} bytes)")
    elif args.cmd in ("burden-fetch", "burden-import"):
        from research_agent.db import get_conn, init_schema
        from research_agent.tools import burden

        init_schema()
        with get_conn() as pg:
            try:
                res = burden.fetch_who(pg) if args.cmd == "burden-fetch" else burden.import_csv(pg, args.path, args.source)
            except Exception as exc:
                print(f"Could not load burden data: {exc}"); return 1
        print(f"Loaded {res}")
    elif args.cmd == "usage":
        from research_agent.db import get_conn

        with get_conn() as pg:
            row = pg.execute("SELECT content FROM run_notes WHERE run_id=%s AND agent='usage'",
                             (args.run_id,)).fetchone()
        if not row:
            print("No usage breakdown for that run (runs before this feature only have totals)."); return 1
        print(format_usage(row["content"]["by_step"], row["content"]["total"]))
    elif args.cmd == "runs":
        from research_agent.db import get_conn

        with get_conn() as pg:
            rows = pg.execute(
                """SELECT r.run_id, r.status, r.created_at, left(r.question, 70) q,
                          (SELECT count(*) FROM run_papers p WHERE p.run_id = r.run_id) n_papers,
                          (SELECT string_agg(agent, ',' ORDER BY ts) FROM run_notes n WHERE n.run_id = r.run_id) done
                   FROM runs r ORDER BY r.created_at DESC LIMIT 10""").fetchall()
        for r in rows:
            print(f"{r['run_id']}  {r['status']:<7} {r['created_at']:%Y-%m-%d %H:%M}  papers={r['n_papers']:<3} "
                  f"done=[{r['done'] or ''}]  {r['q']}")
    elif args.cmd == "report":
        from research_agent.db import get_conn

        with get_conn() as pg:
            row = pg.execute("SELECT status, report_md, error FROM runs WHERE run_id=%s", (args.run_id,)).fetchone()
        print(row["report_md"] if row and row["report_md"] else row)
    return 0


def draft_cmd(args, ap) -> int:
    from research_agent.agents import draft
    from research_agent.exports import export_draft

    if args.export:
        try:
            data, name, _ = export_draft(args.export, args.format)
        except Exception as exc:
            print(f"Export failed: {exc}"); return 1
        Path(args.out or name).write_bytes(data)
        print(f"saved {args.out or name} ({len(data):,} bytes)")
        return 0
    if not args.run_id:
        ap.error("give the run_id to draft from (or --export DRAFT_ID)")
    if args.list:
        for d in draft.list_drafts(args.run_id):
            print(f"#{d['id']:<5} {d['kind']:<9} {d['status']:<8} {d['created_at']:%Y-%m-%d %H:%M}  "
                  f"{d['title'] or d['direction'][:70]}")
        return 0
    if args.directions:
        from research_agent.runstate import RunContext

        ctx = RunContext.attach(args.run_id, llm_factory=lambda strong=False: None)
        try:
            items = draft.directions(ctx)
        finally:
            ctx.close()
        for it in items:
            print(f"{(it['id'] or '-'):<5} {it['kind']:<20} {it['label']}")
        if not items:
            print("No gaps recorded for this run; give your own --direction.")
        return 0
    about = [a for a in args.about.replace(";", ",").split(",") if a.strip()]
    if not args.direction and not about:
        print("Tip: --direction \"...\" or --about G2 says what the draft should argue for; "
              "without it the draft builds on the run's main gap.")
    did = draft.start(args.run_id, args.kind, args.direction, about)
    print(f"Drafting {args.kind} #{did} (planning, then one section at a time; this takes a few minutes)...")
    try:
        out = draft.run_draft(did, provider=args.provider)
    except Exception as exc:
        print(f"\n✖ Draft failed: {exc}"); return 1
    a = out["meta"]["audit"]
    print(f"\n{out['content_md']}\n")
    print(f"draft #{did}: {len(out['meta']['cited_papers'])} papers cited, "
          f"{len(out['meta']['new_claims'])} new counts, {a['unverified_numbers']} numbers marked [unverified], "
          f"{len(a['removed_paper_citations'])} citations removed")
    data, name, _ = export_draft(did, args.format)
    Path(args.out or name).write_bytes(data)
    print(f"saved {args.out or name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
