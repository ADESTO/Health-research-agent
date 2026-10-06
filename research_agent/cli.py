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
    p_ask.add_argument("--systematic", action="store_true",
                       help="find papers by a fixed search, screen every hit against the protocol, and analyse all eligible ones (a random sample past SYSTEMATIC_MAX_ANALYSED) instead of letting an agent choose")
    p_map = sub.add_parser("map", help="build a Research Opportunity Map for a question")
    p_map.add_argument("question", nargs="?", help="omit when using --resume")
    p_map.add_argument("--resume", metavar="RUN_ID", help="continue a failed map run")
    p_map.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_map.add_argument("--out", help="write the map to this .md file")
    p_map.add_argument("--systematic", action="store_true",
                       help="find papers by a fixed search, screen every hit against the protocol, and analyse all eligible ones (a random sample past SYSTEMATIC_MAX_ANALYSED) instead of letting an agent choose")
    p_rep = sub.add_parser("report"); p_rep.add_argument("run_id")
    p_gl = sub.add_parser("guideline-add", help="add a clinical guideline to the reference library, from PMC "
                                             "or a PDF/text file; its recommendations are recorded word for word")
    src = p_gl.add_mutually_exclusive_group(required=True)
    src.add_argument("--pmc", help="PMC id of an open-access guideline paper, e.g. PMC9876543")
    src.add_argument("--file", help="a PDF, text or Markdown file of the guideline")
    p_gl.add_argument("--issuer", required=True, help='who issued it, e.g. EULAR, ACR, "Kenya Ministry of Health"')
    p_gl.add_argument("--condition", action="append", required=True,
                      help='what it covers; repeat for several, e.g. --condition "rheumatoid arthritis"')
    p_gl.add_argument("--title", help="title (taken from PMC when omitted)")
    p_gl.add_argument("--year", type=int, help="year (taken from PMC when omitted)")
    p_gl.add_argument("--region", default="international", help="international, Kenya, Uganda, East Africa...")
    p_gl.add_argument("--url", default="", help="where it was obtained")
    p_gl.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_gls = sub.add_parser("guidelines", help="list the guideline library, or show one guideline's recommendations")
    p_gls.add_argument("guideline_id", nargs="?", type=int)
    p_gls.add_argument("--condition")
    p_glr = sub.add_parser("guideline-remove", help="remove a guideline and its recommendations")
    p_glr.add_argument("guideline_id", type=int)
    p_glc = sub.add_parser("guideline-compare", help="set a finished run's studies against the library's "
                                                 "recommendations for a condition (added to drafts made after)")
    p_glc.add_argument("run_id"); p_glc.add_argument("--condition")
    p_glc.add_argument("--guideline", type=int, action="append", help="only these guideline ids")
    p_meta = sub.add_parser("meta", help="meta-analysis of reported prevalence: read a run's papers for one "
                                         "condition, check every number against a quote, pool in code")
    p_meta.add_argument("run_id")
    p_meta.add_argument("condition", help='e.g. "arthritis", "rheumatoid arthritis", "knee osteoarthritis"')
    p_meta.add_argument("--limit", type=int, help="how many papers to read (default and maximum 150)")
    p_meta.add_argument("--refresh", action="store_true", help="read papers already read for this condition again")
    p_meta.add_argument("--out", default=".", help="folder for the forest plot, the table and the data (CSV)")
    p_meta.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_fp = sub.add_parser("fieldpass", help="fill ONE field from full text in the papers that left it blank")
    p_fp.add_argument("run_id")
    p_fp.add_argument("field", help="e.g. q_validation_split, q_forecast_horizon, validation_level")
    p_fp.add_argument("--limit", type=int, help="how many papers (default and maximum 60)")
    p_fp.add_argument("--values", help="define a NEW, finer field with these comma-separated categories, then "
                                       "fill it (for when the existing field merges things the question "
                                       "separates); the old field is left as it is")
    p_fp.add_argument("--definition", default="", help="with --values: what the new field means")
    p_fp.add_argument("--list", dest="as_list", action="store_true",
                      help="with --values: make it a list field rather than one category per paper")
    p_fp.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_co = sub.add_parser("cohort", help="set the scope a run's numbers are counted over (applied to every "
                                        "count, so the scope need not be restated in each test)")
    p_co.add_argument("run_id")
    p_co.add_argument("--include", action="append", default=[], metavar="FIELD=v1,v2",
                      help="a paper must match this to be counted, e.g. geography=kenya,uganda,east africa. "
                           "Repeatable: every --include must hold.")
    p_co.add_argument("--exclude", action="append", default=[], metavar="FIELD=v1,v2",
                      help="a paper matching this is out of scope, e.g. geography=india,china. Repeatable.")
    p_co.add_argument("--unstated", choices=["keep", "exclude"], default="keep",
                      help="papers silent on an --include field: keep them (default; silence is about how a "
                           "paper was read) or exclude them")
    p_co.add_argument("--note", default="", help="why this is the scope, for the report")
    p_co.add_argument("--clear", action="store_true", help="count every shortlisted paper again")
    p_pr = sub.add_parser("precedent", help="for each of a run's opportunities, search the whole corpus "
                                           "for studies that already do it")
    p_pr.add_argument("run_id")
    p_pr.add_argument("--item", help="just one opportunity, e.g. G2 or N1")
    p_pr.add_argument("--candidates", type=int, default=40, help="papers to pull per opportunity")
    p_cb = sub.add_parser("corroborate", help="which papers back each other up: every reported association set "
                                             "beside every other about the same driver and outcome")
    p_cb.add_argument("run_id")
    p_cb.add_argument("--driver", help="just one driver, e.g. testosterone therapy")
    p_cb.add_argument("--all", dest="show_all", action="store_true",
                      help="also list findings only one paper reports")
    p_rr = sub.add_parser("reread", help="read a finished run's papers in full again and re-code their records")
    p_rr.add_argument("run_id")
    p_rr.add_argument("--limit", type=int, help="how many papers (default MAX_FULLTEXT)")
    p_rr.add_argument("--force", action="store_true",
                      help="read again even papers already read in full (after a protocol or prompt change)")
    p_rr.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_chat = sub.add_parser("chat", help="ask follow-up questions about a finished run")
    p_chat.add_argument("run_id")
    p_chat.add_argument("--question", "-q", help="ask one question and exit")
    p_chat.add_argument("--about", help="the item the question is about, e.g. C512 or G2")
    p_chat.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_exp = sub.add_parser("export", help="export a finished run (report, references, data, review record)")
    p_exp.add_argument("run_id")
    p_exp.add_argument("--format", "-f", default="docx",
                       help="md, docx, html, bib, ris, csv, xlsx, protocol, screening, burden, burden_chart, "
                            "opportunities, or package (everything above as one zip, with a manifest saying "
                            "what each file is and what it cannot tell you)")
    p_exp.add_argument("--out", "-o", help="file to write (default: a name based on the run and format)")
    p_dr = sub.add_parser("draft", help="draft a research proposal or review manuscript from a finished run")
    p_dr.add_argument("run_id", nargs="?", help="the run to build on (not needed with --export)")
    p_dr.add_argument("--type", "-t", choices=["proposal", "review"], default="proposal", dest="kind")
    p_dr.add_argument("--direction", "-d", default="", help="the gap or direction to argue for, in your words")
    p_dr.add_argument("--about", default="", help="map items it builds on, e.g. G2 or G2,D1")
    p_dr.add_argument("--style", choices=["author-year", "numbered"], default="author-year",
                      help="citation style: author-year (Smith et al., 2021) or numbered (Vancouver)")
    p_dr.add_argument("--directions", action="store_true", help="list the run's gaps and directions, then exit")
    p_dr.add_argument("--list", action="store_true", help="list the run's drafts, then exit")
    p_dr.add_argument("--export", metavar="DRAFT_ID", type=int, help="export an existing draft instead")
    p_dr.add_argument("--format", "-f", default="docx", choices=["md", "docx", "html", "bib", "ris"])
    p_dr.add_argument("--out", "-o", help="file to write")
    p_dr.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    p_rs = sub.add_parser("research", help="an open-ended researcher working on a finished run")
    rs = p_rs.add_subparsers(dest="rcmd", required=True)
    r_start = rs.add_parser("start", help="start a researcher (its cycles run on the worker)")
    r_start.add_argument("run_id")
    r_start.add_argument("--goal", "-g", required=True)
    r_start.add_argument("--scope", default="")
    r_start.add_argument("--out-of-bounds", default="", help="comma-separated topics it must not pursue")
    r_start.add_argument("--max-cycles", type=int, default=20)
    r_start.add_argument("--daily-tokens", type=int, default=2_000_000)
    r_start.add_argument("--total-tokens", type=int, default=10_000_000)
    r_start.add_argument("--split", choices=["random", "year"], default="random")
    r_start.add_argument("--split-year", type=int)
    r_start.add_argument("--provider", choices=["anthropic", "groq", "deepseek"])
    r_start.add_argument("--here", action="store_true", help="run the cycles in this terminal instead of the worker")
    r_cyc = rs.add_parser("cycle", help="run cycles now, in this terminal")
    r_cyc.add_argument("researcher_id", type=int)
    r_cyc.add_argument("-n", type=int, default=1)
    r_st = rs.add_parser("status", help="agenda, findings and notebook")
    r_st.add_argument("researcher_id", type=int)
    for name in ("pause", "resume", "stop"):
        rs.add_parser(name).add_argument("researcher_id", type=int)
    r_bud = rs.add_parser("budget", help="give a researcher more cycles or tokens, and set it going again")
    r_bud.add_argument("researcher_id", type=int)
    r_bud.add_argument("--add-cycles", type=int, help="raise the cycle limit by this many")
    r_bud.add_argument("--max-cycles", type=int, help="set the cycle limit to this")
    r_bud.add_argument("--add-tokens", type=int, help="raise the total token budget by this many")
    r_bud.add_argument("--total-tokens", type=int, help="set the total token budget to this")
    r_bud.add_argument("--daily-tokens", type=int, help="set the per-day token budget")
    r_ap = rs.add_parser("approve", help="approve (or --decline) a question waiting for approval")
    r_ap.add_argument("researcher_id", type=int)
    r_ap.add_argument("agenda_id", type=int)
    r_ap.add_argument("--decline", action="store_true")
    p_w = sub.add_parser("worker", help="work the job queue (runs, maps, follow-ups started from the web page)")
    p_w.add_argument("--concurrency", "-c", type=int, default=1)
    sub.add_parser("burden-fetch", help="download WHO estimates of malaria cases and deaths by country")
    p_bi = sub.add_parser("burden-import", help="import burden estimates from a CSV (iso3 or country, year, cases)")
    p_bi.add_argument("path"); p_bi.add_argument("--source", default="CSV")
    p_use = sub.add_parser("usage", help="tokens (and cost) per step for a finished run")
    p_use.add_argument("run_id")
    sub.add_parser("runs", help="list recent runs (to find a run_id to resume)")
    p_inv = sub.add_parser("invite", help="hosted use: add a user and print their invite link")
    p_inv.add_argument("name"); p_inv.add_argument("--email"); p_inv.add_argument("--admin", action="store_true")
    sub.add_parser("users", help="hosted use: list users")
    p_relink = sub.add_parser("relink", help="hosted use: replace a user's link (lost or leaked); the old one stops")
    p_relink.add_argument("user", help="user id, name or email")
    p_deact = sub.add_parser("deactivate", help="hosted use: switch a user off (their runs are kept)")
    p_deact.add_argument("user", help="user id, name or email")
    p_share = sub.add_parser("share", help="hosted use: let every user read a run (e.g. the sample run)")
    p_share.add_argument("run_id"); p_share.add_argument("--off", action="store_true")
    p_own = sub.add_parser("assign", help="hosted use: give a run to a user (e.g. runs made before users existed)")
    p_own.add_argument("run_id"); p_own.add_argument("user", help="user id, name or email")
    args = ap.parse_args(argv)

    if args.cmd in ("invite", "users", "relink", "deactivate", "share", "assign"):
        _users_cmd(args)
        return 0
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
                               resume=args.resume, search="systematic" if args.systematic else None)
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
            res = run_map(args.question, provider=args.provider, on_event=_print_event, resume=args.resume,
                          search="systematic" if args.systematic else None)
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
    elif args.cmd == "research":
        return research_cmd(args)
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
    elif args.cmd == "meta":
        return meta_cmd(args)
    elif args.cmd in ("guideline-add", "guidelines", "guideline-remove", "guideline-compare"):
        return guidelines_cmd(args)
    elif args.cmd == "fieldpass":
        return fieldpass_cmd(args)
    elif args.cmd == "reread":
        return reread_cmd(args)
    elif args.cmd == "cohort":
        return cohort_cmd(args)
    elif args.cmd == "precedent":
        return precedent_cmd(args)
    elif args.cmd == "corroborate":
        return corroborate_cmd(args)
    elif args.cmd == "report":
        from research_agent.db import get_conn

        with get_conn() as pg:
            row = pg.execute("SELECT status, report_md, error FROM runs WHERE run_id=%s", (args.run_id,)).fetchone()
        print(row["report_md"] if row and row["report_md"] else row)
    return 0


def guidelines_cmd(args) -> int:
    from research_agent.db import get_conn, init_schema
    from research_agent.tools import guidelines as GL

    init_schema()
    if args.cmd == "guideline-compare":
        from research_agent.agents.report import _cite
        from research_agent.runstate import RunContext

        ctx = RunContext.attach(args.run_id)
        try:
            res = GL.compare(ctx, args.condition, args.guideline)
        finally:
            ctx.close()
        if not res["recommendations"]:
            print("No recommendations in the library for this condition. Add a guideline first (guideline-add).")
            return 1
        n = res["papers_stating_interventions"]
        print(f"{n} of the run's {res['papers_counted']} papers state which treatments they used or studied.")
        for r in res["recommendations"]:
            what = ", ".join(r["interventions"]) or "no treatment named"
            print(f"- {r['guideline']} {r['number']}: [{r['strength_as_written'] or 'strength not stated'}] {what}: "
                  f"{r['papers']} of {n} papers" + (f" ({', '.join(_cite(p) for p in r['paper_ids'][:4])})"
                                                    if r["papers"] else ""))
        print("Drafts made from this run now include the comparison. " + res["note"])
        return 0
    with get_conn() as pg:
        if args.cmd == "guidelines" and args.guideline_id:
            for r in GL.recommendations(pg, [args.guideline_id]):
                grade = " / ".join(x for x in (r["strength_as_written"], r["evidence_as_written"]) if x)
                print(f"{r['seq']:>3}. {('(' + r['number'] + ') ') if r['number'] else ''}{r['text']}")
                print(f"     {r['action']}; {grade or 'grade not stated'}; about: {', '.join(r['interventions']) or '-'}")
            return 0
        if args.cmd == "guidelines":
            rows = GL.list_guidelines(pg, args.condition)
            if not rows:
                print("The library is empty. Add one with guideline-add --pmc PMC... or --file guideline.pdf")
            for g in rows:
                print(f"{g['id']:>3}  {GL.label(g)}  [{', '.join(g['conditions'])}; {g['region']}]  "
                      f"{g['n_recs']} recommendations" + (f"  ({g['chunks_failed']} parts unread)" if g["chunks_failed"] else ""))
            return 0
        if args.cmd == "guideline-remove":
            n = pg.execute("DELETE FROM guidelines WHERE id=%s RETURNING id", (args.guideline_id,)).fetchall()
            print("removed" if n else "no such guideline")
            return 0
        # guideline-add
        from research_agent.llm import get_llm

        if args.pmc:
            meta, sections = GL.text_from_pmc(args.pmc)
            source, licence = args.pmc.upper(), meta["licence"]
            title, year = args.title or meta["title"], args.year or meta["year"]
        else:
            sections = GL.text_from_file(args.file)
            source, licence = args.file.rsplit("/", 1)[-1], ""
            title, year = args.title or source, args.year
        print(f"Reading {title} ({sum(len(s['text']) for s in sections):,} characters) for its recommendations...")
        res = GL.add_guideline(pg, lambda strong=False, step=None: get_llm(args.provider, strong=strong, step=step),
                               title, args.issuer, year, args.condition, args.region, sections, source, licence,
                               args.url)
        if "error" in res:
            print(res["error"]); return 1
        print(f"Added guideline {res['guideline_id']}: {res['recommendations']} recommendations recorded word for word; "
              f"{res['dropped_not_found_in_text']} dropped because they were not found in the text"
              + (f"; {res['chunks_failed']} parts could not be read (run again or check the file)" if res["chunks_failed"] else ""))
        print(f"Check them with: python -m research_agent.cli guidelines {res['guideline_id']}")
        return 0


def meta_cmd(args) -> int:
    """Prevalence meta-analysis for one condition over a finished run's papers."""
    import csv
    from pathlib import Path

    from research_agent.runstate import RunContext
    from research_agent.tools import meta

    ctx = RunContext.attach(args.run_id, provider=args.provider)
    try:
        res = meta.prevalence_pass(ctx, args.condition, limit=args.limit, refresh=args.refresh)
        if "error" in res:
            print(res["error"]); return 1
        print(f"{res['papers_read']} papers read ({res['read_in_full']} in full); "
              f"{res['papers_reporting_prevalence']} report a prevalence of {res['condition']}; "
              f"{res['studies_pooled']} study estimates pooled.")
        for a in res["analyses"]:
            o = a["overall"]
            if o:
                pi = f", prediction interval {100 * o['pi'][0]:.1f} to {100 * o['pi'][1]:.1f}%" if o["pi"] else ""
                print(f"  {a['condition']}: {100 * o['pooled']:.1f}% (95% CI {100 * o['ci'][0]:.1f} to "
                      f"{100 * o['ci'][1]:.1f}%{pi}), {o['k']} studies, {o['n_total']:,} people, I2 {100 * o['i2']:.0f}%")
            else:
                print(f"  {a['condition']}: one study, not pooled")
        if res["not_pooled"]:
            print(f"  {len(res['not_pooled'])} estimates not pooled (numbers could not be checked); see the CSV")
        if res["possible_duplicates"]:
            print(f"  possible duplicate samples: {res['possible_duplicates']}")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = "meta_" + meta._note_key(args.condition).removeprefix("meta:")
        (out / f"{stem}.md").write_text("\n".join(meta.markdown(ctx, args.condition)))
        header, rows = meta.export_rows(ctx, args.condition)
        with open(out / f"{stem}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        png = meta.forest_plot(ctx, args.condition, str(out / f"{stem}_forest.png"))
        print(f"Wrote {out / (stem + '.md')}, {out / (stem + '.csv')}" + (f" and {png}" if png else ""))
        return 0
    finally:
        ctx.close()


def fieldpass_cmd(args) -> int:
    """Fill one field from full text in the papers that left it blank, and say what changed."""
    from research_agent.runstate import RunContext
    from research_agent.tools.fieldpass import field_pass

    vals = [v.strip() for v in (args.values or "").split(",") if v.strip()]
    if vals and not args.definition:
        print("--values needs --definition: say what the field means so papers can be coded by it"); return 1
    ctx = RunContext.attach(args.run_id, provider=args.provider)
    try:
        res = field_pass(ctx, args.field, limit=args.limit, values=vals or None,
                         definition=args.definition, kind="list" if args.as_list else "enum")
        if "error" in res:
            print(res["error"]); return 1
        if res.get("added_field"):
            a = res["added_field"]
            print(f"Added {a['field']} ({a['type']}): {', '.join(a['values'])}")
            for n in a.get("notes") or []:
                print(f"  note: {n}")
        args.field = res.get("added_field", {}).get("field", args.field)
        if res.get("note"):
            print(res["note"]); return 0
        print(f"{args.field}: {res['blank_before']} papers left it blank; read {res['read_now']} in full.")
        print(f"  filled: {res['filled']}   still not stated: {res['still_not_stated']}   "
              f"dropped for want of a quote: {res['dropped_without_a_quote']}")
        if res["no_full_text_available"]:
            print(f"  no full text available: {len(res['no_full_text_available'])} "
                  f"({', '.join(res['no_full_text_available'][:5])})")
        for value, n in (res["values_found"] or {}).items():
            print(f"    {n:>3}  {value}")
        for ex in res["examples"]:
            print(f"\n  {ex['paper_id']} -> {ex['value']}\n    \u201c{ex['quote']}\u201d")
        print("\nThe report and its claims are unchanged. Count the field again (or re-run the researcher) "
              "to use the fuller records.")
    finally:
        ctx.close()
    return 0


def _cohort_conditions(specs: list[str]) -> tuple[list[dict], str | None]:
    """'geography=kenya,uganda' -> {"field": "geography", "any_of": [...]}; 'year=2015-2025' -> min/max."""
    import re

    out = []
    for spec in specs:
        field, _, values = str(spec).partition("=")
        field, values = field.strip(), values.strip()
        if not field or not values:
            return [], f"could not read {spec!r}: write it as FIELD=value1,value2"
        m = re.fullmatch(r"(\d{4})?-(\d{4})?", values)
        if field == "year" and m and (m.group(1) or m.group(2)):
            out.append({"field": "year", **({"min": int(m.group(1))} if m.group(1) else {}),
                        **({"max": int(m.group(2))} if m.group(2) else {})})
            continue
        out.append({"field": field, "any_of": [v.strip() for v in values.split(",") if v.strip()]})
    return out, None


def corroborate_cmd(args) -> int:
    """Print which findings independent papers back, dispute or leave alone."""
    from research_agent.runstate import RunContext
    from research_agent.tools.corroboration import MEANS, corroborate

    ctx = RunContext.attach(args.run_id, llm_factory=lambda strong=False, step=None: None)
    try:
        res = corroborate(ctx, driver=args.driver)
        if not res["findings"]:
            print(res.get("note") or "No findings to compare."); return 0
        print(f"{res['statements']} statements from the papers, in {len(res['findings'])} findings: "
              + ", ".join(f"{n} {v.replace('_', ' ')}" for v, n in res["summary"].items()))
        for f in res["findings"]:
            if f["verdict"] == "not_addressed_elsewhere" and not args.show_all:
                continue
            dirs = ", ".join(f"{d}: {v['independent_sources']} source(s) {v['papers']}"
                             for d, v in f["by_direction"].items())
            print(f"\n{f['driver']} -> {f['outcome']}   [{f['verdict'].replace('_', ' ')}]")
            print(f"   {dirs}")
            if f.get("what_separates_the_sides"):
                print("   sides differ in: " + "; ".join(x["attribute"] for x in f["what_separates_the_sides"][:3]))
            if len(f["driver_wordings"]) > 1:
                print("   treated as one driver: " + ", ".join(f["driver_wordings"]))
        if not args.show_all and res["summary"].get("not_addressed_elsewhere"):
            print(f"\n{res['summary']['not_addressed_elsewhere']} findings are reported by one paper only "
                  "(--all to list them). " + MEANS["not_addressed_elsewhere"].capitalize() + ".")
    finally:
        ctx.close()
    return 0


def precedent_cmd(args) -> int:
    """Say, for each opportunity, whether the corpus already holds a study that does it."""
    from research_agent.runstate import RunContext
    from research_agent.tools import precedent as PR

    ctx = RunContext.attach(args.run_id, llm_factory=lambda strong=False, step=None: None)
    try:
        res = PR.check(ctx, item_id=args.item, limit=args.candidates)
        if "error" in res:
            print(res["error"]); return 1
        print(f"Checked {res['checked']} opportunities against the whole corpus: "
              + ", ".join(f"{n} {lv}" for lv, n in res["by_verdict"].items()))
        for o in res["opportunities"]:
            print(f"\n{o['item_id']:<4} {o['verdict']:<9} {o['label']}")
            if o["counts"].get("direct"):
                print(f"     {o['counts']['direct']} paper(s) already do this")
        print("\nThe papers behind each verdict are on the opportunity itself "
              "(`export opportunities`, or the Map tab).")
    finally:
        ctx.close()
    return 0


def cohort_cmd(args) -> int:
    """Set, clear or show the scope a run's numbers are counted over."""
    from research_agent.runstate import RunContext
    from research_agent.tools import cohort as CO

    ctx = RunContext.attach(args.run_id, llm_factory=lambda strong=False, step=None: None)
    try:
        if args.clear:
            print(CO.clear(ctx)["note"]); return 0
        include, err = _cohort_conditions(args.include)
        exclude, err2 = _cohort_conditions(args.exclude) if not err else ([], None)
        if err or err2:
            print(err or err2); return 1
        if not include and not exclude:                     # no conditions given: show what is stored
            s = CO.summary(ctx)
            print(f"Cohort: {s.get('conditions') or 'every shortlisted paper'}")
            print(f"Counted: {s['counted']} papers")
            if s.get("cohort") is not None or "excluded_out_of_scope" in s:
                print(f"Excluded as out of scope: {s.get('excluded_out_of_scope', 0)}")
                print(f"Silent on a cohort field: {s.get('unstated_on_a_cohort_field', 0)} "
                      f"({s.get('unstated_are', '')})")
            if s.get("warning"):
                print("\n" + s["warning"])
            return 0
        res = CO.set_cohort(ctx, include=include, exclude=exclude, unstated=args.unstated, note=args.note)
        if "error" in res:
            print(res["error"]); return 1
        print(f"Cohort: {res['conditions']}")
        print(f"Counted: {res['counted']} papers; out of scope: {res['excluded_out_of_scope']}; "
              f"silent on a cohort field: {res['unstated_on_a_cohort_field']} ({res['unstated_are']})")
        if res.get("warning"):
            print("\n" + res["warning"])
        print("\nEvery count from now on is over this cohort: claims, the researcher's tests, the map and the "
              "exports. The report already written is unchanged; re-run `--redo synthesis` to rewrite it.")
    finally:
        ctx.close()
    return 0


def reread_cmd(args) -> int:
    """Read a finished run's papers in full again and re-code them. Abstract-only papers often leave a
    question-specific field empty simply because the abstract has no sentence to quote for it; this fills
    those in. The report is left as it is: counts change only where the papers themselves say so."""
    from research_agent.runstate import RunContext
    from research_agent.tools.extraction import extract_papers, extraction_coverage

    ctx = RunContext.attach(args.run_id, provider=args.provider)
    try:
        before = extraction_coverage(ctx)
        print(f"{before['extracted']} papers extracted; read in full: {before['by_source'].get('fulltext', 0)}, "
              f"from the abstract only: {before['by_source'].get('abstract', 0)}")
        res = extract_papers(ctx, depth="fulltext", limit=args.limit, force=args.force)
        after = extraction_coverage(ctx)
        print(f"Read now: {res['newly_extracted']}; failed: {len(res['failed'])}; "
              f"no full text available: {sum(1 for k, v in res['fulltext_status'].items() if k != 'ok')}")
        moved = [(f, before["field_stated_rate"].get(f, 0), r) for f, r in after["field_stated_rate"].items()
                 if abs(r - before["field_stated_rate"].get(f, 0)) >= 0.01]
        if moved:
            print("\nFields that changed (share of papers stating them):")
            for f, b, a in sorted(moved, key=lambda x: -(x[2] - x[1])):
                print(f"  {f:<34} {b:>5.0%} -> {a:>5.0%}")
        else:
            print("\nNo field changed: the papers read again say no more than their abstracts did.")
        print("\nThe report and its claims are unchanged. Re-run the researcher, or ask a follow-up, to use "
              "the fuller records.")
    finally:
        ctx.close()
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
    did = draft.start(args.run_id, args.kind, args.direction, about, args.style)
    print(f"Drafting {args.kind} #{did} (planning, then one section at a time; this takes a few minutes)...")
    try:
        out = draft.run_draft(did, provider=args.provider)
    except Exception as exc:
        print(f"\n✖ Draft failed: {exc}"); return 1
    a = out["meta"]["audit"]
    print(f"\n{out['content_md']}\n")
    print(f"draft #{did}: {len(out['meta']['cited_in_text'])} studies cited in the text "
          f"({len(out['meta']['cited_papers'])} in the references), "
          f"{len(out['meta']['new_claims'])} new counts, {a['unverified_numbers']} numbers marked [unverified], "
          f"{len(a['removed_paper_citations'])} citations removed")
    data, name, _ = export_draft(did, args.format)
    Path(args.out or name).write_bytes(data)
    print(f"saved {args.out or name}")
    return 0


def research_cmd(args) -> int:
    from research_agent.research import researcher as R

    if args.rcmd == "start":
        kw = dict(scope=args.scope, out_of_bounds=[x for x in args.out_of_bounds.split(",") if x.strip()],
                  max_cycles=args.max_cycles, daily_tokens=args.daily_tokens, total_tokens=args.total_tokens,
                  split=args.split, split_year=args.split_year, provider=args.provider)
        if args.here:
            rid = R.create(args.run_id, args.goal, **kw)
            print(f"researcher #{rid} started; running its cycles here (Ctrl+C to stop, it can be resumed)")
            return _research_loop(rid, args.max_cycles)
        out = R.start_in_background(args.run_id, args.goal, **kw)
        print(f"researcher #{out['researcher_id']} started; its cycles run on the worker "
              "(python -m research_agent.cli worker). Follow it with: research status "
              f"{out['researcher_id']}")
        return 0
    if args.rcmd == "cycle":
        return _research_loop(args.researcher_id, args.n)
    if args.rcmd in ("pause", "resume", "stop"):
        R.set_status(args.researcher_id, {"pause": "paused", "resume": "active", "stop": "stopped"}[args.rcmd],
                     f"{args.rcmd}d from the terminal")
        print(f"researcher #{args.researcher_id}: {args.rcmd}d")
        return 0
    if args.rcmd == "budget":
        res = R.set_budget(args.researcher_id, max_cycles=args.max_cycles, add_cycles=args.add_cycles,
                           total_tokens=args.total_tokens, add_tokens=args.add_tokens,
                           daily_tokens=args.daily_tokens)
        if "error" in res:
            print(res["error"]); return 1
        print(f"researcher #{res['researcher_id']}: {res['cycles_done']}/{res['max_cycles']} cycles, "
              f"{res['tokens_used']:,}/{res['total_tokens']:,} tokens")
        print("It is working again; follow it with research status." if res["resumed"]
              else res.get("note", "Limits raised. It was already running, so nothing else changed."))
        return 0
    if args.rcmd == "approve":
        R.decide_question(args.researcher_id, args.agenda_id, not args.decline)
        print("declined" if args.decline else "approved")
        return 0
    g = R.get(args.researcher_id)
    if not g:
        print("no such researcher"); return 1
    r = g["researcher"]
    print(f"#{r['id']} {r['status']}{' (' + r['status_note'] + ')' if r['status_note'] else ''}  "
          f"cycles {r['cycles_done']}/{r['max_cycles']}  tokens {r['tokens_used']:,}/{r['total_tokens']:,}  "
          f"today {g['tokens_today']:,}/{r['daily_tokens']:,}  patterns tested {g['tests']}")
    print(f"goal: {r['charter']['goal']}\n\nAgenda:")
    for a in g["agenda"]:
        print(f"  [{a['status']:<14}] Q{a['id']} {a['question']}" + (f"  ({a['status_note']})" if a['status_note'] else ""))
    print("\nFindings (the researcher's own; graded by code):")
    for f in g["findings"]:
        print(f"  [{f['grade']:<11}] F{f['id']} {f['statement']}")
        for reason in (f["reasons"] or [])[:4]:
            print(f"      - {reason}")
    return 0


def _research_loop(rid: int, n: int) -> int:
    from research_agent.research import researcher as R

    for _ in range(max(1, n)):
        try:
            out = R.run_cycle(rid)
        except KeyboardInterrupt:
            print("stopped; resume with: research cycle", rid); return 1
        print(json.dumps(out, default=str))
        if any(k in out for k in ("skipped", "finished", "paused", "waiting")):
            break
    return 0


def _find_user(pg, key: str) -> dict:
    rows = pg.execute("SELECT id, name, email FROM users WHERE id::text=%s OR lower(name)=lower(%s) "
                      "OR lower(email)=lower(%s)", (key, key, key)).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"{'no user' if not rows else 'more than one user'} matches {key!r}; use the id from `users`")
    return rows[0]


def _users_cmd(args) -> None:
    from research_agent import auth
    from research_agent.db import get_conn, init_schema

    init_schema()
    with get_conn() as pg:
        if args.cmd == "invite":
            user, token = auth.create_user(pg, args.name, args.email, admin=args.admin)
            print(f"{user['name']}{' (admin)' if user['is_admin'] else ''}: {user['id']}")
            print(f"Invite link (shown once; send it privately): {auth.invite_link(token)}")
        elif args.cmd == "users":
            for u in pg.execute("SELECT u.id, u.name, u.email, u.is_admin, u.active, u.last_seen, "
                                "(SELECT count(*) FROM runs r WHERE r.owner_id = u.id) AS runs "
                                "FROM users u ORDER BY u.created_at").fetchall():
                flags = ", ".join(f for f, on in (("admin", u["is_admin"]), ("inactive", not u["active"])) if on)
                seen = u["last_seen"].strftime("%Y-%m-%d %H:%M") if u["last_seen"] else "never"
                print(f"{u['id']}  {u['name']:<24} {u['email'] or '':<30} runs: {u['runs']:<4} last seen: {seen}"
                      + (f"  [{flags}]" if flags else ""))
        elif args.cmd == "relink":
            u = _find_user(pg, args.user)
            print(f"New link for {u['name']} (the old one no longer works): "
                  f"{auth.invite_link(auth.new_token(pg, u['id']))}")
        elif args.cmd == "deactivate":
            u = _find_user(pg, args.user)
            pg.execute("UPDATE users SET active=false WHERE id=%s", (u["id"],))
            print(f"{u['name']} can no longer sign in; their runs are kept.")
        elif args.cmd == "share":
            n = pg.execute("UPDATE runs SET shared=%s WHERE run_id=%s RETURNING run_id",
                           (not args.off, args.run_id)).fetchall()
            print("no such run" if not n else f"run {args.run_id} is {'private' if args.off else 'shared with every user'}")
        elif args.cmd == "assign":
            u = _find_user(pg, args.user)
            n = pg.execute("UPDATE runs SET owner_id=%s WHERE run_id=%s RETURNING run_id", (u["id"], args.run_id)).fetchall()
            print("no such run" if not n else f"run {args.run_id} now belongs to {u['name']}")


if __name__ == "__main__":
    sys.exit(main())
