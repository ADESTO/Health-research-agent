"""Exports of a finished run.

    report      md     the report as written
                docx   Word document
                html   standalone web page (the PRISMA diagram renders in the browser)
    references  bib    BibTeX for every analysed paper
                ris    RIS (Zotero, EndNote, Mendeley)
    data        csv    one row per paper: extracted fields and their quotes
                xlsx   workbook: papers, reported results, associations, claims, screening log,
                       research versus burden, PRISMA counts
    review      protocol   protocol document with the PRISMA flow (markdown)
                screening  screening log (csv)
    burden      burden     research versus burden table (csv)
                burden_chart  chart (png)

Exports read the database only: no model calls, and the run is never changed.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re

FORMATS = {
    "md": ("text/markdown", "report.md"), "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "report.docx"),
    "html": ("text/html", "report.html"), "bib": ("application/x-bibtex", "references.bib"),
    "ris": ("application/x-research-info-systems", "references.ris"), "csv": ("text/csv", "extractions.csv"),
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "run_data.xlsx"),
    "protocol": ("text/markdown", "protocol_and_prisma.md"), "screening": ("text/csv", "screening_log.csv"),
    "burden": ("text/csv", "research_vs_burden.csv"), "burden_chart": ("image/png", "research_vs_burden.png"),
    "opportunities": ("text/csv", "research_opportunities.csv"),
    "package": ("application/zip", "evidence_package.zip"),
    "corroboration": ("text/csv", "corroboration.csv"),
}


def export(run_id: str, fmt: str) -> tuple[bytes, str, str]:
    """(content, filename, media type) for one export of a run."""
    from research_agent.db import init_schema
    from research_agent.runstate import RunContext

    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
    init_schema()
    ctx = RunContext.attach(run_id, llm_factory=lambda strong=False: None)
    try:
        media, name = FORMATS[fmt]
        data = globals()[f"_export_{fmt}"](ctx)
        if data is None:
            raise ValueError(f"nothing to export as {fmt} for this run")
        return (data if isinstance(data, bytes) else data.encode("utf-8")), f"{run_id[:8]}_{name}", media
    finally:
        ctx.close()


# ---------------------------------------------------------------- report
def _report(ctx) -> str:
    row = ctx.pg.execute("SELECT report_md FROM runs WHERE run_id=%s", (ctx.run_id,)).fetchone()
    if not row or not row["report_md"]:
        raise ValueError("this run has no report yet")
    return row["report_md"]


def _export_md(ctx):
    return _report(ctx)


def blocks(md: str) -> list[tuple]:
    """Markdown as blocks: ('h', level, text), ('p', text), ('ul'/'ol', [items]), ('table', header, rows),
    ('code', lang, text). Enough for the reports this system writes."""
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("```"):
            lang, body = line[3:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i]); i += 1
            out.append(("code", lang, "\n".join(body))); i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            out.append(("h", len(m.group(1)), m.group(2))); i += 1
            continue
        if re.match(r"^\|.*\|$", line):
            rows = []
            while i < len(lines) and re.match(r"^\|.*\|$", lines[i].rstrip()):
                if not re.match(r"^[\s|:-]+$", lines[i]):
                    rows.append([c.strip() for c in lines[i].rstrip()[1:-1].split("|")])
                i += 1
            out.append(("table", rows[0], rows[1:]) if rows else ("p", ""))
            continue
        m = re.match(r"^\s*([-*]|\d+[.)])\s+(.*)$", line)
        if m:
            kind = "ol" if m.group(1)[0].isdigit() else "ul"
            items = []
            while i < len(lines):
                mm = re.match(r"^\s*([-*]|\d+[.)])\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append(mm.group(2)); i += 1
            out.append((kind, items))
            continue
        if line.strip():
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|\s*([-*]|\d+[.)])\s)", lines[i]):
                para.append(lines[i].strip()); i += 1
            out.append(("p", " ".join(para)))
            continue
        i += 1
    return out


def _inline_html(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"\[([^\]]+)\]\((https?:[^)\s]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(^|\W)\*([^*]+)\*", r"\1<em>\2</em>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return re.sub(r"(?<![\"'>])(https?://[^\s<]+)", r'<a href="\1">\1</a>', t)


def _export_html(ctx):
    return render_html(_report(ctx), ctx.question[:120], f"Question: {ctx.question}")


def render_html(md: str, title: str, subtitle: str = "") -> str:
    body = []
    for b in blocks(md):
        if b[0] == "h":
            body.append(f"<h{b[1]}>{_inline_html(b[2])}</h{b[1]}>")
        elif b[0] == "p":
            body.append(f"<p>{_inline_html(b[1])}</p>")
        elif b[0] in ("ul", "ol"):
            body.append(f"<{b[0]}>" + "".join(f"<li>{_inline_html(x)}</li>" for x in b[1]) + f"</{b[0]}>")
        elif b[0] == "table":
            body.append("<table><thead><tr>" + "".join(f"<th>{_inline_html(c)}</th>" for c in b[1]) + "</tr></thead><tbody>"
                        + "".join("<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>" for r in b[2])
                        + "</tbody></table>")
        elif b[0] == "code":
            cls = ' class="mermaid"' if b[1] == "mermaid" else ""
            body.append(f"<pre{cls}>{html.escape(b[2])}</pre>")
    title = html.escape(title)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:24px 16px;color:#1c1a17}}
h2{{border-bottom:1px solid #e5e0d8;padding-bottom:4px;margin-top:28px}}table{{border-collapse:collapse;width:100%;
font-size:13px;display:block;overflow-x:auto}}td,th{{border:1px solid #e5e0d8;padding:6px 8px;text-align:left;
vertical-align:top}}code{{background:#f4f1ec;padding:1px 4px;border-radius:4px}}a{{color:#7a4f2c}}</style>
<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
mermaid.initialize({{startOnLoad:true}});</script></head><body>
<p style="color:#6b6560;font-size:13px">{html.escape(subtitle)}</p>
{chr(10).join(body)}
</body></html>"""


def _add_runs(par, text: str) -> None:
    """Bold, italic and code spans inside a Word paragraph."""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|(?<!\w)\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(re.sub(r"\[([^\]]+)\]\((https?:[^)\s]+)\)", r"\1 (\2)", part))


def _export_docx(ctx):
    return render_docx(_report(ctx), f"Question: {ctx.question}")


def render_docx(md: str, subtitle: str = "") -> bytes:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.size = Pt(10.5)
    if subtitle:
        doc.add_paragraph().add_run(subtitle).italic = True
    for b in blocks(md):
        if b[0] == "h":
            doc.add_heading(re.sub(r"[*`]", "", b[2]), level=min(b[1], 3) if b[1] > 1 else 0)
        elif b[0] == "p":
            _add_runs(doc.add_paragraph(), b[1])
        elif b[0] in ("ul", "ol"):
            for item in b[1]:
                _add_runs(doc.add_paragraph(style="List Bullet" if b[0] == "ul" else "List Number"), item)
        elif b[0] == "table":
            t = doc.add_table(rows=1, cols=len(b[1]))
            t.style = "Light Grid Accent 1"
            for c, text in zip(t.rows[0].cells, b[1]):
                _add_runs(c.paragraphs[0], text)
            for row in b[2]:
                cells = t.add_row().cells
                for c, text in zip(cells, row + [""] * (len(b[1]) - len(row))):
                    _add_runs(c.paragraphs[0], text)
        elif b[0] == "code" and b[1] != "mermaid":
            r = doc.add_paragraph().add_run(b[2]); r.font.name = "Consolas"; r.font.size = Pt(9)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- drafts (proposals, review manuscripts)
DRAFT_FORMATS = {"md": FORMATS["md"][0], "docx": FORMATS["docx"][0], "html": FORMATS["html"][0],
                 "bib": FORMATS["bib"][0], "ris": FORMATS["ris"][0]}


def export_draft(draft_id: int, fmt: str) -> tuple[bytes, str, str]:
    """(content, filename, media type) for a finished draft; bib and ris hold the papers it cites."""
    from research_agent.agents.draft import get_draft
    from research_agent.runstate import RunContext

    if fmt not in DRAFT_FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(DRAFT_FORMATS)}")
    d = get_draft(int(draft_id))
    if not d:
        raise ValueError(f"no draft {draft_id}")
    if d["status"] != "done" or not d["content_md"]:
        raise ValueError("this draft is not finished")
    slug = re.sub(r"[^a-z0-9]+", "_", (d["title"] or d["kind"]).lower()).strip("_")[:50] or d["kind"]
    name = f"{d['kind']}_{d['id']}_{slug}.{fmt}"
    md = d["content_md"]
    if fmt == "md":
        data = md
    elif fmt == "html":
        data = render_html(md, d["title"] or d["kind"])
    elif fmt == "docx":
        data = render_docx(md)
    else:
        ctx = RunContext.attach(str(d["run_id"]), llm_factory=lambda strong=False: None)
        try:
            ctx._only_papers = set((d["meta"] or {}).get("cited_papers") or [])
            data = _export_bib(ctx) if fmt == "bib" else _export_ris(ctx)
        finally:
            ctx.close()
    return (data if isinstance(data, bytes) else data.encode("utf-8")), name, DRAFT_FORMATS[fmt]


# ---------------------------------------------------------------- references
def _papers(ctx) -> list[dict]:
    only = getattr(ctx, "_only_papers", None)     # a draft exports the papers it cites
    rows = ctx.pg.execute(
        """SELECT p.paper_id, p.source, p.title, p.year, p.authors, p.doi, p.journal_ref, p.abstract
           FROM run_papers rp JOIN papers p USING (paper_id) WHERE rp.run_id=%s ORDER BY p.year DESC, p.paper_id""",
        (ctx.run_id,)).fetchall()
    return rows if only is None else [r for r in rows if r["paper_id"] in only]


def _authors(s: str) -> list[str]:
    return [a.strip() for a in re.split(r",| and ", s or "") if a.strip()]


def _url(p: dict) -> str:
    if p["source"] == "upload":
        return ""                         # a user's own document has no public address
    if p["source"] == "pubmed":
        return f"https://pubmed.ncbi.nlm.nih.gov/{p['paper_id'][4:]}/"
    return (f"https://pmc.ncbi.nlm.nih.gov/articles/{p['paper_id']}/" if p["source"] == "pmc"
            else f"https://arxiv.org/abs/{p['paper_id']}")


def _export_bib(ctx):
    out = []
    for p in _papers(ctx):
        key = re.sub(r"[^A-Za-z0-9]", "", (_authors(p["authors"]) or ["anon"])[0].split()[-1]) + str(p["year"] or "") \
            + re.sub(r"[^A-Za-z0-9]", "", p["paper_id"])[-5:]
        fields = {"title": "{" + (p["title"] or "") + "}", "author": " and ".join(_authors(p["authors"])),
                  "year": str(p["year"] or ""), "url": _url(p)}
        if p["doi"]:
            fields["doi"] = p["doi"]
        if p["source"] == "arxiv":
            fields.update(eprint=p["paper_id"], archivePrefix="arXiv")
        if p["journal_ref"]:
            fields["journal"] = p["journal_ref"]
        kind = "article" if (p["source"] == "pmc" or p["journal_ref"]) else "misc"
        out.append(f"@{kind}{{{key},\n" + ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v) + "\n}")
    return "\n\n".join(out) + "\n"


def _export_ris(ctx):
    out = []
    for p in _papers(ctx):
        lines = ["TY  - JOUR" if p["source"] == "pmc" else "TY  - UNPB", f"TI  - {p['title']}"]
        lines += [f"AU  - {a}" for a in _authors(p["authors"])]
        lines += [f"PY  - {p['year']}", f"UR  - {_url(p)}"]
        if p["doi"]:
            lines.append(f"DO  - {p['doi']}")
        if p["journal_ref"]:
            lines.append(f"JO  - {p['journal_ref']}")
        if p["abstract"]:
            lines.append(f"AB  - {' '.join(p['abstract'].split())[:3000]}")
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


# ---------------------------------------------------------------- data
def _paper_rows(ctx) -> tuple[list[str], list[list]]:
    from research_agent.tools import claims as C
    from research_agent.tools.extraction import ENUM_FIELDS, TEXT_FIELDS, fields_for, protocol_of

    LIST_FIELDS, EVIDENCE_FIELDS, _ = fields_for(ctx.extraction_version)
    protocol = protocol_of(ctx) or {}
    qf = [f["name"] for f in protocol.get("fields", [])]
    header = (["paper_id", "corpus", "year", "title", "read"] + LIST_FIELDS + list(ENUM_FIELDS) + TEXT_FIELDS + qf
              + ["reported_results", "reported_associations"] + [f"quote: {f}" for f in EVIDENCE_FIELDS + qf])
    rows = []
    for r in C._rows(ctx):
        d = r["data"]
        ev = d.get("evidence") or {}

        def val(f):
            v = d.get(f)
            return "; ".join(map(str, v)) if isinstance(v, list) else ("" if v in (None, "not_stated") else str(v))
        results = "; ".join(f"{x['metric']} {x['value']} ({x['model']})" for x in d.get("reported_results") or [])
        assocs = "; ".join(f"{x['driver']}: {x['direction']}" for x in d.get("reported_associations") or [])
        rows.append([r["paper_id"], r.get("corpus"), r["year"], r["title"], r["source"]]
                    + [val(f) for f in LIST_FIELDS + list(ENUM_FIELDS) + TEXT_FIELDS + qf] + [results, assocs]
                    + [" | ".join(ev.get(f) or []) for f in EVIDENCE_FIELDS + qf])
    return header, rows


def _csv(header, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _export_csv(ctx):
    return _csv(*_paper_rows(ctx))


def _claim_rows(ctx):
    from research_agent.tools.epistemics import PLAIN

    rows = []
    for c in ctx.pg.execute("SELECT id, agent, text, claim_type, status, state, state_facts, result "
                            "FROM claims WHERE run_id=%s ORDER BY id", (ctx.run_id,)).fetchall():
        r = c["result"] or {}
        f = c["state_facts"] or {}
        counted = (f"{r.get('n_matching')} of {r.get('denominator')}" if r.get("denominator") is not None
                   else json.dumps(r.get("window")) if r.get("window") else "")
        rows.append([f"C{c['id']}", c["agent"], c["text"], c["claim_type"], c["status"], counted,
                     c["state"] or "", PLAIN.get(c["state"] or "", ""), "; ".join(f.get("reasons") or []),
                     f.get("read_in_full", ""), f.get("abstract_only", "")])
    return (["claim", "made_by", "statement", "type", "verdict", "counted", "state", "state_means",
             "why_that_state", "papers_read_in_full", "papers_abstract_only"], rows)


def _opportunity_rows(ctx):
    from research_agent.tools.opportunities import listing

    header = ["id", "kind", "what", "research_question", "state", "grade", "already_done",
              "papers_that_already_do_it", "supporting_claims",
              "weakening_claims", "supporting_papers", "prior_studies", "unresolved_questions",
              "alternative_explanations", "candidate_methods", "required_data", "validation", "from"]
    rows = []
    for o in listing(ctx):
        e, c = o["evidence"] or {}, o["confidence"] or {}
        j = lambda k: "; ".join(str(x) for x in (e.get(k) or []))        # noqa: E731
        pr = e.get("precedent") or {}
        nearest = pr.get(pr.get("verdict") or "") or []
        rows.append([o["item_id"], o["kind"], o["label"], o["question"] or "", o["state"],
                     (c.get("grade") or c.get("level") or "") if isinstance(c, dict) else str(c),
                     pr.get("verdict", ""), "; ".join(x.get("paper_id", "") for x in nearest),
                     j("supporting_claims"), j("weakening_claims"), j("supporting_papers"), j("prior_studies"),
                     j("unresolved_questions"), j("alternative_explanations"), j("candidate_methods"),
                     j("required_data"), j("validation_requirements"),
                     (o["provenance"] or {}).get("agent", "")])
    return header, rows


def _corroboration_rows(ctx):
    from research_agent.tools.corroboration import corroborate

    header = ["driver", "outcome", "verdict", "paper_id", "direction", "significant", "study_system",
              "independent_support", "independent_support_same_system", "backed_by", "against", "quote",
              "linked_to_other_papers_by"]
    rows = []
    for f in corroborate(ctx, save=False).get("findings") or []:
        for st in f["statements"]:
            rows.append([f["driver"], f["outcome"], f["verdict"], st["paper_id"], st["direction"], st["significant"],
                         st["system"], st["independent_support"], st["independent_support_same_system"],
                         "; ".join(st["backed_by"]), "; ".join(st["against"]), st["quote"],
                         "; ".join(st["linked_by"])])
    return header, rows


def _export_corroboration(ctx):
    return _csv(*_corroboration_rows(ctx))


def _screening_rows(ctx):
    from research_agent.tools.review import screening_rows

    return (["paper_id", "corpus", "year", "title", "stage", "reason", "found_by"],
            [[r["paper_id"], r["corpus"], r["year"], r["title"], r["stage"], r["reason"] or "", r["found_by"] or ""]
             for r in screening_rows(ctx)])


def _burden_rows(ctx):
    from research_agent.tools.burden import research_vs_burden

    rb = research_vs_burden(ctx)
    return (["iso3", "country", "papers", "share_of_papers", "cases", "share_of_cases", "ratio", "example_papers"],
            [[x["iso3"], x["country"], x["papers"], x["research_share"], x["cases"], x["burden_share"], x["ratio"],
              "; ".join(x["paper_ids"])] for x in rb["countries"]])


def _export_screening(ctx):
    return _csv(*_screening_rows(ctx))


def _export_opportunities(ctx):
    return _csv(*_opportunity_rows(ctx))


def _export_burden(ctx):
    return _csv(*_burden_rows(ctx))


def _export_burden_chart(ctx):
    import os
    import tempfile

    from research_agent.tools.burden import burden_chart

    with tempfile.TemporaryDirectory() as d:
        path = burden_chart(ctx, os.path.join(d, "chart.png"))
        if not path:
            raise ValueError("no burden data loaded: run `python -m research_agent.cli burden-fetch` first")
        with open(path, "rb") as fh:
            return fh.read()


def _export_protocol(ctx):
    from research_agent.tools.review import prisma_markdown, protocol_markdown

    return protocol_markdown(ctx) + "\n" + "\n".join(prisma_markdown(ctx))


# ---------------------------------------------------------------- the evidence package
# One archive holding everything a reader needs to check the run without asking for anything else: the
# report as written, every number with the papers and quotes behind it, what each number's state means, what
# the run says is worth doing and whether anyone has done it, the papers themselves in citation form, and
# the record of what was screened in or out. The manifest says what each file is and, more importantly,
# which questions the package can and cannot answer.
PACKAGE = [
    ("report.md", "md", "The report as it was written, with the computed sections appended."),
    ("claims.csv", None, "Every claim: its statement, the predicate code tested, the count it produced, its "
                         "state and why that state, and how its papers were read."),
    ("opportunities.csv", "opportunities", "What the run says is worth doing: what supports each one, what "
                                          "weakens it, what is unresolved, and whether the corpus already "
                                          "holds a study that does it."),
    ("corroboration.csv", "corroboration", "Every association a paper reports, with the papers that back it, "
                                           "dispute it or qualify it, counted in independent sources."),
    ("extractions.csv", "csv", "One row per analysed paper: every extracted field and the verified quote "
                               "behind each one."),
    ("screening_log.csv", "screening", "Every paper the search surfaced and what happened to it "
                                       "(identified, included, excluded, duplicate, over the limit)."),
    ("protocol_and_prisma.md", "protocol", "The protocol the run followed and the PRISMA flow counts."),
    ("references.bib", "bib", "BibTeX for every analysed paper."),
    ("references.ris", "ris", "RIS for every analysed paper (Zotero, EndNote, Mendeley)."),
    ("run_data.xlsx", "xlsx", "The same tables as one workbook, plus reported results and associations."),
]


def _manifest(ctx, included: list[tuple[str, str]], missing: list[tuple[str, str]]) -> str:
    from research_agent.tools import cohort

    facts = {}
    try:
        from research_agent.agents.report import run_facts

        facts = run_facts(ctx)
    except Exception:
        pass
    L = [f"# Evidence package", "", f"**Question:** {ctx.question}", "",
         f"Run `{ctx.run_id}`. Papers analysed: {facts.get('extracted', '?')} "
         f"({facts.get('fulltext', '?')} read in full, {facts.get('abstract_only', '?')} from the abstract "
         f"alone), from a corpus of {facts.get('corpus', '?')} papers.", "",
         f"**Counted over:** {cohort.describe(cohort.of(ctx))}.", ""]
    states = facts.get("claims_by_state") or {}
    if states:
        from research_agent.tools.epistemics import PLAIN

        L += ["## What the numbers are worth", "",
              "Each claim carries the state its measurement supports, not the state its wording claims:", ""]
        L += [f"- **{s}** ({n}): {PLAIN.get(s, '')}" for s, n in states.items()]
        L += [""]
    L += ["## What is in here", ""]
    L += [f"- `{name}` — {what}" for name, what in included]
    if missing:
        L += [""] + ["## Not in here", ""] + [f"- `{name}` — {why}" for name, why in missing]
    L += ["", "## What this package cannot tell you", "",
          "- A count of zero means no analysed paper reports it. Whether it is absent from the literature is "
          "a different question, which this run does not answer; `claims.csv` marks those as `not_reported` "
          "or `not_searched_enough`.",
          "- Fields are stated far more often in full texts than in abstracts, so a field's rate across a "
          "mostly abstract-read set says as much about reading depth as about the papers.",
          "- Shares of the corpus over time are not shares of the literature: the PMC part was loaded as "
          "topic slices, so its year-on-year shares track what was loaded.",
          "- Every quote in `extractions.csv` was checked against the paper's own text by code. Values whose "
          "quote was not found were discarded, not kept unverified.", ""]
    return "\n".join(L)


def _export_package(ctx):
    import zipfile

    buf = io.BytesIO()
    included: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    payloads: list[tuple[str, bytes]] = []
    for name, fmt, what in PACKAGE:
        try:
            if fmt is None:                                  # claims have no single-file export of their own
                data = _csv(*_claim_rows(ctx))
            else:
                data = globals()[f"_export_{fmt}"](ctx)
            if data is None:
                raise ValueError("nothing to export")
            payloads.append((name, data if isinstance(data, bytes) else data.encode("utf-8")))
            included.append((name, what))
        except Exception as exc:          # one missing part must not cost the reader the whole package
            missing.append((name, f"{what} Not included: {str(exc)[:160]}"))
    if not payloads:
        raise ValueError("this run has nothing to package yet")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MANIFEST.md", _manifest(ctx, included, missing))
        for name, data in payloads:
            z.writestr(name, data)
    return buf.getvalue()


def _export_xlsx(ctx):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    from research_agent.tools.review import prisma_counts
    from research_agent.tools import claims as C

    wb = Workbook()
    sheets = [("Papers", *_paper_rows(ctx)), ("Claims", *_claim_rows(ctx)),
              ("Opportunities", *_opportunity_rows(ctx)), ("Corroboration", *_corroboration_rows(ctx)), ("Screening", *_screening_rows(ctx)),
              ("Research vs burden", *_burden_rows(ctx))]
    res_rows, assoc_rows = [], []
    for r in C._rows(ctx):
        for x in r["data"].get("reported_results") or []:
            res_rows.append([r["paper_id"], x["metric"], x["value"], x.get("value_num"),
                             x.get("unit_as_written", ""), x.get("unit"), x.get("value_canonical"),
                             x["model"], x["is_baseline"], x["split"], x["setting"], x["horizon"], x["quote"]])
        for x in r["data"].get("reported_associations") or []:
            assoc_rows.append([r["paper_id"], x["driver"], x["direction"], x["lag"], x["significant"], x["quote"]])
    sheets.insert(1, ("Reported results", ["paper_id", "metric", "value", "value_number", "unit_as_written",
                                           "unit", "value_in_that_unit", "model", "baseline", "split",
                                           "setting", "horizon", "quote"], res_rows))
    sheets.insert(2, ("Associations", ["paper_id", "driver", "direction", "lag", "significant", "quote"], assoc_rows))
    from research_agent.tools import meta

    for key in [k for k in ctx.notes() if k.startswith("meta:") and not k.endswith(":result")]:
        cond = (ctx.notes()[key] or {}).get("condition") or key.removeprefix("meta:")
        header, rows = meta.export_rows(ctx, cond)
        if rows:
            sheets.append((f"Prevalence {cond}"[:31], header, rows))
    pc = prisma_counts(ctx)
    sheets.append(("PRISMA", ["stage", "papers"], [[k, json.dumps(v) if isinstance(v, dict) else v] for k, v in pc.items()]))
    wb.remove(wb.active)
    for name, header, rows in sheets:
        ws = wb.create_sheet(name[:31])
        ws.append(header)
        for c in ws[1]:
            c.font = Font(bold=True)
        for row in rows:
            ws.append([("" if v is None else v) for v in row])
        ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
