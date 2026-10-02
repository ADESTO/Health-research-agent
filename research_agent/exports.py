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
    rows = []
    for c in ctx.pg.execute("SELECT id, agent, text, claim_type, status, result FROM claims WHERE run_id=%s ORDER BY id",
                            (ctx.run_id,)).fetchall():
        r = c["result"] or {}
        counted = (f"{r.get('n_matching')} of {r.get('denominator')}" if r.get("denominator") is not None
                   else json.dumps(r.get("window")) if r.get("window") else "")
        rows.append([f"C{c['id']}", c["agent"], c["text"], c["claim_type"], c["status"], counted])
    return ["claim", "made_by", "statement", "type", "verdict", "counted"], rows


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


def _export_xlsx(ctx):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    from research_agent.tools.review import prisma_counts
    from research_agent.tools import claims as C

    wb = Workbook()
    sheets = [("Papers", *_paper_rows(ctx)), ("Claims", *_claim_rows(ctx)), ("Screening", *_screening_rows(ctx)),
              ("Research vs burden", *_burden_rows(ctx))]
    res_rows, assoc_rows = [], []
    for r in C._rows(ctx):
        for x in r["data"].get("reported_results") or []:
            res_rows.append([r["paper_id"], x["metric"], x["value"], x.get("value_num"), x["model"], x["is_baseline"],
                             x["split"], x["setting"], x["horizon"], x["quote"]])
        for x in r["data"].get("reported_associations") or []:
            assoc_rows.append([r["paper_id"], x["driver"], x["direction"], x["lag"], x["significant"], x["quote"]])
    sheets.insert(1, ("Reported results", ["paper_id", "metric", "value", "value_number", "model", "baseline", "split",
                                           "setting", "horizon", "quote"], res_rows))
    sheets.insert(2, ("Associations", ["paper_id", "driver", "direction", "lag", "significant", "quote"], assoc_rows))
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
