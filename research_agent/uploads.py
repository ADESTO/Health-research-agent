"""A user's own documents: papers, reports and links they add, private to them, used in the runs they choose.

Testers often hold studies the corpus lacks: paywalled papers, local journals, theses, programme reports,
unpublished surveys. They can add them here and attach them to a run, where each is treated exactly like a
study from the corpus: screened against the protocol in a systematic run, read in full, every extracted value
backed by a quote checked by code, and counted in claims, PRISMA (as "added by the user") and every table.

Privacy is structural, not a filter someone can forget: a document becomes a row in `papers` with
source='upload' and the id UP<n>, and
  - every corpus-wide search and count excludes source='upload' (search, systematic identification, recall
    checks, precedent, gap checks, trends, corpus sizes), so no run ever finds a document by searching;
  - agents cannot add one to a shortlist (add_to_shortlist ignores uploads), so a document enters a run only
    when its owner attaches it;
  - the document list, the original file and deletion are limited to the owner.

    add       a file (PDF, Word .docx, text, Markdown, HTML) or a link, which is fetched; a link to a PMC or arXiv
              paper already in the corpus attaches that paper instead of a copy
    attach    at run start: in a systematic run the documents join the identified records and are screened;
              in an agent-chosen run they go straight onto the shortlist, marked "added by the user"
    delete    removes the document, its text and its records everywhere, including runs that used it
"""
from __future__ import annotations

import base64
import io
import json
import re
from datetime import date
from html.parser import HTMLParser

MAX_BYTES = 20 * 1024 * 1024
MIN_CHARS = 300                       # less than this is not a readable document (a scanned PDF, an error page)
KINDS = {"paper": "Paper supplied by the user", "report": "Unpublished report supplied by the user"}
TYPES = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         ".txt": "text/plain", ".md": "text/markdown", ".html": "text/html", ".htm": "text/html"}
_UP = re.compile(r"^UP(\d+)$")


def is_upload(paper_id: str) -> bool:
    return bool(_UP.match(paper_id or ""))


def doc_id(paper_id: str) -> int | None:
    m = _UP.match(paper_id or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- text out of files
class _HTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections, self._head, self._buf, self._skip, self._in_h = [], "Text", [], 0, False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header", "noscript"):
            self._skip += 1
        elif tag in ("h1", "h2", "h3"):
            self._flush(); self._in_h = True; self._head = ""
        elif tag in ("p", "br", "li", "tr", "div"):
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header", "noscript"):
            self._skip = max(0, self._skip - 1)
        elif tag in ("h1", "h2", "h3"):
            self._in_h = False

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_h:
            self._head += data
        else:
            self._buf.append(data)

    def _flush(self):
        text = re.sub(r"[ \t]+", " ", "".join(self._buf)).strip()
        text = re.sub(r"\n\s*\n+", "\n", text)
        if text:
            self.sections.append({"heading": " ".join(self._head.split())[:200] or "Text", "text": text})
        self._buf = []


def sections_of(filename: str, data: bytes) -> list[dict]:
    """[{heading, text}] for a PDF (one per page), a Word file or text file (split at headings), or HTML."""
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        from pypdf import PdfReader

        pages = [(p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(data)).pages]
        return [{"heading": f"Page {i}", "text": t} for i, t in enumerate(pages, 1) if t]
    if name.endswith(".docx"):
        import docx

        out, head, buf = [], "Text", []
        for para in docx.Document(io.BytesIO(data)).paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if (para.style.name or "").lower().startswith(("heading", "title")):
                if buf:
                    out.append({"heading": head, "text": "\n".join(buf)})
                head, buf = text[:200], []
            else:
                buf.append(text)
        if buf:
            out.append({"heading": head, "text": "\n".join(buf)})
        return out
    text = data.decode("utf-8", errors="replace")
    if name.endswith((".html", ".htm")) or re.search(r"<html|<body|<p[ >]", text[:3000], re.I):
        p = _HTML()
        p.feed(text)
        p._flush()
        return p.sections
    parts = re.split(r"^#{1,3} +(.+)$", text, flags=re.M)
    if len(parts) == 1:
        return [{"heading": "Text", "text": text.strip()}] if text.strip() else []
    out = [{"heading": "Introduction", "text": parts[0].strip()}] if parts[0].strip() else []
    return out + [{"heading": h.strip()[:200], "text": t.strip()} for h, t in zip(parts[1::2], parts[2::2]) if t.strip()]


def fetch_link(url: str, client=None) -> tuple[str, bytes, str]:
    """(file name, bytes, content type) for a link. Only http(s); the size is capped."""
    import httpx

    if not re.match(r"^https?://", url or "", re.I):
        raise ValueError("a link must start with http:// or https://")
    client = client or httpx.Client(timeout=60, follow_redirects=True,
                                    headers={"User-Agent": "Mozilla/5.0 (research document fetch)"})
    r = client.get(url)
    r.raise_for_status()
    data = r.content[:MAX_BYTES + 1]
    if len(data) > MAX_BYTES:
        raise ValueError("the linked file is larger than 20 MB")
    ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
    name = url.rstrip("/").rsplit("/", 1)[-1] or "page"
    if ctype == "application/pdf" and not name.lower().endswith(".pdf"):
        name += ".pdf"
    elif "html" in ctype and not name.lower().endswith((".html", ".htm")):
        name += ".html"
    return name, data, ctype


def corpus_paper_for_link(pg, url: str) -> str | None:
    """A link to a PMC or arXiv paper the corpus already holds: attach that paper rather than a copy."""
    m = re.search(r"(PMC\d{4,9})", url or "", re.I)
    if m:
        pid = m.group(1).upper()
    else:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url or "", re.I)
        pid = m.group(1) if m else None
    if pid and pg.execute("SELECT 1 FROM papers WHERE paper_id=%s AND source <> 'upload'", (pid,)).fetchone():
        return pid
    return None


def _abstract(sections: list[dict]) -> str:
    for s in sections:
        if "abstract" in s["heading"].lower() or "summary" in s["heading"].lower():
            return " ".join(s["text"].split())[:3000]
    full = " ".join(" ".join(s["text"].split()) for s in sections)
    m = re.search(r"\babstract\b[:.\s]*(.{200,3000}?)(?:\b(?:keywords|introduction|background)\b|$)", full, re.I)
    return (m.group(1) if m else full[:1500]).strip()


def _year(sections: list[dict]) -> int:
    first = " ".join(s["text"] for s in sections[:2])[:4000]
    years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", first) if int(y) <= date.today().year]
    return max(years) if years else date.today().year


# ---------------------------------------------------------------- the library
def add_document(pg, owner_id: str | None, *, kind: str = "paper", title: str = "", year: int | None = None,
                 authors: str = "", filename: str | None = None, data: bytes | None = None,
                 url: str | None = None, client=None) -> dict:
    """Store a document and make it readable as a study. Returns the document, or {"corpus_paper_id"} when a
    link points to a paper the corpus already has, or {"error"}."""
    from research_agent.embeddings import get_embedder, paper_text_for_embedding

    kind = kind if kind in KINDS else "paper"
    if url and not data:
        existing = corpus_paper_for_link(pg, url)
        if existing:
            return {"corpus_paper_id": existing, "note": "This paper is already in the corpus; it is attached as is."}
        try:
            filename, data, _ctype = fetch_link(url, client)
        except Exception as exc:
            return {"error": f"the link could not be fetched: {str(exc)[:200]}"}
    if not data:
        return {"error": "add a file or a link"}
    if len(data) > MAX_BYTES:
        return {"error": "the file is larger than 20 MB"}
    try:
        sections = sections_of(filename or "", data)
    except Exception as exc:
        return {"error": f"the file could not be read: {str(exc)[:200]}"}
    clean = "\n\n".join(f"{s['heading']}\n{s['text']}" for s in sections)
    if len(clean) < MIN_CHARS:
        return {"error": "almost no text could be read from it. A scanned PDF has no text layer: export it with "
                         "text recognition (OCR) and add it again."}
    title = " ".join((title or "").split())[:300] or next(
        (ln.strip() for s in sections[:1] for ln in s["text"].splitlines() if len(ln.strip()) > 15), filename or "Document")[:300]
    year = int(year) if year else _year(sections)
    abstract = _abstract(sections)
    ext = "." + (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    row = pg.execute(
        "INSERT INTO user_documents (owner_id, kind, title, filename, url, content, content_type, n_chars) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id, created_at",
        (owner_id, kind, title, (filename or "")[:200], (url or "")[:500], data, TYPES.get(ext, "application/octet-stream"),
         len(clean))).fetchone()
    pid = f"UP{row['id']}"
    vec = get_embedder().embed_documents([paper_text_for_embedding(title, abstract)])[0]
    pg.execute(
        "INSERT INTO papers (paper_id, source, title, abstract, authors, categories, primary_category, year, "
        "journal_ref, license, health_reason, embedding) VALUES (%s,'upload',%s,%s,%s,%s,'upload',%s,%s,%s,%s,%s)",
        (pid, title, abstract, (authors or "")[:500], [kind], year,
         KINDS[kind] + (f": {url}" if url else f" ({filename})" if filename else ""), "supplied by the user",
         "upload", vec))
    from research_agent.ingestion.fulltext import _json

    pg.execute("INSERT INTO paper_fulltext (paper_id, status, clean_text, sections, resolution, n_chars) "
               "VALUES (%s,'ok',%s,%s::jsonb,'upload',%s) ON CONFLICT (paper_id) DO UPDATE SET status='ok', "
               "clean_text=EXCLUDED.clean_text, sections=EXCLUDED.sections, n_chars=EXCLUDED.n_chars",
               (pid, clean, _json(sections), len(clean)))
    return {"id": row["id"], "paper_id": pid, "title": title, "year": year, "kind": kind, "n_chars": len(clean),
            "pages_or_sections": len(sections)}


def list_documents(pg, owner_id: str | None) -> list[dict]:
    return pg.execute(
        "SELECT id, 'UP' || id AS paper_id, kind, title, filename, url, n_chars, created_at, "
        "(SELECT year FROM papers WHERE paper_id = 'UP' || d.id) AS year, "
        "(SELECT count(*) FROM run_papers rp WHERE rp.paper_id = 'UP' || d.id) AS runs "
        "FROM user_documents d WHERE owner_id IS NOT DISTINCT FROM %s ORDER BY id DESC", (owner_id,)).fetchall()


def owned(pg, owner_id: str | None, ids: list[int]) -> list[int]:
    """The ids among these that belong to this user (all of them on a local install with no users)."""
    if not ids:
        return []
    rows = pg.execute("SELECT id FROM user_documents WHERE id = ANY(%s) AND owner_id IS NOT DISTINCT FROM %s",
                      ([int(i) for i in ids], owner_id)).fetchall()
    return [r["id"] for r in rows]


def file_of(pg, owner_id: str | None, doc: int) -> dict | None:
    return pg.execute("SELECT filename, content, content_type, url FROM user_documents WHERE id=%s "
                      "AND owner_id IS NOT DISTINCT FROM %s", (doc, owner_id)).fetchone()


def delete_document(pg, owner_id: str | None, doc: int) -> bool:
    """Gone everywhere: the file, its text, its records, and its place in any run that used it."""
    if not owned(pg, owner_id, [doc]):
        return False
    pid = f"UP{doc}"
    for table in ("run_papers", "screening", "extractions", "protocol_extractions", "paper_fulltext"):
        try:
            pg.execute(f"DELETE FROM {table} WHERE paper_id=%s", (pid,))
        except Exception:
            pass
    pg.execute("DELETE FROM papers WHERE paper_id=%s", (pid,))
    pg.execute("DELETE FROM user_documents WHERE id=%s", (doc,))
    return True


# ---------------------------------------------------------------- runs
def set_for_run(pg, run_id: str, paper_ids: list[str]) -> None:
    """Remember which documents (UP ids, or corpus ids from links) the user attached to a run."""
    pg.execute("INSERT INTO run_notes (run_id, agent, content) VALUES (%s,'user_documents',%s::jsonb) "
               "ON CONFLICT (run_id, agent) DO UPDATE SET content = EXCLUDED.content",
               (run_id, json.dumps({"paper_ids": list(dict.fromkeys(paper_ids))})))


def for_run(ctx) -> list[str]:
    ids = (ctx.notes().get("user_documents") or {}).get("paper_ids") or []
    if not ids:
        return []
    have = {r["paper_id"] for r in ctx.pg.execute("SELECT paper_id FROM papers WHERE paper_id = ANY(%s)",
                                                  (ids,)).fetchall()}
    return [p for p in ids if p in have]          # a document deleted since is simply gone


def attach_to_run(ctx) -> int:
    """Agent-chosen runs: the user's documents go straight onto the shortlist, marked as theirs."""
    from research_agent.tools.review import log_decision, log_identified

    ids = for_run(ctx)
    if not ids:
        return 0
    log_identified(ctx, ids, "added by the user")
    with ctx.pg.cursor() as cur:
        for pid in ids:
            cur.execute("INSERT INTO run_papers (run_id, paper_id, added_by, reason, score) VALUES "
                        "(%s,%s,'user','added by the user',1.0) ON CONFLICT (run_id, paper_id) DO NOTHING",
                        (ctx.run_id, pid))
    log_decision(ctx, ids, "included", "added by the user")
    return len(ids)


def where(p: dict) -> str:
    """How a document is identified in a reference list: what it is and where it came from."""
    return p.get("journal_ref") or "Document supplied by the user"


def b64(data: str) -> bytes:
    return base64.b64decode(data.split(",", 1)[-1] if data.startswith("data:") else data)
