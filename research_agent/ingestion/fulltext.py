"""Full text for shortlisted papers: fetch assembled TeX from `paper_text`, clean it, split sections.

We never bulk-load the 70 GB `paper_text` config. The Literature agent asks for full text only for
the papers it is about to read closely; results are cached in `paper_fulltext`.

`paper_text` is raw LaTeX with known problems (see the dataset card): withdrawal stubs,
`\\includepdf` wrappers with no text, and some rows where the resolver picked a publisher template
instead of the paper. Each row gets a `status` so downstream agents know what they are reading.
"""
from __future__ import annotations

import re

from research_agent.config import settings

_COMMENT = re.compile(r"(?<!\\)%.*")
_ENV_DROP = re.compile(
    r"\\begin\{(figure\*?|table\*?|equation\*?|align\*?|eqnarray\*?|multline\*?|gather\*?|"
    r"tikzpicture|algorithm\*?|algorithmic|lstlisting|verbatim)\}.*?\\end\{\1\}",
    re.S,
)
_CAPTION = re.compile(r"\\caption\*?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}")
_SECTION = re.compile(r"\\(section|chapter)\*?\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_BIB = re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}|\\bibliography\{[^}]*\}", re.S)

PRIORITY = [
    (re.compile(r"method|approach|model|architecture|framework|algorithm", re.I), 5000),
    (re.compile(r"data|dataset|cohort|material|participant|population|study design", re.I), 4000),
    (re.compile(r"experiment|evaluation|result|validation|performance", re.I), 3500),
    (re.compile(r"limitation|discussion|threat", re.I), 3000),
    (re.compile(r"conclusion|future", re.I), 1500),
    (re.compile(r"introduction|background", re.I), 1500),
]


def _latex_to_text(tex: str) -> str:
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        text = LatexNodes2Text(math_mode="with-delimiters", strict_latex_spaces=True).latex_to_text(tex)
    except Exception:  # pylatexenc is tolerant but can still choke on exotic macros
        text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", tex)
        text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\$[^$]{0,300}\$", " [math] ", text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def clean_latex(tex: str) -> tuple[str, list[dict]]:
    """Return (clean_text, sections)."""
    tex = _COMMENT.sub("", tex)
    m = re.search(r"\\begin\{document\}(.*?)(\\end\{document\}|$)", tex, re.S)
    body = m.group(1) if m else tex
    body = _BIB.sub("", body)
    captions = [c.strip() for c in _CAPTION.findall(body)]
    body = _ENV_DROP.sub(" ", body)
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", " ", body, flags=re.S)  # we already have it

    parts = _SECTION.split(body)  # [pre, kind, heading, text, kind, heading, text, ...]
    sections = []
    if parts[0].strip():
        sections.append({"heading": "Front matter", "text": _latex_to_text(parts[0])})
    for i in range(1, len(parts) - 2, 3):
        heading = _latex_to_text(parts[i + 1])
        sections.append({"heading": heading, "text": _latex_to_text(parts[i + 2])})
    if captions:
        sections.append({"heading": "Figure and table captions",
                         "text": "\n".join(_latex_to_text(c) for c in captions[:40])})
    sections = [s for s in sections if len(s["text"]) > 10]
    clean = "\n\n".join(f"## {s['heading']}\n{s['text']}" for s in sections)
    return clean, sections


def assess(raw: str, clean: str, title: str) -> str:
    head = raw.lstrip()[:400]
    if len(raw) < 1000 or head.startswith("%auto-ignore") and len(raw) < 5000:
        return "stub"
    if "\\includepdf" in raw and len(clean) < 2000:
        return "stub"
    title_tokens = {t for t in re.findall(r"[a-z]{4,}", title.lower())}
    if title_tokens:
        found = sum(1 for t in title_tokens if t in clean[:20000].lower())
        if found / len(title_tokens) < 0.4:
            return "template_suspect"
    return "ok"


def select_for_reading(sections: list[dict], budget: int = 14000) -> str:
    """Pick the most informative sections within a character budget (methods, data, results, limits)."""
    picked: dict[int, str] = {}  # section index -> text taken (keeps original paper order)
    used = 0
    for pattern, cap in PRIORITY:
        for idx, s in enumerate(sections):
            if used >= budget:
                break
            if idx in picked or not pattern.search(s["heading"]):
                continue
            take = s["text"][: min(cap, budget - used)]
            if take:
                picked[idx] = take
                used += len(take)
    if not picked and sections:  # headings didn't match anything: take the start of the paper
        return "\n\n".join(f"## {s['heading']}\n{s['text']}" for s in sections)[:budget]
    return "\n\n".join(f"## {sections[i]['heading']}\n{picked[i]}" for i in sorted(picked))


def fetch_fulltext(pg, paper_ids: list[str]) -> dict[str, str]:
    """Ensure full text is cached for these papers. Returns {paper_id: status}."""
    if not paper_ids:
        return {}
    have = {r["paper_id"]: r["status"] for r in pg.execute(
        "SELECT paper_id, status FROM paper_fulltext WHERE paper_id = ANY(%s)", (paper_ids,)).fetchall()}
    todo = sorted(set(paper_ids) - set(have))
    if todo:
        from research_agent.ingestion.load import duck

        con = duck()
        id_list = ", ".join("'" + pid.replace("'", "''") + "'" for pid in todo)
        rows = con.execute(
            f"SELECT paper_id, text, resolution, title FROM read_parquet('{settings.paper_text_glob}') "
            f"WHERE paper_id IN ({id_list})"
        ).fetchall()
        found = set()
        for pid, raw, resolution, title in rows:
            found.add(pid)
            clean, sections = clean_latex(raw or "")
            status = assess(raw or "", clean, title or "")
            pg.execute(
                "INSERT INTO paper_fulltext (paper_id, status, clean_text, sections, resolution, n_chars) "
                "VALUES (%s,%s,%s,%s::jsonb,%s,%s) ON CONFLICT (paper_id) DO NOTHING",
                (pid, status, clean, _json(sections), resolution, len(clean)),
            )
            have[pid] = status
        for pid in set(todo) - found:
            pg.execute("INSERT INTO paper_fulltext (paper_id, status) VALUES (%s,'missing') "
                       "ON CONFLICT DO NOTHING", (pid,))
            have[pid] = "missing"
    return have


def _json(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
