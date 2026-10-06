"""Citations for documents that leave the tool: drafts (and later exports).

Inside the system a paper is cited by its id, [arXiv:2401.00001] or [PMC1234567], because ids can be checked
against the run. A reader of an exported proposal has no access to the run, so the final document needs real
citations and a real reference list. This module:

- normalises the citation forms models write, (arXiv:2401.00001), [2401.00001], [arXiv 2401.00001v2],
  (PMC123; PMC456), into the checked form, for ids that are in the run (anything else is left alone, so the
  audit can remove it);
- checks numbers attributed to papers: a decimal or a percentage in a sentence that cites papers must appear
  in one of those papers' own text or extracted results, otherwise it is tagged [unverified];
- renders citations as author-year, "(Smith et al., 2021; Okello and Mwangi, 2019)", or numbered (Vancouver),
  "[1, 3-5]", with a matching reference list built from the paper records.
"""
from __future__ import annotations

import re

STYLES = ("author-year", "numbered")
_ID = r"(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})"          # new and old arXiv ids
_ONE = re.compile(rf"(?:arXiv[:\s]?\s*)?({_ID})(?:v\d+)?|(PMC)\s?(\d{{4,9}})|\b(UP\d{{1,9}})\b", re.I)
_GROUP = re.compile(rf"[\[(]\s*((?:(?:arXiv[:\s]?\s*)?{_ID}(?:v\d+)?|PMC\s?\d{{4,9}}|UP\d{{1,9}})"
                    rf"(?:\s*[,;]\s*(?:(?:arXiv[:\s]?\s*)?{_ID}(?:v\d+)?|PMC\s?\d{{4,9}}|UP\d{{1,9}}))*)\s*[\])]",
                    re.I)
_BARE = re.compile(rf"(?<![\[\w/.:])arXiv:\s?({_ID})(?:v\d+)?(?![\w\]])|(?<![\[\w])(PMC)(\d{{4,9}})(?![\w\]])"
                   rf"|(?<![\[\w/])(UP\d{{1,9}})(?![\w\]])")
CITE = re.compile(r"\[(?:arXiv:([^\]\s]+)|(PMC\d+|UP\d+))\]")
_C1 = r"\[(?:arXiv:[^\]\s]+|PMC\d+|UP\d+)\]"
_RUN = re.compile(rf"{_C1}(?:\s*[,;]?\s*{_C1})*")


def _tag(pid: str) -> str:
    return f"[{pid}]" if pid.startswith(("PMC", "UP")) else f"[arXiv:{pid}]"


def normalise(text: str, known: set[str]) -> str:
    """Bring every citation of a known paper into the checked form [arXiv:ID] / [PMCn]."""
    def ids_in(chunk: str) -> list[str]:
        out = []
        for m in _ONE.finditer(chunk):
            pid = m.group(1) or (m.group(4).upper() if m.group(4) else f"PMC{m.group(3)}")
            out.append(pid)
        return out

    def group(m):   # unknown ids are converted too, so the audit sees them and removes them
        ids = ids_in(m.group(1))
        return " ".join(_tag(p) for p in ids) if ids else m.group(0)

    text = _GROUP.sub(group, text)

    def bare(m):
        pid = m.group(1) or (m.group(4).upper() if m.group(4) else f"PMC{m.group(3)}")
        return _tag(pid) if pid in known else m.group(0)

    return _BARE.sub(bare, text)


def cited_ids(text: str) -> list[str]:
    return [a or b for a, b in CITE.findall(text)]


# ---------------------------------------------------------------- numbers attributed to papers
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")
_NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d+(?:\.\d+)?\s?%)(?![\w.]*\d)(?!\s*\[unverified)")


def _num_in(num: str, text: str) -> bool:
    core = num.replace("%", "").strip()
    if re.search(rf"(?<![\d.]){re.escape(core)}(?![\d])", text):
        return True
    try:                                   # 0.870 in the text backs 0.87; 87% backs 0.87 and the reverse
        v = float(core)
    except ValueError:
        return False
    for m in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?", text):
        try:
            w = float(m.group(0))
        except ValueError:
            continue
        if abs(w - v) < 1e-9 or (num.endswith("%") and abs(w * 100 - v) < 0.051) or abs(w - v * 100) < 1e-6:
            return True
    return False


def check_attributions(pg, text: str) -> tuple[str, int]:
    """Tag what a sentence attributes to the papers it cites but none of them contains: decimals and
    percentages (per sentence), and places and method families (per clause, against the citation that closes
    the clause). Returns the text and the number of tags added."""
    text, n_nums = _check_numbers(pg, text)
    text, n_terms = _check_terms(pg, text)
    return text, n_nums + n_terms


def _paper_texts(pg, ids: list[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for r in pg.execute(
            """SELECT p.paper_id, p.title, p.abstract, f.clean_text FROM papers p
               LEFT JOIN paper_fulltext f ON f.paper_id = p.paper_id AND f.status = 'ok'
               WHERE p.paper_id = ANY(%s)""", (ids,)).fetchall():
        sources[r["paper_id"]] = " ".join(x or "" for x in (r["title"], r["abstract"], r["clean_text"]))
    for r in pg.execute("SELECT paper_id, data FROM extractions WHERE paper_id = ANY(%s)", (ids,)).fetchall():
        d = r["data"] or {}
        extra = [f"{x.get('value')} {x.get('quote', '')}" for x in d.get("reported_results") or []]
        extra += [str(d.get("key_findings") or ""), str(d.get("sample_size") or "")]
        for f in ("methods", "geography", "datasets", "data_modalities", "study_designs", "populations",
                  "organisms", "interventions", "mechanisms", "targets", "outcomes"):
            extra += [str(v) for v in (d.get(f) or [])]
        sources[r["paper_id"]] = sources.get(r["paper_id"], "") + " " + " ".join(extra)
    return sources


def _check_numbers(pg, text: str) -> tuple[str, int]:
    """Decimals and percentages in paper-citing sentences that none of the cited papers contains."""
    ids = sorted(set(cited_ids(text)))
    if not ids:
        return text, 0
    sources = _paper_texts(pg, ids)
    flagged = 0

    def fix(sentence: str) -> str:
        nonlocal flagged
        pids = cited_ids(sentence)
        if not pids:
            return sentence
        pool = " ".join(sources.get(p, "") for p in pids)
        # a clause that ends in a count [C12] is checked by the claim audit instead: hide it here
        skip, prev = [], 0
        for cm in re.finditer(r"\[(?:arXiv:[^\]\s]+|PMC\d+|C\d+)[^\]]*\]", sentence):
            if cm.group(0).startswith("[C"):
                skip.append((prev, cm.start()))
            prev = cm.end()

        def tag(m):
            nonlocal flagged
            if _num_in(m.group(1), pool):
                return m.group(0)
            flagged += 1
            return m.group(0) + " [unverified]"
        # hide citations and bracketed notes, whose ids (1607.00001) look like decimals
        held: list[str] = []

        def hold(m):
            held.append(m.group(0))
            return f"\x00{len(held) - 1}\x00"

        def hold_text(t: str) -> str:
            held.append(t)
            return f"\x00{len(held) - 1}\x00"
        for lo, hi in reversed(skip):
            sentence = sentence[:lo] + hold_text(sentence[lo:hi]) + sentence[hi:]
        masked = re.sub(r"\[[^\]]*\]", hold, sentence)
        masked = _NUM.sub(tag, masked)
        return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], masked)

    out, last = [], 0
    for m in _SENT.finditer(text):
        out.append(fix(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(fix(text[last:]))
    return "".join(out), flagged


# ---------------------------------------------------------------- rendering for readers
def _names(authors: str) -> list[tuple[str, str]]:
    """'Grace A. Okello, Peter Mwangi and Li Wei' -> [('Okello', 'GA'), ('Mwangi', 'P'), ('Wei', 'L')]."""
    from research_agent.agents.report import _clean_author

    out = []
    for raw in re.split(r",\s*|\s+and\s+|;\s*", authors or ""):
        name = _clean_author(raw).strip().strip(".")
        if not name or name.lower() in ("et al", "others"):
            continue
        parts = name.split()
        if len(parts) >= 2 and re.fullmatch(r"[A-Z]{1,3}", parts[-1]):           # "Okello GA"
            surname, given = " ".join(parts[:-1]), parts[-1]
        else:
            particles = {"van", "von", "de", "da", "del", "der", "di", "la", "le", "dos", "du"}
            i = len(parts) - 1
            while i > 0 and parts[i - 1].lower() in particles:
                i -= 1
            surname, given = " ".join(parts[i:]), "".join(p[0] for p in parts[:i] if p[:1].isalpha())
        out.append((surname, given.upper()))
    return out


def _inline(names: list[tuple[str, str]], year) -> str:
    y = year or "n.d."
    if not names:
        return f"Anonymous, {y}"
    if len(names) == 1:
        return f"{names[0][0]}, {y}"
    if len(names) == 2:
        return f"{names[0][0]} and {names[1][0]}, {y}"
    return f"{names[0][0]} et al., {y}"


def _ref_authors(names: list[tuple[str, str]], style: str) -> str:
    if not names:
        return "Anonymous"
    shown = names[:6]
    if style == "numbered":
        s = ", ".join(f"{sn} {gv}".strip() for sn, gv in shown)
    else:
        parts = [f"{sn}, {'. '.join(gv)}{'.' if gv else ''}".rstrip(", ") for sn, gv in shown]
        s = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + (", " if len(names) > 6 else ", & ") + parts[-1]
    return s + (", et al." if len(names) > 6 else "")


def _where(p: dict) -> str:
    if p["source"] == "upload":
        from research_agent.uploads import where

        return where(p)
    if p["source"] == "pmc":
        bits = [f"*{p['journal_ref']}*" if p.get("journal_ref") else "", f"https://doi.org/{p['doi']}" if p.get("doi") else "",
                f"PMCID: {p['paper_id']}", f"https://pmc.ncbi.nlm.nih.gov/articles/{p['paper_id']}/"]
    else:
        bits = [f"arXiv:{p['paper_id']} [preprint]", f"*{p['journal_ref']}*" if p.get("journal_ref") else "",
                f"https://doi.org/{p['doi']}" if p.get("doi") and not p["doi"].lower().startswith("10.48550") else "",
                f"https://arxiv.org/abs/{p['paper_id']}"]
    return ". ".join(b for b in bits if b)


def render(pg, text: str, style: str = "author-year") -> tuple[str, list[str], list[str]]:
    """Replace [arXiv:ID]/[PMCn] runs with reader citations. Returns (text, reference lines, cited ids in
    reference order)."""
    if style not in STYLES:
        style = "author-year"
    order = list(dict.fromkeys(cited_ids(text)))
    if not order:
        return text, [], []
    papers = {r["paper_id"]: r for r in pg.execute(
        "SELECT paper_id, source, title, year, authors, doi, journal_ref FROM papers WHERE paper_id = ANY(%s)",
        (order,)).fetchall()}
    order = [p for p in order if p in papers]
    names = {p: _names(papers[p]["authors"]) for p in order}
    label: dict[str, str] = {}
    if style == "author-year":
        base = {p: _inline(names[p], papers[p]["year"]) for p in order}
        dupes: dict[str, list[str]] = {}
        for p in sorted(order, key=lambda x: (base[x], x)):
            dupes.setdefault(base[p], []).append(p)
        for b, ps in dupes.items():
            for i, p in enumerate(ps):
                label[p] = b + ("abcdefghijklmnopqrstuvwxyz"[i] if len(ps) > 1 else "")
    else:
        label = {p: str(i + 1) for i, p in enumerate(order)}

    def run(m):
        ids = [p for p in dict.fromkeys(cited_ids(m.group(0))) if p in label]
        if not ids:
            return m.group(0)
        if style == "author-year":
            return "(" + "; ".join(label[p] for p in sorted(ids, key=lambda x: label[x])) + ")"
        nums = sorted(int(label[p]) for p in ids)
        parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else ", ".join(map(str, nums[i:j + 1])))
            i = j + 1
        return "[" + ", ".join(parts) + "]"

    text = _RUN.sub(run, text)
    refs = []
    if style == "author-year":
        for p in sorted(order, key=lambda x: (names[x][0][0].lower() if names[x] else "~", papers[x]["year"] or 0, label[x])):
            r = papers[p]
            year = label[p].rsplit(", ", 1)[-1]
            refs.append(f"- {_ref_authors(names[p], style)} ({year}). {r['title']}. {_where(r)}")
    else:
        for p in order:
            r = papers[p]
            refs.append(f"{label[p]}. {_ref_authors(names[p], style)}. {r['title']}. {r['year'] or 'n.d.'}. {_where(r)}")
    return text, refs, order


# ---------------------------------------------------------------- places and methods, clause by clause
DEMONYMS = {
    "ugandan": "UGA", "kenyan": "KEN", "tanzanian": "TZA", "rwandan": "RWA", "burundian": "BDI", "ethiopian": "ETH",
    "somali": "SOM", "sudanese": "SDN", "south sudanese": "SSD", "eritrean": "ERI", "malawian": "MWI",
    "mozambican": "MOZ", "zambian": "ZMB", "zimbabwean": "ZWE", "namibian": "NAM", "botswanan": "BWA",
    "south african": "ZAF", "angolan": "AGO", "congolese": "COD", "cameroonian": "CMR", "nigerian": "NGA",
    "ghanaian": "GHA", "beninese": "BEN", "togolese": "TGO", "burkinabe": "BFA", "burkinabè": "BFA",
    "malian": "MLI", "nigerien": "NER", "chadian": "TCD", "senegalese": "SEN", "gambian": "GMB", "guinean": "GIN",
    "sierra leonean": "SLE", "liberian": "LBR", "ivorian": "CIV", "malagasy": "MDG", "indian": "IND",
    "pakistani": "PAK", "afghan": "AFG", "bangladeshi": "BGD", "nepali": "NPL", "nepalese": "NPL",
    "sri lankan": "LKA", "burmese": "MMR", "thai": "THA", "cambodian": "KHM", "laotian": "LAO", "vietnamese": "VNM",
    "indonesian": "IDN", "filipino": "PHL", "papua new guinean": "PNG", "chinese": "CHN", "brazilian": "BRA",
    "peruvian": "PER", "colombian": "COL", "venezuelan": "VEN", "ecuadorian": "ECU", "bolivian": "BOL",
    "yemeni": "YEM", "saudi": "SAU", "iranian": "IRN", "haitian": "HTI",
}
_AMBIGUOUS = {"chad", "jordan", "georgia", "turkey", "guinea", "niger", "china", "mali", "togo", "peru", "india"}
_METHODS = {
    "random forest": [r"random[- ]forests?"],
    "gradient boosting": [r"gradient[- ]boost\w*", r"xgboost", r"lightgbm", r"catboost", r"\bgbm\b",
                          r"boosted (?:regression )?trees?"],
    "LSTM": [r"\blstms?\b", r"long short[- ]term memory"],
    "transformer": [r"\btransformers?\b", r"self[- ]attention", r"attention[- ]based"],
    "convolutional network": [r"\bcnns?\b", r"convolutional"],
    "support vector machine": [r"support vector", r"\bsvms?\b", r"\bsvr\b"],
    "ARIMA": [r"\b(?:s?arima\w*)\b", r"autoregressive integrated"],
    "neural network": [r"neural net\w*", r"\bann\b", r"\bmlp\b", r"multilayer perceptron", r"deep learning",
                       r"\blstm", r"\bgru\b", r"convolutional", r"\bcnn", r"transformer"],
    "neuro-fuzzy": [r"\banfis\b", r"neuro[- ]fuzzy"],
    "Gaussian process": [r"gaussian process"],
    "exponential smoothing": [r"holt[- ]winters?", r"exponential smoothing"],
    "INLA": [r"\binla\b", r"integrated nested laplace"],
    "generalized additive model": [r"generali[sz]ed additive", r"\bgams?\b"],
    "distributed lag": [r"distributed[- ]lags?", r"\bdlnm\b"],
    "agent-based model": [r"agent[- ]based"],
    "k-nearest neighbours": [r"k[- ]nearest", r"\bknn\b"],
}
_METHOD_RX = {k: re.compile("|".join(v), re.I) for k, v in _METHODS.items()}
_PLACE_RX = None
_PLACE_ISO: dict[str, str] = {}


def _places():
    """One pattern for every country name, alias, known subnational place and demonym, mapped to ISO3."""
    global _PLACE_RX
    if _PLACE_RX is None:
        from research_agent.tools import burden

        names = dict(burden._countries()) if hasattr(burden, "_countries") else {}
        names.update(burden.ALIASES)
        names.update(burden.PLACES)
        names.update(DEMONYMS)
        _PLACE_ISO.update({k.lower(): v for k, v in names.items() if v and len(k) > 3})
        _PLACE_RX = re.compile(r"(?<![\w-])(" + "|".join(re.escape(k) for k in sorted(_PLACE_ISO, key=len, reverse=True))
                               + r")(?![\w-])", re.I)
    return _PLACE_RX


def _isos(text: str) -> set[str]:
    return {_PLACE_ISO[m.group(1).lower()] for m in _places().finditer(text or "")}


_ANY_CITE = re.compile(r"\[(?:arXiv:[^\]\s]+|PMC\d+|C\d+)[^\]]*\](?:\s*[,;]?\s*\[(?:arXiv:[^\]\s]+|PMC\d+|C\d+)[^\]]*\])*")
_CLAUSE_BREAK = re.compile(r"[;:]\s|\n")


def _check_terms(pg, text: str) -> tuple[str, int]:
    ids = sorted(set(cited_ids(text)))
    if not ids:
        return text, 0
    sources = _paper_texts(pg, ids)
    iso_of = {p: _isos(t) for p, t in sources.items()}
    flagged = 0
    edits: list[tuple[int, str]] = []            # (position after a term, tag), applied right to left
    bounds = [0] + [m.end() for m in _SENT.finditer(text)] + [len(text)]
    for s0, s1 in zip(bounds, bounds[1:]):
        stext = text[s0:s1]
        prev = 0
        for cm in _ANY_CITE.finditer(stext):
            seg_lo = prev
            prev = cm.end()
            pids = [p for p in cited_ids(cm.group(0)) if p in sources]
            if not pids:
                continue                                   # a count [C12] closes a clause but is checked elsewhere
            seg = stext[seg_lo:cm.start()]
            cut = max((m.end() for m in _CLAUSE_BREAK.finditer(seg)), default=0)
            cut = max(cut, len(seg) - 240)
            seg_start = s0 + seg_lo + cut
            seg = seg[cut:]
            masked = re.sub(r"\[[^\]]*\]", lambda m: " " * len(m.group(0)), seg)
            have_iso = set().union(*(iso_of.get(p, set()) for p in pids))
            pool = " ".join(sources.get(p, "") for p in pids)
            for m in _places().finditer(masked):
                name = m.group(1).lower()
                if name in _AMBIGUOUS and not m.group(1)[0].isupper():
                    continue
                if _PLACE_ISO[name] not in have_iso:
                    edits.append((seg_start + m.end(), " [unverified]"))
            for label, rx in _METHOD_RX.items():
                m = rx.search(masked)
                if m and not rx.search(pool):
                    edits.append((seg_start + m.end(), " [unverified]"))
    for pos, tag in sorted(set(edits), reverse=True):
        if not text[pos:pos + 13].startswith(" [unverified"):
            text = text[:pos] + tag + text[pos:]
            flagged += 1
    return text, flagged
