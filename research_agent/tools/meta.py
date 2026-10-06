"""Meta-analysis of reported prevalence: a pass that reads for counts, and code that pools them.

The results table describes a literature: the median of the prevalences papers report. That is not a pooled
prevalence. A pooled prevalence weights each study by how much it can tell you (a survey of 2,000 people
counts for more than a clinic series of 40), allows for the studies genuinely differing, and says how much
they do. This module does that, in two steps that keep the system's rules.

1. read (LLM, one call per paper, quotes checked by code)
   Each counted paper is read for the condition alone: the number of cases, the number studied, the
   percentage, who and where, how a case was defined, and the ten items of the Hoy et al. (2012) risk-of-bias
   tool for prevalence studies. Every number must appear in a quote copied from the paper, and every quote is
   checked against the text. A number not in its quote is dropped; a risk-of-bias answer without a found
   quote becomes "unclear". The model never computes anything.

2. pool (code only)
   cases / sample size per study (derived from a percentage and a sample size when the count is not written,
   and marked as derived); one overall estimate per study per condition; logit-transformed proportions;
   DerSimonian-Laird random effects for the between-study variance; a Hartung-Knapp-Sidik-Jonkman confidence
   interval (never narrower than the DerSimonian-Laird one), because with few studies the plain interval is
   too narrow; I2; a 95% prediction interval; subgroups by condition, setting, case definition and country;
   and a sensitivity analysis without studies at high risk of bias.

What it is not: a registered systematic review. The corpus is what was loaded (open-access slices), screening
and extraction were done by a model with quotes checked by code, and no second reviewer checked them. The
output says so, and the per-study table with its quotes is there so a reviewer can check every number.
"""
from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings
from research_agent.tools.base import INT, STR, Tool, obj

MAX_PAPERS = 150             # the systematic cap on analysed papers: a pass reads all of them, no more
MAX_ESTIMATES = 8
SETTINGS = ["community", "hospital_or_clinic", "workplace", "school", "other", "not_stated"]
CASE_DEFINITIONS = ["classification_criteria", "clinical_diagnosis", "imaging", "self_report",
                    "records_or_registry", "not_stated"]
ROB_ANSWERS = ["low", "high", "unclear"]
# Hoy D, Brooks P, Woolf A, et al. Assessing risk of bias in prevalence studies: modification of an existing
# tool and evidence of interrater agreement. J Clin Epidemiol 2012;65:934-9.
HOY_ITEMS = [
    ("representative_population", "Was the study's target population a close representation of the national "
                                  "population in relation to relevant variables (age, sex, occupation)?"),
    ("sampling_frame", "Was the sampling frame a true or close representation of the target population?"),
    ("random_selection", "Was some form of random selection used to select the sample, OR was a census "
                         "undertaken?"),
    ("non_response", "Was the likelihood of non-response bias minimal (response rate 75% or more, or no "
                     "difference between responders and non-responders)?"),
    ("direct_collection", "Were data collected directly from the subjects (as opposed to a proxy)?"),
    ("case_definition", "Was an acceptable case definition used in the study?"),
    ("valid_instrument", "Was the study instrument that measured the parameter of interest shown to have "
                         "reliability and validity?"),
    ("same_mode", "Was the same mode of data collection used for all subjects?"),
    ("prevalence_period", "Was the length of the shortest prevalence period for the parameter of interest "
                          "appropriate?"),
    ("numerator_denominator", "Were the numerator(s) and denominator(s) for the parameter of interest "
                              "appropriate?"),
]
# Summary judgement from the number of items at high risk. Reviews using the Hoy tool set their own cut-offs;
# this is a common one, stated in the output so a reader can apply another.
ROB_CUTOFFS = {"low": (0, 2), "moderate": (3, 4), "high": (5, 10)}
UNCLEAR_COUNTS_AS_HIGH = False

SYSTEM = """You read one research paper to record the PREVALENCE of {condition} it reports, and to assess the
study's risk of bias. Nothing else.

Prevalence means the share of a defined population that has the condition: cases out of people studied. Record
an estimate only when the paper reports it as its own finding for its own sample. Prevalences quoted from other
studies, incidence rates, and the share of PATIENTS with some feature (for example the share of arthritis
patients who are women) are not prevalence of the condition: leave them out.

For each estimate (up to {max_est}):
- condition_type: the specific condition, e.g. rheumatoid arthritis, knee osteoarthritis, gout, any arthritis.
- cases, sample_size, percent: each exactly as written (e.g. "212", "1,650", "12.8%"). Give the ones the paper
  states; leave the others ''. Never calculate one from the others.
- is_overall: true for the estimate for the whole sample; false for a subgroup (women, over 60, one district).
- quote: the sentence with the numbers, copied word for word. If the sample size is stated in a different
  sentence (often in the methods), copy that sentence into sample_size_quote. Code checks every number against
  these quotes and every quote against the paper; a number that is not in its quote is discarded.

Risk of bias: answer each of the ten items low, high or unclear, with the sentence that shows it, copied word
for word. Answer unclear when the text does not say; an answer whose quote is not found counts as unclear.

If the paper reports no prevalence of {condition} in its own sample, set reports_prevalence to false and leave
the rest empty: that is a correct and useful answer."""


def _schema() -> dict:
    estimate = {"type": "object", "properties": {
        "condition_type": STR,
        "population": {**STR, "description": "Who was studied, e.g. adults 18+ in Kisumu county"},
        "country": STR,
        "setting": {"type": "string", "enum": SETTINGS},
        "case_definition": {"type": "string", "enum": CASE_DEFINITIONS,
                            "description": "classification_criteria = published criteria such as ACR/EULAR "
                                           "2010 or ACR 1987; clinical_diagnosis = a clinician's diagnosis "
                                           "without stated criteria; imaging = radiographic or ultrasound; "
                                           "self_report = questionnaire or interview answers"},
        "cases": STR, "sample_size": STR, "percent": STR,
        "is_overall": {"type": "boolean"},
        "subgroup": {**STR, "description": "For a subgroup estimate: which subgroup, else ''"},
        "data_years": {**STR, "description": "When the data were collected, e.g. 2016-2017, if stated"},
        "quote": STR,
        "sample_size_quote": STR}, "required": ["condition_type", "quote"]}
    rob = {"type": "object", "properties": {
        key: {"type": "object", "description": question, "properties": {
            "answer": {"type": "string", "enum": ROB_ANSWERS}, "quote": STR}, "required": ["answer"]}
        for key, question in HOY_ITEMS}}
    return {"name": "record_prevalence", "description": "Record this paper's prevalence estimates and risk of bias.",
            "input_schema": obj({"reports_prevalence": {"type": "boolean"},
                                 "estimates": {"type": "array", "items": estimate},
                                 "risk_of_bias": rob}, ["reports_prevalence"])}


# ---------------------------------------------------------------- numbers and quotes (code)
_THOUSANDS = re.compile(r"(?<=\d)[,   ](?=\d{3}(?!\d))")


def _numbers_in(text: str) -> list[float]:
    """Every number in a text, reading 1,650 / 1 650 as one thousand six hundred and fifty."""
    clean = _THOUSANDS.sub("", text or "")
    return [float(m.replace(",", ".")) for m in re.findall(r"\d+(?:[.,]\d+)?", clean)]


def _number(raw) -> float | None:
    vals = _numbers_in(str(raw or ""))
    return vals[0] if vals else None


def _stated(raw, quotes: list[str]) -> float | None:
    """The number as written, if it appears in one of its quotes; otherwise None (it is discarded)."""
    x = _number(raw)
    if x is None:
        return None
    for q in quotes:
        if any(abs(x - y) < 1e-9 for y in _numbers_in(q)):
            return x
    return None


def check_estimate(est: dict, words: list[str], shingles: set) -> dict | None:
    """One estimate as code accepts it: quotes found in the paper, numbers found in the quotes, the count and
    the sample size consistent with the percentage. Returns None when the estimate has no usable quote."""
    from research_agent.tools.extraction import quote_found

    quote = " ".join(str(est.get("quote") or "").split())[:500]
    if not quote or not quote_found(quote, words, shingles):
        return None
    nq = " ".join(str(est.get("sample_size_quote") or "").split())[:500]
    quotes = [quote] + ([nq] if nq and quote_found(nq, words, shingles) else [])
    cases = _stated(est.get("cases"), quotes)
    n = _stated(est.get("sample_size"), quotes)
    pct = _stated(est.get("percent"), [quote])
    out = {"condition_type": " ".join(str(est.get("condition_type") or "").split())[:80],
           "condition": condition_key(est.get("condition_type")),
           "population": str(est.get("population") or "")[:160], "country": str(est.get("country") or "")[:60],
           "setting": est.get("setting") if est.get("setting") in SETTINGS else "not_stated",
           "case_definition": est.get("case_definition") if est.get("case_definition") in CASE_DEFINITIONS
           else "not_stated",
           "is_overall": bool(est.get("is_overall")), "subgroup": str(est.get("subgroup") or "")[:80],
           "data_years": str(est.get("data_years") or "")[:30],
           "cases_as_written": str(est.get("cases") or "")[:30], "sample_size_as_written":
           str(est.get("sample_size") or "")[:30], "percent_as_written": str(est.get("percent") or "")[:30],
           "quote": quote, "sample_size_quote": quotes[1] if len(quotes) > 1 else "",
           "cases": None, "n": None, "derived": False, "problem": None}
    if n is not None and (n < 1 or n != int(n)):
        n = None
    if cases is not None and cases != int(cases):
        cases = None
    if pct is not None and not 0 <= pct <= 100:
        pct = None
    if n is None:
        out["problem"] = "no sample size found in the quotes: listed, not pooled"
    elif cases is not None:
        if cases > n:
            out["problem"] = "more cases than people studied: not pooled"
        elif pct is not None and abs(100 * cases / n - pct) > max(0.6, 0.02 * pct):
            out["problem"] = (f"{int(cases)}/{int(n)} is {100 * cases / n:.1f}%, but the paper writes {pct:g}%: "
                              "not pooled (one of the numbers belongs to something else)")
        else:
            out["cases"], out["n"] = int(cases), int(n)
    elif pct is not None:
        out["cases"], out["n"], out["derived"] = int(round(pct / 100 * n)), int(n), True
    else:
        out["problem"] = "neither a count nor a percentage found in the quote: listed, not pooled"
    return out


def check_rob(rob: dict, words: list[str], shingles: set) -> dict:
    """Each Hoy item: low or high only with a quote found in the paper; otherwise unclear."""
    from research_agent.tools.extraction import quote_found

    out = {}
    for key, _q in HOY_ITEMS:
        item = rob.get(key) if isinstance(rob, dict) else None
        item = item if isinstance(item, dict) else {}
        answer = item.get("answer") if item.get("answer") in ROB_ANSWERS else "unclear"
        quote = " ".join(str(item.get("quote") or "").split())[:300]
        if answer != "unclear" and not (quote and quote_found(quote, words, shingles)):
            answer, quote = "unclear", ""
        out[key] = {"answer": answer, "quote": quote if answer != "unclear" else ""}
    return out


def rob_summary(rob: dict) -> dict:
    high = sum(1 for v in rob.values() if v["answer"] == "high"
               or (UNCLEAR_COUNTS_AS_HIGH and v["answer"] == "unclear"))
    unclear = sum(1 for v in rob.values() if v["answer"] == "unclear")
    level = next(k for k, (lo, hi) in ROB_CUTOFFS.items() if lo <= high <= hi)
    return {"high_items": high, "unclear_items": unclear, "overall": level}


_CANON = [("rheumatoid arthritis", r"rheumatoid|\bra\b"), ("osteoarthritis", r"osteo-?arthr|\boa\b"),
          ("gout", r"\bgout"), ("psoriatic arthritis", r"psoria"), ("juvenile arthritis", r"juvenile|\bjia\b"),
          ("spondyloarthritis", r"spondyl"), ("septic arthritis", r"septic|infectious|bacterial arthr")]


def condition_key(name) -> str:
    """Estimates are pooled within one condition only. Arthritis types get a canonical name; anything else
    is grouped by its own lower-cased name."""
    low = " ".join(str(name or "").lower().split())
    for key, rx in _CANON:
        if re.search(rx, low):
            # "knee osteoarthritis" and "hand osteoarthritis" are different prevalences: keep the joint
            joint = re.search(r"\b(knee|hip|hand|spine|lumbar)\b", low)
            return f"{key} ({joint.group(1)})" if key == "osteoarthritis" and joint else key
    return low or "unknown"


# ---------------------------------------------------------------- the read (LLM, quotes checked)
def _read(llm, condition: str, text: str) -> dict:
    from research_agent.tools.extraction import _words

    resp = llm.chat(SYSTEM.format(condition=condition, max_est=MAX_ESTIMATES),
                    [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    tools=[_schema()], force_tool="record_prevalence",
                    max_tokens=max(settings.extraction_max_tokens, 2500))
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    if "_raw_arguments" in args:
        from research_agent.agents.base import repair_truncated_json

        args = repair_truncated_json(args["_raw_arguments"] or "") or {}
    words = _words(text)
    shingles = {tuple(words[i:i + k]) for k in (3, 4, 5) for i in range(len(words) - k + 1)}
    raw = [e for e in (args.get("estimates") or []) if isinstance(e, dict)][:MAX_ESTIMATES]
    checked = [c for c in (check_estimate(e, words, shingles) for e in raw) if c]
    rob = check_rob(args.get("risk_of_bias") or {}, words, shingles)
    return {"reports_prevalence": bool(args.get("reports_prevalence")) and bool(checked),
            "estimates": checked, "dropped_without_quote": len(raw) - len(checked),
            "risk_of_bias": rob, "rob_summary": rob_summary(rob)}


def prevalence_pass(ctx, condition: str, limit: int | None = None, refresh: bool = False) -> dict:
    """Read the run's counted papers for the prevalence of `condition`, then pool. Papers already read for
    this condition are not read again unless refresh=True."""
    from research_agent.ingestion.fulltext import fetch_fulltext, select_for_reading
    from research_agent.tools.extraction import _rows

    condition = " ".join((condition or "").split())[:120]
    if len(condition) < 3:
        return {"error": "name the condition, e.g. rheumatoid arthritis or arthritis"}
    key = _note_key(condition)
    before = (ctx.notes().get(key) or {}).get("papers") or {}
    rows = {r["paper_id"]: r for r in _rows(ctx)}
    ids = [p for p in rows if refresh or p not in before][: max(1, min(int(limit or MAX_PAPERS), MAX_PAPERS))]
    status = fetch_fulltext(ctx.pg, ids) if ids else {}
    texts = {r["paper_id"]: select_for_reading(r["sections"] or [], budget=settings.fulltext_read_chars)
             for r in ctx.pg.execute("SELECT paper_id, sections FROM paper_fulltext WHERE paper_id = ANY(%s) "
                                     "AND status='ok'", (ids or [""],)).fetchall()}
    meta = {r["paper_id"]: r for r in ctx.pg.execute(
        "SELECT paper_id, title, abstract, year, source AS corpus FROM papers WHERE paper_id = ANY(%s)",
        (ids or [""],)).fetchall()}
    llm = ctx.llm_factory(step="meta")
    setattr(llm, "_step", "meta")
    ctx.emit("meta", "start", {"task": f"reading {len(ids)} papers for the prevalence of {condition}"})
    papers = dict(before)
    failed = []

    def work(pid):
        p = meta[pid]
        text = f"Title: {p['title']}\n\nAbstract: {p['abstract']}"
        if texts.get(pid):
            text += f"\n\nFull-text sections:\n{texts[pid]}"
        res = _read(llm, condition, text)
        return pid, {**res, "read": "fulltext" if texts.get(pid) else "abstract", "year": p["year"],
                     "title": (p["title"] or "")[:200], "corpus": p["corpus"]}

    with ThreadPoolExecutor(max_workers=settings.extraction_workers) as pool:
        futures = [pool.submit(work, pid) for pid in ids if pid in meta]
        for fut in as_completed(futures):
            try:
                pid, res = fut.result()
                papers[pid] = res
            except Exception as exc:          # one failed read leaves that paper out, not the pass broken
                failed.append(str(exc)[:200])
                ctx.emit("meta", "error", {"error": str(exc)[:200]})
    papers = {p: v for p, v in papers.items() if p in rows}       # the cohort may have changed since
    note = {"condition": condition, "papers": papers, "failed": len(failed),
            "full_text_status": {s: sum(1 for v in status.values() if v == s) for s in set(status.values())}}
    ctx.save_note(key, note)
    result = pool_run(ctx, condition)
    ctx.emit("meta", "finish", {"output": {"studies_pooled": result.get("studies_pooled", 0)}})
    return result


def _note_key(condition: str) -> str:
    return "meta:" + re.sub(r"[^a-z0-9]+", "_", condition.lower()).strip("_")[:60]


# ---------------------------------------------------------------- the pool (code only)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
         29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    """Two-sided 95% quantile of Student's t (table, linear in 1/df between rows, 1.96 beyond)."""
    if df < 1:
        return float("nan")
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    if df > keys[-1]:
        lo, hi = keys[-1], None
    else:
        lo = max(k for k in keys if k < df)
        hi = min(k for k in keys if k > df)
    a, fa = 1 / lo, _T975[lo]
    b, fb = (1 / hi, _T975[hi]) if hi else (0.0, 1.959964)
    x = 1 / df
    return fa + (fb - fa) * (x - a) / (b - a)


def _logit(cases: int, n: int) -> tuple[float, float]:
    """Logit of the proportion and its variance; 0.5 is added to both cells when either is zero."""
    c = 0.5 if cases == 0 or cases == n else 0.0
    a, b = cases + c, n - cases + c
    return math.log(a / b), 1 / a + 1 / b


def _expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def wilson(cases: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% CI for one study's proportion (Wilson score), for the per-study table and the forest plot."""
    p = cases / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def random_effects(studies: list[tuple[int, int]]) -> dict | None:
    """Pool [(cases, n), ...]: logit scale, DerSimonian-Laird tau2, HKSJ confidence interval (never narrower
    than DL), I2 and a 95% prediction interval. None for fewer than two studies."""
    k = len(studies)
    if k < 2:
        return None
    ys, vs = zip(*(_logit(c, n) for c, n in studies))
    w = [1 / v for v in vs]
    sw = sum(w)
    y_fe = sum(wi * yi for wi, yi in zip(w, ys)) / sw
    q = sum(wi * (yi - y_fe) ** 2 for wi, yi in zip(w, ys))
    df = k - 1
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    ws = [1 / (v + tau2) for v in vs]
    sws = sum(ws)
    mu = sum(wi * yi for wi, yi in zip(ws, ys)) / sws
    se_dl = math.sqrt(1 / sws)
    q_hk = sum(wi * (yi - mu) ** 2 for wi, yi in zip(ws, ys)) / df
    se_hk = math.sqrt(max(q_hk, 1.0) / sws)               # the "never narrower than DL" form of HKSJ
    half = max(t975(df) * se_hk, 1.959964 * se_dl)
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    out = {"k": k, "n_total": sum(n for _, n in studies), "cases_total": sum(c for c, _ in studies),
           "pooled": _expit(mu), "ci": (_expit(mu - half), _expit(mu + half)),
           "tau2": tau2, "i2": i2, "q": q, "q_df": df, "pi": None}
    if k >= 3:
        pi_half = t975(k - 2) * math.sqrt(tau2 + se_dl ** 2)
        out["pi"] = (_expit(mu - pi_half), _expit(mu + pi_half))
    return out


def study_table(papers: dict) -> list[dict]:
    """Every checked estimate, one row each, with the paper it came from and that paper's risk of bias."""
    rows = []
    for pid, p in papers.items():
        for e in p.get("estimates") or []:
            rows.append({"paper_id": pid, "year": p.get("year"), "read": p.get("read"),
                         "rob": p["rob_summary"]["overall"], "rob_high_items": p["rob_summary"]["high_items"],
                         **e})
    return rows


def _one_per_study(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """For each paper and condition, its overall estimate (the largest sample when it gives several). A paper
    with only subgroup estimates contributes none: pooling its subgroups would count one sample twice."""
    chosen, left = {}, []
    for r in rows:
        if r["cases"] is None:
            continue
        key = (r["paper_id"], r["condition"])
        if not r["is_overall"]:
            left.append(r)
            continue
        if key not in chosen or r["n"] > chosen[key]["n"]:
            if key in chosen:
                left.append(chosen[key])
            chosen[key] = r
        else:
            left.append(r)
    return list(chosen.values()), left


def possible_duplicates(rows: list[dict]) -> list[list[str]]:
    """Papers that may report the same sample: same condition, same country and the same sample size."""
    groups: dict[tuple, set] = {}
    for r in rows:
        if r["n"]:
            groups.setdefault((r["condition"], r["country"].lower(), r["n"]), set()).add(r["paper_id"])
    return [sorted(v) for v in groups.values() if len(v) > 1]


def pool_run(ctx, condition: str) -> dict:
    """Pool what the pass recorded for this condition. Code only; no model call."""
    note = ctx.notes().get(_note_key(condition)) or {}
    papers = note.get("papers") or {}
    rows = study_table(papers)
    used, _subgroups = _one_per_study(rows)
    dupes = possible_duplicates(used)
    by_condition: dict[str, list[dict]] = {}
    for r in used:
        by_condition.setdefault(r["condition"], []).append(r)
    analyses = []
    for cond, rs in sorted(by_condition.items(), key=lambda kv: -len(kv[1])):
        overall = random_effects([(r["cases"], r["n"]) for r in rs])
        entry = {"condition": cond, "k": len(rs), "overall": overall, "studies": [r["paper_id"] for r in rs],
                 "subgroups": {}, "without_high_risk_of_bias": None}
        for dim in ("setting", "case_definition", "country"):
            groups: dict[str, list] = {}
            for r in rs:
                groups.setdefault(r[dim] or "not stated", []).append((r["cases"], r["n"]))
            if len(groups) > 1:
                entry["subgroups"][dim] = {g: (random_effects(v) or {"k": len(v), "single": v[0]})
                                           for g, v in groups.items()}
        kept = [(r["cases"], r["n"]) for r in rs if r["rob"] != "high"]
        if 0 < len(kept) < len(rs):
            entry["without_high_risk_of_bias"] = random_effects(kept) or {"k": len(kept)}
        analyses.append(entry)
    read = {p: v for p, v in papers.items()}
    out = {"condition": condition, "papers_read": len(read), "reads_failed": note.get("failed", 0),
           "papers_reporting_prevalence": sum(1 for v in read.values() if v.get("reports_prevalence")),
           "read_in_full": sum(1 for v in read.values() if v.get("read") == "fulltext"),
           "estimates_checked": len(rows), "studies_pooled": len(used),
           "not_pooled": [{"paper_id": r["paper_id"], "condition": r["condition"], "why": r["problem"]}
                          for r in rows if r["problem"]],
           "derived_counts": sum(1 for r in used if r["derived"]),
           "possible_duplicates": dupes, "analyses": analyses, "rows": rows,
           "method": METHOD_NOTE}
    ctx.save_note(_note_key(condition) + ":result", {k: v for k, v in out.items() if k != "rows"})
    return out


METHOD_NOTE = ("Random-effects meta-analysis of logit-transformed proportions: DerSimonian-Laird between-study "
               "variance, Hartung-Knapp-Sidik-Jonkman 95% confidence interval (never narrower than the "
               "DerSimonian-Laird one), I2, and a 95% prediction interval for three or more studies. One overall "
               "estimate per study and condition. Risk of bias by the Hoy et al. (2012) tool; overall judgement "
               "from the number of items at high risk (0-2 low, 3-4 moderate, 5 or more high).")


# ---------------------------------------------------------------- output
def _pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _ci(t) -> str:
    return "n/a" if not t else f"{100 * t[0]:.1f} to {100 * t[1]:.1f}%"


def markdown(ctx, condition: str | None = None) -> list[str]:
    """The meta-analysis section for a report or a draft. Empty when no pass has been run."""
    from research_agent.agents.report import _cite

    conds = [condition] if condition else [k.removeprefix("meta:") for k in ctx.notes()
                                           if k.startswith("meta:") and not k.endswith(":result")]
    L: list[str] = []
    for cond_key in conds:
        note = ctx.notes().get(_note_key(cond_key)) or {}
        if not note:
            continue
        res = pool_run(ctx, note.get("condition") or cond_key)
        L += [f"## Meta-analysis of reported prevalence: {res['condition']} (computed)", "",
              f"{res['papers_read']} analysed papers were read for the prevalence of {res['condition']} "
              f"({res['read_in_full']} in full). {res['papers_reporting_prevalence']} report a prevalence in their "
              f"own sample; {res['studies_pooled']} study estimates with a count and a sample size checked "
              "against the paper's own words are pooled"
              + (f" ({res['derived_counts']} counts derived from a stated percentage and sample size)"
                 if res["derived_counts"] else "") + ".", ""]
        if res["analyses"]:
            L += ["| Condition | Studies | People | Pooled prevalence | 95% CI | 95% prediction interval | I² |",
                  "|---|---|---|---|---|---|---|"]
            for a in res["analyses"]:
                o = a["overall"]
                if o:
                    L.append(f"| {a['condition']} | {o['k']} | {o['n_total']:,} | {_pct(o['pooled'])} | "
                             f"{_ci(o['ci'])} | {_ci(o['pi']) if o['pi'] else 'needs 3+ studies'} | "
                             f"{100 * o['i2']:.0f}% |")
                else:
                    r = next(x for x in res["rows"] if x["paper_id"] == a["studies"][0] and x["condition"] == a["condition"]
                             and x["cases"] is not None)
                    L.append(f"| {a['condition']} | 1 | {r['n']:,} | {_pct(r['cases'] / r['n'])} (one study, "
                             f"not pooled) | {_ci(wilson(r['cases'], r['n']))} | n/a | n/a |")
            L.append("")
            for a in res["analyses"]:
                for dim, groups in a["subgroups"].items():
                    parts = []
                    for g, s in groups.items():
                        if s.get("pooled") is not None:
                            parts.append(f"{g.replace('_', ' ')}: {_pct(s['pooled'])} ({s['k']} studies)")
                        elif s.get("single"):
                            c, n = s["single"]
                            parts.append(f"{g.replace('_', ' ')}: {_pct(c / n)} (1 study)")
                    if parts:
                        L.append(f"- **{a['condition']}, by {dim.replace('_', ' ')}:** " + "; ".join(parts) + ".")
                w = a.get("without_high_risk_of_bias")
                if w and w.get("pooled") is not None:
                    L.append(f"- **{a['condition']}, without studies at high risk of bias:** {_pct(w['pooled'])} "
                             f"(95% CI {_ci(w['ci'])}, {w['k']} studies).")
            L.append("")
            high_i2 = [a["condition"] for a in res["analyses"] if a["overall"] and a["overall"]["i2"] >= 0.75]
            if high_i2:
                L += [f"Heterogeneity is high for {', '.join(high_i2)} (I² of 75% or more): the studies disagree "
                      "more than chance explains, so the pooled figure is an average over different settings, not a "
                      "single true prevalence. The prediction interval shows the range a new study could find.", ""]
        L += ["**Studies**", "", "| Study | Condition | Setting | Case definition | Cases / people | Prevalence (95% CI) "
              "| Risk of bias |", "|---|---|---|---|---|---|---|"]
        for r in sorted(res["rows"], key=lambda x: (x["condition"], -(x["n"] or 0))):
            if r["cases"] is None:
                continue
            flag = " (derived)" if r["derived"] else ""
            sub = f", {r['subgroup']}" if not r["is_overall"] and r["subgroup"] else (", subgroup" if not r["is_overall"] else "")
            L.append(f"| {_cite(r['paper_id'])} | {r['condition']}{sub} | {r['setting'].replace('_', ' ')} | "
                     f"{r['case_definition'].replace('_', ' ')} | {r['cases']:,} / {r['n']:,}{flag} | "
                     f"{_pct(r['cases'] / r['n'])} ({_ci(wilson(r['cases'], r['n']))}) | {r['rob']} |")
        L.append("")
        notes = []
        if res.get("reads_failed"):
            notes.append(f"{res['reads_failed']} papers could not be read in the last pass and are missing; run the "
                         "pass again to read them.")
        if res["not_pooled"]:
            notes.append(f"{len(res['not_pooled'])} reported estimates were not pooled because their numbers could "
                         "not be checked (listed in the data export with the reason).")
        if res["possible_duplicates"]:
            notes.append("Possibly the same sample reported twice (same condition, country and sample size): "
                         + "; ".join(", ".join(_cite(p) for p in g) for g in res["possible_duplicates"])
                         + ". Check before citing the pooled figure.")
        notes.append(METHOD_NOTE)
        notes.append("This is a model-assisted analysis of the loaded corpus, not a registered systematic review: "
                     "every count is backed by a quote from the paper, but no second reviewer has checked the "
                     "extraction or the risk-of-bias judgements.")
        L += [" ".join(notes), ""]
    return L


def forest_plot(ctx, condition: str, path: str) -> str | None:
    """Forest plot (PNG): each study's prevalence with its 95% CI, and the pooled estimate per condition."""
    res = pool_run(ctx, condition)
    blocks = [a for a in res["analyses"] if a["overall"]]
    if not blocks:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from research_agent.agents.report import _cite

    rows_by = {(r["paper_id"], r["condition"]): r for r in res["rows"] if r["cases"] is not None and r["is_overall"]}
    lines = []
    for a in blocks:
        lines.append(("head", a["condition"], None, None, None))
        for pid in a["studies"]:
            r = rows_by.get((pid, a["condition"]))
            if r:
                lo, hi = wilson(r["cases"], r["n"])
                lines.append(("study", f"{_cite(pid)}  {r['cases']}/{r['n']}", r["cases"] / r["n"], lo, hi))
        o = a["overall"]
        lines.append(("pool", f"Pooled ({o['k']} studies, I² {100 * o['i2']:.0f}%)", o["pooled"], *o["ci"]))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(lines) + 1.2))
    for i, (kind, label, p, lo, hi) in enumerate(lines):
        y = len(lines) - i
        if kind == "head":
            ax.text(-0.02, y, label, fontsize=9, fontweight="bold", va="center", ha="right",
                    transform=ax.get_yaxis_transform())
            continue
        ax.text(-0.02, y, label, fontsize=8, va="center", ha="right", transform=ax.get_yaxis_transform())
        if kind == "study":
            ax.plot([100 * lo, 100 * hi], [y, y], color="#555", linewidth=1)
            ax.plot(100 * p, y, "s", color="#2f3e8f", markersize=4)
        else:
            ax.fill([100 * lo, 100 * p, 100 * hi, 100 * p], [y, y + 0.3, y, y - 0.3], color="#b0254f")
    ax.set_yticks([])
    ax.set_xlabel("Prevalence, %")
    ax.set_xlim(left=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Reported prevalence: {res['condition']}", fontsize=10, pad=12)
    ax.set_ylim(0.3, len(lines) + 0.7)
    fig.subplots_adjust(left=0.42, right=0.97)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_rows(ctx, condition: str) -> tuple[list[str], list[list]]:
    """Every checked estimate with its numbers, quotes and risk of bias: the dataset to re-analyse elsewhere."""
    note = ctx.notes().get(_note_key(condition)) or {}
    header = ["paper_id", "year", "read", "condition", "condition_as_written", "is_overall", "subgroup", "country",
              "setting", "case_definition", "cases", "sample_size", "derived", "percent_as_written", "problem",
              "quote", "sample_size_quote", "rob_overall", "rob_high_items"] + [f"rob_{k}" for k, _ in HOY_ITEMS]
    out = []
    for pid, p in (note.get("papers") or {}).items():
        for e in p.get("estimates") or []:
            out.append([pid, p.get("year"), p.get("read"), e["condition"], e["condition_type"], e["is_overall"],
                        e["subgroup"], e["country"], e["setting"], e["case_definition"], e["cases"], e["n"],
                        e["derived"], e["percent_as_written"], e["problem"] or "", e["quote"], e["sample_size_quote"],
                        p["rob_summary"]["overall"], p["rob_summary"]["high_items"]]
                       + [p["risk_of_bias"][k]["answer"] for k, _ in HOY_ITEMS])
    return header, out


def _pass_tool(ctx, condition: str, limit: int | None = None) -> dict:
    res = prevalence_pass(ctx, condition, limit=limit)
    if "error" in res:
        return res
    return {k: v for k, v in res.items() if k != "rows"}


META_TOOL = Tool(
    "prevalence_meta_analysis",
    "Read the analysed papers for the prevalence of one condition (cases, sample size, setting, case definition "
    "and Hoy risk of bias, every number checked against a quote) and pool them by random-effects meta-analysis "
    "in code. Use it when the question asks how common a condition is.",
    obj({"condition": {**STR, "description": "e.g. arthritis, rheumatoid arthritis, knee osteoarthritis"},
         "limit": {**INT, "description": f"How many papers to read (default and maximum {MAX_PAPERS})"}},
        ["condition"]),
    _pass_tool)
