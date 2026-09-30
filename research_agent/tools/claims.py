"""Claims and deterministic evidence checks.

Analysis agents do not write free-floating statements into the report. They *propose claims*, each
with a machine-checkable predicate. The Evidence layer evaluates predicates with plain code against
the extraction table (prevalence claims) or the corpus (trend claims). The Synthesis agent may only
state claims whose status is `supported`, and must quote the computed numbers.

prevalence predicate:
    {"field": "datasets", "any_of": ["MIMIC-III", "MIMIC-IV", "MIMIC"], "over": "all"|"stated",
     "min_share": 0.5, "max_share": null, "min_count": null, "max_count": null}
trend predicate:
    {"keywords": "\"foundation model\"", "early": [2018, 2020], "late": [2023, 2025],
     "direction": "increase"|"decrease", "min_ratio": 1.5}
"""
from __future__ import annotations

import json
import re

from research_agent.config import settings
from research_agent.tools.base import INT, NUM, STR, STRS, Tool, obj
from research_agent.tools.extraction import ENUM_FIELDS, LIST_FIELDS, SPECIAL_FIELDS, _rows, _values, known_fields
from research_agent.tools.trends import topic_series, window_ratio


def _validate_where(p: dict, lists: list[str], enums: dict) -> str | None:
    where = p.get("where")
    if where is None:
        return None
    if isinstance(where, dict):
        where = p["where"] = [where]
    if not isinstance(where, list):
        return "where must be a list of conditions like {field, any_of} or {field: 'year', min, max}"
    for c in where:
        if not isinstance(c, dict) or not c.get("field"):
            return "each where condition needs a field"
        f = c["field"]
        if f == "source":          # the paper's corpus; 'source' elsewhere means read depth
            c["field"] = f = "corpus"
        if f not in SPECIAL_FIELDS and f not in lists and f not in enums:
            return f"where field must be one of {lists + list(enums) + sorted(SPECIAL_FIELDS)}"
        if f == "year":
            if c.get("min") is None and c.get("max") is None:
                return "a year condition needs min and/or max"
        elif not (c.get("any_of") or c.get("none_of")):
            return f"where condition on {f} needs any_of or none_of"
    return None


# Some concepts are never recorded in some fields, so counting them there measures the form, not the papers.
# evaluation_metrics holds metric names (RMSE, AUC); how a model was tested lives in validation_level or a
# question-specific field. datasets holds named datasets, not the covariates a model uses.
_MISFIT = {
    "evaluation_metrics": (re.compile(r"cross[- ]?valid|out[- ]of[- ]sample|hold[- ]?out|test (set|period|data)|"
                                      r"train(ing)?[- /]test|split|spatial|temporal|external|prospective|"
                                      r"leave[- ]one|rolling|backtest|walk[- ]forward", re.I),
                           "evaluation_metrics lists metric names such as RMSE or AUC; how a model was tested is "
                           "not recorded there. Use validation_level or a question-specific field, or read the "
                           "papers (read_paper) before claiming a test design is rare."),
    "datasets": (re.compile(r"ndvi|evi\b|rainfall|precipitation|temperature|humidity|vegetation|covariate", re.I),
                 "datasets lists named datasets, not the covariates a model uses. Count covariates in "
                 "data_modalities or a question-specific field, or re-check with recheck_field."),
}


def field_misfit(field: str, any_of) -> str | None:
    rule = _MISFIT.get(field)
    if rule and any(rule[0].search(str(x)) for x in any_of or []):
        return rule[1]
    return None


def _validate(claim_type: str, p: dict, ctx=None) -> str | None:
    if claim_type == "prevalence":
        lists, enums = known_fields(ctx) if ctx is not None else (LIST_FIELDS, ENUM_FIELDS)
        if p.get("field") not in lists + list(enums):
            return f"field must be one of {lists + list(enums)}"
        if not p.get("any_of"):
            return "any_of must list at least one value (include synonyms/spellings)"
        if p.get("over", "all") not in ("all", "stated"):
            return "over must be 'all' or 'stated'"
        if all(p.get(k) is None for k in ("min_share", "max_share", "min_count", "max_count")):
            return "give at least one bound (min_share/max_share/min_count/max_count) so the claim is testable"
        misfit = field_misfit(p["field"], p["any_of"])
        if misfit:
            return misfit
        return _validate_where(p, lists, enums)
    if claim_type == "trend":
        for k in ("keywords", "early", "late", "direction"):
            if not p.get(k):
                return f"trend predicate needs {k}"
        if p["direction"] not in ("increase", "decrease"):
            return "direction must be increase or decrease"
        if p.get("within") is not None and not str(p["within"]).strip():
            return "within must be a topic query such as 'malaria', or left out"
        return None
    return "claim_type must be 'prevalence' or 'trend'"


_UPPER = re.compile(r"\b(less than|fewer than|below|under|at most|no more than|only|rare|rarely|few|"
                    r"minority|scarce|limited|lack|lacking|underrepresented|under-represented|gap|seldom|"
                    r"none|no study|no paper)\b", re.I)
_LOWER = re.compile(r"\b(more than|greater than|over|above|at least|most|majority|dominat\w*|"
                    r"common|widely|prevalent|frequent\w*|majority)\b", re.I)
_STRICT = re.compile(r"\b(less than|fewer than|more than|greater than|below|above)\b", re.I)


def bound_direction(text: str) -> str | None:
    """'upper' if the sentence claims something is small/rare, 'lower' if it claims it is large/common."""
    up, low = bool(_UPPER.search(text)), bool(_LOWER.search(text))
    if up and not low:
        return "upper"
    if low and not up:
        return "lower"
    return None


def direction_problem(text: str, p: dict) -> str | None:
    """Catch claims whose test points the wrong way, e.g. 'fewer than 20%' tested with min_share."""
    has_min = p.get("min_share") is not None or p.get("min_count") is not None
    has_max = p.get("max_share") is not None or p.get("max_count") is not None
    d = bound_direction(text)
    if d == "upper" and has_min and not has_max:
        return ("the claim says something is small/rare ('less than', 'few', 'only'...) but the predicate uses "
                "min_share/min_count, which tests the opposite. Use max_share or max_count.")
    if d == "lower" and has_max and not has_min:
        return ("the claim says something is large/common ('most', 'more than'...) but the predicate uses "
                "max_share/max_count, which tests the opposite. Use min_share or min_count.")
    return None


_CORPUS_WIDE = re.compile(r"\b(corpus|arxiv|all (?:\w+ )?papers|the (?:whole|entire) (?:field|literature))\b", re.I)
_SHORTLIST = re.compile(r"\b(shortlist\w*|analy[sz]ed|surveyed|reviewed|these \d* ?papers|the \d+ papers)\b", re.I)


def scope_problem(text: str) -> str | None:
    """Prevalence is counted over the analysed shortlist only, so the text must not claim a corpus-wide share."""
    if _CORPUS_WIDE.search(text) and not _SHORTLIST.search(text):
        return ("prevalence claims are counted over the analysed shortlist only, but the text talks about the "
                "corpus/arXiv as a whole. Rephrase it as 'in the N-paper shortlist ...', or use a trend claim for "
                "corpus-wide statements.")
    return None


def _bound_kind(p: dict) -> tuple[bool, bool]:
    return (p.get("min_share") is not None or p.get("min_count") is not None,
            p.get("max_share") is not None or p.get("max_count") is not None)


def _same_evidence(ctx, predicate: dict) -> dict | None:
    """An existing claim that counts exactly the same papers, over the same base, bounded the same way."""
    if not _rows(ctx):
        return None
    new = evaluate_prevalence(ctx, predicate)
    others = ctx.pg.execute(
        "SELECT id, text, predicate FROM claims WHERE run_id=%s AND claim_type='prevalence' AND status <> 'rejected'",
        (ctx.run_id,)).fetchall()
    for o in others:
        op = o["predicate"]
        if op.get("field") != predicate.get("field") or _bound_kind(op) != _bound_kind(predicate) \
                or (op.get("where") or []) != (predicate.get("where") or []):
            continue
        old = evaluate_prevalence(ctx, op)
        if set(old["matched_paper_ids"]) == set(new["matched_paper_ids"]) and old["denominator"] == new["denominator"]:
            return o
    return None


def _near_duplicate(ctx, predicate: dict) -> dict | None:
    """Not identical, but counting almost the same papers: worth telling the agent about."""
    if not _rows(ctx):
        return None
    new = set(evaluate_prevalence(ctx, predicate)["matched_paper_ids"])
    if not new:
        return None
    for o in ctx.pg.execute(
            "SELECT id, text, predicate FROM claims WHERE run_id=%s AND claim_type='prevalence' "
            "AND status <> 'rejected'", (ctx.run_id,)).fetchall():
        if o["predicate"].get("field") != predicate.get("field") or \
                (o["predicate"].get("where") or []) != (predicate.get("where") or []):
            continue
        old = set(evaluate_prevalence(ctx, o["predicate"])["matched_paper_ids"])
        if old and len(new & old) / len(new | old) >= 0.8:
            return o
    return None


def _measure(ctx, claim_type: str, predicate: dict, text: str | None = None) -> dict:
    """Evaluate a predicate now. With `text`, also check the text against the result. A rarity count that
    papers' own text contradicts triggers one focused re-check of those papers, then is measured again."""
    if claim_type == "prevalence":
        if not _rows(ctx):
            return {}
        r = evaluate_prevalence(ctx, predicate)
        if r.get("absence_problem") and getattr(ctx, "llm_factory", None) and settings.recheck:
            from research_agent.tools.recheck import recheck

            rc = recheck(ctx, predicate["field"], predicate["any_of"])
            r = evaluate_prevalence(ctx, predicate)
            r["recheck"] = {k: rc.get(k) for k in ("confirmed", "only_mentioned", "unclear", "not_checked")}
        summary = (f"{r['n_matching']} of {r['denominator']} papers ({r['share']:.0%}); "
                   f"bounds {'pass' if r['supported'] else 'fail'}")
        problem = text and (wording_problem(text, r["share"])
                            or text_count_problem(text, r["n_matching"], r["denominator"])
                            or percent_problem(text, r["share"]))
        problem = problem or r.get("absence_problem")
        return {"summary": summary, "text_problem": problem or None, "result": r}
    r = evaluate_trend(ctx, predicate)
    w = r["window"]
    unit = f"% of '{predicate['within']}' papers" if predicate.get("within") else " per 10k"
    summary = (f"{w.get('early_mean')} -> {w.get('late_mean')}{unit} (x{w.get('ratio_late_over_early')}), "
               f"{r['total_matching_papers']} matching papers; bounds {'pass' if r['supported'] else 'fail'}")
    problem = text and trend_text_problem(text, w.get("ratio_late_over_early"))
    return {"summary": summary, "text_problem": problem or None, "result": r}


def test_claim(ctx, claim_type: str, predicate: dict) -> dict:
    """Dry run: evaluate a predicate WITHOUT storing a claim, so the claim text can be written from the
    real numbers instead of guessed before them."""
    p = dict(predicate or {})
    err = _validate(claim_type, p, ctx)
    if err:
        return {"error": err}
    m = _measure(ctx, claim_type, p)
    if not m:
        return {"error": "no extractions yet for this run; run literature first"}
    r = m["result"]
    if claim_type == "prevalence":
        ctx.emit("test_claim", "count", {"n": r["n_matching"], "total": r["denominator"]})
        return {"n_matching": r["n_matching"], "denominator": r["denominator"], "share": r["share"],
                "over": r["over"], "bounds_pass": r["supported"], "matched_values": r["matched_values"][:15],
                "caveat": r["caveat"],
                "next": "Write the claim text using exactly these numbers, then call propose_claim."}
    return {"window": r["window"], "total_matching_papers": r["total_matching_papers"], "source": r["source"],
            "bounds_pass": r["supported"], "caveat": r["caveat"],
            "next": "Write the claim text using this measured ratio, then call propose_claim."}


_TOPIC_PAPERS = re.compile(r"\b(?:in|of|among|across)\s+(?:the\s+|all\s+)?([a-z][\w-]+)\s+"
                           r"(?:papers|studies|articles|literature|research|publications)\b", re.I)
_NOT_A_TOPIC = {"health", "all", "arxiv", "pmc", "corpus", "published", "recent", "these", "those", "shortlisted",
                "preprint", "other", "such", "earlier", "later", "new", "relevant", "analysed", "analyzed",
                "extracted", "matching", "biomedical", "medical", "scientific", "open-access", "total"}


def propose_claim(ctx, text: str, claim_type: str, predicate: dict, _agent: str = "unknown") -> dict:
    predicate = dict(predicate or {})
    err = _validate(claim_type, predicate, ctx)
    if err:
        return {"error": err}
    if claim_type == "trend":
        if re.search(r"\bshortlist", text, re.I):
            return {"error": "trend claims are measured on the whole corpus, not on the shortlist. Rephrase "
                             "the claim for the corpus, or use shortlist_years to describe the shortlist."}
        m = _TOPIC_PAPERS.search(text)
        if m and not predicate.get("within") and m.group(1).lower() not in _NOT_A_TOPIC:
            return {"error": f"The claim is about '{m.group(0)}', so the trend must be measured as a share of "
                             f"those papers: add within='{m.group(1)}' (or the topic's full query) to the "
                             "predicate. Without it the ratio is diluted by every other paper in the corpus."}
        if not predicate.get("source"):   # "In PMC, ..." means the trend should be measured inside PMC
            if re.search(r"\bPMC\b|PubMed Central", text):
                predicate["source"] = "pmc"
            elif re.search(r"\barXiv\b", text, re.I):
                predicate["source"] = "arxiv"
    if claim_type == "prevalence":
        problem = direction_problem(text, predicate) or scope_problem(text)
        if problem:
            return {"error": "Predicate does not match the claim text: " + problem}
        if _STRICT.search(text):
            predicate["strict"] = True  # 'less than 50%' must not pass at exactly 50%
        dup = ctx.pg.execute(
            "SELECT id FROM claims WHERE run_id=%s AND claim_type=%s AND predicate=%s::jsonb",
            (ctx.run_id, claim_type, json.dumps(predicate))).fetchone()
        if dup:
            return {"claim_id": dup["id"], "status": "duplicate",
                    "note": "An identical claim already exists; reuse its id instead of adding another."}
        near = _near_duplicate(ctx, predicate)
        same = _same_evidence(ctx, predicate)
        if same:
            return {"claim_id": same["id"], "status": "duplicate", "existing_text": same["text"],
                    "note": "An existing claim already counts exactly these papers the same way; cite "
                            f"[C{same['id']}] instead of adding another."}
    measured = _measure(ctx, claim_type, predicate, text)
    if measured.get("text_problem"):
        return {"error": "The claim text does not match what the data shows: " + measured["text_problem"],
                "measured": measured["summary"],
                "fix": "Rewrite the text from the measured numbers (test_claim shows them) and propose again."}
    r = ctx.pg.execute(
        "INSERT INTO claims (run_id, agent, text, claim_type, predicate) VALUES (%s,%s,%s,%s,%s::jsonb) RETURNING id",
        (ctx.run_id, _agent, text, claim_type, json.dumps(predicate))).fetchone()
    out = {"claim_id": r["id"], "status": "pending", "note": "Evidence agent will verify it."}
    if measured.get("summary"):
        out["measured_now"] = measured["summary"]
    if claim_type == "prevalence" and near:
        out["note"] += (f" Note: C{near['id']} already counts nearly the same papers (\"{near['text'][:80]}\"). "
                        "Prefer citing it unless this claim tests something different.")
    return out


_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"})


def _norm(text: str) -> str:
    """Lower-case and unify the many dash characters models use (U‑Net vs U-Net)."""
    return text.translate(_DASHES).lower().strip()


def _matcher(needle: str):
    """Short terms (CT, MRI, SAM, GAN, CNN) must match as whole words, so 'CT' does not hit 'electron
    microscopy' and 'SAM' does not hit 'sampling'. Longer terms keep substring matching (U-Net in U-Net++)."""
    if len(needle) <= 4:
        rx = re.compile(r"(?<![a-z0-9])" + re.escape(needle) + r"s?(?![a-z0-9])")
        return lambda v: bool(rx.search(v))
    return lambda v: needle in v


def _hits(r: dict, field: str, needles: list[str], exact: bool) -> list[str]:
    vals = _values(r["data"], field)
    if exact:
        return [v for v in vals if any(n == _norm(v) for n in needles)]
    matchers = [_matcher(n) for n in needles]
    return [v for v in vals if any(m(_norm(v)) for m in matchers)]


def row_matches(r: dict, cond: dict, enums) -> bool:
    """Does a paper satisfy one `where` condition? (corpus / read depth / year / any extracted field)"""
    f = "corpus" if cond["field"] == "source" else cond["field"]   # 'source' in a filter means the corpus
    if f == "year":
        y = r.get("year") or 0
        return (cond.get("min") is None or y >= int(cond["min"])) and (cond.get("max") is None or y <= int(cond["max"]))
    if f in ("corpus", "read"):
        have = {(r.get("corpus") if f == "corpus" else r.get("source")) or ""}
        have = {h.lower() for h in have}
        any_ok = not cond.get("any_of") or bool(have & {str(v).lower() for v in cond["any_of"]})
        none_ok = not (have & {str(v).lower() for v in cond.get("none_of") or []})
        return any_ok and none_ok
    exact = f in enums
    any_ok = not cond.get("any_of") or bool(_hits(r, f, [_norm(x) for x in cond["any_of"] if str(x).strip()], exact))
    none_ok = not cond.get("none_of") or not _hits(r, f, [_norm(x) for x in cond["none_of"] if str(x).strip()], exact)
    return any_ok and none_ok


def evaluate_prevalence(ctx, p: dict) -> dict:
    """Count papers whose `field` matches `any_of`. With `where`, count only within the subgroup of papers
    that meet every condition (e.g. PMC papers, papers from East Africa, papers read in full)."""
    all_rows = _rows(ctx)
    _, enums = known_fields(ctx)
    where = p.get("where") or []
    if isinstance(where, dict):
        where = [where]
    rows = [r for r in all_rows if all(row_matches(r, c, enums) for c in where)]
    field, needles = p["field"], [_norm(s) for s in p["any_of"] if s.strip()]
    exact = field in enums
    matched, matched_values = [], set()
    stated = 0
    for r in rows:
        if _values(r["data"], field):
            stated += 1
        hits = _hits(r, field, needles, exact)
        if hits:
            matched.append(r["paper_id"])
            matched_values.update(h.lower() for h in hits)
    over = p.get("over", "all")
    denom = stated if over == "stated" else len(rows)
    n = len(matched)
    share = (n / denom) if denom else 0.0
    checks = []
    strict = bool(p.get("strict"))
    if p.get("min_share") is not None:
        checks.append(share > float(p["min_share"]) if strict else share >= float(p["min_share"]))
    if p.get("max_share") is not None:
        checks.append(share < float(p["max_share"]) if strict else share <= float(p["max_share"]))
    if p.get("min_count") is not None:
        checks.append(n > int(p["min_count"]) if strict else n >= int(p["min_count"]))
    if p.get("max_count") is not None:
        checks.append(n < int(p["max_count"]) if strict else n <= int(p["max_count"]))
    abstract_only = sum(1 for r in rows if r["source"] == "abstract")
    ok = bool(denom) and all(checks)
    out = {
        "supported": ok, "n_matching": n, "denominator": denom, "over": over,
        "share": round(share, 3), "n_papers_extracted": len(all_rows), "n_field_stated": stated,
        "matched_paper_ids": matched[:100], "matched_values": sorted(matched_values)[:25],
        "caveat": (f"{abstract_only}/{len(rows)} papers were read from abstracts only; absence of a value "
                   f"often means 'not reported in the abstract'.") if abstract_only else "",
    }
    problem = _absence_problem(ctx, p, rows, n, denom, exact)
    if problem:
        out.update(supported=False, absence_problem=problem)
    if where:
        out["subgroup"] = {"conditions": where, "size": len(rows)}
        if len(rows) < 5:
            out["caveat"] = (out["caveat"] + " " if out["caveat"] else "") + \
                f"Subgroup has only {len(rows)} papers: too few to generalise."
    return out


def _asserts_rarity(p: dict, denom: int) -> bool:
    if p.get("max_share") is not None and float(p["max_share"]) <= 0.1:
        return True
    return p.get("max_count") is not None and int(p["max_count"]) <= max(2, 0.1 * denom)


def _absence_problem(ctx, p: dict, rows: list[dict], n: int, denom: int, exact: bool) -> str | None:
    """'0 of 100 list bed nets as a data source' can be an artifact: the extraction field may simply not
    record that kind of thing. For a rarity claim on a free-text field, count how many of the same papers
    mention the terms in their title or abstract; if many do, absence is not established."""
    # a count that comes out rare is checked even when the claim's wording did not assert rarity
    # ("a minority use NDVI" measured at 0 of 49 reads as absence in any report that quotes it)
    if exact or not rows or not (_asserts_rarity(p, denom) or (denom and n / denom <= 0.1)):
        return None
    from research_agent.tools.recheck import _terms, mentioning, verdicts

    terms = _terms(p["any_of"])
    if not terms:
        return None
    try:
        ids = mentioning(ctx.pg, [r["paper_id"] for r in rows], terms)
        # only a 'no' settles a paper (it merely mentions the term). A confirmed 'yes' must show up in the
        # count itself; if it somehow does not, the paper stays open rather than silently excused.
        settled = {pid for pid, v in verdicts(ctx.pg, ctx.run_id, p["field"], terms).items() if v == "no"}
    except Exception:   # the check is a safeguard; never let it break counting
        return None
    matched = {r["paper_id"] for r in rows if _hits(r, p["field"], [_norm(t) for t in terms], False)}
    open_ = ids - matched - settled
    if len(open_) >= 3 and len(ids) >= 2 * n + 3:
        return (f"{len(open_)} of these {len(rows)} papers mention {', '.join(terms[:4])} in their text but "
                f"their extracted '{p['field']}' does not record it, and no re-check has settled them. The "
                "field may not capture this, so rarity is not established. Run recheck_field on these terms, "
                "or report it as not measured rather than absent.")
    return None


def evaluate_trend(ctx, p: dict) -> dict:
    within = p.get("within")
    series = topic_series(ctx, p["keywords"], source=p.get("source"), within=within)["series"]
    w = window_ratio(series, tuple(p["early"]), tuple(p["late"]),
                     metric="pct_of_topic" if within else "per_10k_arxiv")
    ratio = w["ratio_late_over_early"]
    min_ratio = float(p.get("min_ratio") or 1.5)
    if ratio is None:
        ok = False
    elif ratio == "inf":
        ok = p["direction"] == "increase"
    else:
        ok = ratio >= min_ratio if p["direction"] == "increase" else ratio <= 1 / min_ratio
    total = sum(s["matching"] for s in series)
    src = (p.get("source") or "all").lower()
    caveats = [] if total >= 10 else ["fewer than 10 matching papers, too few to call a trend"]
    if src in ("pmc", "all") and not within:
        caveats.append(PMC_TREND_CAVEAT)
    return {"supported": ok and total >= 10, "window": w, "min_ratio": min_ratio, "total_matching_papers": total,
            "source": src, "within": within, "caveat": "; ".join(caveats)}


# The PMC part of the corpus is made of topic slices chosen when it was loaded (e.g. malaria), so the share of
# PMC papers on a topic tracks what was loaded, not the published literature. arXiv is loaded whole.
PMC_TREND_CAVEAT = ("PMC papers in this corpus were loaded as topic slices, so their shares over time reflect what "
                    "was loaded rather than the published literature; do not compare them with arXiv trends or "
                    "present them as field-wide")


_WORDING = [  # (pattern, test on the observed share, what the words promise)
    (re.compile(r"\b(vast|large|overwhelming) majority|near[- ]universal|nearly all|almost all|virtually all|"
                r"ubiquitous|overwhelmingly\b", re.I), lambda s: s >= 0.75, "at least 75%"),
    (re.compile(r"(?<!the )(?<!at )\b(most|majority)\b(?! (common|frequent|widely|popular|used|cited))", re.I),
     lambda s: s > 0.5, "more than half"),
    (re.compile(r"\b(almost never|very rare(ly)?|virtually no|hardly any|small minority)\b", re.I),
     lambda s: s <= 0.25, "at most 25%"),
]


_NEGATED = re.compile(r"\b(silent|not reported|unreported|do(es)? not|no |none|never|absent|lack\w*|omit\w*|"
                      r"fail\w* to|without)\b", re.I)
_THRESHOLD_BEFORE = re.compile(r"(at most|at least|fewer than|less than|more than|greater than|over|under|up to|"
                               r"about|around|approximately|roughly|~)\s*$", re.I)
_TEXT_COUNT = re.compile(r"(?<![\d.])(\d{1,4})\s*(?:/|of)\s*(?:the\s*)?(\d{1,4})(?![\d+])")
_LOWER_BOUND = re.compile(r"(at least|more than|over|no fewer than)\s*$", re.I)
_UPPER_BOUND = re.compile(r"(at most|fewer than|less than|under|no more than|up to)\s*$", re.I)


def wording_problem(text: str, share: float) -> str | None:
    """The predicate's threshold can be looser than the words ('large majority' tested with min_share 0.5).
    Hold the claim to what its wording promises, measured on the observed share. A claim about what is
    NOT there ('most papers are silent') is judged on the share that does not match."""
    negated = bool(_NEGATED.search(text))
    for rx, ok, promise in _WORDING:
        m = rx.search(text)
        if m and not ok(share) and not (negated and ok(1 - share)):
            return f"the wording '{m.group(0)}' implies {promise}, but the observed share is {share:.0%}"
    return None


def text_count_problem(text: str, n: int, denom: int) -> str | None:
    """A count written into the claim text ('(18/46 silent)') must be the count the check produces.
    Thresholds ('at most 4 of 60') are the claim's bound, not an asserted count, so they are skipped."""
    for m in _TEXT_COUNT.finditer(text):
        said_n, said_d = int(m.group(1)), int(m.group(2))
        if said_d != denom:
            if said_d >= 10 and said_n <= said_d:   # a paper population the check did not count
                return (f"the text counts out of {said_d} ('{m.group(0)}'), but the check counts {n} of {denom}. "
                        f"State the counted population, or measure the {said_d} papers you mean with `where` "
                        "or over='stated'.")
            continue
        if said_n == n:
            continue
        before = text[max(0, m.start() - 20):m.start()]
        if _LOWER_BOUND.search(before):          # "at least 8 of 33": the count must reach it
            if n >= said_n:
                continue
            return f"the text says at least {said_n} of {denom}, but the check counts {n}."
        if _UPPER_BOUND.search(before):          # "at most 4 of 60": the count must stay under it
            if n <= said_n:
                continue
            return f"the text says at most {said_n} of {denom}, but the check counts {n}."
        if _THRESHOLD_BEFORE.search(before):     # "about 8 of 60": vague, leave it
            continue
        if said_n == denom - n:                  # "32 of 46 do not" is the complement of 14 of 46
            continue
        return (f"the text states '{m.group(0)}' but the check counts {n} of {denom}. "
                "Revise the text to the counted number, or fix the predicate.")
    return None


_PERCENT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d)?)\s*%(?!\s*(?:ci\b|confidence|credible|prediction|"
                      r"uncertainty|interval))", re.I)
_PCT_THRESHOLD = re.compile(r"(at most|at least|fewer than|less than|more than|greater than|over|under|up to|"
                            r"above|below|exceed\w*|about|around|approximately|roughly|~|>|<|≥|≤)\s*$", re.I)


def percent_problem(text: str, share: float) -> str | None:
    """The first percentage written as a share ('15 (26%)') must be the measured share, or its complement.
    Thresholds ('above 95%', 'at least 30%') and interval levels ('95% CI') are not asserted shares."""
    for m in _PERCENT.finditer(text):
        before = text[max(0, m.start() - 20):m.start()]
        if _PCT_THRESHOLD.search(before):
            continue
        said = float(m.group(1))
        measured = 100 * share
        if abs(said - measured) <= 1.5 or abs(said - (100 - measured)) <= 1.5:
            return None
        return (f"the text says {m.group(0)} but the measured share is {measured:.0f}%. Revise the text to the "
                "measured share, or measure the population the percentage refers to.")
    return None


_RATIO_WORDS = re.compile(
    r"ratio\s*(?:of\s*)?(?P<op>>=|<=|≥|≤|>|<|~|≈|=)?\s*~?\s*(?P<r>\d+(?:\.\d+)?)"
    r"|(?P<f>\d+(?:\.\d+)?)\s*[-\u2011 ]?\s*fold\b"
    r"|(?<![\w.])(?P<x>\d+(?:\.\d+)?)\s*[x×](?!\w)", re.I)


def trend_text_problem(text: str, ratio) -> str | None:
    """A multiplier written into a trend claim ('a 9-fold rise', 'ratio ~1.1') must match the measured
    ratio. Only the first one is checked: later ones usually describe a comparison topic."""
    if ratio in (None, "inf"):
        return None
    m = _RATIO_WORDS.search(text)
    if not m:
        return None
    said = float(m.group("r") or m.group("f") or m.group("x"))
    before = text[max(0, m.start() - 20):m.start()].lower()
    op = (m.group("op") or "").strip()
    measured = float(ratio)
    if op in (">=", "≥", ">") or re.search(r"(at least|more than|over)\s*$", before):
        ok = measured >= said
    elif op in ("<=", "≤", "<") or re.search(r"(at most|less than|under)\s*$", before):
        ok = measured <= said
    else:
        ok = abs(measured / said - 1) <= 0.25 if said else measured == 0
    return None if ok else (f"the text says '{m.group(0).strip()}' but the measured ratio is ×{measured:g}. "
                            "Revise the number in the claim text.")


def evaluate(ctx, claim: dict) -> dict:
    p = claim["predicate"]
    if claim["claim_type"] != "prevalence":
        res = evaluate_trend(ctx, p)
        problem = trend_text_problem(claim.get("text") or "", res["window"].get("ratio_late_over_early"))
        if res["supported"] and problem:
            res.update(supported=False, wording_problem=problem)
        return res
    res = evaluate_prevalence(ctx, p)
    text = claim.get("text") or ""
    problem = (wording_problem(text, res["share"])
               or text_count_problem(text, res["n_matching"], res["denominator"])
               or percent_problem(text, res["share"]))
    if res["supported"] and problem:
        res.update(supported=False, wording_problem=problem + " Soften the wording or revise the claim.")
    return res


def verify_claims(ctx, claim_ids: list[int] | None = None) -> dict:
    sql = "SELECT * FROM claims WHERE run_id=%s AND status <> 'rejected'"
    params: list = [ctx.run_id]
    if claim_ids:
        sql += " AND id = ANY(%s)"; params.append(list(claim_ids))
    else:
        sql += " AND status = 'pending'"
    out = []
    for c in ctx.pg.execute(sql, params).fetchall():
        problem = direction_problem(c["text"], c["predicate"]) if c["claim_type"] == "prevalence" else None
        if problem:  # a mis-specified test cannot support OR refute the sentence
            ctx.pg.execute("UPDATE claims SET status='rejected', review_note=%s WHERE id=%s",
                           ("auto-rejected: " + problem, c["id"]))
            out.append({"claim_id": c["id"], "agent": c["agent"], "text": c["text"], "status": "rejected",
                        "reason": problem})
            continue
        res = evaluate(ctx, c)
        status = "supported" if res["supported"] else "unsupported"
        ctx.pg.execute("UPDATE claims SET status=%s, result=%s::jsonb WHERE id=%s",
                       (status, json.dumps(res, default=str), c["id"]))
        brief = {k: res[k] for k in ("n_matching", "denominator", "share", "matched_values", "caveat")
                 if k in res} if c["claim_type"] == "prevalence" else \
                {"window": res["window"], "total_matching_papers": res["total_matching_papers"]}
        out.append({"claim_id": c["id"], "agent": c["agent"], "text": c["text"], "status": status, **brief})
    return {"verified": len(out), "claims": out}


def list_claims(ctx, status: str | None = None) -> dict:
    sql = "SELECT id, agent, text, claim_type, predicate, status, review_note FROM claims WHERE run_id=%s"
    params: list = [ctx.run_id]
    if status:
        sql += " AND status=%s"; params.append(status)
    return {"claims": ctx.pg.execute(sql + " ORDER BY id", params).fetchall()}


def reject_claim(ctx, claim_id: int, reason: str) -> dict:
    ctx.pg.execute("UPDATE claims SET status='rejected', review_note=%s WHERE id=%s AND run_id=%s",
                   (reason, claim_id, ctx.run_id))
    return {"claim_id": claim_id, "status": "rejected"}


def revise_claim(ctx, claim_id: int, predicate: dict, note: str, text: str | None = None) -> dict:
    """Fix a predicate that does not faithfully test its claim text (e.g. missing synonyms).
    Allowed once per claim, and logged — revising until a claim passes is not allowed."""
    c = ctx.pg.execute("SELECT * FROM claims WHERE id=%s AND run_id=%s", (claim_id, ctx.run_id)).fetchone()
    if not c:
        return {"error": "no such claim"}
    if (c["review_note"] or "").startswith("revised:"):
        return {"error": "claim already revised once; reject it or keep the verdict"}
    err = _validate(c["claim_type"], predicate, ctx)
    if err:
        return {"error": err}
    ctx.pg.execute("UPDATE claims SET predicate=%s::jsonb, text=coalesce(%s, text), status='pending', "
                   "review_note=%s WHERE id=%s",
                   (json.dumps(predicate), text, f"revised: {note} | original predicate: {json.dumps(c['predicate'])}",
                    claim_id))
    return verify_claims(ctx, [claim_id])


PREDICATE_DOC = (
    "prevalence: {field, any_of:[synonyms], over:'all'|'stated', min_share|max_share|min_count|max_count, "
    "optional where:[{field, any_of|none_of} | {field:'year', min, max} | {field:'corpus', any_of:['pmc']} | "
    "{field:'read', any_of:['fulltext']}] to count within a subgroup}; "
    "trend: {keywords, early:[y1,y2], late:[y3,y4], direction:'increase'|'decrease', min_ratio, "
    "within:'<topic query>' (required for claims about '<topic> papers': measures the share of that topic's "
    "papers instead of all health papers)}"
)

TEST_TOOL = Tool(
    "test_claim",
    "Dry run: measure a claim's predicate WITHOUT saving it. Returns the real count (n of N, share) or trend "
    "ratio. Test every candidate predicate first (several in one turn is fine), then write each claim's text "
    "from the numbers returned. Predicate formats: " + PREDICATE_DOC + ".",
    obj({"claim_type": {"type": "string", "enum": ["prevalence", "trend"]}, "predicate": {"type": "object"}},
        ["claim_type", "predicate"]),
    test_claim, read_only=True,
)

PROPOSE_TOOL = Tool(
    "propose_claim",
    "Propose a checkable claim for the report. It will be verified by code before it can be used. Its text "
    "must agree with the measured numbers (use test_claim first) or it is refused. "
    "Predicate formats: " + PREDICATE_DOC + ". Include all synonyms/spellings in any_of.",
    obj({"text": STR, "claim_type": {"type": "string", "enum": ["prevalence", "trend"]},
         "predicate": {"type": "object"}}, ["text", "claim_type", "predicate"]),
    propose_claim,
)

EVIDENCE_TOOLS = [
    Tool("list_claims", "List this run's claims, optionally by status.",
         obj({"status": {"type": "string", "enum": ["pending", "supported", "unsupported", "rejected"]}}),
         list_claims),
    Tool("verify_claims", "Evaluate claims by code (all pending ones if no ids given). Returns n/N, shares, "
         "matched values and caveats.", obj({"claim_ids": {"type": "array", "items": INT}}), verify_claims),
    Tool("revise_claim", "Once per claim: replace a predicate that does not faithfully test the claim text "
         "(e.g. missing synonyms, wrong field). Re-verifies immediately. Explain why in note.",
         obj({"claim_id": INT, "predicate": {"type": "object"}, "note": STR, "text": STR},
             ["claim_id", "predicate", "note"]), revise_claim),
    Tool("reject_claim", "Reject a claim whose predicate cannot faithfully test it, or that overreaches.",
         obj({"claim_id": INT, "reason": STR}, ["claim_id", "reason"]), reject_claim),
]
_ = (NUM, STRS)
