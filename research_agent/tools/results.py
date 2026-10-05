"""Reported results and reported associations: what papers FOUND, not only what they did.

Extraction records, per paper:
  reported_results       a number the paper reports: metric, value, which model/drug/arm, baseline or not, split, setting
  reported_associations  a driver's reported effect on a health outcome: direction, lag, significance

Every item needs a quote copied from the paper, checked by code; a result's quote must also contain its
number, and an association's quote must name its driver. Items that fail are dropped, as with other fields.

On top of that, code (not the model) builds:
  results_table       the numbers, grouped by metric, with medians across papers (descriptive only)
  method_comparison   head-to-head results INSIDE the same paper (same data, same metric): the fair test
  contradictions      drivers with papers reporting opposite directions, and what separates the two sides
"""
from __future__ import annotations

import math
import re

from research_agent.tools.base import STR, STRS, Tool, obj

MAX_ITEMS = 12
STRUCT_FIELDS = ["reported_results", "reported_associations"]
SPLITS = ["test_or_holdout", "cross_validation", "in_sample", "not_stated"]
DIRECTIONS = ["positive", "negative", "none", "nonlinear", "mixed"]

# Measured quantities the extractor names itself (the `analyte` field). An abbreviation such as "E2" means
# oestradiol in one paper and prostaglandin E2 in another, which no pattern can tell apart: the model reading
# the paper decides, and code uses its answer only when the metric's name is not already unambiguous.
ANALYTES = ["total testosterone", "free testosterone", "oestradiol", "psa", "body weight",
            "waist circumference", "mic", "other"]

RESULT_SCHEMA = {
    "type": "array", "description": (
        "Main numbers the paper reports (up to 12): model performance (RMSE, AUC, accuracy, R2, sensitivity...), "
        "effect estimates (odds ratio, hazard ratio, relative risk, mean difference), laboratory and "
        "pharmacological measurements (MIC, IC50, EC50, Cmax, plasma AUC, half-life) and prevalences. One entry "
        "per number. quote must be copied word for word and must contain the number."),
    "items": {"type": "object", "properties": {
        "metric": STR, "value": {**STR, "description": "The number exactly as written, e.g. 0.87 or 12.4%"},
        "unit": {**STR, "description": "The unit exactly as written next to the number (ng/dL, nmol/L, mg/L, "
                 "kg, cm, years, %...), or '' for a dimensionless number such as an AUC, an odds ratio or a "
                 "correlation. Only the unit: never a number, a range or a confidence interval. Copy it; do "
                 "not convert it."},
        "analyte": {"type": "string", "enum": ANALYTES, "description": (
            "What is measured, when the number is a level or amount rather than a statistic. Decide from the "
            "paper's meaning, not the abbreviation: 'E2' is oestradiol only when it is the hormone estradiol; "
            "prostaglandin E2 (PGE2), E2 ubiquitin enzymes, a group or experiment called E2, or any other E2 "
            "is 'other'. Organ, tumour or birth weight is 'other', not body weight. Use 'other' for a "
            "statistic (AUC, odds ratio, RMSE...) and for anything not listed.")},
        "model": {**STR, "description": "Which model, method, drug, arm or group it belongs to"},
        "is_baseline": {"type": "boolean", "description": "true for a comparator: baseline model, placebo, "
                        "control arm"},
        "split": {"type": "string", "enum": SPLITS},
        "setting": {**STR, "description": "Place, dataset or subgroup the number is for"},
        "horizon": {**STR, "description": "Forecast lead time if stated, else ''"},
        "quote": STR}, "required": ["metric", "value", "model", "quote"]}}

ASSOCIATION_SCHEMA = {
    "type": "array", "description": (
        "Associations the paper reports between a driver (a risk factor, exposure, drug, dose or intervention, "
        "e.g. rainfall, prior antibiotic use, childhood adversity, sertraline dose, TSH level) and a health "
        "outcome (up to 12). Only what the paper states as its own finding. quote must be "
        "copied word for word and name the driver."),
    "items": {"type": "object", "properties": {
        "driver": STR, "outcome": {**STR, "description": "e.g. incidence, resistance, depressive symptoms, mortality"},
        "direction": {"type": "string", "enum": DIRECTIONS},
        "lag": {**STR, "description": "Lag if stated, e.g. '2 months', else ''"},
        "significant": {"type": "string", "enum": ["yes", "no", "not_stated"]},
        "quote": STR}, "required": ["driver", "direction", "quote"]}}

STRUCT_SCHEMA = {"reported_results": RESULT_SCHEMA, "reported_associations": ASSOCIATION_SCHEMA}

_NOT_BODY = (r"\b(prostate|prostatic|birth|organ|liver|hepatic|kidney|renal|testis|testes|testicular|testicle|"
             r"heart|cardiac|spleen|splenic|brain|uterine|uterus|ovarian|ovary|seminal vesicle|epididymal|"
             r"epididymis|adrenal|thymus|thymic|lung|placental|placenta|fetal|foetal|tumou?r|wet|dry|molecular|"
             r"gland|tissue|seed|egg|carcass|fat pad|muscle)")

# lower is better for errors; higher is better for skill scores
_METRICS = [
    ("rmse", r"\brmse\b|root mean squared? error", False), ("mae", r"\bmae\b|mean absolute error", False),
    ("mape", r"\bmape\b|mean absolute percentage", False), ("mse", r"(?<!r)\bmse\b|mean squared error", False),
    ("crps", r"\bcrps\b", False), ("wis", r"\bwis\b|weighted interval score", False),
    ("plasma auc", r"auc\s*[(_]?\s*(0|inf|tau|last|ss)|area under the (plasma |serum |blood )?"
                   r"(concentration|curve of (plasma|serum|blood))|\bng\s*[·.*]?\s*h|\bmg\s*[·.*]?\s*h", None),
    ("odds ratio", r"odds ratio|\baor\b|\bor\b", None), ("hazard ratio", r"hazard ratio|\bhr\b", None),
    ("relative risk", r"relative risk|risk ratio|\brr\b", None), ("mic", r"\bmic(50|90)?\b|minimum inhibitory", None),
    ("ic50", r"\bic50\b|half[- ]maximal inhibitory", None), ("ec50", r"\bec50\b", None),
    ("cmax", r"\bcmax\b|peak (plasma )?concentration", None), ("half-life", r"half[- ]life|\bt\s*1/2\b", None),
    ("prevalence", r"prevalence", None),
    # hormone levels: total and free testosterone differ by a factor of about fifty, so they are never one
    # bucket, and "serum testosterone" with no fraction named is treated as total, which is what papers mean
    ("free testosterone", r"free\s+(and\s+)?(serum\s+|plasma\s+)?testosterone|testosterone,?\s*free|\bfai\b|"
                          r"free androgen index", None),
    ("total testosterone", r"testosterone", None),
    # spelled out only: a bare "E2" is ambiguous (oestradiol, prostaglandin E2...) and is left to the
    # extractor's `analyte` label, see ANALYTES
    ("oestradiol", r"o?estradiol", None),
    ("psa", r"\bpsa\b|prostate[- ]specific antigen", None),
    ("waist circumference", r"waist", None),
    # "weight" alone is body weight, unless it is an organ's, a tumour's, a baby's at birth or a molecule's
    ("body weight", r"body ?weight|^(?!.*" + _NOT_BODY + r"\s*weights?\b).*\bweight\b(?!ed)", None),
    ("auc", r"\bau(roc|c)\b|area under", True), ("r2", r"\br\s*(2|²|squared)\b|coefficient of determination", True),
    ("accuracy", r"accuracy", True), ("sensitivity", r"sensitivity|recall", True),
    ("specificity", r"specificity", True), ("f1", r"\bf1\b|f-?score", True),
    ("correlation", r"correlation|pearson|spearman|\br\b", True), ("nse", r"\bnse\b|nash", True),
]
_DRIVERS = [
    ("rainfall", r"rain|precipitation"), ("temperature", r"temperature|\blst\b|\btemp\b"),
    ("humidity", r"humidity"), ("vegetation (NDVI/EVI)", r"ndvi|\bevi\b|vegetation"),
    ("bed nets (ITN/LLIN)", r"bed ?net|\bitn|\bllin|insecticide[- ]treated"),
    ("indoor residual spraying", r"\birs\b|residual spray"), ("elevation", r"elevation|altitude"),
    ("ENSO / sea surface temperature", r"enso|el ni|sea surface"), ("surface water / hydrology", r"water|flood|river|soil moisture|hydrolog"),
    ("wealth / socioeconomic status", r"wealth|poverty|socio|income"), ("urbanisation", r"urban"),
    ("housing quality", r"housing|house|roof"), ("health-care access", r"access|travel time|distance to"),
    ("mosquito density / entomology", r"mosquito|anophel|vector|entomolog|larva|\beir\b"),
]


# ---------------------------------------------------------------- units
# A number without its unit is not a measurement. 350 ng/dL and 12.1 nmol/L are the SAME testosterone
# level, and pooling them produced a median of 318 next to a median of 0.6 for the same quantity in one
# report. So each measured metric declares a canonical unit and the factors that reach it; a value whose
# unit cannot be read is kept with its quote but never enters a median.
#
# Molar conversions depend on the analyte's molecular weight, so they cannot be generic: each analyte
# carries its own table. Anything not listed here is pooled only with values sharing its written unit.
_CANONICAL = {
    "total testosterone": ("nmol/L", {"nmol/l": 1.0, "nm": 1.0, "ng/dl": 0.03467, "ng/ml": 3.467,
                                      "ng/l": 0.003467, "pg/ml": 0.003467, "µg/l": 3.467, "ug/l": 3.467,
                                      "mcg/l": 3.467}),
    "free testosterone": ("pmol/L", {"pmol/l": 1.0, "pg/ml": 3.467, "ng/dl": 34.67, "nmol/l": 1000.0}),
    "oestradiol": ("pmol/L", {"pmol/l": 1.0, "pg/ml": 3.671, "ng/l": 3.671, "nmol/l": 1000.0}),
    "psa": ("ng/mL", {"ng/ml": 1.0, "µg/l": 1.0, "ug/l": 1.0, "mcg/l": 1.0}),
    "body weight": ("kg", {"kg": 1.0, "g": 0.001, "lb": 0.4536, "lbs": 0.4536}),
    "waist circumference": ("cm", {"cm": 1.0, "mm": 0.1, "m": 100.0, "in": 2.54, "inch": 2.54}),
    "mic": ("mg/L", {"mg/l": 1.0, "µg/ml": 1.0, "ug/ml": 1.0, "mcg/ml": 1.0, "µg/l": 0.001, "ug/l": 0.001}),
}
# Dimensionless metrics with a known range. A value outside it is a reading error, not a result: one report
# carried an R2 of 2, which is arithmetically impossible and came from a number that was never an R2.
_BOUNDS = {"auc": (0.0, 1.0), "r2": (-1.0, 1.0), "accuracy": (0.0, 1.0), "sensitivity": (0.0, 1.0),
           "specificity": (0.0, 1.0), "f1": (0.0, 1.0), "correlation": (-1.0, 1.0), "nse": (-1e6, 1.0),
           "prevalence": (0.0, 1.0), "mape": (0.0, 1e4)}
_UNIT_CLEAN = re.compile(r"[\s()\[\]]+")


def normalise_unit(metric: str, unit: str, value: float | None) -> tuple[float | None, str | None]:
    """(value in the metric's canonical unit, that unit), or (None, None) when it cannot be converted.

    A metric with no canonical unit keeps its own written unit, so values are still only pooled with others
    measured the same way."""
    canon = _CANONICAL.get(metric)
    written = _UNIT_CLEAN.sub("", str(unit or "")).lower().replace("μ", "µ")
    if canon is None:
        return value, (written or None)
    target, factors = canon
    factor = factors.get(written)
    if factor is None:
        return None, None
    return (None if value is None else round(value * factor, 6)), target


def in_bounds(metric: str, value: float | None) -> bool:
    lo, hi = _BOUNDS.get(metric, (None, None))
    return value is None or lo is None or lo <= value <= hi


def resolve_metric(name: str, analyte: str | None = None) -> tuple[str, bool | None]:
    """The metric bucket for a reported number. The name decides when it is unambiguous (a pattern in
    _METRICS matches it); otherwise the extractor's `analyte` label does. That is how "E2" lands with
    oestradiol in an endocrinology paper and stays apart as "pge2" or "e2" everywhere else: the model that
    read the paper knows which one it meant, and code cannot."""
    key, higher = metric_key(name)
    if any(key == k for k, _, _ in _METRICS):
        return key, higher
    if analyte in ANALYTES and analyte != "other":
        if analyte == "body weight" and re.search(_NOT_BODY + r"\s*weights?\b", (name or "").lower()):
            return key, higher           # an organ's or a baby's weight is never body weight
        return analyte, None
    return key, higher


def metric_key(name: str) -> tuple[str, bool | None]:
    low = (name or "").lower()
    for key, rx, higher in _METRICS:
        if re.search(rx, low):
            return key, higher
    return low.strip() or "unknown", None


def driver_key(name: str) -> str:
    low = (name or "").lower()
    for key, rx in _DRIVERS:
        if re.search(rx, low):
            return key
    return low.strip()


def parse_number(value: str) -> float | None:
    """The first number in a value as written. A percentage is returned as a proportion (12.4% -> 0.124, and
    0.5% -> 0.005: a percentage below one is still a percentage, not a proportion already)."""
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(value or ""))
    if not m:
        return None
    try:
        x = float(m.group(0).replace(",", "."))
    except ValueError:
        return None
    return x / 100 if _is_percent(str(value)) else x


# Metrics that are proportions: stored and pooled on 0 to 1, whether the paper wrote 0.87 or 87%.
PROPORTIONS = {"auc", "accuracy", "sensitivity", "specificity", "f1", "prevalence"}
# What a unit can look like: letters, µ, %, /, ·, ^, superscripts and digits only AFTER a letter (m2, m³).
# Anything else written where a unit should be (a value, a range, a confidence interval) is not a unit.
_UNIT_SHAPE = re.compile(r"^(%|[a-zµμ°][a-zµμ°0-9²³⁻¹/·.*^ -]{0,22})$", re.I)
_NUMBER_THEN_UNIT = re.compile(r"-?\d+(?:[.,]\d+)?\s*(%|[a-zµμ°][a-zµμ°0-9²³⁻¹/·.*^-]{0,22})?", re.I)


def _is_percent(raw: str) -> bool:
    """Is the FIRST number in this text a percentage? ("12.4% (95% CI ...)" yes; "0.87 (95% CI 0.8-0.9)" no)"""
    m = re.search(r"-?\d+(?:[.,]\d+)?\s*(%)?", raw or "")
    return bool(m and m.group(1))


def read_unit(raw_value: str, unit_field: str) -> tuple[str | None, bool]:
    """(the unit as written, or None when there is none or it cannot be read; whether it is a percentage).

    The unit comes from the unit field, or failing that from the text right after the number in the value
    ("318 ng/dL"). It is never the value itself: an empty unit field used to fall back to the whole value, so
    "0.87" or "12.4%" was printed in the Unit column of the results table."""
    field = " ".join(str(unit_field or "").split()).strip(" ()[],;")
    if field:
        if field in ("%", "percent", "per cent", "percentage"):
            return "%", True
        return (field if _UNIT_SHAPE.match(field) else None), False
    m = _NUMBER_THEN_UNIT.search(str(raw_value or ""))
    if not m or not m.group(1):
        return None, False
    unit = m.group(1).strip(" .-")
    if unit == "%":
        return "%", True
    # words that follow a number without being its unit
    if unit.lower() in ("in", "of", "to", "and", "or", "vs", "versus", "at", "for", "ci", "with", "per", "x"):
        return None, False
    return (unit if _UNIT_SHAPE.match(unit) else None), False


def verify_structured(args: dict, text: str) -> tuple[dict, dict]:
    """Keep only result and association items whose quote is in the text (and carries the number / names
    the driver). Returns ({field: kept items}, {field: number dropped})."""
    from research_agent.tools.extraction import _words, quote_found

    words = _words(text)
    shingles = {tuple(words[i:i + k]) for k in (3, 4, 5) for i in range(len(words) - k + 1)}
    kept, dropped = {}, {}
    for field in STRUCT_FIELDS:
        items = args.get(field) if isinstance(args.get(field), list) else []
        good = []
        for it in items[:MAX_ITEMS]:
            if not isinstance(it, dict):
                continue
            quote = " ".join(str(it.get("quote") or "").split())[:400]
            if not quote or not quote_found(quote, words, shingles):
                continue
            if field == "reported_results":
                raw = str(it.get("value") or "").strip()
                num = re.search(r"\d+(?:[.,]\d+)?", raw)
                if not num or num.group(0) not in quote:
                    continue
                key, higher = resolve_metric(it.get("metric"), it.get("analyte"))
                unit_written, percent = read_unit(raw, it.get("unit"))
                value_num = parse_number(raw)
                if percent and not _is_percent(raw) and value_num is not None:
                    value_num = value_num / 100          # "87" with the unit "%" written separately
                # a bare "AUC" above 1 (and not a percentage) is a plasma exposure, not a ROC curve
                if key == "auc" and value_num is not None and value_num > 1 and not percent:
                    key, higher = "plasma auc", None
                if percent and key not in PROPORTIONS and value_num is not None:
                    value_num = round(value_num * 100, 6)  # a percentage that is not a proportion stays in %
                if not in_bounds(key, value_num):
                    continue            # outside what the metric can be: a reading error, not a result
                if key in PROPORTIONS:
                    canon_num, canon_unit = value_num, "proportion"
                else:
                    canon_num, canon_unit = normalise_unit(key, unit_written or "", value_num)
                good.append({"metric": key, "metric_as_written": str(it.get("metric") or "")[:60],
                             "value": raw[:40], "value_num": value_num,
                             "unit_as_written": (unit_written or "")[:24],
                             "unit": canon_unit, "value_canonical": canon_num,
                             "higher_is_better": higher, "model": str(it.get("model") or "")[:80],
                             "is_baseline": bool(it.get("is_baseline")),
                             "split": it.get("split") if it.get("split") in SPLITS else "not_stated",
                             "setting": str(it.get("setting") or "")[:80], "horizon": str(it.get("horizon") or "")[:40],
                             "quote": quote})
            else:
                driver = str(it.get("driver") or "").strip()
                tokens = [t for t in re.findall(r"[a-z]{4,}", driver.lower())] or [driver.lower()]
                if not driver or not any(t in quote.lower() for t in tokens):
                    continue
                good.append({"driver": driver_key(driver), "driver_as_written": driver[:60],
                             "outcome": str(it.get("outcome") or "")[:40],
                             "direction": it.get("direction") if it.get("direction") in DIRECTIONS else "mixed",
                             "lag": str(it.get("lag") or "")[:30],
                             "significant": it.get("significant") if it.get("significant") in ("yes", "no") else "not_stated",
                             "quote": quote})
        kept[field] = good
        if len([i for i in items[:MAX_ITEMS] if isinstance(i, dict)]) > len(good):
            dropped[field] = len([i for i in items[:MAX_ITEMS] if isinstance(i, dict)]) - len(good)
    return kept, dropped


# ---------------------------------------------------------------- analyses (code only)
def _rows(ctx):
    from research_agent.tools import claims

    return claims._rows(ctx)


def _median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _quartiles(xs: list[float]) -> tuple[float | None, float | None]:
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 4:
        return None, None
    return xs[len(xs) // 4], xs[(3 * len(xs)) // 4]


def results_table(ctx, metric: str | None = None, split: str | None = None, model_contains: str | None = None,
                  limit: int = 60) -> dict:
    """Reported performance numbers across the analysed papers. Values from different papers are on
    different data and scales, so the summary is descriptive; use method_comparison for fair comparisons."""
    want = metric_key(metric)[0] if metric else None
    rows, by_metric = [], {}
    for r in _rows(ctx):
        for it in r["data"].get("reported_results") or []:
            if want and it["metric"] != want:
                continue
            if split and it["split"] != split:
                continue
            if model_contains and model_contains.lower() not in it["model"].lower():
                continue
            rows.append({"paper_id": r["paper_id"], "year": r["year"],
                         **{k: it.get(k) for k in ("metric", "value", "unit_as_written", "unit", "model",
                                                   "is_baseline", "split", "setting", "horizon", "quote")}})
            # pooled per metric AND per unit: a median over mixed units is not a number about anything
            key = (it["metric"], it.get("unit"))
            by_metric.setdefault(key, []).append((r["paper_id"], it.get("value_canonical", it.get("value_num"))))
    summary, unconvertible = {}, 0
    for (m, unit), vals in by_metric.items():
        nums = [v for _, v in vals if v is not None]
        if m in _CANONICAL and unit is None:
            unconvertible += len(vals)       # its unit could not be read: kept with its quote, never pooled
            continue
        q1, q3 = _quartiles(nums)
        summary[f"{m} ({unit})" if unit else m] = {
            "metric": m, "unit": unit, "papers": len({p for p, _ in vals}), "values": len(vals),
            "median": _median(nums), "q1": q1, "q3": q3}
    out = {"summary": summary, "results": rows[:max(1, min(int(limit), 200))], "n_results": len(rows),
           "note": "Across-paper values come from different data, places and scales: describe them, do not "
                   "rank methods with them. Within-paper comparisons (method_comparison) are the fair test. "
                   "Values are pooled only within one metric and one unit."}
    if unconvertible:
        out["values_whose_unit_could_not_be_read"] = unconvertible
    return out


def _sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def method_comparison(ctx, method_a: list[str], method_b: list[str], metric: str | None = None,
                      rows: list[dict] | None = None) -> dict:
    """Head-to-head: papers that report the SAME metric for a model of family A and a model of family B
    (same data, same split). Counts how often A beats B, with a two-sided sign test."""
    a_terms = [t.lower() for t in method_a if t.strip()]
    b_terms = [t.lower() for t in method_b if t.strip()]
    want = metric_key(metric)[0] if metric else None
    pairs = []
    for r in (rows if rows is not None else _rows(ctx)):
        items = [it for it in r["data"].get("reported_results") or [] if it.get("value_num") is not None
                 and it.get("higher_is_better") is not None and (not want or it["metric"] == want)]
        for m in {it["metric"] for it in items}:
            same = [it for it in items if it["metric"] == m]
            for split in {it["split"] for it in same}:
                group = [it for it in same if it["split"] == split]
                a = [it for it in group if any(t in it["model"].lower() for t in a_terms)]
                b = [it for it in group if any(t in it["model"].lower() for t in b_terms)]
                a = [it for it in a if it not in b]
                if not a or not b:
                    continue
                higher = group[0]["higher_is_better"]
                best_a = max(a, key=lambda i: i["value_num"]) if higher else min(a, key=lambda i: i["value_num"])
                best_b = max(b, key=lambda i: i["value_num"]) if higher else min(b, key=lambda i: i["value_num"])
                if best_a["value_num"] == best_b["value_num"]:
                    winner = "tie"
                else:
                    winner = "A" if (best_a["value_num"] > best_b["value_num"]) == higher else "B"
                pairs.append({"paper_id": r["paper_id"], "metric": m, "split": split, "winner": winner,
                              "A": {"model": best_a["model"], "value": best_a["value"], "quote": best_a["quote"]},
                              "B": {"model": best_b["model"], "value": best_b["value"], "quote": best_b["quote"]}})
    per_paper: dict[str, str] = {}
    for p in pairs:   # one vote per paper: its test-set result when there is one
        if p["paper_id"] not in per_paper or p["split"] == "test_or_holdout":
            per_paper[p["paper_id"]] = p["winner"]
    wins = sum(1 for w in per_paper.values() if w == "A")
    losses = sum(1 for w in per_paper.values() if w == "B")
    return {"A": method_a, "B": method_b, "metric": want or "any", "papers_with_head_to_head": len(per_paper),
            "A_better": wins, "B_better": losses, "ties": sum(1 for w in per_paper.values() if w == "tie"),
            "sign_test_p": round(_sign_test_p(wins, losses), 4), "comparisons": pairs[:40],
            "note": "Each paper counts once. Few head-to-heads means little evidence either way."}


def associations(ctx, driver: str | None = None) -> dict:
    """Reported associations grouped by driver. Drivers are grouped by meaning, not only by the malaria list:
    "TRT", "testosterone therapy" and "testosterone replacement therapy" are one driver here (see
    tools/corroboration.group_phrases), which is what lets contradictions work outside malaria."""
    from research_agent.tools.corroboration import group_phrases

    rows = _rows(ctx)
    written = [it.get("driver_as_written") or it["driver"] for r in rows
               for it in r["data"].get("reported_associations") or []]
    canon = {w: driver_key(w) for w in written if driver_key(w) != w.lower().strip()}
    groups = group_phrases(written, canonical=canon) if written else {}

    def label(it):
        w = it.get("driver_as_written") or it["driver"]
        return groups.get(w, {}).get("group", it["driver"])

    want = None
    if driver:
        want = groups.get(driver, {}).get("group") or driver_key(driver)
    out: dict[str, dict] = {}
    for r in rows:
        seen = set()
        for it in r["data"].get("reported_associations") or []:
            name = label(it)
            if want and name != want and it["driver"] != want:
                continue
            key = (name, it["direction"])
            if key in seen:
                continue
            seen.add(key)
            d = out.setdefault(name, {d: [] for d in DIRECTIONS})
            d[it["direction"]].append({"paper_id": r["paper_id"], "lag": it["lag"], "significant": it["significant"],
                                       "quote": it["quote"]})
    return {"drivers": {k: {d: v for d, v in dirs.items() if v} for k, dirs in out.items()}}


def _attributes(r: dict) -> set[str]:
    d = r["data"]
    attrs = {f"place: {g}" for g in (d.get("geography") or [])}
    attrs |= {f"method: {m.lower()}" for m in (d.get("methods") or [])}
    attrs |= {f"read: {r['source']}", f"period: {'2020 or later' if (r.get('year') or 0) >= 2020 else 'before 2020'}"}
    for k, v in d.items():
        if k.startswith("q_") and isinstance(v, str) and v not in ("", "not_stated"):
            attrs.add(f"{k[2:]}: {v}")
    return attrs


def contradictions(ctx, min_each_side: int = 2) -> dict:
    """Drivers where papers report opposite directions (positive vs negative), each side with at least
    `min_each_side` papers, with the attributes that most separate the two sides (possible explanations,
    not proven ones)."""
    rows = {r["paper_id"]: r for r in _rows(ctx)}
    found = []
    for drv, dirs in associations(ctx)["drivers"].items():
        pos, neg = dirs.get("positive", []), dirs.get("negative", [])
        pos_ids = {x["paper_id"] for x in pos} - {x["paper_id"] for x in neg}
        neg_ids = {x["paper_id"] for x in neg} - {x["paper_id"] for x in pos}
        if len(pos_ids) < min_each_side or len(neg_ids) < min_each_side:
            continue
        contrasts = []
        attrs_pos = [_attributes(rows[p]) for p in pos_ids if p in rows]
        attrs_neg = [_attributes(rows[p]) for p in neg_ids if p in rows]
        for a in set().union(*attrs_pos, *attrs_neg):
            sp = sum(a in s for s in attrs_pos) / len(attrs_pos)
            sn = sum(a in s for s in attrs_neg) / len(attrs_neg)
            if abs(sp - sn) >= 0.5 and max(sum(a in s for s in attrs_pos), sum(a in s for s in attrs_neg)) >= 2:
                contrasts.append({"attribute": a, "share_positive_side": round(sp, 2), "share_negative_side": round(sn, 2)})
        contrasts.sort(key=lambda c: -abs(c["share_positive_side"] - c["share_negative_side"]))
        found.append({"driver": drv, "positive": [x for x in pos if x["paper_id"] in pos_ids],
                      "negative": [x for x in neg if x["paper_id"] in neg_ids],
                      "no_effect": dirs.get("none", []), "separating_attributes": contrasts[:6]})
    found.sort(key=lambda c: -(len(c["positive"]) + len(c["negative"])))
    return {"contradictions": found,
            "note": "Separating attributes are candidate explanations (setting, method, lag, period). Test one "
                    "with test_claim or test_hypothesis before stating it."}


def results_markdown(ctx) -> list[str]:
    """Code-built appendix: reported performance and conflicting findings, for the end of a report."""
    lines: list[str] = []
    rt = results_table(ctx, limit=200)
    if rt["summary"]:  # the table itself decides whether it has anything worth printing
        # A median over one paper's one number is not a summary of a literature, and fifty such rows read
        # as a results table while saying nothing. Only metrics several papers report are shown; the rest are
        # counted, and every number remains in the data export with its quote.
        shown = {k: v for k, v in rt["summary"].items() if v["papers"] >= 2}
        single = len(rt["summary"]) - len(shown)
        if shown:
            lines += ["## Reported performance (computed)", "",
                      "Numbers papers report for their own models, each backed by a quote and pooled only "
                      "within one metric and one unit. Values from different papers come from different data "
                      "and scales, so the medians describe the literature; they do not rank methods.", "",
                      "| Metric | Unit | Papers | Reported values | Median | Middle half |",
                      "|---|---|---|---|---|---|"]
            for m, s in sorted(shown.items(), key=lambda kv: -kv[1]["papers"]):
                mid = f"{_fmt(s['q1'])} to {_fmt(s['q3'])}" if s["q1"] is not None else "n/a"
                name = s["metric"]
                lines.append(f"| {name.upper() if len(name) <= 5 else name} | {s['unit'] or 'none'} | "
                             f"{s['papers']} | {s['values']} | {_fmt(s['median'])} | {mid} |")
            notes = []
            if single:
                notes.append(f"{single} further metrics are reported by a single paper each and are left out: "
                             "a median of one number describes nothing. They are in the data export, with "
                             "their quotes.")
            if rt.get("values_whose_unit_could_not_be_read"):
                notes.append(f"{rt['values_whose_unit_could_not_be_read']} values were excluded from these "
                             "medians because their unit could not be read, so they could not be put on one "
                             "scale.")
            lines += ["", " ".join(notes)] if notes else []
            lines.append("")
    co = contradictions(ctx)["contradictions"]
    if co:
        lines += ["## Where studies disagree (computed)", ""]
        for c in co:
            sep = "; ".join(f"{x['attribute']} ({round(100 * x['share_positive_side'])}% of positive vs "
                            f"{round(100 * x['share_negative_side'])}% of negative)" for x in c["separating_attributes"][:3])
            lines.append(f"- **{c['driver']}**: {len(c['positive'])} studies report a positive association "
                         f"({_ids(c['positive'])}) and {len(c['negative'])} a negative one ({_ids(c['negative'])})"
                         + (f"; {len(c['no_effect'])} report no effect" if c["no_effect"] else "")
                         + (f". What differs between the two sides: {sep}." if sep else "."))
        lines.append("")
    return lines


def _fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.3g}"


def _ids(items, k=4):
    from research_agent.agents.report import _cite

    ids = list(dict.fromkeys(x["paper_id"] for x in items))
    return ", ".join(_cite(p) for p in ids[:k]) + (" …" if len(ids) > k else "")


RESULT_TOOLS = [
    Tool("results_table", "Performance numbers the analysed papers report (RMSE, AUC, accuracy...), each with its "
         "quote, plus per-metric medians. Descriptive only: values from different papers are not comparable.",
         obj({"metric": STR, "split": {"type": "string", "enum": SPLITS}, "model_contains": STR,
              "limit": {"type": "integer"}}), results_table, read_only=True, max_chars=16000),
    Tool("method_comparison", "Fair comparison of two method families: papers that report the same metric for "
         "both (same data), how often A beats B, with a sign test. E.g. A=['lstm','transformer'], "
         "B=['arima','sarima'].", obj({"method_a": STRS, "method_b": STRS, "metric": STR}, ["method_a", "method_b"]),
         method_comparison, read_only=True, max_chars=16000),
    Tool("contradictions", "Drivers (rainfall, temperature, bed nets...) where papers report opposite directions, "
         "with quotes, and the attributes (place, method, period, read depth) that separate the two sides.",
         obj({}), contradictions, read_only=True, max_chars=16000),
]
