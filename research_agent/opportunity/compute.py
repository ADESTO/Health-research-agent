"""The code-computed part of the Research Opportunity Map.

Nothing here asks an LLM for an opinion. Each section is a rule over the extracted, evidence-checked
records of the shortlisted papers, and every grade carries the reasons that produced it:

- established: common (>= 30% of papers, >= 5 papers)
- emerging:    less common, but its share among recent papers is >= 1.5x its share among earlier ones
- missing:     values the protocol marks as desirable (plus external/prospective validation and open code
               or data) that almost no paper has
- evidence strength (established / emerging) and gap confidence (missing) are point scores from paper
  counts, full-text confirmation, agreement across arXiv and PMC, regional spread, recency, how often the
  field is reported at all, and a corpus-wide keyword check
- novelty:     pairs of components from DIFFERENT dimensions (e.g. a method and a validation design) that
               are each common but almost never occur together, with the probability of seeing so few
               co-occurrences by chance if the two were chosen independently (hypergeometric)

Sampling uncertainty is reported, not hidden: every share carries a 95% Wilson interval, a gap reports
the upper bound of its true rate, and an "emerging" rise carries a one-sided Fisher exact p-value.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from research_agent.tools.claims import _hits, _norm
from research_agent.tools.extraction import _rows, _values, known_fields

THRESHOLDS = {
    "established_min_share": 0.30, "established_min_n": 5,
    "emerging_min_n": 3, "emerging_min_ratio": 1.5, "recent_years": 5,
    "missing_max_share": 0.05, "missing_max_n": 2,
    "novelty_min_component_n": 3, "novelty_max_observed": 1, "novelty_min_expected": 1.5,
    "novelty_max_p": 0.2, "corpus_min_papers": 5, "emerging_max_p": 0.2,
    # two components are the same fact under two names when their paper sets overlap this much
    "mirror_min_jaccard": 0.8,
    # a stratum needs at least this many papers to say anything about what co-occurs inside it
    "stratum_min_papers": 4,
    # a watch item must at least lean the right way; at p = 1 the data says nothing is happening
    "watch_max_p": 0.9,
}

# Generic statistics are not methods of the kind the map is about ("Pearson correlation" is how a paper
# analyses results, not a modelling approach), so they never count as established, emerging or novel.
_GENERIC_STATS = re.compile(
    r"^(pearson|spearman|kendall)?\s*(rank\s*)?correlation( analysis| coefficient)?$|descriptive statistic|"
    r"\b(t|chi[- ]square|mann[- ]kendall|wilcoxon|kruskal[- ]wallis|shapiro[- ]wilk|granger)[- ]test\b|"
    r"^anova$|^statistical analysis$|^regression analysis$|^sensitivity analysis$|^cross[- ]validation$|"
    r"^hypothesis test", re.I)

# Values that record an absence ("none", "not stated", "point estimate only") are not components anyone
# can combine, so they never enter novelty pairs.
_ABSENCE = re.compile(r"(^|_)(none|no|not_stated|not_reported|unknown)($|_)|^point_estimate", re.I)

# Free-text values that name the same thing. Counted as one item so the map does not list
# "satellite imagery" and "remote sensing" as two separate trends.
_SYNONYMS = {
    "data_modalities": [
        ("satellite / remote sensing", r"satellite|remote[- ]sens|earth observation|\bmodis\b|landsat|sentinel"),
        ("electronic health records", r"\behr\b|electronic (health|medical) record|clinical record"),
        ("climate / weather", r"climat|weather|meteorolog|rainfall|temperature|reanalysis"),
        ("geospatial / GIS", r"geospatial|\bgis\b|spatial covariate"),
    ],
    "methods": [
        ("random forest", r"random forest"),
        ("gradient boosting", r"gradient boost|xgboost|lightgbm|catboost"),
        ("LSTM / recurrent network", r"\blstm\b|long short[- ]term memory|recurrent neural|\bgru\b"),
        ("ARIMA family", r"\b(s)?arima(x)?\b"),
    ],
}


def _is_code_field(name: str) -> bool:
    return bool(re.search(r"code|shar|availab|reproduc|open_data", name, re.I))


# Dimensions: novelty only pairs components from different dimensions, so "statistical time series" (a
# model class) is never "novelly combined" with "random forest" (also a model class).
_ROLE_HINTS = [("method", r"model|method|algorithm|architecture|approach|technique|learner"),
               ("data", r"data|input|predictor|covariate|source|modalit|feature"),
               ("evaluation", r"valid|baseline|metric|evaluat|benchmark|skill|calibrat|uncertain|probabil"),
               ("outcome", r"target|outcome|endpoint|label"),
               ("setting", r"spatial|region|geograph|setting|population|scale|unit|horizon|lead"),
               ("deployment", r"operational|deploy|readiness|implement|translation"),
               ("reporting", r"report|delay|missing|share|code|open")]
BASE_ROLES = {"methods": "method", "data_modalities": "data", "validation_level": "evaluation",
              "code_or_data_available": "reporting"}


def role_of(field: dict | str) -> str:
    if isinstance(field, str):
        return BASE_ROLES.get(field, field)
    if field.get("role"):
        return field["role"]
    name = field["name"].removeprefix("q_")
    for role, rx in _ROLE_HINTS:
        if re.search(rx, name, re.I):
            return role
    return field["name"]


# ---------------------------------------------------------------- novelty: the null has to be right
def strata(rows: list[dict], protocol: dict, min_papers: int) -> list[tuple[str, list[set]]]:
    """Ways the corpus splits into groups that cannot overlap, one entry per single-choice enum field:
    (field name, [paper-id set per value]). Only levels with enough papers to say anything are kept."""
    out = []
    for f in protocol.get("fields", []):
        if f["type"] != "enum":
            continue
        groups = []
        for v in f["values"]:
            if v == "not_stated":
                continue
            ids = {r["paper_id"] for r in rows if v.lower() in {x.lower() for x in _values(r["data"], f["name"])}}
            if len(ids) >= min_papers:
                groups.append(ids)
        if len(groups) >= 2:
            out.append((f["name"], groups))
    return out


def stratified_expected(ia: set, ib: set, groups: list[set]) -> float:
    """How many papers would use BOTH components if, INSIDE each group, papers chose them independently.

    This is the number the novelty claim needs, and it is not the same as n_a * n_b / N. When a corpus is
    really two literatures that never meet (animal experiments and human trials, say), a component confined
    to one of them and a component confined to the other never co-occur, and the plain formula calls that a
    striking absence. Stratified, each group contributes only what that group could have produced: the
    animal group has no women's trials and the human group has no animal experiments, so the expectation is
    zero and the pair says nothing. A combination that is genuinely untried behaves differently, because the
    group where both components live still expects to see it."""
    total = 0.0
    for ids in groups:
        n = len(ids)
        if n:
            total += len(ia & ids) * len(ib & ids) / n
    return total


def explained_by_a_stratum(ia: set, ib: set, all_strata, min_expected: float) -> str | None:
    """The field, if any, whose levels on their own account for two components never appearing together.

    One such field is enough, exactly as one rival explanation is enough to drop a researcher's finding: if
    knowing which kind of study a paper is already predicts that it cannot use both, their absence together
    is arithmetic rather than an opportunity."""
    for field, groups in all_strata:
        if stratified_expected(ia, ib, groups) < min_expected:
            return field
    return None


def _collapse_mirrors(items: list[dict], ids: list[set], min_jaccard: float) -> tuple[list[dict], int]:
    """Fold established items that cover the same papers into one, keeping the better-evidenced wording.

    Two fields often record one fact: a population of `animal model` and a design of
    `animal or in vitro experiment` are the same eleven papers. Reported separately they make a run look
    like it found two things. The survivor carries the others under `also_recorded_as` so nothing is lost."""
    keep: list[int] = []
    dropped = 0
    for i, item in enumerate(items):
        twin = next((k for k in keep if mirrors(item, items[k], ids[i], ids[k], min_jaccard)), None)
        if twin is None:
            keep.append(i)
            continue
        # keep whichever is better confirmed: more full-text reads, then the larger paper count
        better = i if (item.get("n_fulltext", 0), item.get("n", 0)) > \
            (items[twin].get("n_fulltext", 0), items[twin].get("n", 0)) else twin
        other = twin if better == i else i
        items[better].setdefault("also_recorded_as", []).append(items[other]["label"])
        keep[keep.index(twin)] = better
        dropped += 1
    order = sorted(keep, key=lambda k: items[k].get("share", 0), reverse=True)
    return [items[k] for k in order], dropped


_MIRROR_STOP = {"only", "other", "stated", "reported", "none", "with", "without", "study", "studies", "type",
                "level", "general", "unspecified", "multiple", "single", "both", "mixed", "true", "false"}


def _concept_words(label: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", label.lower()) if len(w) >= 4 and w not in _MIRROR_STOP}


def mirrors(a: dict, b: dict, ia: set, ib: set, min_jaccard: float) -> bool:
    """Are these two items the same fact recorded in two fields?

    Two conditions, and both are needed. The paper sets must be nearly identical, AND the two values must
    share a word. The second condition is what keeps this honest: in a twenty-paper run plenty of unrelated
    fields coincide exactly by chance, and folding those together would hide real findings. `animal model`
    under population and `animal or in vitro experiment` under design share "animal" and are one fact; a
    one-month horizon and the use of satellite data may cover the same papers and are still two facts."""
    union = ia | ib
    if not union or len(ia & ib) / len(union) < min_jaccard:
        return False
    return bool(_concept_words(str(a.get("value") or a.get("label") or ""))
                & _concept_words(str(b.get("value") or b.get("label") or "")))


# ---------------------------------------------------------------- small-sample statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (sensible at 0 and with small n)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def _hyper_pmf(x: int, N: int, K: int, n: int) -> float:
    if x < max(0, n - (N - K)) or x > min(K, n):
        return 0.0
    return math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)


def hyper_le(x: int, N: int, K: int, n: int) -> float:
    """P(X <= x) for X ~ Hypergeometric(population N, K successes, n draws)."""
    return min(1.0, sum(_hyper_pmf(i, N, K, n) for i in range(0, x + 1)))


def hyper_ge(x: int, N: int, K: int, n: int) -> float:
    """P(X >= x): one-sided Fisher exact test."""
    return min(1.0, sum(_hyper_pmf(i, N, K, n) for i in range(x, min(K, n) + 1)))

# Practices every question cares about, each counted as ONE gap (any of its values fills it)
BASE_DESIRABLE = [
    ("validation_level", ["external", "prospective", "clinical_trial"],
     "validation beyond the development data (external, prospective or trial)"),
    ("code_or_data_available", ["yes"], "code or data shared"),
]


@dataclass
class Feature:
    field: str
    value: str
    label: str
    exact: bool
    desirable: bool = False
    search: str = ""
    values: list | None = None   # several values that all count as this feature (a practice group)
    role: str = ""               # dimension (method, data, evaluation...) used to keep novelty pairs meaningful
    members: list | None = None  # readable member values of a practice group

    def needles(self) -> list[str]:
        return [_norm(v) for v in (self.values or [self.value])]


def _label(field: str, value: str) -> str:
    return f"{field.removeprefix('q_').replace('_', ' ')}: {value.replace('_', ' ')}"


def _features(rows: list[dict], protocol: dict, enums: dict) -> list[Feature]:
    feats: list[Feature] = []
    for f in protocol.get("fields", []):
        role = role_of(f)
        grouped = {v for g in f.get("groups", []) for v in g["values"]}
        if f["type"] == "enum":
            values = [v for v in f["values"] if v != "not_stated"]
        else:
            counts: dict[str, int] = {}
            for r in rows:
                for v in _values(r["data"], f["name"]):
                    counts[v.lower()] = counts.get(v.lower(), 0) + 1
            values = sorted(counts, key=counts.get, reverse=True)[:12] + [d for d in f["desirable"]
                                                                          if d not in counts]
        for v in values:
            # a grouped value still counts on its own for established/emerging, but its gap is the group's
            feats.append(Feature(f["name"], v, _label(f["name"], v), f["type"] == "enum",
                                 v in f["desirable"] and v not in grouped, f.get("search", {}).get(v, ""),
                                 role=role))
        for g in f.get("groups", []):
            search = " OR ".join(f"({q})" for q in (f.get("search", {}).get(v) for v in g["values"]) if q)
            feats.append(Feature(f["name"], "|".join(g["values"]),
                                 f"{f['name'].removeprefix('q_').replace('_', ' ')}: {g['label']}",
                                 f["type"] == "enum", True, search, values=g["values"], role=role,
                                 members=[v.replace("_", " ") for v in g["values"]]))
    # a protocol field about code/data sharing replaces the general one, so it is not counted twice
    has_code_field = any(_is_code_field(f["name"]) for f in protocol.get("fields", []))
    for field, values, label in BASE_DESIRABLE:
        if field == "code_or_data_available" and has_code_field:
            continue
        feats.append(Feature(field, "|".join(values), label, True, True, values=values, role=role_of(field)))
    for field, top in (("methods", 12), ("data_modalities", 10)):
        counts: dict[str, int] = {}
        for r in rows:
            for v in {x.lower() for x in _values(r["data"], field)}:
                if not _GENERIC_STATS.search(v.strip()):
                    counts[v] = counts.get(v, 0) + 1
        groups: dict[str, list[str]] = {}
        for v in counts:
            label = next((lab for lab, rx in _SYNONYMS.get(field, []) if re.search(rx, v, re.I)), None)
            if label:
                groups.setdefault(label, []).append(v)
        grouped = {v for vs in groups.values() for v in vs}
        items = [(counts[v], Feature(field, v, _label(field, v), False, role=role_of(field)))
                 for v in counts if v not in grouped]
        for label, vs in groups.items():
            members = sorted(vs, key=counts.get, reverse=True)
            feat = Feature(field, "|".join(members), f"{field.replace('_', ' ')}: {label}", False,
                           values=members, role=role_of(field),
                           members=members if len(members) > 1 else None)
            items.append((sum(counts[v] for v in vs), feat))
        for _, feat in sorted(items, key=lambda x: -x[0])[:top]:
            feats.append(feat)
    return feats


def _stats(rows: list[dict], feat: Feature, cut_year: int) -> dict:
    match = [r for r in rows if _hits(r, feat.field, feat.needles(), feat.exact)]
    ft = [r for r in rows if r["source"] == "fulltext"]
    recent = [r for r in rows if (r["year"] or 0) >= cut_year]
    early = [r for r in rows if (r["year"] or 0) < cut_year]
    n_recent = sum(1 for r in match if (r["year"] or 0) >= cut_year)
    n_early = len(match) - n_recent
    stated = sum(1 for r in rows if _values(r["data"], feat.field))
    geos = {g.lower() for r in match for g in _values(r["data"], "geography")}
    lo, hi = wilson(len(match), len(rows))
    p_rise = hyper_ge(n_recent, len(rows), len(match), len(recent)) if match and recent else 1.0
    return {
        "n": len(match), "N": len(rows), "share": round(len(match) / len(rows), 3) if rows else 0.0,
        "ci95": [round(lo, 3), round(hi, 3)], "p_rise": round(p_rise, 4),
        "n_fulltext": sum(1 for r in match if r["source"] == "fulltext"), "N_fulltext": len(ft),
        "n_recent": n_recent, "N_recent": len(recent), "n_early": n_early, "N_early": len(early),
        "share_recent": round(n_recent / len(recent), 3) if recent else 0.0,
        "share_early": round(n_early / len(early), 3) if early else 0.0,
        "corpora": sorted({r.get("corpus") or "?" for r in match}),
        "regions": len(geos), "stated_rate": round(stated / len(rows), 2) if rows else 0.0,
        "paper_ids": [r["paper_id"] for r in match][:40],   # for display
        "_ids": {r["paper_id"] for r in match},              # the complete set, for co-occurrence
    }


def _both_corpora(corpus_sizes: dict) -> bool:
    """Only count cross-corpus agreement when each corpus contributes enough papers to mean something."""
    return len(corpus_sizes) > 1 and all(n >= THRESHOLDS["corpus_min_papers"] for n in corpus_sizes.values())


def _strength(s: dict, corpus_sizes: dict, emerging: bool = False) -> dict:
    pts, why = 0, []
    if s["n"] >= 15:
        pts += 2; why.append(f"{s['n']} papers")
    elif s["n"] >= 8:
        pts += 1; why.append(f"{s['n']} papers")
    else:
        why.append(f"only {s['n']} papers")
    if s["n_fulltext"] >= 5:
        pts += 1; why.append(f"confirmed in {s['n_fulltext']} full-text reads")
    elif s["n_fulltext"] == 0:
        pts -= 1; why.append("never confirmed in a full-text read")
    if _both_corpora(corpus_sizes) and set(s["corpora"]) >= set(corpus_sizes):
        pts += 1; why.append("found in both arXiv and PMC")
    if s["regions"] >= 3:
        pts += 1; why.append(f"{s['regions']} distinct place names")
    if s["n_recent"] >= 2:
        pts += 1; why.append(f"{s['n_recent']} papers in the last {THRESHOLDS['recent_years']} years")
    if emerging:
        pts = min(pts, 4) if s["p_rise"] >= 0.05 else pts   # a rise that may be chance is never "high"
        if s["p_rise"] < 0.05:
            pts += 1; why.append(f"the rise is unlikely to be chance (one-sided Fisher p={s['p_rise']:.3f})")
        elif s["p_rise"] > 0.2:
            pts -= 1; why.append(f"the rise could be chance at this sample size (p={s['p_rise']:.2f})")
        else:
            why.append(f"the rise is suggestive (p={s['p_rise']:.2f})")
    grade = "high" if pts >= 5 else ("moderate" if pts >= 3 else "low")
    return {"grade": grade, "points": pts, "reasons": why}


def _gap_confidence(s: dict, corpus: dict | None, corpus_sizes: dict) -> dict:
    pts, why = 0, []
    n_ft, N_ft = s["n_fulltext"], s["N_fulltext"]
    if N_ft >= 15:
        if n_ft == 0:
            pts += 2; why.append(f"absent from all {N_ft} papers read in full")
        elif n_ft / N_ft <= 0.05:
            pts += 1; why.append(f"in only {n_ft} of {N_ft} papers read in full")
        else:
            why.append(f"present in {n_ft} of {N_ft} papers read in full")
    elif N_ft >= 8:
        if n_ft == 0:
            pts += 1; why.append(f"absent from all {N_ft} papers read in full (a small set)")
        else:
            why.append(f"present in {n_ft} of {N_ft} papers read in full")
    else:
        why.append(f"only {N_ft} papers read in full, too few to confirm absence")
    if s["stated_rate"] >= 0.6:
        pts += 1; why.append(f"the field is reported by {s['stated_rate']:.0%} of papers, so absence is informative")
    elif s["stated_rate"] < 0.3:
        pts -= 1; why.append(f"the field is reported by only {s['stated_rate']:.0%} of papers; "
                             "absence may be non-reporting")
    if _both_corpora(corpus_sizes) and s["n"] == 0:
        pts += 1; why.append("absent in both arXiv and PMC papers")
    elif len(corpus_sizes) > 1 and s["n"] == 0:
        why.append("absent in both corpora, but one contributes fewer than "
                   f"{THRESHOLDS['corpus_min_papers']} papers, so that agreement carries little weight")
    if corpus:
        share = corpus["share"]
        if share <= 0.02:
            pts += 1; why.append(f"rare across the corpus too: {corpus['matching']} of {corpus['topic_total']} "
                                 "topic papers mention it in their title or abstract")
        elif share >= 0.10:
            pts -= 1; why.append(f"common in the wider corpus ({corpus['matching']} of {corpus['topic_total']} "
                                 "topic papers mention it in their title or abstract): the gap may be a "
                                 "shortlist artifact")
        else:
            why.append(f"{corpus['matching']} of {corpus['topic_total']} topic papers across the corpus mention "
                       "it in their title or abstract")
        unread = corpus.get("not_analysed") or 0
        if unread:
            # papers that mention it and were never read are the most direct threat to calling it a gap
            if unread >= 5:
                pts -= 1
            why.append(f"{unread} papers outside the analysed set mention it in their title or abstract and were "
                       "not read, so the gap holds for the analysed papers only until they are")
        ft = corpus.get("fulltext") or {}
        if ft.get("full_texts_read", 0) >= 30:
            if ft["matching"] == 0:
                pts += 1
                why.append(f"absent from all {ft['full_texts_read']} topic papers whose full text was searched")
            else:
                pts -= 1
                why.append(f"{ft['matching']} of {ft['full_texts_read']} topic papers mention it in their full "
                           "text: read those passages before calling it absent")
        elif ft.get("full_texts_read"):
            why.append(f"only {ft['full_texts_read']} topic full texts could be searched, too few to weigh")
    why.append(f"true rate in this literature plausibly up to {s['ci95'][1]:.0%} (95% upper bound)")
    grade = "high" if pts >= 4 else ("moderate" if pts >= 2 else "low")
    return {"grade": grade, "points": pts, "reasons": why}


def _corpus_check(ctx, topic_query: str, search: str) -> dict | None:
    """Among corpus papers on the question's topic, how many mention the practice at all."""
    if not topic_query or not search:
        return None
    from research_agent.tools.query import tsquery_sql

    t_sql, t_params = tsquery_sql(topic_query, prefix="t")
    s_sql, s_params = tsquery_sql(search, prefix="s")
    row = ctx.pg.execute(
        f"""SELECT count(*) FILTER (WHERE tsv @@ ({t_sql})) AS topic_total,
                   count(*) FILTER (WHERE tsv @@ ({t_sql}) AND tsv @@ ({s_sql})) AS matching
            FROM papers""", {**t_params, **s_params}).fetchone()
    if not row["topic_total"]:
        return None
    out = {"topic_query": topic_query, "search": search, "topic_total": row["topic_total"],
           "matching": row["matching"], "share": round(row["matching"] / row["topic_total"], 4)}
    # the papers that mention it and were NOT analysed: each one could overturn the gap, so name them
    unread = ctx.pg.execute(
        f"""SELECT paper_id, title, year FROM papers p WHERE tsv @@ ({t_sql}) AND tsv @@ ({s_sql})
              AND NOT EXISTS (SELECT 1 FROM run_papers rp WHERE rp.run_id = %(run)s AND rp.paper_id = p.paper_id)
            ORDER BY year DESC NULLS LAST, paper_id LIMIT 12""",
        {**t_params, **s_params, "run": ctx.run_id}).fetchall()
    n_unread = ctx.pg.execute(
        f"""SELECT count(*) n FROM papers p WHERE tsv @@ ({t_sql}) AND tsv @@ ({s_sql})
              AND NOT EXISTS (SELECT 1 FROM run_papers rp WHERE rp.run_id = %(run)s AND rp.paper_id = p.paper_id)""",
        {**t_params, **s_params, "run": ctx.run_id}).fetchone()["n"]
    out["not_analysed"] = n_unread
    out["not_analysed_examples"] = [{"paper_id": r["paper_id"], "title": (r["title"] or "")[:160], "year": r["year"]}
                                    for r in unread]
    try:
        from research_agent.tools.fulltext_check import search_fulltexts

        ft = search_fulltexts(ctx, topic_query, search)
        if ft is not None:
            out["fulltext"] = ft
    except Exception as exc:          # the full-text tier is an addition; the abstract count stands without it
        out["fulltext_problem"] = str(exc)[:200]
    return out


def compute_map(ctx, protocol: dict) -> dict:
    rows = _rows(ctx)
    _, enums = known_fields(ctx)
    if not rows:
        return {"N": 0, "established": [], "emerging": [], "gaps": [], "novelty": [], "thresholds": THRESHOLDS}
    years = sorted(r["year"] for r in rows if r["year"])
    cut = (years[-1] if years else 2025) - THRESHOLDS["recent_years"] + 1
    corpora = {r.get("corpus") or "?" for r in rows}
    corpus_sizes = {c: sum(1 for r in rows if (r.get("corpus") or "?") == c) for c in corpora}
    T = THRESHOLDS
    established, emerging, watch, gaps, pool = [], [], [], [], []
    established_ids: list[set] = []
    for feat in _features(rows, protocol, enums):
        s = _stats(rows, feat, cut)
        ids = s.pop("_ids")
        item = {"field": feat.field, "value": feat.value, "label": feat.label, "role": feat.role, **s}
        if feat.members:
            item["members"] = feat.members
        if feat.desirable and (s["n"] <= T["missing_max_n"] or s["share"] <= T["missing_max_share"]):
            corpus = _corpus_check(ctx, protocol.get("topic_query", ""), feat.search)
            gaps.append({**item, "corpus_check": corpus,
                         "confidence": _gap_confidence(s, corpus, corpus_sizes)})
            continue
        if (s["n"] >= T["novelty_min_component_n"] and feat.field != "code_or_data_available"
                and not _ABSENCE.search(feat.value)):
            pool.append((feat, s, ids))
        if s["share"] >= T["established_min_share"] and s["n"] >= T["established_min_n"]:
            established.append({**item, "strength": _strength(s, corpus_sizes)})
            established_ids.append(ids)
        elif s["n"] >= T["emerging_min_n"] and s["share"] < T["established_min_share"]:
            rising = ((s["share_early"] > 0 and s["share_recent"] / s["share_early"] >= T["emerging_min_ratio"])
                      or (s["share_early"] == 0 and s["n_recent"] >= T["emerging_min_n"]))
            if rising:
                ratio = round(s["share_recent"] / s["share_early"], 2) if s["share_early"] else None
                entry = {**item, "trend": {"since": cut, "ratio": ratio, "p": s["p_rise"]}}
                if s["p_rise"] > T["emerging_max_p"]:
                    watch.append(entry)     # rising, but plausibly chance: listed, not claimed
                else:
                    emerging.append({**entry, "strength": _strength(s, corpus_sizes, emerging=True)})

    # Novelty: two components from different dimensions, each used by several papers, that (almost) never
    # appear together. p = chance of so few co-occurrences if papers picked them independently.
    novelty = []
    N = len(rows)
    by_id = {r["paper_id"]: r for r in rows}
    # One fact recorded in two fields is one established item, not two. Collapsing them here keeps the
    # section honest about how much the run found, and is what stops novelty pairing a value against the
    # complement of its own field-mate.
    established, folded = _collapse_mirrors(established, established_ids, T["mirror_min_jaccard"])

    all_strata = strata(rows, protocol, T["stratum_min_papers"])
    for i, (fa, sa, ia) in enumerate(pool):
        for fb, sb, ib in pool[i + 1:]:
            if fa.field == fb.field or fa.role == fb.role:
                continue
            both = ia & ib
            expected = sa["n"] * sb["n"] / N
            if len(both) > T["novelty_max_observed"] or expected < T["novelty_min_expected"]:
                continue
            # the pair must still be unexpectedly absent once we know what KIND of study each component
            # belongs to; if any single field accounts for it, their absence together is not news
            explained = explained_by_a_stratum(ia, ib, all_strata, T["novelty_min_expected"])
            if explained is not None:
                continue
            p = hyper_le(len(both), N, sa["n"], sb["n"])
            if p > T["novelty_max_p"]:
                continue
            within = max(((f, stratified_expected(ia, ib, g)) for f, g in all_strata),
                         key=lambda x: x[1], default=None)
            novelty.append({"a": {"label": fa.label, "field": fa.field, "value": fa.value, "role": fa.role,
                                  "n": sa["n"], "paper_ids": sa["paper_ids"][:5]},
                            "b": {"label": fb.label, "field": fb.field, "value": fb.value, "role": fb.role,
                                  "n": sb["n"], "paper_ids": sb["paper_ids"][:5]},
                            "observed": len(both), "expected": round(expected, 1), "p": round(p, 4),
                            "score": round(expected - len(both), 2), "together_in": sorted(both),
                            **({"expected_within": {"field": within[0], "expected": round(within[1], 1)}}
                               if within else {})})
    novelty.sort(key=lambda x: (x["p"], -x["score"]))

    def number(items, prefix):
        for k, it in enumerate(items, 1):
            it["id"] = f"{prefix}{k}"
        return items

    established.sort(key=lambda x: -x["n"])
    emerging.sort(key=lambda x: (x["trend"]["p"], -(x["share_recent"] - x["share_early"])))
    watch.sort(key=lambda x: x["trend"]["p"])
    # A watch list exists to say "this might be rising". An item whose p is 1.00 says the opposite, and a
    # list where nothing at all separates from chance is the field inventory with a p-value stapled on: it
    # fills a page and tells a reader nothing. Drop those, and drop the list if that empties it.
    watch = [w for w in watch if w["trend"]["p"] < T["watch_max_p"]]
    gaps.sort(key=lambda x: -x["confidence"]["points"])
    lists, _ = known_fields(ctx)
    coverage = {f: round(sum(1 for r in rows if _values(r["data"], f)) / len(rows), 2)
                for f in lists + list(enums)}
    return {
        "N": len(rows), "N_fulltext": sum(1 for r in rows if r["source"] == "fulltext"),
        "corpora": {c: corpus_sizes[c] for c in sorted(corpora)},
        "years": [years[0], years[-1]] if years else None, "recent_since": cut,
        "established": number(established, "E"), "emerging": number(emerging, "R"),
        "watch": number(watch, "W"),
        "gaps": number(gaps, "G"), "novelty": number(novelty[:8], "N"),
        "field_coverage": coverage, "thresholds": T,
        "titles": {pid: r["title"] for pid, r in by_id.items()},
    }


def compact_map(m: dict) -> dict:
    """What the reasoning agents see: every item with its numbers and grades, without long id lists."""
    def slim(it):
        keep = {k: it[k] for k in ("id", "label", "field", "value", "role", "members", "n", "N", "share",
                                   "ci95", "n_fulltext", "N_fulltext", "share_early", "share_recent", "p_rise",
                                   "stated_rate", "corpora", "regions") if k in it}
        for k in ("strength", "confidence", "trend", "corpus_check"):
            if it.get(k):
                keep[k] = it[k]
        cc = keep.get("corpus_check")
        if cc and cc.get("fulltext"):
            # the agent needs the passages to judge them, but not twelve long ones per gap
            ft = cc["fulltext"]
            keep["corpus_check"] = {**cc, "fulltext": {
                **{k: ft[k] for k in ("topic_papers_checked", "full_texts_read", "matching", "not_analysed", "note")
                   if k in ft},
                "candidates": [{**x, "passage": x["passage"][:300]} for x in ft.get("candidates", [])[:5]]}}
        keep["example_paper_ids"] = it.get("paper_ids", [])[:6]
        return keep
    return {"N": m["N"], "N_fulltext": m.get("N_fulltext"), "corpora": m.get("corpora"),
            "established": [slim(x) for x in m["established"]], "emerging": [slim(x) for x in m["emerging"]],
            "watch": [slim(x) for x in m.get("watch", [])],
            "gaps": [slim(x) for x in m["gaps"]],
            "novelty": [{k: v for k, v in x.items()} for x in m["novelty"]],
            "field_coverage": m.get("field_coverage")}
