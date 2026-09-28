"""Trend tools. Corpus-wide (the whole health subset, not just the shortlist) and normalised:
arXiv grows every year, so raw counts make every topic look like it is rising. We report each
topic's share of all arXiv papers that year (per 10,000) and its share of the health corpus."""
from __future__ import annotations

from research_agent.tools.base import INT, STR, STRS, Tool, obj
from research_agent.tools.query import tsquery_sql
from research_agent.tools.search import SOURCE


def corpus_sources(ctx) -> dict:
    """Which corpora are loaded, and how many papers and years each contributes."""
    rows = ctx.pg.execute(
        "SELECT source, count(*) n, min(year) lo, max(year) hi FROM papers GROUP BY source ORDER BY source"
    ).fetchall()
    return {"sources": {r["source"]: {"papers": r["n"], "years": [r["lo"], r["hi"]]} for r in rows}}


def _partial_year(ctx) -> int | None:
    r = ctx.pg.execute("SELECT max(first_version_date) AS d FROM papers").fetchone()
    d = r["d"]
    if d is None:
        return None
    return d.year if (d.month, d.day) < (12, 15) else None


def topic_series(ctx, keywords: str, year_from: int | None = None, year_to: int | None = None,
                 source: str | None = None, within: str | None = None) -> dict:
    """Yearly prevalence of a topic, normalised against the same corpus it is counted in.

    With both arXiv and PMC loaded, pass source='arxiv' or 'pmc' to keep a trend inside one corpus:
    the two have different year coverage, so a mixed series can move for reasons other than the topic.

    With `within` (a topic query such as 'malaria'), the trend is the share of THAT topic's papers that
    match `keywords` (pct_of_topic). Use it for claims about "malaria papers": dividing by all health
    papers makes every malaria practice look like it is falling whenever the corpus grows elsewhere."""
    src = (source or "all").lower()
    tq_sql, tq_params = tsquery_sql(keywords)
    w_sql, w_params = tsquery_sql(within, prefix="w") if within else ("NULL::tsquery", {})
    scope = ("AND tsv @@ (" + w_sql + ")") if within else ""
    params = {**tq_params, **w_params, "yf": year_from or 0, "yt": year_to or 9999,
              "src": None if src == "all" else src}
    rows = ctx.pg.execute(
        """WITH m AS (SELECT year, count(*) n FROM papers
                      WHERE tsv @@ (""" + tq_sql + """) """ + scope + """
                        AND year BETWEEN %(yf)s AND %(yt)s
                        AND (%(src)s::text IS NULL OR source = %(src)s) GROUP BY year),
                h AS (SELECT year, count(*) n FROM papers WHERE year BETWEEN %(yf)s AND %(yt)s """ + scope + """
                        AND (%(src)s::text IS NULL OR source = %(src)s) GROUP BY year),
                a AS (SELECT year, sum(n_papers) n FROM corpus_year_stats
                      WHERE year BETWEEN %(yf)s AND %(yt)s
                        AND (%(src)s::text IS NULL OR source = %(src)s) GROUP BY year)
           SELECT h.year, coalesce(m.n, 0) AS matching, h.n AS health_total, a.n AS arxiv_total
           FROM h LEFT JOIN m USING (year) LEFT JOIN a USING (year) ORDER BY h.year""", params).fetchall()
    partial = _partial_year(ctx)
    series = []
    for r in rows:
        series.append({
            "year": r["year"], "matching": r["matching"], "health_total": r["health_total"],
            "per_10k_arxiv": round(10000 * r["matching"] / r["arxiv_total"], 2) if r["arxiv_total"] else None,
            "pct_of_health": round(100 * r["matching"] / r["health_total"], 2) if r["health_total"] else None,
            "partial_year": r["year"] == partial,
        })
        if within:   # health_total is the topic's own paper count here
            series[-1]["topic_total"] = series[-1].pop("health_total")
            series[-1]["pct_of_topic"] = series[-1].pop("pct_of_health")
            series[-1].pop("per_10k_arxiv")
    out = {"keywords": keywords, "source": src, "series": series}
    if within:
        out["within"] = within
    return out


def window_ratio(series: list[dict], early: tuple[int, int], late: tuple[int, int],
                 metric: str = "per_10k_arxiv") -> dict:
    def mean(a, b):
        vals = [s[metric] for s in series if a <= s["year"] <= b and s[metric] is not None
                and not s["partial_year"]]
        return (sum(vals) / len(vals)) if vals else None

    e, l_ = mean(*early), mean(*late)
    ratio = (l_ / e) if (e and l_ is not None) else (float("inf") if (l_ and not e) else None)
    return {"metric": metric, "early_window": list(early), "late_window": list(late),
            "early_mean": None if e is None else round(e, 3), "late_mean": None if l_ is None else round(l_, 3),
            "ratio_late_over_early": None if ratio is None else (round(ratio, 2) if ratio != float("inf") else "inf")}


def topic_trend(ctx, keywords: str, year_from: int | None = None, year_to: int | None = None,
                source: str | None = None, within: str | None = None) -> dict:
    out = topic_series(ctx, keywords, year_from, year_to, source, within)
    full = [s["year"] for s in out["series"] if not s["partial_year"]]
    if len(full) >= 6:
        out["summary"] = window_ratio(out["series"], (full[0], full[2]), (full[-3], full[-1]),
                                      metric="pct_of_topic" if within else "per_10k_arxiv")
    out["note"] = ("per_10k_arxiv is the topic's share of ALL papers in this corpus that year (per 10,000), "
                   "which normalises for its growth; the latest year is partial and excluded from summaries. "
                   "With both corpora loaded, set source to keep a trend inside arxiv or pmc. With within, "
                   "pct_of_topic is the share of the topic's own papers, which is what a claim about "
                   "'<topic> papers' means.")
    return out


def compare_topics(ctx, topics: list[str], year_from: int | None = None, year_to: int | None = None,
                   source: str | None = None, within: str | None = None) -> dict:
    result = []
    metric = "pct_of_topic" if within else "per_10k_arxiv"
    for kw in topics[:8]:
        t = topic_trend(ctx, kw, year_from, year_to, source, within)
        total = sum(s["matching"] for s in t["series"])
        result.append({"keywords": kw, "total_papers": total, "summary": t.get("summary"),
                       f"{metric}_by_year": {s["year"]: s[metric] for s in t["series"]}})
    return {"topics": result}


def shortlist_years(ctx) -> dict:
    rows = ctx.pg.execute(
        """SELECT p.year, count(*) n FROM run_papers rp JOIN papers p USING (paper_id)
           WHERE rp.run_id=%s GROUP BY p.year ORDER BY p.year""", (ctx.run_id,)).fetchall()
    return {"by_year": {r["year"]: r["n"] for r in rows}}


WITHIN = {"type": "string", "description": "Topic query (e.g. 'malaria'). Measures the share of THAT topic's "
          "papers matching keywords, instead of the share of all health papers. Use it whenever the trend "
          "is about practice within a topic ('ML in malaria papers')."}

TREND_TOOLS = [
    Tool("topic_trend",
         "Yearly trend of a topic across the WHOLE health corpus (keyword query, web-search syntax, "
         "e.g. '\"vision transformer\" OR ViT'). Returns counts, share per 10k arXiv papers and % of the "
         "health corpus, plus an early-vs-late window ratio.",
         obj({"keywords": STR, "year_from": INT, "year_to": INT, "source": SOURCE, "within": WITHIN},
             ["keywords"]), topic_trend, read_only=True),
    Tool("compare_topics", "Run topic_trend for up to 8 keyword queries side by side.",
         obj({"topics": STRS, "year_from": INT, "year_to": INT, "source": SOURCE, "within": WITHIN}, ["topics"]),
         compare_topics, read_only=True),
    Tool("corpus_sources", "Which corpora are loaded (arxiv preprints, pmc open-access articles), with "
         "paper counts and year coverage for each.", obj({}), corpus_sources, read_only=True),
    Tool("shortlist_years", "Year distribution of this run's shortlist.", obj({}), shortlist_years),
]
