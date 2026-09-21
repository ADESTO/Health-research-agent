"""Trend tools. Corpus-wide (the whole health subset, not just the shortlist) and normalised:
arXiv grows every year, so raw counts make every topic look like it is rising. We report each
topic's share of all arXiv papers that year (per 10,000) and its share of the health corpus."""
from __future__ import annotations

from research_agent.tools.base import INT, STR, STRS, Tool, obj


def _partial_year(ctx) -> int | None:
    r = ctx.pg.execute("SELECT max(first_version_date) AS d FROM papers").fetchone()
    d = r["d"]
    if d is None:
        return None
    return d.year if (d.month, d.day) < (12, 15) else None


def topic_series(ctx, keywords: str, year_from: int | None = None, year_to: int | None = None) -> dict:
    params = {"tq": keywords, "yf": year_from or 0, "yt": year_to or 9999}
    rows = ctx.pg.execute(
        """WITH m AS (SELECT year, count(*) n FROM papers
                      WHERE tsv @@ websearch_to_tsquery('english', %(tq)s)
                        AND year BETWEEN %(yf)s AND %(yt)s GROUP BY year),
                h AS (SELECT year, count(*) n FROM papers WHERE year BETWEEN %(yf)s AND %(yt)s GROUP BY year),
                a AS (SELECT year, sum(n_papers) n FROM corpus_year_stats
                      WHERE year BETWEEN %(yf)s AND %(yt)s GROUP BY year)
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
    return {"keywords": keywords, "series": series}


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


def topic_trend(ctx, keywords: str, year_from: int | None = None, year_to: int | None = None) -> dict:
    out = topic_series(ctx, keywords, year_from, year_to)
    full = [s["year"] for s in out["series"] if not s["partial_year"]]
    if len(full) >= 6:
        out["summary"] = window_ratio(out["series"], (full[0], full[2]), (full[-3], full[-1]))
    out["note"] = ("per_10k_arxiv normalises for arXiv's overall growth; the latest year is partial "
                   "(snapshot cut-off) and excluded from summaries.")
    return out


def compare_topics(ctx, topics: list[str], year_from: int | None = None, year_to: int | None = None) -> dict:
    result = []
    for kw in topics[:8]:
        t = topic_trend(ctx, kw, year_from, year_to)
        total = sum(s["matching"] for s in t["series"])
        result.append({"keywords": kw, "total_papers": total, "summary": t.get("summary"),
                       "per_10k_by_year": {s["year"]: s["per_10k_arxiv"] for s in t["series"]}})
    return {"topics": result}


def shortlist_years(ctx) -> dict:
    rows = ctx.pg.execute(
        """SELECT p.year, count(*) n FROM run_papers rp JOIN papers p USING (paper_id)
           WHERE rp.run_id=%s GROUP BY p.year ORDER BY p.year""", (ctx.run_id,)).fetchall()
    return {"by_year": {r["year"]: r["n"] for r in rows}}


TREND_TOOLS = [
    Tool("topic_trend",
         "Yearly trend of a topic across the WHOLE health corpus (keyword query, web-search syntax, "
         "e.g. '\"vision transformer\" OR ViT'). Returns counts, share per 10k arXiv papers and % of the "
         "health corpus, plus an early-vs-late window ratio.",
         obj({"keywords": STR, "year_from": INT, "year_to": INT}, ["keywords"]), topic_trend),
    Tool("compare_topics", "Run topic_trend for up to 8 keyword queries side by side.",
         obj({"topics": STRS, "year_from": INT, "year_to": INT}, ["topics"]), compare_topics),
    Tool("shortlist_years", "Year distribution of this run's shortlist.", obj({}), shortlist_years),
]
