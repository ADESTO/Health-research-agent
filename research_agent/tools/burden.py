"""Research versus burden: where the analysed research comes from, against where the disease is.

1. Extracted geography ("Kenya", "western Kenya", "Amhara region", "Côte d'Ivoire") is turned into countries.
   Regions such as "sub-Saharan Africa" are counted separately, never spread over countries.
2. Burden estimates come from the WHO Global Health Observatory (estimated malaria cases and deaths by
   country; `burden-fetch`) or from a CSV you import (`burden-import`, e.g. Malaria Atlas Project figures).
3. For each country: its share of the analysed papers against its share of cases. A country with 20% of
   cases but 2% of papers is under-researched relative to burden (ratio 0.1).
"""
from __future__ import annotations

import csv
import re

import httpx

GHO = "https://ghoapi.azureedge.net/api"
GHO_INDICATORS = {"cases": "MALARIA_EST_CASES", "deaths": "MALARIA_EST_DEATHS"}
_transport: httpx.BaseTransport | None = None     # tests plug a mock transport in here

ALIASES = {
    "ivory coast": "CIV", "cote d'ivoire": "CIV", "côte d'ivoire": "CIV", "drc": "COD", "dr congo": "COD",
    "democratic republic of congo": "COD", "democratic republic of the congo": "COD", "congo-kinshasa": "COD",
    "republic of congo": "COG", "congo-brazzaville": "COG", "the gambia": "GMB", "gambia": "GMB",
    "tanzania": "TZA", "zanzibar": "TZA", "swaziland": "SWZ", "eswatini": "SWZ", "cape verde": "CPV",
    "south sudan": "SSD", "laos": "LAO", "lao pdr": "LAO", "vietnam": "VNM", "burma": "MMR", "myanmar": "MMR",
    "bolivia": "BOL", "venezuela": "VEN", "iran": "IRN", "usa": "USA", "united states": "USA", "uk": "GBR",
    "south korea": "KOR", "north korea": "PRK", "russia": "RUS", "syria": "SYR", "car": "CAF",
    "central african republic": "CAF", "guinea-bissau": "GNB", "sao tome": "STP", "são tomé": "STP",
    "papua new guinea": "PNG", "east timor": "TLS", "timor-leste": "TLS",
}
# Subnational places malaria papers often name without their country
PLACES = {
    "amhara": "ETH", "oromia": "ETH", "tigray": "ETH", "addis ababa": "ETH", "bahir dar": "ETH",
    "limpopo": "ZAF", "kwazulu-natal": "ZAF", "mpumalanga": "ZAF", "dar es salaam": "TZA", "kilombero": "TZA",
    "nairobi": "KEN", "kisumu": "KEN", "western kenya": "KEN", "kenyan highlands": "KEN", "kampala": "UGA",
    "west nile": "UGA", "acholi": "UGA", "lagos": "NGA", "kano": "NGA", "adamawa": "NGA", "ibadan": "NGA",
    "accra": "GHA", "ashanti": "GHA", "tamale": "GHA", "kumasi": "GHA", "ouagadougou": "BFA", "nouna": "BFA",
    "bamako": "MLI", "dangassa": "MLI", "blantyre": "MWI", "lilongwe": "MWI", "maputo": "MOZ",
    "kinshasa": "COD", "yaoundé": "CMR", "yaounde": "CMR", "dakar": "SEN", "cotonou": "BEN", "lusaka": "ZMB",
    "lake kariba": "ZMB", "harare": "ZWE", "kigali": "RWA", "bujumbura": "BDI", "antananarivo": "MDG",
    "odisha": "IND", "orissa": "IND", "chhattisgarh": "IND", "amazon": "BRA", "amazonas": "BRA", "loreto": "PER",
    "sokoto": "NGA", "khartoum": "SDN", "bannu": "PAK", "khyber pakhtunkhwa": "PAK",
}
REGIONS = {"sub-saharan africa": "Sub-Saharan Africa", "east africa": "East Africa", "west africa": "West Africa",
           "southern africa": "Southern Africa", "central africa": "Central Africa", "africa": "Africa",
           "south asia": "South Asia", "southeast asia": "Southeast Asia", "latin america": "Latin America",
           "amazon basin": "Amazon basin", "global": "Global", "worldwide": "Global"}


def _countries():
    import pycountry

    names = {}
    for c in pycountry.countries:
        for n in {c.name, getattr(c, "common_name", None), getattr(c, "official_name", None)}:
            if n:
                names[n.lower()] = c.alpha_3
    names.update(ALIASES)
    return names


_NAMES: dict | None = None


def to_country(place: str) -> tuple[str | None, str | None]:
    """(ISO3, None) for a country or known subnational place; (None, region) for a region; (None, None)."""
    global _NAMES
    if _NAMES is None:
        _NAMES = _countries()
    low = " ".join((place or "").lower().replace("’", "'").split()).strip(" .,;")
    if not low:
        return None, None
    if low in _NAMES:
        return _NAMES[low], None
    if low in PLACES:
        return PLACES[low], None
    for name in sorted(list(_NAMES) + list(PLACES), key=len, reverse=True):   # "western Kenya", "Amhara region"
        if len(name) >= 4 and re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", low):
            return (_NAMES.get(name) or PLACES.get(name)), None
    for key, region in REGIONS.items():
        if re.search(r"(?<![a-z])" + re.escape(key) + r"(?![a-z])", low):
            return None, region
    return None, None


def country_name(iso3: str) -> str:
    import pycountry

    c = pycountry.countries.get(alpha_3=iso3)
    return getattr(c, "common_name", None) or (c.name if c else iso3)


# ---------------------------------------------------------------- burden data
def fetch_who(pg, client: httpx.Client | None = None) -> dict:
    """Download WHO GHO estimates of malaria cases and deaths by country and year."""
    client = client or httpx.Client(timeout=60, transport=_transport)
    rows: dict[tuple[str, int], dict] = {}
    for measure, code in GHO_INDICATORS.items():
        r = client.get(f"{GHO}/{code}")
        r.raise_for_status()
        for x in r.json().get("value", []):
            iso, year, val = x.get("SpatialDim"), x.get("TimeDim"), x.get("NumericValue")
            if not iso or len(iso) != 3 or year is None or val is None or x.get("SpatialDimType") not in (None, "COUNTRY"):
                continue
            rows.setdefault((iso, int(year)), {})[measure] = float(val)
    for (iso, year), v in rows.items():
        pg.execute("INSERT INTO burden (iso3, year, cases, deaths, source) VALUES (%s,%s,%s,%s,'WHO GHO') "
                   "ON CONFLICT (iso3, year, source) DO UPDATE SET cases=EXCLUDED.cases, deaths=EXCLUDED.deaths",
                   (iso, year, v.get("cases"), v.get("deaths")))
    return {"country_years": len(rows), "source": "WHO GHO"}


def import_csv(pg, path: str, source: str = "CSV") -> dict:
    """Columns (case-insensitive, any order): iso3 or country; year; cases; deaths (optional)."""
    n, skipped = 0, []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            r = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            iso = (r.get("iso3") or r.get("iso") or "").upper() or to_country(r.get("country", ""))[0]
            try:
                year = int(float(r.get("year", "")))
                cases = float(r["cases"].replace(",", "")) if r.get("cases") else None
                deaths = float(r["deaths"].replace(",", "")) if r.get("deaths") else None
            except ValueError:
                skipped.append(r.get("country") or r.get("iso3") or "?")
                continue
            if not iso or len(iso) != 3:
                skipped.append(r.get("country") or "?")
                continue
            pg.execute("INSERT INTO burden (iso3, year, cases, deaths, source) VALUES (%s,%s,%s,%s,%s) "
                       "ON CONFLICT (iso3, year, source) DO UPDATE SET cases=EXCLUDED.cases, deaths=EXCLUDED.deaths",
                       (iso, year, cases, deaths, source))
            n += 1
    return {"rows": n, "skipped": skipped[:20], "source": source}


def _latest_burden(pg) -> tuple[dict[str, float], int | None, str | None]:
    row = pg.execute("SELECT source, max(year) AS y FROM burden WHERE cases IS NOT NULL GROUP BY source "
                     "ORDER BY (source = 'WHO GHO') DESC, max(year) DESC LIMIT 1").fetchone()
    if not row:
        return {}, None, None
    cases = {r["iso3"]: r["cases"] for r in pg.execute(
        "SELECT iso3, cases FROM burden WHERE source=%s AND year=%s AND cases > 0", (row["source"], row["y"])).fetchall()}
    return cases, row["y"], row["source"]


# ---------------------------------------------------------------- research versus burden
_MALARIA = re.compile(r"malaria|plasmod", re.I)


def is_malaria_run(ctx, rows=None) -> bool:
    """The burden data are malaria cases, so they only mean something for a run about malaria: the question
    names it, or at least half of the analysed papers have it as a health domain."""
    from research_agent.tools.extraction import _values

    if _MALARIA.search(ctx.question or ""):
        return True
    rows = rows or []
    hits = sum(1 for r in rows if any(_MALARIA.search(str(v)) for v in _values(r["data"], "health_domains")))
    return bool(rows) and hits * 2 >= len(rows)


def research_vs_burden(ctx) -> dict:
    """Countries: papers analysed in this run that use data from them, against their share of malaria cases
    (only for runs about malaria; other runs get where their studies come from)."""
    from research_agent.tools import claims as C
    from research_agent.tools.extraction import _values

    papers: dict[str, set] = {}
    regions: dict[str, set] = {}
    unplaced = 0
    all_rows = C._rows(ctx)
    for r in all_rows:
        isos, regs = set(), set()
        for g in _values(r["data"], "geography"):
            iso, region = to_country(g)
            if iso:
                isos.add(iso)
            elif region:
                regs.add(region)
        for i in isos:
            papers.setdefault(i, set()).add(r["paper_id"])
        for g in regs:
            regions.setdefault(g, set()).add(r["paper_id"])
        if not isos and not regs:
            unplaced += 1
    applies = is_malaria_run(ctx, all_rows)
    try:
        cases, year, source = _latest_burden(ctx.pg) if applies else ({}, None, None)
    except Exception:
        cases, year, source = {}, None, None
    total_cases = sum(cases.values()) or 0
    country_papers = sum(len(v) for v in papers.values()) or 1
    rows = []
    for iso in set(papers) | set(cases):
        n = len(papers.get(iso, ()))
        c = cases.get(iso)
        rs = n / country_papers
        bs = (c / total_cases) if (c and total_cases) else None
        rows.append({"iso3": iso, "country": country_name(iso), "papers": n, "research_share": round(rs, 4),
                     "cases": c, "burden_share": round(bs, 4) if bs is not None else None,
                     "ratio": round(rs / bs, 3) if bs else None, "paper_ids": sorted(papers.get(iso, ()))[:10]})
    rows.sort(key=lambda x: (-(x["burden_share"] or 0), -x["papers"]))
    under = [x for x in rows if x["burden_share"] and x["burden_share"] >= 0.01 and (x["ratio"] or 0) < 0.5]
    over = [x for x in rows if x["burden_share"] is not None and x["papers"] >= 2 and (x["ratio"] or 0) > 2]
    return {"countries": rows, "regions": {k: len(v) for k, v in sorted(regions.items(), key=lambda kv: -len(kv[1]))},
            "papers_without_place": unplaced, "burden_year": year, "burden_source": source,
            "burden_applies": applies,
            "under_researched": under[:10], "over_researched": sorted(over, key=lambda x: -x["ratio"])[:10],
            "note": "Shares are of papers with a country-level place, and of cases in the burden data. A paper "
                    "using data from several countries counts once for each."}


def burden_markdown(ctx) -> list[str]:
    rb = research_vs_burden(ctx)
    placed = [x for x in rb["countries"] if x["papers"]]
    if not placed:
        return []
    if not rb.get("burden_applies", True):
        L = ["## Where the studies come from (computed)", "",
             "Countries the analysed papers take their data or samples from.", "",
             "| Country | Papers |", "|---|---|"]
        L += [f"| {x['country']} | {x['papers']} |" for x in sorted(placed, key=lambda y: -y["papers"])[:15]] + [""]
        if rb["regions"]:
            L += ["Papers describing only a region: " + ", ".join(f"{k} ({v})" for k, v in rb["regions"].items())
                  + ".", ""]
        return L
    L = ["## Research versus burden (computed)", ""]
    if rb["burden_year"]:
        L += [f"Where the analysed papers' data come from, against each country's share of estimated malaria "
              f"cases ({rb['burden_source']}, {rb['burden_year']}). A ratio below 1 means less research than the "
              "burden would suggest.", "", "| Country | Papers | Share of papers | Share of cases | Ratio |",
              "|---|---|---|---|---|"]
        for x in [y for y in rb["countries"] if y["burden_share"]][:15]:
            L.append(f"| {x['country']} | {x['papers']} | {x['research_share']:.0%} | {x['burden_share']:.1%} | "
                     f"{x['ratio'] if x['ratio'] is not None else 'n/a'} |")
        L.append("")
        if rb["under_researched"]:
            L += ["Under-researched relative to burden: " + ", ".join(
                f"{x['country']} ({x['burden_share']:.0%} of cases, {x['papers']} papers)" for x in rb["under_researched"][:6])
                + ".", ""]
    else:
        L += ["Where the analysed papers' data come from. Add burden estimates (`python -m research_agent.cli "
              "burden-fetch`) to compare against each country's share of malaria cases.", "",
              "| Country | Papers |", "|---|---|"]
        L += [f"| {x['country']} | {x['papers']} |" for x in sorted(placed, key=lambda y: -y["papers"])[:15]] + [""]
    if rb["regions"]:
        L += ["Papers describing only a region: " + ", ".join(f"{k} ({v})" for k, v in rb["regions"].items()) + ".", ""]
    return L


def burden_chart(ctx, path: str) -> str | None:
    """Scatter of share of cases against share of papers (PNG). Needs burden data and matplotlib."""
    rb = research_vs_burden(ctx)
    pts = [x for x in rb["countries"] if x["burden_share"]]
    if not pts:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    xs = [100 * x["burden_share"] for x in pts]
    ys = [100 * x["research_share"] for x in pts]
    ax.scatter(xs, ys, s=[30 + 20 * x["papers"] for x in pts], alpha=0.6, color="#7a4f2c")
    lim = max(max(xs), max(ys)) * 1.1 or 1
    ax.plot([0, lim], [0, lim], color="#999", linewidth=1, linestyle="--")
    for x, y, p in zip(xs, ys, pts):
        if p["burden_share"] >= 0.02 or p["papers"] >= 3:
            ax.annotate(p["country"], (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"Share of estimated malaria cases, % ({rb['burden_source']} {rb['burden_year']})")
    ax.set_ylabel("Share of analysed papers, %")
    ax.set_title("Research versus burden (below the line: under-researched)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _research_vs_burden_tool(ctx) -> dict:
    out = research_vs_burden(ctx)
    out["countries"] = out["countries"][:30]
    return out


from research_agent.tools.base import Tool, obj  # noqa: E402

BURDEN_TOOL = Tool("research_vs_burden", "Countries the analysed papers take data from, against each country's "
                   "share of estimated malaria cases: which high-burden countries are under-researched.",
                   obj({}), _research_vs_burden_tool, read_only=True, max_chars=12000)
