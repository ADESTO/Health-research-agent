"""Do the papers back each other up?

Each paper's reported associations are statements with a direction: a driver (a drug, an exposure, a risk
factor) moves an outcome up, down, or not at all, with a quote that code has already checked against the
paper. Until now those statements were only compared to find conflict (`results.contradictions`), and only
for drivers on a hand-written malaria list; "testosterone therapy", "TRT" and "exogenous testosterone" were
three drivers that could never meet. This puts every statement next to every other statement about the
same thing and says, with quotes from both sides, which papers support it, contradict it or qualify it.

Three steps, and only the last could ever need a model, which it does not use:

  group    phrases that name the same driver (or outcome) are put together: identical after removing
           measurement noise ("serum testosterone levels" is "testosterone"), the same entry on the malaria
           list, an acronym of the other ("TRT"), one a slightly longer form of the other, or very close in
           meaning by embedding. Every merge is listed, so a reader can see what was treated as one thing.
  relate   two statements about the same driver and outcome are graded from their extracted directions:
           same direction supports, opposite directions contradict, an effect against no effect disputes
           whether there is one at all, a nonlinear or mixed finding qualifies.
  weigh    support is counted in INDEPENDENT sources. Papers sharing a named dataset, two or more authors,
           or a citation between them are one source: two papers from the same registry agreeing is one
           finding, not a replication. Support from the same kind of study (human, animal, in vitro) is
           counted apart from support across kinds, because a rat result backing a human trial is not the
           same as two trials agreeing.

What this is not. Agreement is not truth: papers can share an assay bias or a publication bias, so a
statement's corroboration sits beside a claim's state and never raises it. And a finding no other paper
addresses is "not addressed elsewhere", which is a fact about this corpus, never evidence against it.
"""
from __future__ import annotations

import re

import numpy as np

from research_agent.config import settings

# words that say how something was measured, not what it is
_NOISE = {"level", "levels", "concentration", "concentrations", "serum", "plasma", "circulating", "the", "of",
          "a", "an", "in", "on", "to", "and", "with", "exposure", "use", "amount", "value", "values", "status"}
_LINKING = {"of", "and", "the", "in", "on", "for", "to", "with", "a", "an"}
# For an OUTCOME, the measure is noise too: a paper reporting that rainfall raises malaria cases supports one
# reporting that it raises malaria incidence. Mortality and severity are not measures of how much disease
# there is, so they stay distinct.
OUTCOME_MEASURES = {"incidence", "case", "cases", "count", "counts", "rate", "rates", "number", "numbers",
                    "occurrence", "burden", "prevalence", "positivity", "risk", "frequency", "episodes"}
# A qualified exposure is a different exposure. "Heavy rainfall" lowering malaria does not contradict
# "rainfall" raising it: excess rain flushing breeding sites and ordinary rain creating them can both be
# true. Phrases are only ever merged with phrases carrying the same qualifiers.
_QUALIFIERS = {"heavy", "excess", "excessive", "extreme", "intense", "prolonged", "drought", "flood", "flooding",
               "low", "high", "minimum", "maximum", "min", "max", "reduced", "elevated", "deficient", "deficiency",
               "lagged", "cumulative", "anomaly", "anomalies"}
_WORD = re.compile(r"[a-z0-9]+")
_ANIMAL = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|porcine|pigs?|rabbits?|zebrafish|primates?|monkeys?|"
                     r"dogs?|sheep|animal)\b", re.I)
_IN_VITRO = re.compile(r"in vitro|cell line|cell culture|organoid|ex vivo|cells\b", re.I)
SUPPORT = ("supports", "agrees_in_direction")
AGAINST = ("contradicts", "disputes_existence")


# ---------------------------------------------------------------- grouping phrases that mean the same thing
def _tokens(phrase: str, extra_noise: frozenset = frozenset()) -> list[str]:
    out = []
    for w in _WORD.findall(str(phrase or "").lower()):
        if w in _NOISE or w in extra_noise:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def _acronym_of(short: str, long: str) -> bool:
    """'TRT' and 'testosterone replacement therapy'. Only a written-in-capitals token counts as an acronym, so
    'age' is never read as the initials of 'androgen gel exposure'."""
    s = str(short or "").strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9]{1,5}", s):
        return False
    words = [w for w in _WORD.findall(str(long or "").lower()) if w not in _LINKING]
    return len(words) == len(s) and "".join(w[0] for w in words) == s.lower()


def _close(a: list[str], b: list[str]) -> bool:
    """One is the other with at most one word added: 'testosterone therapy' and 'testosterone replacement
    therapy'. The shorter needs two words of its own, or 'testosterone' alone would swallow both an
    endogenous level and a treatment, which are different drivers."""
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 2 and set(short) <= set(long) and len(set(long) - set(short)) <= 1


class _Union:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _qualifiers(phrase: str) -> frozenset:
    return frozenset(w for w in _WORD.findall(str(phrase or "").lower()) if w in _QUALIFIERS)


def group_phrases(phrases: list[str], canonical: dict[str, str] | None = None,
                  threshold: float | None = None, outcomes: bool = False) -> dict[str, dict]:
    """{phrase as written: {"group": label, "why": reason it joined, "members": [...]}} for every phrase.

    `canonical` maps a phrase to a key it already shares with others (the malaria driver list), which is
    always a reason to merge. The label of a group is its canonical key if it has one, else its most
    frequent wording."""
    from collections import Counter

    canonical = canonical or {}
    threshold = settings.corroboration_similarity if threshold is None else threshold
    counts = Counter(p for p in phrases if str(p or "").strip())
    uniq = list(counts)
    if not uniq:
        return {}
    extra = frozenset(OUTCOME_MEASURES) if outcomes else frozenset()
    toks = {p: (_tokens(p, extra) or _tokens(p)) for p in uniq}
    quals = {p: _qualifiers(p) for p in uniq}
    u = _Union(uniq)
    why: dict[tuple, str] = {}

    def join(a, b, reason):
        if quals[a] != quals[b]:
            return                              # "heavy rainfall" is not "rainfall", by any route
        if u.find(a) != u.find(b):
            u.join(a, b)
            why[(a, b)] = reason

    by_norm: dict[str, str] = {}
    for p in uniq:
        key = " ".join(toks[p])
        if key and key in by_norm:
            join(by_norm[key], p, "the same after removing how it was measured")
        elif key:
            by_norm[key] = p
    by_canon: dict[tuple, str] = {}
    for p in uniq:
        c = canonical.get(p)
        key = (c, quals[p])                     # the list says "rain"; "heavy rain" is still its own driver
        if c and key in by_canon:
            join(by_canon[key], p, f"both are '{c}' on the driver list")
        elif c:
            by_canon[key] = p
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            if _acronym_of(a, b) or _acronym_of(b, a):
                join(a, b, "one is the other's acronym")
            elif _close(toks[a], toks[b]):
                join(a, b, "one is the other with a word added")
    # meaning, last and strictest: only phrases with real words, and only very close ones
    real = [p for p in uniq if toks[p]]
    if len(real) > 1 and threshold < 1:
        from research_agent.embeddings import get_embedder

        vecs = np.asarray(get_embedder().embed_documents([" ".join(toks[p]) for p in real]), dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1, norms)
        sims = vecs @ vecs.T
        for i, a in enumerate(real):
            for j in range(i + 1, len(real)):
                if sims[i, j] >= threshold:
                    join(a, real[j], f"close in meaning ({sims[i, j]:.2f})")
    groups: dict[str, list[str]] = {}
    for p in uniq:
        groups.setdefault(u.find(p), []).append(p)
    out = {}
    for root, members in groups.items():
        canon = next((canonical[m] for m in members if canonical.get(m) and not quals[m]), None)
        # the most used wording; on a tie the spelled-out one, so a report reads "testosterone replacement
        # therapy" rather than "TRT"
        label = canon or max(members, key=lambda m: (counts[m], len(m)))
        reasons = sorted({r for (a, b), r in why.items() if a in members and b in members})
        for m in members:
            out[m] = {"group": label, "members": sorted(members), "why_merged": reasons}
    return out


# ---------------------------------------------------------------- which papers are one source
def _authors(text: str) -> set[str]:
    return {" ".join(a.lower().split()) for a in re.split(r",|;| and ", str(text or "")) if len(a.strip()) > 3}


def independence(ctx, paper_ids: list[str]) -> dict[str, dict]:
    """{paper_id: {"source": cluster id, "linked_by": [...]}}. Papers sharing a named dataset, two or more
    authors, or a citation between them are one source of evidence."""
    from research_agent.tools.extraction import _rows

    ids = list(dict.fromkeys(paper_ids))
    u = _Union(ids)
    links: dict[str, set[str]] = {p: set() for p in ids}
    rows = {r["paper_id"]: r for r in _rows(ctx) if r["paper_id"] in set(ids)}
    data_of = {p: {d.lower().strip() for d in (rows.get(p, {}).get("data", {}).get("datasets") or []) if d}
               for p in ids}
    try:
        authors = {r["paper_id"]: _authors(r["authors"]) for r in ctx.pg.execute(
            "SELECT paper_id, authors FROM papers WHERE paper_id = ANY(%s)", (ids,)).fetchall()}
    except Exception:
        authors = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = data_of[a] & data_of[b]
            if shared:
                u.join(a, b)
                links[a].add(f"shares the dataset {sorted(shared)[0]} with {b}")
                links[b].add(f"shares the dataset {sorted(shared)[0]} with {a}")
            if len(authors.get(a, set()) & authors.get(b, set())) >= 2:
                u.join(a, b)
                links[a].add(f"shares authors with {b}")
                links[b].add(f"shares authors with {a}")
    try:
        from research_agent.tools.citations import _works

        works = _works(ctx.pg, ids)
        oa = {w["openalex_id"]: pid for pid, w in works.items()}
        for pid, w in works.items():
            for ref in w.get("referenced_works") or []:
                other = oa.get(ref)
                if other and other != pid:
                    u.join(pid, other)
                    links[pid].add(f"cites {other}")
                    links[other].add(f"is cited by {pid}")
    except Exception:            # no citation data: independence rests on datasets and authors alone
        pass
    return {p: {"source": u.find(p), "linked_by": sorted(links[p])} for p in ids}


def study_system(row: dict) -> tuple[str, str]:
    """(human | animal | in_vitro_or_ex_vivo, recorded | inferred) for one analysed paper."""
    d = row.get("data") or {}
    recorded = d.get("q_study_system")
    if isinstance(recorded, str) and recorded not in ("", "not_stated"):
        return recorded, "recorded"
    text = " ".join(str(x) for f in ("organisms", "populations", "study_designs", "data_modalities")
                    for x in (d.get(f) or []))
    if _IN_VITRO.search(text) and not re.search(r"patients|participants|volunteers|women|men\b", text, re.I):
        return "in_vitro_or_ex_vivo", "inferred"
    if _ANIMAL.search(text):
        return "animal", "inferred"
    return "human", "inferred"


# ---------------------------------------------------------------- relating statements
def relation(a: str, b: str, sig_a: str = "not_stated", sig_b: str = "not_stated") -> str:
    """How statement B stands to statement A, from their directions alone."""
    if a in ("nonlinear", "mixed") or b in ("nonlinear", "mixed"):
        return "qualifies"
    if a == b:
        if a in ("positive", "negative") and {sig_a, sig_b} == {"yes", "no"}:
            return "agrees_in_direction"         # same way, but one of them did not find it significant
        return "supports"
    if {a, b} == {"positive", "negative"}:
        return "contradicts"
    if "none" in (a, b):
        return "disputes_existence"              # one finds an effect, the other finds none
    return "qualifies"


def _statements(ctx) -> list[dict]:
    from research_agent.tools.extraction import _rows
    from research_agent.tools.results import driver_key

    out = []
    for r in _rows(ctx):
        system, how = study_system(r)
        seen = set()
        for it in r["data"].get("reported_associations") or []:
            written = it.get("driver_as_written") or it.get("driver") or ""
            out.append({"paper_id": r["paper_id"], "year": r.get("year"), "read": r.get("source"),
                        "driver": written, "driver_key": driver_key(written),
                        "outcome": str(it.get("outcome") or "").strip(),
                        "direction": it.get("direction") or "mixed",
                        "significant": it.get("significant") or "not_stated",
                        "lag": it.get("lag") or "", "quote": it.get("quote") or "",
                        "system": system, "system_is": how, "_row": r})
            key = (written.lower(), out[-1]["outcome"].lower(), out[-1]["direction"])
            if key in seen:
                out.pop()
            seen.add(key)
    return out


def _attributes(row: dict) -> set[str]:
    from research_agent.tools.results import _attributes as attrs

    return attrs(row)


def corroborate(ctx, driver: str | None = None, save: bool = True) -> dict:
    """Every reported association set beside every other about the same driver and outcome."""
    stmts = _statements(ctx)
    if not stmts:
        return {"findings": [], "statements": 0,
                "note": "No paper in this run reports an association with a direction, so there is nothing "
                        "to set side by side. Papers that only report model performance are compared with "
                        "method_comparison instead."}
    drivers = group_phrases([s["driver"] for s in stmts],
                            canonical={s["driver"]: s["driver_key"] for s in stmts
                                       if s["driver_key"] != s["driver"].lower().strip()})
    outcomes = group_phrases([s["outcome"] for s in stmts if s["outcome"]], outcomes=True)
    for s in stmts:
        s["driver_group"] = drivers.get(s["driver"], {}).get("group", s["driver"])
        s["outcome_group"] = outcomes.get(s["outcome"], {}).get("group") if s["outcome"] else None
    indep = independence(ctx, [s["paper_id"] for s in stmts])
    for s in stmts:
        s["source"] = indep[s["paper_id"]]["source"]

    findings: dict[tuple, list[dict]] = {}
    for s in stmts:
        findings.setdefault((s["driver_group"], s["outcome_group"]), []).append(s)
    want = None
    if driver:
        want = drivers.get(driver, {}).get("group") or next(
            (g["group"] for p, g in drivers.items() if driver.lower() in p.lower()), driver)

    out = []
    for (dg, og), group in findings.items():
        if want and dg != want:
            continue
        by_dir: dict[str, list[dict]] = {}
        for s in group:
            by_dir.setdefault(s["direction"], []).append(s)
        sources = {d: {s["source"] for s in v} for d, v in by_dir.items()}
        papers = {s["paper_id"] for s in group}
        verdict = _verdict(group)
        statements = []
        for s in group:
            others = [o for o in group if o["paper_id"] != s["paper_id"]]
            rel = [{"paper_id": o["paper_id"], "relation": relation(s["direction"], o["direction"],
                                                                      s["significant"], o["significant"]),
                    "direction": o["direction"], "quote": o["quote"][:300], "same_system": o["system"] == s["system"],
                    "independent": o["source"] != s["source"], "system": o["system"]} for o in others]
            backing = [x for x in rel if x["relation"] in SUPPORT]
            statements.append({
                "paper_id": s["paper_id"], "driver": s["driver"], "outcome": s["outcome"],
                "direction": s["direction"], "significant": s["significant"], "quote": s["quote"][:300],
                "system": s["system"], "system_is": s["system_is"], "read": s["read"],
                "independent_support": len({o["source"] for o in others
                                            if o["source"] != s["source"]
                                            and relation(s["direction"], o["direction"], s["significant"],
                                                         o["significant"]) in SUPPORT}),
                "independent_support_same_system": len({o["source"] for o in others
                                                        if o["source"] != s["source"] and o["system"] == s["system"]
                                                        and relation(s["direction"], o["direction"], s["significant"],
                                                                     o["significant"]) in SUPPORT}),
                "related": rel, "backed_by": [x["paper_id"] for x in backing],
                "against": [x["paper_id"] for x in rel if x["relation"] in AGAINST],
                "linked_by": indep[s["paper_id"]]["linked_by"]})
        item = {"driver": dg, "outcome": og or "outcome not stated", "verdict": verdict,
                "means": MEANS[verdict], "papers": len(papers),
                "independent_sources": len({s["source"] for s in group}),
                "by_direction": {d: {"papers": sorted({s["paper_id"] for s in v}),
                                     "independent_sources": len(sources[d])} for d, v in by_dir.items()},
                "systems": sorted({s["system"] for s in group}),
                "driver_wordings": sorted({s["driver"] for s in group}),
                "outcome_wordings": sorted({s["outcome"] for s in group if s["outcome"]}),
                "merged_because": drivers.get(group[0]["driver"], {}).get("why_merged", []),
                "statements": statements}
        if verdict in ("contested", "contradicted"):
            item["what_separates_the_sides"] = _separate(by_dir)
        if og is None:
            item["caveat"] = ("None of these statements names its outcome, so they are grouped by driver alone and "
                              "may be about different outcomes.")
        out.append(item)
    order = list(MEANS)
    out.sort(key=lambda f: (order.index(f["verdict"]), -f["independent_sources"], -f["papers"]))
    res = {"statements": len(stmts), "findings": out,
           "summary": {v: sum(1 for f in out if f["verdict"] == v) for v in order if any(f["verdict"] == v for f in out)},
           "driver_groups": {g["group"]: g["members"] for g in drivers.values() if len(g["members"]) > 1},
           "outcome_groups": {g["group"]: g["members"] for g in outcomes.values() if len(g["members"]) > 1},
           "note": "Support is counted in independent sources: papers sharing a dataset, two authors or a "
                   "citation are one source. Agreement is not truth, and a finding no other paper addresses "
                   "is not addressed elsewhere, which is not evidence against it."}
    if save and not driver:
        ctx.save_note("corroboration", {k: v for k, v in res.items()})
    return res


MEANS = {
    "contested": "independent papers report it both ways",
    "corroborated": "independent papers report the same direction, and none the opposite",
    "corroborated_across_systems_only": "the only independent agreement comes from a different kind of study "
                                         "(animal or in vitro behind a human finding, or the reverse)",
    "contradicted": "independent papers report opposite directions with no paper in agreement",
    "repeated_by_related_papers_only": "several papers agree, but they share a dataset, authors or a "
                                        "citation, so this is one source, not a replication",
    "qualified": "other papers report it as nonlinear or mixed",
    "not_addressed_elsewhere": "only one paper reports this, which is a fact about this corpus, not "
                               "evidence against it",
}


def _verdict(group: list[dict]) -> str:
    """One finding's verdict, from who says what and how independent they are."""
    papers = {x["paper_id"] for x in group}
    if len(papers) < 2:
        return "not_addressed_elsewhere"
    sides = {}
    for x in group:
        if x["direction"] in ("positive", "negative", "none"):
            sides.setdefault(x["direction"], set()).add(x["source"])
    if not sides:
        return "qualified"                      # every statement is nonlinear or mixed
    if len(sides) >= 2:
        # a disagreement. Contested when at least one side is backed by independent sources; when every side
        # stands on a single source, it is two findings contradicting each other with no replication either way
        return "contested" if max(len(v) for v in sides.values()) >= 2 else "contradicted"
    direction, sources = next(iter(sides.items()))
    if len(sources) < 2:
        return "repeated_by_related_papers_only"
    systems: dict[str, set] = {}
    for x in group:
        if x["direction"] == direction:
            systems.setdefault(x["system"], set()).add(x["source"])
    if max(len(v) for v in systems.values()) < 2:
        return "corroborated_across_systems_only"
    return "corroborated"


def _separate(by_dir: dict[str, list[dict]]) -> list[dict]:
    """What the two sides of a disagreement differ in: candidate explanations, not proven ones."""
    sides = [(d, [s["_row"] for s in v]) for d, v in by_dir.items() if d in ("positive", "negative", "none")]
    if len(sides) < 2:
        return []
    (da, ra), (db, rb) = sorted(sides, key=lambda x: -len(x[1]))[:2]
    aa = [_attributes(r) | {f"system: {_sys(r)}"} for r in ra]
    ab = [_attributes(r) | {f"system: {_sys(r)}"} for r in rb]
    out = []
    for a in set().union(*aa, *ab):
        sa, sb = sum(a in s for s in aa) / len(aa), sum(a in s for s in ab) / len(ab)
        if abs(sa - sb) >= 0.5:
            out.append({"attribute": a, f"share_{da}": round(sa, 2), f"share_{db}": round(sb, 2),
                        "_gap": abs(sa - sb)})
    out.sort(key=lambda x: -x["_gap"])
    for x in out:
        x.pop("_gap")
    return out[:6]


def _sys(row: dict) -> str:
    return study_system(row)[0]


def markdown(ctx) -> list[str]:
    """The report section: which findings independent papers back, and which they dispute."""
    note = ctx.notes().get("corroboration") or {}
    findings = [f for f in note.get("findings") or [] if f["verdict"] != "not_addressed_elsewhere"]
    alone = sum(1 for f in note.get("findings") or [] if f["verdict"] == "not_addressed_elsewhere")
    if not findings and not alone:
        return []
    L = ["## Do the papers back each other up? (computed)", "",
         "Each association a paper reports (a driver moving an outcome, with a quote) set beside every other "
         "paper's statement about the same driver and outcome. Support is counted in independent sources: "
         "papers that share a dataset, two or more authors, or a citation count once. Agreement is not truth, "
         "so none of this changes what a claim's own evidence supports.", ""]
    if findings:
        L += ["| Driver | Outcome | Verdict | Same direction | Opposite or no effect | Papers |",
              "|---|---|---|---|---|---|"]
        for f in findings[:15]:
            dirs = f["by_direction"]
            real = {d: v for d, v in dirs.items() if d in ("positive", "negative", "none")} or dirs
            main = max(real.items(), key=lambda kv: kv[1]["independent_sources"])
            against = sum(v["independent_sources"] for d, v in dirs.items() if d != main[0]
                          and d in ("positive", "negative", "none"))
            ids = ", ".join(f"[{p}]" for p in sorted({s["paper_id"] for s in f["statements"]})[:4])
            L.append(f"| {f['driver']} | {f['outcome']} | {f['verdict'].replace('_', ' ')} | "
                     f"{main[1]['independent_sources']} ({main[0]}) | {against} | {ids} |")
        L.append("")
        for f in findings:
            if f.get("what_separates_the_sides"):
                sep = "; ".join(x["attribute"] for x in f["what_separates_the_sides"][:3])
                L.append(f"- **{f['driver']} → {f['outcome']}** is contested. What differs between the sides: "
                         f"{sep}. Candidate explanations, not tested ones.")
        if any(f.get("what_separates_the_sides") for f in findings):
            L.append("")
    if alone:
        L.append(f"{alone} further associations are reported by a single paper each. That means no other "
                 "paper in this run addresses them, not that they are unsupported.")
    groups = note.get("driver_groups") or {}
    if groups:
        L += ["", "Treated as the same driver: " + "; ".join(
            f"{k} ({', '.join(v)})" for k, v in list(groups.items())[:8]) + "."]
    return L + [""]


def corroborate_tool(ctx, driver: str | None = None, limit: int = 12, per_finding: int = 8) -> dict:
    """What a follow-up or the researcher sees: findings with their verdicts and the quotes on each side."""
    res = corroborate(ctx, driver=driver, save=False)
    found = res.get("findings") or []
    slim = []
    for f in found[: max(1, min(int(limit), 30))]:
        slim.append({k: f[k] for k in ("driver", "outcome", "verdict", "means", "papers", "independent_sources",
                                       "by_direction", "systems", "driver_wordings", "outcome_wordings") if k in f}
                    | {"what_separates_the_sides": f.get("what_separates_the_sides", []),
                       "statements": [{k: st[k] for k in ("paper_id", "direction", "significant", "quote", "system",
                                                          "independent_support", "backed_by", "against")}
                                      for st in f["statements"]][:max(1, int(per_finding))]})
    return {"findings": slim, "statements": res.get("statements", 0), "summary": res.get("summary", {}),
            "driver_groups": res.get("driver_groups", {}), "note": res.get("note", "")}


from research_agent.tools.base import INT, STR, Tool, obj  # noqa: E402

CORROBORATION_TOOL = Tool(
    "corroborate",
    "Which papers back each other up: every association a paper reports (a driver moving an outcome, with a "
    "quote) set beside every other paper's statement about the same driver and outcome, graded supports / "
    "contradicts / disputes whether there is an effect / qualifies, with support counted in independent "
    "sources (papers sharing a dataset, authors or a citation count once). Give a driver to look at one.",
    obj({"driver": STR, "limit": INT}), corroborate_tool, read_only=True, max_chars=16000)
