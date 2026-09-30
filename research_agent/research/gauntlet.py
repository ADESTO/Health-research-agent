"""The gauntlet: what a pattern must survive before the researcher may call it a finding.

    1. significant in the discovery half            (Fisher exact / exact binomial / sign test)
    2. subgroup tests, run automatically             corpus, read depth, period, the leading places:
                                                     does the direction hold inside each?
    3. the trap library                              known ways counts mislead: field misfit, abstract-only
                                                     absence, PMC topic slices, tiny cells, overlapping groups,
                                                     unequal places, text that contradicts a rarity count
    4. a critic                                      a separate agent proposes rival explanations as testable
                                                     specs; code tests each one
    5. the held-out half                             the same spec on papers the researcher never saw, at a
                                                     threshold that tightens with every finding it proposes
    6. a computed grade                              from the above; the model's confidence plays no part

Findings are the researcher's own hypotheses with their evidence trail. They never enter a report as fact.
"""
from __future__ import annotations

import re
from collections import Counter

from research_agent.research.patterns import MIN_CELL, _conds, evaluate, significant

GRADES = ("strong", "moderate", "provisional", "rejected")


# ---------------------------------------------------------------- 2. subgroups
def _strata(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    out = [("arXiv preprints", [r for r in rows if r.get("corpus") == "arxiv"]),
           ("PMC articles", [r for r in rows if r.get("corpus") == "pmc"]),
           ("read in full", [r for r in rows if r.get("source") == "fulltext"]),
           ("read from abstract", [r for r in rows if r.get("source") == "abstract"])]
    years = sorted(r["year"] for r in rows if r.get("year"))
    if years:
        mid = years[len(years) // 2]
        out += [(f"published up to {mid}", [r for r in rows if r.get("year") and r["year"] <= mid]),
                (f"published after {mid}", [r for r in rows if r.get("year") and r["year"] > mid])]
    places = Counter(g.lower() for r in rows for g in set((r["data"].get("geography") or [])) if g)
    for place, _ in places.most_common(2):
        out.append((f"studies in {place}", [r for r in rows if place in {g.lower() for g in r["data"].get("geography") or []}]))
    return out


def subgroup_tests(spec: dict, rows: list[dict], enums: dict, direction: int, ctx=None) -> dict:
    base_diff = evaluate(spec, rows, enums, ctx).get("difference")
    tests = []
    for name, sub in _strata(rows):
        if len(sub) < 2 * MIN_CELL:
            continue
        r = evaluate(spec, sub, enums, ctx)
        if not r["testable"]:
            tests.append({"subgroup": name, "n": r["n"], "verdict": "too few papers"})
            continue
        if r["direction"] != direction:
            verdict = "does not hold"
        elif base_diff and r.get("difference") is not None and abs(r["difference"]) < abs(base_diff) / 2:
            verdict = "much weaker"            # same sign, but most of the effect is gone inside this subgroup
        else:
            verdict = "holds"
        tests.append({"subgroup": name, "n": r["n"], "verdict": verdict,
                      "p": r["p"], **({"difference": r["difference"]} if "difference" in r else {})})
    tested = [t for t in tests if t["verdict"] != "too few papers"]
    score = {"holds": 1.0, "much weaker": 0.5, "does not hold": 0.0}
    agreement = (sum(score[t["verdict"]] for t in tested) / len(tested)) if tested else None
    return {"tests": tests, "agreement": round(agreement, 2) if agreement is not None else None,
            "tested": len(tested)}


# ---------------------------------------------------------------- 3. traps
def traps(spec: dict, rows: list[dict], result: dict, enums: dict, ctx=None) -> list[dict]:
    """Known ways a count misleads. status: fail (the pattern cannot stand), warn (it can, with a caveat)."""
    from research_agent.tools.claims import field_misfit

    out = []
    all_conds = [c for k in ("outcome", "group_a", "group_b", "within") for c in _conds(spec.get(k))]

    def add(name, status, detail):
        out.append({"trap": name, "status": status, "detail": detail})

    for c in all_conds:                                                   # measured where it is not recorded
        misfit = field_misfit(c.get("field"), c.get("any_of") or [])
        if misfit:
            add("field does not record this", "fail", misfit)
    if any(c.get("field") in ("corpus", "source") for c in _conds(spec.get("group_a")) + _conds(spec.get("group_b"))):
        add("corpus comparison", "fail", "The PMC part of the corpus is topic slices chosen when loading; comparing "
            "arXiv with PMC measures what was loaded, not how the field differs.")
    if spec["kind"] in ("difference", "trend") and result.get("cells"):
        if min(result["cells"]) < MIN_CELL:
            add("small cells", "warn", f"A cell of the 2x2 table has fewer than {MIN_CELL} papers "
                f"({result['cells']}); one paper more or less changes the picture.")
        if result.get("overlap_removed"):
            add("overlapping groups", "warn", f"{result['overlap_removed']} papers met both group definitions and were "
                "left out of both.")
    shares = [s for s in (result.get("share"), (result.get("group_a") or {}).get("share"),
                          (result.get("group_b") or {}).get("share")) if s is not None]
    rare = bool(shares) and min(shares) <= 0.1
    abstract_share = sum(r.get("source") == "abstract" for r in rows) / max(1, len(rows))
    if rare and abstract_share > 0.5:
        full = [r for r in rows if r.get("source") == "fulltext"]
        fr = evaluate(spec, full, enums, ctx) if len(full) >= 2 * MIN_CELL else None
        if fr and fr["testable"] and fr["direction"] != result["direction"]:
            add("absence in abstracts", "fail", "The pattern rests on something being rare, and among papers read "
                "in full it does not hold: abstracts omit what full texts report.")
        else:
            add("absence in abstracts", "warn", f"{abstract_share:.0%} of these papers were read from abstracts only; "
                "rarity may mean 'not in the abstract'.")
    if rare and ctx is not None:                                          # text that contradicts a rarity count
        try:
            from research_agent.tools.recheck import mentioning

            outcome = _conds(spec.get("outcome"))
            terms = [t for c in outcome for t in (c.get("any_of") or []) if c.get("field") not in enums]
            if terms:
                ids = [r["paper_id"] for r in rows]
                mention = mentioning(ctx.pg, ids, terms)
                recorded = {r["paper_id"] for r in rows if evaluate({"kind": "prevalence", "outcome": outcome,
                                                                      "max_share": 1}, [r], enums)["k"]}
                missed = mention - recorded
                if len(missed) >= 3:
                    add("text contradicts the count", "fail", f"{len(missed)} papers mention {', '.join(terms[:3])} "
                        "in their text but their record does not; the count understates it. Re-check with "
                        "recheck_field before relying on it.")
        except Exception:
            pass
    for c in all_conds:                                                   # short terms match too much
        short = [t for t in c.get("any_of") or [] if len(str(t).strip()) <= 3]
        if short and c.get("field") not in enums and c.get("field") not in ("corpus", "read", "source"):
            add("ambiguous term", "warn", f"Short terms {short} in {c['field']} may match unrelated values.")
    if spec["kind"] == "difference":                                      # the groups come from different places
        ga = [r for r in rows if evaluate_group(spec, "group_a", r, enums)]
        gb = [r for r in rows if (evaluate_group(spec, "group_b", r, enums) if spec.get("group_b")
                                  else not evaluate_group(spec, "group_a", r, enums))]
        top = lambda g: Counter(x.lower() for r in g for x in set(r["data"].get("geography") or []))
        ta, tb = top(ga), top(gb)
        for place, n in ta.most_common(1):
            share_a = n / max(1, len(ga))
            share_b = tb.get(place, 0) / max(1, len(gb))
            if share_a - share_b >= 0.5:
                add("groups differ in place", "warn", f"{share_a:.0%} of group A is from {place} against "
                    f"{share_b:.0%} of group B; place may explain the difference.")
    return out


def evaluate_group(spec: dict, side: str, row: dict, enums: dict) -> bool:
    from research_agent.research.patterns import _match

    return _match(row, spec.get(side), enums)


# ---------------------------------------------------------------- 4. rival explanations
def test_alternative(spec: dict, alt: dict, rows: list[dict], enums: dict, base: dict, ctx=None) -> dict:
    """A rival explanation, tested. stratify: does the pattern hold inside each level of a possible confounder?
    redefine: does it hold when a term list is defined differently?"""
    import copy

    kind = alt.get("kind")
    out = {"explanation": str(alt.get("explanation") or "")[:300], "kind": kind}
    if kind == "stratify" and alt.get("field") and (alt.get("any_of") or alt.get("none_of")):
        cond = {k: alt[k] for k in ("field", "any_of", "none_of") if alt.get(k)}
        from research_agent.research.patterns import _match

        levels = [("with", [r for r in rows if _match(r, [cond], enums)]),
                  ("without", [r for r in rows if not _match(r, [cond], enums)])]
        if spec["kind"] in ("difference", "trend"):
            # adjust for the confounder: is there still an association once its levels are held apart?
            from research_agent.research.patterns import mantel_haenszel

            tables = [evaluate(spec, sub, enums, ctx).get("cells") for _, sub in levels]
            tables = [t for t in tables if t and sum(t) >= 2]
            mh = mantel_haenszel(tables)
            out["adjusted"] = mh
            out["levels"] = [{"level": f"{name} {cond['field']} {cond.get('any_of') or cond.get('none_of')}",
                              "n": len(sub)} for name, sub in levels]
            if len(tables) < 2 or mh["adjusted_difference"] is None:
                out["verdict"] = "untestable"
            elif (mh["p"] < 0.05 and (mh["adjusted_difference"] > 0) == (base["direction"] > 0)
                  and abs(mh["adjusted_difference"]) >= abs(base.get("difference") or 0) / 2):
                out["verdict"] = "does not explain it"
            elif mh["p"] >= 0.1 or abs(mh["adjusted_difference"]) < abs(base.get("difference") or 0) / 2:
                out["verdict"] = "explains it"
            else:
                out["verdict"] = "inconclusive"    # weakened but not gone: the rival is neither ruled out nor proven
            return out
        verdicts = []
        for name, sub in levels:
            r = evaluate(spec, sub, enums, ctx)
            if not r["testable"]:
                verdicts.append((name, "untestable", r))
                continue
            weak = ("difference" in r and base.get("difference") and r["difference"] is not None
                    and abs(r["difference"]) < abs(base["difference"]) / 3)
            verdicts.append((name, "holds" if r["direction"] == base["direction"] and not weak else "fails", r))
        tested = [v for v in verdicts if v[1] != "untestable"]
        out["levels"] = [{"level": f"{name} {cond['field']} {cond.get('any_of') or cond.get('none_of')}", "verdict": v,
                          "n": r["n"], "p": r.get("p")} for name, v, r in verdicts]
        out["verdict"] = ("untestable" if not tested else "explains it" if any(v[1] == "fails" for v in tested)
                          else "does not explain it")
        return out
    if kind == "redefine" and alt.get("side") in ("outcome", "group_a", "group_b", "within") and alt.get("any_of"):
        new = copy.deepcopy(spec)
        conds = _conds(new.get(alt["side"]))
        if not conds:
            out["verdict"] = "untestable"
            return out
        conds[0]["any_of"] = list(alt["any_of"])
        new[alt["side"]] = conds
        r = evaluate(new, rows, enums, ctx)
        out["redefined"] = {alt["side"]: alt["any_of"]}
        out["result"] = {k: r.get(k) for k in ("n", "p", "direction", "difference", "share")}
        out["verdict"] = ("untestable" if not r["testable"] else
                          "does not explain it" if r["direction"] == base["direction"] and r["p"] < 0.1 else "explains it")
        return out
    out["verdict"] = "untestable"
    out["why"] = "the rival was not given as a testable spec"
    return out


# ---------------------------------------------------------------- 5-6. held-out half and grade
def holdout_test(spec: dict, rows: list[dict], enums: dict, direction: int, alpha: float, ctx=None) -> dict:
    r = evaluate(spec, rows, enums, ctx)
    if not r["testable"]:
        return {"result": r, "verdict": "too few papers", "alpha": alpha}
    if spec["kind"] == "prevalence":
        ok = r["direction"] == 1
        verdict = "replicated" if ok else "not replicated"
    elif r["direction"] != direction:
        verdict = "reversed"
    elif r["p"] < alpha:
        verdict = "replicated"
    elif r["p"] < 0.2:
        verdict = "same direction, not significant"
    else:
        verdict = "not replicated"
    return {"result": r, "verdict": verdict, "alpha": round(alpha, 4)}


def grade(discovery: dict, subgroups: dict, trap_list: list[dict], alternatives: list[dict], holdout: dict) -> dict:
    reasons = []
    if not significant(discovery):
        return {"grade": "rejected", "status": "rejected", "reasons": ["not significant in the discovery half"]}
    fails = [t for t in trap_list if t["status"] == "fail"]
    warns = [t for t in trap_list if t["status"] == "warn"]
    explained = [a for a in alternatives if a.get("verdict") == "explains it"]
    survived = [a for a in alternatives if a.get("verdict") == "does not explain it"]
    if fails:
        reasons += [f"trap: {t['trap']}" for t in fails]
    if explained:
        reasons += [f"a rival explanation accounts for it: {a['explanation'][:120]}" for a in explained]
    hv = holdout["verdict"]
    if hv in ("reversed", "not replicated"):
        reasons.append(f"held-out half: {hv}")
    if reasons:
        return {"grade": "rejected", "status": "rejected", "reasons": reasons}
    agreement = subgroups.get("agreement")
    open_rivals = [a for a in alternatives if a.get("verdict") == "inconclusive"]
    if open_rivals:
        g = "provisional"                          # a rival partly accounts for it: it may be someone else's effect
        reasons += [f"a rival explanation may partly account for it: {a['explanation'][:120]}" for a in open_rivals]
    elif hv == "replicated" and (agreement is None or agreement >= 0.75) and len(survived) >= 2 and len(warns) <= 1:
        g = "strong"
    elif hv == "replicated" and (agreement is None or agreement >= 0.5):
        g = "moderate"
    else:
        g = "provisional"
    reasons.append(f"held-out half: {hv}")
    reasons.append(f"subgroups: {agreement if agreement is not None else 'too few to test'} agreement across "
                   f"{subgroups.get('tested', 0)} tested")
    reasons.append(f"rival explanations: {len(survived)} tested and ruled out, {len(open_rivals)} inconclusive, "
                   f"{sum(a.get('verdict') == 'untestable' for a in alternatives)} untestable")
    if warns:
        reasons.append("caveats: " + "; ".join(t["trap"] for t in warns))
    return {"grade": g, "status": "confirmed" if g in ("strong", "moderate") else "provisional", "reasons": reasons}


def plain_statement_problem(statement: str) -> str | None:
    """Statements are the researcher's words; every number comes from code, so statements carry none."""
    if re.search(r"\d", statement or ""):
        return "state the pattern in words without numbers; the counts and tests are added by code"
    if len((statement or "").split()) < 5:
        return "state the pattern as a full sentence"
    return None
