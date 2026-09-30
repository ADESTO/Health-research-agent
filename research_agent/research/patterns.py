"""Patterns the researcher can test, and the statistics that test them.

A pattern is a spec code can count, never a sentence. Four kinds:

    difference   the share of papers with an OUTCOME differs between GROUP A and GROUP B
                 {"kind": "difference", "outcome": [cond...], "group_a": [cond...], "group_b": [cond...]}
                 (group_b may be omitted: then it is every paper not in group A)
    prevalence   an OUTCOME is rare (max_share) or common (min_share), optionally WITHIN a subgroup
                 {"kind": "prevalence", "outcome": [cond...], "within": [cond...], "max_share": 0.1}
    comparison   within the same papers, method family A beats family B on the same metric and split
                 {"kind": "comparison", "method_a": [...], "method_b": [...], "metric": "RMSE"}
    trend        an OUTCOME becomes more (or less) common after a year
                 {"kind": "trend", "outcome": [cond...], "split_year": 2020, "within": [cond...]}

A condition is {"field": ..., "any_of": [...]} or {"field": ..., "none_of": [...]}, on any extracted field,
the run's question-specific fields, or "corpus" (arxiv|pmc), "read" (fulltext|abstract) and "year" (min/max).

Everything here is pure: functions take the paper rows to count over, so the same code tests a pattern on the
discovery half, the held-out half, or any subgroup.
"""
from __future__ import annotations

import json
import math

KINDS = ("difference", "prevalence", "comparison", "trend")
MIN_CELL = 3            # below this many papers in a group, a comparison is not testable


# ---------------------------------------------------------------- statistics
def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]] (rows: groups; columns: outcome yes/no)."""
    n1, n2, k, n = a + b, c + d, a + c, a + b + c + d
    if n == 0 or n1 == 0 or n2 == 0 or k == 0 or k == n:
        return 1.0

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    observed = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= observed * (1 + 1e-9)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_tail(k: int, n: int, p0: float, upper: bool) -> float:
    """One-sided exact binomial p-value: P(X >= k) if upper else P(X <= k), under share p0."""
    if n == 0:
        return 1.0
    rng = range(k, n + 1) if upper else range(0, k + 1)
    return min(1.0, sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in rng))


# ---------------------------------------------------------------- specs
def _conds(x) -> list[dict]:
    if x is None:
        return []
    return [x] if isinstance(x, dict) else [c for c in x if isinstance(c, dict)]


def validate(spec: dict, lists: list[str], enums: dict) -> str | None:
    """None when the spec can be counted; otherwise what to fix."""
    from research_agent.tools.claims import SPECIAL_FIELDS, field_misfit

    kind = spec.get("kind")
    if kind not in KINDS:
        return f"kind must be one of {', '.join(KINDS)}"
    known = set(lists) | set(enums) | set(SPECIAL_FIELDS) | {"source"}

    def check(conds, name, required=True):
        conds = _conds(conds)
        if required and not conds:
            return f"{name} needs at least one condition like {{'field': 'methods', 'any_of': ['random forest']}}"
        for c in conds:
            f = c.get("field")
            if f not in known:
                return f"{name}: field must be one of {sorted(known)}"
            if f == "year":
                if c.get("min") is None and c.get("max") is None:
                    return f"{name}: a year condition needs min and/or max"
            elif not (c.get("any_of") or c.get("none_of")):
                return f"{name}: condition on {f} needs any_of or none_of"
            misfit = field_misfit(f, c.get("any_of") or [])
            if misfit:
                return f"{name}: {misfit}"
        return None

    if kind == "comparison":
        if not spec.get("method_a") or not spec.get("method_b"):
            return "comparison needs method_a and method_b (lists of model-name terms)"
        return None
    err = check(spec.get("outcome"), "outcome")
    if err:
        return err
    if kind == "difference":
        return check(spec.get("group_a"), "group_a") or check(spec.get("group_b"), "group_b", required=False)
    if kind == "prevalence":
        if spec.get("max_share") is None and spec.get("min_share") is None:
            return "prevalence needs max_share (a rarity pattern) or min_share (a commonness pattern)"
        return check(spec.get("within"), "within", required=False)
    if kind == "trend":
        if not isinstance(spec.get("split_year"), int):
            return "trend needs split_year (an integer year: papers after it are 'late')"
        return check(spec.get("within"), "within", required=False)
    return None


def signature(spec: dict) -> str:
    """The same pattern tested twice has the same signature, whatever the order of its lists."""
    def norm(x):
        if isinstance(x, dict):
            return {k: norm(v) for k, v in sorted(x.items()) if k not in ("description",)}
        if isinstance(x, list):
            items = [norm(v) for v in x]
            return sorted(items, key=lambda v: json.dumps(v, sort_keys=True).lower())
        return x.lower().strip() if isinstance(x, str) else x
    return json.dumps(norm(spec), sort_keys=True)


def describe(spec: dict) -> str:
    def conds(cs):
        out = []
        for c in _conds(cs):
            if c.get("field") == "year":
                out.append(f"year {c.get('min', '')}-{c.get('max', '')}")
            elif c.get("any_of"):
                out.append(f"{c['field']} in {c['any_of']}")
            else:
                out.append(f"{c['field']} not in {c.get('none_of')}")
        return " and ".join(out) or "all papers"
    k = spec.get("kind")
    if k == "difference":
        return (f"share with [{conds(spec.get('outcome'))}] differs between [{conds(spec.get('group_a'))}] and "
                f"[{conds(spec.get('group_b')) if spec.get('group_b') else 'the other papers'}]")
    if k == "prevalence":
        bound = f"at most {spec['max_share']:.0%}" if spec.get("max_share") is not None else f"at least {spec['min_share']:.0%}"
        return f"[{conds(spec.get('outcome'))}] is {bound} of [{conds(spec.get('within'))}]"
    if k == "trend":
        return f"share with [{conds(spec.get('outcome'))}] changes after {spec.get('split_year')} among [{conds(spec.get('within'))}]"
    if k == "comparison":
        return f"{spec.get('method_a')} vs {spec.get('method_b')} within the same papers ({spec.get('metric') or 'any metric'})"
    return str(spec)


# ---------------------------------------------------------------- evaluation
def _match(r: dict, conds, enums) -> bool:
    from research_agent.tools.claims import row_matches

    return all(row_matches(r, c, enums) for c in _conds(conds))


def evaluate(spec: dict, rows: list[dict], enums: dict, ctx=None) -> dict:
    """Count the pattern over `rows`. Returns n, groups, effect, p, direction (+1 / -1 / 0), testable."""
    kind = spec["kind"]
    if kind == "comparison":
        from research_agent.tools.results import method_comparison

        r = method_comparison(ctx, spec["method_a"], spec["method_b"], spec.get("metric"), rows=rows)
        wins, losses = r["A_better"], r["B_better"]
        return {"kind": kind, "n": r["papers_with_head_to_head"], "a_better": wins, "b_better": losses,
                "ties": r["ties"], "p": r["sign_test_p"], "direction": (wins > losses) - (wins < losses),
                "testable": wins + losses >= MIN_CELL,
                "papers": sorted({c["paper_id"] for c in r["comparisons"]})}
    if kind == "prevalence":
        pool = [r for r in rows if _match(r, spec.get("within"), enums)]
        hits = [r["paper_id"] for r in pool if _match(r, spec.get("outcome"), enums)]
        n, k = len(pool), len(hits)
        lo, hi = wilson(k, n)
        if spec.get("max_share") is not None:
            bound = float(spec["max_share"])
            p = binom_tail(k, n, bound, upper=False)        # evidence the share is below the bound
            holds = hi <= bound or p < 0.05
        else:
            bound = float(spec["min_share"])
            p = binom_tail(k, n, bound, upper=True)
            holds = lo >= bound or p < 0.05
        return {"kind": kind, "n": n, "k": k, "share": round(k / n, 3) if n else None, "ci": [round(lo, 3), round(hi, 3)],
                "bound": bound, "p": round(p, 4), "direction": 1 if holds else -1, "testable": n >= 2 * MIN_CELL,
                "papers": hits[:40]}
    # difference and trend are both a 2x2 table: group x outcome
    if kind == "trend":
        pool = [r for r in rows if _match(r, spec.get("within"), enums)]
        y = int(spec["split_year"])
        group_a = [r for r in pool if (r.get("year") or 0) > y]           # late
        group_b = [r for r in pool if 0 < (r.get("year") or 0) <= y]      # early
        overlap = 0
    else:
        group_a = [r for r in rows if _match(r, spec.get("group_a"), enums)]
        if spec.get("group_b"):
            group_b = [r for r in rows if _match(r, spec.get("group_b"), enums)]
        else:
            ids = {r["paper_id"] for r in group_a}
            group_b = [r for r in rows if r["paper_id"] not in ids]
        both = {r["paper_id"] for r in group_a} & {r["paper_id"] for r in group_b}
        overlap = len(both)
        group_a = [r for r in group_a if r["paper_id"] not in both]       # a paper cannot be on both sides
        group_b = [r for r in group_b if r["paper_id"] not in both]
    ya = [r["paper_id"] for r in group_a if _match(r, spec.get("outcome"), enums)]
    yb = [r["paper_id"] for r in group_b if _match(r, spec.get("outcome"), enums)]
    na, nb = len(group_a), len(group_b)
    a, c = len(ya), len(yb)
    pa, pb = (a / na if na else None), (c / nb if nb else None)
    p = fisher_exact(a, na - a, c, nb - c)
    direction = 0 if pa is None or pb is None else (pa > pb) - (pa < pb)
    return {"kind": kind, "n": na + nb, "group_a": {"n": na, "with_outcome": a, "share": round(pa, 3) if pa is not None else None},
            "group_b": {"n": nb, "with_outcome": c, "share": round(pb, 3) if pb is not None else None},
            "difference": round(pa - pb, 3) if pa is not None and pb is not None else None,
            "p": round(p, 4), "direction": direction, "testable": na >= MIN_CELL and nb >= MIN_CELL,
            "overlap_removed": overlap, "cells": [a, na - a, c, nb - c],
            "papers": {"a_with": ya[:25], "b_with": yb[:25]}}


def mantel_haenszel(tables: list[list[int]]) -> dict:
    """Association between group and outcome after adjusting for strata (a confounder's levels).
    tables: [[a, b, c, d], ...] with rows group A / group B and columns outcome yes / no.
    Returns the adjusted risk difference and the Cochran-Mantel-Haenszel test's p-value."""
    num = den = sa = se = sv = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        n1, n0, m1 = a + b, c + d, a + c
        if n < 2 or n1 == 0 or n0 == 0:
            continue
        num += (a * n0 - c * n1) / n
        den += n1 * n0 / n
        sa += a
        se += n1 * m1 / n
        sv += n1 * n0 * m1 * (n - m1) / (n * n * (n - 1))
    if not den or not sv:
        return {"adjusted_difference": None, "p": 1.0}
    chi2 = max(0.0, abs(sa - se) - 0.5) ** 2 / sv
    return {"adjusted_difference": round(num / den, 3), "p": round(math.erfc(math.sqrt(chi2 / 2)), 4)}


def significant(result: dict, alpha: float = 0.05) -> bool:
    if result["kind"] == "prevalence":
        return result["direction"] == 1 and result["testable"]
    return result["testable"] and result["direction"] != 0 and result["p"] < alpha


# ---------------------------------------------------------------- the held-out split
def split_rows(rows: list[dict], seed: str, method: str = "random", year: int | None = None) -> tuple[list, list]:
    """(discovery, holdout). Random halves by a stable hash of the paper id, or by year (discovery = up to and
    including `year`, holdout = later papers: a pattern must then also hold for newer work)."""
    import hashlib

    if method == "year" and year:
        return [r for r in rows if (r.get("year") or 0) <= year], [r for r in rows if (r.get("year") or 0) > year]
    disc, hold = [], []
    for r in rows:
        h = int(hashlib.sha256(f"{seed}:{r['paper_id']}".encode()).hexdigest(), 16)
        (disc if h % 2 == 0 else hold).append(r)
    return disc, hold
