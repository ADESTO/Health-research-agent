"""Number check: resolve "[unverified]" numbers after the report is written.

The report audit tags every "n of N" that code cannot trace to a claim, a run fact or a counting tool.
Here an agent works through those numbers, but it cannot make a number verified by itself:

    measure  it writes a claim that counts exactly what the sentence describes; code verifies the claim,
             then CODE writes the measured count into the sentence (same number: backed; different: corrected)
    drop     it rewrites the sentence without the number; code refuses rewrites that add numbers, lose
             citations or grow the sentence
    keep     it leaves the number (it stays tagged)

Then the report is audited again. Anything still untraceable keeps its tag.
"""
from __future__ import annotations

import json
import re

from research_agent.agents.base import Agent
from research_agent.agents.report import _N_OF_M, _int, allowed_counts, normalise_citations
from research_agent.tools.base import INT, STR, obj
from research_agent.tools.claims import PREDICATE_DOC, PROPOSE_TOOL, TEST_TOOL, verify_claims
from research_agent.tools.extraction import ANALYSIS_TOOLS
from research_agent.tools.recheck import RECHECK_TOOL

MAX_ITEMS = 40
ROUNDS = 2
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z*_(\[])|\n")
_LIST_MARK = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_CITES = re.compile(r"\[(?:C\d+|PMC\d+|arXiv:[^\]\s]+)\]")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_PCT_AFTER = re.compile(r"^((?:\s+(?:papers|studies|articles|records))?\s*\(?\s*)(\d{1,3}(?:\.\d)?)\s*%")
# Words that say how many. A rewrite may not introduce one the original did not have: turning "3 of 16"
# into "most papers" claims something nobody counted.
_QUANTIFIERS = re.compile(r"\b(most|majority|minority|almost all|nearly all|all|every|vast|many|few|rare|rarely|"
                          r"none|no papers?|never|only|dominant|dominat\w*|common|uncommon|widely|seldom)\b", re.I)


def _sentence_at(body: str, start: int, end: int) -> str:
    """The sentence (or list item / table row) containing body[start:end]."""
    lo = max((m.end() for m in _SENTENCE_END.finditer(body, 0, start)), default=0)
    nxt = _SENTENCE_END.search(body, end)
    sentence = body[lo:nxt.start() if nxt else len(body)].strip()
    return _LIST_MARK.sub("", sentence)   # a bullet or number belongs to the list, not to the sentence


def unverified_items(body: str, allowed: set[tuple[int, int]]) -> list[dict]:
    """Every "n of N" in the report that code cannot trace, with the sentence it sits in."""
    items = []
    for m in _N_OF_M.finditer(body):
        n, d = _int(m.group(1)), _int(m.group(2))
        if (n, d) in allowed or n > d or d < 5:
            continue
        sentence = _sentence_at(body, m.start(), m.end())
        if sentence.startswith("|") or sentence.startswith("#"):
            continue   # tables and headings are code-generated or structural
        items.append({"item": f"U{len(items) + 1}", "number": m.group(0), "n": n, "N": d, "sentence": sentence})
        if len(items) >= MAX_ITEMS:
            break
    return items


NUMBER_CHECK = Agent(
    name="number_check",
    role="Backs, corrects or removes report numbers that code could not trace to a count.",
    system=f"""You are the Number Check agent. The report has been written, and code found numbers in it
("n of N") that no verified count backs. For each item decide one action:

- measure: find what the sentence counts, test it with test_claim, then call propose_claim with a claim that
  counts EXACTLY the population the sentence describes (use `where` for subgroups such as full-text papers,
  PMC papers or a region; over='stated' for "papers that report X"). Put its claim_id in your fix. Code then
  writes the measured count into the sentence, even if it differs from the old one. Never adjust a claim to
  reproduce the old number: the measured count is the answer.
- drop: the number cannot be measured from the extracted fields (for example a sum across several method
  families that would double-count papers, or a fact about the corpus). Give the sentence rewritten without
  the number. Keep its meaning, its citations and every other number; add no new numbers; do not make it
  stronger ("only a few" when you do not know the count is small is not allowed: say "not measured").
- keep: leave it. It stays tagged [unverified].

Predicate formats: {PREDICATE_DOC}.
Work through all items; several test_claim calls per turn are fine. Then call finish with one fix per item.""",
    tools=[TEST_TOOL, PROPOSE_TOOL, RECHECK_TOOL] + [t for t in ANALYSIS_TOOLS if t.name in
                                       ("value_counts", "extraction_coverage", "cross_tab", "list_extractions")],
    finish_schema=obj({"fixes": {"type": "array", "items": obj({
        "item": STR, "action": {"type": "string", "enum": ["measure", "drop", "keep"]},
        "claim_id": INT, "sentence": {**STR, "description": "for drop: the rewritten sentence"},
        "reason": STR}, ["item", "action"])}}, ["fixes"]),
    max_turns=16,
)


def _measured_sentence(sentence: str, number: str, n: int, d: int, cite: str | None) -> str | None:
    i = sentence.find(number)
    if i < 0:
        return None
    rest = sentence[i + len(number):]
    pct = _PCT_AFTER.match(rest)
    if pct:   # "(26%)" just after the count ("7 of 40 papers (18%)") follows the count
        rest = pct.group(1) + f"{round(100 * n / d) if d else 0}%" + rest[pct.end():]
    out = sentence[:i] + f"{n} of {d}" + rest
    if cite and cite not in out:   # the citation goes at the end of the sentence, where readers expect it
        m = re.search(r"[.!?]\s*$", out)
        out = (out[:m.start()] + f" {cite}" + out[m.start():]) if m else f"{out} {cite}"
    return out


def _drop_ok(old: str, new: str, number: str) -> str | None:
    new = (new or "").strip()
    if not new:
        return "empty sentence"
    if len(new) > len(old) + 40:
        return "rewrite is much longer than the original"
    if set(_CITES.findall(old)) - set(_CITES.findall(new)):
        return "rewrite loses a citation"
    if _N_OF_M.search(new):
        return "rewrite still contains an 'n of N' count"
    old_nums = _NUM.findall(old.replace(number, " "))
    if set(_NUM.findall(new)) - set(old_nums):
        return "rewrite adds a number"
    added = {w.lower() for w in _QUANTIFIERS.findall(new)} - {w.lower() for w in _QUANTIFIERS.findall(old)}
    if added:
        return f"rewrite adds an uncounted quantity word ({', '.join(sorted(added))})"
    return None


def apply_fixes(ctx, body: str, items: list[dict], fixes: list[dict]) -> tuple[str, dict]:
    by_item = {it["item"]: it for it in items}
    measured_ids = [int(f["claim_id"]) for f in fixes if f.get("action") == "measure" and f.get("claim_id")]
    if measured_ids:
        verify_claims(ctx, measured_ids)
    claims = {c["id"]: c for c in ctx.pg.execute(
        "SELECT id, status, claim_type, result FROM claims WHERE run_id=%s AND id = ANY(%s)",
        (ctx.run_id, measured_ids or [0])).fetchall()}
    stats = {"backed": 0, "corrected": 0, "removed": 0, "refused": 0, "log": []}
    current: dict[str, str] = {}   # original sentence -> its text after earlier edits
    for f in fixes:
        it = by_item.get(f.get("item"))
        if not it or f.get("action") not in ("measure", "drop"):
            continue
        sentence = current.get(it["sentence"], it["sentence"])
        if sentence not in body:
            stats["refused"] += 1; stats["log"].append(f"{it['item']}: sentence no longer found")
            continue
        if f["action"] == "measure":
            c = claims.get(int(f.get("claim_id") or 0))
            r = (c or {}).get("result") or {}
            if not c or c["claim_type"] != "prevalence" or c["status"] not in ("supported", "unsupported") \
                    or r.get("denominator") is None:
                stats["refused"] += 1; stats["log"].append(f"{it['item']}: claim missing, rejected or not a count")
                continue
            n, d = int(r.get("n_matching") or 0), int(r["denominator"])
            if d != it["N"]:
                stats["refused"] += 1
                stats["log"].append(f"{it['item']}: claim counts out of {d}, the sentence out of {it['N']}")
                continue
            new = _measured_sentence(sentence, it["number"], n, d,
                                     f"[C{c['id']}]" if c["status"] == "supported" else None)
            if new is None:
                stats["refused"] += 1; stats["log"].append(f"{it['item']}: number no longer in its sentence")
                continue
            stats["backed" if n == it["n"] else "corrected"] += 1
            stats["log"].append(f"{it['item']}: {it['number']} -> {n} of {d} (C{c['id']})")
        else:
            problem = _drop_ok(sentence, f.get("sentence"), it["number"])
            if problem:
                stats["refused"] += 1; stats["log"].append(f"{it['item']}: drop refused, {problem}")
                continue
            new = f["sentence"].strip()
            stats["removed"] += 1
            stats["log"].append(f"{it['item']}: {it['number']} removed")
        body = body.replace(sentence, new, 1)
        current[it["sentence"]] = new
    return body, stats


def resolve_numbers(ctx, body: str) -> tuple[str, dict]:
    """Run the Number Check agent on the report's untraceable numbers (at most ROUNDS rounds)."""
    body = normalise_citations(body)
    total = {"backed": 0, "corrected": 0, "removed": 0, "refused": 0, "log": []}
    for rnd in range(ROUNDS):
        items = unverified_items(body, allowed_counts(ctx))
        if not items:
            break
        ctx.emit("number_check", "message", {"text": f"round {rnd + 1}: {len(items)} untraced numbers"})
        raw = NUMBER_CHECK.run(ctx, "Resolve these numbers:\n" + json.dumps(
            [{k: it[k] for k in ("item", "number", "sentence")} for it in items], ensure_ascii=False))
        body, stats = apply_fixes(ctx, body, items, raw.get("fixes") or [])
        for k in ("backed", "corrected", "removed", "refused"):
            total[k] += stats[k]
        total["log"] += stats["log"]
        if not (stats["backed"] or stats["corrected"] or stats["removed"]):
            break   # nothing changed: a second round would repeat the first
    ctx.save_note("number_check", total)
    return body, total
