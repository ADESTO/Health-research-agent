"""The run as a graph and as a mind map, for the web page.

The graph answers "how is this literature connected?"; the mind map answers "what does it contain, what
evidence supports it, where does it disagree, and where are the gaps?".

Graph nodes:  question, category (one per extracted field), concept (a value papers share: "random forest",
              "Kenya"), paper, claim (C12), map item (gap G2, untried combination N1, design D1), reported gap
              (ordinary runs), driver (a factor studies disagree about).
Graph links:  paper -> concept it uses; claim -> concepts it counted and the papers it counted; gap -> concept it
              is about; design -> gaps it addresses and papers it builds on; driver -> papers reporting a positive
              or negative association; paper -> paper it cites (when citation data is cached); concept -> concept
              used together in the same papers ("co", weighted by the number of papers); concept -> concept that
              are both common but never used together in these papers ("untried", with the number expected if
              they were independent).

Mind map:     a short title over the question, then sections (Methods, Data, Study settings, Forecast or study
              design, Evidence, Research gaps). Concepts carry "n/N papers"; under each concept its papers, and
              under each paper its other key attributes, so the map is a way through the literature. A gap opens
              into its evidence, how the papers are distributed on that point (the trail behind the gap), and
              the designs that address it. A claim opens into the papers it counted and those it did not.

Everything comes from the database: extracted records, claims, map notes, results and the citation cache.
No model calls, so building it is free and instant.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from research_agent.tools.claims import _matcher, _norm

# (field, label, section) for the general extraction fields
FIELDS = [("methods", "Methods", "methods"), ("data_modalities", "Data types", "data"),
          ("datasets", "Datasets", "data"), ("geography", "Study settings", "settings"),
          ("validation_level", "Validation", "design"), ("code_or_data_available", "Code or data shared", "design"),
          ("study_designs", "Study designs", "design"), ("populations", "Populations", "population"),
          ("organisms", "Organisms", "population"), ("interventions", "Drugs and interventions", "interventions"),
          ("outcomes", "Outcomes", "interventions"), ("mechanisms", "Mechanisms", "biology"),
          ("targets", "Molecular targets", "biology")]
SECTIONS = [("methods", "Methods"), ("data", "Data"), ("settings", "Study settings"),
            ("population", "Who or what was studied"), ("interventions", "Interventions and outcomes"),
            ("biology", "Mechanisms and targets"), ("design", "Study design")]
TOP_PER_FIELD = 12
CO_LINKS = 40                # most frequent pairs of concepts used together, across different fields
# field pairs where a common pair that nobody combined is worth seeing
UNTRIED_PAIRS = [("methods", "data_modalities"), ("interventions", "organisms"), ("interventions", "populations"),
                 ("study_designs", "interventions"), ("methods", "geography")]
UNTRIED_MAX = 8
PAPERS_PER_BRANCH = 15       # mind map: papers listed under one concept before "+N more"
ATTRS_PER_PAPER = 6          # mind map: other branches listed under a paper
_SKIP = {"", "not_stated", "none_stated", "n/a"}
_SECTION_WORDS = [("interventions", r"drug|dose|regimen|compound|intervention|therap|antibiotic|treatment|route"),
                  ("biology", r"mechanism|pathway|target|receptor|gene|enzyme|hormone"),
                  ("population", r"organism|pathogen|species|isolate|strain|patient|participant|cohort|population"),
                  ("data", r"data|satellite|source|covariate|climate|remote|sensing|input|variable|feature"),
                  ("settings", r"country|region|setting|district|site|location|place|population"),
                  ("methods", r"model|method|architecture|algorithm|learner|approach")]


def _key(value) -> str:
    return re.sub(r"\s+", " ", _norm(str(value))).strip()


def _short(title: str, n: int = 48) -> str:
    title = " ".join((title or "").split())
    return title if len(title) <= n else title[:n].rsplit(" ", 1)[0] + "…"


def _url(pid: str, corpus: str | None) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pid}/" if (corpus == "pmc" or pid.startswith("PMC")) \
        else f"https://arxiv.org/abs/{pid}"


def _counted(result) -> str | None:
    r = result or {}
    if r.get("denominator") is not None:
        return f"{r.get('n_matching')} of {r['denominator']}"
    w = r.get("window") or {}
    if w.get("early_mean") is not None and w.get("late_mean") is not None:
        unit = f"% of '{r['within']}' papers" if r.get("within") else " per 10,000 papers"
        return f"from {round(w['early_mean'], 1)} to {round(w['late_mean'], 1)}{unit}"
    return None


def _section_of(field: str) -> str:
    name = field.removeprefix("q_")
    for section, rx in _SECTION_WORDS:
        if re.search(rx, name):
            return section
    return "design"


def _values(d: dict, field: str) -> list[str]:
    v = d.get(field)
    return [x for x in (v if isinstance(v, list) else [v]) if x is not None and _key(x) not in _SKIP]


# what tells one paper from another at a glance, in order of preference
_TAG_FIELDS = [("geography",), ("interventions", "methods", "organisms", "study_designs"),
               ("data_modalities", "populations", "outcomes")]


def paper_tags(rows: list[dict]) -> dict[str, str]:
    """A short label per paper that tells it apart from the others: "Uganda · 2021 · LSTM · full text".
    Titles in one literature often differ only at the end, so they are no use cut short."""
    def part(d: dict, group: tuple, skip: set) -> str | None:
        for f in group:
            for v in _values(d, f):
                v = str(v).replace("_", " ")
                if v.lower() not in skip:
                    return v if len(v) <= 28 else v[:27] + "…"
        return None

    tags, extra = {}, {}
    for r in rows:
        d = r["data"]
        place = part(d, _TAG_FIELDS[0], set())
        what = part(d, _TAG_FIELDS[1], set())
        bits = [b for b in (place, str(r.get("year") or ""), what) if b]
        if r.get("source") == "fulltext":
            bits.append("full text")
        tags[r["paper_id"]] = " · ".join(bits) if len(bits) >= 2 else _short(r.get("title") or r["paper_id"], 40)
        extra[r["paper_id"]] = part(d, _TAG_FIELDS[2], {(what or "").lower()}) or part(
            d, ("methods", "interventions"), {(what or "").lower()})
    seen = Counter(tags.values())
    for pid, t in list(tags.items()):           # still the same as another paper: add one more detail, then the id
        if seen[t] > 1 and extra.get(pid):
            tags[pid] = f"{t} · {extra[pid]}"
    seen = Counter(tags.values())
    for pid, t in list(tags.items()):
        if seen[t] > 1:
            tags[pid] = f"{t} · {pid}"
    return tags


def short_title(ctx, rows: list[dict]) -> str:
    """"Malaria forecasting" from the papers' most common health domain and task; the question as fallback."""
    domains = Counter(_key(x) for r in rows for x in _values(r["data"], "health_domains"))
    tasks = Counter(_key(x) for r in rows for x in _values(r["data"], "task_types"))
    dom = domains.most_common(1)[0][0] if domains else ""
    task = tasks.most_common(1)[0][0] if tasks else ""
    if dom:
        return (f"{dom} {task}" if task else f"{dom} research").strip().capitalize()
    return _short(ctx.question, 40)


def build_graph(ctx) -> dict:
    from research_agent.tools.extraction import _rows, protocol_of

    rows = _rows(ctx)
    n_papers = len(rows)
    nodes: dict[str, dict] = {}
    links: list[dict] = []
    seen_links: set[tuple] = set()

    def node(nid: str, **kw) -> str:
        if nid not in nodes:
            nodes[nid] = {"id": nid, **kw}
        return nid

    def link(a: str, b: str, kind: str) -> None:
        if a in nodes and b in nodes and a != b and (a, b, kind) not in seen_links:
            seen_links.add((a, b, kind))
            links.append({"source": a, "target": b, "type": kind})

    title = short_title(ctx, rows)
    tasks = Counter(_key(x) for r in rows for x in _values(r["data"], "task_types"))
    forecasting = bool(tasks) and "forecast" in tasks.most_common(1)[0][0]
    root = node("question", type="question", label=title, text=ctx.question, papers=n_papers)

    # ---- categories (one per field) and the concepts papers share
    fields = list(FIELDS)
    protocol = protocol_of(ctx) or {}
    fields += [(f["name"], f["name"].removeprefix("q_").replace("_", " ").capitalize(), _section_of(f["name"]))
               for f in protocol.get("fields", [])]
    field_label = {f: lab for f, lab, _ in fields}
    field_section = {f: sec for f, _, sec in fields}
    min_papers = 2 if n_papers >= 15 else 1
    concept_of: dict[str, dict[str, str]] = {}      # field -> normalised value -> node id
    for field, label, section in fields:
        counts, spelling = Counter(), defaultdict(Counter)
        for r in rows:
            for k, v in {_key(x): x for x in _values(r["data"], field)}.items():
                counts[k] += 1
                spelling[k][str(v)] += 1
        kept = [(k, c) for k, c in counts.most_common() if c >= min_papers][:TOP_PER_FIELD]
        if not kept:
            continue
        cat = node(f"cat:{field}", type="category", label=label, field=field, section=section)
        link(root, cat, "has")
        concept_of[field] = {}
        for k, c in kept:
            shown = spelling[k].most_common(1)[0][0].replace("_", " ")
            if shown.lower() in ("yes", "no", "true", "false", "none", "other", "partial", "unclear"):
                shown = f"{label}: {shown}"            # "yes" alone says nothing on a graph
            cid = node(f"concept:{field}:{k}", type="concept", label=shown, field=field, category=label,
                       section=section, papers=c, share=round(c / max(1, n_papers), 3))
            concept_of[field][k] = cid
            link(cat, cid, "has")

    # ---- papers, with the attributes the compare table and the mind map show
    paper_value: dict[str, dict[str, list[str]]] = {}
    tags = paper_tags(rows)
    for r in rows:
        d, pid = r["data"], r["paper_id"]
        ev = d.get("evidence") or {}
        attrs = {label: [str(x).replace("_", " ") for x in _values(d, f)][:5] for f, label, _ in fields}
        paper_value[pid] = {f: [_key(x) for x in _values(d, f)] for f, _, _ in fields}
        node(f"paper:{pid}", type="paper", label=tags[pid], short=_short(r["title"]), title=r["title"], paper_id=pid,
             year=r["year"], corpus=r.get("corpus"), read=r["source"], url=_url(pid, r.get("corpus")),
             findings=str(d.get("key_findings") or "")[:300],
             results=[f"{x['metric']} {x['value']} ({x['model']})" for x in (d.get("reported_results") or [])[:4]],
             attrs={k: v for k, v in attrs.items() if v},
             quotes={f: (ev.get(f) or [""])[0][:240] for f, _, _ in fields if ev.get(f)})
        for field, _, _ in fields:
            for k in paper_value[pid][field]:
                cid = concept_of.get(field, {}).get(k)
                if cid:
                    link(f"paper:{pid}", cid, "uses")

    # ---- concepts used together, and common pairs nobody combined
    used: dict[str, set[str]] = defaultdict(set)          # concept -> papers
    for l in links:
        if l["type"] == "uses":
            used[l["target"]].add(l["source"])
    pair_n: Counter = Counter()
    for pid in paper_value:
        mine = sorted(c for c in (concept_of.get(f, {}).get(k) for f, vals in paper_value[pid].items() for k in vals)
                      if c)
        mine = list(dict.fromkeys(mine))
        for i, a in enumerate(mine):
            for b in mine[i + 1:]:
                if nodes[a]["field"] != nodes[b]["field"]:
                    pair_n[(a, b)] += 1
    # Pairs are ranked by how much more often they go together than chance would give (lift), so a concept
    # nearly every paper has does not link to everything.
    common = {c for c, ps in used.items() if len(ps) > 0.75 * n_papers}
    scored = []
    for (a, b), n in pair_n.items():
        if n < 2 or a in common or b in common:
            continue
        lift = n * n_papers / max(1, len(used[a]) * len(used[b]))
        if lift > 1.1:
            scored.append((n * (lift - 1), n, round(lift, 2), a, b))
    for _, n, lift, a, b in sorted(scored, reverse=True)[:CO_LINKS]:
        links.append({"source": a, "target": b, "type": "co", "n": n, "lift": lift})
        seen_links.add((a, b, "co"))
    untried = []
    floor = max(3, round(0.1 * n_papers))
    for fa, fb in UNTRIED_PAIRS:
        tops = {f: sorted((c for c in concept_of.get(f, {}).values() if len(used[c]) >= floor),
                          key=lambda c: -len(used[c]))[:5] for f in (fa, fb)}
        for a in tops[fa]:
            for b in tops[fb]:
                if used[a] & used[b]:
                    continue
                expected = len(used[a]) * len(used[b]) / max(1, n_papers)
                if expected >= 1.5:
                    untried.append((expected, a, b))
    for expected, a, b in sorted(untried, reverse=True)[:UNTRIED_MAX]:
        links.append({"source": a, "target": b, "type": "untried", "n": 0, "expected": round(expected, 1)})
        seen_links.add((a, b, "untried"))

    # ---- claims: linked to what they counted
    enum_fields = {"validation_level", "code_or_data_available"} | {
        f["name"] for f in protocol.get("fields", []) if f.get("type") == "enum"}
    all_ids = [r["paper_id"] for r in rows]
    for c in ctx.pg.execute("SELECT id, text, claim_type, predicate, status, result FROM claims WHERE run_id=%s "
                            "ORDER BY id", (ctx.run_id,)).fetchall():
        res = c["result"] or {}
        p = c["predicate"] or {}
        matched = [x for x in res.get("matched_paper_ids") or [] if f"paper:{x}" in nodes]
        not_counted = ([x for x in all_ids if x not in set(matched)]
                       if c["claim_type"] == "prevalence" and res.get("denominator") is not None and not p.get("where")
                       else None)
        cid = node(f"claim:C{c['id']}", type="claim", label=f"C{c['id']}", text=c["text"], status=c["status"],
                   counted=_counted(res), claim_id=f"C{c['id']}", denominator=res.get("denominator"),
                   matched=matched, not_counted=not_counted)
        field = p.get("field")
        needles = [_norm(str(x)) for x in p.get("any_of") or []]
        hit = False
        for k, concept in concept_of.get(field, {}).items():
            if any((n == k) if field in enum_fields else _matcher(n)(k) for n in needles):
                link(cid, concept, "counts")
                hit = True
        if not hit and f"cat:{field}" in nodes:
            link(cid, f"cat:{field}", "counts")
        for pid in matched[:15]:
            link(cid, f"paper:{pid}", "counted")
        if not any(l["source"] == cid for l in links):
            link(cid, root, "about")

    # ---- gaps: map items or the Gap agent's gaps, each with the trail behind it
    notes = ctx.notes()
    m = (notes.get("map") or {}).get("map") or {}

    def concept_for(field: str, value: str, label: str, n=None) -> str | None:
        if not field:
            return None
        members = [_key(v) for v in str(value or "").split("|") if v]
        for k in members:
            if k in concept_of.get(field, {}):
                return concept_of[field][k]
        if not members:
            return None
        cat = f"cat:{field}"
        if cat not in nodes:
            cat = node(cat, type="category", label=field_label.get(field) or field.removeprefix("q_").replace("_", " ").capitalize(),
                       field=field, section=field_section.get(field) or _section_of(field))
            link(root, cat, "has")
        cid = node(f"concept:{field}:{members[0]}", type="concept", label=label.split(": ")[-1], field=field,
                   category=nodes[cat]["label"], section=nodes[cat]["section"], papers=n or 0,
                   share=round((n or 0) / max(1, n_papers), 3))
        concept_of.setdefault(field, {})[members[0]] = cid
        link(cat, cid, "has")
        return cid

    def trail(field: str, value: str, label: str = "") -> list[dict] | None:
        """How the papers are distributed on the gap's field: the evidence behind "X is rare"."""
        if not field or not any(field in pv for pv in paper_value.values()):
            return None
        target = {_key(v) for v in str(value or "").split("|") if v}
        groups: dict[str, list[str]] = defaultdict(list)
        for pid, pv in paper_value.items():
            vals = pv.get(field) or ["not stated"]
            for v in dict.fromkeys(vals):
                groups[v].append(pid)
        out = [{"value": v.replace("_", " "), "papers": ids, "is_gap": v in target} for v, ids in groups.items()]
        if target and not any(g["is_gap"] for g in out):      # nobody has it: say so, first
            label = label or next((nodes[c]["label"] for k, c in concept_of.get(field, {}).items() if k in target), None)
            out.append({"value": label or sorted(target)[0].replace("_", " "), "papers": [], "is_gap": True})
        return sorted(out, key=lambda g: (not g["is_gap"], -len(g["papers"])))[:8]

    for it in m.get("gaps", []):
        gid = node(f"item:{it['id']}", type="gap", label=f"{it['id']} {it['label']}"[:60], item_id=it["id"],
                   text=it["label"], counted=f"{it.get('n')} of {it.get('N')}" if it.get("N") else None,
                   confidence=it.get("confidence"), field=it.get("field"),
                   trail=trail(it.get("field"), it.get("value"), (it.get("label") or "").split(": ")[-1]))
        link(gid, root, "gap")
        c = concept_for(it.get("field"), it.get("value"), it.get("label", ""), it.get("n"))
        if c:
            link(gid, c, "about")
    for it in m.get("novelty", []):
        nid = node(f"item:{it['id']}", type="combination", label=f"{it['id']} {it['a']['label']} + {it['b']['label']}"[:60],
                   item_id=it["id"], text=f"Untried combination: {it['a']['label']} + {it['b']['label']}")
        link(nid, root, "gap")
        for side in (it["a"], it["b"]):
            c = concept_for(side.get("field"), side.get("value"), side.get("label", ""))
            if c:
                link(nid, c, "about")
    for d in (notes.get("design") or {}).get("designs", []):
        did = node(f"item:{d['id']}", type="design", label=f"{d['id']} {d.get('title', '')}"[:60], item_id=d["id"],
                   text=d.get("research_question") or d.get("title"), title=d.get("title"),
                   validation=d.get("validation_strategy"))
        link(did, root, "gap")
        for a in d.get("addresses") or []:
            link(did, f"item:{a}", "addresses")
        for pid in d.get("builds_on") or []:
            link(did, f"paper:{pid}", "builds_on")
    for i, g in enumerate((notes.get("gaps") or {}).get("gaps") or [], 1):
        if not isinstance(g, dict) or not g.get("gap"):
            continue
        gid = node(f"gap:{i}", type="gap", label=_short(g["gap"], 60), text=g["gap"],
                   why=g.get("why_it_matters"), confidence=g.get("confidence"))
        link(gid, root, "gap")
        for cid in g.get("claim_ids") or []:
            cid = str(cid).upper()
            link(gid, f"claim:{cid if cid.startswith('C') else 'C' + cid}", "rests_on")
        # the trail comes from the first claim it rests on that counted a field
        for l in [l for l in links if l["source"] == gid and l["type"] == "rests_on"]:
            cn = nodes[l["target"]]
            cat = next((x["target"] for x in links if x["source"] == cn["id"] and x["type"] == "counts"), None)
            if cat and cat in nodes and nodes[cat].get("field"):
                f = nodes[cat]["field"]
                nodes[gid]["field"] = f
                nodes[gid]["trail"] = trail(f, "|".join(k for k, v in concept_of.get(f, {}).items() if v == cat)
                                            if nodes[cat]["type"] == "concept" else "")
                nodes[gid]["counted"] = cn.get("counted")
                break

    for i, text in enumerate((notes.get("gaps") or {}).get("research_directions") or [], 1):
        did = node(f"direction:{i}", type="direction", label=_short(str(text), 60), text=str(text))
        link(did, root, "gap")

    # ---- where studies disagree
    try:
        from research_agent.tools.results import contradictions

        for co in contradictions(ctx)["contradictions"]:
            drv = node(f"driver:{co['driver']}", type="driver", label=f"{co['driver']} (disputed)",
                       text=f"Studies disagree about {co['driver']}: {len(co['positive'])} report a positive "
                            f"association, {len(co['negative'])} a negative one",
                       separating=[x["attribute"] for x in co["separating_attributes"][:3]],
                       no_effect=[x["paper_id"] for x in co.get("no_effect") or []])
            link(drv, root, "about")
            for side in ("positive", "negative"):
                for x in co[side]:
                    link(drv, f"paper:{x['paper_id']}", side)
    except Exception:
        pass

    # ---- citations between the papers (only what is cached; no network)
    try:
        from research_agent.tools.citations import citation_graph

        for a, b in citation_graph(ctx, fetch=False).get("edges", []):
            link(f"paper:{a}", f"paper:{b}", "cites")
    except Exception:
        pass

    degree = Counter()
    for l in links:
        if l["type"] in ("co", "untried"):
            continue
        degree[l["source"]] += 1
        degree[l["target"]] += 1
    for nid, n in nodes.items():
        n["degree"] = degree[nid]
    design_name = "Forecast design" if forecasting else "Study design"
    return {"nodes": list(nodes.values()), "links": links,
            "tree": _tree(nodes, links, n_papers, design_name),
            "stats": {"papers": n_papers, "nodes": len(nodes), "links": len(links),
                      "citations": sum(1 for l in links if l["type"] == "cites")}}


# ---------------------------------------------------------------- the mind map
def _tree(nodes: dict, links: list[dict], n_papers: int, design_name: str) -> dict:
    out_links = defaultdict(list)
    for l in links:
        out_links[l["source"]].append(l)
    papers_of = defaultdict(list)                 # concept -> papers that use it
    concepts_of = defaultdict(list)               # paper -> concepts it uses
    for l in links:
        if l["type"] == "uses":
            papers_of[l["target"]].append(l["source"])
            concepts_of[l["source"]].append(l["target"])
    N = n_papers

    def frac(k: int) -> str:
        return f"{k}/{N} papers"

    def paper(pid: str, skip: str | None = None) -> dict:
        """A paper, opening into its other key attributes (setting, data, design...)."""
        order = {"settings": 0, "population": 1, "interventions": 2, "design": 3, "biology": 4, "data": 5,
                 "methods": 6}
        others = sorted((c for c in dict.fromkeys(concepts_of[pid]) if c != skip and c in nodes),
                        key=lambda c: (order.get(nodes[c].get("section"), 9), -nodes[c].get("papers", 0)))
        n = nodes[pid]
        # where else this paper sits in the map: each opens the branch it belongs to
        return {"ref": pid, "label": n["label"], "type": "paper", "title": n.get("title"),
                "count": f"also in {len(others)}" if others else "",
                "children": [{"ref": c, "label": nodes[c]["label"], "type": "attr",
                              "count": nodes[c]["category"].lower()} for c in others[:ATTRS_PER_PAPER]]
                + ([{"label": f"+{len(others) - ATTRS_PER_PAPER} more (see the graph)", "type": "more"}]
                   if len(others) > ATTRS_PER_PAPER else [])}

    def papers(ids: list[str], skip: str | None = None) -> list[dict]:
        ids = sorted(set(i for i in ids if i in nodes), key=lambda p: -(nodes[p].get("year") or 0))
        kids = [paper(p, skip) for p in ids[:PAPERS_PER_BRANCH]]
        if len(ids) > PAPERS_PER_BRANCH:
            kids.append({"label": f"+{len(ids) - PAPERS_PER_BRANCH} more papers (see the graph)", "type": "more"})
        return kids

    def concept(c: str) -> dict:
        n = nodes[c]
        return {"ref": c, "label": n["label"], "type": "concept", "count": frac(n.get("papers", 0)),
                "children": papers(papers_of[c], skip=c)}

    def field_branch(cat: dict) -> list[dict]:
        cs = sorted((l["target"] for l in out_links[cat["id"]] if l["type"] == "has"),
                    key=lambda c: -nodes[c].get("papers", 0))
        return [concept(c) for c in cs]

    sections = []
    cats = [n for n in nodes.values() if n["type"] == "category"]
    for key, label in SECTIONS:
        mine = [c for c in cats if c.get("section") == key]
        if not mine:
            continue
        if len(mine) == 1:
            kids = field_branch(mine[0])
        else:
            kids = [{"ref": c["id"], "label": c["label"], "type": "field", "children": field_branch(c)} for c in mine]
        sections.append({"label": design_name if key == "design" else label, "type": "section", "section": key,
                         "children": kids})

    # evidence: claims (papers counted and not counted, size of the evidence base) and disagreements
    claims = [n for n in nodes.values() if n["type"] == "claim"]

    def claim(c: dict) -> dict:
        kids = []
        if c.get("counted"):
            base = c.get("denominator")
            kids.append({"label": f"Counted: {c['counted']} papers" if base is not None else f"Trend: {c['counted']}",
                         "type": "fact"})
            if base is not None and base < 20:
                kids.append({"label": f"Limited evidence base ({base} papers)", "type": "fact"})
        if c.get("matched"):
            kids.append({"label": "Papers counted", "type": "group", "count": str(len(c["matched"])),
                         "children": papers(c["matched"])})
        if c.get("not_counted"):
            kids.append({"label": "Papers not counted", "type": "group", "count": str(len(c["not_counted"])),
                         "children": papers(c["not_counted"])})
        return {"ref": c["id"], "label": f"{c['label']}: {_short(c['text'], 80)}", "type": "claim", "children": kids}

    evidence = []
    sup = [c for c in claims if c["status"] == "supported"]
    uns = [c for c in claims if c["status"] != "supported"]
    if sup:
        evidence.append({"label": "Supported claims", "type": "group", "count": str(len(sup)),
                         "children": [claim(c) for c in sup]})
    drivers = [n for n in nodes.values() if n["type"] == "driver"]
    if drivers:
        evidence.append({"label": "Where studies disagree", "type": "group", "count": str(len(drivers)), "children": [
            {"ref": d["id"], "label": d["label"], "type": "driver", "children": [
                {"label": f"{side.capitalize()} association", "type": "group",
                 "count": str(sum(1 for l in out_links[d["id"]] if l["type"] == side)),
                 "children": papers([l["target"] for l in out_links[d["id"]] if l["type"] == side])}
                for side in ("positive", "negative")]
                + ([{"label": "No effect reported", "type": "group", "count": str(len(d["no_effect"])),
                     "children": papers([f"paper:{p}" for p in d["no_effect"]])}] if d.get("no_effect") else [])
                + ([{"label": "What differs between the sides: " + "; ".join(d["separating"]), "type": "fact"}]
                   if d.get("separating") else [])}
            for d in drivers]})
    if uns:
        evidence.append({"label": "Claims not supported", "type": "group", "count": str(len(uns)),
                         "children": [claim(c) for c in uns]})
    if evidence:
        sections.append({"label": "Evidence", "type": "section", "section": "evidence", "children": evidence})

    # research gaps: evidence, the trail behind the gap, and the directions that address it
    gaps = [n for n in nodes.values() if n["type"] in ("gap", "combination")]
    designs = [n for n in nodes.values() if n["type"] == "design"]

    def gap(g: dict) -> dict:
        kids = []
        if g.get("counted"):
            kids.append({"label": f"Evidence: {g['counted']} papers", "type": "fact"})
        if g.get("why"):
            kids.append({"label": f"Why it matters: {_short(g['why'], 110)}", "type": "fact"})
        if g.get("trail"):
            kids.append({"label": "How the papers are distributed", "type": "group", "children": [
                {"label": t["value"] + (" (the gap)" if t["is_gap"] else ""), "type": "trail",
                 "count": frac(len(t["papers"])), "gap": t["is_gap"],
                 "children": papers([f"paper:{p}" for p in t["papers"]])} for t in g["trail"]]})
        for l in out_links[g["id"]]:
            if l["type"] == "rests_on" and l["target"] in nodes:
                kids.append(claim(nodes[l["target"]]))
        about = [nodes[l["target"]] for l in out_links[g["id"]] if l["type"] == "about" and l["target"] in nodes]
        for c in about:
            if c.get("papers") and not g.get("trail"):
                kids.append({**concept(c["id"]), "label": f"Papers with {c['label']}"})
        directions = [d for d in designs if any(l["type"] == "addresses" and l["target"] == g["id"]
                                                for l in out_links[d["id"]])]
        if directions:
            kids.append({"label": "Research directions", "type": "group", "children": [
                {"ref": d["id"], "label": d["label"], "type": "design",
                 "children": [{"label": f"Validation: {d['validation']}", "type": "fact"}] if d.get("validation") else []}
                for d in directions]})
        return {"ref": g["id"], "label": g["label"], "type": g["type"], "children": kids}

    gap_kids = [gap(g) for g in gaps]
    loose = [d for d in designs if not any(l["type"] == "addresses" and l["target"] in {g["id"] for g in gaps}
                                           for l in out_links[d["id"]])]
    reported = [n for n in nodes.values() if n["type"] == "direction"]
    if reported:
        gap_kids.append({"label": "Suggested research directions", "type": "group",
                         "children": [{"ref": d["id"], "label": d["label"], "type": "direction"} for d in reported]})
    if loose:
        gap_kids.append({"label": "Other research directions", "type": "group",
                         "children": [{"ref": d["id"], "label": d["label"], "type": "design"} for d in loose]})
    if gap_kids:
        sections.append({"label": "Research gaps", "type": "section", "section": "gaps", "count": str(len(gaps)),
                         "children": gap_kids})
    return {"ref": "question", "label": nodes["question"]["label"], "subtitle": nodes["question"]["text"],
            "type": "question", "count": f"{N} papers", "children": sections}
