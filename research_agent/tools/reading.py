"""One careful read per paper.

Every paper is read once for everything the run needs: the general extraction fields and this run's
question-specific (protocol) fields come out of the same call, from the same text. Papers read in full get
a call of their own; abstract-only papers are read in batches (their instructions and field definitions are
sent once for the batch instead of once per paper).

Nothing about checking changes: every value still needs a quote copied from THAT paper's own text, verified
by code, or it is dropped.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from research_agent.config import settings

READ_SYSTEM = """You extract structured facts from health research papers for a literature database.

Read the whole text you are given, section by section and line by line, before you record anything. Do not
stop at the abstract or the first mention: methods, data and validation details are often only in later
sections, and a value you miss is lost.

Rules:
- Record only what the text states. If something is not stated, use an empty list, '' or 'not_stated'.
  Never guess: an empty value is correct and useful.
- Keep list items short (1-5 words), canonical names (e.g. 'vision transformer', not 'our ViT-based model').
- geography = where the data or samples were collected, not where authors work.
- The form covers every kind of health research: modelling and AI papers, trials, cohorts, laboratory and
  animal studies, pharmacology and reviews. Fill the fields that apply to THIS paper and leave the rest
  empty; an AI field left empty on a drug trial, or organisms left empty on a survey, is correct.
- interventions = what was given or studied as an exposure (drug, compound, therapy, programme);
  mechanisms = how something works or resists (mechanism of action, pathway, resistance mechanism), only as
  the paper reports or studies it, not textbook background; targets = named receptors, enzymes, hormones
  or genes central to the study.
- evidence: for each field listed under evidence that has a value, copy 1-2 short passages (up to 25 words)
  WORD FOR WORD from this paper's text. Code checks every passage against the text; a value whose passages
  are not found is discarded. Do not paraphrase.
- reported_results: the paper's own main numbers, one entry per number: model performance (RMSE, AUC,
  accuracy...), effect estimates (odds ratio, hazard ratio, relative risk, mean difference), laboratory and
  pharmacological measurements (MIC, IC50, EC50, Cmax, plasma AUC, half-life) and prevalences. `model` is
  the model, drug, arm or group the number belongs to; is_baseline marks a comparator (baseline model,
  placebo, control arm). The quote must contain the number exactly as written. Look in results sections
  and tables.
- reported_associations: effects the paper itself reports between a driver (a risk factor, exposure,
  drug, dose or intervention) and a health outcome, with direction and lag if stated. Background statements
  about other studies do not count.{protocol_rules}
- {source_note}"""

PROTOCOL_RULES = """
- The question-specific fields (names starting with q_) serve this research question: {question}
  Follow each field's definition exactly and pick the single best category for enum fields.
  Field definitions:
{definitions}"""

BATCH_NOTE = ("You are given several papers, each introduced by '=== Paper <id> ==='. Record one entry per "
              "paper, with its paper_id. Each paper's values and evidence must come from that paper's own "
              "text only; never carry a fact from one paper to another.")


def _schema(protocol: dict | None, include_base: bool) -> tuple[dict, list[str]]:
    """Field properties and required names for a read covering base and/or protocol fields."""
    from research_agent.opportunity.protocol import protocol_tool
    from research_agent.tools.extraction import ALL_FIELDS, EVIDENCE_FIELDS, EXTRACTION_TOOL, _list

    props, required, ev_fields = {}, [], []
    if include_base:
        base = EXTRACTION_TOOL["input_schema"]["properties"]
        props.update({k: v for k, v in base.items() if k != "evidence"})
        from research_agent.tools.results import STRUCT_FIELDS

        required += ALL_FIELDS + STRUCT_FIELDS
        ev_fields += EVIDENCE_FIELDS
    if protocol and protocol.get("fields"):
        pt = protocol_tool(protocol)["input_schema"]["properties"]
        props.update({k: v for k, v in pt.items() if k != "evidence"})
        names = [f["name"] for f in protocol["fields"]]
        required += names
        ev_fields += names
    props["evidence"] = {"type": "object", "description": "Passages copied word for word, per field.",
                         "properties": {f: _list for f in ev_fields}}
    return props, required + ["evidence"]


def _system(ctx, protocol: dict | None, source_note: str) -> str:
    rules = ""
    if protocol and protocol.get("fields"):
        rules = PROTOCOL_RULES.format(question=ctx.question, definitions="\n".join(
            f"  - {f['name']}: {f['definition']}" for f in protocol["fields"]))
    return READ_SYSTEM.format(protocol_rules=rules, source_note=source_note)


def paper_text(paper: dict) -> str:
    text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    if paper.get("fulltext"):
        text += f"\n\nFull-text sections:\n{paper['fulltext']}"
    return text


def _verify(args: dict, text: str, protocol: dict | None, include_base: bool,
            version: str | None = None) -> tuple[dict | None, dict | None]:
    from research_agent.opportunity.protocol import _normalise_protocol
    from research_agent.tools.extraction import check_evidence, normalise

    evidence = args.get("evidence") if isinstance(args.get("evidence"), dict) else {}
    base = check_evidence(normalise(args, version), evidence, text) if include_base else None
    if base is not None:
        from research_agent.tools.results import verify_structured

        found, dropped = verify_structured(args, text)   # results and associations carry their own quotes
        base.update(found)
        if dropped:
            base["_unverified"] = {**(base.get("_unverified") or {}), **dropped}
    proto = None
    if protocol and protocol.get("fields"):
        names = [f["name"] for f in protocol["fields"]]
        lists = [f["name"] for f in protocol["fields"] if f["type"] == "list"]
        proto = check_evidence(_normalise_protocol(args, protocol), evidence, text, fields=names, list_fields=lists)
    return base, proto


def _read_one(llm, ctx, paper: dict, protocol, include_base: bool) -> dict:
    source = "fulltext" if paper.get("fulltext") else "abstract"
    note = ("You are reading the title, abstract and full-text sections of one paper." if paper.get("fulltext")
            else "You are reading only the title and abstract, so many fields may be not stated.")
    props, required = _schema(protocol, include_base)
    tool = {"name": "record_reading", "description": "Record the extraction for this paper.",
            "input_schema": {"type": "object", "properties": props, "required": required}}
    text = paper_text(paper)
    budget = settings.extraction_max_tokens + (1500 if protocol and include_base else 0)
    resp = llm.chat(_system(ctx, protocol, note), [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    tools=[tool], force_tool="record_reading", max_tokens=budget)
    if not resp.tool_calls:
        raise ValueError("model returned no extraction")
    args = resp.tool_calls[0].input or {}
    if "_raw_arguments" in args:
        raise ValueError("extraction was cut off by the output limit; raise EXTRACTION_MAX_TOKENS")
    base, proto = _verify(args, text, protocol, include_base, ctx.extraction_version)
    return {"source": source, "base": base, "protocol": proto}


def _read_batch(llm, ctx, papers: list[dict], protocol, include_base: bool) -> dict[str, dict]:
    """Abstract-only papers in one call. Returns only the papers whose record came back complete."""
    from research_agent.agents.base import repair_truncated_json

    props, required = _schema(protocol, include_base)
    ids = [p["paper_id"] for p in papers]
    record = {"type": "object", "properties": {"paper_id": {"type": "string", "enum": ids}, **props},
              "required": ["paper_id"] + required}
    tool = {"name": "record_readings", "description": "Record one extraction per paper.",
            "input_schema": {"type": "object", "properties": {"records": {"type": "array", "items": record}},
                             "required": ["records"]}}
    texts = {p["paper_id"]: paper_text(p) for p in papers}
    user = "\n\n".join(f"=== Paper {pid} ===\n{texts[pid]}" for pid in ids)
    note = "You are reading only titles and abstracts, so many fields may be not stated. " + BATCH_NOTE
    per_paper = settings.extraction_max_tokens + (800 if protocol and include_base else 0)
    resp = llm.chat(_system(ctx, protocol, note), [{"role": "user", "content": [{"type": "text", "text": user}]}],
                    tools=[tool], force_tool="record_readings",
                    max_tokens=min(settings.long_output_max_tokens, per_paper * len(papers)))
    args = (resp.tool_calls[0].input if resp.tool_calls else {}) or {}
    if "_raw_arguments" in args:   # cut off: keep the records that were finished, re-read the rest singly
        args = repair_truncated_json(args["_raw_arguments"] or "") or {}
    out = {}
    for rec in args.get("records") or []:
        pid = rec.get("paper_id") if isinstance(rec, dict) else None
        if pid in texts and pid not in out:
            base, proto = _verify(rec, texts[pid], protocol, include_base, ctx.extraction_version)
            out[pid] = {"source": "abstract", "base": base, "protocol": proto}
    return out


def read_papers(ctx, papers: list[dict], protocol: dict | None, include_base: bool = True,
                step: str = "extraction") -> tuple[dict[str, dict], list[dict], str]:
    """Read each paper once. `papers`: [{paper_id, title, abstract, fulltext or None}].
    Returns ({paper_id: {source, base, protocol}}, failures, model name)."""
    llm = ctx.llm_factory(step=step)
    setattr(llm, "_step", step)
    full = [p for p in papers if p.get("fulltext")]
    abstract = [p for p in papers if not p.get("fulltext")]
    size = max(1, settings.abstract_batch)
    batches = [abstract[i:i + size] for i in range(0, len(abstract), size)] if size > 1 else []
    singles = full + ([] if size > 1 else abstract)
    results: dict[str, dict] = {}
    failed: list[dict] = []

    with ThreadPoolExecutor(max_workers=settings.extraction_workers) as pool:
        futures = {pool.submit(_read_batch, llm, ctx, b, protocol, include_base): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                got = fut.result()
            except Exception:
                got = {}
            results.update(got)
            singles += [p for p in batch if p["paper_id"] not in got]   # missing from the batch: read alone
        futures = {pool.submit(_read_one, llm, ctx, p, protocol, include_base): p for p in singles}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                results[p["paper_id"]] = fut.result()
            except Exception as exc:   # one bad paper must not sink the run
                failed.append({"paper_id": p["paper_id"], "error": str(exc)[:200]})
    return results, failed, getattr(llm, "model", "")
