"""A scripted stand-in for the LLM so the multi-agent plumbing can be tested offline and for free.

It plays each agent's role with a fixed script, but reacts to real tool results (e.g. it shortlists
the paper ids that the real search returned), so the tests exercise real SQL, real tools and the real
agent loop — only the "thinking" is canned."""
from __future__ import annotations

import itertools
import json
import re

from research_agent.llm.base import LLMResponse, ToolCall, Usage

_ids = itertools.count(1)
PLACES = ["Kenya", "Uganda", "Brazil", "India", "Malawi", "Tanzania", "USA"]
DATASETS = ["CheXpert", "ChestX-ray14", "MIMIC-IV", "BreakHis", "TCGA"]
METHODS = ["random forest", "LSTM", "convolutional neural network", "vision transformer", "transformer",
           "foundation model", "large language model"]


def _call(name, **inp):
    return ToolCall(f"tc_{next(_ids)}", name, inp)


def _quote(text: str, needle: str) -> str:
    """Up to ~12 words of the text around `needle`, copied exactly, like a careful extractor would."""
    words = text.split()
    for i, w in enumerate(words):
        if needle.lower().split()[0] in w.lower():
            return " ".join(words[max(0, i - 5): i + 7])
    return ""


def fake_extraction(text: str) -> dict:
    data = _fake_fields(text)
    keys = {"methods": data["methods"], "datasets": data["datasets"], "geography": data["geography"],
            "data_modalities": [{"surveillance counts": "surveillance", "chest X-ray": "x-ray",
                                 "EHR structured": "health", "histopathology": "histopathology",
                                 "climate/environmental": "rainfall"}[m] for m in data["data_modalities"]],
            "validation_level": ["validation"] if data["validation_level"] != "not_stated" else []}
    data["evidence"] = {f: [q for q in (_quote(text, v) for v in vals[:2]) if q] for f, vals in keys.items()}
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    data["reported_results"] = [
        {"metric": "AUROC", "value": m.group(1), "model": "proposed model", "is_baseline": False,
         "split": "test_or_holdout", "quote": next(s for s in sentences if m.group(1) in s)}
        for m in re.finditer(r"AUROC(?: of)? (\d\.\d+)", text)]
    data["reported_associations"] = [
        {"driver": "rainfall", "outcome": "cases", "direction": "positive",
         "quote": next(s for s in sentences if "driven by rainfall" in s)}] if "driven by rainfall" in text else []
    return data


def _fake_fields(text: str) -> dict:
    low = text.lower()
    return {
        "problem": text.split("\n")[0][7:120],
        "task_types": ["forecasting"] if "forecast" in low else ["classification"],
        "health_domains": [d for d in ("malaria", "pneumonia", "sepsis", "breast cancer") if d in low],
        "data_modalities": [m for k, m in (("surveillance", "surveillance counts"), ("x-ray", "chest X-ray"),
                            ("health records", "EHR structured"), ("histopathology", "histopathology"),
                            ("rainfall", "climate/environmental")) if k in low],
        "datasets": [d for d in DATASETS if d.lower() in low],
        "geography": [p for p in PLACES if p.lower() in low],
        "methods": [m for m in METHODS if m in low],
        "evaluation_metrics": [m for m in ("AUROC", "RMSE", "accuracy") if m.lower() in low],
        "sample_size": "42 sites" if "42 sites" in low else "",
        "validation_level": "external" if "external validation" in low else
                            ("internal" if "internal only" in low else "not_stated"),
        "code_or_data_available": "not_stated",
        "key_findings": "Improved over baseline.",
        "limitations": ["single-centre data"] if "single-centre" in low else
                       (["retrospective data"] if "retrospective" in low else []),
    }


def fake_protocol_extraction(text: str) -> dict:
    """Plays the protocol extractor on the fixture malaria abstracts, quoting its evidence."""
    low = text.lower()
    out = {"q_forecast_horizon": "not_stated", "q_validation_split": "not_stated",
           "q_probabilistic": "not_stated", "q_uses_satellite": "not_stated", "evidence": {}}
    if "monthly" in low:
        out["q_forecast_horizon"] = "one_month"
        out["evidence"]["q_forecast_horizon"] = [_quote(text, "monthly")]
    if "internal only" in low:
        out["q_validation_split"] = "random_split"
        out["evidence"]["q_validation_split"] = [_quote(text, "internal")]
    if "satellite" in low:
        out["q_uses_satellite"] = "yes"
        out["evidence"]["q_uses_satellite"] = [_quote(text, "satellite")]
    if "lstm" in low:                            # claims a probabilistic forecast with an invented quote
        out["q_probabilistic"] = "yes"
        out["evidence"]["q_probabilistic"] = ["we issue calibrated probabilistic forecasts"]
    return out


def fake_reading(text: str, tools) -> dict:
    """Plays the single reader: general fields and/or question-specific fields, as the tool asks for."""
    props = tools[0]["input_schema"]["properties"]
    if "records" in props:
        props = props["records"]["items"]["properties"]
    out, evidence = {}, {}
    if "methods" in props:
        base = fake_extraction(text)
        evidence.update(base.pop("evidence"))
        out.update(base)
    if any(k.startswith("q_") for k in props):
        proto = fake_protocol_extraction(text)
        evidence.update(proto.pop("evidence"))
        out.update({k: v for k, v in proto.items() if k in props})
    out["evidence"] = evidence
    return out


def fake_recheck(text: str, tools) -> dict:
    """Plays the re-checker: 'yes' with the exact sentence when the paper's own work uses the term,
    'no' when the term only appears in a background sentence."""
    import re as _re

    terms = tools[0]["input_schema"]["properties"]["which"]["enum"]
    for sentence in _re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        low = sentence.lower()
        hit = next((t for t in terms if t.lower() in low), None)
        if hit:
            own = any(w in low for w in ("we ", "using ", "driven by", "our "))
            return {"uses": "yes" if own else "no", "which": hit, "quote": sentence.strip() if own else ""}
    return {"uses": "unclear"}


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self):
        self.usage = Usage()

    # ------------------------------------------------------------------
    def chat(self, system, messages, tools=None, force_tool=None, max_tokens=2048, temperature=0.0):
        self.usage.add({"input_tokens": 100, "output_tokens": 50})
        if force_tool == "record_reading":
            return self._resp([_call("record_reading", **fake_reading(messages[0]["content"][0]["text"], tools))])
        if force_tool == "record_readings":
            text = messages[0]["content"][0]["text"]
            blocks = re.split(r"^=== Paper (\S+) ===\n", text, flags=re.M)[1:]
            records = [{"paper_id": pid, **fake_reading(body, tools)} for pid, body in zip(blocks[::2], blocks[1::2])]
            return self._resp([_call("record_readings", records=records)])
        if force_tool == "record_check":
            return self._resp([_call("record_check", **fake_recheck(messages[0]["content"][0]["text"], tools))])
        if not tools and "Draft writer" in system:
            text = self._draft_section(messages[0]["content"])
            return LLMResponse(text, [], [{"type": "text", "text": text}], "stop", {})
        agent = self._agent(system)
        step = sum(1 for m in messages if m["role"] == "assistant")
        last = self._last_results(messages)
        if force_tool == "finish":
            return self._resp([_call("finish", **self._finish(agent, messages))])
        calls = getattr(self, f"_{agent}")(step, last, messages)
        return self._resp(calls)

    @staticmethod
    def _resp(calls):
        content = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        return LLMResponse("", calls, content, "tool_use", {})

    @staticmethod
    def _agent(system: str) -> str:
        for key, name in (("Protocol agent", "protocol"), ("Gap Reasoning agent", "gap_reasoning"),
                          ("Research Design agent", "design"),
                          ("Research Manager", "orchestrator"), ("Discovery agent", "discovery"),
                          ("Literature Analyst", "literature"), ("Methods agent", "methods"),
                          ("Trend agent", "trends"), ("Research Gap agent", "gaps"),
                          ("Evidence agent", "evidence"), ("Synthesis agent", "synthesis"),
                          ("Number Check agent", "number_check"), ("Follow-up agent", "followup"),
                          ("Draft Planner agent", "draft_plan"), ("Research Supervisor agent", "research_supervisor"),
                          ("Critic agent", "critic"), ("Researcher agent", "researcher")):
            if key in system:
                return name
        raise AssertionError("unknown agent prompt")

    @staticmethod
    def _last_results(messages) -> list[dict]:
        if not messages or messages[-1]["role"] != "user":
            return []
        out = []
        for b in messages[-1]["content"]:
            if b.get("type") == "tool_result":
                try:
                    out.append(json.loads(b["content"]))
                except json.JSONDecodeError:
                    out.append({"raw": b["content"]})
        return out

    def _finish(self, agent, messages):
        return {"summary": f"{agent} forced finish"}

    # ------------------------------------------------------------------ scripts
    def _orchestrator(self, step, last, messages):
        plan = [
            [_call("call_discovery", task="Find papers on malaria forecasting with climate data.")],
            [_call("call_literature", task="Extract all; full text for the most central.")],
            [_call("call_methods", task="Characterise methods and data."),
             _call("call_trends", task="Trends of methods over time.")],
            [_call("call_gaps", task="Find evidence-grounded gaps.")],
            [_call("call_evidence", task="Verify all claims.")],
            [_call("call_synthesis", task="Write the report.")],
            [_call("get_status")],
        ]
        if step < len(plan):
            return plan[step]
        return [_call("finish", summary="done")]

    def _discovery(self, step, last, messages):
        if step == 0:
            return [_call("hybrid_search", query="malaria incidence forecasting with climate data",
                          keywords="malaria", limit=40),
                    _call("corpus_count", keywords="malaria")]
        if step == 1:
            ids = [r["paper_id"] for r in last[0]["results"] if "malaria" in r["title"].lower()]
            return [_call("add_to_shortlist", paper_ids=ids + ["0000.00000"], reason="malaria forecasting")]
        if step == 2:
            return [_call("view_shortlist")]
        return [_call("finish", shortlist_size=last[0]["size"] if last and "size" in last[0] else 0,
                      coverage_notes="Malaria forecasting well covered in fixture corpus.")]

    def _literature(self, step, last, messages):
        if step == 0:
            return [_call("extract_papers", depth="abstract")]
        if step == 1:
            return [_call("extract_papers", depth="fulltext")]
        if step == 2:
            return [_call("extraction_coverage")]
        return [_call("finish", extracted=last[0]["extracted"], fulltext_read=last[0]["by_source"]["fulltext"],
                      quality_notes="ok", rarely_stated_fields=["code_or_data_available"])]

    def _methods(self, step, last, messages):
        if step == 0:
            return [_call("value_counts", field="methods"), _call("cross_tab", field_a="methods",
                                                                   field_b="geography")]
        if step == 1:
            return [_call("propose_claim", text="Most studies use data from East Africa.",
                          claim_type="prevalence",
                          predicate={"field": "geography", "any_of": ["Kenya", "Uganda", "Malawi", "Tanzania"],
                                     "over": "all", "min_share": 0.3}),
                    _call("propose_claim", text="No study uses MIMIC data.", claim_type="prevalence",
                          predicate={"field": "datasets", "any_of": ["MIMIC"], "max_count": 0})]
        return [_call("finish", method_families=[{"family": "sequence models", "members": ["LSTM"]}],
                      observations=["LSTM common"], claim_ids=[])]

    def _trends(self, step, last, messages):
        if step == 0:
            return [_call("topic_trend", keywords='"foundation model"'),
                    _call("compare_topics", topics=["LSTM", '"foundation model"'])]
        if step == 1:
            return [_call("propose_claim", text="Foundation-model papers have increased sharply.",
                          claim_type="trend",
                          predicate={"keywords": '"foundation model"', "early": [2016, 2018],
                                     "late": [2023, 2025], "direction": "increase"}),
                    _call("propose_claim", text="Malaria research is declining.", claim_type="trend",
                          predicate={"keywords": "malaria", "early": [2016, 2018], "late": [2023, 2025],
                                     "direction": "decrease", "min_ratio": 3})]
        return [_call("finish", trends=[{"topic": "foundation models", "direction": "up"}], observations=["up"])]

    def _gaps(self, step, last, messages):
        if step == 0:
            return [_call("value_counts", field="validation_level"), _call("corpus_count", keywords="malaria")]
        if step == 1:
            return [_call("propose_claim", text="Few malaria forecasting studies use external validation.",
                          claim_type="prevalence",
                          predicate={"field": "validation_level", "any_of": ["external", "prospective",
                                     "clinical_trial"], "max_share": 0.2})]
        return [_call("finish", gaps=[{"gap": "external validation is rare", "claim_ids": [5],
                                       "confidence": "medium"}], research_directions=["validate externally"])]

    def _evidence(self, step, last, messages):
        if step == 0:
            return [_call("verify_claims")]
        if step == 1:
            return [_call("list_claims", status="supported")]
        return [_call("finish", supported=len(last[0]["claims"]), unsupported=1, rejected=0)]

    def _synthesis(self, step, last, messages):
        if step == 0:
            return [_call("get_brief")]
        brief = last[0]
        first = brief["papers"][0]["paper_id"]
        supported = [c["claim"] for c in brief["claims"] if c["status"] == "supported"]
        unsupported = [c["claim"] for c in brief["claims"] if c["status"] != "supported"]
        n = len(brief["papers"])
        body = (f"## Summary\n- Malaria forecasting relies on climate covariates [arXiv:{first}] "
                f"[{supported[0] if supported else 'C0'}].\n- A fake citation [arXiv:9999.99999] and an "
                f"unverified one [{unsupported[0] if unsupported else 'C0'}].\n"
                f"- Satellite-derived inputs appear in {n - 1} of {n} papers.\n"
                f"- Point metrics are reported by 3 of {n} papers (a hand-made sum).\n")
        return [_call("finish", report_markdown=body)]

    def _followup(self, step, last, messages):
        """Looks up the claim asked about, counts something new, and answers (with one invented id and one
        untraced number, which the audit must catch)."""
        first = messages[0]["content"][0]["text"]
        if step == 0:
            ids = re.findall(r"\bC\d+\b", first.split("New question from the researcher:")[-1])
            return [_call("get_item", item_id=ids[0]) if ids else _call("run_overview")]
        if step == 1:
            self._looked_up = last[0]
            return [_call("test_claim", claim_type="prevalence",
                          predicate={"field": "data_modalities", "any_of": ["surveillance"], "min_count": 1})]
        if step == 2:
            self._n, self._d = last[0]["n_matching"], last[0]["denominator"]
            return [_call("add_claim", text=f"Surveillance counts appear in {self._n} of {self._d} papers.",
                          claim_type="prevalence",
                          predicate={"field": "data_modalities", "any_of": ["surveillance"], "min_count": 1})]
        new = last[0]["claim_id"]
        remembered = "Conversation so far:" in first
        return [_call("finish", answer=(
            f"{'As discussed before, ' if remembered else ''}the claim you asked about is "
            f"{self._looked_up.get('status', 'an overview')}. Surveillance counts appear in {self._n} of {self._d} papers "
            f"[C{new}]. One more study [arXiv:0000.00000] reports 7 of 9 sites."))]

    def _draft_plan(self, step, last, messages):
        """Orients, counts one thing for the argument, and returns an outline citing it."""
        if step == 0:
            return [_call("run_overview")]
        if step == 1:
            return [_call("test_claim", claim_type="prevalence",
                          predicate={"field": "validation_level", "any_of": ["external"], "min_count": 1})]
        if step == 2:
            self._n, self._d = last[0]["n_matching"], last[0]["denominator"]
            return [_call("add_claim", text=f"External validation is reported by {self._n} of {self._d} papers.",
                          claim_type="prevalence",
                          predicate={"field": "validation_level", "any_of": ["external"], "min_count": 1})]
        cid = f"C{last[0]['claim_id']}"
        return [_call("finish", title="Validating malaria forecasts where they are used",
                      argument="Few studies validate externally, so reported skill may not transfer.",
                      sections=[{"key": "gap", "points": ["external validation is rare"], "claims": [cid]}],
                      caveats=["small evidence base"])]

    # ------------------------------------------------------------------ open-ended researcher
    def _researcher(self, step, last, messages):
        """Seeds an agenda (one question out of bounds), or tests a pattern, re-tests it (the notebook must
        return the earlier result), proposes it as a finding and adds a sub-question."""
        first = messages[0]["content"][0]["text"]
        if "The agenda is empty" in first:
            plan = [[_call("research_overview")],
                    [_call("add_question", question="Which forecasting methods are used in studies from Malawi?",
                           why="methods by setting", priority=80),
                     _call("add_question", question="How is external validation reported across forecasting studies?",
                           why="validation practice", priority=60),
                     _call("add_question", question="How should clinical treatment doses be adjusted for children?",
                           why="off the charter", priority=40)],
                    [_call("finish", progress="some", summary="agenda seeded")]]
            return plan[min(step, len(plan) - 1)]
        spec = {"kind": "difference", "outcome": [{"field": "methods", "any_of": ["random forest"]}],
                "group_a": [{"field": "geography", "any_of": ["Malawi"]}]}
        if step == 0:
            return [_call("research_overview")]
        if step == 1:
            return [_call("test_pattern", spec=spec, description="random forest in Malawi vs elsewhere")]
        if step == 2:
            self._test_id = last[0].get("test_id")
            return [_call("test_pattern", spec=spec)]
        if step == 3:
            self._repeat = last[0]
            return [_call("propose_finding", test_id=self._test_id,
                          statement="Random forest studies cluster in Malawi more than in other settings.")]
        if step == 4:
            self._verdict = last[0]
            return [_call("add_question", question="Do the Malawi random forest studies share their data sources?",
                          why="follow the cluster")]
        return [_call("finish", progress="some", summary="tested one pattern", next_step="check data sources")]

    def _critic(self, step, last, messages):
        return [_call("finish", alternatives=[
            {"explanation": "The corpus could explain it.", "kind": "stratify", "field": "corpus", "any_of": ["arxiv"]},
            {"explanation": "A looser definition of the method.", "kind": "redefine", "side": "outcome",
             "any_of": ["forest"]}])]

    def _research_supervisor(self, step, last, messages):
        if step == 0:
            return [_call("research_overview")]
        if step == 1:
            open_items = [a for a in last[0]["agenda"] if a["status"] == "open"]
            return [_call("set_priority", agenda_id=open_items[-1]["id"], priority=95)] if open_items else \
                [_call("finish", assessment="nothing open", keep_going=True)]
        return [_call("finish", assessment="on charter; keep going", keep_going=True)]

    def _draft_section(self, blocks) -> str:
        """Writes a section from the pack: one real claim and paper, plus an invented paper id, an untraced
        number and an em dash, which the checks must catch."""
        pack = json.loads(blocks[0]["text"].split("EVIDENCE PACK (JSON):\n", 1)[1])
        supported = [c["claim"] for c in pack["claims"] if c["status"] == "supported"]
        paper = pack["papers"][0]["id"]
        cite = f"[arXiv:{paper}]" if not paper.startswith("PMC") else f"[{paper}]"
        task = next(b["text"] for b in reversed(blocks) if b["text"].startswith("Write the section"))
        heading = re.search(r'section "([^"]+)"', task).group(1)
        second = pack["papers"][1]["id"] if len(pack["papers"]) > 1 else paper
        return (f"## {heading}\n\nMalaria forecasting draws on climate data {cite} — a common design. "
                f"The key count is cited [{supported[-1]}]. A further study [arXiv:0000.00000] reports 7 of 9 "
                f"sites, and another (arXiv:{second}) reported an error of 9.99 on its holdout. "
                f"We will use [to be confirmed: number of districts] districts.")

    def _number_check(self, step, last, messages):
        """Measures the first untraced number with a claim, drops the second, keeps the rest."""
        items = json.loads(messages[0]["content"][0]["text"].split("Resolve these numbers:\n", 1)[1].split("\n\n")[0])
        if step == 0:
            return [_call("propose_claim", text="Satellite-derived inputs are used by some shortlisted papers.",
                          claim_type="prevalence",
                          predicate={"field": "data_modalities", "any_of": ["satellite"], "min_count": 1})]
        fixes = [{"item": items[0]["item"], "action": "measure", "claim_id": last[0].get("claim_id")}]
        if len(items) > 1:
            fixes.append({"item": items[1]["item"], "action": "drop",
                          "sentence": "Point metrics are reported by some papers (count not measured)."})
        return [_call("finish", fixes=fixes)]

    # ------------------------------------------------------------------ opportunity map agents
    def _protocol(self, step, last, messages):
        if step == 0:
            return [_call("corpus_count", keywords="malaria forecast")]
        return [_call("finish", setting="Malaria forecasting, any country", topic_query="malaria forecast",
                      inclusion=["forecasts or predicts malaria incidence"], exclusion=["no forecasting"],
                      fields=[
                          {"name": "forecast_horizon", "type": "enum",
                           "values": ["one_month", "one_to_three_months", "three_to_six_months", "over_six_months"],
                           "definition": "How far ahead forecasts are made.", "desirable": ["three_to_six_months"],
                           "search": {"three_to_six_months": '"6 months" OR "six months"'}},
                          {"name": "validation_split", "type": "enum",
                           "values": ["random_split", "temporal_holdout", "spatial_holdout",
                                      "spatiotemporal_holdout", "none"], "role": "evaluation",
                           "definition": "How evaluation data were separated.",
                           "desirable": ["spatial_holdout", "spatiotemporal_holdout"],
                           "groups": [{"label": "tests on unseen places",
                                       "values": ["spatial_holdout", "spatiotemporal_holdout"]},
                                      {"label": "lonely group", "values": ["temporal_holdout"]}],
                           "search": {"spatial_holdout": '"spatial cross-validation" OR "held-out districts"',
                                      "spatiotemporal_holdout": '"space-time cross-validation"'}},
                          {"name": "probabilistic", "type": "enum", "values": ["yes", "no"],
                           "definition": "Forecasts give probabilities or intervals.", "desirable": ["yes"]},
                          {"name": "uses_satellite", "type": "enum", "values": ["yes", "no"],
                           "definition": "Uses satellite-derived inputs."},
                          {"name": "methods", "type": "list", "definition": "duplicate of a base field"},
                          {"name": "bad", "type": "enum", "values": ["only_one"], "definition": "x"},
                      ])]

    def _gap_reasoning(self, step, last, messages):
        if step == 0:
            return [_call("get_map")]
        if step == 1:
            self._map = last[0]
            return [_call("test_claim", claim_type="prevalence",
                          predicate={"field": "validation_level", "any_of": ["internal"], "min_share": 0.5})]
        gaps = [g["id"] for g in self._map["gaps"]]
        if step == 2:
            n, d = last[0]["n_matching"], last[0]["denominator"]
            return [_call("test_hypothesis", gap_id=gaps[0], role="cause", claim_type="prevalence",
                          text=f"Internal-only validation appears in {n} of {d} papers.",
                          predicate={"field": "validation_level", "any_of": ["internal"], "min_share": 0.5}),
                    _call("test_hypothesis", gap_id=gaps[0], role="artifact", claim_type="prevalence",
                          text="Among recent abstract-only papers, internal validation is reported.",
                          predicate={"field": "validation_level", "any_of": ["internal"], "min_share": 0.5,
                                     "where": [{"field": "read", "any_of": ["abstract"]},
                                               {"field": "year", "min": 2099}]}),
                    _call("test_hypothesis", gap_id=gaps[0], role="cause", claim_type="prevalence",
                          text=f"External validation appears in {d - 1} of {d} papers.",
                          predicate={"field": "validation_level", "any_of": ["external"], "min_share": 0.5}),
                    _call("test_hypothesis", gap_id=gaps[-1], role="cause", claim_type="prevalence",
                          text=f"Internal-only validation appears in {n} of {d} papers.",
                          predicate={"field": "validation_level", "any_of": ["internal"], "min_share": 0.5})]
        if step == 3:
            return [_call("find_passages", terms=["internal"])]
        ids = [p["paper_id"] for p in last[0]["papers"]]
        return [_call("finish", gaps=[{
            "gap_id": gaps[0], "explanation": "Papers validate on the data they were fitted on (H1).",
            "hypothesis_ids": ["H1", "H99"], "artifact_risk": "low", "artifact_reason": "stated in full text",
            "linked_gaps": [{"gap_id": gaps[-1], "relation": "no shared data, no external test", "hypothesis_id": "H1"},
                            {"gap_id": gaps[-1], "relation": "untested link", "hypothesis_id": "H42"}],
            "near_misses": [{"paper_id": ids[0], "what_it_does": "forecasts monthly", "what_is_missing": "holdout"},
                            {"paper_id": "0000.00000", "what_it_does": "invented", "what_is_missing": "all"}],
            "would_change_if": "papers with district holdouts appear"}],
            cross_cutting=["Validation practice follows data access (H1)."])]

    def _design(self, step, last, messages):
        if step == 0:
            return [_call("get_map")]
        m = last[0]
        gap = m["gaps"][0]["id"]
        pid = m["established"][0]["example_paper_ids"][0]
        return [_call("finish", designs=[
            {"title": "District-holdout 3-6 month forecasts", "target": "monthly cases per district",
             "predictors": ["rainfall", "temperature"], "horizon": "3-6 months",
             "validation_strategy": "leave-one-district-out", "baseline": "seasonal naive",
             "addresses": [gap, "G99", "H1"], "rests_on": ["H77"], "builds_on": [pid],
             "supporting": [{"paper_id": pid, "why": "uses the same surveillance data"}],
             "challenging": [{"paper_id": "9999.99999", "why": "invented"}]},
            {"title": "Unanchored idea", "target": "x", "validation_strategy": "y", "addresses": ["G99"]}])]
