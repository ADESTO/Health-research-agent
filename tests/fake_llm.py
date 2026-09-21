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


def fake_extraction(text: str) -> dict:
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


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self):
        self.usage = Usage()

    # ------------------------------------------------------------------
    def chat(self, system, messages, tools=None, force_tool=None, max_tokens=2048, temperature=0.0):
        self.usage.add({"input_tokens": 100, "output_tokens": 50})
        if force_tool == "record_extraction":
            return self._resp([_call("record_extraction", **fake_extraction(messages[0]["content"][0]["text"]))])
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
        for key, name in (("Research Manager", "orchestrator"), ("Discovery agent", "discovery"),
                          ("Literature Analyst", "literature"), ("Methods agent", "methods"),
                          ("Trend agent", "trends"), ("Research Gap agent", "gaps"),
                          ("Evidence agent", "evidence"), ("Synthesis agent", "synthesis")):
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
        body = (f"## Summary\n- Malaria forecasting relies on climate covariates [arXiv:{first}] "
                f"[{supported[0]}].\n- A fake citation [arXiv:9999.99999] and an unverified one "
                f"[{unsupported[0]}].\n")
        return [_call("finish", report_markdown=body)]
