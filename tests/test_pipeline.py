from __future__ import annotations

import re

import duckdb
import pytest

from research_agent.config import settings
from research_agent.ingestion.fulltext import assess, clean_latex, select_for_reading
from research_agent.ingestion.health_filter import health_term_regex
from tests.fake_llm import FakeLLM


# ---------------------------------------------------------------- unit: health filter
@pytest.mark.parametrize("text,expected", [
    ("Deep learning on electronic health records", "electronic health records"),
    ("Sepsis prediction in the ICU", "sepsis"),
    ("COVID-19 spread modelling", "covid-19"),
    ("Hospitality industry demand forecasting", ""),
    ("Diseased trees in forests", ""),
])
def test_health_regex(text, expected):
    got = duckdb.execute("SELECT lower(regexp_extract(?, ?, 1))", [text, health_term_regex()]).fetchone()[0]
    assert got == expected


# ---------------------------------------------------------------- unit: LaTeX cleaning
def test_clean_latex_sections_and_comments():
    tex = (r"\documentclass{article}\begin{document}\section{Introduction} Hello world, this is the introduction % secret" "\n"
           r"\section{Methods} We used a \textbf{CNN}. \begin{equation} a=b \end{equation}"
           r"\begin{figure}\caption{Pipeline of the CNN}\end{figure}\section{Limitations} Small sample."
           r"\end{document}")
    clean, sections = clean_latex(tex)
    heads = [s["heading"] for s in sections]
    assert heads[:3] == ["Introduction", "Methods", "Limitations"]
    assert "secret" not in clean and "a=b" not in clean
    assert "CNN" in clean and "Pipeline of the CNN" in clean
    picked = select_for_reading(sections, budget=10_000)
    assert picked.index("Methods") < picked.index("Limitations")  # keeps paper order


def test_assess_flags_stubs_and_templates():
    assert assess("%auto-ignore", "", "Anything") == "stub"
    long_raw = "x" * 5000
    assert assess(long_raw, "## Intro\nInstructions for authors of AIP journals " * 50,
                  "Malaria forecasting in Kenya") == "template_suspect"


# ---------------------------------------------------------------- ingestion
def test_ingest_selects_health_subset(loaded_db):
    from research_agent.db import get_conn

    assert loaded_db["inserted"] > 50
    with get_conn() as pg:
        titles = [r["title"] for r in pg.execute("SELECT title FROM papers").fetchall()]
        reasons = {r["health_reason"].split(":")[0] for r in pg.execute("SELECT health_reason FROM papers")}
        n_stats = pg.execute("SELECT sum(n_papers) n FROM corpus_year_stats").fetchone()["n"]
        n_papers = pg.execute("SELECT count(*) n FROM papers").fetchone()["n"]
        n_emb = pg.execute("SELECT count(*) n FROM papers WHERE embedding IS NOT NULL").fetchone()["n"]
    assert not any(t.startswith(("Robust quadruped", "Next-to-leading", "Graph neural networks for traffic",
                                 "Efficient transformers")) for t in titles)
    assert reasons <= {"keyword", "category"}
    assert n_stats > n_papers  # denominators include non-health papers
    assert n_emb == n_papers


def test_ingest_is_resumable(loaded_db):
    from research_agent.db import connect
    from research_agent.ingestion.load import load_papers

    pg = connect()
    try:
        assert load_papers(pg, log=lambda *_: None) == 0
    finally:
        pg.close()


# ---------------------------------------------------------------- tools
@pytest.fixture
def ctx(loaded_db):
    from research_agent.runstate import RunContext

    c = RunContext.create("test question", llm_factory=lambda strong=False: FakeLLM())
    yield c
    c.close()


def test_hybrid_search_ranks_topic_first(ctx):
    from research_agent.tools.search import hybrid_search

    res = hybrid_search(ctx, "forecasting malaria incidence with rainfall", keywords="malaria", limit=10)
    assert res["n"] == 10
    assert all("malaria" in r["title"].lower() for r in res["results"][:5])
    res2 = hybrid_search(ctx, "sepsis", year_from=2024, year_to=2025)
    assert all(2024 <= r["year"] <= 2025 for r in res2["results"])


def test_shortlist_rejects_unknown_ids_and_caps(ctx):
    from research_agent.tools.search import add_to_shortlist, hybrid_search

    ids = [r["paper_id"] for r in hybrid_search(ctx, "malaria", limit=5)["results"]]
    out = add_to_shortlist(ctx, ids + ["nope.123"], reason="t")
    assert out["added"] == 5 and out["unknown_ids"] == ["nope.123"]


def test_trend_is_normalised_and_flags_partial_year(ctx):
    from research_agent.tools.trends import topic_trend

    t = topic_trend(ctx, '"foundation model"')
    last = t["series"][-1]
    assert last["year"] == 2026 and last["partial_year"] is True
    assert all(s["per_10k_arxiv"] is not None for s in t["series"])
    assert t["summary"]["ratio_late_over_early"] in ("inf",) or t["summary"]["ratio_late_over_early"] > 1


def test_claim_validation_rejects_untestable(ctx):
    from research_agent.tools.claims import propose_claim

    assert "error" in propose_claim(ctx, "Most use CNNs", "prevalence", {"field": "methods", "any_of": ["CNN"]})
    assert "error" in propose_claim(ctx, "x", "prevalence", {"field": "authors", "any_of": ["a"], "min_share": .1})


# ---------------------------------------------------------------- full multi-agent runs
@pytest.mark.parametrize("mode", ["orchestrated", "pipeline"])
def test_end_to_end_run(loaded_db, mode):
    from research_agent.agents.orchestrator import run_research
    from research_agent.db import get_conn

    events = []
    res = run_research("What ML methods are used for malaria forecasting and what is missing?",
                       mode=mode, llm_factory=lambda strong=False: FakeLLM(),
                       on_event=lambda a, k, p: events.append((a, k)))
    report = res["report"]
    assert report and "## Evidence table" in report and "## References" in report
    # hallucinated citation removed, unverified claim flagged
    assert "9999.99999" not in report and "citation removed" in report
    assert "not verified]" in report

    with get_conn() as pg:
        run = pg.execute("SELECT * FROM runs WHERE run_id=%s", (res["run_id"],)).fetchone()
        claims = {c["text"]: c for c in pg.execute("SELECT * FROM claims WHERE run_id=%s", (res["run_id"],))}
        n_ft = pg.execute("""SELECT count(*) n FROM run_papers rp JOIN extractions e USING (paper_id)
                             WHERE rp.run_id=%s AND e.source='fulltext'""", (res["run_id"],)).fetchone()["n"]
        stub = pg.execute("SELECT count(*) n FROM paper_fulltext WHERE status='stub'").fetchone()["n"]
        n_q = pg.execute("SELECT count(*) n FROM protocol_extractions WHERE run_id=%s", (res["run_id"],)).fetchone()["n"]
        n_ex = pg.execute("""SELECT count(*) n FROM run_papers rp JOIN extractions e USING (paper_id)
                             WHERE rp.run_id=%s""", (res["run_id"],)).fetchone()["n"]
    assert run["status"] == "done" and run["llm_calls"] > 10
    # ask runs get question-specific fields, extracted for every analysed paper after the literature step
    assert n_q == n_ex > 0 and ("protocol", "finish") in events
    assert "Numbers checked after writing" in report

    # Deterministic evidence verdicts
    assert claims["Most studies use data from East Africa."]["status"] == "supported"
    assert claims["No study uses MIMIC data."]["status"] == "supported"
    assert claims["Foundation-model papers have increased sharply."]["status"] == "supported"
    assert claims["Malaria research is declining."]["status"] == "unsupported"
    ext = claims["Few malaria forecasting studies use external validation."]
    assert ext["status"] == "supported" and ext["result"]["n_matching"] == 0

    assert 0 < n_ft <= settings.max_fulltext
    assert stub <= 1
    agents_seen = {a for a, _ in events}
    assert {"discovery", "literature", "methods", "trends", "gaps", "evidence", "synthesis"} <= agents_seen
    if mode == "orchestrated":
        assert "orchestrator" in agents_seen
    # every paper in the reference list is from the shortlist
    assert len(re.findall(r"^- \*\*arXiv:", report, flags=re.M)) == \
        len(set(re.findall(r"^- \*\*arXiv:([^*]+)\*\*", report, flags=re.M)))


# ---------------------------------------------------------------- provider seam
def test_groq_message_translation_roundtrip():
    from research_agent.llm.groq_client import _to_openai_messages

    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "searching"},
                                          {"type": "tool_use", "id": "t1", "name": "hybrid_search",
                                           "input": {"query": "malaria"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}",
                                      "is_error": False}]},
    ]
    out = _to_openai_messages("sys", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "hybrid_search"
    assert out[3]["tool_call_id"] == "t1"


def test_api_health_and_missing_run(loaded_db):
    from fastapi.testclient import TestClient

    from research_agent.api.main import app

    client = TestClient(app)
    assert client.get("/health").json()["papers"] == loaded_db["total_papers"]
    assert client.get("/runs/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.post("/runs", json={"question": "short"}).status_code == 422


def test_api_builds_a_map(loaded_db, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    from research_agent.api import main as api
    from research_agent.opportunity import pipeline

    real = pipeline.run_map
    monkeypatch.setattr(pipeline, "run_map", lambda q, provider=None, run_id=None:
                        real(q, llm_factory=lambda strong=False: FakeLLM(), run_id=run_id))
    api._hits.clear()
    client = TestClient(api.app)
    assert client.post("/runs", json={"question": "Can ML improve malaria forecasts?", "mode": "maps"}).status_code == 422
    res = client.post("/runs", json={"question": "Can machine learning improve 1-6 month malaria forecasting?",
                                     "mode": "map"}).json()
    assert res["mode"] == "map"
    for _ in range(120):
        run = client.get(f"/runs/{res['run_id']}").json()["run"]
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done" and run["is_map"] and run["has_report"]
    assert client.get(f"/runs/{res['run_id']}/report").json()["report_markdown"].startswith("# Research Opportunity Map")
    listed = {r["run_id"]: r["is_map"] for r in client.get("/runs").json()["runs"]}
    assert listed[res["run_id"]] is True
    # a map counts as two runs against the hourly limit: 2 used, 3 left, so one more map and one run fit
    assert api._hits and len(next(iter(api._hits.values()))) == 2
    api._hits.clear()


def test_anthropic_adapter_parses_sdk_response(monkeypatch):
    import anthropic
    from anthropic.types import Message

    from research_agent.llm.anthropic_client import AnthropicClient

    msg = Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5-20251001",
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 5},
        "content": [{"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "id": "tu1", "name": "hybrid_search", "input": {"query": "malaria"}}],
    })
    captured = {}

    def fake_create(**kw):
        captured.update(kw)
        return msg

    client = AnthropicClient("claude-haiku-4-5-20251001", api_key="test")
    monkeypatch.setattr(client._client.messages, "create", fake_create)
    resp = client.chat("sys", [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                       tools=[{"name": "hybrid_search", "description": "d",
                               "input_schema": {"type": "object", "properties": {}}}],
                       force_tool="hybrid_search")
    assert captured["tool_choice"] == {"type": "tool", "name": "hybrid_search"}
    assert resp.tool_calls[0].input == {"query": "malaria"} and resp.text == "Let me search."
    assert client.usage.input_tokens == 12
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert captured["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    if not client._accepts_temperature:
        assert "temperature" not in captured
    _ = anthropic


def test_health_filter_rejects_passing_mentions(tmp_path):
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq

    from research_agent.ingestion.health_filter import health_subset_sql
    from tests.fixtures.make_fixtures import META_SCHEMA, _row

    rng = __import__("random").Random(1)
    filler = " We evaluate on several benchmarks and report consistent improvements over strong baselines."
    cases = {
        # (title, abstract, categories): expected
        ("Fair Influence Maximization: A Welfare Optimization Approach",
         "We study fairness in influence maximization with applications to public health campaigns.", "cs.SI"): False,
        ("Contrasting Effects of Strong Ties on SIR and SIS Processes in Temporal Networks",
         "We study epidemic spreading on temporal networks.", "physics.soc-ph"): False,
        ("Spreading processes on temporal contact networks",
         "We model epidemic spreading and infection risk for infectious disease control.", "physics.soc-ph"): True,
        ("Early sepsis prediction with gradient boosting",
         "We build a model on routinely collected variables.", "cs.LG"): True,
        ("Single-molecule tracking with a new estimator",
         "We propose an estimator for diffusion coefficients in living cells.", "q-bio.QM"): False,
        ("Dose calculation for proton therapy",
         "A fast Monte Carlo dose engine.", "physics.med-ph"): True,
    }
    rows = [_row(f"2001.{i:05d}", t, a + filler, c, 2020, rng) for i, (t, a, c) in enumerate(cases)]
    path = tmp_path / "m.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=META_SCHEMA), path)
    got = {r[1] for r in duckdb.execute(health_subset_sql(str(path), 2010)).fetchall()}
    for (title, _, _), expected in cases.items():
        assert (title in got) == expected, title
    _ = dt


def test_run_with_nothing_found_ends_gracefully(loaded_db, monkeypatch):
    from research_agent.agents.orchestrator import run_research
    from tests import fake_llm

    def discovery_finds_nothing(self, step, last, messages):
        if step == 0:
            return [fake_llm._call("hybrid_search", query="quantum chromodynamics", year_from=2030)]
        return [fake_llm._call("finish", shortlist_size=0, coverage_notes="Nothing on this topic.")]

    monkeypatch.setattr(fake_llm.FakeLLM, "_discovery", discovery_finds_nothing)
    for mode in ("pipeline", "orchestrated"):
        res = run_research("Unrelated physics question?", mode=mode, llm_factory=lambda strong=False: FakeLLM())
        assert res["report"].startswith("# No relevant papers found")
        assert "Nothing on this topic." in res["report"]


def test_compaction_shortens_old_results_only():
    from research_agent.agents.base import KEEP_RECENT_MESSAGES, compact

    big = "x" * 5000
    msgs = [{"role": "user", "content": [{"type": "text", "text": "task"}]}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "s", "input": {}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": big}]})
    out = compact(msgs, budget=12000)
    assert out[0] == msgs[0] and len(out) == len(msgs)
    assert len(out[2]["content"][0]["content"]) < 500                    # oldest shortened
    assert out[-1]["content"][0]["content"] == big                        # latest untouched
    assert all(m["content"][0].get("content") == big for m in out[-KEEP_RECENT_MESSAGES:] if m["role"] == "user")
    assert msgs[2]["content"][0]["content"] == big                        # input not mutated


def test_retry_waits_honour_retry_after(monkeypatch):
    from research_agent.llm import base

    class RateLimited(Exception):
        status_code = 429
        response = type("R", (), {"headers": {"retry-after": "12"}})()

    waits, calls = [], {"n": 0}
    monkeypatch.setattr(base.time, "sleep", lambda s: waits.append(s))

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimited()
        return "ok"

    assert base.with_retries(flaky) == "ok" and waits == [12.0, 12.0]


def test_agent_recovers_from_rejected_tool_call_and_forces_finish_with_finish_only(loaded_db):
    from research_agent.agents.base import Agent
    from research_agent.llm.base import LLMResponse, ToolCall, Usage
    from research_agent.runstate import RunContext
    from research_agent.tools.search import SEARCH_TOOLS

    class Stubborn:
        """Always searches; once rejects with a Groq-style error; checks what the forced finish offers."""
        provider, model = "stub", "stub"

        def __init__(self):
            self.usage, self.n, self.forced_tools = Usage(), 0, None

        def chat(self, system, messages, tools=None, force_tool=None, max_tokens=0, temperature=0):
            self.n += 1
            if force_tool == "finish":
                self.forced_tools = [t["name"] for t in tools]
                c = ToolCall("f", "finish", {"summary": "done"})
                return LLMResponse("", [c], [{"type": "tool_use", "id": "f", "name": "finish", "input": c.input}])
            if self.n == 2:
                raise RuntimeError("Error code: 400 - tool_use_failed: tool call validation failed")
            c = ToolCall(f"s{self.n}", "hybrid_search", {"query": "malaria"})  # identical every time
            return LLMResponse("", [c], [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}])

    stub = Stubborn()
    ctx = RunContext.create("q", llm_factory=lambda strong=False: stub)
    events = []
    ctx.on_event = lambda a, k, p: events.append((k, p))
    agent = Agent("probe", "r", "You are a probe.", SEARCH_TOOLS, {"type": "object", "properties": {}}, max_turns=5)
    out = agent.run(ctx, "t")
    ctx.close()
    assert out["summary"] == "done" and stub.forced_tools == ["finish"]
    assert any(k == "error" and "invalid tool call" in p["error"] for k, p in events)
    previews = [p["preview"] for k, p in events if k == "tool_result"]
    assert any("already made this exact call" in pv for pv in previews)


def test_resume_skips_finished_agents_after_daily_limit(loaded_db):
    from research_agent.agents.orchestrator import run_research
    from research_agent.llm.base import DailyLimitError

    class QuotaHit(FakeLLM):
        def chat(self, system, messages, tools=None, force_tool=None, **kw):
            if "Literature Analyst" in system:
                raise DailyLimitError("The provider's DAILY token limit is used up")
            return super().chat(system, messages, tools=tools, force_tool=force_tool, **kw)

    with pytest.raises(RuntimeError) as err:
        run_research("What ML methods are used for malaria forecasting?", mode="pipeline",
                     llm_factory=lambda strong=False: QuotaHit())
    run_id = err.value.run_id

    events = []
    res = run_research(resume=run_id, mode="pipeline", llm_factory=lambda strong=False: FakeLLM(),
                       on_event=lambda a, k, p: events.append((a, k)))
    assert res["run_id"] == run_id and "## References" in res["report"]
    assert ("discovery", "skipped") in events and ("literature", "start") in events
    assert ("discovery", "start") not in events


def test_agent_ignores_invented_tool_arguments(ctx):
    from research_agent.agents.base import Agent
    from research_agent.tools.extraction import EXTRACTION_TOOLS

    agent = Agent("probe", "r", "s", EXTRACTION_TOOLS, {"type": "object", "properties": {}})
    tool = next(t for t in EXTRACTION_TOOLS if t.name == "extraction_coverage")
    out = agent._call_tool(ctx, tool, {"": ""})
    assert out["_ignored_arguments"] == [""] and "extracted" in out


def test_coerce_to_schema_only_fixes_mismatches():
    from research_agent.agents.base import coerce_to_schema

    schema = {"properties": {"observations": {"type": "array"}, "n": {"type": "integer"},
                             "summary": {"type": "string"}, "p": {"type": "object"}}}
    good = {"observations": ["a", "b"], "n": 3, "summary": "s", "p": {"x": 1}}
    assert coerce_to_schema(good, schema) == good  # well-formed calls are untouched
    fixed = coerce_to_schema({"observations": "one finding", "n": "4", "p": '{"x": 1}'}, schema)
    assert fixed == {"observations": ["one finding"], "n": 4, "p": {"x": 1}}


def test_groq_rejected_call_is_recovered_only_when_valid():
    from research_agent.llm.groq_client import _recover_rejected_call

    tools = [{"name": "finish", "description": "", "input_schema": {"type": "object"}}]

    class Err(Exception):
        def __init__(self, gen, code="tool_use_failed"):
            self.body = {"error": {"code": code, "failed_generation": gen}}

    ok = _recover_rejected_call(Err('{"name": "finish", "arguments": {"observations": "x"}}'), tools)
    assert ok.tool_calls[0].name == "finish" and ok.tool_calls[0].input == {"observations": "x"}
    assert _recover_rejected_call(Err('{"name": "finish", "arguments": {"obs'), tools) is None  # truncated
    assert _recover_rejected_call(Err('{"name": "nope", "arguments": {}}'), tools) is None     # unknown tool
    assert _recover_rejected_call(Err("{}", code="other"), tools) is None


def test_report_normalises_brackets_and_flags_unbacked_numbers():
    from research_agent.agents.report import audit_numbers, normalise_citations

    body = normalise_citations("U-Net dominates【C1】 and diffusion is rare【C3】 ［arXiv:2401.00722］")
    assert body == "U-Net dominates[C1] and diffusion is rare[C3] [arXiv:2401.00722]"
    text, n = audit_numbers("12 of 16 papers use U-Net; 12 of 16 were read in full; 3/16 validate.",
                            {(12, 16)})
    assert n == 1 and "3/16 validate" not in text and "3/16 [unverified]" in text
    assert text.count("[unverified]") == 1


def test_claim_direction_and_strictness(ctx):
    from research_agent.tools.claims import direction_problem, propose_claim

    # the three mis-specified claims from the first real run
    assert direction_problem("Diffusion methods appear in less than 15% of the papers",
                             {"min_share": 0.15})
    assert direction_problem("External validation is reported in less than 20% of the papers",
                             {"min_share": 0.2})
    assert direction_problem("U-Net variants dominate, used in most papers", {"max_share": 0.5})
    assert direction_problem("U-Net variants are used in at least 50% of papers", {"min_share": 0.5}) is None

    bad = propose_claim(ctx, "Few papers use MIMIC.", "prevalence",
                        {"field": "datasets", "any_of": ["MIMIC"], "min_count": 1})
    assert "error" in bad
    good = propose_claim(ctx, "Fewer than 50% of papers use MIMIC.", "prevalence",
                         {"field": "datasets", "any_of": ["MIMIC"], "max_share": 0.5})
    again = propose_claim(ctx, "Fewer than 50% of papers use MIMIC.", "prevalence",
                          {"field": "datasets", "any_of": ["MIMIC"], "max_share": 0.5})
    assert again["status"] == "duplicate" and again["claim_id"] == good["claim_id"]
    row = ctx.pg.execute("SELECT predicate FROM claims WHERE id=%s", (good["claim_id"],)).fetchone()
    assert row["predicate"]["strict"] is True


def test_deepseek_adapter_request_and_response():
    import json as _json

    import httpx

    from research_agent.llm.deepseek_client import DeepSeekClient

    seen = []

    def handler(request):
        body = _json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": "", "reasoning_content": "thinking about it",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "ok"}'}}]}}],
                "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 80, "completion_tokens": 7}})
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    client = DeepSeekClient("deepseek-flash", api_key="k", thinking="enabled",
                            transport=httpx.MockTransport(handler))
    tools = [{"name": "finish", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    resp = client.chat("sys", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                       tools=tools, force_tool="finish", max_tokens=500)
    assert seen[0]["tool_choice"] == {"type": "function", "function": {"name": "finish"}}
    assert seen[0]["thinking"] == {"type": "enabled"} and "temperature" not in seen[0]
    assert resp.tool_calls[0].input == {"summary": "ok"}
    assert client.usage.cached_input_tokens == 80 and client.usage.output_tokens == 7

    # reasoning must be sent back on the next request
    history = [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
               {"role": "assistant", "content": resp.content},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{}"}]}]
    with pytest.raises(Exception) as err:
        client.chat("sys", history, tools=tools)
    assert seen[1]["messages"][2]["reasoning_content"] == "thinking about it"
    assert "402" in str(err.value) and "top up" in str(err.value)


def test_claim_scope_and_same_evidence_dedupe(ctx, monkeypatch):
    from research_agent.tools import claims as C

    assert C.scope_problem("Diffusion methods appear in at least 3% of 2024 arXiv health papers.")
    assert C.scope_problem("In the 16-paper shortlist, U-Net appears in most papers.") is None
    assert "error" in C.propose_claim(ctx, "CNNs appear in at least 6% of all health papers in the corpus.",
                                      "prevalence", {"field": "methods", "any_of": ["CNN"], "min_share": 0.06})

    rows = [{"paper_id": f"p{i}", "source": "fulltext",
             "data": {"methods": ["U-Net"] if i < 3 else ["random forest"]}} for i in range(5)]
    monkeypatch.setattr(C, "_rows", lambda _ctx: rows)
    first = C.propose_claim(ctx, "U-Net is used by most papers in the shortlist.", "prevalence",
                            {"field": "methods", "any_of": ["U-Net"], "min_share": 0.5})
    same = C.propose_claim(ctx, "U-Net variants remain widely used in the shortlist.", "prevalence",
                           {"field": "methods", "any_of": ["U-Net", "UNet", "nnU-Net"], "min_share": 0.4})
    assert same["status"] == "duplicate" and same["claim_id"] == first["claim_id"]
    rare = C.propose_claim(ctx, "Random forests are a minority in the shortlist.", "prevalence",
                           {"field": "methods", "any_of": ["random forest"], "max_share": 0.5})
    assert rare["status"] == "pending"


def test_truncated_finish_is_retried_then_salvaged(ctx):
    from research_agent.agents.base import Agent
    from research_agent.llm.base import LLMResponse, ToolCall

    raw = '{"summary": "short", "report_markdown": "## Summary\\n- U-Net dominates [C1]\\n- Transf'

    class CutOff:
        calls = 0

        def chat(self, system, messages, tools=None, force_tool=None, max_tokens=0, temperature=0.0):
            CutOff.calls += 1
            tc = ToolCall(f"f{CutOff.calls}", "finish", {"_raw_arguments": raw})
            return LLMResponse("", [tc], [{"type": "tool_use", "id": tc.id, "name": "finish", "input": tc.input}],
                               "length", {})

    schema = {"type": "object", "properties": {"summary": {"type": "string"},
                                               "report_markdown": {"type": "string"}}}
    agent = Agent("synthesis_probe", "r", "s", [], schema, max_turns=6)
    ctx.llm_factory = lambda strong=False: CutOff()
    out = agent.run(ctx, "write")
    assert CutOff.calls == 2  # asked once to shorten, then salvaged
    assert out["summary"] == "short"
    assert out["report_markdown"].startswith("## Summary\n- U-Net dominates [C1]")
    assert "cut off" in out["_note"]


def test_truncated_list_output_keeps_finished_items(ctx):
    from research_agent.agents.base import Agent, repair_truncated_json
    from research_agent.llm.base import LLMResponse, ToolCall
    from research_agent.opportunity.agents import DESIGN, GAP_REASONING
    from research_agent.opportunity.protocol import PROTOCOL

    raw = ('{"designs": [{"title": "A", "addresses": ["G1"], "why": "has \\"quotes\\", commas"}, '
           '{"title": "B", "addresses": ["G2", "G3"]}, {"title": "C half", "addre')
    seen_budgets = []

    class CutOff:
        def chat(self, system, messages, tools=None, force_tool=None, max_tokens=0, temperature=0.0):
            seen_budgets.append(max_tokens)
            tc = ToolCall(f"f{len(seen_budgets)}", "finish", {"_raw_arguments": raw})
            return LLMResponse("", [tc], [{"type": "tool_use", "id": tc.id, "name": "finish", "input": tc.input}],
                               "length", {})

    schema = {"type": "object", "properties": {"designs": {"type": "array", "items": {"type": "object"}}}}
    agent = Agent("design_probe", "r", "s", [], schema, max_turns=6, long_output=True)
    ctx.llm_factory = lambda strong=False: CutOff()
    out = agent.run(ctx, "design")
    # the two finished designs survive; the half-written third is dropped rather than kept half-empty
    assert [d["title"] for d in out["designs"]] == ["A", "B"] and out["designs"][0]["why"].startswith('has "quotes"')
    assert "cut off" in out["_note"] and len(seen_budgets) == 2
    assert seen_budgets[0] >= 8192 and all(a.long_output for a in (PROTOCOL, GAP_REASONING, DESIGN))
    assert repair_truncated_json('{"summary": "abc') is None and repair_truncated_json("not json") is None
    assert repair_truncated_json('{"a": 1, "b": [1, 2') == {"a": 1, "b": [1]}


def test_short_terms_match_whole_words(ctx, monkeypatch):
    from research_agent.tools import claims as C

    rows = [{"paper_id": "a", "source": "fulltext", "data": {"data_modalities": ["electron microscopy"]}},
            {"paper_id": "b", "source": "fulltext", "data": {"data_modalities": ["CT", "MRI scans"]}},
            {"paper_id": "c", "source": "fulltext", "data": {"data_modalities": ["contrast-enhanced CT"]}},
            {"paper_id": "d", "source": "fulltext", "data": {"data_modalities": ["CNNs", "U-Net++"]}}]
    monkeypatch.setattr(C, "_rows", lambda _ctx: rows)
    r = C.evaluate_prevalence(ctx, {"field": "data_modalities", "any_of": ["CT"], "min_count": 1})
    assert sorted(r["matched_paper_ids"]) == ["b", "c"]
    r = C.evaluate_prevalence(ctx, {"field": "data_modalities", "any_of": ["CNN", "U-Net"], "min_count": 1})
    assert r["matched_paper_ids"] == ["d"]


def test_wording_must_match_observed_share():
    from research_agent.tools.claims import wording_problem

    assert wording_problem("CT and MRI account for the large majority of papers", 0.625)
    assert wording_problem("Convolutions remain near-universal", 0.69)
    assert wording_problem("Convolutions remain near-universal", 0.81) is None
    assert wording_problem("Most papers use U-Net", 0.5)
    assert wording_problem("U-Net is the most common family", 0.4) is None
    assert wording_problem("Geography is almost never reported", 0.12) is None
    assert wording_problem("External validation is a small minority", 0.4)


def test_claim_citations_are_aligned_with_their_numbers():
    from research_agent.agents.report import _clean_author, align_claim_citations
    from research_agent.tools.claims import wording_problem

    def c(n, d, text, status="supported"):
        return {"status": status, "claim_type": "prevalence", "text": text,
                "result": {"n_matching": n, "denominator": d}}

    claims = {45: c(18, 60, "Transformer methods are mainstream"), 46: c(14, 60, "SAM foundation models"),
              47: c(16, 60, "Attention is near-ubiquitous", "unsupported"), 48: c(32, 60, "MRI and CT dominate"),
              49: c(18, 60, "Public benchmark datasets are concentrated"), 50: c(23, 60, "Transformer/attention")}
    body = ("SAM models appear in 14 of 60 papers [C48]. MRI and CT match 32 of 60 papers [C50].\n"
            "A benchmark query matches 18 of 60 papers [C50]. Attention appears in 16 of 60 [C46]. "
            "Transformers are common [C45].")
    out, fixes = align_claim_citations(body, claims)
    assert "14 of 60 papers [C46]" in out and "32 of 60 papers [C48]" in out
    assert "18 of 60 papers [C49]" in out                      # tie with C45 broken by wording
    assert "16 of 60 [C46 — number does not match this claim]" in out
    assert "Transformers are common [C45]" in out              # no number: left alone
    assert _clean_author(r"Ziya Ata Yaz{\i}c{\i}") == "Ziya Ata Yazıcı"
    assert _clean_author("Douwe J. Spaanderman (1)") == "Douwe J. Spaanderman"
    assert wording_problem("Diffusion is rare, at most 15% of papers", 0.05) is None


def test_brief_never_drops_claims_when_notes_are_long(ctx):
    import json as _json

    from research_agent.agents.base import _dump
    from research_agent.agents.report import REPORT_TOOLS, get_brief
    from research_agent.tools.claims import propose_claim

    ids = [propose_claim(ctx, f"In the shortlist, method {i} appears in fewer than half of papers.", "prevalence",
                         {"field": "methods", "any_of": [f"m{i}"], "max_share": 0.5})["claim_id"] for i in range(15)]
    for agent in ("discovery", "literature", "methods", "trends", "gaps", "evidence"):
        ctx.save_note(agent, {"summary": "x" * 5000, "old_ids": "C1 C2 C3"})
    tool = REPORT_TOOLS[0]
    text = _dump(get_brief(ctx), tool.max_chars)
    assert all(f'"C{i}"' in text for i in ids)
    assert "truncated" not in text[: text.index('"agent_outputs"')]
    _json.loads(text)  # still valid JSON


def test_claim_text_numbers_and_negated_wording():
    from research_agent.tools.claims import text_count_problem, wording_problem

    # a claim about what is missing is judged on the share that does not match
    assert wording_problem("Most papers are silent on validation level.", 0.30) is None
    assert wording_problem("Most papers report a validation level.", 0.30)
    # counts written into the claim text must be the counted ones; thresholds are not counts
    # 18 silent is the complement of 28 stating, so the number itself is fine
    assert text_count_problem("Validation is unreported in a minority (18/46 silent).", 28, 46) is None
    assert text_count_problem("Validation is unreported in a minority (12/46 silent).", 28, 46)
    assert text_count_problem("Named datasets are reported in fewer than half (19/46).", 14, 46)
    assert text_count_problem("Human judgment appears in at most 4 of 60 papers.", 3, 60) is None
    assert text_count_problem("Classical time series appear in 9+ papers.", 8, 46) is None
    assert text_count_problem("Code availability is stated in 9/46 papers.", 9, 46) is None
    # a count over a population the check did not measure is not evidence for the claim (C432 in a real run:
    # "15 of 58 (26%)" passed while the check counted 19 of 100)
    assert "out of 33" in text_count_problem("Benchmarks appear in 8 of the 33 dataset papers.", 10, 60)
    assert text_count_problem("In the 58 papers with methods, ML appears in 15 of 58 (26%).", 19, 100)
    assert text_count_problem("Used by 3 of 5 models in one paper.", 3, 60) is None   # not a paper population


def test_claim_percentages_must_match_the_measured_share():
    from research_agent.tools.claims import percent_problem

    assert percent_problem("In 58 shortlisted papers, 50 (86%) use climate covariates.", 0.80)
    assert percent_problem("Climate covariates appear in 80 of 100 papers (80%).", 0.80) is None
    assert percent_problem("Silent on code in 85% of papers.", 0.15) is None                 # complement
    assert percent_problem("Most models report accuracy above 95%.", 0.60) is None           # threshold
    assert percent_problem("17 of 100 report 95% credible intervals (17%).", 0.17) is None   # interval level


def test_text_thresholds_and_complements(ctx, monkeypatch):
    from research_agent.tools import claims as C
    from research_agent.agents.report import align_claim_citations

    assert C.text_count_problem("benchmarks appear in at least 8 of the 33 papers", 6, 33)
    assert C.text_count_problem("benchmarks appear in at least 8 of the 33 papers", 10, 33) is None
    assert C.text_count_problem("human judgment appears in at most 4 of 60 papers", 7, 60)
    assert C.text_count_problem("datasets are unnamed in 32 of 46 papers", 14, 46) is None  # complement

    claims = {86: {"status": "supported", "claim_type": "prevalence", "text": "Named datasets are rare",
                   "result": {"n_matching": 14, "denominator": 46}}}
    out, fixes = align_claim_citations("No dataset is named in 32 of 46 papers [C86].", claims)
    assert "number does not match" not in out and not fixes

    rows = [{"paper_id": f"p{i}", "source": "fulltext",
             "data": {"methods": ["ARIMA"] if i < 9 else (["Box-Jenkins"] if i == 9 else ["LSTM"])}}
            for i in range(12)]
    monkeypatch.setattr(C, "_rows", lambda _ctx: rows)
    first = C.propose_claim(ctx, "ARIMA is used by most shortlisted papers.", "prevalence",
                            {"field": "methods", "any_of": ["ARIMA"], "min_share": 0.5})
    near = C.propose_claim(ctx, "Classical time series appear in most shortlisted papers.", "prevalence",
                           {"field": "methods", "any_of": ["ARIMA", "Box-Jenkins"], "min_count": 5})
    assert near["status"] == "pending" and f"C{first['claim_id']}" in near["note"]


# ---------------------------------------------------------------- PubMed Central
def _pmc_article(pmcid, title, licence, year=2021, abstract="A study of malaria forecasting in Kenya "
                 "using routine surveillance data and gradient boosting models over five seasons."):
    return f"""<article><front>
      <journal-meta><journal-title>Journal of Test Medicine</journal-title></journal-meta>
      <article-meta>
        <article-id pub-id-type="pmcid">{pmcid}</article-id>
        <article-id pub-id-type="doi">10.1234/{pmcid}</article-id>
        <title-group><article-title>{title}</article-title></title-group>
        <contrib-group><contrib contrib-type="author"><name><surname>Otieno</surname>
          <given-names>A</given-names></name></contrib></contrib-group>
        <pub-date pub-type="epub"><year>{year}</year></pub-date>
        <abstract><p>{abstract}</p></abstract>
        <permissions><license license-type="{licence}">
          <ext-link xmlns:xlink="http://www.w3.org/1999/xlink"
            xlink:href="https://creativecommons.org/licenses/{licence}/4.0/">licence</ext-link>
        </license></permissions>
        <subj-group><subject>Public Health</subject></subj-group>
      </article-meta></front>
      <body><sec><title>Methods</title><p>We fitted a gradient boosting model.</p>
        <table-wrap><p>dropped table</p></table-wrap></sec>
        <sec><title>Limitations</title><p>Single district, no external validation.</p></sec></body>
    </article>"""


def _pmc_transport(articles, fail_first: int = 0):
    """A fake NCBI E-utilities server holding `articles`; optionally fails the first N efetch calls."""
    import re as _re
    from urllib.parse import parse_qs

    import httpx

    by_id = {_re.search(r"PMC(\d+)", a).group(1): a for a in articles}
    state = {"fails": fail_first}

    def handler(request):
        url = str(request.url)
        params = {k: v for k, v in request.url.params.items()}
        if request.content:
            params.update({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
        if "/esearch.fcgi" in url and params.get("retmax") == "0":      # per-year denominators
            return httpx.Response(200, text="<eSearchResult><Count>1234</Count></eSearchResult>")
        if "/esearch.fcgi" in url:
            ids = "".join(f"<Id>{i}</Id>" for i in by_id)
            return httpx.Response(200, text=f"<eSearchResult><Count>{len(by_id)}</Count>"
                                            f"<IdList>{ids}</IdList></eSearchResult>")
        if state["fails"]:
            state["fails"] -= 1
            return httpx.Response(503, text="busy")
        wanted = [i.removeprefix("PMC") for i in params.get("id", "").split(",") if i]
        return httpx.Response(200, text="<pmc-articleset>" + "".join(by_id[i] for i in wanted if i in by_id)
                                        + "</pmc-articleset>")

    return httpx.MockTransport(handler)


def test_pmc_licence_filter_and_parsing():
    import xml.etree.ElementTree as ET

    from research_agent.ingestion import pmc

    assert pmc.licence_allows_reuse("cc-by https://creativecommons.org/licenses/by/4.0/")
    assert pmc.licence_allows_reuse("CC0 public domain")
    assert not pmc.licence_allows_reuse("cc-by-nc https://creativecommons.org/licenses/by-nc/4.0/")
    assert not pmc.licence_allows_reuse("")

    row = pmc.parse_article(ET.fromstring(_pmc_article("PMC111", "Malaria forecasting", "by")))
    assert row["paper_id"] == "PMC111" and row["year"] == 2021 and row["doi"] == "10.1234/PMC111"
    assert row["authors"] == "A Otieno" and row["categories"] == ["Public Health"]
    assert pmc.parse_article(ET.fromstring("<article><front/></article>")) is None  # no abstract


def test_pmc_ingest_writes_source_tagged_papers(loaded_db):
    from research_agent.db import get_conn
    from research_agent.ingestion import pmc

    client = pmc.PMCClient(transport=_pmc_transport([
        _pmc_article("PMC111", "Malaria forecasting with boosted trees", "by"),
        _pmc_article("PMC222", "Non-commercial study of malaria", "by-nc"),
    ]))
    res = pmc.ingest("malaria forecasting", from_year=2010, log=lambda *_: None, client=client)
    assert res["inserted"] == 1 and res["skipped_licence"] == 1

    with get_conn() as pg:
        row = pg.execute("SELECT source, title, journal_ref, health_reason FROM papers "
                         "WHERE paper_id='PMC111'").fetchone()
        n_arxiv = pg.execute("SELECT count(*) n FROM papers WHERE source='arxiv'").fetchone()["n"]
        stats = pg.execute("SELECT count(*) n FROM corpus_year_stats WHERE source='pmc'").fetchone()["n"]
    assert row["source"] == "pmc" and row["journal_ref"] == "Journal of Test Medicine"
    assert row["health_reason"].startswith("pmc:") and n_arxiv > 0   # arXiv papers untouched
    assert stats > 0                                                  # PMC denominators loaded

    again = pmc.ingest("malaria forecasting", from_year=2010, log=lambda *_: None, client=client)
    assert again["inserted"] == 0                                     # resumable


def test_pmc_fulltext_and_source_aware_tools(loaded_db, ctx):
    from research_agent.db import get_conn
    from research_agent.ingestion import pmc
    from research_agent.tools.search import corpus_count, hybrid_search
    from research_agent.tools.trends import corpus_sources, topic_series

    client = pmc.PMCClient(transport=_pmc_transport([
        _pmc_article("PMC111", "Malaria forecasting with boosted trees", "by")]))
    pmc.ingest("malaria forecasting", from_year=2010, log=lambda *_: None, client=client)

    with get_conn() as pg:
        status = pmc.fetch_fulltext(pg, ["PMC111"], client=client)
        text = pg.execute("SELECT clean_text, sections FROM paper_fulltext "
                          "WHERE paper_id='PMC111'").fetchone()
    assert status["PMC111"] in ("ok", "stub", "template_suspect")
    assert "gradient boosting" in text["clean_text"] and "dropped table" not in text["clean_text"]
    assert [s["heading"] for s in text["sections"]] == ["Methods", "Limitations"]

    assert corpus_sources(ctx)["sources"].keys() >= {"arxiv", "pmc"}
    assert corpus_count(ctx, "malaria")["by_source"].get("pmc", 0) == 1
    only_pmc = hybrid_search(ctx, "malaria forecasting", source="pmc", limit=10)["results"]
    assert only_pmc and all(r["source"] == "pmc" for r in only_pmc)
    assert all(r["source"] == "arxiv" for r in hybrid_search(ctx, "malaria", source="arxiv")["results"])
    assert topic_series(ctx, "malaria", source="pmc")["series"]     # normalised against PMC totals


def test_schema_upgrades_a_pre_pmc_database(loaded_db):
    """An existing database from before PMC support must migrate in place, keeping its papers."""
    from research_agent.db import get_conn, init_schema

    with get_conn() as pg:
        n_before = pg.execute("SELECT count(*) n FROM papers").fetchone()["n"]
        # put the database back into its old shape
        pg.execute("DELETE FROM corpus_year_stats WHERE source <> 'arxiv'")
        pg.execute("ALTER TABLE papers DROP COLUMN source")
        pg.execute("ALTER TABLE corpus_year_stats DROP COLUMN source")
        pg.execute("ALTER TABLE corpus_year_stats ADD PRIMARY KEY (year, primary_category)")

    init_schema()  # must not fail
    init_schema()  # and must be safe to run again

    with get_conn() as pg:
        assert pg.execute("SELECT count(*) n FROM papers").fetchone()["n"] == n_before
        assert pg.execute("SELECT count(*) n FROM papers WHERE source='arxiv'").fetchone()["n"] == n_before
        pk = pg.execute("""SELECT array_length(conkey, 1) k FROM pg_constraint
                           WHERE conname = 'corpus_year_stats_pkey'""").fetchone()["k"]
        idx = pg.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'papers_source_idx'").fetchone()
    assert pk == 3 and idx


def test_pmc_noncommercial_flag_and_citations(loaded_db, ctx):
    from research_agent.agents.report import finalize_report
    from research_agent.db import get_conn
    from research_agent.ingestion import pmc
    from research_agent.tools.search import add_to_shortlist

    assert pmc.licence_is_noncommercial("by-nc https://creativecommons.org/licenses/by-nc/4.0/")
    assert not pmc.licence_is_noncommercial("by https://creativecommons.org/licenses/by/4.0/")
    assert not pmc.licence_is_noncommercial("")

    articles = [_pmc_article("PMC501", "Malaria forecasting with boosted trees", "by"),
                _pmc_article("PMC502", "Malaria early warning in Kenya", "by-nc")]
    client = pmc.PMCClient(transport=_pmc_transport(articles))
    first = pmc.ingest("malaria nc test", from_year=2010, log=lambda *_: None, client=client)
    assert first["inserted"] == 1 and first["skipped_licence"] == 1
    # rerun with the flag: only the skipped non-commercial article is added
    second = pmc.ingest("malaria nc test", from_year=2010, log=lambda *_: None, client=client,
                        include_noncommercial=True)
    assert second["inserted"] == 1 and second["skipped_licence"] == 0

    add_to_shortlist(ctx, ["PMC501", "PMC502"], reason="t")
    report, audit = finalize_report(
        ctx, "## Summary\n- Boosted trees [PMC501] and early warning [arXiv:PMC502]; bogus [PMC999].")
    assert "[PMC501]" in report and "[PMC502]" in report
    assert "PMC999" not in report.split("## Run facts")[0].replace("citation removed", "")
    assert audit["removed_paper_citations"] == ["PMC999"]
    refs = report.split("## References")[1]
    assert "https://pmc.ncbi.nlm.nih.gov/articles/PMC502/ *(licence: non-commercial)*" in refs
    assert "PMC501/" in refs and "PMC501/ *(licence" not in refs
    with get_conn() as pg:
        pg.execute("DELETE FROM run_papers WHERE paper_id IN ('PMC501','PMC502')")


def test_pmc_search_splits_large_results_and_retries(monkeypatch):
    from datetime import date

    import httpx

    from research_agent.ingestion import pmc

    # a fake search where every year holds 3 ids, and one request may return at most 4
    calls = []

    def handler(request):
        term = request.url.params["term"]
        calls.append(term)
        years = [int(y) for y in __import__("re").findall(r'"(\d{4})/\d\d/\d\d"', term)]
        span = range(years[0], years[1] + 1)
        ids = [f"{y}{k}" for y in span for k in range(3)]
        shown = "".join(f"<Id>{i}</Id>" for i in ids[:4])
        return httpx.Response(200, text=f"<eSearchResult><Count>{len(ids)}</Count>"
                                        f"<IdList>{shown}</IdList></eSearchResult>")

    monkeypatch.setattr(pmc, "MAX_IDS", 4)
    client = pmc.PMCClient(transport=httpx.MockTransport(handler))
    ids = pmc.search_ids(client, "malaria", 2018, 2021, log=lambda *_: None)
    assert len(ids) == 12 and len(set(ids)) == 12 and all(i.startswith("PMC") for i in ids)
    assert len(calls) > 1                       # the range was split to stay under the cap

    # transient 503s are retried instead of killing a long ingest
    flaky = pmc.PMCClient(transport=_pmc_transport([_pmc_article("PMC777", "Malaria", "by")], fail_first=2))
    flaky._min_gap = 0
    monkeypatch.setattr(pmc.time, "sleep", lambda *_: None)
    assert [r["paper_id"] for r in pmc.fetch_ids(flaky, ["PMC777"])] == ["PMC777"]


def test_licence_rules_and_audit_of_loaded_papers(loaded_db):
    from research_agent.db import get_conn
    from research_agent.ingestion import pmc

    k = pmc.licence_kind
    assert k("© 2020 by the authors. All rights reserved.") == "unknown"   # a bare "by" is not CC BY
    assert k("open-access") == "unknown"
    assert k("open-access © 2019 by the authors. Licensee MDPI. " + "x" * 250 +
             " distributed under the Creative Commons Attribution (CC BY) license.") == "reusable"
    assert k("Creative Commons Attribution-NonCommercial 4.0 International License") == "noncommercial"
    assert k("https://creativecommons.org/publicdomain/zero/1.0/") == "reusable"

    with get_conn() as pg:
        base = dict(abstract="x" * 100, categories=["pmc"], year=2020, health_reason="pmc:test")
        for pid, lic in [("PMC9001", "cc-by https://creativecommons.org/licenses/by/4.0/"),
                         ("PMC9002", "© 2020 by the authors"),          # would have slipped in before
                         ("PMC9003", "by-nc https://creativecommons.org/licenses/by-nc/4.0/")]:
            pg.execute("INSERT INTO papers (paper_id, source, title, abstract, categories, year, license, "
                       "health_reason) VALUES (%s,'pmc',%s,%s,%s,%s,%s,%s)",
                       (pid, pid, base["abstract"], base["categories"], base["year"], lic, base["health_reason"]))
        dry = pmc.audit_licences(pg)
        assert dry["removed"] == 0 and dry["failing"] >= 2
        kept_nc = pmc.audit_licences(pg, remove=True, allow_noncommercial=True)
        left = {r["paper_id"] for r in pg.execute(
            "SELECT paper_id FROM papers WHERE paper_id IN ('PMC9001','PMC9002','PMC9003')").fetchall()}
        assert kept_nc["removed"] >= 1 and left == {"PMC9001", "PMC9003"}
        pg.execute("DELETE FROM papers WHERE paper_id IN ('PMC9001','PMC9003')")


def test_trend_claim_numbers_and_scope(ctx):
    from research_agent.tools.claims import propose_claim, trend_text_problem

    # real claims from the mixed-corpus malaria report
    assert trend_text_problem("ML/DL rose sharply, roughly a 9-fold increase", 5.32)            # C172
    assert trend_text_problem("validation grew only marginally (ratio ~1.1)", 2.38)             # C184
    assert trend_text_problem("did not increase (ratio <= 1.0)", 2.78)                          # C185
    assert trend_text_problem("grew in normalised share (ratio >= 1.5)", 4.83) is None         # C183
    assert trend_text_problem("grew several-fold", 6.6) is None                                 # no number
    assert trend_text_problem("(ratio ~3.2)", 3.17) is None and trend_text_problem("an ~1.8x rise", 1.8) is None
    # only the first multiplier is the claim's own; later ones describe a comparison topic
    assert trend_text_problem("rose ~2.1x while bias terms rose only ~1.5x", 2.07) is None

    bad = propose_claim(ctx, "ML use increased within the shortlist over time.", "trend",
                        {"keywords": "machine learning", "early": [2015, 2017], "late": [2023, 2025],
                         "direction": "increase"})
    assert "error" in bad
    ok = propose_claim(ctx, "In PMC, malaria ML papers grew.", "trend",
                       {"keywords": "malaria machine learning", "early": [2015, 2017], "late": [2023, 2025],
                        "direction": "increase"})
    row = ctx.pg.execute("SELECT predicate FROM claims WHERE id=%s", (ok["claim_id"],)).fetchone()
    assert row["predicate"]["source"] == "pmc"


def test_preprint_and_published_version_are_not_both_shortlisted(loaded_db, ctx):
    from research_agent.db import get_conn
    from research_agent.tools.search import add_to_shortlist

    with get_conn() as pg:
        for pid, src, title, doi in [
            ("2412.99999", "arxiv", "Forecasting Malaria in Indian States: A Time Series Approach with R Shiny "
                                    "Integration", None),
            ("PMC8888888", "pmc", "Forecasting Malaria in Indian States: A Time Series Approach with R Shiny "
                                  "Integration", "10.1/x"),
            ("1906.99999", "arxiv", "Data-Driven Malaria Prevalence Prediction in Large Densely-Populated Urban "
                                    "Holoendemic sub-Saharan West Africa: Harnessing Machine Learning", None),
            ("PMC7777777", "pmc", "Data-driven malaria prevalence prediction in large densely populated urban "
                                  "holoendemic sub-Saharan West Africa", None)]:
            pg.execute("INSERT INTO papers (paper_id, source, title, abstract, authors, categories, year, doi, "
                       "health_reason) VALUES (%s,%s,%s,%s,%s,%s,2024,%s,'t') ON CONFLICT DO NOTHING",
                       (pid, src, title, "x" * 100, "Sujit K. Ghosh, A. B" if "Indian" in title
                        else "Biobele J. Brown, C. D", ["t"], doi))

    first = add_to_shortlist(ctx, ["2412.99999", "1906.99999"], reason="t")
    assert first["added"] == 2
    # published versions replace the preprints already on the list
    second = add_to_shortlist(ctx, ["PMC8888888", "PMC7777777"], reason="t")
    assert len(second["replaced_preprints"]) == 2
    ids = set(ctx.shortlist_ids())
    assert {"PMC8888888", "PMC7777777"} <= ids and not {"2412.99999", "1906.99999"} & ids
    # and a preprint is not added back when its published version is there
    third = add_to_shortlist(ctx, ["2412.99999"], reason="t")
    assert third["added"] == 0 and third["skipped_duplicates"][0]["same_work_as"] == "PMC8888888"
    with get_conn() as pg:
        pg.execute("DELETE FROM run_papers WHERE run_id=%s", (ctx.run_id,))
        pg.execute("DELETE FROM papers WHERE paper_id IN ('2412.99999','PMC8888888','1906.99999','PMC7777777')")


def test_unsupported_counts_may_be_quoted_but_not_rejected_ones():
    from research_agent.agents.report import align_claim_citations, audit_numbers

    def c(n, d, status, text="t"):
        return {"status": status, "claim_type": "prevalence", "text": text,
                "result": {"n_matching": n, "denominator": d}}

    claims = {166: c(24, 56, "unsupported", "classical methods majority"),
              167: c(27, 56, "supported", "ML minority"), 109: c(10, 60, "rejected")}
    body = "Classical methods reach 24 of 56, short of half (claim C166, unsupported), and ML is a minority [C167]."
    out, fixes = align_claim_citations(body, claims)
    assert "number does not match" not in out and not fixes
    allowed = {(24, 56), (32, 56), (27, 56), (29, 56)}   # what finalize_report derives for C166 and C167
    assert audit_numbers("24 of 56 and 10 of 60", allowed)[1] == 1   # the rejected claim's count is flagged


def test_pmc_year_is_publication_not_collection():
    import xml.etree.ElementTree as ET

    from research_agent.ingestion import pmc

    xml = _pmc_article("PMC123", "Malaria forecasting", "by", year=2026).replace(
        '<pub-date pub-type="epub"><year>2026</year></pub-date>',
        '<pub-date pub-type="collection"><year>2027</year></pub-date>'
        '<pub-date pub-type="epub"><year>2026</year></pub-date>')
    assert pmc.parse_article(ET.fromstring(xml))["year"] == 2026


def test_extraction_evidence_is_checked_against_the_text():
    from research_agent.tools.extraction import check_evidence, normalise

    text = ("Title: Forecasting malaria in Kenya\n\nAbstract: We trained a gradient boosting model on "
            "weekly surveillance counts from 42 sites in western Kenya, with external validation on two "
            "held-out counties.")
    raw = {"methods": ["gradient boosting"], "geography": ["Kenya"], "datasets": ["MIMIC-IV"],
           "data_modalities": ["surveillance counts"], "validation_level": "external",
           "code_or_data_available": "yes"}
    evidence = {
        "methods": ["We trained a gradient boosting model on weekly"],          # verbatim
        "geography": ["from 42 sites in western Kenya,"],                      # verbatim, punctuation differs
        "data_modalities": ["weekly surveillance counts from 42 sites"],
        "validation_level": ["with external validation on two held-out counties"],
        "datasets": ["the MIMIC-IV database was used"],                        # invented: not in the text
        # code_or_data_available: no quote at all
    }
    out = check_evidence(normalise(raw), evidence, text)
    assert out["methods"] == ["gradient boosting"] and out["geography"] == ["Kenya"]
    assert out["validation_level"] == "external"
    assert out["datasets"] == [] and out["code_or_data_available"] == "not_stated"   # dropped
    assert out["_unverified"] == {"datasets": ["MIMIC-IV"], "code_or_data_available": "yes"}
    assert set(out["evidence"]) == {"methods", "geography", "data_modalities", "validation_level"}
    # a paraphrase is not a quote
    para = check_evidence(normalise({"methods": ["gradient boosting"]}),
                          {"methods": ["the authors used a boosted tree ensemble"]}, text)
    assert para["methods"] == [] and "methods" in para["_unverified"]


def test_cut_off_extraction_fails_loudly():
    from research_agent.llm.base import LLMResponse, ToolCall
    from research_agent.tools.reading import _read_one

    class CutOff:
        model = "x"

        def chat(self, *a, **k):
            tc = ToolCall("1", "record_reading", {"_raw_arguments": '{"problem": "malaria", "meth'})
            return LLMResponse("", [tc], [], "length", {})

    class Ctx:
        question = "q"

    import pytest as _pytest
    with _pytest.raises(ValueError, match="cut off"):
        _read_one(CutOff(), Ctx(), {"paper_id": "p", "title": "t", "abstract": "a"}, None, True)


def test_claims_are_measured_before_they_are_written(ctx, monkeypatch):
    from research_agent.tools import claims as C

    rows = [{"paper_id": f"p{i}", "source": "fulltext", "year": 2020, "title": "t",
             "data": {"methods": ["ARIMA"] if i < 9 else ["LSTM"]}} for i in range(46)]
    monkeypatch.setattr(C, "_rows", lambda _ctx: rows)

    dry = C.test_claim(ctx, "prevalence", {"field": "methods", "any_of": ["ARIMA"], "min_count": 5})
    assert dry["n_matching"] == 9 and dry["denominator"] == 46 and dry["bounds_pass"]
    assert not ctx.pg.execute("SELECT 1 FROM claims WHERE run_id=%s", (ctx.run_id,)).fetchone()  # nothing stored

    # text written before measuring, contradicting the count: refused, with the real number returned
    bad = C.propose_claim(ctx, "Classical methods appear in 18 of 46 papers.", "prevalence",
                          {"field": "methods", "any_of": ["ARIMA"], "min_count": 5})
    assert "error" in bad and "9 of 46" in bad["measured"]
    good = C.propose_claim(ctx, "Classical methods appear in 9 of 46 shortlisted papers.", "prevalence",
                           {"field": "methods", "any_of": ["ARIMA"], "min_count": 5})
    assert good["status"] == "pending" and good["measured_now"].startswith("9 of 46")


# ---------------------------------------------------------------- Research Opportunity Map
def test_protocol_validation_cleans_fields():
    from research_agent.opportunity.protocol import validate_protocol

    clean, problems = validate_protocol({"fields": [
        {"name": "Forecast Horizon", "type": "enum", "values": ["1 month", "3-6 months", "not stated"],
         "definition": "how far ahead", "desirable": ["3-6 months", "nonsense"],
         "search": {"3-6 months": '"six months"'}},
        {"name": "methods", "type": "list", "definition": "already a base field"},
        {"name": "lonely", "type": "enum", "values": ["only"], "definition": "x"},
        {"name": "forecast_horizon", "type": "enum", "values": ["a", "b"], "definition": "duplicate"},
        {"name": "covariates", "type": "list", "values": ["rainfall"], "definition": "inputs",
         "desirable": ["Intervention coverage"]}]})
    f = {x["name"]: x for x in clean["fields"]}
    assert set(f) == {"q_forecast_horizon", "q_covariates"} and len(problems) == 3
    assert f["q_forecast_horizon"]["values"] == ["1_month", "3_6_months", "not_stated"]
    assert f["q_forecast_horizon"]["desirable"] == ["3_6_months"]
    assert f["q_forecast_horizon"]["search"] == {"3_6_months": '"six months"'}
    assert f["q_covariates"]["desirable"] == ["intervention coverage"]


def test_subgroup_counts_with_where(ctx, monkeypatch):
    from research_agent.tools import claims as C

    rows = []
    for i in range(20):
        rows.append({"paper_id": f"p{i}", "source": "fulltext" if i < 8 else "abstract",
                     "corpus": "pmc" if i % 2 else "arxiv", "year": 2012 + i % 12, "title": "t",
                     "data": {"geography": ["Kenya"] if i < 10 else ["Brazil"],
                              "data_modalities": ["satellite imagery"] if i in (0, 1, 2, 11) else ["survey"],
                              "validation_level": "external" if i in (0, 3) else "internal"}})
    monkeypatch.setattr(C, "_rows", lambda _ctx: rows)
    sat_kenya = C.test_claim(ctx, "prevalence", {"field": "data_modalities", "any_of": ["satellite"],
                                                 "min_count": 1, "where": {"field": "geography", "any_of": ["Kenya"]}})
    assert (sat_kenya["n_matching"], sat_kenya["denominator"]) == (3, 10)
    pmc_ext = C.evaluate_prevalence(ctx, {"field": "validation_level", "any_of": ["external"], "min_count": 1,
                                          "where": [{"field": "source", "any_of": ["pmc"]}]})
    assert (pmc_ext["n_matching"], pmc_ext["denominator"]) == (1, 10) and pmc_ext["subgroup"]["size"] == 10
    not_kenya_recent = C.evaluate_prevalence(ctx, {
        "field": "data_modalities", "any_of": ["satellite"], "min_count": 1,
        "where": [{"field": "geography", "none_of": ["Kenya"]}, {"field": "year", "min": 2020},
                  {"field": "read", "any_of": ["abstract"]}]})
    assert not_kenya_recent["subgroup"]["size"] < 5 and "too few" in not_kenya_recent["caveat"]
    assert "error" in C.test_claim(ctx, "prevalence", {"field": "methods", "any_of": ["x"], "min_count": 1,
                                                       "where": [{"field": "nonsense", "any_of": ["y"]}]})


def test_research_opportunity_map_end_to_end(loaded_db):
    from research_agent.db import get_conn
    from research_agent.opportunity.pipeline import run_map

    with get_conn() as pg:   # a database from before the map existed has no protocol table
        pg.execute("DROP TABLE IF EXISTS protocol_extractions")
    events = []
    res = run_map("Can machine learning improve 1-6 month malaria forecasting?",
                  llm_factory=lambda strong=False: FakeLLM(), on_event=lambda a, k, p: events.append((a, k)))
    md, m = res["report"], res["map"]
    for section in ("## What is established", "## What is emerging", "## What is missing",
                    "## Evidence strength at a glance", "## Potential novelty", "## Candidate research designs",
                    "## Protocol", "## Method", "## References"):
        assert section in md, section

    labels = {g["label"] for g in m["gaps"]}
    # desirable practices no fixture paper has: 3-6 month horizon, spatial holdout, probabilistic output
    assert {"forecast horizon: three to six months", "validation split: tests on unseen places",
            "probabilistic: yes"} <= labels
    # grouped values make one gap (the group), never one gap per member
    assert not labels & {"validation split: spatial holdout", "validation split: spatiotemporal holdout"}
    group = next(g for g in m["gaps"] if g["label"] == "validation split: tests on unseen places")
    assert group["members"] == ["spatial holdout", "spatiotemporal holdout"] and group["role"] == "evaluation"
    assert "counts any of: spatial holdout, spatiotemporal holdout" in md
    assert all(0 <= g["ci95"][0] <= g["share"] <= g["ci95"][1] <= 1 for g in m["gaps"])
    assert all(g["confidence"]["grade"] in ("high", "moderate", "low") and g["confidence"]["reasons"]
               for g in m["gaps"])
    est = {e["label"] for e in m["established"]}
    assert "forecast horizon: one month" in est and "uses satellite: yes" in est
    assert all(e["strength"]["reasons"] for e in m["established"])

    with get_conn() as pg:
        notes = {r["agent"]: r["content"] for r in pg.execute(
            "SELECT agent, content FROM run_notes WHERE run_id=%s", (res["run_id"],)).fetchall()}
        n_proto = pg.execute("SELECT count(*) n FROM protocol_extractions WHERE run_id=%s",
                             (res["run_id"],)).fetchone()["n"]
    assert n_proto == m["N"] > 0
    # the invented probabilistic quote on non-malaria papers was discarded, so nobody counts as probabilistic
    assert notes["protocol_extraction"]["fields_dropped_without_evidence"].get("q_probabilistic", 0) > 0

    hyps = notes["hypotheses"]["items"]
    verdicts = {h["id"]: h["verdict"] for h in hyps}
    assert verdicts["H1"] == "supported" and verdicts["H2"] == "untestable"
    # the hypothesis whose text contradicted its count was refused, and repeating H1's test reused H1
    assert len(hyps) == 2
    gr = notes["gap_reasoning"]["checked"]["gaps"][0]
    assert [h["id"] for h in gr["hypotheses"]] == ["H1", "H2"]
    assert [ln["hypothesis_id"] for ln in gr["linked_gaps"]] == ["H1"]
    assert all(nm["paper_id"] != "0000.00000" for nm in gr["near_misses"])
    assert "supported" in md and "untestable" in md

    designs = notes["design"]["designs"]
    assert [d["title"] for d in designs] == ["District-holdout 3-6 month forecasts"]
    assert designs[0]["addresses"] == [m["gaps"][0]["id"]] and not designs[0]["challenging"]
    # a hypothesis listed under 'addresses' is kept as a reason, with its code-decided verdict
    assert [(h["id"], h["verdict"]) for h in designs[0]["rests_on"]] == [("H1", "supported")]
    assert "**Rests on:** H1 (supported)" in md
    assert designs[0]["supporting"][0].get("title")
    dropped = notes["design"]["dropped"]
    assert any("G99" in x for x in dropped) and any("9999.99999" in x for x in dropped)
    assert any("H77" in x for x in dropped) and not any("reference H1" in x for x in dropped)
    # invented ids are never cited; they only appear in the list of what the checks removed
    assert "[arXiv:9999.99999]" not in md and "[arXiv:0000.00000]" not in md
    assert "**Removed during checks:**" in md and "9999.99999" in md.split("**Removed during checks:**")[1]
    assert {"protocol", "gap_reasoning", "design"} <= {a for a, _ in events}


# ---------------------------------------------------------------- query parsing and map statistics
def test_query_parser_structure():
    from research_agent.tools.query import tsquery_sql

    sql, params = tsquery_sql('malaria (forecast OR prediction) -vaccine "early warning" climat*')
    assert sql.count("||") == 1 and "!!" in sql and "phraseto_tsquery" in sql and "'climat:*'" in str(params)
    assert list(params.values())[:3] == ["malaria", "forecast", "prediction"]
    # malformed input never raises and never widens the query
    for q in ("malaria (forecast", "malaria forecast)", ")(", "NOT", '"', "", "-", "a OR", "(( OR ))"):
        sql, params = tsquery_sql(q)
        assert isinstance(sql, str) and all(isinstance(v, str) for v in params.values())
    assert tsquery_sql("")[1] == {} and tsquery_sql("OR AND")[1] == {}


def test_query_parser_matches_postgres(loaded_db):
    from research_agent.db import get_conn
    from research_agent.tools.query import tsquery_sql

    docs = {"a": "malaria forecasting with rainfall", "b": "heart failure prediction", "c": "malaria vaccine trial",
            "d": "early warning system for dengue", "e": "climate-driven malaria prediction"}
    cases = {"malaria (forecast OR prediction)": {"a", "e"},
             "malaria -vaccine": {"a", "e"},
             '"early warning"': {"d"},
             "malaria forecast) OR heart": {"a", "b"},     # a stray ')' cannot split the query
             "(malaria OR dengue) AND NOT vaccine": {"a", "d", "e"},
             "climate-driven": {"e"},
             "predict*": {"b", "e"},
             "": set()}
    with get_conn() as pg:
        for q, want in cases.items():
            sql, params = tsquery_sql(q)
            got = {k for k, text in docs.items() if pg.execute(
                f"SELECT to_tsvector('english', %(doc)s) @@ ({sql}) AS m", {**params, "doc": text}).fetchone()["m"]}
            assert got == want, (q, got)


def test_corpus_count_respects_parentheses(ctx):
    from research_agent.tools.search import corpus_count

    malaria = corpus_count(ctx, "malaria")["matching_papers"]
    grouped = corpus_count(ctx, "malaria (forecast OR prediction)")["matching_papers"]
    prediction = corpus_count(ctx, "prediction")["matching_papers"]
    assert 0 < grouped <= malaria and prediction > 0
    # before the fix this read as 'malaria & forecast | prediction' and counted every sepsis 'prediction' paper
    assert corpus_count(ctx, "malaria forecast OR prediction")["matching_papers"] > grouped


def test_statistics_helpers():
    from research_agent.opportunity.compute import hyper_ge, hyper_le, wilson

    lo, hi = wilson(0, 40)
    assert lo == 0 and 0.08 < hi < 0.09            # zero of 40 still allows a true rate up to ~9%
    lo, hi = wilson(20, 40)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9
    assert wilson(0, 0) == (0.0, 1.0)
    # hypergeometric tails are proper probabilities and complementary
    assert abs(hyper_le(3, 50, 20, 10) + hyper_ge(4, 50, 20, 10) - 1) < 1e-9
    assert hyper_le(0, 60, 30, 30) < 1e-15 and hyper_ge(0, 60, 30, 30) == 1


def test_clip_and_protocol_groups():
    from research_agent.opportunity.protocol import _clip, validate_protocol

    assert _clip("one two three four", 11) == "one two …" and _clip("short", 10) == "short"
    clean, problems = validate_protocol({"fields": [
        {"name": "split", "type": "enum", "role": "evaluation",
         "values": ["random", "spatial_holdout", "spatiotemporal_holdout", "temporal"],
         "definition": "x", "desirable": ["spatial_holdout", "spatiotemporal_holdout"],
         "groups": [{"label": "unseen places", "values": ["spatial_holdout", "spatiotemporal_holdout"]},
                    {"label": "not desirable", "values": ["random", "temporal"]},
                    {"label": "reuses a member", "values": ["spatial_holdout", "random"]}]},
        {"name": "horizon", "type": "enum", "values": ["a", "b"], "definition": "y", "role": "made_up"}]})
    f = {x["name"]: x for x in clean["fields"]}
    assert f["q_split"]["groups"] == [{"label": "unseen places",
                                       "values": ["spatial_holdout", "spatiotemporal_holdout"]}]
    assert f["q_split"]["role"] == "evaluation" and f["q_horizon"]["role"] != "made_up"
    assert sum("group" in p for p in problems) == 2


def _synthetic_rows():
    rows = []
    for i in range(60):
        data = {"methods": ["random forest" if i < 30 else "pearson correlation"],
                "data_modalities": (["satellite imagery"] if i % 2 else ["remote sensing"]) if i >= 30
                else ["case counts"],
                "q_model_class": "machine_learning" if i >= 30 else "statistical",
                "q_interventions": "none" if i >= 30 else "itn_only"}
        if i >= 50:
            data["data_modalities"].append("mobile phone data")
        if i in (2, 40, 50):          # a weak rise: 1 of 35 early papers, 2 of 25 recent ones
            data["methods"].append("support vector machine")
        rows.append({"paper_id": f"p{i}", "source": "fulltext" if i % 3 == 0 else "abstract",
                     "corpus": "pmc" if i % 2 else "arxiv", "year": 2010 + i // 5, "title": "t", "data": data})
    return rows


def test_map_novelty_is_cross_dimension_and_tested(ctx, monkeypatch):
    from research_agent.opportunity import compute as C

    monkeypatch.setattr(C, "_rows", lambda _ctx: _synthetic_rows())
    monkeypatch.setattr(C, "known_fields", lambda _ctx: (["methods", "data_modalities"], {
        "q_model_class": ["machine_learning", "statistical"], "q_interventions": ["none", "itn_only"]}))
    monkeypatch.setattr(C, "_corpus_check", lambda *a: None)
    protocol = {"fields": [{"name": "q_model_class", "type": "enum", "role": "method",
                            "values": ["machine_learning", "statistical"], "desirable": [], "definition": "x"},
                           {"name": "q_interventions", "type": "enum", "role": "data",
                            "values": ["none", "itn_only"], "desirable": [], "definition": "y"},
                           {"name": "q_code_shared", "type": "enum", "role": "reporting",
                            "values": ["yes", "no"], "desirable": ["yes"], "definition": "z"}]}
    m = C.compute_map(ctx, protocol)
    labels = {e["label"] for sec in ("established", "emerging", "gaps") for e in m[sec]}
    assert not any("pearson" in x for x in labels)          # generic statistics are not research components
    pairs = [(n["a"], n["b"]) for n in m["novelty"]]
    assert pairs, m["novelty"]
    for a, b in pairs:
        assert a["field"] != b["field"] and a["role"] != b["role"]
    # random forest (method) never meets satellite imagery (data): a cross-dimension, significant pair
    rf_sat = next(n for n in m["novelty"] if {n["a"]["label"], n["b"]["label"]} ==
                  {"methods: random forest", "data modalities: satellite / remote sensing"})
    assert rf_sat["observed"] == 0 and rf_sat["p"] < 0.001
    # model class and methods are the same dimension, so they are never paired however disjoint
    assert not any({a["role"], b["role"]} == {"method"} for a, b in pairs)
    assert all(n["p"] <= C.THRESHOLDS["novelty_max_p"] for n in m["novelty"])
    for e in m["established"]:
        assert e["ci95"][0] <= e["share"] <= e["ci95"][1]
    rise = next(e for e in m["emerging"] if e["label"] == "data modalities: mobile phone data")
    assert rise["trend"]["p"] < 0.05 and any("p" in r for r in rise["strength"]["reasons"])
    # synonyms are one item: satellite imagery and remote sensing count together, 30 papers
    sat = next(e for e in m["established"] if e["label"] == "data modalities: satellite / remote sensing")
    assert sat["n"] == 30 and set(sat["members"]) == {"satellite imagery", "remote sensing"}
    assert not any(e["label"] in ("data modalities: satellite imagery", "data modalities: remote sensing")
                   for sec in ("established", "emerging", "watch") for e in m[sec])
    # an absence value ('none') is never half of a novel combination
    assert not any("interventions: none" in (n["a"]["label"], n["b"]["label"]) for n in m["novelty"])
    # the protocol's own code-sharing field replaces the general one: one code gap, not two
    labels = {g["label"] for g in m["gaps"]}
    assert "code shared: yes" in labels and "code or data shared" not in labels
    for e in m["emerging"]:
        assert e["trend"]["p"] <= C.THRESHOLDS["emerging_max_p"]
        assert e["strength"]["grade"] != "high" or e["trend"]["p"] < 0.05
    assert all(w["trend"]["p"] > C.THRESHOLDS["emerging_max_p"] for w in m["watch"])
    assert "methods: support vector machine" in {w["label"] for w in m["watch"]}
    assert "methods: support vector machine" not in {e["label"] for e in m["emerging"]}


def test_hypotheses_dedupe_and_flag_restatements(ctx, monkeypatch):
    from research_agent.opportunity import agents as A
    from research_agent.tools import claims as Cl

    rows = _synthetic_rows()
    monkeypatch.setattr(Cl, "_rows", lambda _ctx: rows)
    gap = {"id": "G1", "field": "data_modalities", "value": "mobile phone data|call records", "label": "x"}
    monkeypatch.setattr(A, "_map", lambda _ctx: {"gaps": [gap], "novelty": [], "emerging": [], "established": []})
    pred = {"field": "methods", "any_of": ["random forest"], "min_share": 0.4}
    first = A.test_hypothesis(ctx, "G1", "Random forest appears in 30 of 60 papers.", "prevalence", pred)
    again = A.test_hypothesis(ctx, "G1", "Random forests are used in 30 of 60 papers.", "prevalence", dict(pred))
    assert first["hypothesis_id"] == again["hypothesis_id"] == "H1" and "already" in again["note"]
    same = A.test_hypothesis(ctx, "G1", "Mobile phone data appears in 10 of 60 papers.", "prevalence",
                             {"field": "data_modalities", "any_of": ["mobile phone data"], "max_share": 0.2})
    assert same["hypothesis_id"] == "H2" and "re-measures" in same["note"]
    items = ctx.notes()["hypotheses"]["items"]
    assert [h["restates_item"] for h in items] == [False, True]


def test_topic_scoped_trends_and_absence_checks(ctx, monkeypatch):
    from research_agent.tools import claims as C
    from research_agent.tools.trends import topic_series

    scoped = topic_series(ctx, "rainfall", within="malaria")
    plain = topic_series(ctx, "malaria rainfall")
    assert "within" in scoped and all("pct_of_topic" in y and "per_10k_arxiv" not in y for y in scoped["series"])
    assert sum(y["matching"] for y in scoped["series"]) == sum(y["matching"] for y in plain["series"]) > 0
    assert all(y["matching"] <= y["topic_total"] for y in scoped["series"])
    # a claim about "malaria papers" must be measured within malaria papers
    pred = {"keywords": "rainfall", "early": [2015, 2017], "late": [2021, 2023], "direction": "increase"}
    refused = C.propose_claim(ctx, "Rainfall covariates rose in malaria papers.", "trend", dict(pred))
    assert "within" in refused["error"]
    assert "error" not in C.test_claim(ctx, "trend", {**pred, "within": "malaria"})

    # an absence claim on a free-text field is checked against the papers' own abstracts
    ids = [r["paper_id"] for r in ctx.pg.execute(
        "SELECT paper_id FROM papers WHERE tsv @@ plainto_tsquery('english', 'malaria rainfall') LIMIT 12").fetchall()]
    assert len(ids) >= 5
    rows = [{"paper_id": pid, "source": "abstract", "corpus": "arxiv", "year": 2020,
             "data": {"data_modalities": ["surveillance counts"]}} for pid in ids]
    monkeypatch.setattr(C, "_rows", lambda _ctx: rows)
    absent = C.evaluate_prevalence(ctx, {"field": "data_modalities", "any_of": ["rainfall"], "max_count": 0})
    assert absent["n_matching"] == 0 and not absent["supported"] and "not established" in absent["absence_problem"]
    # with re-checks off, proposing the absence is refused outright
    import dataclasses
    monkeypatch.setattr(C, "settings", dataclasses.replace(C.settings, recheck=False))
    refused = C.propose_claim(ctx, "No paper uses rainfall data (0 of %d)." % len(ids), "prevalence",
                              {"field": "data_modalities", "any_of": ["rainfall"], "max_count": 0})
    assert "not established" in refused["error"]
    # a term the abstracts do not mention stays a valid rarity claim
    fine = C.evaluate_prevalence(ctx, {"field": "data_modalities", "any_of": ["mobile phone"], "max_count": 0})
    assert fine["supported"] and "absence_problem" not in fine


def test_number_check_backs_corrects_and_drops(ctx, monkeypatch):
    from research_agent.agents import number_check as NC
    from research_agent.agents.report import allowed_counts
    from research_agent.tools import claims as C

    rows = [{"paper_id": f"p{i}", "source": "fulltext" if i < 10 else "abstract", "corpus": "pmc", "year": 2020,
             "title": "t", "data": {"methods": ["random forest"] if i < 7 else ["ARIMA"],
                                    "code_or_data_available": "yes" if i < 3 else "no"}} for i in range(40)]
    monkeypatch.setattr(C, "_rows", lambda _ctx: rows)
    body = ("## Summary\n"
            "- Random forests appear in 7 of 40 papers (18%) [PMC1].\n"
            "- Code is shared by 5 of 40 papers (13%).\n"
            "- Method families cover 34 of 40 papers across several groups.\n"
            "- Tree models appear in 9 of 40 papers.\n"
            "| C1 | row 3 of 40 | supported | x |\n")
    items = NC.unverified_items(body, allowed_counts(ctx))
    assert [it["number"] for it in items] == ["7 of 40", "5 of 40", "34 of 40", "9 of 40"]   # tables skipped
    assert items[0]["sentence"].startswith("Random forests")                                  # no bullet
    rf = C.propose_claim(ctx, "Random forests appear in 7 of 40 papers.", "prevalence",
                         {"field": "methods", "any_of": ["random forest"], "min_count": 5})
    code = C.propose_claim(ctx, "Code is shared by 3 of 40 papers.", "prevalence",
                           {"field": "code_or_data_available", "any_of": ["yes"], "max_share": 0.1})
    ft = C.propose_claim(ctx, "Random forests appear in 7 of 10 full-text papers.", "prevalence",
                         {"field": "methods", "any_of": ["random forest"], "min_count": 5,
                          "where": [{"field": "read", "any_of": ["fulltext"]}]})
    fixes = [{"item": "U1", "action": "measure", "claim_id": rf["claim_id"]},
             {"item": "U2", "action": "measure", "claim_id": code["claim_id"]},
             {"item": "U3", "action": "drop", "sentence": "Method families cover most papers across several groups."},
             {"item": "U4", "action": "measure", "claim_id": ft["claim_id"]}]
    out, stats = NC.apply_fixes(ctx, body, items, fixes)
    assert f"7 of 40 papers (18%) [PMC1] [C{rf['claim_id']}]." in out    # backed: same number, now cited
    assert f"Code is shared by 3 of 40 papers (8%) [C{code['claim_id']}]." in out   # corrected, % follows
    assert "- Method families cover 34 of 40" in out                    # 'most' is an uncounted quantity: refused
    assert "9 of 40" in out                                              # claim counted out of 10: refused
    assert (stats["backed"], stats["corrected"], stats["removed"], stats["refused"]) == (1, 1, 0, 2)
    # a faithful drop is accepted, and the bullet stays
    out2, stats2 = NC.apply_fixes(ctx, body, items, [
        {"item": "U3", "action": "drop", "sentence": "Method families span several groups (not counted per paper)."}])
    assert "- Method families span several groups (not counted per paper)." in out2 and stats2["removed"] == 1
    # dropping may not lose a citation or add numbers
    assert NC._drop_ok("X in 7 of 40 papers [PMC1].", "X in some papers.", "7 of 40")
    assert NC._drop_ok("X in 7 of 40 papers.", "X in 12 papers.", "7 of 40")


def test_recheck_confirms_uses_the_extraction_missed(ctx, monkeypatch):
    from research_agent.tools import claims as C
    from research_agent.tools import extraction as X
    from research_agent.tools import recheck as R

    ids = [r["paper_id"] for r in ctx.pg.execute(
        "SELECT paper_id FROM papers WHERE tsv @@ plainto_tsquery('english', 'malaria rainfall') LIMIT 8").fetchall()]
    base = [{"paper_id": pid, "source": "abstract", "corpus": "arxiv", "year": 2020, "title": "t",
             "data": {"data_modalities": ["surveillance counts"]}} for pid in ids]

    def rows(_ctx):   # the real merge of confirmed re-checks, on these rows
        out = [{**r, "data": dict(r["data"])} for r in base]
        X._merge_rechecks(_ctx, out)
        return out
    monkeypatch.setattr(C, "_rows", rows)
    before = C.evaluate_prevalence(ctx, {"field": "data_modalities", "any_of": ["rainfall"], "max_count": 0})
    assert before["n_matching"] == 0 and before.get("absence_problem")
    # proposing "no paper uses rainfall" triggers the re-check, which finds every paper does
    res = C.propose_claim(ctx, "No paper uses rainfall data (0 of %d)." % len(ids), "prevalence",
                          {"field": "data_modalities", "any_of": ["rainfall"], "max_count": 0})
    assert f"{len(ids)} of {len(ids)}" in (res.get("measured") or res.get("measured_now"))
    if "claim_id" in res:     # accepted as pending, it fails verification: its bound (none) is not met
        C.verify_claims(ctx, [res["claim_id"]])
        assert ctx.pg.execute("SELECT status FROM claims WHERE id=%s", (res["claim_id"],)).fetchone()["status"] \
            == "unsupported"
    stored = ctx.pg.execute("SELECT verdict, quote FROM rechecks WHERE run_id=%s", (ctx.run_id,)).fetchall()
    assert len(stored) == len(ids) and all(s["verdict"] == "yes" and "rainfall" in s["quote"] for s in stored)
    after = C.evaluate_prevalence(ctx, {"field": "data_modalities", "any_of": ["rainfall"], "min_count": 1})
    assert after["n_matching"] == len(ids) and after["supported"] and "absence_problem" not in after
    assert all("rainfall" in r["data"]["_rechecked"]["data_modalities"] for r in rows(ctx))
    # a second call re-checks nobody: answers are cached per paper and concept
    assert R.recheck(ctx, "data_modalities", ["rainfall"])["checked_now"] == 0
    assert "list fields" in R.recheck(ctx, "validation_level", ["external"])["error"]


def test_recheck_answers_need_a_real_quote():
    from research_agent.llm.base import LLMResponse, ToolCall
    from research_agent.tools.recheck import _ask

    text = ("Title: Forecasting malaria\n\nAbstract: Earlier studies used transformer models. We fit an "
            "ARIMA model to monthly case counts.")

    class Stub:
        def __init__(self, args):
            self.args = args

        def chat(self, *a, **k):
            tc = ToolCall("1", "record_check", self.args)
            return LLMResponse("", [tc], [], "tool_use", {})

    terms = ["transformer"]
    invented = _ask(Stub({"uses": "yes", "which": "transformer", "quote": "we train a transformer network"}),
                    "transformers", terms, {}, text)
    assert invented["verdict"] == "unclear" and not invented["quote"]      # quote is not in the paper
    real = _ask(Stub({"uses": "yes", "which": "transformer",
                      "quote": "Earlier studies used transformer models."}), "transformers", terms, {}, text)
    assert real["verdict"] == "yes"
    assert _ask(Stub({"uses": "maybe"}), "transformers", terms, {}, text)["verdict"] == "unclear"


def test_coverage_probe_finds_what_discovery_missed(ctx):
    from research_agent.tools.search import add_to_shortlist, coverage_probe

    malaria = [r["paper_id"] for r in ctx.pg.execute(
        "SELECT paper_id FROM papers WHERE tsv @@ plainto_tsquery('english', 'malaria') ORDER BY paper_id").fetchall()]
    add_to_shortlist(ctx, malaria[:1], "seed")
    res = coverage_probe(ctx, "malaria", ["rainfall", "(sepsis)"])
    rain, sepsis = res["probes"]
    assert res["topic_papers_in_corpus"] >= rain["corpus_papers"] > 1
    assert rain["on_shortlist"] <= 1 and rain["under_covered"] == (rain["corpus_papers"] >= 5)
    assert all(p["paper_id"] not in malaria[:1] for p in rain["not_shortlisted_examples"])
    assert sepsis["corpus_papers"] == 0 and not sepsis["under_covered"]
    # the corpus counts are recorded, so a report may quote them
    assert ctx.pg.execute("SELECT count(*) n FROM run_events WHERE run_id=%s AND agent='coverage_probe' "
                          "AND kind='count'", (ctx.run_id,)).fetchone()["n"] == 2


def test_trimming_is_done_once_and_kept():
    from research_agent.agents.base import _msg_chars, compact

    msgs = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
    for i in range(12):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "call"}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": str(i), "content": "x" * 3000}]})
    budget = 20000
    trimmed = compact(msgs, int(budget * 0.6))
    assert _msg_chars(trimmed) <= budget * 0.6
    # adding one more turn keeps it under budget, so the trimmed start (the cached prefix) is not touched again
    trimmed.append({"role": "assistant", "content": [{"type": "text", "text": "next"}]})
    assert _msg_chars(trimmed) <= budget and compact(trimmed, budget) is trimmed


def test_usage_is_broken_down_by_step(loaded_db, monkeypatch):
    import dataclasses

    from research_agent import runstate
    from research_agent.agents.orchestrator import run_research
    from research_agent.cli import format_usage

    monkeypatch.setattr(runstate, "settings", dataclasses.replace(
        runstate.settings, price_input_per_m=1.0, price_cached_input_per_m=0.1, price_output_per_m=2.0))
    res = run_research("What ML methods are used for malaria forecasting?", mode="pipeline",
                       llm_factory=lambda strong=False: FakeLLM())
    steps = res["usage_by_step"]
    assert {"discovery", "extraction", "synthesis", "protocol_fields"} <= set(steps)
    assert sum(s["llm_calls"] for s in steps.values()) == res["usage"]["llm_calls"]
    assert abs(sum(s["cost_usd"] for s in steps.values()) - res["usage"]["cost_usd"]) < 0.01
    assert "cost $" in format_usage(steps, res["usage"])


def _reading_stub(script):
    from research_agent.llm.base import LLMResponse, ToolCall

    class Stub:
        model = "stub"
        calls = []

        def chat(self, system, messages, tools=None, force_tool=None, max_tokens=0, temperature=0.0):
            Stub.calls.append(force_tool)
            args = script(force_tool, messages[0]["content"][0]["text"], tools)
            return LLMResponse("", [ToolCall(str(len(Stub.calls)), force_tool, args)], [], "tool_use", {})
    return Stub


def test_one_read_per_paper_and_batched_abstracts(ctx, monkeypatch):
    import dataclasses

    from research_agent.tools import reading as RD
    from tests.fake_llm import fake_reading

    protocol = {"fields": [{"name": "q_probabilistic", "type": "enum", "values": ["yes", "no", "not_stated"],
                            "definition": "Gives intervals.", "desirable": ["yes"]}]}
    papers = [{"paper_id": f"f{i}", "title": f"Paper f{i}", "abstract": "We forecast monthly malaria cases using "
               "an LSTM driven by rainfall.", "fulltext": "## Methods\nWe used surveillance counts from Kenya."}
              for i in range(3)]
    papers += [{"paper_id": f"a{i}", "title": f"Paper a{i}", "abstract": "We forecast monthly malaria cases "
                "using ARIMA in Uganda.", "fulltext": None} for i in range(7)]

    def script(tool, text, tools):
        if tool == "record_readings":
            blocks = re.split(r"^=== Paper (\S+) ===\n", text, flags=re.M)[1:]
            return {"records": [{"paper_id": pid, **fake_reading(body, tools)} for pid, body in zip(blocks[::2], blocks[1::2])]}
        return fake_reading(text, tools)
    Stub = _reading_stub(script)
    monkeypatch.setattr(RD, "settings", dataclasses.replace(RD.settings, abstract_batch=5))
    ctx.llm_factory = lambda strong=False: Stub()
    results, failed, _ = RD.read_papers(ctx, papers, protocol)
    # 3 full-text papers alone + 7 abstracts in batches of 5 and 2 = 5 calls, each covering BOTH sets of fields
    assert sorted(Stub.calls) == ["record_reading"] * 3 + ["record_readings"] * 2 and not failed
    assert set(results) == {p["paper_id"] for p in papers}
    assert all(r["base"] is not None and r["protocol"] is not None for r in results.values())
    assert results["f0"]["source"] == "fulltext" and results["a0"]["source"] == "abstract"
    assert "Kenya" in results["f0"]["base"]["geography"]            # read from the full-text section
    assert "Uganda" in results["a3"]["base"]["geography"]


def test_batched_reads_cannot_borrow_quotes_and_recover_missing_papers(ctx):
    import re as _re

    from research_agent.tools import reading as RD

    papers = [{"paper_id": "a1", "title": "One", "abstract": "We used ARIMA models on clinic data from Ghana.",
               "fulltext": None},
              {"paper_id": "a2", "title": "Two", "abstract": "A survey of bed net use in Malawi households.",
               "fulltext": None},
              {"paper_id": "a3", "title": "Three", "abstract": "Rainfall predicted cases in Zambia districts.",
               "fulltext": None}]

    def script(tool, text, tools):
        if tool == "record_readings":
            # a2's record borrows a1's sentence as evidence; a3 is left out altogether
            return {"records": [
                {"paper_id": "a1", "methods": ["ARIMA"], "geography": ["Ghana"],
                 "evidence": {"methods": ["We used ARIMA models on clinic data"], "geography": ["clinic data from Ghana"]}},
                {"paper_id": "a2", "methods": ["ARIMA"], "geography": ["Malawi"],
                 "evidence": {"methods": ["We used ARIMA models on clinic data"],
                              "geography": ["bed net use in Malawi households"]}}]}
        assert "Zambia" in text and "Ghana" not in text                  # the fallback reads a3 alone
        return {"geography": ["Zambia"], "evidence": {"geography": ["Rainfall predicted cases in Zambia districts"]}}
    Stub = _reading_stub(script)
    ctx.llm_factory = lambda strong=False: Stub()
    results, failed, _ = RD.read_papers(ctx, papers, None)
    assert results["a1"]["base"]["methods"] == ["ARIMA"]
    assert results["a2"]["base"]["methods"] == [] and results["a2"]["base"]["geography"] == ["Malawi"]
    assert results["a3"]["base"]["geography"] == ["Zambia"] and Stub.calls.count("record_reading") == 1
    _ = _re


def test_cut_off_batch_keeps_finished_records(ctx):
    import json as _json

    from research_agent.tools import reading as RD

    papers = [{"paper_id": f"b{i}", "title": f"T{i}", "abstract": f"Study {i} used ARIMA in Ghana.", "fulltext": None}
              for i in range(3)]
    done = {"paper_id": "b0", "methods": ["ARIMA"], "evidence": {"methods": ["Study 0 used ARIMA in Ghana"]}}

    def script(tool, text, tools):
        if tool == "record_readings":
            raw = _json.dumps({"records": [done]})[:-2] + ', {"paper_id": "b1", "meth'
            return {"_raw_arguments": raw}
        return {"methods": [], "evidence": {}}
    Stub = _reading_stub(script)
    ctx.llm_factory = lambda strong=False: Stub()
    results, failed, _ = RD.read_papers(ctx, papers, None)
    assert results["b0"]["base"]["methods"] == ["ARIMA"]                 # kept from the cut-off batch
    assert Stub.calls.count("record_reading") == 2 and set(results) == {"b0", "b1", "b2"}
