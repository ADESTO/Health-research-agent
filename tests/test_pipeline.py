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
    assert run["status"] == "done" and run["llm_calls"] > 10

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
