"""Central configuration, read from environment variables (and a .env file if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader so we don't need python-dotenv. Existing env vars win."""
    env_path = Path(os.getenv("RA_ENV_FILE", ".env"))
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

HF_DATASET = "hf://datasets/secemp9/arxiv-complete"


@dataclass(frozen=True)
class Settings:
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://research:research@localhost:5432/research"
    )

    # Where the arXiv parquet lives. Either the Hugging Face path (streamed by DuckDB)
    # or a local folder you downloaded with `huggingface-cli download`.
    metadata_glob: str = os.getenv("METADATA_GLOB", f"{HF_DATASET}/metadata/*.parquet")
    paper_text_glob: str = os.getenv("PAPER_TEXT_GLOB", f"{HF_DATASET}/paper_text/*.parquet")

    # Health subset
    min_year: int = int(os.getenv("MIN_YEAR", "2010"))

    # Embeddings: "fastembed" (real model, CPU/ONNX) or "hashing" (offline fallback, tests)
    embedder: str = os.getenv("EMBEDDER", "fastembed")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # LLM provider seam: "anthropic" | "groq" | "deepseek"
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # Optional stronger model just for the orchestrator + synthesis (falls back to the main one)
    anthropic_model_strong: str = os.getenv("ANTHROPIC_MODEL_STRONG", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_model_strong: str = os.getenv("DEEPSEEK_MODEL_STRONG", "")

    # Per-step model routing, e.g. "cheap=deepseek:deepseek-flash; strong=anthropic:claude-sonnet-4-5"
    model_routes: str = os.getenv("MODEL_ROUTES", "")

    # Run limits (cost guards)
    max_shortlist: int = int(os.getenv("MAX_SHORTLIST", "60"))
    max_fulltext: int = int(os.getenv("MAX_FULLTEXT", "20"))
    extraction_workers: int = int(os.getenv("EXTRACTION_WORKERS", "4"))
    # characters of each full text given to the reader (methods, data, results and limitations first)
    fulltext_read_chars: int = int(os.getenv("FULLTEXT_READ_CHARS", "14000"))
    # abstract-only papers read per call (1 = one call per paper)
    abstract_batch: int = int(os.getenv("ABSTRACT_BATCH", "5"))
    agent_max_turns: int = int(os.getenv("AGENT_MAX_TURNS", "14"))
    orchestrator_max_turns: int = int(os.getenv("ORCHESTRATOR_MAX_TURNS", "16"))

    # Per-request size controls. Defaults suit Claude; for Groq's free tier (8K tokens/minute, and
    # Groq counts max_tokens towards that) see the "Groq free tier" block in .env.example.
    agent_max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    report_max_tokens: int = int(os.getenv("REPORT_MAX_TOKENS", "4096"))
    # Model prices in USD per million tokens, to show what each run and step costs (leave 0 to hide).
    # Take them from your provider's pricing page; cached input is usually billed at a fraction.
    price_input_per_m: float = float(os.getenv("PRICE_INPUT_PER_M", "0") or 0)
    price_cached_input_per_m: float = float(os.getenv("PRICE_CACHED_INPUT_PER_M", "0") or 0)
    price_output_per_m: float = float(os.getenv("PRICE_OUTPUT_PER_M", "0") or 0)
    # ask runs also get question-specific fields (the map's protocol step)
    ask_protocol: bool = os.getenv("ASK_PROTOCOL", "1") != "0"
    # job workers inside the web server (0 = none; then start `python -m research_agent.cli worker`)
    embedded_workers: int = int(os.getenv("EMBEDDED_WORKERS", "1"))
    # open-ended researcher: pause between background cycles, cycles one question may use before it is parked,
    # supervisor checkpoint interval, and the similarity to the charter below which a new question needs approval
    research_cycle_gap_seconds: int = int(os.getenv("RESEARCH_CYCLE_GAP_SECONDS", "30"))
    research_question_cycles: int = int(os.getenv("RESEARCH_QUESTION_CYCLES", "3"))
    research_checkpoint_every: int = int(os.getenv("RESEARCH_CHECKPOINT_EVERY", "3"))
    research_scope_min: float = float(os.getenv("RESEARCH_SCOPE_MIN", "0.30"))
    # citation graph from OpenAlex (set OPENALEX_API_KEY, free, for a 10x larger daily allowance)
    citations: bool = os.getenv("CITATIONS", "1") != "0"
    # focused quoted re-checks when a rarity count is contradicted by the papers' own text
    recheck: bool = os.getenv("RECHECK", "1") != "0"
    # after the report is written, an agent backs, corrects or removes numbers code could not trace
    number_check: bool = os.getenv("NUMBER_CHECK", "1") != "0"
    # before the report is written, claims whose state rests on reading depth have their abstract-only
    # papers read in full and are measured again (tools/resolve.py). 0 claims turns it off.
    resolve_evidence: bool = os.getenv("RESOLVE_EVIDENCE", "1") != "0"
    resolve_max_claims: int = int(os.getenv("RESOLVE_MAX_CLAIMS", "4"))
    resolve_papers_per_claim: int = int(os.getenv("RESOLVE_PAPERS_PER_CLAIM", "8"))
    # and a budget for the pass as a whole, so a run's full-text reads stay predictable however many claims
    # come out thin: the same allowance the extraction step gets, unless set otherwise
    # how close in meaning two driver or outcome phrases must be to count as the same thing when papers'
    # statements are set beside each other (tools/corroboration.py); 1 turns meaning-based merging off
    corroboration_similarity: float = float(os.getenv("CORROBORATION_SIMILARITY", "0.88"))
    resolve_max_papers: int = int(os.getenv("RESOLVE_MAX_PAPERS", "0")) or int(os.getenv("MAX_FULLTEXT", "20"))
    # agents whose final answer is a long structured list (map protocol, gap reasoning, designs)
    long_output_max_tokens: int = int(os.getenv("LONG_OUTPUT_MAX_TOKENS", "8192"))
    # evidence quotes make extractions longer; a cut-off extraction now fails loudly instead of saving blanks
    extraction_max_tokens: int = int(os.getenv("EXTRACTION_MAX_TOKENS", "2800"))
    tool_result_chars: int = int(os.getenv("TOOL_RESULT_CHARS", "9000"))
    # When an agent's conversation exceeds this many characters, older tool results are shortened.
    context_budget_chars: int = int(os.getenv("CONTEXT_BUDGET_CHARS", "60000"))

    # v2 adds evidence quotes; v1 extractions are not reused because their fields were never checked
    # v3 adds reported results and associations (what papers found); v2 records are read again when needed
    # v4 widens the form beyond AI papers (study designs, populations, organisms, interventions, mechanisms,
    #    targets, outcomes); runs read under v3 keep their records and their form
    # v5 splits validation_level: a test on a later period (temporal_holdout) and a test in a different
    #    place (external_site) were both "external", so reports called them the same thing and any
    #    comparison of the two mixed them. v4 runs keep the old enum (extraction.enums_for)
    # a count over fewer papers than this decides nothing either way: the claim's state is "uncertain"
    min_evidence_base: int = int(os.getenv("MIN_EVIDENCE_BASE", "10"))
    extraction_schema_version: str = "health-v5"
    extra: dict = field(default_factory=dict)


settings = Settings()
