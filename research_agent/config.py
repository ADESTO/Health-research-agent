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

    # Run limits (cost guards)
    max_shortlist: int = int(os.getenv("MAX_SHORTLIST", "60"))
    max_fulltext: int = int(os.getenv("MAX_FULLTEXT", "20"))
    extraction_workers: int = int(os.getenv("EXTRACTION_WORKERS", "4"))
    agent_max_turns: int = int(os.getenv("AGENT_MAX_TURNS", "14"))
    orchestrator_max_turns: int = int(os.getenv("ORCHESTRATOR_MAX_TURNS", "16"))

    # Per-request size controls. Defaults suit Claude; for Groq's free tier (8K tokens/minute, and
    # Groq counts max_tokens towards that) see the "Groq free tier" block in .env.example.
    agent_max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    report_max_tokens: int = int(os.getenv("REPORT_MAX_TOKENS", "4096"))
    # after the report is written, an agent backs, corrects or removes numbers code could not trace
    number_check: bool = os.getenv("NUMBER_CHECK", "1") != "0"
    # agents whose final answer is a long structured list (map protocol, gap reasoning, designs)
    long_output_max_tokens: int = int(os.getenv("LONG_OUTPUT_MAX_TOKENS", "8192"))
    # evidence quotes make extractions longer; a cut-off extraction now fails loudly instead of saving blanks
    extraction_max_tokens: int = int(os.getenv("EXTRACTION_MAX_TOKENS", "2000"))
    tool_result_chars: int = int(os.getenv("TOOL_RESULT_CHARS", "9000"))
    # When an agent's conversation exceeds this many characters, older tool results are shortened.
    context_budget_chars: int = int(os.getenv("CONTEXT_BUDGET_CHARS", "60000"))

    # v2 adds evidence quotes; v1 extractions are not reused because their fields were never checked
    extraction_schema_version: str = "health-v2"
    extra: dict = field(default_factory=dict)


settings = Settings()
