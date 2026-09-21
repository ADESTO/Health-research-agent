"""Test setup: a real Postgres+pgvector test database, fixture parquet, offline hashing embedder.

Set TEST_DATABASE_URL to point at a disposable database (it is wiped)."""
from __future__ import annotations

import os
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "data"

os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL",
                                       "postgresql://postgres@localhost:5433/research_test")
os.environ["EMBEDDER"] = "hashing"
os.environ["METADATA_GLOB"] = str(FIXTURES / "metadata" / "*.parquet")
os.environ["PAPER_TEXT_GLOB"] = str(FIXTURES / "paper_text" / "*.parquet")
os.environ["MIN_YEAR"] = "2015"
os.environ["MAX_FULLTEXT"] = "5"
os.environ["RA_ENV_FILE"] = "/nonexistent"

import psycopg  # noqa: E402
import pytest  # noqa: E402

from tests.fixtures.make_fixtures import build  # noqa: E402


@pytest.fixture(scope="session")
def loaded_db():
    build(FIXTURES)
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    from research_agent.ingestion.load import run_ingest

    stats = run_ingest(log=lambda *_: None)
    return stats
