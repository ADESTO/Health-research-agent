"""Database helpers: connection factory and schema bootstrap."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from research_agent.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or settings.database_url, row_factory=dict_row, autocommit=True)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # extension not created yet (first run) — init_schema will create it
        pass
    return conn


@contextmanager
def get_conn(url: str | None = None) -> Iterator[psycopg.Connection]:
    conn = connect(url)
    try:
        yield conn
    finally:
        conn.close()


_SCHEMA_LOCK = __import__("threading").Lock()
_SCHEMA_KEY = 7423011        # one Postgres advisory lock: the server, workers and CLI never set up at the same time


def init_schema(url: str | None = None, dim: int | None = None) -> None:
    """Create or upgrade tables and indexes. Serialised within this process and across processes, because
    concurrent DDL on the same tables can deadlock (several web requests arriving together, a worker starting)."""
    sql = SCHEMA_PATH.read_text().replace("{dim}", str(dim or settings.embedding_dim))
    with _SCHEMA_LOCK, psycopg.connect(url or settings.database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_KEY,))
        try:
            conn.execute(sql)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_KEY,))
