"""Postgres schema + logging helpers for monitoring: conversations and feedback.

Uses the `monitoring` schema in the same shared Postgres container that dlt's
ingestion pipeline uses (separate `raw` schema there — see PLAN.md's "one shared
Postgres container" decision). Grafana's dashboard (monitoring/grafana/) reads
straight from this schema with plain SQL.
"""

import json
import logging
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCHEMA = "monitoring"


def get_connection_params() -> dict:
    load_dotenv()
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "dbname": os.environ.get("POSTGRES_DB", "sec_rag"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    }


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(**get_connection_params())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the monitoring schema/tables if they don't already exist. Safe to
    call on every app startup — CREATE ... IF NOT EXISTS throughout."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.conversations (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                method TEXT NOT NULL,
                use_rerank BOOLEAN NOT NULL,
                use_query_rewrite BOOLEAN NOT NULL,
                filter_dict JSONB,
                latency_ms INTEGER,
                judge_faithfulness REAL,
                judge_context_precision REAL,
                judge_relevance TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        # Columns added after the initial schema — for an already-existing table
        # (e.g. a prior container run), these ADD COLUMN IF NOT EXISTS cover it
        # since CREATE TABLE IF NOT EXISTS above is a no-op then. judge_score was
        # the original single blended score, superseded by the three columns below
        # (see rag/judge.py) — dropped since it's no longer written.
        cur.execute(f"ALTER TABLE {SCHEMA}.conversations DROP COLUMN IF EXISTS judge_score")
        cur.execute(f"ALTER TABLE {SCHEMA}.conversations ADD COLUMN IF NOT EXISTS judge_faithfulness REAL")
        cur.execute(f"ALTER TABLE {SCHEMA}.conversations ADD COLUMN IF NOT EXISTS judge_context_precision REAL")
        cur.execute(f"ALTER TABLE {SCHEMA}.conversations ADD COLUMN IF NOT EXISTS judge_relevance TEXT")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.feedback (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES {SCHEMA}.conversations(id),
                is_positive BOOLEAN NOT NULL,
                comment TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute(f"ALTER TABLE {SCHEMA}.feedback ADD COLUMN IF NOT EXISTS comment TEXT")
    log.info("Monitoring schema/tables ready")


def log_conversation(
    question: str,
    answer: str,
    method: str,
    use_rerank: bool,
    use_query_rewrite: bool,
    filter_dict: dict | None,
    latency_ms: int,
    judge_faithfulness: float | None = None,
    judge_context_precision: float | None = None,
    judge_relevance: str | None = None,
) -> int:
    """Insert a conversation row, return its id (used to link feedback to it later).
    The judge_* fields are None unless the caller opted into live LLM-as-judge
    scoring (see rag/judge.py, app.py's "Score this answer" toggle) — most
    conversations won't have them, and that's expected, not an error.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.conversations
                (question, answer, method, use_rerank, use_query_rewrite, filter_dict, latency_ms,
                 judge_faithfulness, judge_context_precision, judge_relevance)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (question, answer, method, use_rerank, use_query_rewrite,
             json.dumps(filter_dict) if filter_dict else None, latency_ms,
             judge_faithfulness, judge_context_precision, judge_relevance),
        )
        return cur.fetchone()[0]


def log_feedback(conversation_id: int, is_positive: bool, comment: str | None = None) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {SCHEMA}.feedback (conversation_id, is_positive, comment) VALUES (%s, %s, %s)",
            (conversation_id, is_positive, comment or None),
        )
