"""SQLite: the DocumentsDB plus a full-text index for BM25 keyword search.

documents   one row per source file: which version (content_hash) is
            indexed, and whether the last attempt failed.
chunks_fts  FTS5 table with every chunk's text. SQLite ranks matches with
            BM25 out of the box. The porter tokenizer makes "caching" match
            "cache".

content_hash is only written after a document is fully stored (mark_ok), so
a failure leaves the old hash in place and the next run tries again.
failed_hash remembers which file version failed, so a broken file is not
retried every run unless asked (--retry-failed).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rag.config import CONFIG

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    content_hash    TEXT,               -- version that is indexed (NULL = never indexed)
    updated_at      TEXT,               -- file modified time of that version
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT,
    status          TEXT NOT NULL,      -- ok | failed
    error           TEXT,
    failed_hash     TEXT,               -- version that failed
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    tokenize = 'porter unicode61'
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path = CONFIG.db_path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def all_documents(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {r["doc_id"]: r for r in conn.execute("SELECT * FROM documents")}


def replace_fts(conn: sqlite3.Connection, doc_id: str, rows: list[tuple[str, str]]) -> None:
    """Swap a document's keyword-index rows. rows = [(chunk_id, text)]."""
    with conn:
        conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        conn.executemany("INSERT INTO chunks_fts (text, chunk_id, doc_id) VALUES (?, ?, ?)",
                         [(text, cid, doc_id) for cid, text in rows])


def mark_ok(conn: sqlite3.Connection, doc_id: str, source_path: str, content_hash: str,
            updated_at: str, chunk_count: int, embedding_model: str, attempts: int) -> None:
    with conn:
        conn.execute("""
            INSERT INTO documents (doc_id, source_path, content_hash, updated_at, chunk_count,
                                   embedding_model, status, error, failed_hash, attempts,
                                   last_attempt_at)
            VALUES (?, ?, ?, ?, ?, ?, 'ok', NULL, NULL, ?, ?)
            ON CONFLICT (doc_id) DO UPDATE SET
                source_path = excluded.source_path, content_hash = excluded.content_hash,
                updated_at = excluded.updated_at, chunk_count = excluded.chunk_count,
                embedding_model = excluded.embedding_model, status = 'ok', error = NULL,
                failed_hash = NULL, attempts = excluded.attempts,
                last_attempt_at = excluded.last_attempt_at
        """, (doc_id, source_path, content_hash, updated_at, chunk_count, embedding_model,
              attempts, _now()))


def mark_failed(conn: sqlite3.Connection, doc_id: str, source_path: str, failed_hash: str,
                error: str, attempts: int) -> None:
    """Record a failure. content_hash (the last good version) is left alone."""
    with conn:
        conn.execute("""
            INSERT INTO documents (doc_id, source_path, status, error, failed_hash, attempts,
                                   last_attempt_at)
            VALUES (?, ?, 'failed', ?, ?, ?, ?)
            ON CONFLICT (doc_id) DO UPDATE SET
                source_path = excluded.source_path, status = 'failed', error = excluded.error,
                failed_hash = excluded.failed_hash, attempts = excluded.attempts,
                last_attempt_at = excluded.last_attempt_at
        """, (doc_id, source_path, error[:1000], failed_hash, attempts, _now()))


def delete_document(conn: sqlite3.Connection, doc_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def fts_count(conn: sqlite3.Connection, doc_id: str | None = None) -> int:
    if doc_id is None:
        return conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    return conn.execute("SELECT count(*) FROM chunks_fts WHERE doc_id = ?",
                        (doc_id,)).fetchone()[0]
