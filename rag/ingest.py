"""Ingestion: scan -> decide -> parse -> chunk -> embed -> store.

Run by hand as the "nightly scan" (`python -m rag ingest`). For every file
in the corpus folders it compares the file's sha256 with DocumentsDB:

    new        not seen before                   -> index it
    changed    hash differs (or --force)         -> index it, old chunks removed
    unchanged  same hash, last run ok            -> skip, no work
    deleted    in DocumentsDB but not on disk    -> remove its chunks and row
    failed     this exact version failed before  -> skip unless --retry-failed

Store order for one document (the hash is saved LAST):
    1. upsert new chunks to Qdrant
    2. delete the document's other (old) points
    3. replace its rows in the keyword index
    4. save the new hash + status=ok
If anything fails before step 4, the old hash stays, so the next run sees the
file as changed and redoes it. Steps 1-3 are safe to repeat.
"""
from __future__ import annotations

import fnmatch
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from rag import docs_db, ollama_client, vector_store
from rag.chunk import Chunk, chunk_document
from rag.config import CONFIG
from rag.parse import ParsedDoc, PermanentError, parse_file
from rag.trace import Trace, span, trace


@dataclass
class SourceFile:
    doc_id: str                 # "<corpus name>/<path inside that folder>"
    path: Path
    content_hash: str


@dataclass
class Jobs:
    new: list[SourceFile] = field(default_factory=list)
    changed: list[SourceFile] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    still_failed: list[str] = field(default_factory=list)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan() -> tuple[list[SourceFile], list[str]]:
    """Find corpus files. Also returns which corpus roots were actually
    scanned: a missing folder must not look like 'every file was deleted'."""
    files, scanned = [], []
    for name, root in CONFIG.corpus_roots.items():
        if not root.is_dir():
            continue
        scanned.append(name)
        for path in sorted(root.rglob("*")):
            if (path.is_file() and path.suffix.lower() in CONFIG.corpus_extensions
                    and not any(fnmatch.fnmatch(path.name, g) for g in CONFIG.corpus_exclude_globs)):
                doc_id = f"{name}/{path.relative_to(root).as_posix()}"
                files.append(SourceFile(doc_id, path, file_hash(path)))
    return files, scanned


def plan_jobs(files: list[SourceFile], known: Mapping[str, Mapping], scanned_roots: list[str],
              retry_failed: bool = False, force: bool = False) -> Jobs:
    jobs = Jobs()
    seen = set()
    for f in files:
        seen.add(f.doc_id)
        row = known.get(f.doc_id)
        if row is None:
            jobs.new.append(f)
        elif (row["status"] == "failed" and row["failed_hash"] == f.content_hash
              and not retry_failed):
            jobs.still_failed.append(f.doc_id)
        elif row["status"] == "ok" and row["content_hash"] == f.content_hash and not force:
            jobs.unchanged.append(f.doc_id)
        elif row["content_hash"] is None:
            jobs.new.append(f)                  # only ever failed; never indexed
        else:
            jobs.changed.append(f)
    jobs.deleted = [d for d in known
                    if d not in seen and d.split("/", 1)[0] in scanned_roots]
    return jobs


# ---------------------------------------------------------------- one document

def _embed(chunks: list[Chunk]) -> list[list[float]]:
    bs = CONFIG.embed_batch_size
    with span("embed", model=CONFIG.embed_model, batch_size=bs) as s:
        t0 = time.perf_counter()
        vectors: list[list[float]] = []
        tokens = load_ms = 0
        for i in range(0, len(chunks), bs):
            texts = [CONFIG.embed_doc_prefix + c.embed_text() for c in chunks[i:i + bs]]
            vecs, timings = ollama_client.embed(texts)
            vectors.extend(vecs)
            tokens += timings.get("prompt_eval_count", 0)
            load_ms += timings.get("load_ms", 0)
        secs = time.perf_counter() - t0
        if any(len(v) != CONFIG.embed_dim for v in vectors):
            raise RuntimeError(f"embedding size is not {CONFIG.embed_dim}")
        est = sum(len(c.embed_text()) for c in chunks) // CONFIG.chars_per_token
        s.set("chunks", len(chunks))
        s.set("batches", -(-len(chunks) // bs))
        s.set("tokens", tokens)                 # real count from Ollama
        s.set("est_tokens", est)                # our chars/4 guess, to check it
        s.set("tokens_per_s", round(tokens / secs) if secs else None)
        s.set("load_ms", round(load_ms, 1))
    return vectors


def _store(conn, f: SourceFile, doc: ParsedDoc, chunks: list[Chunk],
           vectors: list[list[float]], attempts: int) -> None:
    updated_at = datetime.fromtimestamp(f.path.stat().st_mtime, timezone.utc).isoformat(
        timespec="seconds")
    ids = [vector_store.chunk_id(f.doc_id, f.content_hash, c.index) for c in chunks]
    payloads = [{
        "chunk_id": cid, "doc_id": f.doc_id, "title": doc.title, "section": c.section,
        "page": c.page, "chunk_index": c.index, "char_start": c.char_start,
        "char_end": c.char_end, "source_path": str(f.path), "updated_at": updated_at,
        "content_hash": f.content_hash, "embedding_model": CONFIG.embed_model, "text": c.text,
    } for cid, c in zip(ids, chunks)]

    with span("store", points=len(ids)):
        with span("store.qdrant_upsert"):
            vector_store.upsert(ids, vectors, payloads)
        with span("store.qdrant_delete_old"):
            vector_store.delete_others(f.doc_id, keep_ids=ids)
        with span("store.fts_replace"):
            docs_db.replace_fts(conn, f.doc_id, [(cid, c.embed_text()) for cid, c in zip(ids, chunks)])
        with span("store.save_hash"):
            docs_db.mark_ok(conn, f.doc_id, str(f.path), f.content_hash, updated_at,
                            len(chunks), CONFIG.embed_model, attempts)


def index_document(conn, f: SourceFile, kind: str) -> str | None:
    """Index one file. Returns None on success, else the error message."""
    with span("ingest.doc", doc_id=f.doc_id, kind=kind) as s:
        # Parse/chunk problems come from the file itself: no point retrying.
        try:
            with span("parse") as ps:
                doc = parse_file(f.path)
                ps.set("sections", len(doc.sections))
                ps.set("chars", len(doc.text))
                if doc.pages is not None:
                    ps.set("pages", doc.pages)
                    ps.set("empty_pages", doc.empty_pages)
            with span("chunk") as cs:
                chunks = chunk_document(doc)
                cs.set("chunks", len(chunks))
                if chunks:
                    cs.set("avg_est_tokens", round(sum(len(c.text) for c in chunks)
                                                   / len(chunks) / CONFIG.chars_per_token))
            if not chunks:
                raise PermanentError("no text left after parsing")
        except Exception as e:
            msg = e if isinstance(e, PermanentError) else f"{type(e).__name__}: {e}"
            docs_db.mark_failed(conn, f.doc_id, str(f.path), f.content_hash, str(msg), 1)
            s.set("outcome", "failed")
            s.set("error", str(msg))
            return str(msg)

        # Embed/store problems are usually the services (Ollama, Qdrant): retry.
        last_error = ""
        for attempt in range(1, CONFIG.max_attempts + 1):
            try:
                vectors = _embed(chunks)
                _store(conn, f, doc, chunks, vectors, attempt)
                s.set("outcome", "ok")
                s.set("chunks", len(chunks))
                s.set("attempts", attempt)
                return None
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < CONFIG.max_attempts:
                    time.sleep(2 ** (attempt - 1))
        docs_db.mark_failed(conn, f.doc_id, str(f.path), f.content_hash, last_error,
                            CONFIG.max_attempts)
        s.set("outcome", "failed")
        s.set("error", last_error)
        s.set("attempts", CONFIG.max_attempts)
        return last_error


def delete_doc(conn, doc_id: str) -> None:
    with span("ingest.delete", doc_id=doc_id):
        vector_store.delete_document(doc_id)
        docs_db.delete_document(conn, doc_id)   # row goes last, like the hash


# ---------------------------------------------------------------- one run

@dataclass
class IngestResult:
    trace: Trace
    jobs: Jobs
    failed: dict[str, str]                      # doc_id -> error
    chunks_written: int


def ingest(retry_failed: bool = False, force: bool = False) -> IngestResult:
    conn = docs_db.connect()
    failed: dict[str, str] = {}
    chunks_written = 0
    with trace("ingest", config=CONFIG.snapshot()) as t:
        with span("scan") as s:
            files, roots = scan()
            s.set("files", len(files))
            s.set("roots", roots)
        with span("connect"):
            vector_store.ensure_collection()
        jobs = plan_jobs(files, docs_db.all_documents(conn), roots, retry_failed, force)

        for doc_id in jobs.deleted:
            delete_doc(conn, doc_id)
        for kind, group in (("new", jobs.new), ("changed", jobs.changed)):
            for f in group:
                error = index_document(conn, f, kind)
                if error:
                    failed[f.doc_id] = error

        rows = docs_db.all_documents(conn)
        chunks_written = sum(rows[f.doc_id]["chunk_count"] for f in jobs.new + jobs.changed
                             if f.doc_id not in failed)
        for key, value in (("scanned", len(files)), ("new", len(jobs.new)),
                           ("changed", len(jobs.changed)), ("unchanged", len(jobs.unchanged)),
                           ("deleted", len(jobs.deleted)), ("failed", len(failed)),
                           ("still_failed", len(jobs.still_failed)),
                           ("chunks_written", chunks_written)):
            t.set(key, value)
    conn.close()
    return IngestResult(t, jobs, failed, chunks_written)
