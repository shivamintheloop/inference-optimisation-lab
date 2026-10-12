from pathlib import Path

from rag.ingest import SourceFile, plan_jobs


def f(doc_id, h):
    return SourceFile(doc_id, Path("/x") / doc_id, h)


def row(status="ok", content_hash="h1", failed_hash=None):
    return {"status": status, "content_hash": content_hash, "failed_hash": failed_hash}


ROOTS = ["notes", "docs"]


def test_new_changed_unchanged_deleted():
    files = [f("notes/a.md", "h1"), f("notes/b.md", "h2-new"), f("notes/c.md", "h3")]
    known = {"notes/a.md": row(content_hash="h1"),
             "notes/b.md": row(content_hash="h2-old"),
             "notes/gone.md": row()}
    jobs = plan_jobs(files, known, ROOTS)
    assert [x.doc_id for x in jobs.new] == ["notes/c.md"]
    assert [x.doc_id for x in jobs.changed] == ["notes/b.md"]
    assert jobs.unchanged == ["notes/a.md"]
    assert jobs.deleted == ["notes/gone.md"]


def test_failed_version_is_skipped_unless_retry():
    files = [f("docs/bad.pdf", "h9")]
    known = {"docs/bad.pdf": row(status="failed", content_hash=None, failed_hash="h9")}
    assert plan_jobs(files, known, ROOTS).still_failed == ["docs/bad.pdf"]
    retry = plan_jobs(files, known, ROOTS, retry_failed=True)
    assert [x.doc_id for x in retry.new] == ["docs/bad.pdf"]   # never indexed -> "new"


def test_failed_file_that_was_edited_is_tried_again():
    files = [f("docs/bad.pdf", "h10-fixed")]
    known = {"docs/bad.pdf": row(status="failed", content_hash="h8", failed_hash="h9")}
    assert [x.doc_id for x in plan_jobs(files, known, ROOTS).changed] == ["docs/bad.pdf"]


def test_force_reindexes_unchanged():
    files = [f("notes/a.md", "h1")]
    jobs = plan_jobs(files, {"notes/a.md": row()}, ROOTS, force=True)
    assert [x.doc_id for x in jobs.changed] == ["notes/a.md"]


def test_missing_root_does_not_delete_its_documents():
    known = {"docs/paper.pdf": row(), "notes/a.md": row()}
    jobs = plan_jobs([f("notes/a.md", "h1")], known, scanned_roots=["notes"])
    assert jobs.deleted == []
