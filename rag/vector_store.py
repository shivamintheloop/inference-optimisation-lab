"""Qdrant: one collection of chunk vectors, with metadata in each point's payload.

Point IDs are stable (uuid5 of doc_id, content_hash and chunk index), so
re-running the same version overwrites instead of duplicating. After new
chunks are written, every other point of that document is deleted, which
removes the old version even if it had more chunks.
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from rag.config import CONFIG

_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "inference-optimization-lab/chunks")


def chunk_id(doc_id: str, content_hash: str, index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}:{content_hash}:{index}"))


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    return QdrantClient(url=CONFIG.qdrant_url, timeout=30)


def _doc_filter(doc_id: str) -> models.FieldCondition:
    return models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))


def ensure_collection() -> None:
    name = CONFIG.qdrant_collection
    c = client()
    if c.collection_exists(name):
        size = c.get_collection(name).config.params.vectors.size
        if size != CONFIG.embed_dim:
            raise RuntimeError(f"collection {name!r} has {size}-dim vectors but "
                               f"{CONFIG.embed_model} makes {CONFIG.embed_dim}; "
                               f"delete the collection and re-ingest")
        return
    c.create_collection(name, vectors_config=models.VectorParams(
        size=CONFIG.embed_dim, distance=models.Distance.COSINE))
    c.create_payload_index(name, field_name="doc_id",
                           field_schema=models.PayloadSchemaType.KEYWORD)


def upsert(ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
    points = [models.PointStruct(id=i, vector=v, payload=p)
              for i, v, p in zip(ids, vectors, payloads)]
    for start in range(0, len(points), 256):
        client().upsert(CONFIG.qdrant_collection, points=points[start:start + 256], wait=True)


def delete_others(doc_id: str, keep_ids: list[str]) -> None:
    """Delete every point of this document except the ones just written."""
    client().delete(CONFIG.qdrant_collection, wait=True,
                    points_selector=models.FilterSelector(filter=models.Filter(
                        must=[_doc_filter(doc_id)],
                        must_not=[models.HasIdCondition(has_id=keep_ids)])))


def delete_document(doc_id: str) -> None:
    client().delete(CONFIG.qdrant_collection, wait=True,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(must=[_doc_filter(doc_id)])))


def count(doc_id: str | None = None) -> int:
    flt = models.Filter(must=[_doc_filter(doc_id)]) if doc_id else None
    return client().count(CONFIG.qdrant_collection, count_filter=flt, exact=True).count
