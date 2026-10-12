"""Every setting for the pipeline, in one place.

Each trace stores a snapshot of this config (plus a short hash of it), so two
runs can be compared later and we always know which settings produced which
numbers. Change values here, not inside the modules.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEARNING_AI_ROOT = PROJECT_ROOT.parent.parent


@dataclass(frozen=True)
class Config:
    # --- services ---
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    # Which engine answers: "ollama" (default) or "llama_server" (llama.cpp, for the
    # inference experiments; start it with scripts/llama-server.sh).
    llm_backend: str = os.environ.get("RAG_LLM_BACKEND", "ollama")
    llama_server_url: str = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8081")

    # --- models ---
    llm_model: str = "llama3.1:8b"
    rewrite_model: str = "llama3.1:8b"   # Llama 3.1 has no small size; llama3.2:3b is a faster option
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    embed_num_ctx: int = 2048          # longer input is an error, not silently cut (truncate=False)
    # nomic-embed-text was trained with these task prefixes; without them search quality drops.
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- corpus: name -> folder. doc_id = "<name>/<path inside folder>" ---
    corpus_roots: dict[str, Path] = field(default_factory=lambda: {
        "notes": LEARNING_AI_ROOT / "all-notes-files",
        "docs": PROJECT_ROOT / "docs",
    })
    corpus_extensions: tuple[str, ...] = (".md", ".pdf")
    # Index tables (00-INDEX*.md) are progress trackers, not knowledge.
    corpus_exclude_globs: tuple[str, ...] = ("00-INDEX*",)

    # --- storage ---
    db_path: Path = PROJECT_ROOT / "data" / "rag.db"
    qdrant_collection: str = "chunks"
    traces_dir: Path = PROJECT_ROOT / "traces"
    results_dir: Path = PROJECT_ROOT / "results"

    # --- ingestion ---
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50
    chars_per_token: int = 4          # rough estimate; good enough for sizing chunks
    embed_batch_size: int = 32
    max_attempts: int = 3             # retries for one document before it is marked failed

    # --- retrieval ---
    vector_top_k: int = 30
    bm25_top_k: int = 30
    rrf_k: int = 60
    rerank_candidates: int = 30       # how many hybrid hits the cross-encoder re-scores
    rerank_top_n: int = 6
    rerank_device: str = "auto"       # auto = Apple GPU (mps) if present, else cpu
    # Re-rank top score below this -> "not found". Phase 3 eval: answerable >= -2.65,
    # unanswerable <= -9.18; -6.0 sits mid-gap. Only 4 (very off-topic) unanswerable
    # questions so far, so re-check when harder ones are added.
    not_found_threshold: float = -6.0

    # --- generation ---
    context_token_budget: int = 3500
    num_ctx: int = 8192               # set explicitly: Ollama silently cuts prompts longer than its default
    temperature: float = 0.0
    max_output_tokens: int = 600
    keep_alive: str = "30m"           # keep models loaded between questions (avoids cold starts)
    history_turns: int = 6

    def snapshot(self) -> dict:
        """Plain-JSON view of the config, stored in every trace."""
        out = {}
        for k, v in asdict(self).items():
            if isinstance(v, Path):
                v = str(v)
            elif isinstance(v, dict):
                v = {kk: str(vv) for kk, vv in v.items()}
            elif isinstance(v, tuple):
                v = list(v)
            out[k] = v
        return out


CONFIG = Config()
