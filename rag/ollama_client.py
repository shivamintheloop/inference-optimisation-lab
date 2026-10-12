"""Thin client for Ollama's native HTTP API.

Uses the native API (/api/embed, /api/chat) instead of the OpenAI-compatible
one because only the native API returns server timings: model load time,
prefill (prompt_eval) and decode (eval). Keep this file thin; moving to
llama-server later should only touch this module.
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from rag.config import CONFIG

_client: httpx.Client | None = None


class OllamaError(RuntimeError):
    pass


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=CONFIG.ollama_url,
                               timeout=httpx.Timeout(300.0, connect=5.0))
    return _client


def server_timings(resp: dict) -> dict:
    """Ollama reports durations in nanoseconds; convert to ms and keep counts."""
    out = {}
    for key, name in (("load_duration", "load_ms"),
                      ("prompt_eval_duration", "prompt_eval_ms"),
                      ("eval_duration", "eval_ms"),
                      ("total_duration", "server_total_ms")):
        if key in resp:
            out[name] = round(resp[key] / 1e6, 2)
    for key in ("prompt_eval_count", "eval_count"):
        if key in resp:
            out[key] = resp[key]
    return out


def embed(texts: list[str], model: str | None = None) -> tuple[list[list[float]], dict]:
    """Embed a batch of texts. Returns (vectors, server timings)."""
    r = _http().post("/api/embed", json={
        "model": model or CONFIG.embed_model,
        "input": texts,
        "truncate": False,                      # fail loudly instead of cutting text
        "keep_alive": CONFIG.keep_alive,
        "options": {"num_ctx": CONFIG.embed_num_ctx},
    })
    if r.status_code != 200:
        raise OllamaError(f"embed failed ({r.status_code}): {r.text[:300]}")
    data = r.json()
    return data["embeddings"], server_timings(data)


def chat_stream(messages: list[dict], model: str | None = None,
                max_tokens: int | None = None) -> Iterator[dict]:
    """Stream a chat answer. Yields {"type": "token", "text": ...} for each piece,
    then one {"type": "done", "timings": ..., "done_reason": ...}.
    done_reason "length" means the answer was cut off at max_tokens."""
    body = {
        "model": model or CONFIG.llm_model,
        "messages": messages,
        "stream": True,
        "keep_alive": CONFIG.keep_alive,
        "options": {
            "temperature": CONFIG.temperature,
            "num_ctx": CONFIG.num_ctx,
            "num_predict": max_tokens or CONFIG.max_output_tokens,
        },
    }
    with _http().stream("POST", "/api/chat", json=body) as r:
        if r.status_code != 200:
            r.read()
            raise OllamaError(f"chat failed ({r.status_code}): {r.text[:300]}")
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise OllamaError(data["error"])
            if data.get("done"):
                yield {"type": "done", "timings": server_timings(data),
                       "done_reason": data.get("done_reason")}
            else:
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield {"type": "token", "text": piece}
