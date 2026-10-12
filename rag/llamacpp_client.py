"""Thin client for llama-server (llama.cpp), used for the inference experiments.

Same interface as ollama_client.chat_stream, and its timings are renamed to
Ollama's keys, so generate() and the traces don't care which engine ran.

Why a second engine: Ollama has no speculative decoding for Llama 3.1, while
llama-server does (a small draft model, or n-gram drafts copied from the
prompt). Start a server with scripts/llama-server.sh and set
RAG_LLM_BACKEND=llama_server.

Draft statistics are not in the per-request timings; llama-server exposes them
as running totals on /metrics (needs --metrics). Requests in the benchmark run
one at a time, so "after minus before" is one request's numbers.
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from rag.config import CONFIG

_client: httpx.Client | None = None

SPEC_METRICS = {                                  # Prometheus name suffix -> our key
    "spec_decode_num_draft_tokens_total": "draft_tokens",
    "spec_decode_num_accepted_tokens_total": "draft_accepted",
    "spec_decode_num_drafts_total": "draft_steps",
}


class LlamaServerError(RuntimeError):
    pass


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=CONFIG.llama_server_url,
                               timeout=httpx.Timeout(300.0, connect=5.0))
    return _client


def server_timings(t: dict) -> dict:
    """llama-server timings -> the key names ollama_client uses."""
    out = {}
    for src, dst in (("prompt_n", "prompt_eval_count"), ("predicted_n", "eval_count")):
        if src in t:
            out[dst] = t[src]
    for src, dst in (("prompt_ms", "prompt_eval_ms"), ("predicted_ms", "eval_ms")):
        if src in t:
            out[dst] = round(t[src], 2)
    if "cache_n" in t:
        out["cache_n"] = t["cache_n"]             # prompt tokens reused from the prefix cache
    out["load_ms"] = 0.0                          # the model is loaded when the server starts
    return out


def parse_sse(lines: Iterator[str]) -> Iterator[dict]:
    """OpenAI-style stream ("data: {...}" lines) -> our token / done events."""
    timings: dict = {}
    finish = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        data = json.loads(payload)
        if data.get("error"):
            raise LlamaServerError(str(data["error"]))
        if data.get("timings"):
            timings = data["timings"]
        for choice in data.get("choices", []):
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                yield {"type": "token", "text": piece}
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    yield {"type": "done", "timings": server_timings(timings), "done_reason": finish}


def chat_stream(messages: list[dict], model: str | None = None,
                max_tokens: int | None = None) -> Iterator[dict]:
    """Stream a chat answer from llama-server. `model` is ignored: the server
    serves the model it was started with."""
    body = {
        "messages": messages,
        "stream": True,
        "temperature": CONFIG.temperature,
        "max_tokens": max_tokens or CONFIG.max_output_tokens,
        "cache_prompt": False,       # every request pays full prefill: fair TTFT between variants
    }
    with _http().stream("POST", "/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            r.read()
            raise LlamaServerError(f"chat failed ({r.status_code}): {r.text[:300]}")
        yield from parse_sse(r.iter_lines())


def parse_metrics(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        for suffix, key in SPEC_METRICS.items():
            if name.endswith(suffix):
                out[key] = int(float(value))
    return out


def spec_counters() -> dict | None:
    """Running totals of draft tokens / accepted tokens, or None without --metrics."""
    try:
        r = _http().get("/metrics")
    except httpx.HTTPError:
        return None
    return parse_metrics(r.text) if r.status_code == 200 else None


def spec_delta(before: dict | None, after: dict | None) -> dict:
    """One request's draft numbers, plus acceptance and tokens kept per check."""
    if not before or not after:
        return {}
    d = {k: after.get(k, 0) - before.get(k, 0) for k in SPEC_METRICS.values()}
    if not d["draft_tokens"]:
        return {}                                 # no speculation happened
    d["acceptance_rate"] = round(d["draft_accepted"] / d["draft_tokens"], 3)
    return d
