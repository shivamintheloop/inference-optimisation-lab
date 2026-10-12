# Inference Optimization Lab

A local RAG pipeline (ingestion + generation) with built-in tracing. It is the system that `PLAN.md` benchmarks and optimises.

- **Answer model + follow-up rewriter:** `llama3.1:8b` (Ollama, Meta)
- **Embeddings:** `nomic-embed-text` (Ollama, 768 dims)
- **Vectors:** Qdrant (Docker). **Documents + keyword search:** SQLite (FTS5)
- **Tracing:** `rag/trace.py` writes one JSON line per request to `traces/`

## Setup

Run every command from this folder (`projects/inference-optimization-lab/`). `docker compose` looks for `docker-compose.yml` in the current folder.

```bash
# 1. Python env (project-local)
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Qdrant (dashboard: http://localhost:6333/dashboard)
open -a Docker
docker compose up -d

# 3. Ollama + models (~5 GB)
ollama serve &           # skip if the Ollama app is already running
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 4. Corpus PDFs (not in git; the eval expects these three)
curl -sSL -o docs/attention-is-all-you-need.pdf https://arxiv.org/pdf/1706.03762
curl -sSL -o docs/rag-lewis-2020.pdf            https://arxiv.org/pdf/2005.11401
curl -sSL -o docs/pagedattention-vllm.pdf       https://arxiv.org/pdf/2309.06180

# 5. Check
.venv/bin/pytest -q
curl -s localhost:6333/healthz
ollama list
```

## Commands

```bash
.venv/bin/python -m rag ingest                 # index new/changed files, remove deleted ones
.venv/bin/python -m rag ingest --retry-failed  # also retry files that failed before
.venv/bin/python -m rag ingest --force         # re-index everything (after changing chunk settings)
.venv/bin/python -m rag status [-v]            # documents, chunk counts in all 3 stores, failures
.venv/bin/python -m rag search "question" [--mode vector|bm25|hybrid|rerank] [-k 8]
.venv/bin/python -m rag ask "question" [--show-prompt]   # cited answer from the notes, streamed
.venv/bin/python -m rag eval [--mode M ...]    # hit@1 / hit@5 / MRR@10 on eval/questions.jsonl
.venv/bin/python -m rag eval --answers         # + full pipeline per question: refusals, citations, TTFT/TPOT
.venv/bin/python -m rag eval --answers --record "Phase N: what changed" [--note "..."]   # also append to results/RESULTS.md
```

Inference experiments (llama.cpp `llama-server` instead of Ollama, same GGUF files):

```bash
scripts/llama-server.sh baseline | draft [k] | ngram [k] [n]   # start a server on :8081
scripts/micro-bench.sh                                         # llama-bench: 8B check cost vs tokens, 1B/8B decode
.venv/bin/python -m rag export-prompts                         # freeze the 24 LLM prompts -> bench/prompts-v1.jsonl
RAG_LLM_BACKEND=llama_server .venv/bin/python -m rag ask "..." # any command, via llama-server
RAG_LLM_BACKEND=llama_server .venv/bin/python -m rag bench-ab --variants baseline draft:3 ngram:16:12 --record
                                                               # interleaved, heat-checked A/B -> results/INFERENCE.md
.venv/bin/python -m rag export-prompts --extract               # quote-the-evidence variant of the 24 prompts
RAG_LLM_BACKEND=llama_server .venv/bin/python -m rag bench-ab --variants baseline@t=f16 draft:3@t=f16,d=1bq4
                                                               # variants: draft:<k>[:<p_min>], ngram:<m>:<n>, @t=<target>,d=<draft>
.venv/bin/python -m rag mlx-micro                              # MLX: check cost vs tokens
.venv/bin/python -m rag mlx-ab --variants baseline draft:2 [--prompts bench/prompts-extract-v1.jsonl] --record
```

**End of every phase:** run `rag eval --answers --record "Phase N: ..."`, then commit. `results/RESULTS.md` is the phase-by-phase history; raw runs are in `results/eval-*.json` and `results/answers-*.json`.

## Retrieval results (phase 3, config `9999f4c0`, 24 questions + 4 unanswerable)

| mode | hit@1 | hit@5 | MRR@10 | keyword hit@5 | paraphrase hit@5 | p50 ms |
|---|---|---|---|---|---|---|
| vector | 0.54 | 0.79 | 0.639 | 0.67 | 1.00 | 19 |
| bm25 | 0.83 | 0.96 | 0.877 | 0.93 | 1.00 | 4 |
| hybrid (RRF) | 0.75 | 0.92 | 0.814 | 0.87 | 1.00 | 23 |
| hybrid + re-rank | 0.79 | 0.96 | 0.861 | 1.00 | 0.89 | 67 |

- `nomic-embed-text` blurs acronyms and names (RRF, SSE, Jev), so vector search drags hybrid below BM25.
- The re-ranker fixes every keyword miss (e.g. "What is RRF?": hybrid rank 12 → #1) but `ms-marco-MiniLM` misses one paraphrase (q22).
- Not found: re-rank top score separates cleanly (answerable ≥ −2.65, unanswerable ≤ −9.18) → `not_found_threshold = -6.0`.
- The re-ranker takes ~2.4 s to load once per process; ~45 ms per question after that.

## Answer results (phase 4, `llama3.1:8b` Q4_K_M via Ollama, M4 Max)

| Refused right | Refused wrong | Cited | Cites expected file | Invalid cites | TTFT p50 / p95 | TPOT p50 | Prompt tok p50 | Output tok p50 | Per question p50 |
|---|---|---|---|---|---|---|---|---|---|
| 4/4 | 1/24 | 0.96 | 0.88 | 0 | 1743 / 2552 ms | 13.8 ms | 1087 | 121 | 3.5 s |

Where one warm question's time goes (`rag ask "How does RRF merge..."`, trace `c368417d`):

```
search (embed + vector + BM25 + RRF)    ~0.1 s
rerank                                  ~0.05 s   (+2.6–3.9 s model load once per process)
prefill  1,282 prompt tokens            1.3 s     (~970 tok/s)   -> this is TTFT
decode   240 output tokens              2.8 s     (~87 tok/s, 11.8 ms/token)
```

- For this long answer decode is about 2/3 of LLM time; for a typical question (p50: 121 output tokens × 13.8 ms ≈ 1.7 s vs 1.7 s TTFT) prefill and decode take about the same time. Both are what the inference ladder in `PLAN.md` attacks.
- Ollama decodes at ~87 tok/s; memory bandwidth caps a 4.9 GB model at ~110 tok/s on this chip (546 GB/s ÷ 4.9 GB), so single-request decode is already near its limit.
- Prompts are only ~1,100 tokens even though the budget is 3,500, because most note cards are short.
- **File-level hit@5 can lie.** q05 (PagedAttention) counts as a retrieval hit, but no retrieved chunk explains PagedAttention, so the LLM rightly refused. Cause: BM25 with OR lets "problem"/"solve" outscore the key word, and the PDF title in every chunk's indexed heading makes "pagedattention" look common.
- Llama 3.1 8B sometimes states things the sources don't say (q16: "GQA makes the KV cache constant"). Citations check *where*, not *whether it's true*.
- Timing is noisy on a laptop: one run had TPOT 37.5 ms during macOS background work (same answers). Repeat runs before trusting a speed number.

## Speculative decoding results (llama.cpp b11146, Llama 3.1 8B Q4_K_M + Llama 3.2 1B Q8_0 draft)

Interleaved A/B on the 24 frozen prompts (`results/ab-20261004-153955.json`): 4-prompt blocks, 2 min cool-down before each, order rotated.

| Setup | TPOT p50 | Speedup | Answer p50 | Guesses kept | Same text as baseline |
|---|---|---|---|---|---|
| 8B alone | 11.9 ms | 1.00× | 2.78 s | – | – |
| + 1B draft, k=3 | 20.4 ms | **0.59× (slower)** | 4.16 s | 66% | 21/24 (3 one-word swaps) |
| + n-gram, k=16, n=12 | 11.9 ms | 1.00× | 2.75 s | 15% (drafts in 14/24 answers) | 24/24 |
| 1B draft k=16, counting 1–60 (easy text) | – | **1.6× faster** | – | ~100% | – |

Answer quality with the draft (`rag eval --answers` via llama-server): identical to Phase 4 (refused 4/4, refused wrong 1/24, cited 0.96, cites expected 0.88).

Why it's slower: the 8B's *check* of several tokens is not nearly free here (`scripts/micro-bench.sh`, `results/micro-bench-20261004-144809.md`):

| 8B pass over N tokens | 1 | 2 | 4 | 8 | 9–32 |
|---|---|---|---|---|---|
| ms | 12.5 | 17.6 | 38.0 | 72.0 | ~56 (flat) |

- The 1B guess costs 3.8 ms (c ≈ 0.31 of an 8B step), as predicted. The wrong assumption was that the 8B checks k tokens for about one step's cost. Up to 8 tokens llama.cpp's Metal code grows ~linearly; from 9 it switches to a matrix routine at a flat ~4.5 steps.
- Per question, time per token falls as more guesses are kept (r = −0.81), but extrapolates to 15.7 ms at 100% kept, still above 11.9 ms. Better guesses shrink the loss; they can't turn it into a gain at this check cost.
- The 3 non-identical answers are near-ties flipped by rounding ("one-directional" vs "one-way"): the batched check and the single-token step use different GPU routines.
- Lesson: "verification is nearly free" is a property of the engine + hardware + quantization. Measure the check cost (one minute of llama-bench) before building the experiment.
- Laptop note: sustained load slows decode 85 → 26 tok/s within ~2 min. Back-to-back benchmarks on this machine are unusable; see `rag/bench_ab.py`.

### Follow-up: where speculative decoding *does* pay off (`analysis/speculative-decoding.md`)

All on the 24 frozen prompts, interleaved A/B with cool-downs, temperature 0:

| Case | Engine | Per token | Whole answer | Same text |
|---|---|---|---|---|
| Normal answers, 8B Q4_K_M + 1B draft k=3 | llama.cpp | 0.59× | 0.72× | 21/24 |
| **Normal answers, 8B F16 + 1B Q4 draft k=3** | llama.cpp | **1.80×** | **1.50×** | 24/24 |
| **"Quote the evidence" answers, 8B Q4_K_M + n-gram m=24 n=4** | llama.cpp | **1.75×** | **1.37×** | 23/24 |
| Normal answers, 8B 4-bit + 1B 4-bit draft k=2 | MLX | 1.26× | 0.99× (1.08× if ≥120 tokens) | 23/24 |
| "Quote the evidence" answers, 8B 4-bit + 1B 4-bit draft k=2 | MLX | 1.43× | 1.14× | 21/24 |

The rule behind every row: it pays off when checking a few extra tokens costs
almost nothing extra. That holds when the target's step is limited by memory
reads (F16), or when the engine's small-batch code is efficient (MLX); 4-bit
llama.cpp is limited by arithmetic instead, so it needs answers that copy their
sources (n-gram). `scripts/micro-bench.sh` and `rag mlx-micro` measure that check
cost in a minute, before any long benchmark.

## Layout

| Path | What it is |
|---|---|
| `rag/config.py` | Every setting in one place; each trace stores a snapshot + hash |
| `rag/trace.py` | `trace()` / `span()` helpers → `traces/<kind>-YYYY-MM-DD.jsonl` |
| `rag/ollama_client.py` | Ollama native API: `embed()` and streaming `chat_stream()`; returns server timings |
| `rag/parse.py` | Markdown → heading sections; PDF → pages without headers/footers |
| `rag/chunk.py` | ~400-token chunks, 50 overlap, cut at paragraph/line/sentence boundaries |
| `rag/docs_db.py` | SQLite: `documents` (hash, status) + `chunks_fts` (BM25 keyword index) |
| `rag/vector_store.py` | Qdrant: stable chunk IDs, upsert, delete old versions, count |
| `rag/ingest.py` | Scan → new/changed/unchanged/deleted/failed → parse → chunk → embed → store (hash saved last) |
| `rag/retrieve.py` | Vector search + BM25 (SQLite FTS5) + RRF merge |
| `rag/rerank.py` | Cross-encoder re-rank of the top 30 → top 6, plus the "not found" check |
| `rag/context.py` | Join overlapping neighbour chunks, number sources `[S1]..`, fit the token budget |
| `rag/generate.py` | System prompt → sources → question; streamed answer; TTFT / TPOT / ITL; citation check |
| `rag/evaluate.py` | hit@1 / hit@5 / MRR@10 per mode → `results/eval-*.json`; `--answers` → `results/answers-*.json` |
| `rag/llamacpp_client.py` | llama-server streaming client (same interface as Ollama's) + draft counters from `/metrics` |
| `rag/bench_llm.py` | Frozen prompts, LLM-only timing, `results/INFERENCE.md` rows |
| `rag/bench_ab.py` | Interleaved A/B of server variants with cool-downs (the laptop slows under sustained load) |
| `rag/bench_mlx.py` | Same A/B design on MLX-LM (`rag mlx-micro`, `rag mlx-ab`) |
| `analysis/` | Write-ups of the inference experiments |
| `scripts/` | `llama-server.sh` (baseline / draft / n-gram), `micro-bench.sh` (llama-bench) |
| `eval/questions.jsonl` | 24 questions with expected files (keyword / paraphrase) + 4 unanswerable |
| `rag/cli.py` | `python -m rag ...` commands |
| `docs/` | Drop PDFs here; they are ingested next to the lecture notes |
| `tests/` | `pytest` unit tests |
