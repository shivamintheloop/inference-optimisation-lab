# Inference Optimization Lab (lite)

I built a small RAG assistant on a laptop, over my own AI-engineering lecture notes and three arXiv papers. I added one layer at a time and measured each one. Then I tried to make the LLM part faster with speculative decoding, on two engines and four model formats.

No framework: plain Python, so every step is visible and traced.

- **Results summary:** [`results/RESULTS.md`](results/RESULTS.md)
- **Caveats, side findings, and the full numbers:** [`detailed_results.md`](detailed_results.md)
- **Full speculative decoding write-up:** [`analysis/speculative-decoding.md`](analysis/speculative-decoding.md)

## Setup

| Part | Choice |
|---|---|
| Machine | MacBook, Apple M4 Max, 64 GB unified memory |
| Corpus | 26 lecture-note files (markdown) + 3 arXiv papers: Attention Is All You Need, RAG (Lewis 2020), PagedAttention/vLLM. 3,017 chunks |
| LLM | Llama 3.1 8B Instruct (Ollama, llama.cpp, MLX), temperature 0 |
| Draft model | Llama 3.2 1B Instruct |
| Embeddings | `nomic-embed-text` (768 dims) |
| Search | Qdrant (vectors) + SQLite FTS5 (BM25 keywords), merged with RRF |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Eval | 24 questions with known source files + 4 off-topic questions |

## Experiments I ran

**Part 1: RAG pipeline, one layer per phase**

| Phase | What I added |
|---|---|
| 0 | Tracing: one JSON line per request with step timings and a config hash |
| 1 | Ingestion: parse, chunk (~400 tokens), embed, store; only changed files are re-embedded |
| 2 | Vector search + keyword search (BM25), merged with RRF |
| 3 | Cross-encoder re-ranker + a "not found" check |
| 4 | Cited, streamed answers + an answer-level eval |

**Part 2: Speculative decoding** (a small model or a text lookup guesses tokens, the 8B checks them in one pass)

| Loop | Question |
|---|---|
| Baseline | Does a 1B draft speed up Llama 3.1 8B Q4 on llama.cpp? |
| 1 | Do copy-heavy ("quote the evidence") answers help? |
| 2 | Does another engine (MLX) behave differently? |
| 3 | Do other 8B formats (Q4_0, Q8_0, F16) change the answer? |

## What I learned

**RAG**

1. **The simplest search won.** BM25 alone found the right file in the top 5 for 96% of questions, in 4 ms. Hybrid search was *worse* (92%), because the embedding model blurs acronyms like RRF and SSE.
2. **The re-ranker's real value was "I don't know".** It didn't beat BM25 on accuracy, but its score cleanly separates real questions from off-topic ones. It refused 4/4 off-topic questions without calling the LLM, for ~45 ms.
3. **File-level retrieval metrics can lie.** A question counted as a retrieval "hit", but no retrieved chunk actually explained the topic. Only the answer eval caught it.
4. **Citations show *where* a claim came from, not *whether it is true*.** The 8B cited a source and still stated something false.
5. **The LLM is ~98% of the wait.** Search + re-rank take ~70 ms. Prompt reading and answer writing take ~1.7 s each.

**Speculative decoding**

6. **It made my default setup slower** (0.59×). The textbook says "checking k tokens costs about one normal step". On llama.cpp with 4-bit weights it doesn't: checking 4 tokens costs 3 steps.
7. **It pays off when the 8B's step is memory-bound.** F16 weights: **1.80×** faster per token, identical answers. MLX (cheaper multi-token checks): **1.26×**.
8. **4-bit quantization and speculative decoding compete for the same headroom.** F16 + draft (17.5 ms/token) is still slower than plain Q4 (11.9 ms/token).
9. **The workload matters as much as the model.** When answers quote their sources, free n-gram guesses give **1.75×** even on Q4.
10. **Count the whole answer, not just per-token speed.** The draft must read the prompt too (+~200 ms). Short answers can end up no faster.
11. **Measure the check cost first.** A one-minute micro-benchmark predicted the direction and rough size of every result.
12. **Laptops throttle.** Decode speed dropped from 85 to 26 tok/s within 2 minutes of sustained load. I had to use interleaved A/B runs with cool-downs.

## Results at a glance

**Retrieval** (24 questions, Phase 3)

| Mode | hit@1 | hit@5 | MRR@10 | Time |
|---|---|---|---|---|
| Vector | 0.54 | 0.79 | 0.639 | 19 ms |
| **BM25** | **0.83** | **0.96** | **0.877** | **4 ms** |
| Hybrid (RRF) | 0.75 | 0.92 | 0.814 | 23 ms |
| Hybrid + re-rank | 0.79 | 0.96 | 0.861 | 67 ms |

**Answers** (Llama 3.1 8B Q4_K_M via Ollama, Phase 4)

| Off-topic refused | Real questions wrongly refused | Cited a source | Fake citations | First token p50 | Per token p50 | Whole question p50 |
|---|---|---|---|---|---|---|
| 4/4 | 1/24 | 96% | 0 | 1.7 s | 13.8 ms | 3.5 s |

**Speculative decoding** (24 prompts each, temperature 0)

| Setup | Engine | Per token | Whole answer | Same text as without |
|---|---|---|---|---|
| 8B Q4_K_M + 1B draft k=3 | llama.cpp | 0.59× ❌ | 0.72× | 21/24 |
| 8B F16 + 1B draft k=3 | llama.cpp | **1.80×** | **1.50×** | 24/24 |
| 8B Q4_K_M + n-gram, "quote the evidence" prompts | llama.cpp | **1.75×** | **1.37×** | 23/24 |
| 8B 4-bit + 1B draft k=2 | MLX | 1.26× | 0.99× | 23/24 |
| 8B 4-bit + 1B draft k=2, "quote the evidence" prompts | MLX | 1.43× | 1.14× | 21/24 |

Answers that differ only swap one near-tie word ("one-way" vs "one-directional"), never the meaning.

## How to run

Run everything from this folder. You need macOS on Apple Silicon, Python 3.13, Docker and [Ollama](https://ollama.com).

```bash
# 1. Python env
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Qdrant (dashboard: http://localhost:6333/dashboard)
open -a Docker
docker compose up -d

# 3. Models (~5 GB)
ollama serve &            # skip if the Ollama app is already running
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 4. Corpus PDFs
curl -sSL -o docs/attention-is-all-you-need.pdf https://arxiv.org/pdf/1706.03762
curl -sSL -o docs/rag-lewis-2020.pdf            https://arxiv.org/pdf/2005.11401
curl -sSL -o docs/pagedattention-vllm.pdf       https://arxiv.org/pdf/2309.06180

# 5. Check
.venv/bin/pytest -q
```

**Your own notes:** the lecture notes are not included. By default they are read from `../../all-notes-files`. Point `corpus_roots` in `rag/config.py` at any folder of `.md` / `.pdf` files. The eval questions in `eval/questions.jsonl` expect my notes, so write your own questions for your corpus.

### Use the RAG assistant

```bash
.venv/bin/python -m rag ingest                       # index new/changed files (--force to redo all)
.venv/bin/python -m rag status                       # what is indexed
.venv/bin/python -m rag search "What is RRF?" --mode rerank
.venv/bin/python -m rag ask "How does RRF merge results?"
```

### Run the evals

```bash
.venv/bin/python -m rag eval                         # retrieval: hit@1, hit@5, MRR@10 per search mode
.venv/bin/python -m rag eval --answers               # + full answers: refusals, citations, timings
.venv/bin/python -m rag eval --answers --record "Phase N: what changed"   # append to results/RESULTS.md
```

### Run the speculative decoding experiments

Needs [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`, `llama-bench`) and `ollama pull llama3.2:1b`. The scripts reuse Ollama's GGUF files.

```bash
scripts/micro-bench.sh                               # 1 minute: how much does checking N tokens cost?
.venv/bin/python -m rag export-prompts               # freeze the 24 prompts -> bench/prompts-v1.jsonl
.venv/bin/python -m rag export-prompts --extract     # "quote the evidence" variant

RAG_LLM_BACKEND=llama_server .venv/bin/python -m rag bench-ab \
    --variants baseline draft:3 ngram:24:4 --record  # interleaved A/B -> results/INFERENCE.md

TARGET=llama3.1:8b-instruct-fp16 scripts/micro-bench.sh   # other 8B formats

.venv/bin/python -m rag mlx-micro                    # MLX check cost
.venv/bin/python -m rag mlx-ab --variants baseline draft:2 --record
```

Variant syntax: `draft:<k>[:<p_min>]`, `ngram:<m>:<n>`, plus `@t=<target>,d=<draft>` to change models (e.g. `draft:3@t=f16,d=1bq4`). Targets `q4km`, `q40`, `q8`, `f16` and drafts `1b`, `1bq4` map to Ollama tags in `rag/bench_ab.py`; pull the ones you use first. Start a server by hand with `scripts/llama-server.sh baseline | draft [k] | ngram [m] [n]`.

Tip: let the laptop cool down between benchmarks, or the numbers drift.

## Project layout

| Path | What it is |
|---|---|
| `rag/` | The pipeline: ingest, retrieve, rerank, generate, evaluate, benchmarks, CLI (`python -m rag ...`) |
| `rag/config.py` | Every setting in one place; each trace stores a snapshot + hash |
| `eval/questions.jsonl` | 24 questions with expected files + 4 off-topic ones |
| `bench/` | Frozen prompts for the inference benchmarks |
| `scripts/` | `llama-server.sh` and `micro-bench.sh` |
| `results/` | `RESULTS.md` (summary + run log), `INFERENCE.md` (benchmark rows), raw JSON per run |
| `analysis/` | Speculative decoding write-up |
| `PLAN.md` | The original plan for the inference lab |
| `tests/` | `pytest` unit tests |
