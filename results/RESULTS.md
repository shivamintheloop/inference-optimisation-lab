# Results

All numbers come from the traces and raw JSON files in this folder. Caveats and side findings are in [`../detailed_results.md`](../detailed_results.md).

- **hit@1 / hit@5**: share of questions whose expected file is the top chunk / in the top 5 chunks
- **MRR@10**: average of 1 / rank of the first correct chunk (0 if not in the top 10)
- **Jargon / Everyday-words**: questions that use the notes' exact terms vs. the same idea in plain words
- **TTFT / TPOT**: time to first token / time per output token
- **Speedup**: time without speculative decoding ÷ time with it (above 1× = faster)

## 1. Retrieval

Phase 3 labels (config `9999f4c0`, questions `8fdcb037`), 24 questions. Raw: `eval-20261001-205238.json`.

| Mode | hit@1 | hit@5 | MRR@10 | Jargon hit@5 | Everyday-words hit@5 | Time per question |
|---|---|---|---|---|---|---|
| Vector | 0.54 | 0.79 | 0.639 | 0.67 | 1.00 | 19 ms |
| **BM25** | **0.83** | **0.96** | **0.877** | 0.93 | 1.00 | **4 ms** |
| Hybrid (RRF) | 0.75 | 0.92 | 0.814 | 0.87 | 1.00 | 23 ms |
| Hybrid + re-rank | 0.79 | 0.96 | 0.861 | **1.00** | 0.89 | 67 ms |

- Hybrid is below BM25: `nomic-embed-text` blurs acronyms and names (RRF, SSE), so vector search drags good keyword hits down.
- The re-ranker fixes every jargon miss ("What is RRF?": rank 12 → 1) but misses one everyday-words question.
- **"Not found" check:** real questions score ≥ -2.65 on the re-ranker, off-topic ones ≤ -9.18. A threshold of -6.0 keeps 24/24 and refuses 4/4, with no LLM call.

## 2. Answers

Full pipeline, Llama 3.1 8B Q4_K_M via Ollama, M4 Max. Raw: `answers-20261001-214918.json`.

| Off-topic refused | Real questions wrongly refused | Cited | Cites expected file | Invalid cites | TTFT p50 / p95 | TPOT p50 | Prompt tokens p50 | Output tokens p50 | Per question p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| 4/4 | 1/24 | 0.96 | 0.88 | 0 | 1.7 / 2.6 s | 13.8 ms | 1,087 | 121 | 3.5 / 6.0 s |

Where a median question's time goes: ~2% search + re-rank, ~49% reading the prompt (prefill), ~49% writing the answer (decode).

## 3. Speculative decoding

Llama 3.1 8B target, Llama 3.2 1B draft. 24 frozen prompts per row, temperature 0, interleaved A/B with cool-downs. "Normal" = the RAG prompts from section 2; "Quote" = the same prompts asking the model to quote its evidence word for word.

| Setup | Engine | Prompts | TPOT (alone → with) | Per token | Whole answer | Same text | Raw |
|---|---|---|---|---|---|---|---|
| 8B Q4_K_M + 1B draft k=3 | llama.cpp | Normal | 11.9 → 20.4 ms | 0.59× ❌ | 0.72× | 21/24 | `ab-20261004-153955.json` |
| 8B Q4_K_M + n-gram (default) | llama.cpp | Normal | 11.9 → 11.9 ms | 1.00× | ~1.0× | 24/24 | `ab-20261004-153955.json` |
| **8B F16 + 1B Q4 draft k=3** | llama.cpp | Normal | 31.5 → 17.5 ms | **1.80×** | **1.50×** | 24/24 | `ab-20261004-184037.json` |
| **8B Q4_K_M + n-gram m=24 n=4** | llama.cpp | Quote | 12.0 → 6.9 ms | **1.75×** | **1.37×** | 23/24 | `ab-20261004-191423.json` |
| 8B 4-bit + 1B 4-bit draft k=2 | MLX | Normal | 10.5 → 8.4 ms | 1.26× | 0.99× | 23/24 | `mlx-ab-20261004-194714.json` |
| 8B 4-bit + 1B 4-bit draft k=2 | MLX | Quote | 10.5 → 7.3 ms | 1.43× | 1.14× | 21/24 | `mlx-ab-20261004-203659.json` |
| 8B 4-bit + 1B 4-bit draft k=4 | MLX | Quote | 10.5 → 7.6 ms | 1.38× | 1.11× | 22/24 | `mlx-ab-20261004-203659.json` |

**Why: the cost of checking N tokens in one 8B pass**, in normal steps (`micro-bench-*.md`, `mlx-micro-20261004-170556.json`). Speculative decoding needs this to stay near 1.

| 8B format | Engine | Normal step | 2 tokens | 3 | 4 | 8 |
|---|---|---|---|---|---|---|
| Q4_K_M (4.6 GB) | llama.cpp | 12.5 ms | 1.4 | 1.9 | 3.0 | 5.8 |
| Q4_0 (4.4 GB) | llama.cpp | 12.1 ms | 1.5 | 2.0 | 2.6 | 5.0 |
| Q8_0 (8.0 GB) | llama.cpp | 18.9 ms | 1.06 | 1.3 | 1.7 | 3.2 |
| **F16 (15 GB)** | llama.cpp | 32.3 ms | **1.03** | **1.05** | **1.07** | 2.1 |
| 4-bit | MLX | 11.4 ms | 1.1 | 1.2 | 1.5 | 2.6 |

**Best setup per goal on this laptop**

| Goal | Setup | Per token |
|---|---|---|
| Fastest normal answers | MLX 8B 4-bit + 1B draft k=2 | 8.4 ms |
| Fastest quote-the-evidence answers | llama.cpp Q4_K_M + n-gram m=24 n=4 | 6.9 ms |
| Full-precision answers, faster | llama.cpp F16 + 1B Q4 draft k=3 | 17.5 ms |

Every benchmark row, with memory and acceptance counters: [`INFERENCE.md`](INFERENCE.md). How each result was found: [`../analysis/speculative-decoding.md`](../analysis/speculative-decoding.md).

---

## Run log

Appended by `python -m rag eval --record "<label>"`; entries below are not edited by hand.
Compare rows only when the `questions` hash matches: a different hash means the eval labels changed in between.

## Phase 2: hybrid retrieval (vector + BM25 + RRF)

Run 2026-10-01T20:41:29 · config `edf47ee8` · questions `v1-labels` · raw: `eval-20261001-204129.json`

| Mode | hit@1 | hit@5 | MRR@10 | Jargon hit@5 | Everyday-words hit@5 | Time per question |
|---|---|---|---|---|---|---|
| vector | 0.50 | 0.79 | 0.602 | 0.67 | 1.00 | 24 ms |
| bm25 | 0.79 | 0.92 | 0.835 | 0.93 | 0.89 | 4 ms |
| hybrid | 0.71 | 0.88 | 0.772 | 0.87 | 0.89 | 27 ms |

Labels v1: before the two Phase 3 label fixes (q13 +lectures 06/01, q24 +lectures 09/10), so not directly comparable with later rows. Finding: hybrid < BM25 because nomic-embed-text blurs acronyms (RRF, SSE, Jev).

## Phase 3: re-ranker + not-found check

Run 2026-10-01T20:52:38 · config `9999f4c0` · questions `8fdcb037` · raw: `eval-20261001-205238.json`

| Mode | hit@1 | hit@5 | MRR@10 | Jargon hit@5 | Everyday-words hit@5 | Time per question |
|---|---|---|---|---|---|---|
| vector | 0.54 | 0.79 | 0.639 | 0.67 | 1.00 | 19 ms |
| bm25 | 0.83 | 0.96 | 0.877 | 0.93 | 1.00 | 4 ms |
| hybrid | 0.75 | 0.92 | 0.814 | 0.87 | 1.00 | 23 ms |
| rerank | 0.79 | 0.96 | 0.861 | 1.00 | 0.89 | 67 ms |

Not found (re-rank top score): answerable ≥ -2.648, unanswerable ≤ -9.175; threshold -6.0 keeps 24/24 and refuses 4/4.

Re-ranker (ms-marco-MiniLM-L-6-v2) fixes every jargon miss ("What is RRF?": hybrid rank 12 -> #1) but misses paraphrase q22. Ties BM25 on hit@5 (1-question gaps are noise at n=24); kept for the not-found check, the only score that separates off-topic questions.

## Phase 4: generation (context builder, cited streaming answers, rag ask)

Run 2026-10-01T21:41:33 · config `9999f4c0` · questions `8fdcb037` · raw: `eval-20261001-214133.json`

| Mode | hit@1 | hit@5 | MRR@10 | Jargon hit@5 | Everyday-words hit@5 | Time per question |
|---|---|---|---|---|---|---|
| vector | 0.54 | 0.79 | 0.639 | 0.67 | 1.00 | 18 ms |
| bm25 | 0.83 | 0.96 | 0.877 | 0.93 | 1.00 | 4 ms |
| hybrid | 0.75 | 0.92 | 0.814 | 0.87 | 1.00 | 21 ms |
| rerank | 0.79 | 0.96 | 0.861 | 1.00 | 0.89 | 67 ms |

Not found (re-rank top score): answerable ≥ -2.648, unanswerable ≤ -9.175; threshold -6.0 keeps 24/24 and refuses 4/4.

Answers (full pipeline, `llama3.1:8b`) · raw: `answers-20261001-214506.json`

| Refused right | Refused wrong | Cited | Cites expected file | Invalid cites | Cut off | TTFT p50 / p95 | TPOT p50 | Prompt tokens p50 | Output tokens p50 | Time per question p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4/4 | 1/24 | 0.96 | 0.88 | 0 | 0 | 3748 / 6750 ms | 37.5 ms | 1087 | 121 | 8.0 / 15.4 s |

Retrieval unchanged from Phase 3 (same config). Answer eval found what file-level hit@5 hides: q05 (PagedAttention) counts as a hit, but no retrieved chunk explains PagedAttention, so the LLM correctly refused. Cause: OR-query BM25 lets 'problem'/'solve' outscore the key term, and the PDF title in every chunk's indexed heading makes 'pagedattention' look common. q16: wrong claim (GQA makes KV cache 'constant') + stray [G2] tag; q22: known paraphrase retrieval miss. TPOT drifted 11.8 -> 14.9 ms (two at 23 ms) over the run with no link to prompt size: machine noise, so the benchmark needs repeats.

## Phase 4 (timing re-run on a quiet run)

Run 2026-10-01T21:47:49 · config `9999f4c0` · questions `8fdcb037` · raw: `eval-20261001-214749.json`

| Mode | hit@1 | hit@5 | MRR@10 | Jargon hit@5 | Everyday-words hit@5 | Time per question |
|---|---|---|---|---|---|---|
| vector | 0.54 | 0.79 | 0.639 | 0.67 | 1.00 | 20 ms |
| bm25 | 0.83 | 0.96 | 0.877 | 0.93 | 1.00 | 5 ms |
| hybrid | 0.75 | 0.92 | 0.814 | 0.87 | 1.00 | 22 ms |
| rerank | 0.79 | 0.96 | 0.861 | 1.00 | 0.89 | 69 ms |

Not found (re-rank top score): answerable ≥ -2.648, unanswerable ≤ -9.175; threshold -6.0 keeps 24/24 and refuses 4/4.

Answers (full pipeline, `llama3.1:8b`) · raw: `answers-20261001-214918.json`

| Refused right | Refused wrong | Cited | Cites expected file | Invalid cites | Cut off | TTFT p50 / p95 | TPOT p50 | Prompt tokens p50 | Output tokens p50 | Time per question p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4/4 | 1/24 | 0.96 | 0.88 | 0 | 0 | 1743 / 2552 ms | 13.8 ms | 1087 | 121 | 3.5 / 6.0 s |

Same answers as the entry above (temperature 0 is deterministic). The run above had TPOT p50 37.5 ms during a burst of macOS background work (mediaanalysisd etc.); this run and an earlier unrecorded one agree (TPOT 13.8 / 14.2 ms). Use this entry as the Phase 4 timing baseline.

## Speculative decoding: llama-server + Llama 3.2 1B draft k=3 (quality check)

Run 2026-10-04T15:41:29 · config `841acbce` · questions `8fdcb037` · raw: `eval-20261004-154129.json`

| Mode | hit@1 | hit@5 | MRR@10 | Jargon hit@5 | Everyday-words hit@5 | Time per question |
|---|---|---|---|---|---|---|
| vector | 0.54 | 0.79 | 0.639 | 0.67 | 1.00 | 21 ms |
| bm25 | 0.83 | 0.96 | 0.877 | 0.93 | 1.00 | 5 ms |
| hybrid | 0.75 | 0.92 | 0.814 | 0.87 | 1.00 | 22 ms |
| rerank | 0.79 | 0.96 | 0.861 | 1.00 | 0.89 | 69 ms |

Not found (re-rank top score): answerable ≥ -2.648, unanswerable ≤ -9.175; threshold -6.0 keeps 24/24 and refuses 4/4.

Answers (full pipeline, `llama3.1:8b`) · raw: `answers-20261004-154338.json`

| Refused right | Refused wrong | Cited | Cites expected file | Invalid cites | Cut off | TTFT p50 / p95 | TPOT p50 | Prompt tokens p50 | Output tokens p50 | Time per question p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4/4 | 1/24 | 0.96 | 0.88 | 0 | 0 | 2063 / 3591 ms | 23.8 ms | 1107 | 133 | 5.5 / 7.9 s |

Same 8B weights as Phase 4 but llama.cpp b11146 instead of Ollama. Checks that drafting doesn't change refusals/citations. Timing here is not a clean speed number (re-ranker shares the GPU; see results/INFERENCE.md for the A/B).
