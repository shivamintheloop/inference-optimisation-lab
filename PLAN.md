# LLM Inference Optimization Lab — Plan

Build → measure → optimize → measure again → explain the improvement.

Benchmark an enterprise-style RAG assistant served locally, then apply inference optimizations one at a time and prove each improvement with measurements.

- **Hardware:** Apple M4 Max, 64 GB unified memory, 40-core GPU
- **Serving:** llama.cpp `llama-server` (Metal)
- **Corpus:** lecture flash notes in `~/Documents/learning-ai/all-notes-files`
- **Background reading:** [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)

---

## Design principles

1. **Define a naive baseline, not "features off".** In vLLM, PagedAttention and continuous batching are always on; llama.cpp has no PagedAttention. The baseline is a deliberately naive setup: 1 slot, F16 weights, no prompt cache.
2. **Use metrics that fit the hardware.** Unified memory instead of VRAM; GPU load and watts from `powermetrics`. CUDA-only features (PagedAttention, chunked prefill) run on an optional cloud GPU day (Track B).
3. **Sweep load, not one point.** Batching only helps under concurrency; speculative decoding helps most at low concurrency. Every experiment runs at concurrency 1, 4, 8, 16, 32.
4. **Report incremental and cumulative deltas.** Each step vs the previous step, and vs baseline. Optimizations interact.
5. **Reproducible runs, noise-aware results.** Frozen dataset, temperature 0, fixed `max_tokens`, warm-up, 3 repeats. An improvement counts only if it beats run-to-run noise.
6. **Evidence-backed explanations.** Scripts compute every number. Every "why" cites a measured counter (cache hits, acceptance rate, batch size, memory). Results that contradict expectations get investigated, not explained away.
7. **Lossless before lossy.** Caching, batching and speculative decoding first; quantization last, behind a quality gate.
8. **Freeze retrieval.** Retrieved chunks are precomputed into the dataset, so the benchmark measures inference, not the vector DB.

---

## Scenario and workload

An enterprise-style RAG assistant over 26 lectures of flash notes. Every request carries a shared system prompt (~1k tokens) plus 2–4k tokens of retrieved chunks.

| Request type | Share | Prompt | Output (`max_tokens`) | Stresses |
|---|---|---|---|---|
| Q&A | 80% | ~3–5k tokens | 150 | Prefill / TTFT |
| Revision summary | 20% | ~3–5k tokens | 600 | Decode / TPOT |

- **Dataset:** 200 questions with retrieved context precomputed into `data/workload.jsonl` (frozen, versioned).
- **Quality set:** 50 questions with reference answers.
- **Load patterns:**
  - Closed loop: concurrency sweep {1, 4, 8, 16, 32}.
  - Open loop: Poisson arrivals at a target rate, for the SLO test.
- **Run protocol:** 60 s warm-up, 5 min measured, 3 repeats per configuration.

---

## Metrics

| Metric | Definition | Source |
|---|---|---|
| TTFT p50/p95 | Request sent → first streamed token | Client timestamps |
| TPOT p50/p95 | (last token − first token) ÷ (output tokens − 1) | Client |
| E2E p95 | Request sent → last token | Client |
| Throughput | Output tokens/s across all users | Client |
| Goodput | Requests/s meeting the SLO: TTFT p95 < 1 s **and** TPOT p95 < 50 ms | Client |
| Max users at SLO | Highest concurrency still meeting the SLO | Sweep |
| Memory / GPU load / watts | Peak unified memory, GPU %, power | `powermetrics`, server metrics |
| Cache-hit tokens, acceptance rate | Evidence counters for explanations | Server timings / metrics |
| Cost per 1M output tokens | Local: watts × time × electricity price. Cloud: $/GPU-hour ÷ (tokens/s × 3600) × 1M | Derived, assumptions stated |
| Quality | LLM-judge score on the 50-question set; temp-0 output match for lossless steps | Eval script |

**Headline metric:** max users at SLO, baseline → final.

---

## Experiment ladder

### Track A — local (M4 Max, llama.cpp)

| # | Change | Lossless? | Expected to move | Evidence counter |
|---|---|---|---|---|
| 0 | Baseline: 1 slot, F16, no prompt cache | — | — | — |
| 1 | Prompt / prefix caching | ✅ | TTFT ↓ | Cached tokens per request |
| 2 | Parallel slots + continuous batching | ✅ | Throughput ↑, max users at SLO ↑, per-user TPOT ↑ (worse) | Active batch size |
| 3 | Flash attention | ✅ | TTFT ↓ on long prompts | — |
| 4 | Speculative decoding (small same-family draft model) | ✅ | TPOT ↓ at low concurrency | Acceptance rate |

> **Done 2026-10-04 (out of order, first):** on this M4 Max + llama.cpp b11146 it made RAG answers *slower* (1B draft k=3: 0.59×; n-gram: 1.00×), with answer quality unchanged. Cause: the 8B's multi-token check costs 1.4–6× a normal step here. Details in README "Speculative decoding results".
| 5 | Weight quantization (Q8 → Q4) | ❌ | TPOT ↓, memory ↓ | Quality score |
| 6 | KV-cache quantization (q8_0) | ❌ | Memory ↓, max users ↑ | Quality score |

**Model:** a ~7–8B instruct target with a tiny same-family draft for speculative decoding (shared tokenizer required), e.g. Llama 3.1 8B Instruct + Llama 3.2 1B Instruct (both use the Llama 3 tokenizer).

### Track B — optional cloud GPU day (vLLM)

- Naive Hugging Face `transformers` serving vs vLLM → shows PagedAttention + continuous batching together.
- Chunked prefill on the mixed workload.
- Prefix caching.
- FP8 / AWQ quantization.
- Compare the **shape** of gains with Track A, not absolute numbers.

---

## Success criteria

- An improvement counts only if it exceeds **2× the std dev** across the 3 repeats; otherwise it is noise.
- A lossy step that drops quality by **more than 3 points** is a regression, however fast it is.
- Lossless steps should match baseline outputs at temperature 0; any mismatch is reported.

---

## Results table format

Generated by the analysis script, never filled in by hand:

| Metric | Baseline | Previous step | This step | Incremental Δ% | Cumulative Δ% |
|---|---|---|---|---|---|
| TTFT p95 | | | | | |
| TPOT p95 | | | | | |
| Throughput | | | | | |
| E2E p95 | | | | | |
| Goodput | | | | | |
| Max users at SLO | | | | | |
| Peak memory | | | | | |
| GPU load / watts | | | | | |
| Cost per 1M tokens | | | | | |
| Quality | | | | | |

---

## Claude prompt

Paste into a fresh Claude Code session in this directory:

```text
You are my engineering partner on an LLM Inference Optimization Lab.
Project dir: ~/Documents/learning-ai/projects/inference-optimization-lab
Read PLAN.md first; it is the source of truth.

GOAL
Benchmark an enterprise-style RAG assistant (corpus: my lecture flash notes in
~/Documents/learning-ai/all-notes-files) served locally on an Apple M4 Max
(64 GB unified memory) with llama.cpp, then optimize it one change at a time
and prove each improvement with measurements.

BUILD FIRST (stop and show me each before moving on)
1. Frozen dataset: 200 questions with retrieved context PRECOMPUTED into
   data/workload.jsonl (80% Q&A max_tokens=150, 20% summaries max_tokens=600),
   plus a 50-question quality set with reference answers.
2. Load generator: async Python, OpenAI-compatible streaming, temperature 0,
   closed-loop concurrency sweep {1,4,8,16,32} and open-loop Poisson mode.
   60s warm-up, 5 min measured, 3 repeats per configuration.
3. Per-request logging to results/<run_id>/requests.jsonl: TTFT, TPOT, E2E,
   prompt/output tokens, cached tokens. Server counters and powermetrics
   sampled to the same run dir. Every run records its full config + a hash.
4. Analysis script that computes p50/p95, throughput, goodput (SLO:
   TTFT p95 < 1s AND TPOT p95 < 50ms), max users at SLO, cost per 1M tokens
   (state assumptions), quality score, and writes a comparison table with
   BOTH incremental Δ% (vs previous step) and cumulative Δ% (vs baseline).
5. Prometheus + Grafana via docker compose for live views.

EXPERIMENT LADDER (exactly one change per run, lossless before lossy)
0 baseline (1 slot, F16, no prompt cache) → 1 prefix caching →
2 parallel slots/continuous batching → 3 flash attention →
4 speculative decoding → 5 Q4 weights → 6 KV-cache q8_0

RULES
- Never compute or estimate numbers in chat; every number comes from the
  analysis script output, which you quote.
- An improvement counts only if it exceeds 2x the std dev across repeats;
  otherwise call it noise.
- Every "why" must cite a measured counter (cache hit tokens, acceptance rate,
  batch size, memory). If a result contradicts expectations, say so and
  investigate before moving on.
- Lossy steps must pass the quality gate (drop ≤ 3 points); lossless steps
  should match baseline outputs at temperature 0. Report any mismatch.
- After each experiment, stop and give me: what changed, the comparison
  table, the evidence, and your recommendation to keep or revert.

Start with step 1. Ask me before installing anything system-wide.
```

---

## Final deliverable

A mini **LLM Inference Optimization Lab** case study: reproducible workload, per-experiment results with evidence, a baseline → final comparison table, throughput vs P95 latency curves, and the headline "max users at SLO" improvement.
