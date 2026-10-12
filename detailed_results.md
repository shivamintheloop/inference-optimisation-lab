# Detailed results: caveats, side findings and nits

The README keeps only the main story. This file holds everything else worth knowing before you trust or reuse a number. Raw data is in `results/`; the speculative decoding story in full is in `analysis/speculative-decoding.md`.

## How much to trust the numbers

- **Small eval.** 24 questions, so one question moves hit@5 by ~0.04. "0.96 vs 0.92" is a single question. Treat small gaps as noise.
- **Two label versions.** I found mistakes in my own eval labels after Phase 2 and fixed two questions (q13, q24). Phase 2 numbers are not directly comparable with Phase 3 and later. `RESULTS.md` shows a `questions` hash per entry for this reason.
- **Only 4 off-topic questions,** and they are very off-topic (e.g. pizza in Naples). "Refused 4/4" is promising, not proven.
- **One person's notes, one laptop, one embedding model.** The general lesson is "measure each layer", not "hybrid search is bad".
- **Single request at a time.** Every inference benchmark ran with one slot and concurrency 1. Speculative decoding is known to help less under load; I did not test that.
- **One pass per variant in the final A/B runs.** Variants were interleaved in 4-prompt blocks with 2 min cool-downs and rotated order, and baseline drift was checked per block (≤ 1.05× in the runs I kept). That controls heat, but it is not 3 independent repeats.
- **Speedup columns use two methods.** "Per token" compares TPOT p50. "Whole answer" in the analysis is a paired per-question comparison, so it does not always equal the ratio of the two answer-time p50s in `INFERENCE.md`.

## Laptop noise

- **macOS background work.** One full answer run had TPOT p50 37.5 ms (vs 13.8 ms) with *identical* answers. storage indexing and photo analysis (`mediaanalysisd`) were busy. The re-run agreed with an earlier run (13.8 vs 14.2 ms).
- **Thermal throttling.** Under sustained load, decode fell from 85 to 26 tok/s within ~2 minutes, even on AC power. Back-to-back benchmarks were unusable; the `baseline (8B alone) AC` row in `INFERENCE.md` is marked NOISY for this reason. `rag/bench_ab.py` exists to work around it (interleaved blocks, cool-downs, speed probes before each block).

## RAG side findings

- **Hybrid < BM25 cause.** I ruled out the embedding prefixes and the chunk headings; the cause is that `nomic-embed-text` blurs acronyms and names (RRF, SSE, Jev).
- **File-level hit@5 hid a bug (q05, PagedAttention).** It counted as a hit because a file that mentions PagedAttention came back, but the chunk was about autoscaling. The LLM correctly refused. Two causes:
  - BM25 uses OR, so chunks repeating "problem" and "solve" (score 11.65) beat the real PagedAttention card (5.93).
  - The paper title (with "PagedAttention") is indexed with every one of its 60 chunks, so BM25 treats the word as common.
  - Fix not done yet: keep titles/headings out of the keyword index or down-weight them; score retrieval at chunk level.
- **The 8B states false things with citations (q16).** It said GQA makes the KV cache "constant" (it shrinks it; it still grows with length) and printed a stray `[G2]` tag. The citation check cannot catch this; it needs a faithfulness judge.
- **The one wrongly refused real question (q22)** is an everyday-words retrieval miss by the re-ranker. The LLM answered faithfully from the wrong notes.
- **Re-ranker load time:** 2.4 to 3.9 s once per process, then ~45 ms per question.
- **Prompts are short.** ~1,100 tokens median, although the context budget is 3,500, because most note cards are short.
- **Decode is near the hardware limit.** Ollama decodes at ~87 tok/s; memory bandwidth caps a 4.9 GB model at ~110 tok/s on this chip (546 GB/s ÷ 4.9 GB). Prefill runs at ~970 tok/s.
- **Ingestion:** a second run embeds 0 chunks; embedding is ~88% of ingest time; the characters ÷ 4 token estimate is within 3% of the real count.

## Speculative decoding side findings

- **Where the 0.59× came from.** On llama.cpp b11146 Metal with Q4_K_M, the 8B's pass over N tokens costs 12.5 ms (1), 17.6 (2), 38.0 (4), 72.0 (8), then ~56 ms flat from 9 to 32 tokens, when llama.cpp switches to a matrix routine. A k=3 draft needs a 4-token check: 3 × 3.8 ms drafting + 38 ms check, divided by ~3 tokens kept, is worse than 12.5 ms.
- **Better guesses would not have saved it.** Per question, time per token fell as more guesses were kept (r = -0.81), but extrapolates to 15.7 ms at 100% kept, still above 11.9 ms.
- **Why quantization matters.** The *extra* cost of checking more tokens is arithmetic and is about the same in every format. F16's normal step reads 15 GB (32 ms), so the extra arithmetic hides under the memory read. Q4 reads 4.6 GB (12 ms), so it shows.
- **F16 + draft is "full precision at ~2× speed", not "faster than Q4".** 17.5 ms vs Q4 alone 11.9 ms.
- **k=3 is the sweet spot on F16.** A 4-token check is still ~free (1.07 steps); a 5-token check already costs 1.36 steps.
- **N-gram on normal answers does nothing** (13 to 15% of guesses kept): there is little to copy.
- **N-gram on quote answers:** longer guesses (m=24 vs 16) help because 17 and 25 tokens cost the same in the flat zone; a shorter lookup (n=4 vs 12) starts copying sooner. Per answer it ranges from ~1.0× to 3.6×.
- **llama.cpp flag gotcha.** `ngram-simple` ignores `--spec-draft-n-max`; its guess length is `--spec-ngram-simple-size-m`. The "k=16" in the first n-gram run was really the default length.
- **MLX draft prefill is a fixed cost.** The 1B must read the ~1,160-token prompt too (+190 to 230 ms TTFT). Break-even is ~120 output tokens. MLX's 4-bit 8B writes shorter answers (p50 77 tokens) than llama.cpp Q4_K_M (~130), so most normal answers fall below break-even: median 0.96× under 120 tokens, 1.08× at 120+, 1.10 to 1.12× at 200+.
- **MLX k=4 keeps more guesses than k=2 (75% vs 63%) but is no faster:** misses cost more at k=4.
- **"Lossless" in practice.** 158 of 168 compared answers were identical. The rest swap one near-tie word ("one-directional" vs "one-way"), because the batched check and the single-token step use different GPU routines that round differently. F16 flipped none.
- **Answer quality with a draft is unchanged.** `rag eval --answers` through llama-server with the 1B draft: refused 4/4, wrongly refused 1/24, cited 0.96, cites expected 0.88, same as Phase 4.

## What the original plan had that I did not run

`PLAN.md` planned a full ladder (prefix caching, parallel slots, flash attention, speculative decoding, Q4 weights, KV-cache quantization) under a concurrency sweep, plus a cloud vLLM day. I ran speculative decoding first and stopped there. Not tried:

- 8B Q8_0 + draft (micro-bench predicts ~1.4×)
- Draft `p_min` tuning
- A bigger target (e.g. Gemma 3 27B + 1B)
- Reusing the draft's prompt cache across requests
- Behaviour under concurrent requests
- Everything else on the `PLAN.md` ladder, and the RAG fixes listed above
