# Speculative decoding on a laptop: when does it pay off?

Living analysis, updated after every experiment (newest results at the bottom of
"Experiment log"). Raw data: `results/INFERENCE.md`, `results/ab-*.json`,
`results/micro-bench-*.md`. Machine: MacBook M4 Max, 64 GB, AC power.

## The idea in one paragraph

A small "draft" model (or a lookup in text already seen) guesses the next k
tokens. The big "target" model checks all k guesses in **one** pass, keeps the
ones it agrees with, and adds one token of its own. At temperature 0 the answer
is the target's own answer; only the speed changes.

## Summary: where it wins (and why)

| Case (Llama 3.1 8B target, Llama 3.2 1B draft) | Engine | Per token | Whole answer | Same text | Status |
|---|---|---|---|---|---|
| Normal RAG answers, 8B **Q4_K_M** + 1B draft k=3 | llama.cpp | 0.59× ❌ | 0.72× | 21/24 | 24 prompts |
| Normal RAG answers, 8B **F16** + 1B Q4 draft k=3 | llama.cpp | **1.80×** ✅ | **1.50×** | 24/24 | 24 prompts |
| "Quote the evidence" answers, 8B Q4_K_M + n-gram m=24 n=4 | llama.cpp | **1.75×** ✅ | **1.37×** | 23/24 | 24 prompts |
| Normal RAG answers, 8B 4-bit + 1B 4-bit draft k=2 | MLX | **1.26×** ✅ | 0.99× (1.08× for answers ≥120 tokens) | 23/24 | 24 prompts |
| "Quote the evidence" answers, 8B 4-bit + 1B 4-bit draft k=2 | MLX | **1.43×** ✅ | **1.14×** | 21/24 | 24 prompts |

One rule explains every row: **speculative decoding pays off when checking a
few extra tokens costs (almost) nothing extra.** That is true when the target's
step is limited by *reading weights from memory* (F16: 15 GB per token) or when
the engine's small-batch code is efficient (MLX), and false when 4-bit weights
make the step limited by *arithmetic* (llama.cpp Q4). Then it needs answers that
copy their sources, so one check keeps many tokens (n-gram on quotes).

## The speedup formula (and the part that was wrong)

    time per token = (k × draft_step + check_pass(k+1)) / tokens_per_check

The textbook assumption is `check_pass(k+1) ≈ one normal step`, because one
request is limited by reading the weights from memory, and a check reads them
once. **Measured on llama.cpp b11146 Metal with the 8B in Q4_K_M, it isn't:**

| 8B pass over N tokens | 1 | 2 | 4 | 8 | 9–32 |
|---|---|---|---|---|---|
| ms | 12.5 | 17.6 | 38.0 | 72.0 | ~56 (flat) |

So a speedup needs either a cheap check (a different quantization, engine or
bigger model) or very many kept tokens per check (copy-heavy answers).

## Results so far

Starting point (experiment 1, `results/ab-20261004-153955.json`, 24 RAG prompts):

| Setup | TPOT | Speedup | Guesses kept | Same text |
|---|---|---|---|---|
| 8B Q4_K_M alone | 11.9 ms | 1.00× | – | – |
| + 1B draft k=3 | 20.4 ms | 0.59× | 66% | 21/24 |
| + n-gram (default size-m, n=12) | 11.9 ms | 1.00× | 15% | 24/24 |

Correction found later: `ngram-simple` ignores `--spec-draft-n-max`; its guess
length is `--spec-ngram-simple-size-m`. The "k=16" in that run was really the
default length (1448 guesses over 31 checks ≈ 47 per check).

## Experiment log

Plan, cheapest and most likely to work first:

1. **Copy-heavy RAG workload** (`bench/prompts-extract-v1.jsonl`: same 24 prompts,
   but "quote the evidence word for word"). Many kept tokens per check.
2. **Other 8B quantizations** (Q4_0, Q8_0, F16): is the multi-token check cheaper?
3. **Draft tuning:** stop guessing when the 1B is unsure (`--spec-draft-p-min`),
   cheaper 1B quantization.
4. **Another engine** (MLX-LM) with its own Apple GPU code.
5. **A bigger target** where a 1B guess is a small share of a step.

Each gets the 1-minute `scripts/micro-bench.sh` first when the model changes.

### Loop 1: copy-heavy answers ("quote the evidence") — first speedup ✅

Screening, 8 extract prompts, 4-prompt blocks, 2 min cool-down, rotated
(`results/ab-20261004-170505.json`):

| Setup | TPOT p50 | Speedup (TPOT) | Speedup (whole answer) | Guesses kept | Tokens per check | Same text |
|---|---|---|---|---|---|---|
| 8B alone | 12.2 ms | 1.00× | 1.00× | – | – | – |
| **n-gram m=24, n=4** | **8.2 ms** | **1.60×** | **1.30×** | 45% | 11.5 | 8/8 |
| n-gram m=16, n=4 | 8.7 ms | 1.45× | 1.24× | 58% | 10.2 | 8/8 |
| n-gram m=16, n=12 | 10.1 ms | 1.27× | 1.15× | 61% | 10.8 | 8/8 |
| 1B draft k=3 | 17.1 ms | 0.72× | 0.78× | 84% | 3.5 | 8/8 |
| 1B draft k=8 | 16.2 ms | 0.82× | 0.85× | 70% | 6.6 | 7/8 |

Why n-gram wins and the 1B still loses, with the same check-cost table:

- **N-gram guesses are free and long.** A check of 25 tokens costs ~56 ms (the
  flat zone), and when the answer is copying a sentence it keeps ~11 tokens:
  56 / 11.5 ≈ 5 ms per token on those stretches; plain steps elsewhere.
- **Longer guesses (m=24 vs 16) help** because 17 vs 25 tokens cost the same
  in the flat zone; more of a copied sentence fits in one check.
- **Shorter lookup (n=4 vs 12) helps**: a 4-token match starts copying sooner
  in each quoted sentence. It also guesses wrong more (45% vs 61% kept), but a
  wrong n-gram guess only wastes a check, not draft-model time.
- **The 1B keeps 84% of its guesses here**, the best it has ever done, yet k=3
  caps a check at 4 tokens: (3 × 3.8 + 38) / 3.5 ≈ 14 ms + overhead > 12.2 ms.
  k=8 lands in the flat zone but pays 8 × 3.8 ms of drafting per check.
- **The whole answer gains less (1.3×)** because prefill (reading the ~1,300
  prompt tokens) doesn't change; only decode speeds up.
- Block drift up to 1.6× for the n-gram setups is the questions, not heat
  (baseline drift ≤ 1.02): how much an answer copies varies per question.

**Confirmed on all 24 extract prompts** (`results/ab-20261004-191423.json`, recorded in `results/INFERENCE.md`):

| 8B Q4_K_M + … | TPOT p50 | Speedup (TPOT) | Speedup (whole answer) | Guesses kept | Tokens per check | Same text |
|---|---|---|---|---|---|---|
| alone | 12.0 ms | 1.00× | 1.00× | – | – | – |
| **n-gram m=24, n=4** | **6.9 ms** | **1.75×** | **1.37×** | 47% | 12.2 | 23/24 |

Per answer it ranges from ~1.0× to 3.6× (best: 3.3 ms per token), depending
on how long the quoted stretches are. Baseline drift ≤ 1.05.

### Loop 2: another engine (MLX-LM 0.31, same Meta models in MLX 4-bit)

Micro-bench, depth 1024 (`rag mlx-micro`, `results/mlx-micro-20261004-170556.json`):

| Pass over N tokens, in normal steps | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16–32 |
|---|---|---|---|---|---|---|---|---|---|
| llama.cpp, Q4_K_M | 1.0 | 1.4 | 1.9 | 3.0 | 4.2 | 5.8 | 4.4 | 4.5 | ~4.5 |
| **MLX, 4-bit** | 1.0 | **1.1** | **1.2** | **1.5** | 2.1 | 2.6 | 3.3 | 3.9 | ~4.2 |

- Normal step: MLX 11.4 ms, llama.cpp 12.5 ms (similar: both near the memory limit).
- MLX 1B 4-bit draft step: 2.7 ms, c = 0.24 (llama.cpp 1B Q8_0: 3.8 ms, 0.31).
- **MLX checks 2–4 tokens almost for free**, which is what speculative
  decoding needs. With ~70% of guesses kept, the formula predicts
  k=1: 1.3×, k=2: 1.3×, k=3: 1.15×, k=4: ~1.0× on normal RAG answers.

Screening, 8 **normal** RAG prompts (`rag mlx-ab --limit 8`, `results/mlx-ab-20261004-173329.json`):

| MLX setup | TPOT p50 | Speedup (TPOT) | Speedup (whole answer) | Tokens from kept guesses | TTFT p50 | Same text |
|---|---|---|---|---|---|---|
| 8B alone | 10.5 ms | 1.00× | 1.00× | – | 1378 ms | – |
| + 1B draft k=1 | 8.8 ms | 1.21× | 1.01× | 44% | 1612 ms | 7/8 |
| **+ 1B draft k=2** | **8.6 ms** | **1.27×** | 1.01× | 57% | 1613 ms | 7/8 |
| + 1B draft k=3 | 9.0 ms | 1.21× | 0.97× | 63% | 1601 ms | 7/8 |
| + 1B draft k=4 | 9.6 ms | 1.10× | 0.94× | 67% | 1606 ms | 7/8 |

- **First speedup on normal RAG answers.** The measured order (k=2 best, then
  k=1/k=3, k=4 fading) matches the micro-bench prediction.
- MLX alone (10.5 ms) is already 12% faster per token than llama.cpp (11.9 ms).
- **But the whole answer barely moves:** the 1B must also read the ~1,160-token
  prompt before its first guess, +230 ms TTFT. These 8 answers are short (p50
  ~65 tokens), so decode saves only ~65 × 1.9 ms ≈ 125 ms. Break-even is
  ~120 output tokens; longer answers gain, shorter ones lose.
- The one different answer (q05) is a near-tie flip again ("what you've learned" → "information").

**Confirmed on all 24 normal prompts** (`results/mlx-ab-20261004-194714.json`, recorded in `results/INFERENCE.md`):

| MLX setup | TPOT p50 | Speedup (TPOT) | Speedup (whole answer) | Tokens from kept guesses | TTFT p50 | Same text |
|---|---|---|---|---|---|---|
| 8B alone | 10.5 ms | 1.00× | 1.00× | – | 1315 ms | – |
| **+ 1B draft k=2** | **8.4 ms** | **1.26×** | 0.99× | 58% | 1524 ms | 23/24 |

Whole-answer speedup by answer length (per question, paired):

| Output tokens | < 120 (18 answers) | ≥ 120 (6 answers) | 200+ (q02, q14, q17) |
|---|---|---|---|
| Median answer speedup | 0.96× | 1.08× | 1.10–1.12× |

- The draft's own prefill (+210 ms TTFT) is a fixed cost per answer; the
  decode saving grows with length. This MLX 4-bit 8B writes shorter answers
  (p50 77 tokens) than llama.cpp's Q4_K_M (~130), so most answers are below
  break-even.
- Ideas to remove that fixed cost: reuse a prompt cache for the draft across
  requests (only the shared system prompt would hit), or give the draft only
  the last few hundred prompt tokens (cheaper prefill, probably lower acceptance).

**MLX on "quote the evidence" prompts, 24 prompts** (`results/mlx-ab-20261004-203659.json`, recorded in `results/INFERENCE.md`):

| MLX setup | TPOT p50 | Speedup (TPOT) | Speedup (whole answer) | Tokens from kept guesses | TTFT p50 | Output p50 | Same text |
|---|---|---|---|---|---|---|---|
| 8B alone | 10.5 ms | 1.00× | 1.00× | – | 1298 ms | 175 | – |
| **+ 1B draft k=2** | **7.3 ms** | **1.43×** | **1.14×** | 63% | 1487 ms | 172 | 21/24 |
| + 1B draft k=4 | 7.6 ms | 1.38× | 1.11× | 75% | 1518 ms | 175 | 22/24 |

- Copy-heavy answers raise the share of tokens that come from kept guesses
  (58% → 63% at k=2) and are long (~175 tokens), so the draft's fixed
  prefill cost (+190 ms) is covered: the whole answer gets faster too.
- k=4 keeps more guesses (75%) but is no faster: at near-full acceptance
  k=2 costs (2 × 2.7 + 14.0) / 3 ≈ 6.5 ms per token and k=4 (4 × 2.7 + 20.0) / 5 ≈ 6.2 ms,
  plus the misses cost more at k=4.

## Conclusions

1. **Measure the check cost first.** One minute of `scripts/micro-bench.sh` /
   `rag mlx-micro` predicted the sign and roughly the size of every result
   here. The textbook "checking k tokens costs one step" held for F16 on
   llama.cpp (1.03–1.07 steps for 2–4 tokens), roughly for MLX 4-bit
   (1.1–1.5), and not for llama.cpp 4-bit (1.4–3.0).
2. **Speculative decoding and 4-bit quantization compete for the same
   headroom.** Both attack the memory-read limit. After Q4, llama.cpp's step is
   arithmetic-bound and extra tokens cost real time. F16 + draft (17.5 ms) is
   1.8× faster than F16 alone but still slower than Q4 alone (11.9 ms).
3. **The workload matters as much as the model.** Answers that quote their
   sources let free n-gram guesses fill whole checks (12 tokens per check),
   giving 1.75× even where the 1B draft loses.
4. **Count the whole answer, not just TPOT.** The draft model reads the prompt
   too (+160 to +230 ms TTFT). Short answers can end up slower overall.
5. **Lossless in practice, with one caveat.** 158 of 168 compared answers were
   identical. The rest differ by one near-tie word (batched vs single-token
   rounding), never in meaning.

Best setup per goal on this laptop:

| Goal | Setup | Per token | Whole answer |
|---|---|---|---|
| Fastest normal answers | MLX 8B 4-bit + 1B draft k=2 | 8.4 ms (1.26× vs MLX alone, 1.42× vs llama.cpp Q4) | ~equal to MLX alone |
| Fastest quote-the-evidence answers | llama.cpp Q4_K_M + n-gram m=24 n=4 | 6.9 ms (1.75×) | 1.37× |
| Full-precision quality, faster | llama.cpp F16 + 1B Q4 draft k=3 | 17.5 ms (1.80×) | 1.50× |

Not tried (stopped here on request): 8B Q8_0 + draft (micro-bench predicts
~1.4×), draft p-min tuning, a bigger target (Gemma 3 27B + 1B), reusing the
draft's prompt cache, and behaviour under concurrent requests.

### Loop 3: other 8B quantizations on llama.cpp (micro-bench first)

`scripts/micro-bench.sh` per format (`results/micro-bench-20261004-1734*.md`). Check cost in normal steps:

| 8B format (size) | Normal step | 2 tokens | 3 | 4 | 8 | 16–32 |
|---|---|---|---|---|---|---|
| Q4_K_M (4.6 GB) | 12.5 ms | 1.4 | 1.9 | 3.0 | 5.8 | ~4.5 |
| Q4_0 (4.4 GB) | 12.1 ms | 1.5 | 2.0 | 2.6 | 5.0 | ~4.3 |
| Q8_0 (8.0 GB) | 18.9 ms | 1.06 | 1.3 | 1.7 | 3.2 | ~3.0 |
| **F16 (15 GB)** | 32.3 ms | **1.03** | **1.05** | **1.07** | 2.1 | **~1.85** |

1B draft in Q4_K_M: 355 tok/s = 2.8 ms per guess (Q8_0: 3.8 ms).

**The key pattern:** the extra cost of checking more tokens is mostly
arithmetic (unpacking weights, multiplying), and that cost is about the same in
every format. What changes is the normal step: reading 15 GB (F16) takes 32 ms,
so the extra arithmetic for 4 tokens hides under the memory read. Reading
4.6 GB (Q4) takes 12 ms, so the same arithmetic shows. **Speculative decoding
pays off when the target step is memory-bound; heavy quantization makes the 8B
fast and compute-bound, leaving little room.** Q4 and speculative decoding
compete for the same headroom.

Predictions with ~70% of guesses kept and the 1B Q4 draft (2.8 ms):
- F16, k=3: (3 × 2.8 + 33.8) / 2.53 ≈ 16.7 ms vs 32.3, **~1.9×**.
- Q8_0, k=1–2: ~13–13.7 ms vs 18.9, ~1.4×.
- Q4_K_M with the cheaper 1B, k=1: ~12 ms vs 11.9, break-even.
- In absolute terms F16 + draft (~17 ms) is still slower than Q4_K_M alone
  (11.9 ms). The win is "full-precision answers at ~2× the speed", not "faster than Q4".

Screening, F16 8B target, 8 **normal** RAG prompts (`results/ab-20261004-180455.json`):

| F16 8B + … | TPOT p50 | Speedup (TPOT) | Speedup (whole answer) | Guesses kept | Tokens per check | Same text |
|---|---|---|---|---|---|---|
| alone | 31.6 ms | 1.00× | 1.00× | – | – | – |
| 1B Q4 draft k=2 | 18.6 ms | 1.71× | 1.44× | 70% | 2.41 | 8/8 |
| **1B Q4 draft k=3** | **17.8 ms** | **1.78×** | **1.50×** | 62% | 2.86 | **8/8** |
| 1B Q4 draft k=4 | 20.4 ms | 1.58× | 1.40× | 57% | 3.26 | 8/8 |
| n-gram m=24 n=4 | 28.4 ms | 1.11× | 1.09× | 13% | – | 8/8 |

- **The textbook result, on normal RAG answers.** Predicted ~16.7 ms / 1.9× for
  k=3; measured 17.8 ms / 1.78×.
- k=3 is the sweet spot: a 4-token check is still ~free (1.07 steps), a 5-token
  check (k=4) already costs 1.36 steps.
- All 8 answers identical: in F16 the check and the single step round the same
  way more often than in Q4 (where 3/24 near-ties flipped).
- The whole answer gains less (1.5×) because prefill is unchanged, and the 1B
  adds its own prefill.
- N-gram barely helps on normal answers (13% kept): nothing to copy.

**Confirmed on all 24 normal RAG prompts** (`results/ab-20261004-184037.json`, recorded in `results/INFERENCE.md`):

| F16 8B + … | TPOT p50 | Speedup (TPOT) | Speedup (whole answer) | Guesses kept | Same text |
|---|---|---|---|---|---|
| alone | 31.5 ms | 1.00× | 1.00× | – | – |
| **1B Q4 draft k=3** | **17.5 ms** | **1.80×** | **1.50×** | 62.5% | **24/24** |

Heat under control: baseline block drift 0.99–1.02.
