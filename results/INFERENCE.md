# Inference experiments

Appended by `python -m rag bench-llm --record`; never edited by hand. Every row runs the same frozen prompts (`bench/prompts-v1.jsonl`, 24 answerable eval questions) one at a time, 3 passes, temperature 0, prefix cache off.

- **TTFT / TPOT**: time to first token / time per output token; median of the per-pass p50s, with the min–max of those p50s (run-to-run noise)
- **Decode tok/s**: the engine's own decode speed
- **Speedup**: baseline TPOT ÷ this TPOT
- **Answer time**: p50 time for the whole LLM answer
- **Acceptance**: share of drafted tokens the big model kept
- **Tokens/check**: tokens produced per big-model pass (kept guesses + 1 of its own)
- **Same as baseline**: answers identical to the baseline run (should be all, at temperature 0)
- **Stable**: answers identical across this run's own passes
- **Server RSS**: llama-server memory

| Variant | Engine | k | TTFT p50 | TPOT p50 (range) | Decode tok/s | Speedup | Answer time p50 | Acceptance | Tokens/check | Same as baseline | Stable | Server RSS | Raw | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (8B alone) | llama_server b11146-7fe450e19 | – | 3153 ms | 19.9 ms (13.3–27.7) | 52 | – | 5.68 s | – | – | – | 24/24 | 9734 MB | `llm-baseline--8b-alone-20261004-101945.json` | fair before: same engine, no drafts |
| baseline (8B alone) AC | llama_server b11146-7fe450e19 | – | 3238 ms | 22.9 ms (14.5–34.7) | 44 | – | 6.44 s | – | – | – | 24/24 | 9589 MB | `llm-baseline--8b-alone--ac-20261004-140556.json` | NOISY, do not use: on AC power, still drifts (sustained-load slowdown: decode 85 -> 26 tok/s in 2 min at GPU 98%); pass 0 clean at ~14 ms |
| baseline (A/B) | llama_server b11146-7fe450e19 | – | 1310 ms | 11.9 ms (11.9–11.9) | 84 | – | 2.78 s | – | – | – | 24/24 | 5869 MB | `ab-20261004-153955.json` | A/B interleaved vs baseline, 1 passes, idle 120 s; probe 84–85 tok/s; block drift max 1.12× |
| draft:3 (A/B) | llama_server b11146-7fe450e19 | 3 | 1467 ms | 20.4 ms (20.4–20.4) | 49 | 0.59× | 4.16 s | 0.66 | 2.97 | 21/24 | 24/24 | 7428 MB | `ab-20261004-153955.json` | A/B interleaved vs baseline, 1 passes, idle 120 s; probe 68–70 tok/s; block drift max 1.39× |
| ngram:16:12 (A/B) | llama_server b11146-7fe450e19 | 16 | 1313 ms | 11.9 ms (11.9–11.9) | 84 | 1.00× | 2.75 s | 0.15 | 8.26 | 24/24 | 24/24 | 5865 MB | `ab-20261004-153955.json` | A/B interleaved vs baseline, 1 passes, idle 120 s; probe 85–86 tok/s; block drift max 1.10× |
| baseline@t=f16 (A/B) | llama_server b11146-7fe450e19 | – | 1202 ms | 31.5 ms (31.5–31.5) | 32 | – | 5.84 s | – | – | – | 24/24 | 16495 MB | `ab-20261004-184037.json` | prompts-v1.jsonl (24); A/B interleaved vs baseline@t=f16, 1 passes, idle 120 s; probe 32–32 tok/s; block drift max 1.02× |
| draft:3@t=f16,d=1bq4 (A/B) | llama_server b11146-7fe450e19 | 3 | 1380 ms | 17.5 ms (17.5–17.5) | 58 | 1.80× | 3.89 s | 0.62 | 2.88 | 24/24 | 24/24 | 17573 MB | `ab-20261004-184037.json` | prompts-v1.jsonl (24); A/B interleaved vs baseline@t=f16, 1 passes, idle 120 s; probe 81–82 tok/s; block drift max 1.12× |
| baseline (A/B) | llama_server b11146-7fe450e19 | – | 1292 ms | 12.0 ms (12.0–12.0) | 83 | – | 3.48 s | – | – | – | 24/24 | 5801 MB | `ab-20261004-191423.json` | prompts-extract-v1.jsonl (24); A/B interleaved vs baseline, 1 passes, idle 120 s; probe 84–86 tok/s; block drift max 1.05× |
| ngram:24:4 (A/B) | llama_server b11146-7fe450e19 | 24 | 1282 ms | 6.9 ms (6.9–6.9) | 150 | 1.75× | 2.44 s | 0.47 | 12.15 | 23/24 | 24/24 | 5867 MB | `ab-20261004-191423.json` | prompts-extract-v1.jsonl (24); A/B interleaved vs baseline, 1 passes, idle 120 s; probe 83–85 tok/s; block drift max 1.65× |
| MLX baseline (A/B) | mlx 0.31.3 | – | 1321 ms | 10.5 ms (10.5–10.5) | 95 | – | 2.19 s | – | – | – | 24/24 | – | `mlx-ab-20261004-194714.json` | prompts-v1.jsonl (24); MLX A/B vs baseline, idle 120 s; tokens from kept guesses –; block drift max 1.02× |
| MLX draft:2 (A/B) | mlx 0.31.3 | 2 | 1538 ms | 8.4 ms (8.4–8.4) | 120 | 1.26× | 2.29 s | – | – | 23/24 | 24/24 | – | `mlx-ab-20261004-194714.json` | prompts-v1.jsonl (24); MLX A/B vs baseline, idle 120 s; tokens from kept guesses 0.578; block drift max 1.23× |
| MLX baseline (A/B) | mlx 0.31.3 | – | 1304 ms | 10.5 ms (10.5–10.5) | 96 | – | 3.15 s | – | – | – | 24/24 | – | `mlx-ab-20261004-203659.json` | prompts-extract-v1.jsonl (24); MLX A/B vs baseline, idle 120 s; tokens from kept guesses –; block drift max 1.03× |
| MLX draft:2 (A/B) | mlx 0.31.3 | 2 | 1492 ms | 7.3 ms (7.3–7.3) | 138 | 1.43× | 2.67 s | – | – | 21/24 | 24/24 | – | `mlx-ab-20261004-203659.json` | prompts-extract-v1.jsonl (24); MLX A/B vs baseline, idle 120 s; tokens from kept guesses 0.631; block drift max 1.09× |
| MLX draft:4 (A/B) | mlx 0.31.3 | 4 | 1528 ms | 7.6 ms (7.6–7.6) | 133 | 1.38× | 2.82 s | – | – | 22/24 | 24/24 | – | `mlx-ab-20261004-203659.json` | prompts-extract-v1.jsonl (24); MLX A/B vs baseline, idle 120 s; tokens from kept guesses 0.753; block drift max 1.11× |
