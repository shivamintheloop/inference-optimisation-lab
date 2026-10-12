#!/usr/bin/env bash
# Short llama-bench runs that explain (and predict) speculative decoding results.
# Run on a cool, idle machine with llama-server stopped. Each test is seconds.
#
#   scripts/micro-bench.sh                              # Llama 3.1 8B Q4_K_M + Llama 3.2 1B
#   TARGET=llama3.1:8b-instruct-q8_0 scripts/micro-bench.sh   # any Ollama name or .gguf path
#   SKIP_DRAFT=1 TARGET=... scripts/micro-bench.sh      # target only
#
# 1. Target "check" cost: process 1..32 new tokens in one pass after 1024
#    tokens of context. A spec-decoding check of k drafts is this pass with k+1
#    tokens. Theory on a bandwidth-bound GPU: about the same time for 1 or 16.
# 2. Target decode: one normal step (the baseline cost per token).
# 3. Draft decode: one draft step (c = draft step / target step).
set -euo pipefail
cd "$(dirname "$0")/.."

blob() {
  if [[ $1 == *.gguf ]]; then echo "$1"; else ollama show "$1" --modelfile | awk '/^FROM \//{print $2}'; fi
}
target=${TARGET:-llama3.1:8b}
draft=${DRAFT:-llama3.2:1b}
big=$(blob "$target")
slug=$(echo "$target" | tr -c 'a-zA-Z0-9.\n' '-')
out=results/micro-bench-$(date +%Y%m%d-%H%M%S)-$slug.md

{
  echo "# llama-bench micro-measurements ($(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "target $target, draft $draft; llama.cpp $(llama-server --version 2>&1 | grep version); -ngl 99, depth 1024, 5 repetitions."
  echo
  echo "## Target: one pass over N new tokens (pp N = check cost for k = N-1 drafts)"
  llama-bench -m "$big" -ngl 99 -d 1024 -p 1,2,3,4,5,6,8,9,12,16,17,24,32 -n 0 -r 5
  echo
  echo "## Target decode (tg = one normal step per token)"
  llama-bench -m "$big" -ngl 99 -d 1024 -p 0 -n 32 -r 5
  if [[ -z ${SKIP_DRAFT:-} ]]; then
    echo
    echo "## Draft decode (the draft model's step)"
    llama-bench -m "$(blob "$draft")" -ngl 99 -d 1024 -p 0 -n 32 -r 5
  fi
} 2>/dev/null | tee "$out"
echo "saved $out"
