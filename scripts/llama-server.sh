#!/usr/bin/env bash
# Start llama-server (llama.cpp) for the inference experiments. Runs in the
# foreground; Ctrl+C stops it. Uses the GGUF files Ollama already downloaded.
#
#   scripts/llama-server.sh baseline            # Llama 3.1 8B alone
#   scripts/llama-server.sh draft [k] [p]       # + Llama 3.2 1B guessing up to k tokens ahead (default 8);
#                                               #   stop guessing early when the 1B's top token
#                                               #   probability drops below p (default 0 = never)
#   scripts/llama-server.sh ngram [m] [n]       # + guesses copied from text already seen:
#                                               #   when the last n tokens appeared before, propose
#                                               #   the m tokens that followed them
#
#   TARGET=llama3.1:8b-instruct-q8_0 DRAFT=llama3.2:1b-instruct-q4_K_M scripts/llama-server.sh draft 3
#                                               # any Ollama model name or a .gguf path
#
# Shared settings: 8k context, all layers on the GPU, one slot (one request at a
# time) and /metrics on, which is where draft acceptance counts come from.
# The exact command is saved to data/llama-server.cmd for the benchmark results.
set -euo pipefail
cd "$(dirname "$0")/.."

blob() {
  if [[ $1 == *.gguf ]]; then echo "$1"; else ollama show "$1" --modelfile | awk '/^FROM \//{print $2}'; fi
}

mode=${1:-baseline}
k=${2:-8}
target=${TARGET:-llama3.1:8b}
draft=${DRAFT:-llama3.2:1b}
args=(-m "$(blob "$target")" --alias llama3.1:8b -c 8192 -ngl 99 -np 1 --metrics
      --port "${PORT:-8081}")
case $mode in
  baseline) ;;
  draft) args+=(-md "$(blob "$draft")" -ngld 99 --spec-type draft-simple --spec-draft-n-max "$k"
                --spec-draft-p-min "${3:-0}") ;;
  # ngram-simple ignores --spec-draft-n-max; its guess length is size-m
  ngram) args+=(--spec-type ngram-simple --spec-ngram-simple-size-m "$k"
                --spec-ngram-simple-size-n "${3:-12}") ;;
  *) echo "usage: $0 baseline | draft [k] [p_min] | ngram [m] [n]" >&2; exit 1 ;;
esac

mkdir -p data
echo "mode=$mode k=$k target=$target draft=$draft llama-server ${args[*]}" > data/llama-server.cmd
exec llama-server "${args[@]}"
