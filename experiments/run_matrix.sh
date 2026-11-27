#!/usr/bin/env bash
# Throwaway: run the abstract ablation through the real eval path (~1h15 total).
# Each run is its own Langfuse dataset run in the "abstract-ablation" experiment,
# and also lands in experiments/results/ for `experiments/ablation.py analyze`.
#
#   experiments/run_matrix.sh                 # baseline, extraction, full, full again
#   experiments/run_matrix.sh full 3          # one run, repeat 3
#   experiments/run_matrix.sh full 3 --samples 5   # extra args go to `orcha evals run`
set -euo pipefail
cd "$(dirname "$0")/.."

run() {  # run <name> <repeat> [extra args]
  echo "=== $1 rep$2 ==="
  uv run orcha evals run --ablation "$1" --repeat "$2" --langfuse "${@:3}"
}

case "${1:-all}" in
  all) run baseline 1; run extraction 1; run full 1; run full 2 ;;
  baseline|extraction|full) run "$1" "${2:-1}" "${@:3}" ;;
  *) echo "usage: $0 [baseline|extraction|full [repeat]]" >&2; exit 2 ;;
esac
uv run python experiments/ablation.py analyze
