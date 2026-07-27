#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <ast|fixed> <1|2|3>" >&2
  exit 2
fi

arm=$1
round=$2
case "$arm" in ast|fixed) ;; *) echo "invalid arm: $arm" >&2; exit 2 ;; esac
case "$round" in 1|2|3) ;; *) echo "invalid round: $round" >&2; exit 2 ;; esac

root=/data/validation/results/cost-codesearch-repr15930-exploratory-20260727
run_id="tag-cost-codesearch-repr15930-${arm}-20260727-r${round}"
run_dir="$root/$run_id"
predictions="$run_dir/raw/tag/preds.json"
output="$run_dir/local-harness-report/tag"

[[ -s "$predictions" ]] || { echo "missing predictions: $predictions" >&2; exit 3; }
[[ ! -e "$run_dir/local-harness-report" ]] || {
  echo "refusing to overwrite existing harness report: $run_dir/local-harness-report" >&2
  exit 4
}
mkdir -p "$run_dir/local-harness-report"
printf '%s\n' "$$" >"$run_dir/harness-controller.pid"

cd /data/validation/worktrees/rag-agentadapt-benchmark-aad40c5/swebench
GOWORK=/data/validation/worktrees/rag-agentadapt-build-20260724/go.work \
  /usr/bin/time -v go run ./evaluator verify \
    --run-id "$run_id" \
    --target tag \
    --predictions "$predictions" \
    --output "$output" \
    --python /data/validation/swebench-py/bin/python \
    --harness-workers 4 \
    --verifier-mode calibrated \
    --cache-level instance \
    --clean=false \
    --apply-harness-compat=true \
    --instances-from-predictions=true \
    >"$run_dir/harness-controller.log" 2>"$run_dir/harness-time.txt"

date --iso-8601=seconds >"$run_dir/harness-completed-at.txt"
