#!/usr/bin/env bash
set -Eeuo pipefail

root=/data/validation/results/cost-codesearch-repr15930-exploratory-20260727
launcher="$root/launch-one.sh"

[[ -f /data/validation/results/cost-delay15930-exploratory-20260727/all-completed-at.txt ]] || {
  echo "V13 completion marker missing" >&2
  exit 10
}
if pgrep -af '[e]valuator verify|[r]un_evaluation|[t]rpc-agent-go-impl' >/dev/null; then
  echo "conflicting evaluator or agent process active" >&2
  exit 11
fi
[[ "$(docker ps --format '{{.Names}}' | grep -vc '^swebench-managed-httpbin$')" -eq 0 ]] || {
  echo "unexpected active experiment containers" >&2
  exit 12
}

run_pair() {
  local left_arm=$1 left_round=$2 right_arm=$3 right_round=$4 left_pid right_pid
  "$launcher" "$left_arm" "$left_round" &
  left_pid=$!
  "$launcher" "$right_arm" "$right_round" &
  right_pid=$!
  wait "$left_pid"
  wait "$right_pid"
}

run_pair ast 1 fixed 1
run_pair fixed 2 ast 2
run_pair ast 3 fixed 3

date --iso-8601=seconds >"$root/all-completed-at.txt"
