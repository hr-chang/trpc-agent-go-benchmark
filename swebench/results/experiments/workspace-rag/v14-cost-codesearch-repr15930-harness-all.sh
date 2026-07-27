#!/usr/bin/env bash
set -Eeuo pipefail

root=/data/validation/results/cost-codesearch-repr15930-exploratory-20260727
launcher="$root/harness-one.sh"

[[ -f "$root/all-completed-at.txt" ]] || { echo "V14 generation completion marker missing" >&2; exit 10; }
if pgrep -af '[e]valuator verify|[r]un_evaluation|[t]rpc-agent-go-impl' >/dev/null; then
  echo "conflicting evaluator or agent process active" >&2
  exit 11
fi
[[ "$(docker ps --format '{{.Names}}' | grep -vc '^swebench-managed-httpbin$')" -eq 0 ]] || {
  echo "unexpected active experiment containers" >&2
  exit 12
}

for item in ast:1 fixed:1 fixed:2 ast:2 ast:3 fixed:3; do
  "$launcher" "${item%%:*}" "${item##*:}"
done

date --iso-8601=seconds >"$root/harness-all-completed-at.txt"
