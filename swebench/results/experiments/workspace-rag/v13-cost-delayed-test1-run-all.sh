#!/usr/bin/env bash
set -Eeuo pipefail

main=/data/validation/results/cost-growth-hot12-20260727
root=/data/validation/results/cost-delay15930-exploratory-20260727
launcher="$root/launch-one.sh"

for _ in $(seq 1 720); do
  [[ ! -f "$main/harness-all-completed-at.txt" ]] || break
  if [[ -f "$main/harness-all.pid" ]] && ! kill -0 "$(cat "$main/harness-all.pid")" 2>/dev/null; then
    echo "main harness controller exited without completion marker" >&2
    exit 10
  fi
  sleep 10
done
[[ -f "$main/harness-all-completed-at.txt" ]] || { echo "timed out waiting for main harness" >&2; exit 11; }
sleep 3
if pgrep -af '[e]valuator verify|[r]un_evaluation' >/dev/null; then
  echo "harness process still active" >&2
  exit 12
fi
[[ "$(docker ps --format '{{.Names}}' | grep -vc '^swebench-managed-httpbin$')" -eq 0 ]] || {
  echo "unexpected active experiment containers" >&2
  exit 13
}

for item in native:1 agentadapt:1 agentadapt:2 native:2 native:3 agentadapt:3; do
  "$launcher" "${item%%:*}" "${item##*:}"
done

date --iso-8601=seconds >"$root/all-completed-at.txt"
