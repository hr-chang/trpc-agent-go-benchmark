#!/usr/bin/env bash
set -Eeuo pipefail

root=/data/validation/results/cost-growth-hot12-20260727
launcher="$root/harness-one.sh"

for item in native:1 agentadapt:1 agentadapt:2 native:2 native:3 agentadapt:3; do
  arm=${item%%:*}
  round=${item##*:}
  "$launcher" "$arm" "$round"
done

date --iso-8601=seconds >"$root/harness-all-completed-at.txt"
