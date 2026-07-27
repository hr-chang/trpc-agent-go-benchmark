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

experiment_id=cost-codesearch-repr15930-exploratory-20260727
root=/data/validation/results/cost-codesearch-repr15930-exploratory-20260727
case_list="$root/case_ids.txt"
cases=/data/validation/trpc-agent-go-benchmark/swebench/data/generated/cases.jsonl
model_config=/data/validation/trpc-agent-go-benchmark/swebench/config/models/glm-5.2.local.yaml
environment_config=/data/validation/trpc-agent-go-benchmark/swebench/config/environments/swebench-testbed.yaml
embedding_config=/data/validation/trpc-agent-go-benchmark/swebench/config/embeddings/workspace-rag.cache.local.yaml
binary=/data/validation/bin/trpc-agent-go-impl-rag-agentadapt-aad40c5

verify_sha256() {
  local expected=$1 file=$2 actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $file expected=$expected actual=$actual" >&2; exit 3; }
}

verify_sha256 2c134c2d838b49fb7c53a51a38a6c81775c92b25f86f654ff4c44a2a0678c835 "$case_list"
verify_sha256 4b2a050a82d356963320cbfa8e2efdf6a133af8863f31b291a973ab4dd349d07 "$cases"
verify_sha256 fbfdf25e9fced3ecc51d244ccbb652c078b02ac640abbdbcca53ddbaa7af27de "$model_config"
verify_sha256 2452624a6a38fbd6f414a70b0c66e48653204662fe1b96c9026350bcc6d79bd1 "$embedding_config"
verify_sha256 3cfa72f92f4010d242e6adb9bb507ccdf9db261ba5726163b5adecd509c140f0 "$environment_config"
verify_sha256 b282a16fd50ea4ec054e9684c441729de98c4c437b7f741645a2f4ababd6f627 "$binary"

if [[ "$arm" == ast ]]; then
  representation=ast-structured
else
  representation=current-fixed
fi

run_id="tag-cost-codesearch-repr15930-${arm}-20260727-r${round}"
run_dir="$root/$run_id"
[[ ! -e "$run_dir" ]] || { echo "refusing to overwrite: $run_dir" >&2; exit 4; }
mkdir -p "$run_dir"
printf '%s\n' "$$" >"$run_dir/controller.pid"

sar_pid=
cleanup() {
  if [[ -n "$sar_pid" ]] && kill -0 "$sar_pid" 2>/dev/null; then
    kill "$sar_pid" 2>/dev/null || true
    wait "$sar_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
sar -u -r -d 60 >"$run_dir/sar.log" 2>&1 &
sar_pid=$!
printf '%s\n' "$sar_pid" >"$run_dir/sar.pid"

cd /data/validation/worktrees/rag-agentadapt-benchmark-aad40c5/swebench
/usr/bin/time -v "$binary" \
  --run-id "$run_id" --cases "$cases" --case-list "$case_list" \
  --model-config "$model_config" --embedding-config "$embedding_config" \
  --environment-config "$environment_config" --output "$run_dir/raw/tag" \
  --agent-workers 6 --observation-codec xml \
  --billing-tag "$run_id" --experiment-id "$experiment_id" \
  --framework-revision 358d376784889fd539a764190e1970ee4f4fc5f6 \
  --code-search=true --workspace-preload=false --workspace-representation="$representation" \
  --command-timeout 1m --case-timeout 2h \
  >"$run_dir/runner.log" 2>"$run_dir/time.txt"

date --iso-8601=seconds >"$run_dir/completed_at.txt"
