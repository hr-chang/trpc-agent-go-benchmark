#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <native|agentadapt> <1|2|3>" >&2
  exit 2
fi

arm=$1
round=$2
case "$arm" in
  native|agentadapt) ;;
  *) echo "invalid arm: $arm" >&2; exit 2 ;;
esac
case "$round" in
  1|2|3) ;;
  *) echo "invalid round: $round" >&2; exit 2 ;;
esac

experiment_id=cost-growth-hot12-20260727
root=/data/validation/results/cost-growth-hot12-20260727
case_list="$root/hot12.case_ids.txt"
cases=/data/validation/trpc-agent-go-benchmark/swebench/data/generated/cases.jsonl
model_config=/data/validation/trpc-agent-go-benchmark/swebench/config/models/glm-5.2.local.yaml
environment_config=/data/validation/trpc-agent-go-benchmark/swebench/config/environments/swebench-testbed.yaml
embedding_config=/data/validation/trpc-agent-go-benchmark/swebench/config/embeddings/workspace-rag.cache.local.yaml

verify_sha256() {
  local expected=$1
  local file=$2
  local actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $file expected=$expected actual=$actual" >&2
    exit 3
  fi
}

verify_sha256 2219d42a35636e646e913145714c589575c1b75c25b46dea13fd5855917cffc8 "$case_list"
verify_sha256 4b2a050a82d356963320cbfa8e2efdf6a133af8863f31b291a973ab4dd349d07 "$cases"
verify_sha256 fbfdf25e9fced3ecc51d244ccbb652c078b02ac640abbdbcca53ddbaa7af27de "$model_config"
verify_sha256 3cfa72f92f4010d242e6adb9bb507ccdf9db261ba5726163b5adecd509c140f0 "$environment_config"

run_id="tag-cost-hot12-${arm}-20260727-r${round}"
run_dir="$root/$run_id"
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing run directory: $run_dir" >&2
  exit 4
fi
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

if [[ "$arm" == native ]]; then
  binary=/data/validation/bin/trpc-agent-go-impl-ast-agent54-1532343
  verify_sha256 a4840b3e05110b7b3a861a58cbb59ee445279c3e7436b77298ca5e9b0135060c "$binary"
  cd /data/validation/trpc-agent-go-benchmark/swebench
  /usr/bin/time -v "$binary" \
    --run-id "$run_id" \
    --cases "$cases" \
    --case-list "$case_list" \
    --model-config "$model_config" \
    --environment-config "$environment_config" \
    --output "$run_dir/raw/tag" \
    --agent-workers 15 \
    --command-timeout 1m \
    --case-timeout 2h \
    --observation-codec xml \
    --billing-tag "$run_id" \
    --experiment-id "$experiment_id" \
    --framework-revision 9c4a839314a2f0f0b782196e52a45c733a94f213 \
    --code-search=false \
    --workspace-representation=current-fixed \
    >"$run_dir/runner.log" 2>"$run_dir/time.txt"
else
  binary=/data/validation/bin/trpc-agent-go-impl-rag-agentadapt-aad40c5
  verify_sha256 b282a16fd50ea4ec054e9684c441729de98c4c437b7f741645a2f4ababd6f627 "$binary"
  verify_sha256 2452624a6a38fbd6f414a70b0c66e48653204662fe1b96c9026350bcc6d79bd1 "$embedding_config"
  cd /data/validation/worktrees/rag-agentadapt-benchmark-aad40c5/swebench
  /usr/bin/time -v "$binary" \
    --run-id "$run_id" \
    --cases "$cases" \
    --case-list "$case_list" \
    --model-config "$model_config" \
    --embedding-config "$embedding_config" \
    --environment-config "$environment_config" \
    --output "$run_dir/raw/tag" \
    --agent-workers 6 \
    --observation-codec xml \
    --billing-tag "$run_id" \
    --experiment-id "$experiment_id" \
    --framework-revision 358d376784889fd539a764190e1970ee4f4fc5f6 \
    --code-search=true \
    --workspace-preload=false \
    --workspace-representation=ast-structured \
    --command-timeout 1m \
    --case-timeout 2h \
    >"$run_dir/runner.log" 2>"$run_dir/time.txt"
fi

date --iso-8601=seconds >"$run_dir/completed_at.txt"
