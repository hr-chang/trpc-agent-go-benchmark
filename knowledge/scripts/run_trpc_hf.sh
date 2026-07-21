#!/usr/bin/env bash

# Tencent is pleased to support the open source community by making
# trpc-agent-go available.
#
# Copyright (C) 2025 Tencent. All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_trpc_hf.sh smoke RUN_DIR PG_TABLE
  run_trpc_hf.sh baseline RUN_DIR PG_TABLE

The caller must load model and PGVector environment variables first and set an
absolute GOWORK path. The smoke mode only needs Embedding and PGVector access.
The baseline mode additionally requires explicit, independent Agent and Judge
model names, URLs, and API keys.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_env() {
  local name=$1
  [[ -n "${!name:-}" ]] || fail "required environment variable is empty: $name"
}

[[ $# -eq 3 ]] || {
  usage >&2
  exit 2
}

mode=$1
run_dir=$2
pg_table=$3
case "$mode" in
  smoke | baseline) ;;
  *)
    usage >&2
    fail "mode must be smoke or baseline"
    ;;
esac

[[ "$run_dir" = /* ]] || fail "RUN_DIR must be an absolute path"
[[ ! -e "$run_dir" ]] || fail "RUN_DIR already exists: $run_dir"
[[ -n "$pg_table" ]] || fail "PG_TABLE must not be empty"

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
script_path="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
knowledge_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
benchmark_root=$(CDPATH= cd -- "$knowledge_root/.." && pwd)
repo_root=$(git -C "$benchmark_root" rev-parse --show-superproject-working-tree)
[[ -n "$repo_root" ]] || fail "benchmark must be checked out as a submodule"

python_bin=${PYTHON_BIN:-python3}
port=${PORT:-8765}
workers=${WORKERS:-30}
evaluation_timeout=${EVALUATION_TIMEOUT:-600}
service_dir="$knowledge_root/knowledge_system/trpc_agent_go/trpc_knowledge"

require_command git
require_command go
require_command curl
[[ -x "$python_bin" ]] || require_command "$python_bin"

[[ -n "${GOWORK:-}" ]] || fail "GOWORK must point to a run-scoped go.work"
[[ "$GOWORK" = /* ]] || fail "GOWORK must be an absolute path"
[[ -f "$GOWORK" ]] || fail "GOWORK does not exist: $GOWORK"

for name in \
  EMBEDDING_MODEL \
  EMBEDDING_API_KEY \
  EMBEDDING_BASE_URL \
  PGVECTOR_HOST \
  PGVECTOR_PORT \
  PGVECTOR_USER \
  PGVECTOR_PASSWORD \
  PGVECTOR_DATABASE; do
  require_env "$name"
done

if [[ "$mode" == "baseline" ]]; then
  for name in \
    MODEL_NAME \
    OPENAI_API_KEY \
    OPENAI_BASE_URL \
    EVAL_MODEL_NAME \
    EVAL_API_KEY \
    EVAL_BASE_URL; do
    require_env "$name"
  done
  [[ "${MODEL_NAME,,}" != "${EVAL_MODEL_NAME,,}" ]] || \
    fail "Agent and Judge model names must differ"
  [[ "${OPENAI_BASE_URL%/}" != "${EVAL_BASE_URL%/}" ]] || \
    fail "Agent and Judge URLs must differ"
  [[ "$OPENAI_API_KEY" != "$EVAL_API_KEY" ]] || \
    fail "Agent and Judge API keys must differ"
fi

[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || \
  fail "framework superproject is dirty: $repo_root"
[[ -z "$(git -C "$benchmark_root" status --porcelain)" ]] || \
  fail "benchmark worktree is dirty: $benchmark_root"

if curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  fail "port $port already serves a healthy benchmark process"
fi

mkdir -p "$run_dir/bin"
export PGVECTOR_TABLE=$pg_table
export PYTHONPATH=$knowledge_root

root_head=$(git -C "$repo_root" rev-parse HEAD)
benchmark_head=$(git -C "$benchmark_root" rev-parse HEAD)
export RUN_MODE=$mode
export RUN_DIR=$run_dir
export RUNNER_PATH=$script_path
export ROOT_HEAD=$root_head
export BENCHMARK_HEAD=$benchmark_head
export PG_TABLE=$pg_table
export RUN_PORT=$port

"$python_bin" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


payload = {
    "schema_version": 1,
    "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "mode": os.environ["RUN_MODE"],
    "root_commit": os.environ["ROOT_HEAD"],
    "benchmark_commit": os.environ["BENCHMARK_HEAD"],
    "runner_sha256": digest(os.environ["RUNNER_PATH"]),
    "gowork_sha256": digest(os.environ["GOWORK"]),
    "pg_table": os.environ["PG_TABLE"],
    "port": int(os.environ["RUN_PORT"]),
    "search_mode": 0,
    "agent_model": os.environ.get("MODEL_NAME", ""),
    "agent_invoked": os.environ["RUN_MODE"] == "baseline",
    "judge_model": (
        os.environ.get("EVAL_MODEL_NAME", "")
        if os.environ["RUN_MODE"] == "baseline"
        else ""
    ),
    "judge_initialized": os.environ["RUN_MODE"] == "baseline",
    "embedding_model": os.environ["EMBEDDING_MODEL"],
}
path = Path(os.environ["RUN_DIR"]) / "run-environment.json"
path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

service_bin="$run_dir/bin/trpc_knowledge"
service_log="$run_dir/service.log"
service_pid=

cleanup() {
  if [[ -n "$service_pid" ]]; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
    service_pid=
  fi
}
trap cleanup EXIT INT TERM

echo "RUN_STAGE=build"
(
  cd "$service_dir"
  go test ./...
  go build -trimpath -o "$service_bin" .
)

echo "RUN_STAGE=start_service"
"$service_bin" \
  --port="$port" \
  --vectorstore=pgvector \
  --search-mode=0 \
  --pg-table="$pg_table" \
  >"$service_log" 2>&1 &
service_pid=$!

for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    tail -100 "$service_log" >&2
    fail "service exited before becoming healthy"
  fi
  sleep 0.5
done
curl -fsS --max-time 2 "http://127.0.0.1:$port/health" \
  >"$run_dir/health.json"
curl -fsS --max-time 10 "http://127.0.0.1:$port/config" \
  >"$run_dir/service-config.json"

if [[ "$mode" == "smoke" ]]; then
  echo "RUN_STAGE=search_smoke"
  curl -fsS --max-time 120 \
    -H 'Content-Type: application/json' \
    --data '{"query":"What is a model card?","k":4}' \
    "http://127.0.0.1:$port/search" \
    >"$run_dir/search.json"

  "$python_bin" - "$run_dir/service-config.json" "$run_dir/search.json" "$pg_table" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
search = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_table = sys.argv[3]

if config.get("vectorstore") != "pgvector":
    raise SystemExit("unexpected vector store")
if config.get("search_mode") != 0:
    raise SystemExit("search mode is not hybrid(0)")
if config.get("pg_table") != expected_table:
    raise SystemExit("PGVector table mismatch")
if not isinstance(config.get("index_document_count"), int):
    raise SystemExit("index document count is missing")
if config["index_document_count"] <= 0:
    raise SystemExit("index document count is not positive")
documents = search.get("documents")
if not isinstance(documents, list) or len(documents) != 4:
    raise SystemExit("search did not return exactly four documents")
if any(not document.get("text") for document in documents):
    raise SystemExit("search returned an empty document")
print(
    "SMOKE_VALIDATION=pass "
    f"documents={len(documents)} rows={config['index_document_count']}"
)
PY
else
  echo "RUN_STAGE=evaluate"
  output="$run_dir/result.json"
  "$python_bin" "$knowledge_root/main.py" \
    --kb=trpc-agent-go \
    --dataset=huggingface \
    --k=4 \
    --workers="$workers" \
    --timeout="$evaluation_timeout" \
    --skip-load \
    --output="$output" \
    2>&1 | tee "$run_dir/evaluation.log"

  "$python_bin" - "$output" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validation = result.get("validation") or {}
if validation.get("evidence_status") != "valid":
    reasons = validation.get("reasons") or []
    raise SystemExit(f"baseline evidence is insufficient: {reasons}")
print("BASELINE_VALIDATION=pass samples=54")
PY
fi

echo "RUN_STAGE=cleanup"
cleanup
trap - EXIT INT TERM
if curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  fail "service still responds after cleanup"
fi
echo "RUN_STAGE=complete"
