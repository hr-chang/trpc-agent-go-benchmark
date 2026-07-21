#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#
#
"""Durable, secret-safe artifacts for knowledge benchmark runs."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import pathlib
import subprocess
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = "knowledge-eval/v2"
BASELINE_RUN_KIND = "baseline_reproduction"
BASELINE_EVIDENCE_SCOPE = "baseline_stability"


def endpoint_identity(value: Optional[str]) -> str:
    """Return an endpoint identity without credentials, query, or fragment."""
    if not value:
        return ""
    raw_value = value.strip()
    has_scheme = "://" in raw_value
    parsed = urlsplit(raw_value if has_scheme else f"//{raw_value}")
    if not parsed.hostname:
        if "@" in raw_value:
            return "invalid_endpoint"
        return raw_value.split("?", 1)[0].split("#", 1)[0]
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        hostname = f"{hostname}:{port}"
    if not has_scheme:
        return f"{hostname}{parsed.path}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def build_sample_id(
    dataset_name: str,
    question: str,
    answer: str,
    sample_index: Optional[int] = None,
) -> str:
    """Build a stable ID for a dataset QA entry."""
    payload = "\x00".join(
        (
            dataset_name,
            str(sample_index) if sample_index is not None else "",
            question,
            answer,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_output(repo: str, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo, *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def git_snapshot(repo: str) -> Dict[str, Any]:
    """Capture a repository revision without mutating the worktree."""
    if not repo:
        return {}
    commit = _git_output(repo, "rev-parse", "HEAD")
    if not commit:
        return {}
    status = _git_output(repo, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git_output(repo, "branch", "--show-current"),
        "dirty": bool(status),
    }


def repository_provenance(benchmark_root: str) -> Dict[str, Any]:
    """Capture benchmark and optional superproject revisions."""
    result: Dict[str, Any] = {
        "benchmark": git_snapshot(benchmark_root),
    }
    superproject = _git_output(
        benchmark_root,
        "rev-parse",
        "--show-superproject-working-tree",
    )
    if superproject:
        result["framework_superproject"] = git_snapshot(superproject)
    return result


def _stable_manifest_fields(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Select fields that must match when reusing a Q&A checkpoint."""
    return {
        "schema_version": manifest.get("schema_version"),
        "run_kind": manifest.get("run_kind"),
        "evidence_scope": manifest.get("evidence_scope"),
        "knowledge_base": manifest.get("knowledge_base"),
        "dataset": manifest.get("dataset"),
        "evaluator": manifest.get("evaluator"),
        "retrieval_k": manifest.get("retrieval_k"),
        "skip_load": manifest.get("skip_load"),
        "models": manifest.get("models"),
        "endpoints": manifest.get("endpoints"),
        "gateway_header_names": manifest.get("gateway_header_names"),
        "prompt_max_searches": manifest.get("prompt_max_searches"),
        "hard_max_tool_iterations": manifest.get(
            "hard_max_tool_iterations"
        ),
        "runtime_config": manifest.get("runtime_config"),
        "evaluator_runtime_config": manifest.get(
            "evaluator_runtime_config"
        ),
        "repositories": manifest.get("repositories"),
    }


def compute_run_fingerprint(
    manifest: Dict[str, Any],
    expected_sample_ids: Iterable[str],
) -> str:
    """Fingerprint the configuration and ordered sample set for checkpoint reuse."""
    payload = {
        "manifest": _stable_manifest_fields(manifest),
        "expected_sample_ids": list(expected_sample_ids),
    }
    encoded = json.dumps(
        _to_jsonable(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_samples_digest(records: List[Dict[str, Any]]) -> str:
    """Hash the complete persisted Q&A content for tamper detection."""
    encoded = json.dumps(
        _to_jsonable(records),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_paths(output_file: str) -> Dict[str, str]:
    """Return deterministic companion paths for a final result file."""
    return {
        "result": output_file,
        "manifest": f"{output_file}.manifest.json",
        "samples": f"{output_file}.samples.json",
        "diagnostics": f"{output_file}.ragas-diagnostics.jsonl",
    }


def _to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_to_jsonable(item) for item in sorted(value, key=str)]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _to_jsonable(item_method())
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        return _to_jsonable(tolist_method())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def atomic_write_json(path: str, payload: Any) -> None:
    """Atomically replace a JSON artifact in its destination directory."""
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=parent,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _to_jsonable(payload),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, absolute_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


class RunArtifactWriter:
    """Write a manifest, per-question checkpoint, and final result."""

    def __init__(
        self,
        output_file: str,
        manifest: Dict[str, Any],
        expected_sample_ids: List[str],
    ):
        self.paths = artifact_paths(output_file)
        self.manifest = dict(manifest)
        self.expected_sample_ids = list(expected_sample_ids)
        self.manifest.setdefault("schema_version", SCHEMA_VERSION)
        self.manifest.setdefault("run_id", str(uuid.uuid4()))
        self.manifest.pop("completed_at", None)
        self.manifest.pop("validation", None)
        self.manifest["artifacts"] = dict(self.paths)
        self.manifest["expected_samples"] = len(expected_sample_ids)
        self.run_fingerprint = compute_run_fingerprint(
            self.manifest,
            self.expected_sample_ids,
        )
        self.manifest["run_fingerprint"] = self.run_fingerprint
        self.manifest["execution_status"] = "running"
        self.manifest["evidence_status"] = "insufficient"
        self.write_manifest()

    def refresh_manifest(self, updates: Dict[str, Any]) -> None:
        """Refresh effective config before any Q&A samples are collected."""
        immutable = {
            "schema_version": self.manifest["schema_version"],
            "run_id": self.manifest["run_id"],
            "artifacts": dict(self.paths),
            "expected_samples": len(self.expected_sample_ids),
        }
        self.manifest.update(updates)
        self.manifest.update(immutable)
        self.run_fingerprint = compute_run_fingerprint(
            self.manifest,
            self.expected_sample_ids,
        )
        self.manifest["run_fingerprint"] = self.run_fingerprint
        self.manifest["execution_status"] = "running"
        self.manifest["evidence_status"] = "insufficient"
        self.write_manifest()

    def write_manifest(self) -> None:
        """Persist the current manifest state."""
        atomic_write_json(self.paths["manifest"], self.manifest)

    def write_samples(self, records: List[Dict[str, Any]]) -> None:
        """Persist the complete ordered Q&A checkpoint."""
        atomic_write_json(
            self.paths["samples"],
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.manifest["run_id"],
                "run_fingerprint": self.run_fingerprint,
                "manifest": self.manifest,
                "expected_sample_ids": self.expected_sample_ids,
                "samples_count": len(records),
                "samples_digest": compute_samples_digest(records),
                "samples": records,
            },
        )

    def write_result(self, payload: Dict[str, Any]) -> None:
        """Persist the final aggregate result."""
        atomic_write_json(self.paths["result"], payload)


def load_sample_checkpoint(
    path: str,
    expected_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and validate a sample checkpoint."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema: {payload.get('schema_version')!r}"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("checkpoint samples must be a list")
    if payload.get("samples_count") != len(samples):
        raise ValueError("checkpoint samples_count does not match samples")
    if payload.get("samples_digest") != compute_samples_digest(samples):
        raise ValueError("checkpoint samples_digest does not match samples")
    actual_fingerprint = payload.get("run_fingerprint")
    if expected_fingerprint and actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "checkpoint fingerprint does not match the requested run"
        )
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be an object")
    expected_sample_ids = payload.get("expected_sample_ids")
    if not isinstance(expected_sample_ids, list):
        raise ValueError("checkpoint expected_sample_ids must be a list")
    if len(expected_sample_ids) != manifest.get("expected_samples"):
        raise ValueError(
            "checkpoint expected_sample_ids does not match expected_samples"
        )
    dataset_name = manifest.get("dataset")
    sample_ids = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"checkpoint sample {index} must be an object")
        sample_id = sample.get("sample_id")
        if not sample_id:
            raise ValueError(f"checkpoint sample {index} has no sample_id")
        expected_sample_id = build_sample_id(
            str(dataset_name or ""),
            str(sample.get("question") or ""),
            str(sample.get("ground_truth") or ""),
            sample.get("sample_index"),
        )
        if sample_id != expected_sample_id:
            raise ValueError(
                f"checkpoint sample {index} identity does not match content"
            )
        sample_ids.append(sample_id)
    if sample_ids != expected_sample_ids[:len(sample_ids)]:
        raise ValueError("checkpoint samples are not the expected ordered prefix")
    computed_fingerprint = compute_run_fingerprint(
        manifest,
        expected_sample_ids,
    )
    if actual_fingerprint != computed_fingerprint:
        raise ValueError(
            "checkpoint fingerprint does not match its manifest and samples"
        )
    return payload
