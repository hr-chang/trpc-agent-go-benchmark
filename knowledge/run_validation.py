#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#
#
"""Evidence validity rules for knowledge benchmark baseline runs."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from run_artifacts import BASELINE_EVIDENCE_SCOPE, BASELINE_RUN_KIND


METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "answer_similarity",
    "context_precision",
    "context_recall",
    "context_entity_recall",
)

BASELINE_DATASET = "huggingface"
BASELINE_SAMPLE_COUNT = 54


def _canonical_model_name(value: Any) -> str:
    return "".join(
        character
        for character in str(value or "").lower()
        if character.isalnum()
    )


def validate_baseline_run(
    manifest: Dict[str, Any],
    sample_records: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    evaluation_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify whether a completed run is valid baseline-stability evidence."""
    reasons: List[str] = []
    expected_samples = int(manifest.get("expected_samples") or 0)
    actual_samples = len(sample_records)

    if manifest.get("run_kind") != BASELINE_RUN_KIND:
        reasons.append("run_kind_is_not_baseline_reproduction")
    if manifest.get("evidence_scope") != BASELINE_EVIDENCE_SCOPE:
        reasons.append("evidence_scope_is_not_baseline_stability")
    if manifest.get("dataset") != BASELINE_DATASET:
        reasons.append("baseline_dataset_is_not_huggingface")
    if manifest.get("knowledge_base") != "trpc-agent-go":
        reasons.append("baseline_knowledge_base_is_not_trpc_agent_go")
    if manifest.get("evaluator") != "ragas":
        reasons.append("baseline_evaluator_is_not_ragas")
    if manifest.get("retrieval_k") != 4:
        reasons.append("baseline_retrieval_k_is_not_4")
    if expected_samples <= 0:
        reasons.append("expected_sample_count_is_missing")
    elif actual_samples != expected_samples:
        reasons.append(
            f"sample_count_mismatch:{actual_samples}/{expected_samples}"
        )
    if expected_samples != BASELINE_SAMPLE_COUNT:
        reasons.append(
            "baseline_expected_sample_count_is_not_54:"
            f"{expected_samples}"
        )

    sample_ids = [record.get("sample_id") for record in sample_records]
    if any(not sample_id for sample_id in sample_ids):
        reasons.append("sample_id_is_missing")
    if len(set(sample_ids)) != len(sample_ids):
        reasons.append("sample_id_is_not_unique")

    agent_errors = sum(
        1 for record in sample_records if record.get("status") != "success"
    )
    if agent_errors:
        reasons.append(f"agent_errors:{agent_errors}")
    missing_tool_counts = sum(
        1
        for record in sample_records
        if (
            not isinstance(record.get("tool_call_count"), int)
            or isinstance(record.get("tool_call_count"), bool)
        )
    )
    if missing_tool_counts:
        reasons.append(f"tool_call_counts_missing:{missing_tool_counts}")
    missing_required_searches = sum(
        1
        for record in sample_records
        if (
            record.get("status") == "success"
            and record.get("tool_call_count") == 0
        )
    )
    if missing_required_searches:
        reasons.append(
            f"required_search_not_observed:{missing_required_searches}"
        )

    if evaluation_error:
        reasons.append("evaluator_failed")
    if manifest.get("runtime_config_error"):
        reasons.append("runtime_config_snapshot_failed")
    if manifest.get("evaluator_runtime_config_error"):
        reasons.append("evaluator_runtime_config_snapshot_failed")

    metric_counts: Dict[str, Any] = {}
    if metrics:
        raw_counts = metrics.get("metric_counts")
        if isinstance(raw_counts, dict):
            metric_counts = raw_counts
    for metric in METRIC_KEYS:
        counts = metric_counts.get(metric)
        if not isinstance(counts, dict):
            reasons.append(f"metric_counts_missing:{metric}")
            continue
        expected = counts.get("expected")
        finite = counts.get("finite")
        missing = counts.get("missing")
        if expected != expected_samples or finite != expected_samples or missing != 0:
            reasons.append(
                f"metric_incomplete:{metric}:{finite}/{expected_samples}"
            )

    if metrics:
        aggregate = metrics.get("aggregate")
        if not isinstance(aggregate, dict):
            reasons.append("aggregate_metrics_are_missing")
        else:
            for metric in METRIC_KEYS:
                value = aggregate.get(metric)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    reasons.append(
                        f"aggregate_metric_is_not_finite:{metric}"
                    )

        per_sample = metrics.get("per_sample")
        if not isinstance(per_sample, list):
            reasons.append("per_sample_metrics_are_missing")
        elif len(per_sample) != expected_samples:
            reasons.append(
                "per_sample_metric_count_mismatch:"
                f"{len(per_sample)}/{expected_samples}"
            )
        else:
            for index, record in enumerate(per_sample):
                if not isinstance(record, dict):
                    reasons.append(f"per_sample_metric_invalid:{index}")
                    continue
                for metric in METRIC_KEYS:
                    value = record.get(metric)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                    ):
                        reasons.append(
                            "per_sample_metric_is_not_finite:"
                            f"{index}:{metric}"
                        )

    models = manifest.get("models")
    if not isinstance(models, dict):
        reasons.append("model_roles_are_missing")
    else:
        for role in ("answer", "judge", "embedding", "judge_embedding"):
            if not models.get(role):
                reasons.append(f"model_role_is_missing:{role}")
        if "glm52" not in _canonical_model_name(models.get("answer")):
            reasons.append("baseline_model_mismatch:answer")
        answer_model = _canonical_model_name(models.get("answer"))
        judge_model = _canonical_model_name(models.get("judge"))
        if answer_model and judge_model and answer_model == judge_model:
            reasons.append("judge_model_is_not_independent")
        for role in ("embedding", "judge_embedding"):
            embedding_model = _canonical_model_name(models.get(role))
            if (
                "bgem3" not in embedding_model
                and embedding_model != "server274214"
            ):
                reasons.append(f"baseline_model_mismatch:{role}")

    runtime_config = manifest.get("runtime_config")
    if manifest.get("knowledge_base") == "trpc-agent-go":
        if not isinstance(runtime_config, dict):
            reasons.append("runtime_config_is_missing")
        else:
            if runtime_config.get("vectorstore") != "pgvector":
                reasons.append("baseline_vectorstore_is_not_pgvector")
            if runtime_config.get("search_mode") != 0:
                reasons.append("baseline_search_mode_is_not_hybrid")
            if runtime_config.get("chunk_size") != 500:
                reasons.append("baseline_chunk_size_is_not_500")
            if runtime_config.get("chunk_overlap") != 50:
                reasons.append("baseline_chunk_overlap_is_not_50")
            if runtime_config.get("embedding_dimensions") != 1024:
                reasons.append(
                    "baseline_embedding_dimensions_is_not_1024"
                )
            module = runtime_config.get("framework_module")
            if (
                not isinstance(module, dict)
                or not module.get("path")
                or not module.get("version")
            ):
                reasons.append("framework_module_provenance_is_missing")
            if runtime_config.get("vectorstore") == "pgvector":
                if not runtime_config.get("pg_table"):
                    reasons.append("pgvector_table_is_missing")
                document_count = runtime_config.get(
                    "index_document_count"
                )
                if document_count is None:
                    reasons.append("pgvector_document_count_is_missing")
                elif (
                    not isinstance(document_count, int)
                    or isinstance(document_count, bool)
                    or document_count <= 0
                ):
                    reasons.append(
                        "pgvector_document_count_is_not_positive"
                    )

    evaluator_runtime_config = manifest.get("evaluator_runtime_config")
    if manifest.get("evaluator") == "ragas":
        if not isinstance(evaluator_runtime_config, dict):
            reasons.append("evaluator_runtime_config_is_missing")
        else:
            if not evaluator_runtime_config.get("model_name"):
                reasons.append("evaluator_model_provenance_is_missing")
            if not evaluator_runtime_config.get("embedding_model"):
                reasons.append(
                    "evaluator_embedding_provenance_is_missing"
                )
            for flag in (
                "model_explicit",
                "api_key_explicit",
                "base_url_explicit",
                "model_separate_from_answer",
                "api_key_separate_from_answer",
                "endpoint_separate_from_answer",
            ):
                if evaluator_runtime_config.get(flag) is not True:
                    reasons.append(f"judge_role_separation_missing:{flag}")

    endpoints = manifest.get("endpoints")
    if not isinstance(endpoints, dict):
        reasons.append("endpoint_identities_are_missing")
    else:
        for role in (
            "answer",
            "judge",
            "embedding",
            "judge_embedding",
        ):
            if not endpoints.get(role):
                reasons.append(f"endpoint_identity_is_missing:{role}")
        if (
            endpoints.get("answer")
            and endpoints.get("judge")
            and endpoints["answer"] == endpoints["judge"]
        ):
            reasons.append("judge_endpoint_is_not_independent")

    repositories = manifest.get("repositories")
    if not isinstance(repositories, dict) or not repositories.get("benchmark"):
        reasons.append("benchmark_revision_is_missing")
    else:
        benchmark_revision = repositories["benchmark"]
        if (
            not isinstance(benchmark_revision, dict)
            or not benchmark_revision.get("commit")
        ):
            reasons.append("benchmark_commit_is_missing")
        elif benchmark_revision.get("dirty") is not False:
            reasons.append("benchmark_worktree_is_dirty")
    if not isinstance(repositories, dict) or not repositories.get(
        "framework_superproject"
    ):
        reasons.append("framework_superproject_revision_is_missing")
    else:
        framework_revision = repositories["framework_superproject"]
        if (
            not isinstance(framework_revision, dict)
            or not framework_revision.get("commit")
        ):
            reasons.append("framework_superproject_commit_is_missing")
        elif framework_revision.get("dirty") is not False:
            reasons.append("framework_superproject_worktree_is_dirty")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        reasons.append("artifact_paths_are_missing")
    else:
        for artifact in ("result", "manifest", "samples", "diagnostics"):
            if not artifacts.get(artifact):
                reasons.append(f"artifact_path_is_missing:{artifact}")

    prompt_limit = manifest.get("prompt_max_searches")
    hard_limit = manifest.get("hard_max_tool_iterations")
    if prompt_limit is None:
        reasons.append("prompt_search_budget_is_missing")
    elif prompt_limit != 3:
        reasons.append(f"baseline_prompt_search_budget_is_not_3:{prompt_limit}")
    if hard_limit is None:
        reasons.append("hard_tool_watchdog_is_missing")
    elif hard_limit != 500:
        reasons.append(f"baseline_hard_tool_watchdog_is_not_500:{hard_limit}")

    budget_violations = sum(
        1
        for record in sample_records
        if (
            isinstance(prompt_limit, int)
            and not isinstance(prompt_limit, bool)
            and isinstance(record.get("tool_call_count"), int)
            and not isinstance(record.get("tool_call_count"), bool)
            and record["tool_call_count"] > prompt_limit
        )
    )
    limitations = []
    if manifest.get("skip_load"):
        limitations.append("reused_index")
    if (
        isinstance(prompt_limit, int)
        and isinstance(hard_limit, int)
        and prompt_limit != hard_limit
    ):
        limitations.append("modified_tool_watchdog")
    if budget_violations:
        limitations.append("prompt_search_budget_exceeded")

    return {
        "execution_status": "failed" if evaluation_error else "complete",
        "evidence_status": "insufficient" if reasons else "valid",
        "evidence_scope": BASELINE_EVIDENCE_SCOPE,
        "formal_ab_eligible": False,
        "reasons": reasons,
        "limitations": limitations,
        "expected_samples": expected_samples,
        "actual_samples": actual_samples,
        "agent_error_count": agent_errors,
        "budget_violation_count": budget_violations,
        "quality_metrics_include_runtime_failures": agent_errors > 0,
    }
