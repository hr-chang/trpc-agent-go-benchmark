#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#
#
"""
Main entry point for RAG evaluation with different evaluators.
"""

import argparse
import json
import os
import platform
import shlex
import sys
import time
from typing import Any, Dict, List, Optional

from dataset.base import BaseDataset
from knowledge_system.base import KnowledgeBase
from evaluator.base import Evaluator, EvaluationSample
from run_artifacts import (
    BASELINE_EVIDENCE_SCOPE,
    BASELINE_RUN_KIND,
    SCHEMA_VERSION,
    RunArtifactWriter,
    build_sample_id,
    endpoint_identity,
    load_sample_checkpoint,
    repository_provenance,
)
from run_validation import validate_baseline_run


def _normalize_query(query: Any) -> Optional[str]:
    """Normalize a query value to a non-empty string."""
    if isinstance(query, str):
        normalized = query.strip()
        if normalized:
            return normalized
    return None


def _extract_query_from_tool_call_arguments(arguments: Any) -> Optional[str]:
    """Extract query text from tool-call arguments."""
    if isinstance(arguments, dict):
        return _normalize_query(arguments.get("query"))

    if isinstance(arguments, str):
        argument_text = arguments.strip()
        if not argument_text:
            return None
        try:
            payload = json.loads(argument_text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return _normalize_query(payload.get("query"))
    return None


def _dedupe_queries(queries: List[str]) -> List[str]:
    """Dedupe queries while preserving original order."""
    seen = set()
    deduped = []
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        deduped.append(query)
    return deduped


def extract_retrieval_queries(
    question: str,
    search_results: List[Any],
    fallback_trace: Optional[dict] = None,
) -> List[str]:
    """Extract retrieval queries from search-result metadata or agent trace."""
    queries: List[str] = []

    # CrewAI path: tool query is attached to per-result metadata.
    for result in search_results:
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        normalized = _normalize_query(metadata.get("tool_query"))
        if normalized:
            queries.append(normalized)

    # tRPC-Agent-Go path: tool query is embedded in trace.tool_calls[*].arguments.
    trace: Optional[dict] = None
    for result in search_results:
        candidate = getattr(result, "trace", None)
        if isinstance(candidate, dict):
            trace = candidate
            break

    if trace is None and isinstance(fallback_trace, dict):
        trace = fallback_trace

    if isinstance(trace, dict):
        tool_queries = trace.get("tool_queries", [])
        if isinstance(tool_queries, list):
            for query in tool_queries:
                normalized = _normalize_query(query)
                if normalized:
                    queries.append(normalized)

        tool_calls = trace.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                extracted = _extract_query_from_tool_call_arguments(call.get("arguments"))
                if extracted:
                    queries.append(extracted)

    return _dedupe_queries(queries)


def _trace_from_results(
    search_results: List[Any],
    fallback_trace: Optional[dict],
) -> Optional[dict]:
    """Read the trace attached to results, falling back to the KB client."""
    for result in search_results:
        trace = getattr(result, "trace", None)
        if isinstance(trace, dict):
            return trace
    return fallback_trace if isinstance(fallback_trace, dict) else None


def count_tool_calls(
    search_results: List[Any],
    fallback_trace: Optional[dict] = None,
) -> int:
    """Count actual agent tool calls in the captured trace."""
    trace = _trace_from_results(search_results, fallback_trace)
    if not trace:
        return 0
    tool_calls = trace.get("tool_calls")
    if not isinstance(tool_calls, list):
        return 0
    count = 0
    seen_ids = set()
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id")
        if call_id:
            if call_id in seen_ids:
                continue
            seen_ids.add(call_id)
        count += 1
    return count


def _runtime_config(kb: KnowledgeBase) -> Dict[str, Any]:
    """Read optional runtime provenance without making it a hard dependency."""
    getter = getattr(kb, "get_runtime_config", None)
    if not callable(getter):
        return {}
    config = getter()
    if not isinstance(config, dict):
        raise TypeError("knowledge-base runtime config must be a dictionary")
    return config


def _evaluator_runtime_config(evaluator: Evaluator) -> Dict[str, Any]:
    """Read optional evaluator provenance without credentials."""
    getter = getattr(evaluator, "get_runtime_config", None)
    if not callable(getter):
        return {}
    config = getter()
    if not isinstance(config, dict):
        raise TypeError("evaluator runtime config must be a dictionary")
    sanitized = {
        "model_name": config.get("model_name", ""),
        "embedding_model": config.get("embedding_model", ""),
        "endpoint": endpoint_identity(
            config.get("endpoint") or config.get("base_url")
        ),
        "embedding_endpoint": endpoint_identity(
            config.get("embedding_endpoint")
            or config.get("embedding_base_url")
        ),
        "header_names": sorted(config.get("header_names") or []),
        "embedding_header_names": sorted(
            config.get("embedding_header_names") or []
        ),
    }
    for flag in (
        "model_explicit",
        "api_key_explicit",
        "base_url_explicit",
        "model_separate_from_answer",
        "api_key_separate_from_answer",
        "endpoint_separate_from_answer",
    ):
        sanitized[flag] = config.get(flag) is True
    return sanitized


def build_run_manifest(
    kb_name: str,
    evaluator_name: str,
    dataset_name: str,
    retrieval_k: int,
    skip_load: bool,
    runtime_config: Optional[Dict[str, Any]] = None,
    runtime_config_error: Optional[str] = None,
    evaluator_runtime_config: Optional[Dict[str, Any]] = None,
    evaluator_runtime_config_error: Optional[str] = None,
    run_kind: str = BASELINE_RUN_KIND,
) -> Dict[str, Any]:
    """Build a manifest capturing all key configuration for reproducibility."""
    from util import get_config

    config = get_config()
    runtime_config = runtime_config or {}
    evaluator_runtime_config = evaluator_runtime_config or {}
    benchmark_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_max_searches = runtime_config.get("prompt_max_searches")
    hard_max_tool_iterations = runtime_config.get("hard_max_tool_iterations")
    if kb_name == "trpc-agent-go":
        # These are the historical modified-harness defaults. The Go service
        # snapshot should normally provide the same effective values.
        if prompt_max_searches is None:
            prompt_max_searches = 3
        if hard_max_tool_iterations is None:
            hard_max_tool_iterations = 500

    return {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "evidence_scope": BASELINE_EVIDENCE_SCOPE,
        "formal_ab_eligible": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "command": shlex.join(sys.argv),
        "working_directory": os.getcwd(),
        "knowledge_base": kb_name,
        "evaluator": evaluator_name,
        "dataset": dataset_name,
        "models": {
            "answer": runtime_config.get("model_name")
            or config.get("model_name", ""),
            "judge": evaluator_runtime_config.get("model_name")
            or config.get("eval_model_name", ""),
            "embedding": runtime_config.get("embedding_model")
            or config.get("embedding_model", ""),
            "judge_embedding": evaluator_runtime_config.get(
                "embedding_model"
            )
            or config.get("embedding_model", ""),
        },
        "endpoints": {
            "answer": runtime_config.get("llm_endpoint")
            or endpoint_identity(config.get("base_url")),
            "judge": endpoint_identity(
                evaluator_runtime_config.get("endpoint")
                or config.get("eval_base_url")
            ),
            "embedding": runtime_config.get("embedding_endpoint")
            or endpoint_identity(config.get("embedding_base_url")),
            "judge_embedding": endpoint_identity(
                evaluator_runtime_config.get("embedding_endpoint")
                or config.get("embedding_base_url")
            ),
        },
        "gateway_header_names": {
            "answer": runtime_config.get("llm_header_names", []),
            "judge": evaluator_runtime_config.get(
                "header_names",
                sorted((config.get("eval_headers") or {}).keys()),
            ),
            "embedding": runtime_config.get(
                "embedding_header_names",
                sorted((config.get("embedding_headers") or {}).keys()),
            ),
            "judge_embedding": evaluator_runtime_config.get(
                "embedding_header_names",
                sorted((config.get("embedding_headers") or {}).keys()),
            ),
        },
        "retrieval_k": retrieval_k,
        "skip_load": skip_load,
        "index_policy": "reuse_existing" if skip_load else "rebuild",
        "prompt_max_searches": prompt_max_searches,
        "hard_max_tool_iterations": hard_max_tool_iterations,
        "runtime_config": runtime_config,
        "runtime_config_error": runtime_config_error,
        "evaluator_runtime_config": evaluator_runtime_config,
        "evaluator_runtime_config_error": evaluator_runtime_config_error,
        "repositories": repository_provenance(benchmark_root),
    }


def _set_effective_runtime_config(
    manifest: Dict[str, Any],
    runtime_config: Dict[str, Any],
    runtime_config_error: Optional[str],
) -> None:
    """Update effective service fields after an index rebuild."""
    manifest["runtime_config"] = runtime_config
    manifest["runtime_config_error"] = runtime_config_error
    if runtime_config_error:
        return
    if runtime_config.get("model_name"):
        manifest["models"]["answer"] = runtime_config["model_name"]
    if runtime_config.get("embedding_model"):
        manifest["models"]["embedding"] = runtime_config[
            "embedding_model"
        ]
    if runtime_config.get("prompt_max_searches") is not None:
        manifest["prompt_max_searches"] = runtime_config[
            "prompt_max_searches"
        ]
    if runtime_config.get("hard_max_tool_iterations") is not None:
        manifest["hard_max_tool_iterations"] = runtime_config[
            "hard_max_tool_iterations"
        ]
    if runtime_config.get("llm_endpoint"):
        manifest["endpoints"]["answer"] = runtime_config["llm_endpoint"]
    if runtime_config.get("embedding_endpoint"):
        manifest["endpoints"]["embedding"] = runtime_config[
            "embedding_endpoint"
        ]
    manifest["gateway_header_names"]["answer"] = runtime_config.get(
        "llm_header_names",
        [],
    )
    manifest["gateway_header_names"]["embedding"] = runtime_config.get(
        "embedding_header_names",
        manifest["gateway_header_names"]["embedding"],
    )


def _set_effective_evaluator_config(
    manifest: Dict[str, Any],
    evaluator_runtime_config: Dict[str, Any],
    evaluator_runtime_config_error: Optional[str],
) -> None:
    """Update Judge fields for a fresh or replayed evaluation attempt."""
    manifest["evaluator_runtime_config"] = evaluator_runtime_config
    manifest[
        "evaluator_runtime_config_error"
    ] = evaluator_runtime_config_error
    if evaluator_runtime_config_error:
        return
    if evaluator_runtime_config.get("model_name"):
        manifest["models"]["judge"] = evaluator_runtime_config[
            "model_name"
        ]
    if evaluator_runtime_config.get("embedding_model"):
        manifest["models"]["judge_embedding"] = (
            evaluator_runtime_config["embedding_model"]
        )
    if evaluator_runtime_config.get("endpoint"):
        manifest["endpoints"]["judge"] = evaluator_runtime_config[
            "endpoint"
        ]
    if evaluator_runtime_config.get("embedding_endpoint"):
        manifest["endpoints"]["judge_embedding"] = (
            evaluator_runtime_config["embedding_endpoint"]
        )
    manifest["gateway_header_names"]["judge"] = (
        evaluator_runtime_config.get("header_names", [])
    )
    manifest["gateway_header_names"]["judge_embedding"] = (
        evaluator_runtime_config.get("embedding_header_names", [])
    )


def _as_evaluation_samples(
    sample_records: List[Dict[str, Any]],
) -> List[EvaluationSample]:
    """Convert durable sample records into the evaluator's input type."""
    return [
        EvaluationSample(
            question=record["question"],
            answer=record["answer"],
            contexts=record["contexts"],
            ground_truth=record["ground_truth"],
        )
        for record in sample_records
    ]


def _evaluate_samples(
    evaluator: Evaluator,
    samples: List[EvaluationSample],
    diagnostics_path: Optional[str],
) -> tuple[str, Optional[Dict[str, Any]], Optional[str], float]:
    """Run an evaluator while preserving a structured failure result."""
    previous_diagnostics_path = os.environ.get("RAGAS_DIAGNOSTICS_PATH")
    if diagnostics_path:
        os.environ["RAGAS_DIAGNOSTICS_PATH"] = diagnostics_path
    if hasattr(evaluator, "last_metrics"):
        evaluator.last_metrics = None

    started = time.time()
    try:
        result = evaluator.evaluate(samples)
        metrics = getattr(evaluator, "last_metrics", None)
        if not isinstance(metrics, dict):
            metrics = None
        return result, metrics, None, time.time() - started
    except Exception as error:
        message = f"❌ Evaluation failed: {error}"
        return message, None, str(error), time.time() - started
    finally:
        if diagnostics_path:
            if previous_diagnostics_path is None:
                os.environ.pop("RAGAS_DIAGNOSTICS_PATH", None)
            else:
                os.environ[
                    "RAGAS_DIAGNOSTICS_PATH"
                ] = previous_diagnostics_path


def _finalize_run(
    manifest: Dict[str, Any],
    sample_records: List[Dict[str, Any]],
    result: str,
    metrics: Optional[Dict[str, Any]],
    evaluation_error: Optional[str],
    timing: Dict[str, Any],
    writer: Optional[RunArtifactWriter],
) -> Dict[str, Any]:
    """Classify a run and persist the final manifest/result atomically."""
    if metrics and isinstance(metrics.get("per_sample"), list):
        for index, metric_record in enumerate(metrics["per_sample"]):
            if (
                isinstance(metric_record, dict)
                and index < len(sample_records)
            ):
                metric_record["sample_id"] = sample_records[index][
                    "sample_id"
                ]

    validation = validate_baseline_run(
        manifest,
        sample_records,
        metrics,
        evaluation_error=evaluation_error,
    )
    manifest["execution_status"] = validation["execution_status"]
    manifest["evidence_status"] = validation["evidence_status"]
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["validation"] = validation

    errors = [
        {
            "sample_id": record["sample_id"],
            "question": record["question"],
            "error": record.get("error"),
            "time": record["elapsed_seconds"],
        }
        for record in sample_records
        if record.get("status") != "success"
    ]
    sample_debug = [
        {
            "sample_id": record["sample_id"],
            "question": record["question"],
            "retrieval_queries": record["retrieval_queries"],
            "tool_call_count": record["tool_call_count"],
            "retrieved_context_count": record.get(
                "retrieved_context_count"
            ),
        }
        for record in sample_records
    ]
    output_data = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest.get("run_id"),
        "manifest": manifest,
        "validation": validation,
        "timing": timing,
        "samples_count": len(sample_records),
        "errors_count": len(errors),
        "result": result,
        "evaluation": {
            "formatted_result": result,
            "error": evaluation_error,
            "metrics": metrics,
        },
        "samples": sample_records,
        "errors": errors,
        "sample_debug": sample_debug,
    }
    if writer:
        writer.manifest.update(manifest)
        writer.write_manifest()
        writer.write_samples(sample_records)
        writer.write_result(output_data)
    return output_data


def run_evaluator_only(
    samples_input: str,
    evaluator: Evaluator,
    output_file: str,
) -> str:
    """Re-run only the judge from a durable Q&A checkpoint."""
    checkpoint = load_sample_checkpoint(samples_input)
    sample_records = checkpoint["samples"]
    manifest = dict(checkpoint["manifest"])
    expected_samples = int(manifest.get("expected_samples") or 0)
    if len(sample_records) != expected_samples:
        raise ValueError(
            "evaluator-only mode requires a complete Q&A checkpoint: "
            f"{len(sample_records)}/{expected_samples}"
        )

    evaluator_runtime_config: Dict[str, Any] = {}
    evaluator_runtime_config_error = None
    try:
        evaluator_runtime_config = _evaluator_runtime_config(evaluator)
    except Exception as error:
        evaluator_runtime_config_error = str(error)
    _set_effective_evaluator_config(
        manifest,
        evaluator_runtime_config,
        evaluator_runtime_config_error,
    )
    manifest["execution_mode"] = "evaluator_only"
    manifest["evaluation_replay"] = {
        "source_checkpoint": os.path.abspath(samples_input),
        "source_run_id": checkpoint.get("run_id"),
        "source_run_fingerprint": checkpoint.get("run_fingerprint"),
        "command": shlex.join(sys.argv),
        "working_directory": os.getcwd(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    sample_ids = [record["sample_id"] for record in sample_records]
    writer = RunArtifactWriter(output_file, manifest, sample_ids)
    manifest = writer.manifest
    writer.write_samples(sample_records)

    samples = _as_evaluation_samples(sample_records)
    result, metrics, evaluation_error, eval_time = _evaluate_samples(
        evaluator,
        samples,
        writer.paths["diagnostics"],
    )
    qa_total_time = sum(
        float(record.get("elapsed_seconds") or 0)
        for record in sample_records
    )
    timing = {
        "qa_total_seconds": round(qa_total_time, 2),
        "qa_avg_seconds": round(
            qa_total_time / len(sample_records) if sample_records else 0,
            2,
        ),
        "eval_seconds": round(eval_time, 2),
        "total_seconds": round(qa_total_time + eval_time, 2),
        "qa_replayed": False,
    }
    output_data = _finalize_run(
        manifest,
        sample_records,
        result,
        metrics,
        evaluation_error,
        timing,
        writer,
    )
    print(f"\n{result}")
    print(
        "\nEvidence status: "
        f"{output_data['validation']['evidence_status']}"
    )
    print(f"📁 Results saved to: {output_file}")
    return result


def run_evaluation(
    kb: KnowledgeBase,
    dataset: BaseDataset,
    evaluator: Evaluator,
    retrieval_k: int = 4,
    skip_load: bool = False,
    force_reload: bool = False,
    full_log: bool = True,
    output_file: Optional[str] = None,
    kb_name: str = "unknown",
    evaluator_name: str = "unknown",
    dataset_name: str = "unknown",
    run_kind: str = BASELINE_RUN_KIND,
) -> str:
    """Collect checkpointed Q&A samples, then evaluate and classify them."""
    print("=== RAG Evaluation ===\n")
    print("1. Loading QA items...")
    qa_items = dataset.load_qa_items()
    print(f"   Loaded {len(qa_items)} QA items.\n")

    runtime_config: Dict[str, Any] = {}
    runtime_config_error = None
    try:
        runtime_config = _runtime_config(kb)
    except Exception as error:
        runtime_config_error = str(error)
        print(f"   ⚠️  Runtime configuration unavailable: {error}")
    evaluator_runtime_config: Dict[str, Any] = {}
    evaluator_runtime_config_error = None
    try:
        evaluator_runtime_config = _evaluator_runtime_config(evaluator)
    except Exception as error:
        evaluator_runtime_config_error = str(error)
        print(f"   ⚠️  Evaluator configuration unavailable: {error}")

    manifest = build_run_manifest(
        kb_name=kb_name,
        evaluator_name=evaluator_name,
        dataset_name=dataset_name,
        retrieval_k=retrieval_k,
        skip_load=skip_load,
        runtime_config=runtime_config,
        runtime_config_error=runtime_config_error,
        evaluator_runtime_config=evaluator_runtime_config,
        evaluator_runtime_config_error=evaluator_runtime_config_error,
        run_kind=run_kind,
    )
    expected_sample_ids = [
        build_sample_id(
            dataset_name,
            qa.question,
            qa.answer,
            sample_index=index,
        )
        for index, qa in enumerate(qa_items)
    ]
    writer = (
        RunArtifactWriter(output_file, manifest, expected_sample_ids)
        if output_file
        else None
    )
    if writer:
        manifest = writer.manifest
        writer.write_samples([])
    else:
        manifest["expected_samples"] = len(expected_sample_ids)

    print("📋 Run Manifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print()

    setup_error = None
    if skip_load:
        print("2. Skipping document loading (--skip-load enabled)...\n")
    else:
        try:
            print("2. Loading documents...")
            doc_dir = dataset.load_documents(force_reload=force_reload)
            file_paths = [
                os.path.join(doc_dir, filename)
                for filename in sorted(os.listdir(doc_dir))
                if os.path.isfile(os.path.join(doc_dir, filename))
            ]
            print(
                f"   Found {len(file_paths)} documents "
                "(sorted for reproducibility)."
            )
            print("3. Building knowledge base...")
            kb.load(file_paths)
            print("   Knowledge base built.\n")
            try:
                refreshed_runtime_config = _runtime_config(kb)
                _set_effective_runtime_config(
                    manifest,
                    refreshed_runtime_config,
                    None,
                )
            except Exception as error:
                runtime_config_error = str(error)
                _set_effective_runtime_config(
                    manifest,
                    runtime_config,
                    runtime_config_error,
                )
                print(
                    "   ⚠️  Post-load runtime configuration "
                    f"unavailable: {error}"
                )
            if writer:
                writer.refresh_manifest(manifest)
                manifest = writer.manifest
                writer.write_samples([])
        except Exception as error:
            setup_error = str(error)
            print(f"   ❌ Knowledge-base setup failed: {error}")

    sample_records: List[Dict[str, Any]] = []
    qa_start_total = time.time()
    prompt_limit = manifest.get("prompt_max_searches")
    if setup_error is None:
        print("4. Running Q&A with fresh sessions...")
        for index, qa in enumerate(qa_items):
            print(
                f"\n   [{index + 1}/{len(qa_items)}] "
                f"Q: {qa.question[:80]}..."
            )
            qa_start = time.time()
            status = "success"
            error_message = None
            retrieval_queries: List[str] = []
            tool_call_count = 0
            retrieved_context_count = 0
            try:
                answer, search_results = kb.answer(
                    qa.question,
                    k=retrieval_k,
                )
                fallback_trace = getattr(kb, "last_trace", None)
                contexts = [result.content for result in search_results]
                retrieved_context_count = len(contexts)
                retrieval_queries = extract_retrieval_queries(
                    question=qa.question,
                    search_results=search_results,
                    fallback_trace=fallback_trace,
                )
                tool_call_count = count_tool_calls(
                    search_results,
                    fallback_trace=fallback_trace,
                )
                if not contexts:
                    print(
                        "   ⚠️  No contexts retrieved, using placeholder"
                    )
                    contexts = ["No relevant context found."]
            except Exception as error:
                status = "agent_error"
                error_message = str(error)
                answer = "Error: failed to generate answer."
                contexts = ["No relevant context found."]

            qa_elapsed = time.time() - qa_start
            record = {
                "sample_index": index,
                "sample_id": expected_sample_ids[index],
                "question": qa.question,
                "ground_truth": qa.answer,
                "source_document": getattr(qa, "source_doc", ""),
                "answer": answer,
                "contexts": contexts,
                "retrieved_context_count": retrieved_context_count,
                "retrieval_queries": retrieval_queries,
                "tool_call_count": tool_call_count,
                "exceeded_prompt_search_budget": (
                    isinstance(prompt_limit, int)
                    and tool_call_count > prompt_limit
                ),
                "status": status,
                "error": error_message,
                "elapsed_seconds": round(qa_elapsed, 6),
            }
            sample_records.append(record)
            if writer:
                writer.write_samples(sample_records)

            if status == "success":
                print(
                    f"   A: {answer[:200]}"
                    f"{'...' if len(answer) > 200 else ''}"
                )
                print(
                    f"   Retrieved {retrieved_context_count} contexts, "
                    f"{tool_call_count} tool calls, "
                    f"took {qa_elapsed:.2f}s"
                )
                if retrieval_queries:
                    print(
                        f"   Retrieval queries "
                        f"({len(retrieval_queries)}):"
                    )
                    for query_index, query in enumerate(
                        retrieval_queries,
                        1,
                    ):
                        print(f"      [{query_index}] {query}")
                else:
                    print("   Retrieval queries: unavailable")
                if full_log:
                    print("\n   === Full Answer ===")
                    print(answer)
                    print("\n   === Contexts ===")
                    for context_index, context in enumerate(contexts, 1):
                        print(f"\n   [{context_index}] {context}")
            else:
                print(
                    f"   ❌ Error: {error_message} "
                    f"(took {qa_elapsed:.2f}s)"
                )

    qa_total_time = time.time() - qa_start_total
    avg_time = (
        sum(record["elapsed_seconds"] for record in sample_records)
        / len(sample_records)
        if sample_records
        else 0
    )
    agent_errors = sum(
        record["status"] != "success" for record in sample_records
    )
    print(
        f"\n   Collected {len(sample_records)} samples, "
        f"{agent_errors} agent errors"
    )
    print(
        f"   ⏱️  Q&A total time: {qa_total_time:.2f}s, "
        f"avg per question: {avg_time:.2f}s"
    )

    samples = _as_evaluation_samples(sample_records)
    evaluation_error = setup_error
    metrics: Optional[Dict[str, Any]] = None
    eval_time = 0.0
    if setup_error:
        result = f"❌ Knowledge-base setup failed: {setup_error}"
    elif not samples:
        evaluation_error = "no samples collected"
        result = "❌ No samples collected. Cannot run evaluation."
    else:
        print("\n5. Running evaluation from persisted Q&A samples...")
        result, metrics, evaluation_error, eval_time = _evaluate_samples(
            evaluator,
            samples,
            writer.paths["diagnostics"] if writer else None,
        )

    timing = {
        "qa_total_seconds": round(qa_total_time, 2),
        "qa_avg_seconds": round(avg_time, 2),
        "eval_seconds": round(eval_time, 2),
        "total_seconds": round(qa_total_time + eval_time, 2),
        "qa_replayed": False,
    }
    output_data = _finalize_run(
        manifest,
        sample_records,
        result,
        metrics,
        evaluation_error,
        timing,
        writer,
    )

    print(f"\n{result}")
    print("\n--- Timing ---")
    print(
        f"Q&A total time: {qa_total_time:.2f}s "
        f"(avg {avg_time:.2f}s/question)"
    )
    print(f"Evaluation time: {eval_time:.2f}s")
    print(f"Total time: {qa_total_time + eval_time:.2f}s")
    print(
        "Evidence status: "
        f"{output_data['validation']['evidence_status']}"
    )
    if output_file:
        print(f"\n📁 Results saved to: {output_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument(
        "--evaluator",
        choices=["ragas"],
        default="ragas",
        help="Evaluator to use (default: ragas)",
    )
    parser.add_argument(
        "--kb",
        choices=["langchain", "langchain_chain", "trpc-agent-go", "agno", "crewai", "autogen"],
        default="langchain",
        help="Knowledge base implementation to use (default: langchain)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve per query (default: 4)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        default=False,
        help="Skip loading documents into knowledge base (default: False)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Force loading documents into knowledge base",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save evaluation results as JSON",
    )
    parser.add_argument(
        "--samples-input",
        type=str,
        default=None,
        help=(
            "Re-run only the evaluator from a .samples.json checkpoint; "
            "requires --output and does not initialize a dataset or agent"
        ),
    )
    parser.add_argument(
        "--run-kind",
        choices=[BASELINE_RUN_KIND],
        default=BASELINE_RUN_KIND,
        help=(
            "Evidence lane for this command. I0 supports baseline "
            "reproduction only."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["huggingface", "rgb", "multihop-rag"],
        default="huggingface",
        help="Dataset to use for evaluation (default: huggingface)",
    )
    parser.add_argument(
        "--rgb-subset",
        choices=["en", "zh", "en_int", "zh_int", "en_fact", "zh_fact"],
        default="en",
        help="RGB dataset subset (default: en). Only used when --dataset=rgb",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for evaluation (default: 600)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        default=False,
        help="Force re-downloading documents even if already cached (default: False)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=30,
        help="Number of concurrent workers for evaluation (default: 30)",
    )
    args = parser.parse_args()

    # Initialize evaluator first so evaluator-only replay never constructs a
    # dataset or knowledge-base client.
    if args.evaluator == "ragas":
        from evaluator.ragas.evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator(max_workers=args.workers, timeout=args.timeout)
        print("Using RAGAS evaluator")
    else:
        raise ValueError(f"Unknown evaluator: {args.evaluator}")

    if args.samples_input:
        if not args.output:
            parser.error("--samples-input requires --output")
        run_evaluator_only(
            samples_input=args.samples_input,
            evaluator=evaluator,
            output_file=args.output,
        )
        return

    # Initialize dataset
    from dataset import create_dataset
    dataset_kwargs = {}
    if args.dataset == "rgb":
        dataset_kwargs = {
            "subset": args.rgb_subset,
        }
    dataset = create_dataset(args.dataset, **dataset_kwargs)
    print(f"Using dataset: {args.dataset}")

    # Initialize knowledge base
    if args.kb == "trpc-agent-go":
        from knowledge_system.trpc_agent_go.knowledge_base import TRPCAgentGoKnowledgeBase
        kb = TRPCAgentGoKnowledgeBase(timeout=300000000, auto_start=False)
        print("Using tRPC-Agent-Go knowledge base")
    elif args.kb == "agno":
        from knowledge_system.agno.knowledge_base import AgnoKnowledgeBase
        kb = AgnoKnowledgeBase(max_results=args.k)
        print("Using Agno knowledge base")
    elif args.kb == "crewai":
        from knowledge_system.crewai.knowledge_base import CrewAIKnowledgeBase
        kb = CrewAIKnowledgeBase(max_results=args.k)
        print("Using CrewAI knowledge base")
    elif args.kb == "autogen":
        from knowledge_system.autogen.knowledge_base import AutoGenKnowledgeBase
        kb = AutoGenKnowledgeBase(max_results=args.k)
        print("Using AutoGen knowledge base")
    elif args.kb == "langchain_chain":
        from knowledge_system.langchain_chain.knowledge_base import LangChainChainKnowledgeBase
        kb = LangChainChainKnowledgeBase()
        print("Using LangChain Chain knowledge base")
    else:
        from knowledge_system.langchain.knowledge_base import LangChainKnowledgeBase
        kb = LangChainKnowledgeBase()
        print("Using LangChain knowledge base")

    # Run evaluation
    # --load overrides --skip-load
    skip_load = args.skip_load and not args.load
    run_evaluation(
        kb=kb,
        dataset=dataset,
        evaluator=evaluator,
        retrieval_k=args.k,
        skip_load=skip_load,
        force_reload=args.force_reload,
        full_log=True,
        output_file=args.output,
        kb_name=args.kb,
        evaluator_name=args.evaluator,
        dataset_name=args.dataset,
        run_kind=args.run_kind,
    )


if __name__ == "__main__":
    main()
