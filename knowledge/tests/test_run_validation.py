#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#

import pathlib
import sys
import unittest


KNOWLEDGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KNOWLEDGE_ROOT))

from run_artifacts import (  # noqa: E402
    BASELINE_EVIDENCE_SCOPE,
    BASELINE_RUN_KIND,
)
from run_validation import METRIC_KEYS, validate_baseline_run  # noqa: E402


def _manifest() -> dict:
    return {
        "run_kind": BASELINE_RUN_KIND,
        "evidence_scope": BASELINE_EVIDENCE_SCOPE,
        "expected_samples": 54,
        "dataset": "huggingface",
        "knowledge_base": "trpc-agent-go",
        "evaluator": "ragas",
        "retrieval_k": 4,
        "models": {
            "answer": "glm-5.2",
            "judge": "economical-independent-judge",
            "embedding": "bge-m3",
            "judge_embedding": "bge-m3",
        },
        "endpoints": {
            "answer": "https://answer.test/v1",
            "judge": "https://judge.test/v1",
            "embedding": "https://embedding.test/v1",
            "judge_embedding": "https://embedding.test/v1",
        },
        "runtime_config": {
            "vectorstore": "pgvector",
            "search_mode": 0,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "embedding_dimensions": 1024,
            "pg_table": "baseline",
            "index_document_count": 100,
            "framework_module": {
                "path": "trpc.group/trpc-go/trpc-agent-go",
                "version": "v1.7.0",
            },
        },
        "evaluator_runtime_config": {
            "model_name": "economical-independent-judge",
            "embedding_model": "bge-m3",
            "model_explicit": True,
            "api_key_explicit": True,
            "base_url_explicit": True,
            "model_separate_from_answer": True,
            "api_key_separate_from_answer": True,
            "endpoint_separate_from_answer": True,
        },
        "repositories": {
            "benchmark": {"commit": "abc123", "dirty": False},
            "framework_superproject": {
                "commit": "def456",
                "dirty": False,
            },
        },
        "artifacts": {
            "result": "run.json",
            "manifest": "run.json.manifest.json",
            "samples": "run.json.samples.json",
            "diagnostics": "run.json.ragas-diagnostics.jsonl",
        },
        "prompt_max_searches": 3,
        "hard_max_tool_iterations": 500,
        "skip_load": True,
    }


def _samples() -> list[dict]:
    return [
        {
            "sample_id": f"sample-{index}",
            "status": "success",
            "tool_call_count": 1,
            "exceeded_prompt_search_budget": False,
        }
        for index in range(54)
    ]


def _metrics() -> dict:
    return {
        "aggregate": {metric: 0.5 for metric in METRIC_KEYS},
        "metric_counts": {
            metric: {"expected": 54, "finite": 54, "missing": 0}
            for metric in METRIC_KEYS
        },
        "per_sample": [
            {
                "sample_index": index,
                **{metric: 0.5 for metric in METRIC_KEYS},
            }
            for index in range(54)
        ],
    }


class RunValidationTest(unittest.TestCase):
    def test_complete_baseline_is_valid_but_never_formal_ab_evidence(self):
        validation = validate_baseline_run(
            _manifest(),
            _samples(),
            _metrics(),
        )
        self.assertEqual(validation["evidence_status"], "valid")
        self.assertFalse(validation["formal_ab_eligible"])
        self.assertIn("reused_index", validation["limitations"])
        self.assertIn(
            "modified_tool_watchdog",
            validation["limitations"],
        )

    def test_agent_or_metric_failure_is_insufficient(self):
        samples = _samples()
        samples[1]["status"] = "agent_error"
        metrics = _metrics()
        metrics["metric_counts"]["faithfulness"]["finite"] = 53
        metrics["metric_counts"]["faithfulness"]["missing"] = 1

        validation = validate_baseline_run(
            _manifest(),
            samples,
            metrics,
        )
        self.assertEqual(validation["evidence_status"], "insufficient")
        self.assertIn("agent_errors:1", validation["reasons"])
        self.assertIn(
            "metric_incomplete:faithfulness:53/54",
            validation["reasons"],
        )
        self.assertTrue(
            validation["quality_metrics_include_runtime_failures"]
        )

    def test_evaluator_failure_is_failed_and_insufficient(self):
        validation = validate_baseline_run(
            _manifest(),
            _samples(),
            None,
            evaluation_error="judge timeout",
        )
        self.assertEqual(validation["execution_status"], "failed")
        self.assertEqual(validation["evidence_status"], "insufficient")
        self.assertIn("evaluator_failed", validation["reasons"])

    def test_shared_judge_model_endpoint_or_credentials_are_insufficient(self):
        manifest = _manifest()
        manifest["models"]["judge"] = "glm-5.2"
        manifest["endpoints"]["judge"] = manifest["endpoints"]["answer"]
        manifest["evaluator_runtime_config"][
            "api_key_separate_from_answer"
        ] = False

        validation = validate_baseline_run(
            manifest,
            _samples(),
            _metrics(),
        )

        self.assertEqual(validation["evidence_status"], "insufficient")
        self.assertIn(
            "judge_model_is_not_independent",
            validation["reasons"],
        )
        self.assertIn(
            "judge_endpoint_is_not_independent",
            validation["reasons"],
        )
        self.assertIn(
            "judge_role_separation_missing:api_key_separate_from_answer",
            validation["reasons"],
        )

    def test_dirty_framework_superproject_is_insufficient(self):
        manifest = _manifest()
        manifest["repositories"]["framework_superproject"]["dirty"] = True

        validation = validate_baseline_run(
            manifest,
            _samples(),
            _metrics(),
        )

        self.assertEqual(validation["evidence_status"], "insufficient")
        self.assertIn(
            "framework_superproject_worktree_is_dirty",
            validation["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
