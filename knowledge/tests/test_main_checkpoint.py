#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#

import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace


KNOWLEDGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KNOWLEDGE_ROOT))

from dataset.base import QAItem  # noqa: E402
from main import run_evaluation, run_evaluator_only  # noqa: E402
from run_artifacts import artifact_paths  # noqa: E402
from run_validation import METRIC_KEYS  # noqa: E402


class FakeDataset:
    def load_qa_items(self):
        return [
            QAItem(
                question="question",
                answer="ground truth",
                context="",
                source_doc="doc.txt",
            )
        ]

    def load_documents(self, force_reload=False):
        raise AssertionError("skip-load should not load documents")


class FakeKnowledgeBase:
    def __init__(self):
        self.last_trace = None
        self.answer_calls = 0

    def answer(self, question, k=4):
        self.answer_calls += 1
        self.last_trace = {
            "tool_calls": [
                {
                    "name": "knowledge_search",
                    "arguments": '{"query":"retrieval query"}',
                }
            ]
        }
        return (
            "answer",
            [
                SimpleNamespace(
                    content="context",
                    trace=self.last_trace,
                )
            ],
        )


class FailingEvaluator:
    last_metrics = None

    def evaluate(self, samples):
        raise RuntimeError("judge timeout")


class CompleteEvaluator:
    def __init__(self):
        self.last_metrics = None
        self.calls = 0

    def get_runtime_config(self):
        return {
            "model_name": "economical-independent-judge",
            "embedding_model": "bge-m3",
            "base_url": (
                "https://user:judge-secret@judge.test/v1"
                "?token=judge-secret"
            ),
            "embedding_base_url": (
                "user:embedding-secret@embedding.test/v1"
            ),
            "header_names": ["X-SMG-Routing-Key"],
            "embedding_header_names": ["X-SMG-Agent-Name"],
            "model_explicit": True,
            "api_key_explicit": True,
            "base_url_explicit": True,
            "model_separate_from_answer": True,
            "api_key_separate_from_answer": True,
            "endpoint_separate_from_answer": True,
        }

    def evaluate(self, samples):
        self.calls += 1
        self.last_metrics = {
            "aggregate": {metric: 0.5 for metric in METRIC_KEYS},
            "metric_counts": {
                metric: {
                    "expected": len(samples),
                    "finite": len(samples),
                    "missing": 0,
                }
                for metric in METRIC_KEYS
            },
            "per_sample": [
                {
                    "sample_index": index,
                    **{metric: 0.5 for metric in METRIC_KEYS},
                }
                for index in range(len(samples))
            ],
        }
        return "complete evaluation"


class MainCheckpointTest(unittest.TestCase):
    def test_agent_samples_survive_evaluator_failure_and_can_be_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            first_output = str(pathlib.Path(directory) / "first.json")
            kb = FakeKnowledgeBase()
            with redirect_stdout(io.StringIO()):
                result = run_evaluation(
                    kb=kb,
                    dataset=FakeDataset(),
                    evaluator=FailingEvaluator(),
                    skip_load=True,
                    full_log=False,
                    output_file=first_output,
                    kb_name="fake",
                    evaluator_name="fake",
                    dataset_name="fake-dataset",
                )
            self.assertIn("Evaluation failed", result)
            self.assertEqual(kb.answer_calls, 1)

            first_paths = artifact_paths(first_output)
            checkpoint = json.loads(
                pathlib.Path(first_paths["samples"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(checkpoint["samples_count"], 1)
            self.assertEqual(
                checkpoint["samples"][0]["answer"],
                "answer",
            )
            first_result = json.loads(
                pathlib.Path(first_output).read_text(encoding="utf-8")
            )
            self.assertEqual(
                first_result["validation"]["evidence_status"],
                "insufficient",
            )
            self.assertIn(
                "evaluator_failed",
                first_result["validation"]["reasons"],
            )

            replay_output = str(pathlib.Path(directory) / "replay.json")
            evaluator = CompleteEvaluator()
            with redirect_stdout(io.StringIO()):
                replay_result = run_evaluator_only(
                    samples_input=first_paths["samples"],
                    evaluator=evaluator,
                    output_file=replay_output,
                )
            self.assertEqual(replay_result, "complete evaluation")
            self.assertEqual(evaluator.calls, 1)
            replay = json.loads(
                pathlib.Path(replay_output).read_text(encoding="utf-8")
            )
            self.assertFalse(replay["timing"]["qa_replayed"])
            self.assertEqual(
                replay["manifest"]["execution_mode"],
                "evaluator_only",
            )
            self.assertEqual(
                replay["manifest"]["models"]["judge"],
                "economical-independent-judge",
            )
            self.assertEqual(
                replay["manifest"]["endpoints"]["judge"],
                "https://judge.test/v1",
            )
            self.assertNotIn("judge-secret", json.dumps(replay))
            self.assertNotIn("embedding-secret", json.dumps(replay))
            self.assertEqual(
                replay["manifest"]["evaluation_replay"][
                    "source_run_fingerprint"
                ],
                checkpoint["run_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
