#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#
#
"""Executable checks for opt-in RAGAS finish-reason diagnostics."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from evaluator.ragas.diagnostics import (
    DIAGNOSTICS_PATH_ENV,
    RAGASFinishReasonDiagnostics,
    ragas_would_accept,
)


def _response(finish_reason, headers=None):
    generation = ChatGeneration(
        message=AIMessage(
            content='{"entities": ["example"]}',
            response_metadata={"finish_reason": finish_reason},
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        ),
        generation_info={
            "finish_reason": finish_reason,
            "headers": headers or {},
        },
    )
    return LLMResult(
        generations=[[generation]],
        llm_output={
            "model_name": "test-model",
            "token_usage": {"total_tokens": 15},
        },
    )


class RAGASFinishReasonDiagnosticsTest(unittest.TestCase):
    def test_matches_ragas_finish_reason_semantics(self):
        self.assertTrue(ragas_would_accept(_response("stop")))
        self.assertTrue(ragas_would_accept(_response("MAX_TOKENS")))
        self.assertTrue(ragas_would_accept(_response(None)))
        self.assertFalse(ragas_would_accept(_response("length")))
        self.assertFalse(ragas_would_accept(_response("content_filter")))

    def test_disabled_without_environment_variable(self):
        old_value = os.environ.pop(DIAGNOSTICS_PATH_ENV, None)
        try:
            self.assertIsNone(
                RAGASFinishReasonDiagnostics.from_environment()
            )
        finally:
            if old_value is not None:
                os.environ[DIAGNOSTICS_PATH_ENV] = old_value

    def test_normal_response_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "failures.jsonl"
            diagnostics = RAGASFinishReasonDiagnostics(str(path))
            run_id = uuid4()
            diagnostics.on_chat_model_start(
                {},
                [[HumanMessage(content="prompt")]],
                run_id=run_id,
            )
            diagnostics.on_llm_end(_response("stop"), run_id=run_id)
            diagnostics.close()
            self.assertFalse(path.exists())

    def test_unfinished_response_is_persisted_and_redacted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "failures.jsonl"
            diagnostics = RAGASFinishReasonDiagnostics(str(path))
            row_id = uuid4()
            metric_id = uuid4()
            prompt_id = uuid4()
            llm_id = uuid4()

            diagnostics.on_chain_start(
                {},
                {"user_input": "Which value?", "reference": "answer"},
                run_id=row_id,
                metadata={"row_index": 2, "type": "ROW"},
                name="row 2",
            )
            diagnostics.on_chain_start(
                {},
                {
                    "reference": "answer",
                    "retrieved_contexts": ["first", "second"],
                },
                run_id=metric_id,
                parent_run_id=row_id,
                metadata={"type": "METRIC"},
                name="context_entity_recall",
            )

            class PromptData:
                text = "answer"

            diagnostics.on_chain_start(
                {},
                {"data": PromptData()},
                run_id=prompt_id,
                parent_run_id=metric_id,
                metadata={"type": "RAGAS_PROMPT"},
                name="text_entity_extraction",
            )
            diagnostics.on_chat_model_start(
                {},
                [[HumanMessage(content="extract answer entities")]],
                run_id=llm_id,
                parent_run_id=prompt_id,
            )
            diagnostics.on_llm_end(
                _response(
                    "length",
                    headers={
                        "x-request-id": "request-123",
                        "authorization": "Bearer secret",
                        "x-smg-routing-key": "private-user",
                    },
                ),
                run_id=llm_id,
                parent_run_id=prompt_id,
            )
            diagnostics.close()

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["row_index"], 2)
            self.assertEqual(record["sample_number"], 3)
            self.assertEqual(record["metric"], "context_entity_recall")
            self.assertEqual(record["input_phase"], "reference")
            self.assertEqual(
                record["finish_states"][0]["finish_reason"],
                "length",
            )
            self.assertFalse(record["ragas_would_accept"])
            serialized = json.dumps(record)
            self.assertNotIn("Bearer secret", serialized)
            self.assertNotIn("private-user", serialized)
            self.assertIn("[REDACTED]", serialized)


if __name__ == "__main__":
    unittest.main()
