#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#

import json
import math
import pathlib
import sys
import tempfile
import unittest


KNOWLEDGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KNOWLEDGE_ROOT))

from run_artifacts import (  # noqa: E402
    BASELINE_EVIDENCE_SCOPE,
    BASELINE_RUN_KIND,
    SCHEMA_VERSION,
    RunArtifactWriter,
    atomic_write_json,
    build_sample_id,
    endpoint_identity,
    load_sample_checkpoint,
)


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_kind": BASELINE_RUN_KIND,
        "evidence_scope": BASELINE_EVIDENCE_SCOPE,
        "knowledge_base": "trpc-agent-go",
        "dataset": "huggingface",
        "evaluator": "ragas",
        "retrieval_k": 4,
        "skip_load": True,
        "models": {
            "answer": "glm-5.2",
            "judge": "glm-5.2",
            "embedding": "bge-m3",
        },
        "runtime_config": {
            "vectorstore": "pgvector",
            "pg_table": "baseline",
            "index_document_count": 123,
            "framework_module": {
                "path": "trpc.group/trpc-go/trpc-agent-go",
                "version": "v1.7.0",
            },
        },
        "repositories": {
            "benchmark": {"commit": "abc123", "dirty": False},
        },
    }


class RunArtifactsTest(unittest.TestCase):
    def test_endpoint_identity_removes_credentials_query_and_fragment(self):
        endpoint = endpoint_identity(
            "https://user:secret@example.test:8443/v1?token=secret#part"
        )
        self.assertEqual(endpoint, "https://example.test:8443/v1")
        self.assertNotIn("secret", endpoint)
        self.assertNotIn("user", endpoint)
        self.assertEqual(
            endpoint_identity("user:secret@example.test:8443/v1?token=x"),
            "example.test:8443/v1",
        )

    def test_atomic_write_converts_non_finite_numbers_to_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            atomic_write_json(
                str(path),
                {"nan": math.nan, "infinity": math.inf},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"nan": None, "infinity": None})
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_checkpoint_supports_ordered_partial_progress_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(pathlib.Path(directory) / "run.json")
            sample_ids = [
                build_sample_id("huggingface", "q1", "g1", 0),
                build_sample_id("huggingface", "q2", "g2", 1),
            ]
            writer = RunArtifactWriter(output, _manifest(), sample_ids)
            initial_fingerprint = writer.run_fingerprint
            refreshed_manifest = _manifest()
            refreshed_manifest["runtime_config"][
                "index_document_count"
            ] = 456
            writer.refresh_manifest(refreshed_manifest)
            self.assertNotEqual(
                writer.run_fingerprint,
                initial_fingerprint,
            )
            records = [
                {
                    "sample_index": 0,
                    "sample_id": sample_ids[0],
                    "question": "q1",
                    "ground_truth": "g1",
                    "answer": "a1",
                    "contexts": ["c1"],
                    "retrieval_queries": ["q1"],
                    "tool_call_count": 1,
                    "status": "success",
                    "error": None,
                    "elapsed_seconds": 1.0,
                }
            ]
            writer.write_samples(records)

            checkpoint = load_sample_checkpoint(writer.paths["samples"])
            self.assertEqual(checkpoint["samples"], records)
            self.assertEqual(checkpoint["expected_sample_ids"], sample_ids)

            checkpoint["samples"][0]["question"] = "tampered"
            atomic_write_json(writer.paths["samples"], checkpoint)
            with self.assertRaisesRegex(ValueError, "samples_digest"):
                load_sample_checkpoint(writer.paths["samples"])


if __name__ == "__main__":
    unittest.main()
