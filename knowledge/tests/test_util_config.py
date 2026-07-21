#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#

import os
import pathlib
import sys
import unittest
from unittest import mock


KNOWLEDGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KNOWLEDGE_ROOT))

from util import get_config  # noqa: E402


class UtilConfigTest(unittest.TestCase):
    def test_explicit_judge_endpoint_and_key_do_not_fall_back(self):
        environment = {
            "MODEL_NAME": "glm-5.2",
            "OPENAI_API_KEY": "answer-key",
            "OPENAI_BASE_URL": "https://answer.test/v1",
            "EVAL_MODEL_NAME": "economical-independent-judge",
            "EVAL_API_KEY": "judge-key",
            "EVAL_BASE_URL": "https://judge.test/v1",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            config = get_config()

        self.assertEqual(
            config["eval_model_name"],
            "economical-independent-judge",
        )
        self.assertEqual(config["eval_api_key"], "judge-key")
        self.assertEqual(
            config["eval_base_url"],
            "https://judge.test/v1",
        )
        self.assertTrue(config["eval_model_explicit"])
        self.assertTrue(config["eval_api_key_explicit"])
        self.assertTrue(config["eval_base_url_explicit"])

    def test_missing_judge_settings_are_marked_as_fallbacks(self):
        environment = {
            "OPENAI_API_KEY": "answer-key",
            "OPENAI_BASE_URL": "https://answer.test/v1",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            config = get_config()

        self.assertEqual(config["eval_api_key"], "answer-key")
        self.assertEqual(
            config["eval_base_url"],
            "https://answer.test/v1",
        )
        self.assertFalse(config["eval_model_explicit"])
        self.assertFalse(config["eval_api_key_explicit"])
        self.assertFalse(config["eval_base_url_explicit"])


if __name__ == "__main__":
    unittest.main()
