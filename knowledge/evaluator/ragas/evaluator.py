#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#
#
"""
RAGAS Evaluation for RAG Systems.

This module provides evaluation capabilities using RAGAS metrics
to assess the quality of RAG systems.
"""

import math
import os
import sys
from typing import Any, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_entities_recall import ContextEntityRecall
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
from util import get_config
from evaluator.base import Evaluator, EvaluationSample
from evaluator.ragas.diagnostics import RAGASFinishReasonDiagnostics
from run_artifacts import endpoint_identity


class RAGASEvaluator(Evaluator):
    """Evaluator using RAGAS metrics for RAG quality assessment."""

    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_workers: int = 4,
        timeout: int = 6000,
    ):
        """
        Initialize the RAGAS evaluator.

        Args:
            llm_model: OpenAI LLM model for evaluation. Defaults to EVAL_MODEL_NAME env var.
            embedding_model: OpenAI embedding model for evaluation. Defaults to EMBEDDING_MODEL env var.
            base_url: OpenAI API base URL. Defaults to EVAL_BASE_URL env var.
            api_key: OpenAI API key. Defaults to EVAL_API_KEY env var.
            max_workers: Maximum number of concurrent workers for evaluation.
            timeout: Timeout in seconds for each LLM call.
        """
        config = get_config()
        model_is_explicit = bool(llm_model) or config[
            "eval_model_explicit"
        ]
        api_key_is_explicit = bool(api_key) or config[
            "eval_api_key_explicit"
        ]
        base_url_is_explicit = bool(base_url) or config[
            "eval_base_url_explicit"
        ]

        # Use evaluation-specific config (can be different from knowledge model)
        llm_model = llm_model or config["eval_model_name"]
        embedding_model = embedding_model or config["embedding_model"]
        base_url = base_url or config["eval_base_url"]
        api_key = api_key or config["eval_api_key"]
        judge_endpoint = endpoint_identity(base_url)
        answer_endpoint = endpoint_identity(config["base_url"])
        self.evaluation_runtime_config = {
            "model_name": llm_model,
            "embedding_model": embedding_model,
            "endpoint": judge_endpoint,
            "embedding_endpoint": endpoint_identity(
                config["embedding_base_url"]
            ),
            "header_names": sorted(config["eval_headers"].keys()),
            "embedding_header_names": sorted(
                config["embedding_headers"].keys()
            ),
            "model_explicit": model_is_explicit,
            "api_key_explicit": api_key_is_explicit,
            "base_url_explicit": base_url_is_explicit,
            "model_separate_from_answer": (
                str(llm_model).strip().lower()
                != str(config["model_name"]).strip().lower()
            ),
            "api_key_separate_from_answer": bool(
                api_key
                and config["api_key"]
                and api_key != config["api_key"]
            ),
            "endpoint_separate_from_answer": bool(
                judge_endpoint
                and answer_endpoint
                and judge_endpoint != answer_endpoint
            ),
        }

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            api_key=SecretStr(api_key) if api_key else None,
            base_url=base_url,
            default_headers=config["eval_headers"] or None,
            max_tokens=40960,
        )
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=SecretStr(config["embedding_api_key"]) if config["embedding_api_key"] else None,
            base_url=config["embedding_base_url"],
            default_headers=config["embedding_headers"] or None,
            tiktoken_enabled=False,  # Disable tiktoken for non-OpenAI models
            check_embedding_ctx_length=False,  # Skip context length check
        )
        self.run_config = RunConfig(
            max_workers=max_workers,
            timeout=timeout,
        )
        self.last_metrics: Optional[dict] = None

    def get_runtime_config(self) -> dict:
        """Return the evaluator configuration without credentials."""
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in self.evaluation_runtime_config.items()
        }

    def evaluate(self, samples: List[EvaluationSample]) -> str:
        """
        Evaluate a list of samples using RAGAS metrics.

        Args:
            samples: List of EvaluationSample objects.

        Returns:
            Formatted evaluation result as string.
        """
        self.last_metrics = None
        self.last_metrics = self._compute_metrics(samples)
        return self._format_results(self.last_metrics)

    def _compute_metrics(self, samples: List[EvaluationSample]) -> dict:
        """Compute RAGAS metrics for samples."""
        dataset = Dataset.from_dict(
            {
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [s.contexts for s in samples],
                "ground_truth": [s.ground_truth for s in samples],
            }
        )

        diagnostics = RAGASFinishReasonDiagnostics.from_environment()
        try:
            result: Any = evaluate(
                dataset,
                metrics=[
                    # Answer quality metrics
                    Faithfulness(),
                    AnswerRelevancy(),
                    AnswerCorrectness(),
                    AnswerSimilarity(),
                    # Context quality metrics
                    ContextPrecision(),
                    ContextRecall(),
                    ContextEntityRecall(),
                ],
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=self.run_config,
                callbacks=[diagnostics] if diagnostics is not None else None,
            )
        finally:
            if diagnostics is not None:
                diagnostics.close()

        records = result.to_pandas().to_dict(orient="records")
        metric_aliases = {
            "faithfulness": ("faithfulness",),
            "answer_relevancy": ("answer_relevancy",),
            "answer_correctness": ("answer_correctness",),
            "answer_similarity": (
                "answer_similarity",
                "semantic_similarity",
            ),
            "context_precision": ("context_precision",),
            "context_recall": ("context_recall",),
            "context_entity_recall": ("context_entity_recall",),
        }

        aggregate = {}
        metric_counts = {}
        per_sample = []
        for index, record in enumerate(records):
            normalized = {"sample_index": index}
            for metric, aliases in metric_aliases.items():
                numeric_value = None
                for name in aliases:
                    if name not in record:
                        continue
                    try:
                        candidate = float(record.get(name))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(candidate):
                        numeric_value = candidate
                        break
                normalized[metric] = numeric_value
            per_sample.append(normalized)

        expected = len(samples)
        for metric in metric_aliases:
            values = [
                record[metric]
                for record in per_sample
                if record[metric] is not None
            ]
            aggregate[metric] = sum(values) / len(values) if values else 0.0
            metric_counts[metric] = {
                "expected": expected,
                "finite": len(values),
                "missing": expected - len(values),
            }

        return {
            "aggregate": aggregate,
            "metric_counts": metric_counts,
            "per_sample": per_sample,
        }

    def _format_results(self, metrics: dict) -> str:
        """Format metrics into a readable string."""
        aggregate = metrics["aggregate"]
        result = []
        result.append("=== RAGAS Evaluation Results ===\n")
        result.append("--- Answer Quality ---")
        result.append(f"Faithfulness:        {aggregate['faithfulness']:.4f}")
        result.append(f"Answer Relevancy:    {aggregate['answer_relevancy']:.4f}")
        result.append(f"Answer Correctness:  {aggregate['answer_correctness']:.4f}")
        result.append(f"Answer Similarity:   {aggregate['answer_similarity']:.4f}")
        result.append("\n--- Context Quality ---")
        result.append(f"Context Precision:   {aggregate['context_precision']:.4f}")
        result.append(f"Context Recall:      {aggregate['context_recall']:.4f}")
        result.append(f"Context Entity Recall: {aggregate['context_entity_recall']:.4f}")
        return "\n".join(result)

    def get_metrics_dict(self, samples: List[EvaluationSample]) -> dict:
        """
        Get raw metrics dictionary (for backward compatibility).

        Args:
            samples: List of EvaluationSample objects.

        Returns:
            Dictionary containing evaluation metrics.
        """
        metrics = self._compute_metrics(samples)
        return {
            **metrics["aggregate"],
            "detailed_results": metrics["per_sample"],
            **metrics,
        }
