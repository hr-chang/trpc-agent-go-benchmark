#
# Tencent is pleased to support the open source community by making trpc-agent-go available.
#
# Copyright (C) 2025 Tencent.  All rights reserved.
#
# trpc-agent-go is licensed under the Apache License Version 2.0.
#
#
"""Opt-in diagnostics for unfinished RAGAS LLM responses."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult


logger = logging.getLogger(__name__)

DIAGNOSTICS_PATH_ENV = "RAGAS_DIAGNOSTICS_PATH"
_ACCEPTED_FINISH_REASONS = frozenset(
    {"stop", "STOP", "MAX_TOKENS", "eos_token"}
)
_SENSITIVE_KEY_FRAGMENTS = (
    "api-key",
    "api_key",
    "authorization",
    "cookie",
    "routing-key",
    "secret",
)
_MAX_CAPTURE_CHARS = 131072
_STOP = object()


def _run_key(run_id: Optional[UUID]) -> Optional[str]:
    return str(run_id) if run_id is not None else None


def _truncate(value: str) -> str:
    if len(value) <= _MAX_CAPTURE_CHARS:
        return value
    return value[:_MAX_CAPTURE_CHARS] + "\n...[truncated]"


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower()
    if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
        return True
    normalized = normalized.replace("-", "_")
    return normalized in {
        "access_token",
        "bearer_token",
        "id_token",
        "refresh_token",
        "token",
    }


def _safe_json_value(value: Any) -> Any:
    """Convert a value to JSON-safe data while removing sensitive fields."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _is_sensitive_key(key)
                else _safe_json_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate(value) if isinstance(value, str) else value
    if hasattr(value, "model_dump"):
        return _safe_json_value(value.model_dump())
    return _truncate(str(value))


def _response_finish_states(response: LLMResult) -> List[Dict[str, Any]]:
    """Extract finish states using the same precedence as RAGAS 0.2.15."""
    states = []
    for generations in response.generations:
        generation = generations[0]
        generation_info = getattr(generation, "generation_info", None)
        reason = None
        source = None

        if generation_info is not None:
            reason = generation_info.get("finish_reason")
            source = "generation_info.finish_reason"
        elif isinstance(generation, ChatGeneration):
            message = getattr(generation, "message", None)
            response_metadata = getattr(message, "response_metadata", {}) or {}
            if response_metadata.get("finish_reason") is not None:
                reason = response_metadata.get("finish_reason")
                source = "response_metadata.finish_reason"
            elif response_metadata.get("stop_reason") is not None:
                reason = response_metadata.get("stop_reason")
                source = "response_metadata.stop_reason"

        accepted_reasons = _ACCEPTED_FINISH_REASONS
        if source == "response_metadata.stop_reason":
            accepted_reasons = accepted_reasons.union({"end_turn"})
        states.append(
            {
                "finish_reason": reason,
                "source": source,
                "accepted": (
                    True
                    if reason is None
                    else reason in accepted_reasons
                ),
            }
        )
    return states


def ragas_would_accept(response: LLMResult) -> bool:
    """Mirror LangchainLLMWrapper.is_finished from RAGAS 0.2.15."""
    accepted = []
    for generations in response.generations:
        generation = generations[0]
        generation_info = getattr(generation, "generation_info", None)

        if generation_info is not None:
            finish_reason = generation_info.get("finish_reason")
            if finish_reason is not None:
                accepted.append(finish_reason in _ACCEPTED_FINISH_REASONS)
            continue

        if not isinstance(generation, ChatGeneration):
            accepted.append(True)
            continue
        message = getattr(generation, "message", None)
        response_metadata = getattr(message, "response_metadata", {}) or {}
        if response_metadata.get("finish_reason") is not None:
            accepted.append(
                response_metadata["finish_reason"]
                in _ACCEPTED_FINISH_REASONS
            )
        elif response_metadata.get("stop_reason") is not None:
            accepted.append(
                response_metadata["stop_reason"]
                in _ACCEPTED_FINISH_REASONS.union({"end_turn"})
            )
        else:
            accepted.append(True)

    return all(accepted)


class _AsyncJSONLWriter:
    """Write queued records on a background thread."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._error: Optional[Exception] = None
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def submit(self, record: Dict[str, Any]) -> None:
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="ragas-diagnostics-writer",
                    daemon=True,
                )
                self._thread.start()
        self._queue.put(record)

    def close(self) -> None:
        if self._thread is None:
            return
        self._queue.put(_STOP)
        self._thread.join()

    def _run(self) -> None:
        output = None
        try:
            while True:
                record = self._queue.get()
                if record is _STOP:
                    break
                if output is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    output = self.path.open("a", encoding="utf-8", buffering=1)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as error:
            self._error = error
            logger.error("Failed to write RAGAS diagnostics: %s", error)
        finally:
            if output is not None:
                output.close()


class RAGASFinishReasonDiagnostics(BaseCallbackHandler):
    """Capture only LLM responses that RAGAS considers unfinished."""

    def __init__(self, path: str):
        self.path = path
        self._writer = _AsyncJSONLWriter(path)
        self._lock = threading.Lock()
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._pending_llm: Dict[str, Dict[str, Any]] = {}
        self._closed = False

    @classmethod
    def from_environment(cls) -> Optional["RAGASFinishReasonDiagnostics"]:
        path = os.environ.get(DIAGNOSTICS_PATH_ENV, "").strip()
        return cls(path) if path else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close()
        if self._writer.error is not None:
            logger.error(
                "RAGAS diagnostics were not fully persisted to %s",
                self.path,
            )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        del tags
        try:
            run_key = _run_key(run_id)
            parent_key = _run_key(parent_run_id)
            with self._lock:
                context = dict(self._contexts.get(parent_key, {}))

            name = kwargs.get("name")
            if not name and isinstance(serialized, dict):
                name = serialized.get("name")
            if name:
                context["chain_name"] = name

            metadata = metadata or {}
            row_index = metadata.get("row_index")
            if row_index is not None:
                context["row_index"] = int(row_index)
                context["sample_number"] = int(row_index) + 1

            chain_type = str(metadata.get("type", ""))
            if chain_type.endswith("METRIC") and name:
                context["metric"] = name
                context["reference"] = inputs.get("reference")
                contexts = inputs.get("retrieved_contexts")
                if isinstance(contexts, list):
                    context["retrieved_contexts"] = "\n".join(
                        str(item) for item in contexts
                    )
            elif chain_type.endswith("RAGAS_PROMPT") and name:
                context["prompt_name"] = name
                prompt_data = inputs.get("data")
                prompt_text = getattr(prompt_data, "text", None)
                if prompt_text is not None:
                    context["input_phase"] = self._input_phase(
                        str(prompt_text),
                        context,
                    )

            question = inputs.get("user_input") or inputs.get("question")
            if question:
                context["question"] = str(question)

            if run_key is not None:
                with self._lock:
                    self._contexts[run_key] = context
        except Exception as error:
            logger.warning("RAGAS diagnostic chain callback failed: %s", error)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        del outputs, parent_run_id, kwargs
        self._discard_context(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        del error, parent_run_id, kwargs
        self._discard_context(run_id)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        del serialized, tags, metadata, kwargs
        try:
            run_key = _run_key(run_id)
            parent_key = _run_key(parent_run_id)
            if run_key is None:
                return
            with self._lock:
                context = dict(self._contexts.get(parent_key, {}))
                self._pending_llm[run_key] = {
                    "context": context,
                    "messages": messages,
                    "started_monotonic": time.monotonic(),
                }
        except Exception as error:
            logger.warning("RAGAS diagnostic LLM start callback failed: %s", error)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        try:
            pending = self._pop_pending(run_id)
            if ragas_would_accept(response):
                return
            self._writer.submit(
                self._failure_record(
                    event="llm_end",
                    pending=pending,
                    response=response,
                )
            )
        except Exception as error:
            logger.warning("RAGAS diagnostic LLM end callback failed: %s", error)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        try:
            pending = self._pop_pending(run_id)
            record = self._failure_record(
                event="llm_error",
                pending=pending,
                response=None,
            )
            record["error_type"] = type(error).__name__
            record["error"] = _truncate(str(error))
            self._writer.submit(record)
        except Exception as callback_error:
            logger.warning(
                "RAGAS diagnostic LLM error callback failed: %s",
                callback_error,
            )

    def _discard_context(self, run_id: UUID) -> None:
        run_key = _run_key(run_id)
        if run_key is None:
            return
        with self._lock:
            self._contexts.pop(run_key, None)

    def _pop_pending(self, run_id: UUID) -> Dict[str, Any]:
        run_key = _run_key(run_id)
        if run_key is None:
            return {}
        with self._lock:
            return self._pending_llm.pop(run_key, {})

    @staticmethod
    def _input_phase(prompt_text: str, context: Dict[str, Any]) -> str:
        if prompt_text == context.get("reference"):
            return "reference"
        if prompt_text == context.get("retrieved_contexts"):
            return "retrieved_contexts"
        return "unknown"

    @staticmethod
    def _serialize_messages(messages: Any) -> Any:
        result = []
        for message_group in messages or []:
            group = []
            for message in message_group:
                group.append(
                    {
                        "type": getattr(message, "type", type(message).__name__),
                        "content": _safe_json_value(
                            getattr(message, "content", str(message))
                        ),
                    }
                )
            result.append(group)
        return result

    @staticmethod
    def _serialize_response(response: Optional[LLMResult]) -> Any:
        if response is None:
            return None
        generations = []
        for group in response.generations:
            serialized_group = []
            for generation in group:
                serialized_group.append(
                    {
                        "text": _truncate(getattr(generation, "text", "")),
                        "generation_info": _safe_json_value(
                            getattr(generation, "generation_info", None)
                        ),
                        "response_metadata": _safe_json_value(
                            getattr(
                                getattr(generation, "message", None),
                                "response_metadata",
                                None,
                            )
                        ),
                        "usage_metadata": _safe_json_value(
                            getattr(
                                getattr(generation, "message", None),
                                "usage_metadata",
                                None,
                            )
                        ),
                    }
                )
            generations.append(serialized_group)
        return {
            "generations": generations,
            "llm_output": _safe_json_value(response.llm_output),
        }

    def _failure_record(
        self,
        event: str,
        pending: Dict[str, Any],
        response: Optional[LLMResult],
    ) -> Dict[str, Any]:
        context = pending.get("context", {})
        messages = self._serialize_messages(pending.get("messages"))
        response_data = self._serialize_response(response)
        response_json = json.dumps(
            response_data,
            ensure_ascii=False,
            sort_keys=True,
        )
        started = pending.get("started_monotonic")
        latency_ms = (
            round((time.monotonic() - started) * 1000, 2)
            if isinstance(started, (int, float))
            else None
        )
        return {
            "schema_version": 1,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "row_index": context.get("row_index"),
            "sample_number": context.get("sample_number"),
            "question": _safe_json_value(context.get("question")),
            "metric": context.get("metric"),
            "prompt_name": context.get("prompt_name"),
            "input_phase": context.get("input_phase"),
            "latency_ms": latency_ms,
            "finish_states": (
                _response_finish_states(response)
                if response is not None
                else []
            ),
            "ragas_would_accept": (
                ragas_would_accept(response)
                if response is not None
                else None
            ),
            "messages": messages,
            "response": response_data,
            "response_sha256": hashlib.sha256(
                response_json.encode("utf-8")
            ).hexdigest(),
        }
