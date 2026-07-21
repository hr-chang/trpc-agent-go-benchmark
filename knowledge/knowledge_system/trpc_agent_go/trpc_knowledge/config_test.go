//
// Tencent is pleased to support the open source community by making trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//

package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestGatewayHeadersUseSeparatePrefixes(t *testing.T) {
	t.Setenv("LLM_SMG_ROUTING_KEY", "llm-route")
	t.Setenv("LLM_SMG_AGENT_NAME", "llm-agent")
	t.Setenv("EMBEDDING_SMG_ROUTING_KEY", "embedding-route")
	t.Setenv("EMBEDDING_SMG_AGENT_NAME", "embedding-agent")

	llmHeaders := gatewayHeaders("LLM")
	embeddingHeaders := gatewayHeaders("EMBEDDING")

	if got := llmHeaders["X-SMG-Routing-Key"]; got != "llm-route" {
		t.Fatalf("LLM routing header = %q, want llm-route", got)
	}
	if got := embeddingHeaders["X-SMG-Routing-Key"]; got != "embedding-route" {
		t.Fatalf("embedding routing header = %q, want embedding-route", got)
	}
	if got := llmHeaders["X-SMG-Agent-Name"]; got != "llm-agent" {
		t.Fatalf("LLM agent header = %q, want llm-agent", got)
	}
	if got := embeddingHeaders["X-SMG-Agent-Name"]; got != "embedding-agent" {
		t.Fatalf("embedding agent header = %q, want embedding-agent", got)
	}
}

func TestBuildRuntimeConfigIsCompleteAndSecretFree(t *testing.T) {
	t.Setenv("EMBEDDING_MODEL", "bge-m3")
	t.Setenv("LLM_SMG_ROUTING_KEY", "secret-llm-route")
	t.Setenv("EMBEDDING_SMG_AGENT_NAME", "secret-embedding-agent")
	t.Setenv("PGVECTOR_PASSWORD", "secret-pg-password")
	t.Setenv(
		"OPENAI_BASE_URL",
		"https://user:secret-llm-url@example.test/v1?token=secret",
	)
	t.Setenv(
		"EMBEDDING_BASE_URL",
		"user:secret-embedding-url@embedding.test:8443/v1?token=secret",
	)

	svc, err := NewKnowledgeServiceWithConfig(&ServiceConfig{
		StoreType:          VectorStoreInMemory,
		ModelName:          "glm-5.2",
		SearchMode:         0,
		HybridVectorWeight: 0.99999,
		HybridTextWeight:   0.00001,
	})
	if err != nil {
		t.Fatalf("NewKnowledgeServiceWithConfig() error = %v", err)
	}

	cfg := buildRuntimeConfig(context.Background(), svc)
	if got := cfg["model_name"]; got != "glm-5.2" {
		t.Fatalf("model_name = %v, want glm-5.2", got)
	}
	if got := cfg["embedding_model"]; got != "bge-m3" {
		t.Fatalf("embedding_model = %v, want bge-m3", got)
	}
	if got := cfg["prompt_max_searches"]; got != promptMaxSearches {
		t.Fatalf("prompt_max_searches = %v, want %d", got, promptMaxSearches)
	}
	if got := cfg["hard_max_tool_iterations"]; got != hardMaxToolIterations {
		t.Fatalf("hard_max_tool_iterations = %v, want %d", got, hardMaxToolIterations)
	}
	if got := cfg["index_document_count"]; got != 0 {
		t.Fatalf("index_document_count = %v, want 0", got)
	}

	module, ok := cfg["framework_module"].(map[string]string)
	if !ok {
		t.Fatalf("framework_module type = %T, want map[string]string", cfg["framework_module"])
	}
	if module["path"] != frameworkModulePath || module["version"] == "" {
		t.Fatalf("framework_module = %#v, want path and version", module)
	}

	encoded, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal(config) error = %v", err)
	}
	for _, secret := range []string{
		"secret-llm-route",
		"secret-embedding-agent",
		"secret-pg-password",
		"secret-llm-url",
		"secret-embedding-url",
		"token=secret",
	} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("runtime config leaked secret %q: %s", secret, encoded)
		}
	}
	if !strings.Contains(string(encoded), "X-SMG-Routing-Key") {
		t.Fatalf("runtime config does not include the LLM header name: %s", encoded)
	}
	if !strings.Contains(string(encoded), "X-SMG-Agent-Name") {
		t.Fatalf("runtime config does not include the embedding header name: %s", encoded)
	}
	if got := cfg["llm_endpoint"]; got != "https://example.test/v1" {
		t.Fatalf("llm_endpoint = %v, want https://example.test/v1", got)
	}
	if got := cfg["embedding_endpoint"]; got != "embedding.test:8443/v1" {
		t.Fatalf("embedding_endpoint = %v, want embedding.test:8443/v1", got)
	}
}
