//
// Tencent is pleased to support the open source community by making trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//

package main

import "testing"

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
