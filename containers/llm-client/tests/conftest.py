"""Test fixtures for the llm-client container.

The OpenAI auto-instrumentation patches the SDK at runtime when
`OpenAIInstrumentor().instrument()` is called. Tests rely on that
mechanism but DO NOT actually hit a real LLM — `respx` mocks the
underlying httpx transport so we get realistic SDK responses without
network.
"""
import os

# Set defaults BEFORE OTel imports so any module-level reads pick them up.
os.environ.setdefault("OTEL_SERVICE_NAME", "llm-client-test")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
