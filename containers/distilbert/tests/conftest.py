"""Test fixtures for distilbert container.

These tests verify behaviors required by Chamber's per-operation dashboard
breakdown. They do NOT require GPU/model weights — `app.model` is mocked
when needed.
"""
import os

# Stable HTTP semconv opt-in MUST be set before any opentelemetry imports
# in any test module — the instrumentation reads it at import time. Setting
# it here in conftest ensures it is in place before pytest collects tests.
os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "http/dup")
os.environ.setdefault("OTEL_SERVICE_NAME", "distilbert-test")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
