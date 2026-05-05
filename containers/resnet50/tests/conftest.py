"""Test fixtures for resnet50 container.

Mirrors `containers/distilbert/tests/conftest.py`. Same OTel constraints
apply: `OTEL_SEMCONV_STABILITY_OPT_IN` MUST be set before the OTel modules
import so the FastAPI instrumentation initializes with stable HTTP semconv.
"""
import os

os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "http/dup")
os.environ.setdefault("OTEL_SERVICE_NAME", "resnet50-test")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
