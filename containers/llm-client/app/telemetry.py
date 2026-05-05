"""OpenTelemetry setup for the llm-client container.

Production builds an OTLP gRPC exporter from env vars
(OTEL_EXPORTER_OTLP_ENDPOINT) and exports both metrics and traces to the
chamber-standalone agent on the host. Tests pass `meter_provider_override`
to capture metrics in-process via an InMemoryMetricReader without going
through OTLP.

Layered next to containers/distilbert/app/telemetry.py for consistency,
even though the OpenAI auto-instrumentation does most of the heavy
lifting (no manual counters / histograms here — gen_ai.* metrics come
from `opentelemetry-instrumentation-openai-v2`).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def setup_telemetry(*, meter_provider_override: Optional[MeterProvider] = None) -> None:
    """Initialize OTel + auto-instrument the OpenAI client.

    `meter_provider_override`: if provided, use this provider (typical test
    pattern with InMemoryMetricReader). If None (production), build a
    PeriodicExportingMetricReader pointed at OTEL_EXPORTER_OTLP_ENDPOINT
    and install the resulting MeterProvider globally.
    """
    service_name = os.environ.get("OTEL_SERVICE_NAME", "llm-client")
    otlp_endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://host.docker.internal:4317"
    )

    if meter_provider_override is not None:
        # Test path — caller-supplied provider. We pass it explicitly into
        # OpenAIInstrumentor.instrument() instead of using globals so each
        # test can have an isolated provider/reader pair. (OTel's
        # set_meter_provider is one-shot — globals would lock the first
        # test's provider in for the whole session.)
        provider = meter_provider_override
    else:
        # Production: build OTLP exporters from env, install globally.
        resource = Resource.create({"service.name": service_name})

        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
        )
        trace.set_tracer_provider(trace_provider)

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True),
            export_interval_millis=10000,
        )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        logger.info("OTel exporters configured for %s -> %s", service_name, otlp_endpoint)

    # Auto-instrument the OpenAI SDK. Passing meter_provider explicitly so
    # the instrumentation binds to OUR provider rather than walking the
    # global registry. Production passes the same provider it just installed
    # globally, so behaviour is identical; tests get isolation.
    OpenAIInstrumentor().instrument(meter_provider=provider)
