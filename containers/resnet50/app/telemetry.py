import logging
import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_model_load_time: float = 0.0


def set_model_load_time(t: float) -> None:
    global _model_load_time
    _model_load_time = t


def _model_load_time_callback(_options):
    yield metrics.Observation(value=_model_load_time)


def setup_telemetry(app, *, metric_readers=None) -> tuple:
    """Initialize OTel metrics and traces, instrument FastAPI, return (latency,
    request_counter, error_counter, tracer).

    `metric_readers`: when None (production default), readers are constructed
    from env vars (OTEL_EXPORTER_OTLP_ENDPOINT and optional CHAMBER_OTLP_ENDPOINT
    for dual export). Tests pass [InMemoryMetricReader()] to capture metrics
    in-process without going through the OTLP gRPC encoder.
    """
    # See setup_telemetry contract in containers/distilbert/app/telemetry.py:
    # caller-supplied readers ⇒ don't install globally so OTel's set-once
    # rule doesn't block per-test isolation.
    caller_provided_readers = metric_readers is not None

    service_name = os.environ.get("OTEL_SERVICE_NAME", "resnet50")
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://metrics-agent:4318")

    resource = Resource.create({"service.name": service_name})

    # Traces
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
    )
    if not caller_provided_readers:
        trace.set_tracer_provider(trace_provider)
    tracer = trace_provider.get_tracer(service_name)

    # Metrics — caller-supplied readers win (tests pass InMemoryMetricReader);
    # otherwise build the default OTLP readers from env vars.
    if metric_readers is None:
        metric_readers = [
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True),
                export_interval_millis=10000,
            )
        ]
        chamber_endpoint = os.environ.get("CHAMBER_OTLP_ENDPOINT")
        if chamber_endpoint:
            logger.info("Chamber OTLP export enabled → %s", chamber_endpoint)
            metric_readers.append(
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=chamber_endpoint, insecure=True),
                    export_interval_millis=10000,
                )
            )

    latency_view = View(
        instrument_name="ml.inference.latency",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
        ),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers, views=[latency_view])

    if not caller_provided_readers:
        metrics.set_meter_provider(meter_provider)
    meter = meter_provider.get_meter(service_name)

    # Custom metrics
    inference_latency = meter.create_histogram(
        name="ml.inference.latency",
        description="Model inference latency in seconds",
        unit="s",
    )
    request_counter = meter.create_counter(
        name="ml.inference.request_count",
        description="Total inference requests",
    )
    error_counter = meter.create_counter(
        name="ml.inference.error_count",
        description="Total inference errors",
    )
    meter.create_observable_gauge(
        name="ml.model.load_time",
        description="Time taken to load the model in seconds",
        unit="s",
        callbacks=[_model_load_time_callback],
    )

    # Auto-instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    return inference_latency, request_counter, error_counter, tracer
