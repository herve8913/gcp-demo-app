"""
TDD spec for Phase 1a step 1a.1.

Expected behavior (derived from Chamber Service Analytics design):
The FastAPI auto-instrumentation MUST attach `http.route` (the *templated*
route, e.g. `/items/{item_id}`) to every `http.server.*` metric data point.
This is the precondition that lets the dashboard slice latency, error rate
and RPS by operation. The Chamber operation-key auto-detector prefers
`http.route` (stable, cross-language semconv) over `http.target` (Python
FastAPI happens to put the templated route here under legacy semconv, but
most other languages put the raw URL there — uncapped cardinality risk).

Two preconditions both needed:
  1. opentelemetry-instrumentation-fastapi >= 0.46b0 (was 0.45b0; the older
     release does not attach http.route to METRICS data points at all).
  2. OTEL_SEMCONV_STABILITY_OPT_IN=http (or http/dup) set BEFORE the
     instrumentation modules import. Default (unset) emits ONLY legacy
     attributes, with the templated route under `http.target` instead.
     conftest.py sets http/dup for test runs.

If a test fails, the root cause is one of those two — fix the dep version
or the env var, do NOT relax the assertion.
"""
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def test_semconv_opt_in_is_set_for_test_process():
    """Defensive: the conftest must set OTEL_SEMCONV_STABILITY_OPT_IN before
    OTel imports. If a future change moves it past the import barrier, the
    other tests would silently fall back to legacy semconv and miss the
    `http.route` regression. This assertion makes the dependency explicit."""
    assert os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN") in ("http", "http/dup"), (
        "Tests in this file require stable HTTP semconv opt-in. "
        f"Got OTEL_SEMCONV_STABILITY_OPT_IN={os.environ.get('OTEL_SEMCONV_STABILITY_OPT_IN')!r}"
    )


def _build_app_with_reader() -> tuple[FastAPI, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    app = FastAPI()

    @app.get("/items/{item_id}")
    def get_item(item_id: str):
        return {"item_id": item_id}

    @app.post("/predict/sentiment")
    def predict_sentiment():
        return {"label": "POSITIVE", "score": 0.95}

    @app.post("/predict/classify")
    def predict_classify():
        return {"label": "NEUTRAL", "score": 0.5}

    FastAPIInstrumentor().instrument_app(app, meter_provider=provider)
    return app, reader


def _flatten_metric_points(reader: InMemoryMetricReader) -> list[tuple[str, dict]]:
    """Return [(metric_name, attributes_dict), ...] across all collected data points."""
    metrics_data = reader.get_metrics_data()
    flat: list[tuple[str, dict]] = []
    if metrics_data is None:
        return flat
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                # `data` may be Histogram, Sum, or Gauge — all expose `data_points`
                for dp in metric.data.data_points:
                    flat.append((metric.name, dict(dp.attributes)))
    return flat


def test_http_route_emitted_on_templated_route():
    """
    A request to a path-templated FastAPI route MUST surface
    http.route = "/items/{item_id}" on the http.server.* histogram metric.
    Anything else (raw URL, missing) breaks dashboard breakdown.
    """
    app, reader = _build_app_with_reader()
    client = TestClient(app)

    response = client.get("/items/abc-123")
    assert response.status_code == 200

    points = _flatten_metric_points(reader)
    http_points = [(name, attrs) for name, attrs in points if name.startswith("http.server")]
    assert http_points, (
        f"Expected at least one http.server.* metric data point. "
        f"Saw metric names: {sorted({n for n, _ in points})}"
    )

    routes = [attrs.get("http.route") for _, attrs in http_points]
    assert any(route == "/items/{item_id}" for route in routes), (
        f"Expected http.route='/items/{{item_id}}' on http.server metric. "
        f"Routes seen: {routes}. Full points: {http_points}"
    )


def test_distinct_routes_produce_distinct_http_route_values():
    """
    Two distinct routes hit in the same process must yield two distinct
    http.route values on the metric. This is the breakdown precondition.
    """
    app, reader = _build_app_with_reader()
    client = TestClient(app)

    client.get("/items/a")
    client.get("/items/b")  # same route, different param
    client.post("/predict/sentiment")
    client.post("/predict/classify")

    points = _flatten_metric_points(reader)
    routes_seen = {
        attrs.get("http.route")
        for name, attrs in points
        if name.startswith("http.server") and attrs.get("http.route")
    }

    expected = {"/items/{item_id}", "/predict/sentiment", "/predict/classify"}
    assert expected.issubset(routes_seen), (
        f"Expected http.route values {expected} to all be present. "
        f"Saw: {routes_seen}"
    )


def test_http_route_uses_template_not_raw_url():
    """
    Requests with different path-param values to the SAME templated route
    must collapse to ONE http.route value (the template), proving the
    instrumentation does not leak per-request URLs into label cardinality.
    """
    app, reader = _build_app_with_reader()
    client = TestClient(app)

    client.get("/items/alpha")
    client.get("/items/beta")
    client.get("/items/gamma")

    points = _flatten_metric_points(reader)
    items_routes = {
        attrs.get("http.route")
        for name, attrs in points
        if name.startswith("http.server")
        and attrs.get("http.route", "").startswith("/items")
    }
    assert items_routes == {"/items/{item_id}"}, (
        f"Expected exactly one templated route value. Saw: {items_routes}"
    )


def test_otlp_encoder_handles_observable_gauge_with_default_exemplar():
    """
    Regression: opentelemetry==1.28.0 raised EncodingException when the
    OTLP encoder serialized observable Gauge metrics that the SDK had
    auto-attached an Exemplar to (span_id=None, trace_id=None,
    filtered_attributes=None). Fixed in 1.28.1.

    Without this guard, the bug surfaces only at deploy time as a runtime
    crash inside the export thread — silent to local pytest, fatal to
    production. This test exercises the encoder on the same metric shape
    that ml.model.load_time produces in main.py.

    See: https://github.com/open-telemetry/opentelemetry-python/issues/4250
    """
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.exporter.otlp.proto.common._internal.metrics_encoder import (
        encode_metrics,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("regression-test")

    def _gauge_cb(_options):
        # main.py passes a real value here; behavior is identical
        yield otel_metrics.Observation(value=2.045)

    meter.create_observable_gauge(
        name="ml.model.load_time",
        description="Time taken to load the model in seconds",
        unit="s",
        callbacks=[_gauge_cb],
    )

    metrics_data = reader.get_metrics_data()
    assert metrics_data is not None

    # The crash happened inside this call. If the deps regress, this raises
    # opentelemetry.exporter.otlp.proto.common._internal.metrics_encoder.EncodingException.
    encoded = encode_metrics(metrics_data)
    assert encoded is not None
