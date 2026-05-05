"""
TDD spec for Phase 1a step 1a.1 — resnet50 mirror of distilbert's test.

Same expectation as distilbert's variant: the FastAPI auto-instrumentation
must attach `http.route` to every `http.server.*` metric data point so the
Chamber dashboard can break metrics down by operation.

This file is the regression guard for resnet50's OTel deps. If it fails,
the dep cohort drifted out of sync — fix the version, do NOT relax the
assertion. See containers/distilbert/tests/test_http_semconv_emission.py
for the full rationale.
"""
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def _build_app_with_reader() -> tuple[FastAPI, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    app = FastAPI()

    @app.post("/predict/classify")
    def predict_classify():
        return {"predictions": [{"class_name": "cat", "score": 0.92}]}

    @app.post("/predict/topk")
    def predict_topk():
        return {"predictions": []}

    @app.post("/predict/features")
    def predict_features():
        return {"features": [0.0] * 16}

    FastAPIInstrumentor().instrument_app(app, meter_provider=provider)
    return app, reader


def _flatten_metric_points(reader: InMemoryMetricReader) -> list[tuple[str, dict]]:
    metrics_data = reader.get_metrics_data()
    flat: list[tuple[str, dict]] = []
    if metrics_data is None:
        return flat
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for dp in metric.data.data_points:
                    flat.append((metric.name, dict(dp.attributes)))
    return flat


def test_semconv_opt_in_is_set_for_test_process():
    assert os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN") in ("http", "http/dup"), (
        "Tests in this file require stable HTTP semconv opt-in. "
        f"Got OTEL_SEMCONV_STABILITY_OPT_IN={os.environ.get('OTEL_SEMCONV_STABILITY_OPT_IN')!r}"
    )


def test_http_route_emitted_on_resnet50_routes():
    """
    All three resnet50 prediction routes must surface http.route on the
    http.server.* metric so the dashboard can break down by operation.
    """
    app, reader = _build_app_with_reader()
    client = TestClient(app)

    client.post("/predict/classify")
    client.post("/predict/topk")
    client.post("/predict/features")

    points = _flatten_metric_points(reader)
    routes_seen = {
        attrs.get("http.route")
        for name, attrs in points
        if name.startswith("http.server") and attrs.get("http.route")
    }
    expected = {"/predict/classify", "/predict/topk", "/predict/features"}
    assert expected.issubset(routes_seen), (
        f"Expected http.route values {expected}. Got: {routes_seen}"
    )
