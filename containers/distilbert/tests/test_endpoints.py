"""
TDD spec for Phase 1a step 1a.2 — distilbert endpoint differentiation.

Expected behavior (derived from operation-breakdown design):
  1. The legacy POST /predict route still works (regression — webapp
     and any external caller pinned to /predict must keep functioning).
  2. Three new POST /predict/{operation} routes exist with distinct
     server-side semantics:
       - /predict/sentiment → full label + score (same model output)
       - /predict/score     → numeric score only (no label string)
       - /predict/binary    → coerced two-bucket POSITIVE/NEGATIVE
  3. Each request, regardless of route, increments
     ml.inference.request_count with `{model: distilbert, operation: <op>}`
     so the Chamber operation-breakdown UI can group RPS by op.
  4. /predict (legacy) emits operation="legacy" so we can spot stragglers
     once webapp / external callers migrate to the new routes.
  5. Latency is recorded with the same operation label.
  6. On model failure, error_count carries the operation label too.
  7. The new /health endpoint behavior is unchanged (smoke check).

The tests construct a FRESH FastAPI app per test via `main.create_app(...)`
with `model.predict` mocked and an InMemoryMetricReader capturing metrics.
The fresh-app pattern is what allows TDD here without restarting the
container or relying on global OTel state.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


# ---------------------------------------------------------------------------
# Test rig: build a fresh distilbert app with model + telemetry mocked.
#
# Strategy: install ONE shared fake `model` module + import `main` ONCE per
# session, so the module-level `app = create_app()` (production fallback)
# runs exactly once. Re-importing per test would spawn a new OTLPMetricExporter
# each time — those run a background retry loop pointed at the unreachable
# default endpoint and add tens of seconds of noise per test.
#
# Per-test isolation is achieved via main.create_app(metric_readers=[reader]),
# which (per setup_telemetry's contract) builds a LOCAL MeterProvider and does
# not touch globals. Each test gets its own counters bound to its own reader.
#
# Tests that need to vary the model's predict behavior reassign
# `fake_model.predict = MagicMock(return_value=...)`. main holds the same
# module reference, so the change is visible immediately.
# ---------------------------------------------------------------------------

# One-time module-level setup: install fake model + path before main imports
_FAKE_MODEL = types.ModuleType("model")
_FAKE_MODEL.is_loaded = lambda: True
_FAKE_MODEL.load_model = lambda: 0.05
_FAKE_MODEL.predict = MagicMock(return_value={"label": "POSITIVE", "score": 0.92})
sys.modules["model"] = _FAKE_MODEL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import main  # noqa: E402  — must follow the model + path setup above


@pytest.fixture
def fresh_app() -> Iterator[tuple]:
    """Each test gets (TestClient, InMemoryMetricReader, _FAKE_MODEL).

    Reset _FAKE_MODEL.predict to the canonical default so a previous test's
    override (e.g. low-confidence binary, error path) doesn't leak.
    """
    _FAKE_MODEL.predict = MagicMock(return_value={"label": "POSITIVE", "score": 0.92})

    reader = InMemoryMetricReader()
    app = main.create_app(metric_readers=[reader])
    # `with TestClient(...)` triggers @app.on_event("startup"), which calls
    # model.load_model() (our mock) and seeds the error counter zero-points.
    with TestClient(app) as client:
        yield client, reader, _FAKE_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_metric_points(reader: InMemoryMetricReader) -> list[tuple[str, dict]]:
    """Return [(metric_name, attributes_dict), ...] across all points."""
    data = reader.get_metrics_data()
    out: list[tuple[str, dict]] = []
    if data is None:
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for dp in metric.data.data_points:
                    out.append((metric.name, dict(dp.attributes)))
    return out


def _operations_seen(reader: InMemoryMetricReader, metric_name: str) -> set[str]:
    """The distinct values of the `operation` label observed on a metric."""
    return {
        attrs.get("operation")
        for name, attrs in _all_metric_points(reader)
        if name == metric_name and attrs.get("operation") is not None
    }


# ---------------------------------------------------------------------------
# Regression: legacy /predict still works
# ---------------------------------------------------------------------------

def test_legacy_predict_returns_200_with_label_and_score(fresh_app):
    """The webapp and any external caller pinned to /predict must keep
    receiving the same response shape. Catching a regression here would
    surface BEFORE we deploy a breaking change to the demo's public API."""
    client, _reader, _model = fresh_app
    response = client.post("/predict", json={"text": "great movie"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "POSITIVE"
    assert body["score"] == 0.92
    assert "latency_ms" in body


def test_legacy_predict_emits_operation_label_legacy(fresh_app):
    """Once we move to the new endpoint set, /predict callers emit
    operation=legacy so we can find them in the dashboard and migrate."""
    client, reader, _model = fresh_app
    client.post("/predict", json={"text": "x"})
    ops = _operations_seen(reader, "ml.inference.request_count")
    assert "legacy" in ops, (
        f"Expected operation='legacy' on /predict request_count. Saw: {ops}"
    )


# ---------------------------------------------------------------------------
# New /predict/sentiment
# ---------------------------------------------------------------------------

def test_predict_sentiment_returns_label_and_score(fresh_app):
    """Functionally equivalent to legacy /predict — the difference is the
    operation label, not the response shape."""
    client, _reader, _model = fresh_app
    response = client.post("/predict/sentiment", json={"text": "great"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "POSITIVE"
    assert body["score"] == 0.92


def test_predict_sentiment_emits_operation_sentiment(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict/sentiment", json={"text": "x"})
    assert "sentiment" in _operations_seen(reader, "ml.inference.request_count")


# ---------------------------------------------------------------------------
# New /predict/score — score-only response
# ---------------------------------------------------------------------------

def test_predict_score_returns_score_only_no_label(fresh_app):
    """Score-only callers don't need (or pay to deserialize) the label
    string. Smaller payload + clearer intent. Still drives the model the
    same way, so latency profile is similar but slightly lower."""
    client, _reader, _model = fresh_app
    response = client.post("/predict/score", json={"text": "x"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["score"] == 0.92
    assert "label" not in body, f"score response should omit label; got {body}"


def test_predict_score_emits_operation_score(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict/score", json={"text": "x"})
    assert "score" in _operations_seen(reader, "ml.inference.request_count")


# ---------------------------------------------------------------------------
# New /predict/binary — coerced POSITIVE / NEGATIVE
# ---------------------------------------------------------------------------

def test_predict_binary_high_confidence_positive_stays_positive(fresh_app):
    """When the model returns POSITIVE with score >= 0.5, /predict/binary
    keeps the POSITIVE label."""
    client, _reader, _model = fresh_app
    response = client.post("/predict/binary", json={"text": "x"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "POSITIVE"
    assert body["score"] == 0.92


def test_predict_binary_low_confidence_falls_to_negative(fresh_app):
    """Low-confidence POSITIVE (score < 0.5) is coerced to NEGATIVE so
    callers can rely on a stable two-bucket signal."""
    client, _reader, fake_model = fresh_app
    fake_model.predict = MagicMock(return_value={"label": "POSITIVE", "score": 0.3})
    response = client.post("/predict/binary", json={"text": "meh"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "NEGATIVE"


def test_predict_binary_negative_stays_negative(fresh_app):
    client, _reader, fake_model = fresh_app
    fake_model.predict = MagicMock(return_value={"label": "NEGATIVE", "score": 0.91})
    response = client.post("/predict/binary", json={"text": "awful"})
    assert response.status_code == 200, response.text
    assert response.json()["label"] == "NEGATIVE"


def test_predict_binary_emits_operation_binary(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict/binary", json={"text": "x"})
    assert "binary" in _operations_seen(reader, "ml.inference.request_count")


# ---------------------------------------------------------------------------
# Per-route metric coverage
# ---------------------------------------------------------------------------

def test_each_route_records_latency_with_its_operation_label(fresh_app):
    """ml.inference.latency must carry `operation` so the dashboard can
    show p50/p95/p99 per operation. Without this, latency rolls up to a
    single service-wide line."""
    client, reader, _model = fresh_app
    client.post("/predict/sentiment", json={"text": "a"})
    client.post("/predict/score", json={"text": "b"})
    client.post("/predict/binary", json={"text": "c"})
    client.post("/predict", json={"text": "d"})

    ops = _operations_seen(reader, "ml.inference.latency")
    assert ops >= {"sentiment", "score", "binary", "legacy"}, (
        f"Expected latency points for all 4 operations. Got: {ops}"
    )


def test_distinct_routes_produce_distinct_request_counts(fresh_app):
    """Three calls to three distinct routes → three distinct
    operation labels in the request_count. Catches a regression where
    a future refactor accidentally hardcodes operation=sentiment."""
    client, reader, _model = fresh_app
    client.post("/predict/sentiment", json={"text": "a"})
    client.post("/predict/score", json={"text": "b"})
    client.post("/predict/binary", json={"text": "c"})

    assert _operations_seen(reader, "ml.inference.request_count") >= {
        "sentiment", "score", "binary"
    }


# ---------------------------------------------------------------------------
# Error path keeps the operation label
# ---------------------------------------------------------------------------

def test_predict_error_emits_error_count_with_operation_label(fresh_app):
    """When the model raises, error_count must include `operation` so the
    dashboard can break down error rate per operation. Without it, all
    errors collapse to one bucket."""
    client, reader, fake_model = fresh_app
    fake_model.predict = MagicMock(side_effect=ValueError("boom"))

    response = client.post("/predict/sentiment", json={"text": "x"})
    assert response.status_code == 500

    ops = _operations_seen(reader, "ml.inference.error_count")
    assert "sentiment" in ops, (
        f"Expected operation=sentiment on error_count. Got: {ops}"
    )


# ---------------------------------------------------------------------------
# Health endpoint regression
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_status_healthy(fresh_app):
    client, _reader, _model = fresh_app
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
