"""
TDD spec for Phase 1a step 1a.3 — resnet50 endpoint differentiation.

Mirrors containers/distilbert/tests/test_endpoints.py. Three new
operation-flavored routes plus a back-compat alias:
  /predict          → alias of /predict/classify, operation=legacy
  /predict/classify → top-5 predictions list (existing /predict shape)
  /predict/top1     → single top-1 result (smaller object)
  /predict/score    → top-1 score number only

Each request emits ml.inference.{request_count, latency, error_count}
with the operation label so the dashboard can group by op.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import types
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from PIL import Image


# ---------------------------------------------------------------------------
# Test rig (see containers/distilbert/tests/test_endpoints.py for the
# rationale behind the import-once + shared-mock pattern).
# ---------------------------------------------------------------------------

_FAKE_MODEL = types.ModuleType("model")
_FAKE_MODEL.is_loaded = lambda: True
_FAKE_MODEL.load_model = lambda: 0.05
_FAKE_PREDICTIONS = [
    {"class": "tabby", "score": 0.50},
    {"class": "tiger_cat", "score": 0.20},
    {"class": "Egyptian_cat", "score": 0.15},
    {"class": "lynx", "score": 0.10},
    {"class": "Persian_cat", "score": 0.05},
]
_FAKE_MODEL.predict = MagicMock(return_value=list(_FAKE_PREDICTIONS))
sys.modules["model"] = _FAKE_MODEL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import main  # noqa: E402


def _tiny_png_b64() -> str:
    """A real, valid 1x1 RGB PNG so the FastAPI handler's
    `Image.open(...).convert("RGB")` step doesn't error before our mocked
    model.predict is reached."""
    img = Image.new("RGB", (1, 1), color="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_PNG_B64 = _tiny_png_b64()


@pytest.fixture
def fresh_app() -> Iterator[tuple]:
    _FAKE_MODEL.predict = MagicMock(return_value=list(_FAKE_PREDICTIONS))
    reader = InMemoryMetricReader()
    app = main.create_app(metric_readers=[reader])
    with TestClient(app) as client:
        yield client, reader, _FAKE_MODEL


def _all_metric_points(reader: InMemoryMetricReader) -> list[tuple[str, dict]]:
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


def _operations_seen(reader: InMemoryMetricReader, metric_name: str) -> set:
    return {
        attrs.get("operation")
        for name, attrs in _all_metric_points(reader)
        if name == metric_name and attrs.get("operation") is not None
    }


# ---------------------------------------------------------------------------
# Regression: legacy /predict still works
# ---------------------------------------------------------------------------

def test_legacy_predict_returns_top5_predictions(fresh_app):
    """webapp and any external caller pinned to /predict must keep
    receiving the existing top-5 list shape."""
    client, _reader, _model = fresh_app
    response = client.post("/predict", json={"image_base64": _PNG_B64})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 5
    assert body["predictions"][0]["class_name"] == "tabby"


def test_legacy_predict_emits_operation_legacy(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict", json={"image_base64": _PNG_B64})
    assert "legacy" in _operations_seen(reader, "ml.inference.request_count")


# ---------------------------------------------------------------------------
# /predict/classify — top-5 (existing behavior, named explicitly)
# ---------------------------------------------------------------------------

def test_predict_classify_returns_top5(fresh_app):
    client, _reader, _model = fresh_app
    response = client.post("/predict/classify", json={"image_base64": _PNG_B64})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["predictions"]) == 5


def test_predict_classify_emits_operation_classify(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict/classify", json={"image_base64": _PNG_B64})
    assert "classify" in _operations_seen(reader, "ml.inference.request_count")


# ---------------------------------------------------------------------------
# /predict/top1 — single top-1 result (no list)
# ---------------------------------------------------------------------------

def test_predict_top1_returns_single_class_and_score(fresh_app):
    """Top-1 callers don't need (or pay to deserialize) the full top-5
    list. Smaller payload + clearer intent."""
    client, _reader, _model = fresh_app
    response = client.post("/predict/top1", json={"image_base64": _PNG_B64})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "predictions" not in body, "top1 should NOT return a list"
    assert body["class_name"] == "tabby"
    assert body["score"] == 0.50


def test_predict_top1_emits_operation_top1(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict/top1", json={"image_base64": _PNG_B64})
    assert "top1" in _operations_seen(reader, "ml.inference.request_count")


# ---------------------------------------------------------------------------
# /predict/score — top-1 score only
# ---------------------------------------------------------------------------

def test_predict_score_returns_score_only(fresh_app):
    client, _reader, _model = fresh_app
    response = client.post("/predict/score", json={"image_base64": _PNG_B64})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["score"] == 0.50
    assert "class_name" not in body, f"score response should omit class_name; got {body}"
    assert "predictions" not in body


def test_predict_score_emits_operation_score(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict/score", json={"image_base64": _PNG_B64})
    assert "score" in _operations_seen(reader, "ml.inference.request_count")


# ---------------------------------------------------------------------------
# Per-route latency + error coverage
# ---------------------------------------------------------------------------

def test_each_route_records_latency_with_its_operation_label(fresh_app):
    client, reader, _model = fresh_app
    client.post("/predict/classify", json={"image_base64": _PNG_B64})
    client.post("/predict/top1", json={"image_base64": _PNG_B64})
    client.post("/predict/score", json={"image_base64": _PNG_B64})
    client.post("/predict", json={"image_base64": _PNG_B64})

    ops = _operations_seen(reader, "ml.inference.latency")
    assert ops >= {"classify", "top1", "score", "legacy"}, (
        f"Expected latency points for all 4 operations. Got: {ops}"
    )


def test_predict_error_emits_error_count_with_operation_label(fresh_app):
    client, reader, fake_model = fresh_app
    fake_model.predict = MagicMock(side_effect=ValueError("boom"))

    response = client.post("/predict/classify", json={"image_base64": _PNG_B64})
    assert response.status_code == 500

    assert "classify" in _operations_seen(reader, "ml.inference.error_count")


def test_invalid_request_returns_400(fresh_app):
    """Regression: a request with neither image_url nor image_base64 must
    still be rejected with 400 — pre-existing behavior."""
    client, _reader, _model = fresh_app
    response = client.post("/predict/classify", json={})
    assert response.status_code == 400, response.text


def test_health_endpoint_returns_status_healthy(fresh_app):
    client, _reader, _model = fresh_app
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
