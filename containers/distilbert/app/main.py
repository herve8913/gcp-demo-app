import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import model
from telemetry import set_model_load_time, setup_telemetry


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    latency_ms: float


class ScoreResponse(BaseModel):
    score: float
    latency_ms: float


# Operation values used as the OTel `operation` attribute on ml.inference.*
# metrics. Kept as a tuple so the startup hook can seed a zero-point counter
# per operation (gives Prometheus a series even before traffic flows).
_OPERATIONS = ("legacy", "sentiment", "score", "binary")


def create_app(*, metric_readers=None) -> FastAPI:
    """Build a fresh FastAPI app. Production calls this once at import time
    (`app = create_app()` below). Tests call it per-test with an
    InMemoryMetricReader to capture metrics without touching OTLP."""
    app = FastAPI(title="DistilBERT Sentiment Analysis")

    inference_latency, request_counter, error_counter, tracer = setup_telemetry(
        app, metric_readers=metric_readers
    )

    def _record(operation: str, latency: float, error: Optional[Exception] = None) -> None:
        labels = {"model": "distilbert", "operation": operation}
        request_counter.add(1, labels)
        if error is None:
            inference_latency.record(latency, labels)
        else:
            error_counter.add(1, {**labels, "error_type": type(error).__name__})

    def _run_inference(text: str, operation: str) -> tuple[dict, float]:
        start = time.time()
        try:
            result = model.predict(text)
        except Exception as e:
            latency = time.time() - start
            _record(operation, latency, error=e)
            raise HTTPException(status_code=500, detail=str(e))
        latency = time.time() - start
        _record(operation, latency)
        return result, latency

    @app.on_event("startup")
    def startup() -> None:
        load_time = model.load_model()
        set_model_load_time(load_time)
        # Seed each operation's error series so panels render even at zero
        # error rate, and so the dashboard's discovery sees the operation
        # label key on day one.
        for op in _OPERATIONS:
            error_counter.add(0, {"model": "distilbert", "operation": op})

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy", "model_loaded": model.is_loaded()}

    @app.post("/predict", response_model=PredictResponse)
    def predict_legacy(request: PredictRequest) -> PredictResponse:
        """Alias of /predict/sentiment kept for back-compat. Tagged
        `operation=legacy` so dashboards can spot stragglers and migrate."""
        with tracer.start_as_current_span("distilbert.inference.legacy"):
            result, latency = _run_inference(request.text, operation="legacy")
        return PredictResponse(
            label=result["label"],
            score=result["score"],
            latency_ms=round(latency * 1000, 2),
        )

    @app.post("/predict/sentiment", response_model=PredictResponse)
    def predict_sentiment(request: PredictRequest) -> PredictResponse:
        with tracer.start_as_current_span("distilbert.inference.sentiment"):
            result, latency = _run_inference(request.text, operation="sentiment")
        return PredictResponse(
            label=result["label"],
            score=result["score"],
            latency_ms=round(latency * 1000, 2),
        )

    @app.post("/predict/score", response_model=ScoreResponse)
    def predict_score(request: PredictRequest) -> ScoreResponse:
        """Numeric-only response. Smaller payload, no label string."""
        with tracer.start_as_current_span("distilbert.inference.score"):
            result, latency = _run_inference(request.text, operation="score")
        return ScoreResponse(
            score=result["score"],
            latency_ms=round(latency * 1000, 2),
        )

    @app.post("/predict/binary", response_model=PredictResponse)
    def predict_binary(request: PredictRequest) -> PredictResponse:
        """Coerced two-bucket polarity: high-confidence POSITIVE stays
        POSITIVE, everything else collapses to NEGATIVE. Useful when the
        caller wants a stable boolean signal."""
        with tracer.start_as_current_span("distilbert.inference.binary"):
            result, latency = _run_inference(request.text, operation="binary")
        coerced = (
            "POSITIVE"
            if result["label"] == "POSITIVE" and result["score"] >= 0.5
            else "NEGATIVE"
        )
        return PredictResponse(
            label=coerced,
            score=result["score"],
            latency_ms=round(latency * 1000, 2),
        )

    return app


app = create_app()
