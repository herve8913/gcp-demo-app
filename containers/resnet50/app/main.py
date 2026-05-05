import asyncio
import base64
import io
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

import model
from telemetry import set_model_load_time, setup_telemetry


class PredictRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class Prediction(BaseModel):
    class_name: str
    score: float


class PredictResponse(BaseModel):
    """Top-5 (or top-K) list response — used by /predict and /predict/classify."""

    predictions: list[Prediction]
    latency_ms: float


class Top1Response(BaseModel):
    """Single-class response — used by /predict/top1."""

    class_name: str
    score: float
    latency_ms: float


class ScoreResponse(BaseModel):
    """Score-only response — used by /predict/score."""

    score: float
    latency_ms: float


# Operation values used as the OTel `operation` attribute on ml.inference.*
# metrics. Kept as a tuple so the startup hook can seed a zero-point counter
# per operation (gives Prometheus a series even before traffic flows).
_OPERATIONS = ("legacy", "classify", "top1", "score")


def create_app(*, metric_readers=None) -> FastAPI:
    """Build a fresh FastAPI app. Production calls this once at import time
    (`app = create_app()` below). Tests call it per-test with an
    InMemoryMetricReader to capture metrics without touching OTLP."""
    app = FastAPI(title="ResNet-50 Image Classification")

    inference_latency, request_counter, error_counter, tracer = setup_telemetry(
        app, metric_readers=metric_readers
    )

    def _record(operation: str, latency: float, error: Optional[Exception] = None) -> None:
        labels = {"model": "resnet50", "operation": operation}
        request_counter.add(1, labels)
        if error is None:
            inference_latency.record(latency, labels)
        else:
            error_counter.add(1, {**labels, "error_type": type(error).__name__})

    async def _load_image(request: PredictRequest, span) -> Image.Image:
        if request.image_url:
            span.set_attribute("input.type", "url")
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(request.image_url)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            span.set_attribute("input.type", "base64")
            assert request.image_base64 is not None  # validated by caller
            image_data = base64.b64decode(request.image_base64)
            return Image.open(io.BytesIO(image_data)).convert("RGB")

    async def _run_inference(request: PredictRequest, operation: str, span) -> tuple[list, float]:
        if not request.image_url and not request.image_base64:
            raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

        start = time.time()
        try:
            image = await _load_image(request, span)
            results = await asyncio.to_thread(model.predict, image)
        except httpx.HTTPError as e:
            latency = time.time() - start
            err = HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
            # Use the original exception class for `error_type` so dashboards
            # can distinguish ImageFetchError from ValueError, etc.
            _record(operation, latency, error=e)
            raise err
        except HTTPException:
            raise
        except Exception as e:
            latency = time.time() - start
            _record(operation, latency, error=e)
            raise HTTPException(status_code=500, detail=str(e))

        latency = time.time() - start
        _record(operation, latency)
        return results, latency

    @app.on_event("startup")
    def startup() -> None:
        load_time = model.load_model()
        set_model_load_time(load_time)
        for op in _OPERATIONS:
            error_counter.add(0, {"model": "resnet50", "operation": op})

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy", "model_loaded": model.is_loaded()}

    @app.post("/predict", response_model=PredictResponse)
    async def predict_legacy(request: PredictRequest) -> PredictResponse:
        """Alias of /predict/classify kept for back-compat. Tagged
        `operation=legacy` so dashboards can spot stragglers and migrate."""
        with tracer.start_as_current_span("resnet50.inference.legacy") as span:
            results, latency = await _run_inference(request, "legacy", span)
        return PredictResponse(
            predictions=[Prediction(class_name=r["class"], score=r["score"]) for r in results],
            latency_ms=round(latency * 1000, 2),
        )

    @app.post("/predict/classify", response_model=PredictResponse)
    async def predict_classify(request: PredictRequest) -> PredictResponse:
        """Top-K classification list — the canonical resnet50 operation."""
        with tracer.start_as_current_span("resnet50.inference.classify") as span:
            results, latency = await _run_inference(request, "classify", span)
        return PredictResponse(
            predictions=[Prediction(class_name=r["class"], score=r["score"]) for r in results],
            latency_ms=round(latency * 1000, 2),
        )

    @app.post("/predict/top1", response_model=Top1Response)
    async def predict_top1(request: PredictRequest) -> Top1Response:
        """Single best-class response. Smaller payload than /predict/classify
        when callers only need the winning class."""
        with tracer.start_as_current_span("resnet50.inference.top1") as span:
            results, latency = await _run_inference(request, "top1", span)
        top = results[0]
        return Top1Response(
            class_name=top["class"],
            score=top["score"],
            latency_ms=round(latency * 1000, 2),
        )

    @app.post("/predict/score", response_model=ScoreResponse)
    async def predict_score(request: PredictRequest) -> ScoreResponse:
        """Numeric-only response. The smallest possible payload — useful as a
        cheap confidence probe."""
        with tracer.start_as_current_span("resnet50.inference.score") as span:
            results, latency = await _run_inference(request, "score", span)
        return ScoreResponse(
            score=results[0]["score"],
            latency_ms=round(latency * 1000, 2),
        )

    return app


app = create_app()
