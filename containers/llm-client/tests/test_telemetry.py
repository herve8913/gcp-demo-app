"""
TDD spec for Phase 1b — llm-client telemetry layer.

Expected behavior (derived from the operation-breakdown design):
The OpenAI SDK auto-instrumentation, once `setup_telemetry()` runs,
must produce `gen_ai.client.operation.duration` on every chat /
completions call, with at minimum these attributes:

  - gen_ai.operation.name      ('chat' for chat.completions.create,
                                'text_completion' for completions.create)
  - gen_ai.request.model       (whatever model= was passed)

These are exactly the attributes the Chamber dashboard's
operation-breakdown UI will key on. If they're missing or wrong, the
breakdown won't have anything to group by.

We mock the underlying httpx transport via respx so tests don't need
a real LLM endpoint.
"""
from __future__ import annotations

import os
import sys
from typing import Iterator

import httpx
import pytest
import respx
from openai import OpenAI
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import telemetry  # noqa: E402


@pytest.fixture
def isolated_telemetry() -> Iterator[tuple[InMemoryMetricReader, OpenAI]]:
    """Each test gets (reader, openai_client). The client is wired through
    the openai-v2 instrumentor so calling chat.completions.create or
    completions.create against the respx-mocked transport produces real
    gen_ai.* metrics into the reader."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    telemetry.setup_telemetry(meter_provider_override=provider)

    client = OpenAI(base_url="http://test/v1", api_key="EMPTY")
    yield reader, client

    # Tear down the instrumentation so the next test gets a clean slate.
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    if OpenAIInstrumentor().is_instrumented_by_opentelemetry:
        OpenAIInstrumentor().uninstrument()


def _operation_durations(reader: InMemoryMetricReader) -> list[dict]:
    """Return [{attrs...}, ...] for every gen_ai.client.operation.duration
    data point currently held by the reader."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    out: list[dict] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name != "gen_ai.client.operation.duration":
                    continue
                for dp in m.data.data_points:
                    out.append(dict(dp.attributes))
    return out


# ---------------------------------------------------------------------------
# setup_telemetry smoke check
# ---------------------------------------------------------------------------

def test_setup_telemetry_can_be_called_with_a_local_meter_provider():
    """Production calls setup_telemetry() with no args (build OTLP exporters
    from env vars). Tests pass meter_provider_override so they can capture
    metrics without needing a collector. The call must not raise."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    telemetry.setup_telemetry(meter_provider_override=provider)

    # OpenAIInstrumentor.instrument() is a one-shot until uninstrument is
    # called — leave a clean slate for subsequent tests in this file.
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    if OpenAIInstrumentor().is_instrumented_by_opentelemetry:
        OpenAIInstrumentor().uninstrument()


# ---------------------------------------------------------------------------
# gen_ai.client.operation.duration — the metric the dashboard breaks down on
# ---------------------------------------------------------------------------

@respx.mock(base_url="http://test/v1")
def test_chat_completion_call_emits_operation_duration_with_chat_op_name(
    respx_mock, isolated_telemetry
):
    reader, client = isolated_telemetry
    respx_mock.post("/chat/completions").mock(return_value=httpx.Response(200, json={
        "id": "chatcmpl-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }],
        "model": "qwen-test",
        "created": 1,
        "object": "chat.completion",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }))

    client.chat.completions.create(
        model="qwen-test",
        messages=[{"role": "user", "content": "Hi"}],
    )

    durations = _operation_durations(reader)
    assert any(
        d.get("gen_ai.operation.name") == "chat"
        and d.get("gen_ai.request.model") == "qwen-test"
        for d in durations
    ), (
        f"Expected gen_ai.client.operation.duration with operation.name=chat "
        f"and request.model=qwen-test. Got: {durations}"
    )


@respx.mock(base_url="http://test/v1")
def test_legacy_completions_call_does_NOT_emit_gen_ai_metric(
    respx_mock, isolated_telemetry
):
    """opentelemetry-instrumentation-openai-v2 (verified at version 2.4b0)
    only wraps `openai.resources.chat.completions.Completions.create` —
    it does NOT instrument the legacy /completions endpoint
    (`client.completions.create`). This test pins that behaviour so
    a future contributor doesn't add `client.completions.create` to
    main.py expecting metrics that won't fire.

    If a future version of the instrumentor adds /completions support,
    this test will start failing — at which point we can re-enable a
    text_completion code path in main.py with confidence."""
    reader, client = isolated_telemetry
    respx_mock.post("/completions").mock(return_value=httpx.Response(200, json={
        "id": "cmpl-test",
        "choices": [{"index": 0, "text": "Hi", "finish_reason": "stop", "logprobs": None}],
        "model": "qwen-test",
        "created": 1, "object": "text_completion",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }))

    client.completions.create(model="qwen-test", prompt="Hi")

    durations = _operation_durations(reader)
    assert not any(d.get("gen_ai.operation.name") == "text_completion" for d in durations), (
        f"Unexpected text_completion metric — has openai-v2 added /completions "
        f"instrumentation? Update main.py to use it. Saw: {durations}"
    )


@respx.mock(base_url="http://test/v1")
def test_repeated_chat_calls_with_different_models_appear_as_distinct_series(
    respx_mock, isolated_telemetry
):
    """The Chamber dashboard's operation-breakdown for an LLM service falls
    back to grouping by `gen_ai.request.model` when only one operation
    name is in play. Mixing model values must yield distinct series so
    the breakdown shows multiple rows."""
    reader, client = isolated_telemetry
    respx_mock.post("/chat/completions").mock(return_value=httpx.Response(200, json={
        "id": "c1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
        "model": "stub",
        "created": 1, "object": "chat.completion",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }))

    client.chat.completions.create(model="model-a", messages=[{"role": "user", "content": "x"}])
    client.chat.completions.create(model="model-b", messages=[{"role": "user", "content": "x"}])

    request_models = {d.get("gen_ai.request.model") for d in _operation_durations(reader)}
    assert request_models >= {"model-a", "model-b"}, (
        f"Expected two distinct gen_ai.request.model values. Got: {request_models}"
    )
