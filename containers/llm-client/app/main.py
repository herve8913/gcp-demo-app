"""Long-running client that exercises vllm-qwen via the OpenAI Python SDK.

The OpenAI auto-instrumentation set up in telemetry.py emits
gen_ai.client.operation.duration on every chat.completions.create call,
which the chamber-standalone agent picks up via OTLP and forwards to the
Chamber dashboard's gen_ai semconv breakdown.

This container does NOT use client.completions.create — only
chat.completions.create is wrapped by opentelemetry-instrumentation-openai-v2
2.4b0; see tests/test_telemetry.py for the regression that pins this.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

from openai import OpenAI

from telemetry import setup_telemetry

logger = logging.getLogger(__name__)

# Prompts vary in length so the gen_ai.client.token.usage histogram has a
# meaningful distribution. They're intentionally lightweight so vllm-qwen
# doesn't spend its compute budget on this synthetic load.
PROMPTS: list[str] = [
    "Tell me a one-sentence fact about GPUs.",
    "What is reinforcement learning?",
    "Explain attention briefly.",
    "What is a tensor core?",
    "Why does softmax need a temperature parameter?",
    "Name two ML inference servers.",
]


def make_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://vllm:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),  # vllm doesn't auth
    )


def run_chat(client: OpenAI, *, model: str, prompt: str) -> str:
    """Single chat.completions request. Returns the assistant content for
    convenience; the auto-instrumentation handles the metric emission."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        stream=False,
    )
    return response.choices[0].message.content


def run_one_cycle(
    client: OpenAI,
    *,
    model: str,
    prompts: Iterable[str],
    sleep_seconds: float,
) -> None:
    """Walk through prompts once, sleeping between each call. A single
    failed prompt is logged and skipped — we don't let one bad request
    take down the demo's continuous traffic."""
    for prompt in prompts:
        try:
            run_chat(client, model=model, prompt=prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("chat call failed for prompt %r: %s", prompt[:40], e)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    setup_telemetry()

    model = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    sleep_seconds = float(os.environ.get("LLM_CLIENT_SLEEP", "5"))
    client = make_client()

    logger.info("llm-client up; model=%s sleep=%ss", model, sleep_seconds)
    while True:
        run_one_cycle(client, model=model, prompts=PROMPTS, sleep_seconds=sleep_seconds)


if __name__ == "__main__":
    main()
