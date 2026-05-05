"""Tests for the llm-client main loop. Verify the loop calls chat.completions
.create with the right args, handles errors, and cycles through the configured
prompts. Independent of telemetry — that's covered in test_telemetry.py."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import main  # noqa: E402


def _fake_chat_response() -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="hi"))]
    return response


def test_prompts_list_is_non_empty():
    """If someone removes all prompts the loop becomes a tight no-op."""
    assert len(main.PROMPTS) >= 1


def test_run_chat_calls_create_with_expected_args():
    """run_chat is the unit of work the loop drives. It must shape the
    openai SDK call correctly (model, messages, max_tokens, stream=False)."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response()

    main.run_chat(fake_client, model="m1", prompt="hello")

    fake_client.chat.completions.create.assert_called_once()
    call = fake_client.chat.completions.create.call_args
    assert call.kwargs["model"] == "m1"
    assert call.kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert call.kwargs["stream"] is False
    assert "max_tokens" in call.kwargs


def test_run_one_cycle_invokes_chat_once_per_prompt():
    """run_one_cycle is the loop body — it should hit chat.completions
    exactly len(prompts) times."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response()
    prompts = ["a", "b", "c"]

    main.run_one_cycle(fake_client, model="m1", prompts=prompts, sleep_seconds=0)

    assert fake_client.chat.completions.create.call_count == len(prompts)


def test_run_one_cycle_logs_and_continues_on_error():
    """An error from one prompt should not stop the cycle. The loop body
    should swallow the exception, log it, and move on."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        Exception("first call boom"),
        _fake_chat_response(),
        _fake_chat_response(),
    ]

    # Must not raise.
    main.run_one_cycle(
        fake_client, model="m1", prompts=["a", "b", "c"], sleep_seconds=0
    )

    assert fake_client.chat.completions.create.call_count == 3
