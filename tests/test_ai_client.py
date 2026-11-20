"""Offline tests for ai_client.py and structured.py."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from deepseek_search_everywhere_mcp import ai_client, structured


class SampleSchema(BaseModel):
    name: str
    count: int


async def test_ai_client_singleton_and_limits() -> None:
    client1 = ai_client.get_shared_http_client()
    client2 = ai_client.get_shared_http_client()
    assert client1 is client2
    assert not client1.is_closed

    ai_client.reset_state()
    # Next call should create a fresh client
    client3 = ai_client.get_shared_http_client()
    assert client3 is not client1
    ai_client.reset_state()


def test_ai_client_tracing_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    ai_client._assert_tracing_off()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    with pytest.raises(RuntimeError, match="Telemetry / tracing"):
        ai_client._assert_tracing_off()


async def test_get_agent_provider_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
    provider = ai_client.get_agent_provider()
    assert provider.name == "deepseek"
    assert provider.client.max_retries == 0
    ai_client.reset_state()


class MockChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Msg", (), {"content": content})()


class MockResponse:
    def __init__(self, content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.choices = [MockChoice(content)]
        self.usage = type(
            "Usage",
            (),
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": 15},
        )()


async def test_generate_structured_success() -> None:
    class MockCompletions:
        async def create(self, **kwargs: Any) -> MockResponse:
            assert kwargs.get("extra_body") == {"response_format": {"type": "json_object"}}
            return MockResponse('{"name": "items", "count": 42}')

    mock_sdk = type("MockSDK", (), {"chat": type("Chat", (), {"completions": MockCompletions()})()})()

    res, usage = await structured.generate_structured(
        mock_sdk,  # type: ignore[arg-type]
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hello"}],
        schema=SampleSchema,
    )
    assert res.name == "items"
    assert res.count == 42
    assert usage["total_tokens"] == 15


async def test_generate_structured_retry_on_invalid_json() -> None:
    calls = 0

    class MockCompletions:
        async def create(self, **kwargs: Any) -> MockResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return MockResponse('{"invalid": "json"}')
            return MockResponse('{"name": "recovered", "count": 99}')

    mock_sdk = type("MockSDK", (), {"chat": type("Chat", (), {"completions": MockCompletions()})()})()

    res, usage = await structured.generate_structured(
        mock_sdk,  # type: ignore[arg-type]
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hello"}],
        schema=SampleSchema,
        attempts=2,
    )
    assert calls == 2
    assert res.name == "recovered"
    assert res.count == 99


async def test_generate_structured_fails_after_retries() -> None:
    class MockCompletions:
        async def create(self, **kwargs: Any) -> MockResponse:
            return MockResponse('{"completely": "wrong"}')

    mock_sdk = type("MockSDK", (), {"chat": type("Chat", (), {"completions": MockCompletions()})()})()

    with pytest.raises(structured.StructuredOutputError, match="Failed to produce structured output"):
        await structured.generate_structured(
            mock_sdk,  # type: ignore[arg-type]
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            schema=SampleSchema,
            attempts=2,
        )
