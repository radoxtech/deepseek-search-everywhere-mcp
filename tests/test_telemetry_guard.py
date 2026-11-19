"""The server refuses to start when an OpenTelemetry provider would record spans."""

from __future__ import annotations

import pytest

from deepseek_search_everywhere_mcp import server


def test_startup_passes_with_no_provider_installed() -> None:
    server._assert_tracing_off()


def test_startup_fails_when_a_provider_records_spans(monkeypatch) -> None:
    class _RecordingSpan:
        def is_recording(self) -> bool:
            return True

    class _RecordingTracer:
        def start_as_current_span(self, _name: str):
            class _Ctx:
                def __enter__(self):
                    return _RecordingSpan()

                def __exit__(self, *_exc: object) -> None:
                    return None

            return _Ctx()

    class _Provider:
        def get_tracer(self, _name: str) -> _RecordingTracer:
            return _RecordingTracer()

    import opentelemetry.trace as otel

    monkeypatch.setattr(otel, "get_tracer_provider", lambda: _Provider())

    with pytest.raises(RuntimeError, match="without telemetry"):
        server._assert_tracing_off()
