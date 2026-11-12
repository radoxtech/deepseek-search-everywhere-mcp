"""Smoke tests verifying that live_llm and live_e2e markers are registered and behave as expected.

These tests hit no real external network endpoints; they verify marker selection and the 3x
repeat runner wiring when executed in the live suite (and skip without DEEPSEEK_API_KEY).
"""

from __future__ import annotations

import pytest
from repeat_runner import repeat_llm


@pytest.mark.live_llm
@repeat_llm(repeats=3)
async def test_live_llm_marker_smoke() -> None:
    """Smoke test ensuring live_llm marker and repeat runner succeed across all 3 iterations."""
    assert True


@pytest.mark.live_e2e
async def test_live_e2e_marker_smoke() -> None:
    """Smoke test ensuring live_e2e marker is recognized."""
    assert True
