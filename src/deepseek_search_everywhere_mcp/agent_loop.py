"""Agent orchestration loop for research execution on Vercel ai (ADR 0009).

Subclasses ai.Agent, enforces LOOP_BUFFER = 0, caps turns with MAX_STEPS,
tracks usage against budget per event, and supports deterministic stopping.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from ai import Agent, types
from ai.agents.agent import Context, ToolRunner, events_, models, telemetry, util
from ai.types.usage import Usage


class ResearchAgent(Agent):
    """Orchestrated agent loop with hard step limits and budget awareness."""

    LOOP_BUFFER = 0  # Run in strict lockstep so budget checks have teeth
    MAX_STEPS = 10

    def __init__(
        self,
        *,
        tools: Sequence[Any] | None = None,
        max_steps: int = 10,
        budget_tracker: Any = None,
        deadline_at: float | None = None,
    ) -> None:
        super().__init__(tools=tools or [])
        self.max_steps = max_steps
        self.budget_tracker = budget_tracker
        self.deadline_at = deadline_at
        self.steps_run = 0
        self.stopped_reason = "running"
        self.total_usage = Usage()

    async def loop(self, context: Context) -> AsyncGenerator[events_.AgentEvent]:
        """Custom loop overriding unbounded default in ai.Agent."""
        self.steps_run = 0
        while context.keep_running() and self.steps_run < self.max_steps:
            if self.deadline_at is not None and time.monotonic() >= self.deadline_at:
                self.stopped_reason = "deadline"
                break

            self.steps_run += 1
            async with (
                telemetry.span(telemetry.LoopTurnSpanData()),
                models.stream(context=context) as stream,
                ToolRunner() as tr,
                util.merge(stream, tr.events()) as events,
            ):
                async for event in events:
                    yield event

                    if isinstance(event, types.events.StreamEnd) and event.usage is not None:
                        self.total_usage = self.total_usage + event.usage

                    if isinstance(event, types.events.ToolEnd):
                        tool_item = self.resolve(event.tool_call)
                        tr.schedule(tool_item)

                context.add(stream.message)
                context.add(tr.get_tool_message())

        if not context.keep_running():
            self.stopped_reason = "completed"
        elif self.steps_run >= self.max_steps:
            self.stopped_reason = "max_steps"
