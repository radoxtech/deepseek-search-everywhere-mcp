"""MCP-layer structured-content contract tests for every tool.

These tests drive the real server through an in-process ``ClientSession`` over the in-memory
transport, so every assertion runs on what a model client actually receives: the ``tools/list``
entry (``title`` + ``outputSchema``), the unstructured ``content`` text block and the
``structuredContent`` dict. No network is touched; ``fake_upstream`` patches ``runtime._request``
exactly as the per-tool suites do.

Per response the contract pins four things:

1. ``structuredContent`` validates against the tool's published ``outputSchema``.
2. The text block's JSON keys are a subset of ``structuredContent`` and shared values are equal.
   The text block is built from the raw tool dict while ``structuredContent`` is the model dump
   (``model_dump(mode="json", by_alias=True)``, no ``exclude_none``), so a field missing from the
   output model is silently dropped only from the structured half; this assertion catches it.
3. ``isError`` is false for every logical ``{"ok": false}`` envelope: tools report failures in the
   envelope, not as protocol errors.
4. ``elapsed_ms`` in the *text block* matches today's behaviour: every ``@_timed`` tool emits it,
   ``deepseek_check_research`` emits it only on its running and polled-done paths (never on the
   early-error, needs-input or verified-report paths).

The structured payload is also asserted to be the model dump, not the raw dict: the inherited
``Envelope.meta`` field is published as ``_meta``, so ``_meta`` must be present and ``meta`` absent.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from jsonschema import Draft202012Validator
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession
from test_agent_research import ReportPlanUpstream
from test_research import DeepResearchUpstream

from deepseek_search_everywhere_mcp import runtime
from deepseek_search_everywhere_mcp.errors import _fail
from deepseek_search_everywhere_mcp.server import FETCH_PATH, SEARCH_PATH, create_server
from deepseek_search_everywhere_mcp.settings import set_settings

TOOL_NAMES = {
    "deepseek_search",
    "deepseek_deep_agent_research",
    "deepseek_fetch",
    "deepseek_quick_research",
    "deepseek_check_research",
    "deepseek_extract",
    "deepseek_check_balance",
}


def _inline_wait(seconds: float) -> None:
    """Set the research inline wait for the current test (the plan/test pattern)."""
    set_settings(replace(runtime.get_settings(), research_inline_wait=seconds))


def _assert_no_key_loss(raw: Any, structured: Any, path: str) -> None:
    """Assert ``raw`` (the text block) is a recursive key-subset of ``structured`` (the model dump)."""
    if isinstance(raw, dict):
        assert isinstance(structured, dict), f"{path}: text is an object, structured is {type(structured)}"
        missing = set(raw) - set(structured)
        assert not missing, f"{path} lost keys from structuredContent: {sorted(missing)}"
        for key, value in raw.items():
            _assert_no_key_loss(value, structured[key], f"{path}.{key}")
    elif isinstance(raw, list):
        assert isinstance(structured, list) and len(raw) == len(structured), f"{path}: list shape changed"
        for index, (left, right) in enumerate(zip(raw, structured, strict=True)):
            _assert_no_key_loss(left, right, f"{path}[{index}]")
    else:
        assert raw == structured, f"{path}: text {raw!r} != structured {structured!r}"


class _Client:
    """A connected client plus the ``tools/list`` schemas keyed by tool name."""

    def __init__(self, session: ClientSession, tools: dict[str, Any]) -> None:
        self._session = session
        self.tools = tools

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        ok: bool,
        expect_elapsed: bool,
    ) -> dict[str, Any]:
        """Call ``name`` over MCP, assert the full contract and return ``structuredContent``."""
        result = await self._session.call_tool(name, arguments or {})
        schema = self.tools[name].output_schema
        assert schema is not None, f"{name} publishes no outputSchema"
        assert isinstance(schema.get("properties"), dict) and schema["properties"]

        structured = result.structured_content
        assert isinstance(structured, dict), f"{name} returned no structuredContent"

        # 1. The published schema accepts the structured payload.
        Draft202012Validator(schema).validate(structured)

        # 2. No key is silently lost from the model dump, and shared values agree. The comparison
        # is recursive: nested dicts are padded with nulls in the model dump, so a dict must be a
        # key-subset recursively rather than strictly equal.
        assert result.content, f"{name} returned an empty content list"
        raw = json.loads(result.content[0].text)
        assert isinstance(raw, dict)
        _assert_no_key_loss(raw, structured, name)

        # 3. Logical `{ok: false}` envelopes are not protocol errors.
        assert result.is_error is False, f"{name} raised a protocol-level error"

        # 4. The logical outcome survives the model round-trip.
        assert structured["ok"] is ok, f"{name}: structured ok={structured['ok']!r}, expected {ok!r}"

        # 5. The dump uses the model's alias, proving it is the model dump and not the raw dict.
        assert "_meta" in structured, f"{name} structuredContent has no `_meta` alias"
        assert "meta" not in structured, f"{name} structuredContent kept the Python field name `meta`"

        # 6. `elapsed_ms` presence in the text block mirrors what the tool actually emitted.
        assert ("elapsed_ms" in raw) is expect_elapsed, (
            f"{name}: text elapsed_ms present={('elapsed_ms' in raw)}, expected {expect_elapsed}"
        )
        if expect_elapsed:
            assert isinstance(structured["elapsed_ms"], int) or isinstance(structured["elapsed_ms"], float)
        else:
            assert structured["elapsed_ms"] is None, f"{name}: unexpected elapsed_ms value"

        return structured


@asynccontextmanager
async def _connected() -> AsyncIterator[_Client]:
    """Start a fresh server and talk to it through a real in-memory MCP client session."""
    server = create_server()
    async with InMemoryTransport(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listing = await session.list_tools()
            yield _Client(session, {tool.name: tool for tool in listing.tools})


def _search_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "data": {"results": results}}


class _SearchUpstream:
    async def __call__(self, method: str, path: str, **_: Any) -> dict[str, Any]:
        assert path == SEARCH_PATH, path
        return _search_results([{"title": "A", "url": "https://a.test", "snippet": "s"}])


class _FetchUpstream:
    """A page for most URLs and a 404 for any URL containing ``missing``."""

    async def __call__(
        self, method: str, path: str, *, payload: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, Any]:
        if path != FETCH_PATH:
            return _fail("not_found", f"unexpected path {path}")
        url = (payload or {}).get("url") or ""
        if "missing" in url:
            return _fail("not_found", "page not found", 404)
        return {"ok": True, "data": {"title": "Test Page", "content": "Readable page content. " * 40}}


class _ExtractUpstream:
    """Answers both the fetch and the chat-completions path used by ``deepseek_extract``."""

    def __init__(self, page_content: str = "Our Pro Widget costs $49.99.", chat_response: str = "") -> None:
        self.page_content = page_content
        self.chat_response = chat_response

    async def __call__(
        self, method: str, path: str, *, payload: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, Any]:
        data = payload or {}
        if path == FETCH_PATH:
            url = data.get("url") or ""
            content = "" if "empty" in url else self.page_content
            return {"ok": True, "data": {"url": url, "title": "Test Page", "content": content}}
        if path == "/v1/chat/completions":
            return {
                "ok": True,
                "data": {"choices": [{"message": {"role": "assistant", "content": self.chat_response}}]},
            }
        return _fail("not_found", f"unexpected path {path}")


class _SlowResearchUpstream(DeepResearchUpstream):
    """A research upstream whose chat calls outlast a tiny inline wait, keeping the job running."""

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/v1/chat/completions":
            await asyncio.sleep(0.15)
        return await super().__call__(method, path, **kwargs)


class _SlowAgentUpstream(ReportPlanUpstream):
    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/v1/chat/completions":
            await asyncio.sleep(0.1)
        return await super().__call__(method, path, **kwargs)


class _FailedSynthesisUpstream(ReportPlanUpstream):
    """The plan call succeeds, every later chat call fails, so no report can be written."""

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        from deepseek_search_everywhere_mcp.deep_research import DECOMPOSE_SYSTEM_PROMPT

        if path == "/v1/chat/completions":
            payload = kwargs.get("payload") or {}
            messages = payload.get("messages", [])
            system = messages[0].get("content", "") if messages else ""
            if system == DECOMPOSE_SYSTEM_PROMPT:
                return await super().__call__(method, path, **kwargs)
            return _fail("upstream_error", "model unavailable")
        return await super().__call__(method, path, **kwargs)


async def test_server_publishes_the_contract_for_all_seven_tools(fake_upstream: Any) -> None:
    fake_upstream(default=_search_results([]))
    async with _connected() as client:
        assert set(client.tools) == TOOL_NAMES
        assert len(client.tools) == 7
        for name, tool in client.tools.items():
            assert tool.title and tool.title.strip(), f"{name} has no non-empty title"
            schema = tool.output_schema
            assert schema is not None, f"{name} has no outputSchema"
            assert isinstance(schema.get("properties"), dict) and schema["properties"], name
            assert schema.get("additionalProperties") is not True, name
            assert schema.get("required") == ["ok"], name


async def test_search_single_list_and_error(fake_upstream: Any) -> None:
    fake_upstream(_SearchUpstream())
    async with _connected() as client:
        single = await client.call("deepseek_search", {"query": "python"}, ok=True, expect_elapsed=True)
        assert single["query_used"] == "python" and single["count"] == 1

        listed = await client.call(
            "deepseek_search", {"query": ["q1", "q2", "q3", "q4", "q5", "q6"]}, ok=True, expect_elapsed=True
        )
        assert listed["skipped_queries"] == ["q6"]
        assert listed["count"] == 5

        error = await client.call(
            "deepseek_search", {"query": "python", "count": 0}, ok=False, expect_elapsed=True
        )
        assert error["error"]["code"] == "invalid_input"


async def test_fetch_single_list_partial_and_error(fake_upstream: Any) -> None:
    fake_upstream(_FetchUpstream())
    async with _connected() as client:
        single = await client.call(
            "deepseek_fetch", {"url": "https://example.com/a"}, ok=True, expect_elapsed=True
        )
        assert single["url"] == "https://example.com/a" and single["content"]

        listed = await client.call(
            "deepseek_fetch",
            {"url": ["https://example.com/a", "https://example.com/missing"]},
            ok=True,
            expect_elapsed=True,
        )
        assert listed["partial"] is True and listed["failed_count"] == 1
        assert listed["results"][1]["ok"] is False

        error = await client.call("deepseek_fetch", {"url": "not-a-url"}, ok=False, expect_elapsed=True)
        assert error["error"]["code"] == "invalid_input"


async def test_extract_single_list_partial_top_level_and_bare_error(fake_upstream: Any) -> None:
    fake_upstream(_ExtractUpstream(chat_response='```json\n{"product": "Pro Widget", "price": 49.99}\n```'))
    async with _connected() as client:
        single = await client.call(
            "deepseek_extract",
            {"url": "https://example.com/product", "fields": ["product", "price"]},
            ok=True,
            expect_elapsed=True,
        )
        assert single["data"] == {"product": "Pro Widget", "price": 49.99}

        listed = await client.call(
            "deepseek_extract",
            {"url": ["https://example.com/valid", "https://example.com/empty"], "fields": ["name"]},
            ok=True,
            expect_elapsed=True,
        )
        assert listed["partial"] is True and listed["failed_count"] == 1
        assert listed["results"][1]["ok"] is False

        top_level = await client.call(
            "deepseek_extract", {"url": "https://example.com/product"}, ok=False, expect_elapsed=True
        )
        assert top_level["error"]["reason"] == "bad_request"

    fake_upstream(_ExtractUpstream(page_content="Hello world", chat_response="Not json"))
    async with _connected() as client:
        bare = await client.call(
            "deepseek_extract",
            {"url": "https://example.com/x", "fields": ["item"]},
            ok=False,
            expect_elapsed=True,
        )
        assert bare["error"]["reason"] == "extraction_failed"


async def test_quick_research_done_inline(fake_upstream: Any) -> None:
    fake_upstream(DeepResearchUpstream())
    async with _connected() as client:
        done = await client.call(
            "deepseek_quick_research",
            {"topic": "quantum computing scaling", "depth": "low"},
            ok=True,
            expect_elapsed=True,
        )
        assert done["status"] == "done" and done["delivery"] == "inline"
        assert done["synthesized"] is True and done["report"]
        assert done["sources"]


async def test_quick_research_running_then_collected(fake_upstream: Any) -> None:
    fake_upstream(_SlowResearchUpstream())
    _inline_wait(0.01)
    async with _connected() as client:
        running = await client.call(
            "deepseek_quick_research",
            {"topic": "slow quantum topic", "depth": "low"},
            ok=True,
            expect_elapsed=True,
        )
        assert running["status"] == "running" and running["job_id"]
        assert running["progress"] and running["next_action"]

        _inline_wait(5.0)
        done = await client.call(
            "deepseek_check_research", {"job_id": running["job_id"]}, ok=True, expect_elapsed=True
        )
        assert done["status"] == "done" and done["delivery"] == "polled"


async def test_quick_research_needs_input(fake_upstream: Any) -> None:
    fake_upstream(DeepResearchUpstream())
    async with _connected() as client:
        waiting = await client.call(
            "deepseek_quick_research",
            {"topic": "client synthesis topic", "depth": "low", "synthesis": "client"},
            ok=True,
            expect_elapsed=True,
        )
        assert waiting["status"] == "needs_input" and waiting["step"] == "report"
        assert waiting["context_page"] == 1 and waiting["context"]


async def test_check_research_running_then_done_polled(fake_upstream: Any) -> None:
    fake_upstream(_SlowResearchUpstream())
    _inline_wait(0.01)
    async with _connected() as client:
        running = await client.call(
            "deepseek_quick_research",
            {"topic": "polled slow topic", "depth": "low"},
            ok=True,
            expect_elapsed=True,
        )
        assert running["status"] == "running"

        still_running = await client.call(
            "deepseek_check_research", {"job_id": running["job_id"]}, ok=True, expect_elapsed=True
        )
        assert still_running["status"] == "running"

        _inline_wait(5.0)
        polled = await client.call(
            "deepseek_check_research", {"job_id": running["job_id"]}, ok=True, expect_elapsed=True
        )
        assert polled["status"] == "done" and polled["delivery"] == "polled"


async def test_check_research_job_not_found(fake_upstream: Any) -> None:
    fake_upstream(default=_search_results([]))
    async with _connected() as client:
        missing = await client.call(
            "deepseek_check_research", {"job_id": str(uuid.uuid4())}, ok=False, expect_elapsed=False
        )
        assert missing["error"]["code"] == "job_not_found"


async def test_check_research_needs_input_then_verified_report(fake_upstream: Any) -> None:
    fake_upstream(DeepResearchUpstream())
    async with _connected() as client:
        waiting = await client.call(
            "deepseek_quick_research",
            {"topic": "verified client topic", "depth": "low", "synthesis": "client"},
            ok=True,
            expect_elapsed=True,
        )
        job_id = waiting["job_id"]

        # Reading the sources of a finished client job does not go through a @_timed wrapper.
        page = await client.call("deepseek_check_research", {"job_id": job_id}, ok=True, expect_elapsed=False)
        assert page["status"] == "needs_input" and page["step"] == "report"

        verified = await client.call(
            "deepseek_check_research",
            {
                "job_id": job_id,
                "answer": ("Surface codes provide fault tolerance [1]. Gate fidelities exceed 99.9% [2]."),
            },
            ok=True,
            expect_elapsed=False,
        )
        assert verified["status"] == "done" and verified["delivery"] == "client"
        assert verified["job_id"] is None


async def test_agent_done_synthesized(fake_upstream: Any) -> None:
    fake_upstream(ReportPlanUpstream())
    async with _connected() as client:
        done = await client.call(
            "deepseek_deep_agent_research",
            {"topic": "quantum error correction", "depth": "medium"},
            ok=True,
            expect_elapsed=True,
        )
        assert done["status"] == "done" and done["delivery"] == "inline"
        assert done["synthesized"] is True
        assert done["plan"] and done["rounds"] and done["stopped_reason"]


async def test_agent_done_unsynthesized(fake_upstream: Any) -> None:
    fake_upstream(_FailedSynthesisUpstream())
    async with _connected() as client:
        done = await client.call(
            "deepseek_deep_agent_research",
            {"topic": "quantum error correction without synthesis", "depth": "medium"},
            ok=True,
            expect_elapsed=True,
        )
        assert done["status"] == "done" and done["synthesized"] is False
        assert done["partial"] is True and done["report"] is None


async def test_agent_running_then_collected(fake_upstream: Any) -> None:
    fake_upstream(_SlowAgentUpstream())
    _inline_wait(0.01)
    async with _connected() as client:
        running = await client.call(
            "deepseek_deep_agent_research",
            {"topic": "slow quantum agent topic", "depth": "medium"},
            ok=True,
            expect_elapsed=True,
        )
        assert running["status"] == "running" and running["job_id"]

        # A second identical call joins the running job; a large inline wait collects it.
        _inline_wait(5.0)
        done = await client.call(
            "deepseek_deep_agent_research",
            {"topic": "slow quantum agent topic", "depth": "medium"},
            ok=True,
            expect_elapsed=True,
        )
        assert done["status"] == "done"


async def test_agent_budget_guard_error(fake_upstream: Any, monkeypatch: Any) -> None:
    import deepseek_search_everywhere_mcp.tools.agent as agent_tools

    fake_upstream(ReportPlanUpstream())
    real_job = agent_tools.ResearchJob

    def _budgeted_job(**kwargs: Any) -> Any:
        # The public tool always builds a job with max_cost_usd=None; pin a tiny budget so the
        # pre-run estimator trips the per-job guard and the tool emits its budget-error envelope.
        kwargs["max_cost_usd"] = 0.001
        return real_job(**kwargs)

    monkeypatch.setattr(agent_tools, "ResearchJob", _budgeted_job)
    async with _connected() as client:
        error = await client.call(
            "deepseek_deep_agent_research",
            {"topic": "budget guarded agent", "depth": "medium"},
            ok=False,
            expect_elapsed=True,
        )
        assert error["error"]["code"] == "budget_exceeded"
        assert error["estimate_usd"] > 0


async def test_balance_list_dict_and_error(fake_upstream: Any) -> None:
    async with _connected() as client:
        fake_upstream(
            default={"ok": True, "data": {"is_available": True, "balance_infos": [{"currency": "USD"}]}}
        )
        listed = await client.call("deepseek_check_balance", {}, ok=True, expect_elapsed=True)
        assert isinstance(listed["balances"], list) and listed["is_available"] is True

        fake_upstream(default={"ok": True, "data": {"is_available": False, "balances": {"currency": "USD"}}})
        mapped = await client.call("deepseek_check_balance", {}, ok=True, expect_elapsed=True)
        assert isinstance(mapped["balances"], dict) and mapped["is_available"] is False

        fake_upstream(default=_fail("auth_failed", "denied", 401))
        error = await client.call("deepseek_check_balance", {}, ok=False, expect_elapsed=True)
        assert error["error"]["code"] == "account_error"
