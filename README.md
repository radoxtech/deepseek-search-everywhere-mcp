# DeepSeek Search Everywhere MCP

[![CI](https://github.com/radoxtech/deepseek-search-everywhere-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/radoxtech/deepseek-search-everywhere-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/deepseek-search-everywhere-mcp)](https://pypi.org/project/deepseek-search-everywhere-mcp/)
[![Python versions](https://img.shields.io/pypi/pyversions/deepseek-search-everywhere-mcp)](https://pypi.org/project/deepseek-search-everywhere-mcp/)
[![Glama score](https://glama.ai/mcp/servers/radoxtech/deepseek-search-everywhere-mcp/badges/score.svg)](https://glama.ai/mcp/servers/radoxtech/deepseek-search-everywhere-mcp)

An [MCP](https://modelcontextprotocol.io) server that gives AI agents web search, page fetching and
multi-source research through the official DeepSeek Search & Fetch API.

It is a thin, careful layer over that API: results are deduplicated, content is capped to a budget you set,
repeated calls are served from an in-memory cache, and outgoing calls are paced by a local rate limiter.
Seven tools, stdio by default, optional bearer-protected streamable HTTP.

It is a standard MCP server, so it works with **any MCP-compatible AI client or agent**: Claude Code,
Claude Desktop, Google Antigravity, Cursor, Windsurf, VS Code (GitHub Copilot), Gemini CLI, Codex CLI, Zed,
opencode and others. Local clients launch it over stdio; remote or shared setups can connect over HTTP.

<!-- mcp-name: io.github.radoxtech/deepseek-search-everywhere-mcp -->

> Not affiliated with DeepSeek. This is an independent, community-maintained project.

## Features

- **Search with content in one request**: `need_content` returns page text inline, so there is usually no
  need for a second fetch per result.
- **Batch fetch and research**: `deepseek_fetch_many` fetches up to 10 URLs concurrently;
  `deepseek_research` gathers sources in a single pass: one search, then direct fetches for pages whose search
  content is missing. It does not iterate or summarise.
- **Deep research and structured extraction**: `deepseek_deep_research` decomposes topics into subqueries and
  synthesizes cited briefing reports; `deepseek_extract` extracts validated structured JSON matching schemas or fields.
- **Result hygiene**: canonical-URL and title/site deduplication, per-result and total character budgets,
  Markdown unescaping.
- **Optional query rewrite** to English through `deepseek-flash` (thinking disabled), plus an explicit `lang`
  operator for non-English queries.
- **Cache and limiter**: TTL + LRU cache with single-flight, per-endpoint token buckets, `refresh` to bypass.
- **Safe by default**: static SSRF checks, secret redaction, no telemetry. See [Security](#security).

## Requirements

- Python 3.11 or newer
- A DeepSeek API key (`DEEPSEEK_API_KEY`) from the [DeepSeek platform](https://platform.deepseek.com)

## Install

The server is published on [PyPI](https://pypi.org/project/deepseek-search-everywhere-mcp/) and listed in the
[official MCP Registry](https://registry.modelcontextprotocol.io/v0/servers?search=deepseek-search-everywhere)
as `io.github.radoxtech/deepseek-search-everywhere-mcp`.

```bash
# Run without installing (uv)
uvx deepseek-search-everywhere-mcp

# Or install with pip
pip install deepseek-search-everywhere-mcp
```

The HTTP transport needs the `http` extra (it pulls in `uvicorn`):

```bash
pip install "deepseek-search-everywhere-mcp[http]"
```

To run the latest unreleased code from `main`, or from a source checkout:

```bash
uvx --from git+https://github.com/radoxtech/deepseek-search-everywhere-mcp deepseek-search-everywhere-mcp

git clone https://github.com/radoxtech/deepseek-search-everywhere-mcp.git
cd deepseek-search-everywhere-mcp
pip install -e ".[http]"
```

Check that the key and connectivity work (prints your balance, then exits):

```bash
export DEEPSEEK_API_KEY=sk-...
deepseek-search-everywhere-mcp --check
```

## Quick start

All examples use stdio. Replace `sk-...` with your key.

The server is client-agnostic: the examples below cover common clients, and any other MCP client
(Antigravity, Cursor, Windsurf, VS Code, Gemini CLI, Codex CLI, Zed, ...) works with the same
command, arguments and environment variable in its own MCP configuration format.

### Claude Code

```bash
claude mcp add deepseek-search-everywhere -e DEEPSEEK_API_KEY=sk-... -- \
  uvx deepseek-search-everywhere-mcp
```

### Claude Desktop

Edit `claude_desktop_config.json` (Settings, Developer, Edit Config):

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

### opencode

In `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "deepseek-search-everywhere": {
      "type": "local",
      "command": ["uvx", "deepseek-search-everywhere-mcp"],
      "environment": { "DEEPSEEK_API_KEY": "{env:DEEPSEEK_API_KEY}" },
      "enabled": true
    }
  }
}
```

### Any stdio client

Most clients (including Antigravity, Cursor and Windsurf) use an `mcpServers` JSON block. Use the `uvx`
form from the Claude Desktop example, or, if the package is already installed, point the client at the
console script:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "deepseek-search-everywhere-mcp",
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

`python -m deepseek_search_everywhere_mcp` is equivalent.

### Streamable HTTP

For clients that connect over HTTP. A bearer token is **always** required; the server refuses to start without one.

```bash
pip install "deepseek-search-everywhere-mcp[http]"
export DEEPSEEK_API_KEY=sk-...
export MCP_BEARER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
deepseek-search-everywhere-mcp --transport streamable-http   # http://127.0.0.1:8000/mcp
```

Options: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--path` (default `/mcp`). The token is read
from the environment only, never from the command line. The server runs stateless and answers with JSON
(no SSE streams). Client entry:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <MCP_BEARER_TOKEN>" }
    }
  }
}
```

Claude Code: `claude mcp add --transport http deepseek-search-everywhere http://127.0.0.1:8000/mcp --header "Authorization: Bearer $MCP_BEARER_TOKEN"`.
The server speaks plain HTTP; when binding beyond loopback, put TLS in front of it (see [SECURITY.md](SECURITY.md)).

### Docker

An official container image is available on GitHub Container Registry (`ghcr.io`):

```bash
docker pull ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest
```

**Stdio mode (Claude Desktop, Cursor, local IDEs):**

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "DEEPSEEK_API_KEY=sk-...",
        "ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest"
      ]
    }
  }
}
```

**Remote HTTP hosting (Docker Compose, VPS, n8n, Dify, cloud agents):**

```bash
docker run -d --name deepseek-mcp -p 8000:8000 \
  -e DEEPSEEK_API_KEY="sk-..." \
  -e MCP_BEARER_TOKEN="your-secret-token" \
  ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest \
  --transport streamable-http --host 0.0.0.0 --port 8000
```

The container runs as an unprivileged non-root user (`UID 10001`) with dependencies isolated from the host.

## Tools

| Tool | Purpose | Parameters (defaults) |
|---|---|---|
| `deepseek_search` | Search the web; returns inline page text (default) or snippets and metadata | `query`, `count=5` (1-50), `ori_question`, `need_content=True`, `lang`, `max_chars_per_result=4000`, `max_total_chars=30000`, `dedupe=True`, `rewrite=False`, `localization=None`, `exclude_terms=None`, `exact_phrases=None`, `after`, `before`, `include_domains`, `exclude_domains`, `refresh=False`, `summary_chars`, `strip_boilerplate=True` |
| `deepseek_fetch` | Fetch one known URL as Markdown (supports lexical snippet extraction, sectional reading and windowed pagination) | `url`, `clean=True`, `max_chars=20000`, `offset=None`, `limit=None`, `find_text=None`, `query=None`, `section=None`, `max_snippets=3`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_fetch_many` | Fetch 1-10 URLs concurrently (up to 5 at a time) | `urls`, `max_chars_per_url=4000`, `max_total_chars=20000`, `clean=True`, `query`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_research` | Single-pass source gathering: search with content and auto-fetch stub pages (< 200 chars). No iteration, no summary | `question`, `max_pages=5` (1-50), `per_page_chars=6000`, `total_chars=30000`, `after`, `before`, `include_domains`, `exclude_domains`, `lang`, `rewrite=False`, `dedupe=True`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_deep_research` | Multi-angle research on a topic; decomposes subqueries, gathers sources, and synthesizes a cited report | `topic`, `max_queries=3` (2-4), `max_pages_per_query=3` (1-5), `lang`, `dedupe=True`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_extract` | Extract structured JSON from a web page conforming to target fields or a schema | `url`, `fields=None`, `schema=None`, `instruction=None`, `clean=True`, `max_chars=40000`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_check_balance` | Account balance and session metrics (upstream queries, cache hits, estimated cost) | none |

Notes on the parameters:

- `query` and `question` are limited to 128 characters and support `site:`, `-site:`, `"phrase"`, `lang:`,
  `after:` and `before:`. `ori_question` (max 128 characters) is the natural-language question and improves ranking.
  `localization` is an optional IP address used by DeepSeek to localize search results.
- `after`, `before`, `include_domains`, `exclude_domains`, `exclude_terms` and `exact_phrases` are explicit
  filter parameters for search tools. Multiple `include_domains` are joined with `OR` (`site:a.com OR site:b.com`),
  `exclude_domains` compiles to `-site:`, `exclude_terms` to `-term`, `exact_phrases` to `"phrase"`, and dates to
  `after:YYYY-MM-DD` / `before:YYYY-MM-DD`. Filters and query are budget-guarded together to stay within DeepSeek's
  128-character limit.
- `offset` (0-based start index) and `limit` in `deepseek_fetch` enable character-level windowed pagination for long pages,
  reporting `total_chars` and `has_more`. `find_text` locates a case-insensitive substring and sets `offset` to its position.
- `query` in `deepseek_fetch` and `deepseek_fetch_many` extracts ranked relevant excerpts using a BM25-lite
  scoring engine without invoking an LLM. `section` (1-based index) enables sectional reading of long documents
  with `has_more` and `total_sections` pagination metadata. Both operate in-memory on cached pages.
- `deepseek_deep_research` decomposes topics into 2-4 subqueries via `deepseek-flash`, searches concurrently,
  merges and dedupes candidate sources, and synthesizes a cited report with numbered bracket citations `[1]`, `[2]`
  capped at 6 sources / 24,000 characters. If the synthesis deadline is approached, it gracefully falls back
  to returning the gathered sources.
- `deepseek_extract` uses `deepseek-flash` to extract structured JSON matching either `fields: list[str]` or
  a `schema: dict` without ingesting the full page text into client context. Missing requested fields are populated
  with `null`, and pages with no readable text fast-fail with zero LLM token consumption.
- `deepseek_check_balance` (and legacy alias `deepseek_balance`) includes a `session` block with uptime, total queries handled, cache hit rate, and estimated
  cost and savings (at $0.00075 per request).
- `lang` is an explicit language operator such as `es`, `de`, `ja` or `pt-BR`. Without it, results for
  non-English queries may be mixed-language; for named entities, compare with and without it.
- With `dedupe=True`, searches ask the upstream for a few extra results (`max(2, ceil(count * 0.5))`, capped at 50)
  and trim to `count` after removing duplicates, so you still get `count` results when the index returns repeats.
  With `dedupe=False` exactly `count` are requested.
- Without `need_content`, each search result carries a short `summary` instead of page text. It is capped at
  `summary_chars` (default 400) and never longer than `max_chars_per_result`; `summary_chars=0` leaves it out.
  With content, a `summary` only stands in for a result that came back without content, and `summary_chars`
  then defaults to `max_chars_per_result`. Summaries count towards `max_total_chars`.
- `rewrite=True` (also in `deepseek_research`) translates the query to English with `deepseek-flash` before searching. It is skipped when `lang`
  is set and falls back silently to the original query if the rewrite fails.
- `clean=True` (fetch tools and search content) removes backslash escapes from every ASCII punctuation character
  (`\(`, `\-`, `\[`, `\/`, `\#`...). Fenced code blocks and inline code spans are left untouched.
- Search results (content, `snippet`, `summary`, and so `deepseek_research` sources) also get JSON string
  escapes decoded when the search index returns a page's text still JSON-escaped (`Kr\u00f3tkie` becomes
  `Krótkie`, `<\/p>` becomes `</p>`), outside code. Simple attribute-less HTML tags left in snippets and
  summaries (`<p>`, `<br>`, `</h2>`...) are replaced with a space.
- `strip_boilerplate=True` (keyword-only, on every tool that returns page content) removes known site-interface
  lines. DeepSeek returns Reddit threads with the site UI localised to Chinese whatever the thread's language:
  the comment-sort menu (`排序方式：`, `最佳`, `新`...), the `# 评论区域` heading, the archived-post banner,
  `[已删除]` author lines and the sign-up footer. Rules are per site (Reddit only for now, matched on
  `reddit.com` and its subdomains) and only drop lines that consist entirely of a known label, so user text,
  Chinese included, and code blocks are kept. One rule applies to every site: a cookie-consent (CMP) dialog
  at the very start of the content ("Informacje o prywatności", "We value your privacy", "Wir verwenden
  Cookies"...) is removed when its first line opens with a known consent phrase and the run of lines shows a
  dialog control ("Accept all", "Zarządzaj preferencjami zgody"...). Removal stops at the first line that is
  not part of the dialog, and text further down is never touched, so an article about privacy is kept. When a
  fetched page is nothing but the banner, `content` is empty and a note says so. It runs after `clean` and
  before the character caps, so budgets count the cleaned text; it does not change caching. `strip_boilerplate=False` returns the content as fetched.
- `deepseek_fetch_many` skips empty entries, fetches duplicate URLs (same canonical URL) once, keeps the input
  order, and reports a failing URL as an item with `ok: false` instead of failing the batch. All fetches share
  a 120 s deadline; URLs still pending then are reported as `deadline_exceeded`. Each page gets at most
  `min(max_chars_per_url, max_total_chars // number of URLs)` characters.
- `deepseek_research` is single-pass source gathering, not an agent: it makes exactly one search plus the gap
  fetches, never refines the query or searches again, and does no summarising. It takes the same `lang`,
  `rewrite` and `dedupe` options as search; the envelope's `query_used`
  shows the query that was sent. It does no summarising and makes no model call unless `rewrite=True`: it returns sources for the caller to synthesise.
  Each source carries `title`, `url`, `date` and `site` (both null when the upstream gave none), `date_source`,
  `authority` and `rerank_score`, so stale or off-topic sources are visible without opening them.
- `deepseek_research` is for reconnaissance: finding candidate sources. There is no quality filter, so results
  can include low-quality or secondary pages (student papers, aggregators, clickbait). Verify facts in the
  primary source before citing them. To reach primary sources, ask in English for the primary study (for
  example "original study on X, journal article") and use `site:` for Polish or other local sources
  (`site:cbos.pl ...`).
- `authority` is passed through from the upstream index as is. Its meaning is not documented and it does not
  track credibility (a student PDF and a reference database can score alike), so do not use it to judge or
  filter sources.

Example calls:

```text
deepseek_search(query="rust async runtime comparison", count=5)
deepseek_search(query="gobierno de España presupuestos 2026", lang="es", need_content=False)
deepseek_fetch(url="https://docs.rs/tokio/latest/tokio/", max_chars=15000)
deepseek_fetch_many(urls=["https://example.com/a", "https://example.com/b"], max_total_chars=12000)
deepseek_research(question="how does HTTP/3 connection migration work", max_pages=4)
deepseek_deep_research(topic="comparing Raft vs Paxos consensus mechanisms in distributed systems")
deepseek_extract(url="https://news.ycombinator.com", fields=["top_story", "points", "author"])
deepseek_check_balance()
```

### Response envelope

Every tool returns a JSON object with `ok`.

| Field | Where | Meaning |
|---|---|---|
| `ok` | all | `true` on success, `false` on failure |
| `cached` | all tools except balance; also per item in `deepseek_fetch_many` and per source in `deepseek_research` | `true` when the response came from the in-memory cache. For `deepseek_fetch_many` and `deepseek_research` it is `true` only when no upstream request was made and at least one response came from the cache |
| `cached_count` | `deepseek_fetch_many`, `deepseek_research` | how many responses in this call were served from the cache |
| `requests_used` | `deepseek_fetch_many`, `deepseek_research` | upstream requests actually made (cache hits are free) |
| `elapsed_ms` | all, including errors | wall time of the tool call in milliseconds (monotonic clock); a cache hit reports its own, near-zero time |
| `notes` | all success envelopes | human-readable remarks: dedupe counts, rewrite applied or failed, `lang` applied, truncation ("Some page content was truncated to respect the character budget.") |
| `error` | failures | `{ "code", "message", "status"? }`; `status` is the upstream HTTP status when there was one |
| `cost_usd` | all retrieval tools | estimated upstream API cost in USD ($0.00075 per upstream call, $0.0 for cache hits) |
| `untrusted_content` | all tools returning web content | `true`; marks fetched web data as untrusted to defend against prompt injection |
| `_meta` | all retrieval tools | metadata object `{ "cost_usd": float, "untrusted_content": bool }` for MCP client telemetry |
| `partial` | `deepseek_fetch_many`, `deepseek_research` | `true` when some items or sources failed; each failed one carries its own `error`. The top-level `ok` stays `true`, so check `partial` |
| `failed_count` | `deepseek_fetch_many`, `deepseek_research` | number of failed items or sources |

Search results carry `title` (the URL host without `www.` when the upstream title is blank; a title repeated around ` - `, such as Reddit's `X - X`, is collapsed to `X`), `url`, `date`, `authority` (0-4, -1 unknown; an upstream field, not a credibility score), `rerank_score`, `site`, `snippet`,
and `content` with `chars` or `summary` where available. When the upstream gives no `date`, a `YYYY-MM-DD` date
(a valid calendar date, year 1990 to next year) found in the title or the first 600 characters of the text is
used instead. Without one, a full English or Polish month name next to a year (`September 2025`,
`wrzesień 2025`, PubMed-style `2025 September`) gives the first of that month, and a day written with it (`15 września 2025`,
`September 15, 2025`) gives that day; Polish names are matched in the nominative and genitive, abbreviations
and a bare year are ignored; `date_source` is `"upstream"`, `"content"` or `null` (no date), on search results and research
sources alike. `deepseek_fetch` returns `url`, `title`, `content`, `chars`,
`truncated`. `deepseek_deep_research` returns `topic`, `subqueries`, `report` (with bracketed citations `[1]`, `[2]`),
`synthesized`, and `sources`. `deepseek_extract` returns `url`, `title`, and `data` (the extracted JSON matching the
requested fields or schema). `chars` is always the length of the `content` returned, in every tool.

### Error codes

| Code | Meaning |
|---|---|
| `missing_api_key` | no key in `DEEPSEEK_API_KEY` or the fallback file |
| `ssrf_blocked` | URL rejected locally: internal host, literal private/loopback address, or embedded credentials |
| `bad_url`, `invalid_url`, `empty_url`, `url_too_long` | malformed or oversized URL |
| `empty_query`, `query_too_long`, `bad_count`, `bad_lang`, `bad_request` | invalid arguments |
| `bad_term`, `bad_phrase`, `bad_domain`, `bad_date`, `filters_too_long` | invalid or oversized search filter options |
| `budget_exceeded` | local daily spend limit (`DEEPSEEK_DAILY_BUDGET_USD`) exceeded |
| `rate_limited_local` | no local rate-limit token became available within the maximum wait |
| `response_too_large` | upstream body exceeded the cap (8 MiB search, 32 MiB fetch) |
| `deadline_exceeded` | batch item not finished within the shared batch deadline |
| `timeout`, `network_error`, `request_failed`, `invalid_json` | transport-level failures |
| `internal_error` | unexpected exception while fetching one batch item |
| other / `http_error` | upstream API errors keep the code from DeepSeek (or `http_error`) and set `error.status` |

## Response size and cost

Serialized JSON size of one response per mode, measured on 2026-10-28 against the live API with the default
parameters of each tool (compact JSON, UTF-8 bytes):

| Call | Size |
|---|---|
| `deepseek_search`, `count=3`, `need_content=false` (default `summary_chars`) | 3.7 KB |
| `deepseek_search`, `count=3`, `need_content=false`, `summary_chars=0` | 2.3 KB |
| `deepseek_search`, `count=3`, `need_content=true` (default) | 15.1 KB |
| `deepseek_fetch`, one documentation page (default `max_chars`) | 20.9 KB |
| `deepseek_fetch_many`, 3 documentation pages | 13.3 KB |
| `deepseek_research`, `max_pages=3` | 20.1 KB |

These are single samples, not guarantees: sizes vary with the query and the pages returned, and are bounded by
`max_chars_per_result`, `max_total_chars` and the other character budgets, not by a fixed number.

Cost: DeepSeek bills Search and Fetch per successful request, not per returned result, so `count` does not change
the price of a call. See the [DeepSeek Search & Fetch guide](https://api-docs.deepseek.com/guides/search) for
current prices, billing rules and limits; they may change and this project does not track them. A tool call that
needs several upstream requests (for example `deepseek_research` fetching pages whose search content is missing,
or `deepseek_fetch_many`) is billed for each of them, and `rewrite=true` additionally uses `deepseek-flash` tokens.
Cache hits are not billed.

## Cache and rate limiting

The cache and the limiter live in the server process and are not shared between processes.

| | Search | Fetch | Query rewrite |
|---|---|---|---|
| Cache TTL | 5 min | 30 min | 24 h |
| Entry limit | 128 | 128 | 1024 |
| Size limit | 16 MiB | 32 MiB | 1 MiB |
| Rate limit (sustained / burst) | 10 / s, 20 | 5 / s, 10 | not limited |

- Only successful, non-empty responses are cached. Errors are never cached.
- Identical concurrent calls share one upstream request (single-flight); only the one that made the request pays a limiter token, and none of them reports `cached: true`.
- `refresh=true` skips the cache read for that call and replaces the stored entry.
- `dedupe` changes the upstream request (the search over-fetches), so `dedupe=true` and `dedupe=false` calls do not share a cache entry.
- Cache hits and rewrite calls cost no limiter token. One logical call takes one token, retries included.
- If a token would take longer than 5 s to become available, the call fails immediately with `rate_limited_local`
  rather than queueing indefinitely.
- Transient failures (HTTP 429 and gateway 5xx) are retried once.
- `DEEPSEEK_MCP_CACHE=0` disables all caches.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | yes | DeepSeek platform key |
| `DEEPSEEK_BASE_URL` | no | API base URL (default `https://api.deepseek.com`) |
| `MCP_BEARER_TOKEN` | HTTP only | token clients must send as `Authorization: Bearer ...` |
| `DEEPSEEK_MCP_CACHE` | no | `1` (default) or `0` to disable caching |
| `DEEPSEEK_MCP_LOG_LEVEL` | no | stderr log level for the server, the MCP SDK and uvicorn: `DEBUG`, `INFO`, `WARNING` (default), `ERROR`, `CRITICAL` |
| `DEEPSEEK_DAILY_BUDGET_USD` | no | optional daily spending limit in USD (e.g. `5.00`); blocks upstream queries with `budget_exceeded` once reached |
| `DEEPSEEK_STATE_DIR` | no | directory to persist local state files (defaults to `~/.cache/deepseek-search-everywhere-mcp`) |
| `DEEPSEEK_AUTH_JSON` | no | path to a JSON file used as the key fallback (see [Security](#security)) |

`.env.example` lists the same variables. The server does not read `.env` files itself; export the variables or
set them in your MCP client's `env`.

## Security

Full policy and reporting instructions: [SECURITY.md](SECURITY.md).

- **Static SSRF checks only.** URLs are validated before the cache, the limiter or the network are touched:
  non-http(s) schemes, internal hostnames, literal private/loopback/link-local addresses and URLs with embedded
  credentials return `ssrf_blocked`. The server **does not resolve DNS**, and DeepSeek performs the actual fetch,
  so this is not network isolation.
- **Key handling.** The key is sent only in the `Authorization` header to the configured API base URL. Keys,
  bearer tokens and similar secrets are redacted from error messages and notes.
- **HTTP transport** requires the bearer token on every request (constant-time comparison, never logged).
- **No telemetry.** The only outbound destination is the DeepSeek API.
- **Legacy opencode fallback.** If `DEEPSEEK_API_KEY` is unset, the server reads the key from
  `~/.local/share/opencode/auth.json` (`{"deepseek": {"key": "..."}}`), or from the file named by
  `DEEPSEEK_AUTH_JSON`. This exists only for opencode users; prefer the environment variable.
- Tool inputs (queries, URLs) are sent to DeepSeek to do the work. Do not pass secrets in them.

## Limitations

- `deepseek_fetch` returns HTTP 404 for PDFs and Reddit and times out on some sites. Use
  `deepseek_search` with the page title or a `site:`-scoped query instead.
- Fetch returns only title and content: no final URL, status code or publication date.
- Fetch latency is DeepSeek's page render, not this server: an already rendered page returns in about 0.5 s,
  others take 3-10 s, and a missing page (HTTP 404) still takes 3-5 s before DeepSeek reports it. A 404 is
  not retried. `deepseek_fetch_many` runs up to 5 fetches at once, so a batch takes about as long as its
  slowest page; plan for that with larger batches.
- Search and fetch only. No site mapping or crawling, no asynchronous jobs, a single provider (DeepSeek).
- The cache and rate limiter are per process; running several server processes multiplies the effective rate.
- Local rate limits are conservative defaults and do not replace DeepSeek's own quotas.
- The HTTP transport is plain HTTP, stateless and JSON-only; there are no OAuth and no SSE streams.

## Development

```bash
git clone https://github.com/radoxtech/deepseek-search-everywhere-mcp.git
cd deepseek-search-everywhere-mcp
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,lint,http]" build twine

pytest -q                     # offline suite; no network, no key needed
ruff check .                  # lint
python -m build               # sdist + wheel in dist/
python -m twine check dist/*  # validate package metadata
```

The offline tests clear `DEEPSEEK_*` variables and point the key fallback at a nonexistent file, so they cannot
reach the network. CI runs lint, tests and build on Python 3.11, 3.12 and 3.13.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and commit conventions.

### Live tests

The live suite calls the real DeepSeek API end to end and uses `deepseek-flash` as an LLM judge to grade whether
results are relevant and answer the question. It is excluded from the default run, skips itself without a key,
and a full run costs a few cents at most (it also makes `deepseek-flash` judge calls).

```bash
cp .env.example .env          # then put your key in DEEPSEEK_API_KEY=
set -a && . ./.env && set +a  # export the variables; tests do not read .env themselves
pytest -m live
```

Judge scores and reasons are printed after the run. `.env` is git-ignored; never commit it.

### Releasing

Versions follow [Semantic Versioning](https://semver.org/) and are automated with
[release-please](https://github.com/googleapis/release-please) from
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Everything runs in
`.github/workflows/release.yml`:

1. Every push to `main` runs release-please. It opens (or updates) a release PR that bumps the version in
   `pyproject.toml`, `server.json` and `.release-please-manifest.json` and prepends a section to `CHANGELOG.md`.
   Standard Semantic Versioning applies: `fix` bumps the patch version, `feat` bumps the minor, and
   breaking changes (`feat!:`) bump the major version; `feat`, `fix`, `perf`, `refactor`, `docs` and `deps`
   appear in the changelog.
2. Merging the release PR makes release-please tag `vX.Y.Z` and create the GitHub Release.
3. The same workflow run then builds the tagged commit, checks that the tag equals the versions in
   `pyproject.toml` and `server.json`, runs `twine check` and a clean-wheel smoke test, publishes to PyPI through
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, no tokens), attaches the artifacts to
   the GitHub Release and updates the MCP Registry.

Publishing happens in the same workflow because tags and PRs created with `GITHUB_TOKEN` do not trigger other
workflows.

Manual fallback: bump the version in `pyproject.toml`, `server.json` and `.release-please-manifest.json`
(`tests/test_version_sync.py` checks the first two agree), update `CHANGELOG.md`, commit, and push a tag. The
workflow then publishes and creates the GitHub Release itself.

```bash
git tag -a v2.0.0 -m "v2.0.0" && git push origin v2.0.0
bash scripts/smoke_package.sh   # optional local check: build, install into a clean venv, run the CLI
```

Running the workflow manually (`workflow_dispatch`) builds and verifies the package and publishes to TestPyPI
only.

One-time repository setup:

- Settings, Actions, General: enable "Allow GitHub Actions to create and approve pull requests" so
  release-please can open its PR.
- Settings, Environments: create a `pypi` environment (and `testpypi` for manual runs); adding required
  reviewers there gates every PyPI publish.
- On PyPI (and TestPyPI), add a trusted publisher for this repository with workflow `release.yml` and
  environment `pypi` (`testpypi` on TestPyPI).

### Publishing to the MCP Registry manually

`server.json` describes the PyPI package for the [MCP Registry](https://modelcontextprotocol.io/registry). The
README carries an `mcp-name` comment that the registry uses to verify package ownership. The release workflow
does this automatically; to publish by hand:

```bash
python scripts/validate_server_json.py   # structural checks; also runs `mcp-publisher validate` if installed
mcp-publisher login github               # the io.github.radoxtech namespace requires GitHub authentication
mcp-publisher publish
```

Publish to PyPI first: the registry checks that the package version exists. release-please keeps `pyproject.toml`
and `server.json` in step; `tests/test_version_sync.py` fails when they drift apart.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

[MIT](LICENSE). Not affiliated with DeepSeek.
