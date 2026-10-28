# DeepSeek Search Everywhere MCP

[![CI](https://github.com/radoxtech/deepseek-search-everywhere-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/radoxtech/deepseek-search-everywhere-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

An [MCP](https://modelcontextprotocol.io) server that gives AI agents web search, page fetching and
multi-source research through the official DeepSeek Search & Fetch API.

It is a thin, careful layer over that API: results are deduplicated, content is capped to a budget you set,
repeated calls are served from an in-memory cache, and outgoing calls are paced by a local rate limiter.
Six tools, stdio by default, optional bearer-protected streamable HTTP.

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

The package is installed from GitHub; no registry release is required.

```bash
# Run without installing (uv)
uvx --from git+https://github.com/radoxtech/deepseek-search-everywhere-mcp deepseek-search-everywhere-mcp

# Or install with pip
pip install git+https://github.com/radoxtech/deepseek-search-everywhere-mcp.git

# Or from a source checkout
git clone https://github.com/radoxtech/deepseek-search-everywhere-mcp.git
cd deepseek-search-everywhere-mcp
pip install -e .
```

The HTTP transport needs the `http` extra (it pulls in `uvicorn`):

```bash
pip install "deepseek-search-everywhere-mcp[http] @ git+https://github.com/radoxtech/deepseek-search-everywhere-mcp.git"
# from a checkout: pip install -e ".[http]"
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
  uvx --from git+https://github.com/radoxtech/deepseek-search-everywhere-mcp deepseek-search-everywhere-mcp
```

### Claude Desktop

Edit `claude_desktop_config.json` (Settings, Developer, Edit Config):

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/radoxtech/deepseek-search-everywhere-mcp",
        "deepseek-search-everywhere-mcp"
      ],
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
      "command": [
        "uvx",
        "--from",
        "git+https://github.com/radoxtech/deepseek-search-everywhere-mcp",
        "deepseek-search-everywhere-mcp"
      ],
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
pip install -e ".[http]"          # or the pip/GitHub form shown above
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

## Tools

| Tool | Purpose | Parameters (defaults) |
|---|---|---|
| `deepseek_web_search` | Search; titles, URLs, dates, authority, rerank scores, optional page content | `query`, `count=10` (1-50), `ori_question`, `need_content=False`, `lang`, `max_chars_per_result=4000`, `max_total_chars=40000`, `dedupe=True`, `rewrite=False`, `refresh=False`, `summary_chars`, `strip_boilerplate=True` |
| `deepseek_search_and_read` | Search with page content in one request; the recommended retrieval path | `query`, `count=5` (1-50), `ori_question`, `lang`, `max_chars_per_result=4000`, `max_total_chars=30000`, `dedupe=True`, `rewrite=False`, `refresh=False`, `summary_chars`, `strip_boilerplate=True` |
| `deepseek_fetch` | Fetch one known URL as Markdown | `url`, `clean=True`, `max_chars=20000`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_fetch_many` | Fetch 1-10 URLs concurrently (up to 5 at a time) | `urls`, `max_chars_per_url=4000`, `max_total_chars=20000`, `clean=True`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_research` | Single-pass source gathering: one search with content, then fetch the sources whose content is missing or shorter than 200 characters (at most 5 fetches). No iteration, no summary | `question`, `max_pages=5` (1-50), `per_page_chars=6000`, `total_chars=30000`, `lang`, `rewrite=False`, `dedupe=True`, `refresh=False`, `strip_boilerplate=True` |
| `deepseek_balance` | Account balance | none |

Notes on the parameters:

- `query` and `question` are limited to 128 characters and support `site:`, `-site:`, `"phrase"`, `lang:`,
  `after:` and `before:`. `ori_question` (max 128 characters) is the natural-language question and improves ranking.
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
- `strip_boilerplate=True` (keyword-only, on every tool that returns page content) removes known site-interface
  lines. DeepSeek returns Reddit threads with the site UI localised to Chinese whatever the thread's language:
  the comment-sort menu (`排序方式：`, `最佳`, `新`...), the `# 评论区域` heading, the archived-post banner,
  `[已删除]` author lines and the sign-up footer. Rules are per site (Reddit only for now, matched on
  `reddit.com` and its subdomains) and only drop lines that consist entirely of a known label, so user text,
  Chinese included, and code blocks are kept. It runs after `clean` and before the character caps, so budgets
  count the cleaned text; it does not change caching. `strip_boilerplate=False` returns the content as fetched.
- `deepseek_fetch_many` skips empty entries, fetches duplicate URLs (same canonical URL) once, keeps the input
  order, and reports a failing URL as an item with `ok: false` instead of failing the batch. All fetches share
  a 120 s deadline; URLs still pending then are reported as `deadline_exceeded`. Each page gets at most
  `min(max_chars_per_url, max_total_chars // number of URLs)` characters.
- `deepseek_research` is single-pass source gathering, not an agent: it makes exactly one search plus the gap
  fetches, never refines the query or searches again, and does no summarising. It takes the same `lang`,
  `rewrite` and `dedupe` options as search; the envelope's `query_used`
  shows the query that was sent. It does no summarising and makes no model call unless `rewrite=True`: it returns sources for the caller to synthesise.
  Each source carries `title`, `url`, `date` and `site` (both null when the upstream gave none), `authority` and
  `rerank_score`, so stale or off-topic sources are visible without opening them.

Example calls:

```text
deepseek_search_and_read(query="rust async runtime comparison", count=5)
deepseek_web_search(query="gobierno de España presupuestos 2026", lang="es", need_content=False)
deepseek_fetch_many(urls=["https://example.com/a", "https://example.com/b"], max_total_chars=12000)
deepseek_research(question="how does HTTP/3 connection migration work", max_pages=4)
deepseek_research(question="CBOS 2025 korzystanie z internetu Polacy", lang="pl")
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
| `partial` | `deepseek_fetch_many`, `deepseek_research` | `true` when some items or sources failed; each failed one carries its own `error`. The top-level `ok` stays `true`, so check `partial` |
| `failed_count` | `deepseek_fetch_many`, `deepseek_research` | number of failed items or sources |

Search results carry `title` (the URL host without `www.` when the upstream title is blank; a title repeated around ` - `, such as Reddit's `X - X`, is collapsed to `X`), `url`, `date`, `authority` (0-4, -1 unknown), `rerank_score`, `site`, `snippet`,
and `content` with `chars` or `summary` where available. `deepseek_fetch` returns `title`, `content`, `chars`,
`truncated`. `chars` is always the length of the `content` returned, in every tool.

`content_chars` is deprecated and will be removed in the next release: on `deepseek_fetch` it equals `chars`;
on search results it still reports the cleaned page length before the per-result cap.

### Error codes

| Code | Meaning |
|---|---|
| `missing_api_key` | no key in `DEEPSEEK_API_KEY` or the fallback file |
| `ssrf_blocked` | URL rejected locally: internal host, literal private/loopback address, or embedded credentials |
| `bad_url`, `invalid_url`, `empty_url`, `url_too_long` | malformed or oversized URL |
| `empty_query`, `query_too_long`, `bad_count`, `bad_lang`, `bad_request` | invalid arguments |
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
| `deepseek_web_search`, `count=3`, `need_content=false` (default `summary_chars`) | 3.7 KB |
| `deepseek_web_search`, `count=3`, `need_content=false`, `summary_chars=0` | 2.3 KB |
| `deepseek_web_search`, `count=3`, `need_content=true` | 15.1 KB |
| `deepseek_search_and_read`, `count=3` | 15.1 KB |
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
  `deepseek_search_and_read` with the page title or a `site:`-scoped query instead.
- Fetch returns only title and content: no final URL, status code or publication date.
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
   While the version is below 1.0.0, `feat` bumps the patch version and breaking changes bump the minor one;
   `feat`, `fix`, `perf`, `refactor`, `docs` and `deps` commits appear in the changelog.
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
git tag -a v0.2.1 -m "v0.2.1" && git push origin v0.2.1
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
